// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Poll;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common_runtime::JoinSet;
use datafusion_common_runtime::SpawnedTask;
use datafusion_datasource::file_sink_config::FileSink;
use datafusion_datasource::file_sink_config::FileSinkConfig;
use datafusion_datasource::sink::DataSink;
use datafusion_datasource::write::demux::DemuxedStreamReceiver;
use datafusion_datasource::write::get_writer_schema;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::DisplayAs;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::metrics::MetricsSet;
use futures::Stream;
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::path::Path;
use uuid::Uuid;
use vortex::array::ArrayRef;
use vortex::array::arrow::FromArrowArray;
use vortex::array::stream::ArrayStream;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexResult;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::ObjectStoreWriter;
use vortex::io::VortexWrite;
use vortex::session::VortexSession;

pub struct VortexSink {
    config: FileSinkConfig,
    schema: SchemaRef,
    session: VortexSession,
    /// Target file size in bytes. The writer will split output files
    /// when they reach approximately this size.
    target_file_size: u64,
}

const DEFAULT_TARGET_FILE_SIZE_BYTES: u64 = 16 * 1024 * 1024;

impl VortexSink {
    pub fn new(
        config: FileSinkConfig,
        schema: SchemaRef,
        session: VortexSession,
        target_file_size: u64,
    ) -> Self {
        let target_file_size = if target_file_size == 0 {
            tracing::warn!(
                "Received target_file_size=0 for VortexSink; defaulting to 16MB"
            );
            DEFAULT_TARGET_FILE_SIZE_BYTES
        } else {
            target_file_size
        };

        Self {
            config,
            schema,
            session,
            target_file_size,
        }
    }
}

impl VortexSink {
    /// Write all data with target file size control, bypassing the DataFusion demuxer.
    ///
    /// This method consumes the input stream directly and writes files sized to approximately
    /// `target_file_size` bytes. It uses an adaptive approach: after writing each file, it
    /// observes the actual compression ratio and adjusts how many bytes to accumulate
    /// for the next file.
    ///
    /// Batches are streamed directly into the file writer without being buffered in memory.
    /// A channel-based stream wrapper feeds each batch from the input into the writer as it
    /// arrives. Once the estimated uncompressed byte target for a file is reached, the channel
    /// is closed so the writer finishes the current file, and a new writer is started for the
    /// next file.
    async fn write_all_with_target_size(
        &self,
        mut data: SendableRecordBatchStream,
        context: &Arc<TaskContext>,
        target_file_size: u64,
    ) -> DFResult<u64> {
        let target_file_size = target_file_size.max(1);

        let object_store = context
            .runtime_env()
            .object_store(&self.config.object_store_url)?;

        let base_output_path = &self.config.table_paths[0];
        let writer_schema = get_writer_schema(&self.config);
        let dtype = DType::from_arrow(writer_schema);

        let write_id = Uuid::new_v4().to_string();

        let mut row_count: u64 = 0;
        let mut file_index: usize = 0;

        // Start with a 1:1 ratio assumption; will be updated after the first file is written.
        // This means we'll aim to accumulate target_file_size bytes of uncompressed data
        // for the first file, then adjust based on actual results.
        let mut uncompressed_target = target_file_size;

        loop {
            let path = base_output_path.prefix().child(format!(
                "{write_id}_{file_index}.{}",
                self.config.file_extension
            ));

            let mut sink = ObjectStoreWriter::new(object_store.clone(), &path)
                .await
                .map_err(|e| {
                    DataFusionError::Execution(format!("Failed to create ObjectStoreWriter: {e}"))
                })?;

            // Create a bounded channel (capacity 1) so at most one batch is in-flight
            // between the feeder and the writer — no batch buffering in memory.
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            let stream = ChannelArrayStream {
                dtype: dtype.clone(),
                rx,
            };

            // Run the feeder and writer concurrently. The feeder pulls from the source
            // stream and pushes into the channel; the writer reads from the channel
            // and writes to the file.
            let write_fut = self.session.write_options().write(&mut sink, stream);
            let feed_fut = feed_record_batch_stream(&mut data, tx, uncompressed_target);

            let (write_result, feed_result) = tokio::join!(write_fut, feed_fut);

            let summary = write_result.map_err(|e| {
                DataFusionError::Execution(format!("Failed to write Vortex file: {e}"))
            })?;
            let feed = feed_result?;

            let file_row_count = summary.row_count();
            let written_size = summary.size();

            // If the stream never produced any rows, we're done (empty input).
            if file_row_count == 0 {
                if !feed.source_exhausted {
                    return Err(DataFusionError::Execution(
                        "Writer produced an empty file before the source was exhausted"
                            .to_string(),
                    ));
                }
                break;
            }

            sink.shutdown().await.map_err(|e| {
                DataFusionError::Execution(format!("Failed to shutdown Vortex writer: {e}"))
            })?;

            row_count += file_row_count;

            // Update compression ratio estimate for the next file.
            if written_size > 0 && !feed.source_exhausted {
                let flushed_uncompressed = feed.uncompressed_bytes;
                uncompressed_target = target_file_size
                    .saturating_mul(flushed_uncompressed)
                    .saturating_div(written_size);
                // Clamp to at least the target to avoid accumulating too little
                uncompressed_target = uncompressed_target.max(target_file_size);
            }

            tracing::debug!(
                path = %path,
                written_bytes = written_size,
                uncompressed_bytes = feed.uncompressed_bytes,
                next_uncompressed_target = uncompressed_target,
                "Wrote file with target size control"
            );

            if feed.source_exhausted {
                break;
            }

            file_index += 1;
        }

        Ok(row_count)
    }
}

impl std::fmt::Debug for VortexSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexSink").finish()
    }
}

impl DisplayAs for VortexSink {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(f, "VortexSink")
            }
        }
    }
}

#[async_trait]
impl DataSink for VortexSink {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn metrics(&self) -> Option<MetricsSet> {
        None
    }

    /// Returns the sink schema
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    async fn write_all(
        &self,
        data: SendableRecordBatchStream,
        context: &Arc<TaskContext>,
    ) -> DFResult<u64> {
        if self.config.table_partition_cols.is_empty() {
            // Non-partitioned: bypass the demuxer and write files directly
            // with size-based splitting.
            self.write_all_with_target_size(data, context, self.target_file_size)
                .await
        } else {
            // Partitioned: use the FileSink/demuxer flow, which will call
            // spawn_writer_tasks_and_join where we also apply size limits.
            FileSink::write_all(self, data, context).await
        }
    }
}

#[async_trait]
impl FileSink for VortexSink {
    fn config(&self) -> &FileSinkConfig {
        &self.config
    }

    async fn spawn_writer_tasks_and_join(
        &self,
        _context: &Arc<TaskContext>,
        demux_task: SpawnedTask<DFResult<()>>,
        mut file_stream_rx: DemuxedStreamReceiver,
        object_store: Arc<dyn ObjectStore>,
    ) -> DFResult<u64> {
        let row_counter = Arc::new(AtomicU64::new(0));

        let mut file_write_tasks: JoinSet<DFResult<Vec<Path>>> = JoinSet::new();

        // TODO(adamg):
        // 1. We can probably be better at signaling how much memory we're consuming (potentially when reading too), see ParquetSink::spawn_writer_tasks_and_join.
        while let Some((path, rx)) = file_stream_rx.recv().await {
            let session = self.session.clone();
            let row_counter = row_counter.clone();
            let object_store = object_store.clone();
            let writer_schema = get_writer_schema(&self.config);
            let dtype = DType::from_arrow(writer_schema);
            let target_file_size = self.target_file_size;

            file_write_tasks.spawn(async move {
                write_with_file_size_limit(
                    session,
                    row_counter,
                    object_store,
                    dtype,
                    path,
                    rx,
                    target_file_size,
                )
                .await
            });
        }

        while let Some(result) = file_write_tasks.join_next().await {
            match result {
                Ok(paths) => {
                    for path in paths? {
                        tracing::info!(path = %path, "Successfully written file");
                    }
                }
                Err(e) => {
                    if e.is_panic() {
                        std::panic::resume_unwind(e.into_panic());
                    } else {
                        unreachable!();
                    }
                }
            }
        }

        demux_task
            .join_unwind()
            .await
            .map_err(|e| DataFusionError::ExecutionJoin(Box::new(e)))??;

        Ok(row_counter.load(Ordering::SeqCst))
    }
}

/// Generate a split path from an original path by inserting a sub-index before the extension.
///
/// For example: `data/abc_0.vortex` with sub_index=1 becomes `data/abc_0_1.vortex`.
fn split_path(original: &Path, sub_index: usize) -> Path {
    let path_str = original.to_string();
    if let Some(dot_pos) = path_str.rfind('.') {
        let (stem, ext) = path_str.split_at(dot_pos);
        Path::from(format!("{stem}_{sub_index}{ext}"))
    } else {
        Path::from(format!("{path_str}_{sub_index}"))
    }
}

/// Write the stream to multiple Vortex files, splitting when the target file size is reached.
///
/// Batches from the input receiver are streamed directly into the file writer without being
/// buffered in memory. A [`SizeLimitedRecordBatchStream`] wraps the receiver and stops
/// yielding batches once the uncompressed byte target is reached, causing the writer to
/// finish the current file. The receiver is then reused for the next file.
async fn write_with_file_size_limit(
    session: VortexSession,
    row_counter: Arc<AtomicU64>,
    object_store: Arc<dyn ObjectStore>,
    dtype: DType,
    base_path: Path,
    rx: tokio::sync::mpsc::Receiver<datafusion_common::arrow::array::RecordBatch>,
    target_file_size: u64,
) -> DFResult<Vec<Path>> {
    let target_file_size = target_file_size.max(1);

    let mut written_paths = Vec::new();
    let mut file_index: usize = 0;

    let mut rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

    loop {
        let path = split_path(&base_path, file_index);

        let mut sink = ObjectStoreWriter::new(object_store.clone(), &path)
            .await
            .map_err(|e| {
                DataFusionError::Execution(format!("Failed to create ObjectStoreWriter: {e}"))
            })?;

        // Create a bounded channel (capacity 1) — at most one batch in-flight.
        let (tx, array_rx) = tokio::sync::mpsc::channel(1);
        let stream = ChannelArrayStream {
            dtype: dtype.clone(),
            rx: array_rx,
        };

        // Run the feeder and writer concurrently.
        let write_fut = session.write_options().write(&mut sink, stream);
        let feed_fut = feed_receiver_stream(&mut rx_stream, tx, target_file_size);

        let (write_result, feed_result) = tokio::join!(write_fut, feed_fut);

        let summary = write_result
            .map_err(|e| DataFusionError::Execution(format!("Failed to write Vortex file: {e}")))?;
        let feed = feed_result?;

        let file_row_count = summary.row_count();

        // If the stream never produced any rows, the source was already exhausted.
        if file_row_count == 0 {
            if !feed.source_exhausted {
                return Err(DataFusionError::Execution(
                    "Writer produced an empty file before the source was exhausted"
                        .to_string(),
                ));
            }
            break;
        }

        sink.shutdown().await.map_err(|e| {
            DataFusionError::Execution(format!("Failed to shutdown Vortex writer: {e}"))
        })?;

        row_counter.fetch_add(file_row_count, Ordering::Relaxed);

        tracing::debug!(path = %path, "Wrote split file at target size");
        written_paths.push(path);

        if feed.source_exhausted {
            break;
        }

        file_index += 1;
    }

    Ok(written_paths)
}

/// A stream adapter that wraps a `tokio::sync::mpsc::Receiver` of `VortexResult<ArrayRef>` items
/// and implements `ArrayStream`. Used to create a `'static + Send` stream from a channel
/// receiver that can be passed into `VortexWriteOptions::write`.
struct ChannelArrayStream {
    dtype: DType,
    rx: tokio::sync::mpsc::Receiver<VortexResult<ArrayRef>>,
}

impl ArrayStream for ChannelArrayStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl Stream for ChannelArrayStream {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        this.rx.poll_recv(cx)
    }
}

/// Result of feeding batches from a source stream into a channel-based writer.
struct FeedResult {
    /// Total uncompressed bytes sent through the channel.
    uncompressed_bytes: u64,
    /// Whether the source stream was fully exhausted.
    source_exhausted: bool,
}

/// Feed record batches from a `SendableRecordBatchStream` into a channel sender,
/// converting each batch to `ArrayRef` and tracking uncompressed byte totals.
/// Stops feeding when accumulated bytes reach `target_bytes` or the source is exhausted.
/// Drops the sender when done to signal end-of-stream to the writer.
async fn feed_record_batch_stream(
    data: &mut SendableRecordBatchStream,
    tx: tokio::sync::mpsc::Sender<VortexResult<ArrayRef>>,
    target_bytes: u64,
) -> DFResult<FeedResult> {
    let mut accumulated = 0u64;
    let mut source_exhausted = false;

    while accumulated < target_bytes {
        match data.next().await.transpose()? {
            Some(rb) => {
                if rb.num_rows() == 0 {
                    continue;
                }

                let batch_size: u64 = rb
                    .columns()
                    .iter()
                    .map(|c| c.get_array_memory_size() as u64)
                    .sum();

                let array = ArrayRef::from_arrow(rb, false);
                accumulated += batch_size;

                // Send the batch through the channel. If the receiver is dropped
                // (writer failed), this will error.
                tx.send(Ok(array)).await.map_err(|_| {
                    DataFusionError::Execution("Writer channel closed unexpectedly".to_string())
                })?;
            }
            None => {
                source_exhausted = true;
                break;
            }
        }
    }

    // Drop the sender to signal end-of-stream to the writer.
    drop(tx);

    Ok(FeedResult {
        uncompressed_bytes: accumulated,
        source_exhausted,
    })
}

/// Feed record batches from a `ReceiverStream` (partitioned path) into a channel sender.
/// Same semantics as [`feed_record_batch_stream`] but for infallible `RecordBatch` items.
async fn feed_receiver_stream(
    data: &mut tokio_stream::wrappers::ReceiverStream<datafusion_common::arrow::array::RecordBatch>,
    tx: tokio::sync::mpsc::Sender<VortexResult<ArrayRef>>,
    target_bytes: u64,
) -> DFResult<FeedResult> {
    let mut accumulated = 0u64;
    let mut source_exhausted = false;

    while accumulated < target_bytes {
        match data.next().await {
            Some(rb) => {
                if rb.num_rows() == 0 {
                    continue;
                }

                let batch_size: u64 = rb
                    .columns()
                    .iter()
                    .map(|c| c.get_array_memory_size() as u64)
                    .sum();

                let array = ArrayRef::from_arrow(rb, false);
                accumulated += batch_size;

                tx.send(Ok(array)).await.map_err(|_| {
                    DataFusionError::Execution("Writer channel closed unexpectedly".to_string())
                })?;
            }
            None => {
                source_exhausted = true;
                break;
            }
        }
    }

    drop(tx);

    Ok(FeedResult {
        uncompressed_bytes: accumulated,
        source_exhausted,
    })
}

/// Write a set of arrays to a single Vortex file, returning the written file size in bytes.
#[cfg(test)]
async fn flush_batches_to_file(
    session: &VortexSession,
    object_store: &Arc<dyn ObjectStore>,
    dtype: &DType,
    path: &Path,
    batches: Vec<ArrayRef>,
) -> DFResult<u64> {
    let stream = futures::stream::iter(batches.into_iter().map(VortexResult::Ok));
    let stream_adapter = vortex::array::stream::ArrayStreamAdapter::new(dtype.clone(), stream);

    let mut sink = ObjectStoreWriter::new(object_store.clone(), path)
        .await
        .map_err(|e| {
            DataFusionError::Execution(format!("Failed to create ObjectStoreWriter: {e}"))
        })?;

    let summary = session
        .write_options()
        .write(&mut sink, stream_adapter)
        .await
        .map_err(|e| DataFusionError::Execution(format!("Failed to write Vortex file: {e}")))?;

    sink.shutdown().await.map_err(|e| {
        DataFusionError::Execution(format!("Failed to shutdown Vortex writer: {e}"))
    })?;

    Ok(summary.size())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use datafusion::arrow::array::Int8Array;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::datasource::DefaultTableSource;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::logical_expr::Expr;
    use datafusion::logical_expr::LogicalPlan;
    use datafusion::logical_expr::LogicalPlanBuilder;
    use datafusion::logical_expr::Values;
    use datafusion::prelude::SessionContext;
    use datafusion_common::ScalarValue;
    use datafusion_datasource::file_format::format_as_file_type;
    use object_store::ObjectStore;
    use object_store::path::Path;
    use rstest::rstest;
    use tempfile::TempDir;
    use tokio_stream::wrappers::ReceiverStream;
    use vortex::VortexSessionDefault;
    use vortex::array::Array;
    use vortex::dtype::DType;
    use vortex::dtype::arrow::FromArrowType;
    use vortex::session::VortexSession;
    use walkdir::WalkDir;

    use super::flush_batches_to_file;
    use super::split_path;
    use crate::persistent::VortexFormatFactory;
    use crate::persistent::register_vortex_format_factory;

    #[tokio::test]
    async fn test_insert_into() {
        let dir = TempDir::new().expect("should create temp dir");

        let factory = VortexFormatFactory::new();

        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE my_tbl \
                    (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION '{}/';",
                dir.path().to_str().expect("should convert path to str")
            ))
            .await
            .expect("should create external table");

        let my_tbl = session.table("my_tbl").await.expect("should get table");

        // It's valuable to have two insert code paths because they actually behave slightly differently
        let values = Values {
            schema: Arc::new(my_tbl.schema().clone()),
            values: vec![vec![
                Expr::Literal(ScalarValue::new_utf8view("hello"), None),
                Expr::Literal(42_i32.into(), None),
            ]],
        };

        let tbl_provider = session
            .table_provider("my_tbl")
            .await
            .expect("should get table provider");

        let logical_plan = LogicalPlanBuilder::insert_into(
            LogicalPlan::Values(values.clone()),
            "my_tbl",
            Arc::new(DefaultTableSource::new(tbl_provider)),
            datafusion::logical_expr::dml::InsertOp::Append,
        )
        .expect("should build insert into plan")
        .build()
        .expect("should build logical plan");

        session
            .execute_logical_plan(logical_plan)
            .await
            .expect("should execute logical plan")
            .collect()
            .await
            .expect("should collect results");

        session
            .sql("INSERT INTO my_tbl VALUES ('world', 24);")
            .await
            .expect("should execute insert SQL")
            .collect()
            .await
            .expect("should collect insert results");

        my_tbl.clone().show().await.expect("should show table");

        assert_eq!(
            session
                .table("my_tbl")
                .await
                .expect("should get table")
                .count()
                .await
                .expect("should count rows"),
            2
        );
    }

    /// Reproduction by <https://github.com/vortex-data/vortex/issues/4315>.
    ///
    /// Uses a 1 MB target file size with varying data sizes to exercise file splitting.
    /// Small data (below target) → 1 file; large data (above target) → multiple files.
    #[rstest]
    #[case(1_000, 1)]
    #[case(5_000_000, 6)]
    #[case(10_000_000, 10)]
    #[tokio::test]
    async fn test_write_large_batch(
        #[case] entries: usize,
        #[case] expected_files: usize,
    ) -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;

        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 1;

        let factory = VortexFormatFactory::new().with_options(opts);

        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        // Use varying values so the data isn't trivially compressible to near-zero.
        let values: Vec<i8> = (0..entries).map(|i| (i % 127) as i8).collect();
        let data = session.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(values))],
        )?)?;

        let logical_plan = LogicalPlanBuilder::copy_to(
            data.logical_plan().clone(),
            dir.path()
                .to_str()
                .expect("should convert path to str")
                .to_string(),
            format_as_file_type(Arc::new(VortexFormatFactory::new().with_options({
                let mut o = crate::persistent::VortexOptions::default();
                o.target_file_size_mb = 1;
                o
            }))),
            Default::default(),
            vec![],
        )?
        .build()?;

        session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        // Validate the output by reading back the written files
        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE written_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{}/';",
                dir.path().to_str().expect("should convert path to str")
            ))
            .await?;

        let result = session
            .sql("SELECT COUNT(*) as count FROM written_data")
            .await?
            .collect()
            .await?;

        assert_eq!(result.len(), 1);
        let count_batch = &result[0];
        assert_eq!(count_batch.num_rows(), 1);

        let count_value = count_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("should downcast to Int64Array")
            .value(0);

        assert_eq!(
            count_value, entries as i64,
            "Expected {} entries, but found {}",
            entries, count_value
        );

        let all_data = session
            .sql("SELECT a FROM written_data")
            .await?
            .collect()
            .await?;

        let mut total_rows = 0;
        for batch in all_data {
            total_rows += batch.num_rows();
        }

        assert_eq!(
            total_rows, entries,
            "Total rows read ({}) doesn't match expected entries ({})",
            total_rows, entries
        );

        let file_count = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "vortex"))
            .count();

        assert_eq!(
            file_count, expected_files,
            "Expected {expected_files} files for {entries} entries with 1 MB target, got {file_count}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_write_partitioned() -> anyhow::Result<()> {
        let dir = TempDir::new()?;
        let data_dir = dir.path().to_str().expect("should convert path to str");

        let factory: VortexFormatFactory = VortexFormatFactory::new();
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        let _unused = session
            .sql(&format!(
                "CREATE EXTERNAL TABLE my_tbl \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION '{data_dir}' \
                PARTITIONED BY (c1);"
            ))
            .await?;

        session
            .sql("INSERT INTO my_tbl (c1, c2) VALUES ('world', 24), ('world', 25), ('hello', 42);")
            .await?
            .collect()
            .await?;

        let table = session.table("my_tbl").await?;
        assert_eq!(table.count().await?, 3);

        for dir in WalkDir::new(data_dir)
            .into_iter()
            .filter_entry(|e| e.path().is_dir())
        {
            let dir = dir?;
            if let Ok(path) = dir.path().strip_prefix(data_dir)
                && !path.as_os_str().is_empty()
            {
                assert!(path.starts_with("c1=hello") || path.starts_with("c1=world"),);
            }
        }

        Ok(())
    }

    /// Test that INSERT INTO with target_file_size produces files whose sizes are
    /// approximately within the target. This uses the ListingTable / INSERT INTO path
    /// which exercises the full DataFusion write pipeline including the demuxer bypass.
    #[tokio::test]
    async fn test_insert_into_with_target_file_size() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;
        let data_dir = dir.path().to_str().expect("should convert path to str");

        // Set a 1 MB target file size.
        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 1;

        let factory = VortexFormatFactory::new().with_options(opts);
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        // Create an external table (ListingTable) for writing.
        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE target_tbl \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{data_dir}/';"
            ))
            .await?;

        // Create a source batch with enough data to produce multiple files.
        // 5M Int8 values ≈ 5 MB uncompressed; with 1 MB target we expect multiple files.
        let entries: usize = 5_000_000;
        let values: Vec<i8> = (0..entries).map(|i| (i % 127) as i8).collect();
        let source_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(values))],
        )?;

        // Register the source as a temp table and INSERT INTO the ListingTable.
        session.register_batch("source_data", source_batch)?;
        session
            .sql("INSERT INTO target_tbl SELECT * FROM source_data")
            .await?
            .collect()
            .await?;

        // Collect written files and their sizes.
        let files: Vec<_> = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "vortex"))
            .collect();

        let file_count = files.len();
        assert!(
            file_count > 1,
            "Expected more than 1 file with 1 MB target and ~5 MB data, got {file_count}"
        );

        // Verify each file is within a reasonable range of the target.
        // Allow up to 4x target as a generous upper bound (compression ratios vary).
        let target_bytes = 1u64 * 1024 * 1024;
        for file in &files {
            let file_size = file.metadata()?.len();
            assert!(
                file_size <= target_bytes * 4,
                "File {:?} is {} bytes, which exceeds 4x target of {} bytes",
                file.path(),
                file_size,
                target_bytes
            );
        }

        // Read back and verify all data is preserved.
        let result = session
            .sql("SELECT COUNT(*) as cnt FROM target_tbl")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("should downcast to Int64Array")
            .value(0);

        assert_eq!(
            count_value, entries as i64,
            "Expected {entries} entries, but found {count_value}"
        );

        Ok(())
    }

    /// Test that INSERT INTO without a target file size does not produce excessive splitting.
    #[tokio::test]
    async fn test_insert_into_without_target_file_size() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;
        let data_dir = dir.path().to_str().expect("should convert path to str");

        // Default options — no target file size.
        let factory = VortexFormatFactory::new();
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE target_tbl \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{data_dir}/';"
            ))
            .await?;

        let entries: usize = 1_000;
        let source_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(vec![0i8; entries]))],
        )?;

        session.register_batch("source_data", source_batch)?;
        session
            .sql("INSERT INTO target_tbl SELECT * FROM source_data")
            .await?
            .collect()
            .await?;

        // With no target file size, the demuxer controls output. Just verify data integrity.
        let result = session
            .sql("SELECT COUNT(*) as cnt FROM target_tbl")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("should downcast to Int64Array")
            .value(0);

        assert_eq!(
            count_value, entries as i64,
            "Expected {entries} entries, but found {count_value}"
        );

        Ok(())
    }

    /// Test that data survives a round-trip through INSERT INTO with target file size:
    /// write data, read it back, and verify individual values.
    #[tokio::test]
    async fn test_insert_into_with_target_file_size_round_trip() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;
        let data_dir = dir.path().to_str().expect("should convert path to str");

        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 1;

        let factory = VortexFormatFactory::new().with_options(opts);
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE target_tbl \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{data_dir}/';"
            ))
            .await?;

        // Write data with a deterministic pattern so we can verify values.
        let entries: usize = 2_000_000;
        let values: Vec<i8> = (0..entries).map(|i| (i % 127) as i8).collect();
        let source_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(values))],
        )?;

        session.register_batch("source_data", source_batch)?;
        session
            .sql("INSERT INTO target_tbl SELECT * FROM source_data")
            .await?
            .collect()
            .await?;

        // Verify total count.
        let result = session
            .sql("SELECT COUNT(*) as cnt FROM target_tbl")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("should downcast to Int64Array")
            .value(0);

        assert_eq!(count_value, entries as i64);

        // Verify value distribution — each value 0..126 should appear roughly equally.
        let dist = session
            .sql(
                "SELECT a, COUNT(*) as cnt FROM target_tbl \
                 GROUP BY a ORDER BY a",
            )
            .await?
            .collect()
            .await?;

        let mut total = 0i64;
        for batch in &dist {
            total += batch.num_rows() as i64;
        }
        // 127 distinct values
        assert_eq!(total, 127, "Expected 127 distinct groups");

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Unit tests for feed_receiver_stream
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_feed_receiver_stream_skips_empty_batches() -> anyhow::Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)]));

        let empty = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int8Array::from(Vec::<i8>::new()))],
        )?;
        let non_empty =
            RecordBatch::try_new(schema, vec![Arc::new(Int8Array::from(vec![1, 2, 3]))])?;

        let (rb_tx, rb_rx) = tokio::sync::mpsc::channel(4);
        rb_tx.send(empty).await?;
        rb_tx.send(non_empty).await?;
        drop(rb_tx);

        let mut rb_stream = ReceiverStream::new(rb_rx);
        let (array_tx, mut array_rx) = tokio::sync::mpsc::channel(4);

        let feed = super::feed_receiver_stream(&mut rb_stream, array_tx, 1).await?;
        assert!(!feed.source_exhausted);

        let mut arrays = Vec::new();
        while let Some(array) = array_rx.recv().await {
            arrays.push(array?);
        }

        assert_eq!(arrays.len(), 1, "Empty batches should not be forwarded");
        assert_eq!(
            arrays[0].len(),
            3,
            "Non-empty batch should still be forwarded"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_feed_receiver_stream_all_empty_batches_exhausts_source() -> anyhow::Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)]));

        let empty =
            RecordBatch::try_new(schema, vec![Arc::new(Int8Array::from(Vec::<i8>::new()))])?;

        let (rb_tx, rb_rx) = tokio::sync::mpsc::channel(2);
        rb_tx.send(empty).await?;
        drop(rb_tx);

        let mut rb_stream = ReceiverStream::new(rb_rx);
        let (array_tx, mut array_rx) = tokio::sync::mpsc::channel(2);

        let feed = super::feed_receiver_stream(&mut rb_stream, array_tx, 1024).await?;
        assert!(
            feed.source_exhausted,
            "All-empty input should exhaust source"
        );

        assert!(
            array_rx.recv().await.is_none(),
            "No arrays should be forwarded for empty input"
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Unit tests for split_path
    // -----------------------------------------------------------------------

    #[test]
    fn test_split_path_with_extension() {
        let path = Path::from("data/abc_0.vortex");
        assert_eq!(split_path(&path, 1).to_string(), "data/abc_0_1.vortex");
        assert_eq!(split_path(&path, 0).to_string(), "data/abc_0_0.vortex");
        assert_eq!(split_path(&path, 42).to_string(), "data/abc_0_42.vortex");
    }

    #[test]
    fn test_split_path_without_extension() {
        let path = Path::from("data/abc_0");
        assert_eq!(split_path(&path, 1).to_string(), "data/abc_0_1");
        assert_eq!(split_path(&path, 0).to_string(), "data/abc_0_0");
    }

    #[test]
    fn test_split_path_with_multiple_dots() {
        let path = Path::from("data/my.file.name.vortex");
        // Should split at the last dot
        assert_eq!(
            split_path(&path, 3).to_string(),
            "data/my.file.name_3.vortex"
        );
    }

    #[test]
    fn test_split_path_root_level_file() {
        let path = Path::from("output.vortex");
        assert_eq!(split_path(&path, 0).to_string(), "output_0.vortex");
    }

    #[test]
    fn test_split_path_deeply_nested() {
        let path = Path::from("a/b/c/d/file.vortex");
        assert_eq!(split_path(&path, 5).to_string(), "a/b/c/d/file_5.vortex");
    }

    // -----------------------------------------------------------------------
    // Unit test for flush_batches_to_file
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_flush_batches_to_file_returns_nonzero_size() {
        use object_store::memory::InMemory;
        use vortex::array::IntoArray;
        use vortex::buffer::buffer;

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let session = VortexSession::default();
        let dtype = DType::from_arrow(Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Int32,
            false,
        )])));

        let batch = buffer![1i32, 2, 3, 4, 5].into_array();
        let path = Path::from("test_output.vortex");

        let written_size =
            flush_batches_to_file(&session, &object_store, &dtype, &path, vec![batch])
                .await
                .expect("should flush batches");

        assert!(
            written_size > 0,
            "Expected non-zero file size, got {written_size}"
        );

        // Verify the file actually exists in the object store
        let head = object_store
            .head(&path)
            .await
            .expect("file should exist in object store");
        assert_eq!(
            head.size as u64, written_size,
            "Object store file size should match returned size"
        );
    }

    #[tokio::test]
    async fn test_flush_batches_multiple_arrays() {
        use object_store::memory::InMemory;
        use vortex::array::IntoArray;
        use vortex::buffer::buffer;

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let session = VortexSession::default();
        let dtype = DType::from_arrow(Arc::new(Schema::new(vec![Field::new(
            "x",
            DataType::Int64,
            false,
        )])));

        let batches = vec![
            buffer![10i64, 20, 30].into_array(),
            buffer![40i64, 50, 60].into_array(),
            buffer![70i64, 80, 90].into_array(),
        ];

        let path = Path::from("multi_batch.vortex");

        let written_size = flush_batches_to_file(&session, &object_store, &dtype, &path, batches)
            .await
            .expect("should flush multiple batches");

        assert!(written_size > 0);
    }

    // -----------------------------------------------------------------------
    // Integration test: COPY TO with target file size
    // -----------------------------------------------------------------------

    /// Test that COPY TO with a target file size produces multiple files and
    /// preserves data integrity. This exercises the write_all_with_target_size
    /// code path (bypassing the demuxer).
    #[tokio::test]
    async fn test_copy_to_with_target_file_size() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;

        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 1;

        let factory = VortexFormatFactory::new().with_options(opts);
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        // 5M Int8 values ≈ 5 MB uncompressed; with 1 MB target we expect multiple files.
        let entries: usize = 5_000_000;
        let values: Vec<i8> = (0..entries).map(|i| (i % 127) as i8).collect();
        let data = session.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(values))],
        )?)?;

        let logical_plan = LogicalPlanBuilder::copy_to(
            data.logical_plan().clone(),
            dir.path()
                .to_str()
                .expect("should convert path to str")
                .to_string(),
            format_as_file_type(Arc::new(VortexFormatFactory::new().with_options({
                let mut o = crate::persistent::VortexOptions::default();
                o.target_file_size_mb = 1;
                o
            }))),
            Default::default(),
            vec![],
        )?
        .build()?;

        session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        let files: Vec<_> = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "vortex"))
            .collect();

        assert!(
            files.len() > 1,
            "Expected multiple files with 1 MB target and ~5 MB data via COPY TO, got {}",
            files.len()
        );

        // Read back and verify row count.
        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE copy_to_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{}/';",
                dir.path().to_str().expect("should convert path to str")
            ))
            .await?;

        let result = session
            .sql("SELECT COUNT(*) as cnt FROM copy_to_data")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("should downcast to Int64Array")
            .value(0);

        assert_eq!(
            count_value, entries as i64,
            "COPY TO round-trip: expected {entries} entries, got {count_value}"
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Integration test: Partitioned writes with target file size
    // -----------------------------------------------------------------------

    /// When partition columns are present, the write should go through the
    /// demuxer / write_with_file_size_limit path even with a target file size.
    #[tokio::test]
    async fn test_write_partitioned_with_target_file_size() -> anyhow::Result<()> {
        let dir = TempDir::new()?;
        let data_dir = dir.path().to_str().expect("should convert path to str");

        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 1;

        let factory = VortexFormatFactory::new().with_options(opts);
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE partitioned_tbl \
                    (category VARCHAR NOT NULL, value INT NOT NULL) \
                STORED AS vortex \
                LOCATION '{data_dir}' \
                PARTITIONED BY (category);"
            ))
            .await?;

        // Insert data across two partitions via SQL VALUES with explicit column names.
        let mut values_clauses = Vec::new();
        for i in 0..200 {
            let cat = if i % 2 == 0 { "A" } else { "B" };
            let val = i % 100;
            values_clauses.push(format!("('{cat}', {val})"));
        }
        let values_str = values_clauses.join(", ");

        session
            .sql(&format!(
                "INSERT INTO partitioned_tbl (category, value) VALUES {values_str}"
            ))
            .await?
            .collect()
            .await?;

        // Verify we can read all rows back.
        let table = session.table("partitioned_tbl").await?;
        let count = table.count().await?;
        assert_eq!(
            count, 200,
            "Partitioned write lost rows: expected 200, got {count}"
        );

        // Verify partition directories exist.
        let mut has_a = false;
        let mut has_b = false;
        for entry in WalkDir::new(data_dir)
            .into_iter()
            .filter_entry(|e| e.path().is_dir())
        {
            let entry = entry?;
            if let Ok(path) = entry.path().strip_prefix(data_dir) {
                if path.starts_with("category=A") {
                    has_a = true;
                }
                if path.starts_with("category=B") {
                    has_b = true;
                }
            }
        }
        assert!(has_a, "Expected partition directory category=A");
        assert!(has_b, "Expected partition directory category=B");

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Edge case: Empty input produces no files
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_insert_into_empty_input() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;
        let data_dir = dir.path().to_str().expect("should convert path to str");

        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 1;

        let factory = VortexFormatFactory::new().with_options(opts);
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE empty_tbl \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{data_dir}/';"
            ))
            .await?;

        // Insert an empty batch.
        let empty_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(Vec::<i8>::new()))],
        )?;
        session.register_batch("empty_source", empty_batch)?;

        session
            .sql("INSERT INTO empty_tbl SELECT * FROM empty_source")
            .await?
            .collect()
            .await?;

        let result = session
            .sql("SELECT COUNT(*) as cnt FROM empty_tbl")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("should downcast")
            .value(0);

        assert_eq!(count_value, 0, "Empty input should yield 0 rows");

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Edge case: Single small batch (below target) produces one file
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_insert_into_single_small_batch_with_target() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;
        let data_dir = dir.path().to_str().expect("should convert path to str");

        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 10; // 10 MB target — much larger than the data.

        let factory = VortexFormatFactory::new().with_options(opts);
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE small_tbl \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{data_dir}/';"
            ))
            .await?;

        // A tiny batch — well below the target.
        let entries = 100usize;
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(vec![7i8; entries]))],
        )?;
        session.register_batch("small_source", batch)?;

        session
            .sql("INSERT INTO small_tbl SELECT * FROM small_source")
            .await?
            .collect()
            .await?;

        let files: Vec<_> = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "vortex"))
            .collect();

        assert_eq!(
            files.len(),
            1,
            "A single small batch should produce exactly 1 file, got {}",
            files.len()
        );

        let result = session
            .sql("SELECT COUNT(*) as cnt FROM small_tbl")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("should downcast")
            .value(0);

        assert_eq!(count_value, entries as i64);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Edge case: Very small target triggers many files
    // -----------------------------------------------------------------------

    /// Using a very small target_file_size_mb doesn't work via the options
    /// (minimum is 1 MB), but we can test with exactly 1 MB and a dataset
    /// large enough to produce many files.
    #[tokio::test]
    async fn test_insert_into_many_small_files() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;
        let data_dir = dir.path().to_str().expect("should convert path to str");

        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 1;

        let factory = VortexFormatFactory::new().with_options(opts);
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE many_files_tbl \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{data_dir}/';"
            ))
            .await?;

        // ~10 MB of Int8 data with a 1 MB target should produce several files.
        let entries: usize = 10_000_000;
        let values: Vec<i8> = (0..entries).map(|i| (i % 50) as i8).collect();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(values))],
        )?;
        session.register_batch("many_source", batch)?;

        session
            .sql("INSERT INTO many_files_tbl SELECT * FROM many_source")
            .await?
            .collect()
            .await?;

        let files: Vec<_> = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "vortex"))
            .collect();

        assert!(
            files.len() >= 3,
            "Expected at least 3 files for ~10 MB data with 1 MB target, got {}",
            files.len()
        );

        // Verify all data is preserved.
        let result = session
            .sql("SELECT COUNT(*) as cnt FROM many_files_tbl")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("should downcast")
            .value(0);

        assert_eq!(count_value, entries as i64);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Integration test: target_file_size_mb = 0 defaults to 16 MB
    // -----------------------------------------------------------------------

    /// With `target_file_size_mb=0`, the writer should default to 16 MB and
    /// still use the custom size-based writer. Data integrity should be
    /// preserved regardless.
    #[tokio::test]
    async fn test_target_file_size_zero_defaults_to_16mb() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;
        let data_dir = dir.path().to_str().expect("should convert path to str");

        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 0;

        let factory = VortexFormatFactory::new().with_options(opts);
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE nosplit_tbl \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{data_dir}/';"
            ))
            .await?;

        // 100K Int8 values is well below 16 MB, so should produce a single file.
        let entries: usize = 100_000;
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(vec![1i8; entries]))],
        )?;
        session.register_batch("nosplit_source", batch)?;

        session
            .sql("INSERT INTO nosplit_tbl SELECT * FROM nosplit_source")
            .await?
            .collect()
            .await?;

        // Verify data integrity.
        let result = session
            .sql("SELECT COUNT(*) as cnt FROM nosplit_tbl")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("should downcast")
            .value(0);

        assert_eq!(count_value, entries as i64);

        // Small data with a 16 MB default should produce exactly 1 file.
        let files: Vec<_> = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "vortex"))
            .collect();

        assert_eq!(
            files.len(),
            1,
            "Small data with default 16 MB target (from target_file_size_mb=0) should produce 1 file, got {}",
            files.len()
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Integration test: Multi-column data round-trip with target file size
    // -----------------------------------------------------------------------

    /// Verify that multi-column, multi-type data survives a round-trip through
    /// the file-size-limited write path.
    #[tokio::test]
    async fn test_multi_column_round_trip_with_target_file_size() -> anyhow::Result<()> {
        use datafusion::arrow::array::Float64Array;
        use datafusion::arrow::array::Int32Array;
        use datafusion::arrow::array::Int64Array;
        use datafusion::arrow::array::StringArray;

        let dir = TempDir::new()?;
        let data_dir = dir.path().to_str().expect("should convert path to str");

        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 1;

        let factory = VortexFormatFactory::new().with_options(opts);
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE multi_col_tbl \
                    (id INT NOT NULL, name VARCHAR NOT NULL, score DOUBLE NOT NULL) \
                STORED AS vortex \
                LOCATION '{data_dir}/';"
            ))
            .await?;

        let entries: usize = 500_000;
        let ids: Vec<i32> = (0..entries).map(|i| i as i32).collect();
        let names: Vec<String> = (0..entries).map(|i| format!("item_{}", i % 1000)).collect();
        let scores: Vec<f64> = (0..entries).map(|i| (i as f64) * 0.1).collect();

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("name", DataType::Utf8, false),
                Field::new("score", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
                Arc::new(Float64Array::from(scores)),
            ],
        )?;
        session.register_batch("multi_source", batch)?;

        session
            .sql("INSERT INTO multi_col_tbl SELECT * FROM multi_source")
            .await?
            .collect()
            .await?;

        // Verify row count.
        let result = session
            .sql("SELECT COUNT(*) as cnt FROM multi_col_tbl")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("should downcast")
            .value(0);

        assert_eq!(count_value, entries as i64);

        // Verify value ranges are preserved.
        let stats = session
            .sql("SELECT MIN(id) as min_id, MAX(id) as max_id, MIN(score) as min_score FROM multi_col_tbl")
            .await?
            .collect()
            .await?;

        let min_id = stats[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("should downcast")
            .value(0);
        let max_id = stats[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("should downcast")
            .value(0);

        assert_eq!(min_id, 0);
        assert_eq!(max_id, (entries - 1) as i32);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Integration test: Multiple sequential INSERT INTOs with target file size
    // -----------------------------------------------------------------------

    /// Verify that multiple INSERT INTO operations into the same table with
    /// target file size accumulate data correctly.
    #[tokio::test]
    async fn test_multiple_inserts_with_target_file_size() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;
        let data_dir = dir.path().to_str().expect("should convert path to str");

        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 1;

        let factory = VortexFormatFactory::new().with_options(opts);
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE multi_insert_tbl \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{data_dir}/';"
            ))
            .await?;

        // Perform three separate insertions.
        let batch_size = 1_000_000usize;
        for i in 0..3 {
            let values: Vec<i8> = (0..batch_size)
                .map(|j| ((i * batch_size + j) % 100) as i8)
                .collect();
            let batch = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
                vec![Arc::new(Int8Array::from(values))],
            )?;
            session.register_batch(&format!("batch_{i}"), batch)?;

            session
                .sql(&format!(
                    "INSERT INTO multi_insert_tbl SELECT * FROM batch_{i}"
                ))
                .await?
                .collect()
                .await?;
        }

        // Total should be 3 * batch_size.
        let result = session
            .sql("SELECT COUNT(*) as cnt FROM multi_insert_tbl")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("should downcast")
            .value(0);

        assert_eq!(count_value, (3 * batch_size) as i64);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Integration test: COPY TO without target file size (default behavior)
    // -----------------------------------------------------------------------

    /// Verify that COPY TO with default options (16 MB target) still works
    /// correctly for small datasets — producing a single file.
    #[tokio::test]
    async fn test_copy_to_small_data_default_target() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;

        let factory = VortexFormatFactory::new();
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        // Small dataset — should fit in a single file with the default 16 MB target.
        let entries = 1000usize;
        let data = session.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(vec![42i8; entries]))],
        )?)?;

        let logical_plan = LogicalPlanBuilder::copy_to(
            data.logical_plan().clone(),
            dir.path()
                .to_str()
                .expect("should convert path to str")
                .to_string(),
            format_as_file_type(Arc::new(VortexFormatFactory::new())),
            Default::default(),
            vec![],
        )?
        .build()?;

        session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        let files: Vec<_> = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "vortex"))
            .collect();

        assert_eq!(
            files.len(),
            1,
            "Small data with default 16 MB target should produce 1 file, got {}",
            files.len()
        );

        // Read it back.
        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE copy_small \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{}/';",
                dir.path().to_str().expect("should convert path to str")
            ))
            .await?;

        let result = session
            .sql("SELECT COUNT(*) as cnt FROM copy_small")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("should downcast")
            .value(0);

        assert_eq!(count_value, entries as i64);

        Ok(())
    }
}
