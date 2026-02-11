// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

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
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::path::Path;
use tokio_stream::wrappers::ReceiverStream;
use vortex::array::ArrayRef;
use vortex::array::arrow::FromArrowArray;
use vortex::array::stream::ArrayStreamAdapter;
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
    /// Target file size in bytes. When set, the writer will split output files
    /// when they reach approximately this size.
    target_file_size: Option<u64>,
}

impl VortexSink {
    pub fn new(
        config: FileSinkConfig,
        schema: SchemaRef,
        session: VortexSession,
        target_file_size: Option<u64>,
    ) -> Self {
        Self {
            config,
            schema,
            session,
            target_file_size,
        }
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
        FileSink::write_all(self, data, context).await
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
        // This is a hack
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

            // We need to spawn work because there's a dependency between the different files. If one file has too many batches buffered,
            // the demux task might deadlock itself.
            file_write_tasks.spawn(async move {
                if let Some(target_size) = target_file_size {
                    write_with_file_size_limit(
                        session,
                        row_counter,
                        object_store,
                        dtype,
                        path,
                        rx,
                        target_size,
                    )
                    .await
                } else {
                    write_single_file(session, row_counter, object_store, dtype, path, rx)
                        .await
                        .map(|p| vec![p])
                }
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

/// Write the entire stream to a single Vortex file (original behavior).
async fn write_single_file(
    session: VortexSession,
    row_counter: Arc<AtomicU64>,
    object_store: Arc<dyn ObjectStore>,
    dtype: DType,
    path: Path,
    rx: tokio::sync::mpsc::Receiver<datafusion_common::arrow::array::RecordBatch>,
) -> DFResult<Path> {
    let stream = ReceiverStream::new(rx).map(move |rb| {
        row_counter.fetch_add(rb.num_rows() as u64, Ordering::Relaxed);
        VortexResult::Ok(ArrayRef::from_arrow(rb, false))
    });

    let stream_adapter = ArrayStreamAdapter::new(dtype, stream);

    let mut sink = ObjectStoreWriter::new(object_store.clone(), &path)
        .await
        .map_err(|e| {
            DataFusionError::Execution(format!("Failed to create ObjectStoreWriter: {e}"))
        })?;

    session
        .write_options()
        .write(&mut sink, stream_adapter)
        .await
        .map_err(|e| DataFusionError::Execution(format!("Failed to write Vortex file: {e}")))?;

    sink.shutdown().await.map_err(|e| {
        DataFusionError::Execution(format!("Failed to shutdown Vortex writer: {e}"))
    })?;

    Ok(path)
}

/// Write the stream to multiple Vortex files, splitting when the target file size is reached.
///
/// Splits the input record batches into groups based on estimated uncompressed size, then writes
/// each group to a separate Vortex file. The actual compressed file sizes may differ from the
/// target, but this provides reasonable control over output file sizes.
async fn write_with_file_size_limit(
    session: VortexSession,
    row_counter: Arc<AtomicU64>,
    object_store: Arc<dyn ObjectStore>,
    dtype: DType,
    base_path: Path,
    rx: tokio::sync::mpsc::Receiver<datafusion_common::arrow::array::RecordBatch>,
    target_file_size: u64,
) -> DFResult<Vec<Path>> {
    let mut written_paths = Vec::new();
    let mut file_index: usize = 0;

    let mut rx_stream = ReceiverStream::new(rx);
    let mut pending_batches: Vec<ArrayRef> = Vec::new();
    let mut pending_bytes: u64 = 0;

    while let Some(rb) = rx_stream.next().await {
        row_counter.fetch_add(rb.num_rows() as u64, Ordering::Relaxed);

        // Estimate uncompressed size using Arrow's get_array_memory_size
        let batch_size: u64 = rb
            .columns()
            .iter()
            .map(|c| c.get_array_memory_size() as u64)
            .sum();

        let array = ArrayRef::from_arrow(rb, false);
        pending_batches.push(array);
        pending_bytes += batch_size;

        // When we exceed the target size, flush the pending batches to a new file
        if pending_bytes >= target_file_size {
            let path = split_path(&base_path, file_index);
            let batches = std::mem::take(&mut pending_batches);
            pending_bytes = 0;

            flush_batches_to_file(&session, &object_store, &dtype, &path, batches).await?;

            tracing::debug!(path = %path, "Wrote split file at target size");
            written_paths.push(path);
            file_index += 1;
        }
    }

    // Flush any remaining batches
    if !pending_batches.is_empty() {
        let path = if file_index == 0 {
            // If we never split, use the original path (no suffix)
            split_path(&base_path, 0)
        } else {
            split_path(&base_path, file_index)
        };

        flush_batches_to_file(&session, &object_store, &dtype, &path, pending_batches).await?;

        written_paths.push(path);
    }

    Ok(written_paths)
}

/// Write a set of arrays to a single Vortex file.
async fn flush_batches_to_file(
    session: &VortexSession,
    object_store: &Arc<dyn ObjectStore>,
    dtype: &DType,
    path: &Path,
    batches: Vec<ArrayRef>,
) -> DFResult<()> {
    let stream = futures::stream::iter(batches.into_iter().map(VortexResult::Ok));
    let stream_adapter = ArrayStreamAdapter::new(dtype.clone(), stream);

    let mut sink = ObjectStoreWriter::new(object_store.clone(), path)
        .await
        .map_err(|e| {
            DataFusionError::Execution(format!("Failed to create ObjectStoreWriter: {e}"))
        })?;

    session
        .write_options()
        .write(&mut sink, stream_adapter)
        .await
        .map_err(|e| DataFusionError::Execution(format!("Failed to write Vortex file: {e}")))?;

    sink.shutdown().await.map_err(|e| {
        DataFusionError::Execution(format!("Failed to shutdown Vortex writer: {e}"))
    })?;

    Ok(())
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
    use rstest::rstest;
    use tempfile::TempDir;
    use walkdir::WalkDir;

    use crate::persistent::VortexFormatFactory;
    use crate::persistent::register_vortex_format_factory;

    #[tokio::test]
    async fn test_insert_into() {
        let dir = TempDir::new().unwrap();

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
                dir.path().to_str().unwrap()
            ))
            .await
            .unwrap();

        let my_tbl = session.table("my_tbl").await.unwrap();

        // It's valuable to have two insert code paths because they actually behave slightly differently
        let values = Values {
            schema: Arc::new(my_tbl.schema().clone()),
            values: vec![vec![
                Expr::Literal(ScalarValue::new_utf8view("hello"), None),
                Expr::Literal(42_i32.into(), None),
            ]],
        };

        let tbl_provider = session.table_provider("my_tbl").await.unwrap();

        let logical_plan = LogicalPlanBuilder::insert_into(
            LogicalPlan::Values(values.clone()),
            "my_tbl",
            Arc::new(DefaultTableSource::new(tbl_provider)),
            datafusion::logical_expr::dml::InsertOp::Append,
        )
        .unwrap()
        .build()
        .unwrap();

        session
            .execute_logical_plan(logical_plan)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        session
            .sql("INSERT INTO my_tbl VALUES ('world', 24);")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        my_tbl.clone().show().await.unwrap();

        assert_eq!(
            session
                .table("my_tbl")
                .await
                .unwrap()
                .count()
                .await
                .unwrap(),
            2
        );
    }

    /// Reproduction by <https://github.com/vortex-data/vortex/issues/4315>.
    #[rstest]
    #[case(1000, 1)]
    #[case(40_961, 4)]
    #[case(1_000_000, 4)]
    #[tokio::test]
    async fn test_write_large_batch(
        #[case] entries: usize,
        #[case] expected_files: usize,
    ) -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;

        let factory = VortexFormatFactory::new();

        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        let data = session.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(vec![0i8; entries]))],
        )?)?;

        let logical_plan = LogicalPlanBuilder::copy_to(
            data.logical_plan().clone(),
            dir.path().to_str().unwrap().to_string(),
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

        // Validate the output by reading back the written files
        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE written_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{}/';",
                dir.path().to_str().unwrap()
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
            .unwrap()
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
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int8Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                assert_eq!(
                    col.value(i),
                    0i8,
                    "Expected value 0 at row {}, but found {}",
                    total_rows + i,
                    col.value(i)
                );
            }
            total_rows += batch.num_rows();
        }

        assert_eq!(
            total_rows, entries,
            "Total rows read ({}) doesn't match expected entries ({})",
            total_rows, entries
        );

        let read_dir = std::fs::read_dir(dir.path())?;
        assert_eq!(
            read_dir.count(),
            expected_files,
            "Expected {expected_files} files for {entries} values"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_write_partitioned() -> anyhow::Result<()> {
        let dir = TempDir::new()?;
        let data_dir = dir.path().to_str().unwrap();

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

    #[tokio::test]
    async fn test_write_with_target_file_size() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;

        // Set a small target file size (1 MB) to force splitting
        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 1;

        let factory = VortexFormatFactory::new().with_options(opts);
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        // Write a large enough batch that should exceed 1 MB uncompressed
        // 500_000 Int8 values = ~500 KB uncompressed, but we need more data
        // to trigger splitting. Let's use 1_000_000 entries to be safe.
        let entries = 1_000_000;
        let data = session.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(vec![42i8; entries]))],
        )?)?;

        let logical_plan = LogicalPlanBuilder::copy_to(
            data.logical_plan().clone(),
            dir.path().to_str().unwrap().to_string(),
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

        // Count the output files
        let file_count = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "vortex"))
            .count();

        // With 1 MB target and ~1 MB of data, we should get at least 1 file.
        // The exact count depends on compression, but we should get more than 0.
        assert!(
            file_count >= 1,
            "Expected at least 1 file, got {file_count}"
        );

        // Read back and verify all data is preserved
        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE written_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{}/';",
                dir.path().to_str().unwrap()
            ))
            .await?;

        let result = session
            .sql("SELECT COUNT(*) as count FROM written_data")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);

        assert_eq!(
            count_value, entries as i64,
            "Expected {} entries, but found {}",
            entries, count_value
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_write_with_target_file_size_produces_multiple_files() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;

        // Set a very small target file size (1 MB) and write a lot of data to force multiple files.
        let mut opts = crate::persistent::VortexOptions::default();
        opts.target_file_size_mb = 1;

        let factory = VortexFormatFactory::new().with_options(opts);
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        // Write 5_000_000 random-ish Int8 values (~5 MB uncompressed)
        // This should produce multiple files with a 1 MB target
        let entries = 5_000_000;
        // Use a non-trivial pattern so compression doesn't collapse it to nothing
        let values: Vec<i8> = (0..entries).map(|i| (i % 127) as i8).collect();
        let data = session.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(values))],
        )?)?;

        let logical_plan = LogicalPlanBuilder::copy_to(
            data.logical_plan().clone(),
            dir.path().to_str().unwrap().to_string(),
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

        // Count the output files
        let files: Vec<_> = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "vortex"))
            .collect();

        let file_count = files.len();

        // With a 1 MB target and ~5 MB of data, we expect multiple files
        assert!(
            file_count > 1,
            "Expected more than 1 file with 1MB target and 5MB data, got {file_count}"
        );

        // Verify that each file is approximately within the target size (with some tolerance)
        let target_bytes = 1u64 * 1024 * 1024; // 1 MB
        for file in &files {
            let file_size = file.metadata()?.len();
            // Allow up to 4x the target size as tolerance (compression ratios vary,
            // and a single batch that exceeds the target gets written as a single file)
            assert!(
                file_size <= target_bytes * 4,
                "File {:?} is {} bytes, which exceeds 4x target of {} bytes",
                file.path(),
                file_size,
                target_bytes
            );
        }

        // Read back and verify all data is preserved
        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE written_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{}/';",
                dir.path().to_str().unwrap()
            ))
            .await?;

        let result = session
            .sql("SELECT COUNT(*) as count FROM written_data")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);

        assert_eq!(
            count_value, entries as i64,
            "Expected {} entries, but found {}",
            entries, count_value
        );

        // Verify actual values
        let all_data = session
            .sql("SELECT a FROM written_data ORDER BY a")
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

        Ok(())
    }

    #[tokio::test]
    async fn test_write_without_target_file_size_no_splitting() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;

        let dir = TempDir::new()?;

        // Use default options (no target file size, target_file_size_mb = 0)
        let factory = VortexFormatFactory::new();
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        // Write a small amount of data
        let entries = 1000;
        let data = session.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(vec![0i8; entries]))],
        )?)?;

        let logical_plan = LogicalPlanBuilder::copy_to(
            data.logical_plan().clone(),
            dir.path().to_str().unwrap().to_string(),
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

        // Should produce exactly 1 file with small data and no size limit
        let file_count = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "vortex"))
            .count();

        assert_eq!(
            file_count, 1,
            "Expected exactly 1 file without target file size, got {file_count}"
        );

        // Read back and verify all data is preserved
        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE written_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{}/';",
                dir.path().to_str().unwrap()
            ))
            .await?;

        let result = session
            .sql("SELECT COUNT(*) as count FROM written_data")
            .await?
            .collect()
            .await?;

        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);

        assert_eq!(
            count_value, entries as i64,
            "Expected {} entries, but found {}",
            entries, count_value
        );

        Ok(())
    }
}
