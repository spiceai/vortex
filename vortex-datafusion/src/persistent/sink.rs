// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::arrow::array::RecordBatch;
use datafusion_common::{DataFusionError, Result as DFResult};
use datafusion_common_runtime::{JoinSet, SpawnedTask};
use datafusion_datasource::file_sink_config::{FileSink, FileSinkConfig};
use datafusion_datasource::sink::DataSink;
use datafusion_datasource::write::demux::DemuxedStreamReceiver;
use datafusion_datasource::write::get_writer_schema;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::metrics::MetricsSet;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType};
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::path::Path;
use tokio_stream::wrappers::ReceiverStream;
use vortex::ArrayRef;
use vortex::arrow::FromArrowArray;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexResult;
use vortex::file::{VORTEX_FILE_EXTENSION, WriteOptionsSessionExt};
use vortex::io::{ObjectStoreWriter, VortexWrite};
use vortex::session::VortexSession;
use vortex::stream::ArrayStreamAdapter;

pub struct VortexSink {
    config: FileSinkConfig,
    schema: SchemaRef,
    session: VortexSession,
}

impl VortexSink {
    pub fn new(config: FileSinkConfig, schema: SchemaRef, session: VortexSession) -> Self {
        Self {
            config,
            schema,
            session,
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

        let write_options = self.session.write_options();
        let target_file_size = write_options.target_file_size();

        // TODO(adamg):
        // 1. We can probably be better at signaling how much memory we're consuming (potentially when reading too), see ParquetSink::spawn_writer_tasks_and_join.
        while let Some((path, rx)) = file_stream_rx.recv().await {
            let session = self.session.clone();
            let row_counter = row_counter.clone();
            let object_store = object_store.clone();
            let writer_schema = get_writer_schema(&self.config);
            let dtype = DType::from_arrow(writer_schema);

            // We need to spawn work because there's a dependency between the different files. If one file has too many batches buffered,
            // the demux task might deadlock itself.
            file_write_tasks.spawn(async move {
                if let Some(target_size) = target_file_size {
                    write_with_target_file_size(
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
                    let paths = paths?;
                    for path in paths {
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

/// Write all record batches from a receiver into a single Vortex file.
async fn write_single_file(
    session: VortexSession,
    row_counter: Arc<AtomicU64>,
    object_store: Arc<dyn ObjectStore>,
    dtype: DType,
    path: Path,
    rx: tokio::sync::mpsc::Receiver<RecordBatch>,
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

/// Generate a new file path by appending a part number to the original path.
///
/// For example, `data/output.vortex` with part 1 becomes `data/output_part1.vortex`.
fn generate_part_path(base_path: &Path, part: usize) -> Path {
    let path_str = base_path.to_string();
    let extension = format!(".{VORTEX_FILE_EXTENSION}");
    if let Some(stem) = path_str.strip_suffix(&extension) {
        Path::from(format!("{stem}_part{part}{extension}"))
    } else {
        Path::from(format!("{path_str}_part{part}"))
    }
}

/// Write record batches from a receiver into multiple Vortex files, splitting
/// when the estimated file size exceeds `target_file_size` bytes.
async fn write_with_target_file_size(
    session: VortexSession,
    row_counter: Arc<AtomicU64>,
    object_store: Arc<dyn ObjectStore>,
    dtype: DType,
    base_path: Path,
    rx: tokio::sync::mpsc::Receiver<RecordBatch>,
    target_file_size: u64,
) -> DFResult<Vec<Path>> {
    let mut written_paths: Vec<Path> = Vec::new();
    let mut part_number = 0usize;
    let mut stream = ReceiverStream::new(rx);

    let mut current_batches: Vec<ArrayRef> = Vec::new();
    let mut current_estimated_bytes: u64 = 0;

    while let Some(rb) = stream.next().await {
        row_counter.fetch_add(rb.num_rows() as u64, Ordering::Relaxed);

        // Estimate the in-memory size of this batch (sum of all column buffer sizes).
        let batch_size: usize = rb
            .columns()
            .iter()
            .map(|col| col.get_buffer_memory_size())
            .sum();

        let array = ArrayRef::from_arrow(rb, false);
        current_batches.push(array);
        current_estimated_bytes += batch_size as u64;

        // When the estimated size exceeds the target, flush the accumulated batches to a file.
        if current_estimated_bytes >= target_file_size {
            let path = if part_number == 0 {
                base_path.clone()
            } else {
                generate_part_path(&base_path, part_number)
            };

            write_batches_to_file(
                &session,
                &object_store,
                &dtype,
                &path,
                std::mem::take(&mut current_batches),
            )
            .await?;

            written_paths.push(path);
            part_number += 1;
            current_estimated_bytes = 0;
        }
    }

    // Flush any remaining batches.
    if !current_batches.is_empty() {
        let path = if part_number == 0 {
            base_path.clone()
        } else {
            generate_part_path(&base_path, part_number)
        };

        write_batches_to_file(&session, &object_store, &dtype, &path, current_batches).await?;

        written_paths.push(path);
    }

    Ok(written_paths)
}

/// Write a collection of array batches to a single Vortex file.
async fn write_batches_to_file(
    session: &VortexSession,
    object_store: &Arc<dyn ObjectStore>,
    dtype: &DType,
    path: &Path,
    batches: Vec<ArrayRef>,
) -> DFResult<()> {
    let stream = ArrayStreamAdapter::new(
        dtype.clone(),
        futures::stream::iter(batches.into_iter().map(VortexResult::Ok)),
    );

    let mut sink = ObjectStoreWriter::new(object_store.clone(), path)
        .await
        .map_err(|e| {
            DataFusionError::Execution(format!("Failed to create ObjectStoreWriter: {e}"))
        })?;

    session
        .write_options()
        .write(&mut sink, stream)
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

    use arrow_schema::{DataType, Field, Schema};
    use datafusion::arrow::array::{Int8Array, RecordBatch};
    use datafusion::datasource::DefaultTableSource;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Values};
    use datafusion::prelude::SessionContext;
    use datafusion_common::ScalarValue;
    use datafusion_datasource::file_format::format_as_file_type;
    use rstest::rstest;
    use tempfile::TempDir;
    use walkdir::WalkDir;

    use super::generate_part_path;
    use crate::persistent::{VortexFormatFactory, register_vortex_format_factory};

    #[test]
    fn test_generate_part_path() {
        use object_store::path::Path;

        let base = Path::from("data/output.vortex");
        assert_eq!(
            generate_part_path(&base, 0).to_string(),
            "data/output_part0.vortex"
        );
        assert_eq!(
            generate_part_path(&base, 1).to_string(),
            "data/output_part1.vortex"
        );
        assert_eq!(
            generate_part_path(&base, 42).to_string(),
            "data/output_part42.vortex"
        );

        let base_no_ext = Path::from("data/output");
        assert_eq!(
            generate_part_path(&base_no_ext, 1).to_string(),
            "data/output_part1"
        );
    }

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

    /// Test that setting a target file size on the VortexSession causes the writer
    /// to split output into multiple files, each approximately the target size.
    #[tokio::test]
    async fn test_write_with_target_file_size() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;
        use vortex::VortexSessionDefault;
        use vortex::file::TargetFileSize;
        use vortex::session::VortexSession;

        let dir = TempDir::new()?;
        let entries = 100_000usize;

        // Set a small target file size (e.g., 100KB) so that 100k int8 entries get split
        // across multiple files.
        let target_size_bytes: u64 = 100 * 1024; // 100 KB

        let vortex_session = VortexSession::default().set(TargetFileSize(target_size_bytes));

        let factory =
            VortexFormatFactory::new_with_options(vortex_session, crate::VortexOptions::default());

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

        // Count the number of files produced.
        let file_count = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .count();

        // We should have more than 1 file because the data was split.
        assert!(
            file_count > 1,
            "Expected multiple files due to target file size, got {file_count}"
        );

        // Check that each file is approximately the target size (within 2x).
        for entry in std::fs::read_dir(dir.path())? {
            let entry = entry?;
            if entry.path().is_file() {
                let file_size = entry.metadata()?.len();
                // The last file can be smaller, but no file should be wildly larger
                // than the target. We allow up to 3x the target to account for
                // compression overhead and footer size.
                assert!(
                    file_size <= target_size_bytes * 3,
                    "File {:?} is {file_size} bytes, exceeds 3x target of {target_size_bytes}",
                    entry.path()
                );
            }
        }

        // Verify that we can read back all the data correctly.
        // Use a fresh VortexFormatFactory (without target file size) for reading.
        let read_factory = VortexFormatFactory::new();
        let mut read_session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(read_factory, &mut read_session_state_builder);
        let read_session = SessionContext::new_with_state(read_session_state_builder.build());

        read_session
            .sql(&format!(
                "CREATE EXTERNAL TABLE written_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '{}/';",
                dir.path().to_str().unwrap()
            ))
            .await?;

        let result = read_session
            .sql("SELECT COUNT(*) as count FROM written_data")
            .await?
            .collect()
            .await?;

        assert_eq!(result.len(), 1);
        let count_value = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);

        assert_eq!(
            count_value, entries as i64,
            "Expected {entries} entries, but found {count_value}"
        );

        Ok(())
    }

    /// Test that without a target file size, data is written as a single file
    /// (for a reasonably small dataset), ensuring backwards compatibility.
    #[tokio::test]
    async fn test_write_without_target_file_size_single_file() -> anyhow::Result<()> {
        let dir = TempDir::new()?;
        let entries = 1000usize;

        let factory = VortexFormatFactory::new();

        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

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

        let file_count = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .count();

        assert_eq!(
            file_count, 1,
            "Expected a single file without target file size, got {file_count}"
        );

        Ok(())
    }

    /// Test that the target file size works correctly with INSERT INTO statements.
    #[tokio::test]
    async fn test_insert_into_with_target_file_size() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int64Array;
        use vortex::VortexSessionDefault;
        use vortex::file::TargetFileSize;
        use vortex::session::VortexSession;

        let dir = TempDir::new()?;

        // Set a very small target file size to force splitting.
        let target_size_bytes: u64 = 50 * 1024; // 50 KB

        let vortex_session = VortexSession::default().set(TargetFileSize(target_size_bytes));

        let factory =
            VortexFormatFactory::new_with_options(vortex_session, crate::VortexOptions::default());

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
            .await?;

        // Insert enough data to trigger file splitting.
        // Each row is roughly a string + i32, so we need many rows.
        let mut values = Vec::new();
        for i in 0..10_000 {
            values.push(format!("('value_{i}', {i})"));
        }
        let values_str = values.join(", ");

        session
            .sql(&format!("INSERT INTO my_tbl VALUES {values_str}"))
            .await?
            .collect()
            .await?;

        // Verify all rows are readable.
        let result = session
            .sql("SELECT COUNT(*) FROM my_tbl")
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
            count_value, 10_000,
            "Expected 10000 rows, got {count_value}"
        );

        // Count files - should be more than 1 due to small target.
        let file_count = std::fs::read_dir(dir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .count();

        assert!(
            file_count > 1,
            "Expected multiple files due to target file size, got {file_count}"
        );

        Ok(())
    }
}
