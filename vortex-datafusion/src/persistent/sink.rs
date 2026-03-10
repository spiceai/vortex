// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::Result as DFResult;
use datafusion_common::arrow::array::RecordBatch;
use datafusion_common::arrow::array::RecordBatchOptions;
use datafusion_common::arrow::array::StructArray;
use datafusion_common::arrow::array::UInt64Builder;
use datafusion_common::arrow::array::cast::AsArray;
use datafusion_common::arrow::compute::take;
use datafusion_common::arrow::util::display::array_value_to_string;
use datafusion_common::exec_datafusion_err;
use datafusion_datasource::ListingTableUrl;
use datafusion_datasource::file_sink_config::FileSinkConfig;
use datafusion_datasource::sink::DataSink;
use datafusion_datasource::write::get_writer_schema;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::DisplayAs;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::metrics::MetricsSet;
use futures::StreamExt;
use object_store::ObjectStore;
use object_store::path::Path;
use uuid::Uuid;
use vortex::array::ArrayRef;
use vortex::array::arrow::FromArrowArray;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteSummary;
use vortex::io::VortexWrite;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

struct WriteOutputOptions<'a> {
    base_output_path: &'a ListingTableUrl,
    target_file_size: Option<u64>,
    extension: &'a str,
    write_id: &'a str,
}

struct PartitionWriteOptions<'a> {
    partition_by: &'a [(String, DataType)],
    keep_partition_by_columns: bool,
}

pub struct VortexSink {
    config: FileSinkConfig,
    schema: SchemaRef,
    session: VortexSession,
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

    fn base_output_path(&self) -> &ListingTableUrl {
        &self.config.table_paths[0]
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
        let object_store = context
            .runtime_env()
            .object_store(&self.config.object_store_url)?;
        let writer_schema = get_writer_schema(&self.config);
        let dtype = DType::from_arrow(writer_schema);
        let write_id = Uuid::new_v4().simple().to_string();

        let summaries = if self.config.table_partition_cols.is_empty() {
            write_record_batch_stream_to_files(
                self.session.clone(),
                object_store,
                dtype,
                data,
                &WriteOutputOptions {
                    base_output_path: self.base_output_path(),
                    target_file_size: self.target_file_size,
                    extension: &self.config.file_extension,
                    write_id: &write_id,
                },
            )
            .await?
        } else {
            write_partitioned_record_batch_stream_to_files(
                self.session.clone(),
                object_store,
                dtype,
                data,
                &WriteOutputOptions {
                    base_output_path: self.base_output_path(),
                    target_file_size: self.target_file_size,
                    extension: &self.config.file_extension,
                    write_id: &write_id,
                },
                &PartitionWriteOptions {
                    partition_by: &self.config.table_partition_cols,
                    keep_partition_by_columns: self.config.keep_partition_by_columns,
                },
            )
            .await?
        };

        let mut row_count = 0_u64;
        for (path, summary) in summaries {
            row_count += summary.row_count();
            tracing::info!(path = %path, "Successfully written file");
        }

        Ok(row_count)
    }
}

/// Write batches from a single input stream to one or more output files.
///
/// For collection paths, files are emitted as `part-00000.{extension}` and so on.
/// For a single-file path, the original target path is used unless rolling is needed,
/// in which case numbered variants are produced.
async fn write_record_batch_stream_to_files(
    session: VortexSession,
    object_store: Arc<dyn ObjectStore>,
    dtype: DType,
    mut data: SendableRecordBatchStream,
    output_options: &WriteOutputOptions<'_>,
) -> DFResult<Vec<(Path, WriteSummary)>> {
    let target = output_options.target_file_size.map(|t| t.max(1));
    let single_file_output = !output_options.base_output_path.is_collection()
        && output_options.base_output_path.file_extension().is_some();

    let mut results: Vec<(Path, WriteSummary)> = Vec::new();
    let mut buffered_batches: Vec<RecordBatch> = Vec::new();
    let mut buffered_bytes = 0_u64;
    let mut file_index = 0_usize;
    let mut compression_ratio = 1.0_f64;

    while let Some(batch) = data.next().await.transpose()? {
        buffered_bytes = buffered_bytes.saturating_add(batch.get_array_memory_size() as u64);
        buffered_batches.push(batch);

        if let Some(target) = target {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "file sizes won't exceed u64::MAX"
            )]
            let estimated_compressed = (buffered_bytes as f64 * compression_ratio) as u64;
            if estimated_compressed >= target {
                let file_path = output_file_path(
                    output_options.base_output_path,
                    file_index,
                    output_options.extension,
                    single_file_output,
                    output_options.write_id,
                );
                let summary = write_batches(
                    &session,
                    object_store.clone(),
                    file_path.clone(),
                    dtype.clone(),
                    std::mem::take(&mut buffered_batches),
                )
                .await?;

                if buffered_bytes > 0 {
                    compression_ratio = summary.size() as f64 / buffered_bytes as f64;
                }

                results.push((file_path, summary));
                buffered_bytes = 0;
                file_index += 1;
            }
        }
    }

    if !buffered_batches.is_empty() {
        let file_path = output_file_path(
            output_options.base_output_path,
            file_index,
            output_options.extension,
            single_file_output,
            output_options.write_id,
        );
        let summary = write_batches(
            &session,
            object_store,
            file_path.clone(),
            dtype,
            buffered_batches,
        )
        .await?;
        results.push((file_path, summary));
    }

    Ok(results)
}

/// Write batches from a partitioned input stream to one or more output files per partition.
async fn write_partitioned_record_batch_stream_to_files(
    session: VortexSession,
    object_store: Arc<dyn ObjectStore>,
    dtype: DType,
    mut data: SendableRecordBatchStream,
    output_options: &WriteOutputOptions<'_>,
    partition_options: &PartitionWriteOptions<'_>,
) -> DFResult<Vec<(Path, WriteSummary)>> {
    let target = output_options.target_file_size.map(|t| t.max(1));

    struct PartitionState {
        buffered_batches: Vec<RecordBatch>,
        buffered_bytes: u64,
        file_index: usize,
        compression_ratio: f64,
    }

    impl PartitionState {
        fn new() -> Self {
            Self {
                buffered_batches: Vec::new(),
                buffered_bytes: 0,
                file_index: 0,
                compression_ratio: 1.0,
            }
        }
    }

    let mut results: Vec<(Path, WriteSummary)> = Vec::new();
    let mut states: HashMap<Vec<String>, PartitionState> = HashMap::new();

    while let Some(batch) = data.next().await.transpose()? {
        let take_map = compute_take_arrays(&batch, partition_options.partition_by)?;

        for (part_key, mut builder) in take_map {
            let take_indices = builder.finish();
            let part_batch_struct: StructArray = batch.clone().into();
            let part_batch =
                RecordBatch::from(take(&part_batch_struct, &take_indices, None)?.as_struct());

            let final_batch = if partition_options.keep_partition_by_columns {
                part_batch
            } else {
                remove_partition_by_columns(&part_batch, partition_options.partition_by)?
            };

            let state = states
                .entry(part_key.clone())
                .or_insert_with(PartitionState::new);
            state.buffered_bytes = state
                .buffered_bytes
                .saturating_add(final_batch.get_array_memory_size() as u64);
            state.buffered_batches.push(final_batch);

            if let Some(target) = target {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "file sizes won't exceed u64::MAX"
                )]
                let estimated_compressed =
                    (state.buffered_bytes as f64 * state.compression_ratio) as u64;

                if estimated_compressed >= target {
                    let file_path = partition_output_file_path(
                        output_options.base_output_path,
                        partition_options.partition_by,
                        &part_key,
                        state.file_index,
                        output_options.extension,
                        output_options.write_id,
                    );
                    let summary = write_batches(
                        &session,
                        object_store.clone(),
                        file_path.clone(),
                        dtype.clone(),
                        std::mem::take(&mut state.buffered_batches),
                    )
                    .await?;

                    if state.buffered_bytes > 0 {
                        state.compression_ratio =
                            summary.size() as f64 / state.buffered_bytes as f64;
                    }

                    results.push((file_path, summary));
                    state.buffered_bytes = 0;
                    state.file_index += 1;
                }
            }
        }
    }

    for (part_key, mut state) in states {
        if state.buffered_batches.is_empty() {
            continue;
        }

        let file_path = partition_output_file_path(
            output_options.base_output_path,
            partition_options.partition_by,
            &part_key,
            state.file_index,
            output_options.extension,
            output_options.write_id,
        );
        let summary = write_batches(
            &session,
            object_store.clone(),
            file_path.clone(),
            dtype.clone(),
            std::mem::take(&mut state.buffered_batches),
        )
        .await?;
        results.push((file_path, summary));
    }

    Ok(results)
}

fn compute_take_arrays(
    batch: &RecordBatch,
    partition_by: &[(String, DataType)],
) -> DFResult<HashMap<Vec<String>, UInt64Builder>> {
    let mut take_map: HashMap<Vec<String>, UInt64Builder> = HashMap::new();
    let partition_columns: Vec<_> = partition_by
        .iter()
        .map(|(name, _)| {
            batch.column_by_name(name).ok_or_else(|| {
                exec_datafusion_err!("Partition column '{name}' does not exist in source data")
            })
        })
        .collect::<DFResult<_>>()?;

    for row in 0..batch.num_rows() {
        let mut part_key = Vec::with_capacity(partition_columns.len());
        for array in &partition_columns {
            let value = array_value_to_string(array.as_ref(), row).map_err(|e| {
                exec_datafusion_err!("Failed to convert partition value to string: {e}")
            })?;
            part_key.push(value);
        }
        let builder = take_map.entry(part_key).or_default();
        builder.append_value(row as u64);
    }

    Ok(take_map)
}

fn remove_partition_by_columns(
    partitioned_batch: &RecordBatch,
    partition_by: &[(String, DataType)],
) -> DFResult<RecordBatch> {
    let partition_names: Vec<_> = partition_by.iter().map(|(name, _)| name.as_str()).collect();
    let (non_partition_columns, non_partition_fields): (Vec<_>, Vec<_>) = partitioned_batch
        .columns()
        .iter()
        .zip(partitioned_batch.schema().fields())
        .filter(|(_, field)| !partition_names.contains(&field.name().as_str()))
        .map(|(array, field)| (Arc::clone(array), (**field).clone()))
        .unzip();

    let schema = Schema::new(non_partition_fields);
    if non_partition_columns.is_empty() {
        let options =
            RecordBatchOptions::default().with_row_count(Some(partitioned_batch.num_rows()));
        return Ok(RecordBatch::try_new_with_options(
            Arc::new(schema),
            non_partition_columns,
            &options,
        )?);
    }

    Ok(RecordBatch::try_new(
        Arc::new(schema),
        non_partition_columns,
    )?)
}

fn partition_output_file_path(
    base_output_path: &ListingTableUrl,
    partition_by: &[(String, DataType)],
    part_key: &[String],
    file_index: usize,
    extension: &str,
    write_id: &str,
) -> Path {
    let mut path = base_output_path.prefix().clone();
    for (idx, value) in part_key.iter().enumerate() {
        path = path.child(format!("{}={}", partition_by[idx].0, value));
    }

    path.child(format!("{write_id}_{file_index:05}.{extension}"))
}

/// Generate a numbered file path from an existing path for size-based splitting.
///
/// Given `base/file.vortex`, produces `base/file_00000.vortex`.
/// If the path has no recognized extension, appends `_00000.{extension}`.
fn numbered_path(original: &Path, index: usize, extension: &str) -> Path {
    let s = original.to_string();
    let suffix = format!(".{extension}");
    if let Some(stem) = s.strip_suffix(&suffix) {
        Path::from(format!("{stem}_{index:05}{suffix}"))
    } else {
        Path::from(format!("{s}_{index:05}.{extension}"))
    }
}
/// Write a set of [`RecordBatch`]es to a single Vortex file at `path`.
async fn write_batches(
    session: &VortexSession,
    object_store: Arc<dyn ObjectStore>,
    path: Path,
    dtype: DType,
    batches: Vec<RecordBatch>,
) -> DFResult<WriteSummary> {
    let stream = futures::stream::iter(
        batches
            .into_iter()
            .map(|rb| ArrayRef::from_arrow(rb, false)),
    );
    let stream_adapter = ArrayStreamAdapter::new(dtype, stream);

    let mut object_writer = ObjectStoreWrite::new(object_store, &path)
        .await
        .map_err(|e| exec_datafusion_err!("Failed to create ObjectStoreWrite: {e}"))?;

    let summary = session
        .write_options()
        .write(&mut object_writer, stream_adapter)
        .await
        .map_err(|e| exec_datafusion_err!("Failed to write Vortex file: {e}"))?;

    object_writer
        .shutdown()
        .await
        .map_err(|e| exec_datafusion_err!("Failed to shutdown Vortex writer: {e}"))?;

    Ok(summary)
}

/// Build the output path for a rolling write.
fn output_file_path(
    base_output_path: &ListingTableUrl,
    file_index: usize,
    extension: &str,
    single_file_output: bool,
    write_id: &str,
) -> Path {
    if single_file_output {
        if file_index == 0 {
            return base_output_path.prefix().clone();
        }
        return numbered_path(base_output_path.prefix(), file_index, extension);
    }

    let mut base = base_output_path.prefix().to_string();
    if !base.ends_with('/') {
        base.push('/');
    }
    Path::from(format!("{base}{write_id}_{file_index:05}.{extension}"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use datafusion::arrow::array::Int8Array;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::assert_batches_sorted_eq;
    use datafusion::datasource::DefaultTableSource;
    use datafusion::logical_expr::Expr;
    use datafusion::logical_expr::LogicalPlan;
    use datafusion::logical_expr::LogicalPlanBuilder;
    use datafusion::logical_expr::Values;
    use datafusion_common::ScalarValue;
    use datafusion_datasource::ListingTableUrl;
    use datafusion_datasource::file_format::format_as_file_type;
    use futures::TryStreamExt;
    use rstest::rstest;

    use crate::common_tests::TestSessionContext;
    use crate::persistent::VortexFormatFactory;
    use crate::persistent::VortexTableOptions;

    fn split_path(
        base_path: &object_store::path::Path,
        file_index: usize,
        extension: &str,
    ) -> object_store::path::Path {
        let mut base = base_path.to_string();
        if !base.ends_with('/') {
            base.push('/');
        }
        let filename = format!("part-{file_index:05}.{extension}");
        object_store::path::Path::from(format!("{base}{filename}"))
    }

    #[tokio::test]
    async fn test_insert_into_sql() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                    (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/';",
            )
            .await?;

        ctx.session
            .sql("INSERT INTO my_tbl VALUES ('hello', 1), ('world', 2);")
            .await?
            .collect()
            .await?;

        let batches = ctx
            .session
            .sql("SELECT * from my_tbl")
            .await?
            .collect()
            .await?;

        assert_batches_sorted_eq!(
            &[
                "+-------+----+",
                "| c1    | c2 |",
                "+-------+----+",
                "| hello | 1  |",
                "| world | 2  |",
                "+-------+----+",
            ],
            &batches
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_insert_into_logical_plan() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                    (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/';",
            )
            .await?;

        let my_tbl = ctx.session.table("my_tbl").await?;

        // It's valuable to have two insert code paths because they actually behave slightly differently
        let values = Values {
            schema: Arc::new(my_tbl.schema().clone()),
            values: vec![vec![
                Expr::Literal(ScalarValue::new_utf8view("hello"), None),
                Expr::Literal(42_i32.into(), None),
            ]],
        };

        let tbl_provider = ctx.session.table_provider("my_tbl").await?;

        let logical_plan = LogicalPlanBuilder::insert_into(
            LogicalPlan::Values(values.clone()),
            "my_tbl",
            Arc::new(DefaultTableSource::new(tbl_provider.clone())),
            datafusion::logical_expr::dml::InsertOp::Append,
        )?
        .build()?;

        ctx.session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        let batches = ctx.session.read_table(tbl_provider)?.collect().await?;

        assert_batches_sorted_eq!(
            [
                "+-------+----+",
                "| c1    | c2 |",
                "+-------+----+",
                "| hello | 42 |",
                "+-------+----+",
            ],
            &batches
        );

        Ok(())
    }

    /// Reproduction by <https://github.com/vortex-data/vortex/issues/4315>.
    /// Uses a 1MB target file size to exercise file splitting behavior.
    #[rstest]
    #[case(1_000, 1)]
    #[case(5_000_000, 6)]
    #[case(10_000_000, 10)]
    #[tokio::test]
    async fn test_write_large_batch(
        #[case] entries: usize,
        #[case] expected_files: usize,
    ) -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        let opts = VortexTableOptions {
            target_file_size_mb: 1,
            ..Default::default()
        };

        let factory = VortexFormatFactory::new().with_options(opts);

        let values: Vec<i8> = (0..entries)
            .map(|i| i8::try_from(i % 127))
            .collect::<Result<_, _>>()?;

        let data = ctx.session.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(values))],
        )?)?;

        let logical_plan = LogicalPlanBuilder::copy_to(
            data.logical_plan().clone(),
            "/table/".to_string(),
            format_as_file_type(Arc::new(factory)),
            Default::default(),
            vec![],
        )?
        .build()?;

        ctx.session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        let result = ctx
            .session
            .sql("SELECT COUNT(*) as count FROM '/table/'")
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

        let file_metas = ctx
            .store
            .list(Some(&"table".into()))
            .try_collect::<Vec<_>>()
            .await?;

        assert!(
            file_metas.len() >= expected_files,
            "Expected at least {expected_files} files for {entries} values, got {}",
            file_metas.len()
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_write_large_batch_default_target_is_128mb() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        let entries = 1_000_000;
        let values: Vec<i8> = (0..entries)
            .map(|i| i8::try_from(i % 127))
            .collect::<Result<_, _>>()?;

        let data = ctx.session.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int8, false)])),
            vec![Arc::new(Int8Array::from(values))],
        )?)?;

        let logical_plan = LogicalPlanBuilder::copy_to(
            data.logical_plan().clone(),
            "/table/".to_string(),
            format_as_file_type(Arc::new(VortexFormatFactory::new())),
            Default::default(),
            vec![],
        )?
        .build()?;

        ctx.session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        let file_metas = ctx
            .store
            .list(Some(&"/table".into()))
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(file_metas.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_write_partitioned() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        let _unused = ctx
            .session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/' \
                PARTITIONED BY (c1);",
            )
            .await?;

        ctx.session
            .sql("INSERT INTO my_tbl (c1, c2) VALUES ('world', 24), ('world', 25), ('hello', 42);")
            .await?
            .collect()
            .await?;

        let table = ctx.session.table("my_tbl").await?;
        assert_eq!(table.count().await?, 3);

        let location = object_store::path::Path::parse("table/")?;
        let file_metas = ctx
            .store
            .list(Some(&location))
            .try_collect::<Vec<_>>()
            .await?;

        for meta in file_metas.into_iter() {
            let location = meta.location;
            assert!(
                location.prefix_matches(&"c1=hello".into())
                    || location.prefix_matches(&"c1=world".into())
            );
        }

        Ok(())
    }

    /// Verify that partitioned writes produce exactly one file per partition value,
    /// not multiple round-robin files from the DataFusion demuxer.
    #[tokio::test]
    async fn test_write_partitioned_single_file_per_partition() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/' \
                PARTITIONED BY (c1);",
            )
            .await?;

        // Insert enough data that DataFusion's default demuxer (4 parallel outputs)
        // would produce multiple files per partition if not overridden.
        let mut inserts = Vec::new();
        for i in 0..200 {
            inserts.push(format!("('alpha', {})", i));
        }
        for i in 0..200 {
            inserts.push(format!("('beta', {})", i));
        }
        let insert_sql = format!("INSERT INTO my_tbl (c1, c2) VALUES {};", inserts.join(", "));
        ctx.session.sql(&insert_sql).await?.collect().await?;

        let table = ctx.session.table("my_tbl").await?;
        assert_eq!(table.count().await?, 400);

        let all_files = ctx.store.list(None).try_collect::<Vec<_>>().await?;

        // With 2 partition values, we expect exactly 2 files (one per partition).
        assert_eq!(
            all_files.len(),
            2,
            "Expected exactly 2 files (one per partition), got {}",
            all_files.len()
        );

        for meta in all_files.into_iter() {
            let loc = meta.location.to_string();
            assert!(
                loc.contains("c1=alpha") || loc.contains("c1=beta"),
                "Unexpected file path: {loc}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_split_path_basic() {
        let path = object_store::path::Path::from("data/output");
        assert_eq!(
            split_path(&path, 0, "vortex").to_string(),
            "data/output/part-00000.vortex"
        );
        assert_eq!(
            split_path(&path, 12, "vortex").to_string(),
            "data/output/part-00012.vortex"
        );
    }

    #[test]
    fn test_split_path_preserves_trailing_slash() {
        let path = object_store::path::Path::from("nested/path/");
        assert_eq!(
            split_path(&path, 3, "vx").to_string(),
            "nested/path/part-00003.vx"
        );
    }

    #[test]
    fn test_numbered_path() {
        use super::numbered_path;

        let path = object_store::path::Path::from("table/c1=alpha/abc123.vortex");
        assert_eq!(
            numbered_path(&path, 0, "vortex").to_string(),
            "table/c1=alpha/abc123_00000.vortex"
        );
        assert_eq!(
            numbered_path(&path, 5, "vortex").to_string(),
            "table/c1=alpha/abc123_00005.vortex"
        );
    }

    #[test]
    fn test_numbered_path_no_extension() {
        use super::numbered_path;

        let path = object_store::path::Path::from("table/output");
        assert_eq!(
            numbered_path(&path, 0, "vortex").to_string(),
            "table/output_00000.vortex"
        );
    }

    #[test]
    fn test_output_file_path_single_file_and_collection() {
        use super::output_file_path;

        let single = ListingTableUrl::parse("file:///tmp/output.vortex").unwrap();
        assert_eq!(
            output_file_path(&single, 0, "vortex", true, "wid").to_string(),
            "tmp/output.vortex"
        );
        assert_eq!(
            output_file_path(&single, 2, "vortex", true, "wid").to_string(),
            "tmp/output_00002.vortex"
        );

        let collection = ListingTableUrl::parse("file:///tmp/table/").unwrap();
        assert_eq!(
            output_file_path(&collection, 3, "vortex", false, "wid").to_string(),
            "tmp/table/wid_00003.vortex"
        );
    }

    #[test]
    fn test_partition_output_file_path() {
        use super::partition_output_file_path;

        let base = ListingTableUrl::parse("file:///tmp/table/").unwrap();
        let partition_by = vec![
            ("p1".to_string(), DataType::Utf8),
            ("p2".to_string(), DataType::Int32),
        ];
        let part_key = vec!["alpha".to_string(), "42".to_string()];

        let path = partition_output_file_path(&base, &partition_by, &part_key, 7, "vortex", "wid");
        assert_eq!(
            path.to_string(),
            "tmp/table/p1=alpha/p2=42/wid_00007.vortex"
        );
    }

    #[test]
    fn test_compute_take_arrays_groups_rows() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int32Array;
        use datafusion::arrow::array::StringArray;

        use super::compute_take_arrays;

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("part", DataType::Utf8, false),
                Field::new("bucket", DataType::Int32, false),
                Field::new("val", DataType::Int64, false),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b", "a"])),
                Arc::new(Int32Array::from(vec![1, 1, 2, 3])),
                Arc::new(Int64Array::from(vec![10, 11, 12, 13])),
            ],
        )?;

        let partition_by = vec![
            ("part".to_string(), DataType::Utf8),
            ("bucket".to_string(), DataType::Int32),
        ];

        let groups = compute_take_arrays(&batch, &partition_by)?;
        assert_eq!(groups.len(), 3);
        assert!(groups.contains_key(&vec!["a".to_string(), "1".to_string()]));
        assert!(groups.contains_key(&vec!["a".to_string(), "3".to_string()]));
        assert!(groups.contains_key(&vec!["b".to_string(), "2".to_string()]));

        // Every row must be present in exactly one group.
        let total_indices: usize = groups
            .into_values()
            .map(|mut builder| builder.finish().len())
            .sum();
        assert_eq!(total_indices, batch.num_rows());

        Ok(())
    }

    #[test]
    fn test_compute_take_arrays_with_null_partition_values() -> anyhow::Result<()> {
        use datafusion::arrow::array::Int32Array;
        use datafusion::arrow::array::StringArray;

        use super::compute_take_arrays;

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("part", DataType::Utf8, true),
                Field::new("bucket", DataType::Int32, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec![Some("a"), None, Some("a"), None])),
                Arc::new(Int32Array::from(vec![Some(1), Some(1), None, None])),
            ],
        )?;

        let partition_by = vec![
            ("part".to_string(), DataType::Utf8),
            ("bucket".to_string(), DataType::Int32),
        ];

        let groups = compute_take_arrays(&batch, &partition_by)?;
        let total_indices: usize = groups
            .into_values()
            .map(|mut builder| builder.finish().len())
            .sum();

        assert_eq!(total_indices, batch.num_rows());

        Ok(())
    }

    #[test]
    fn test_compute_take_arrays_missing_partition_column_errors() -> anyhow::Result<()> {
        use super::compute_take_arrays;

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("part", DataType::Utf8, false)])),
            vec![Arc::new(datafusion::arrow::array::StringArray::from(vec![
                "a", "b",
            ]))],
        )?;

        let partition_by = vec![("missing".to_string(), DataType::Utf8)];
        let err = compute_take_arrays(&batch, &partition_by).unwrap_err();
        assert!(err.to_string().contains("does not exist in source data"));

        Ok(())
    }

    #[test]
    fn test_remove_partition_by_columns() -> anyhow::Result<()> {
        use datafusion::arrow::array::StringArray;

        use super::remove_partition_by_columns;

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("part", DataType::Utf8, false),
                Field::new("val", DataType::Int64, false),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["x", "y"])),
                Arc::new(Int64Array::from(vec![1, 2])),
            ],
        )?;

        let out = remove_partition_by_columns(&batch, &[("part".to_string(), DataType::Utf8)])?;
        assert_eq!(out.num_columns(), 1);
        assert_eq!(out.schema().field(0).name(), "val");
        assert_eq!(out.num_rows(), 2);

        Ok(())
    }

    #[test]
    fn test_remove_partition_by_columns_all_columns_partitioned() -> anyhow::Result<()> {
        use datafusion::arrow::array::StringArray;

        use super::remove_partition_by_columns;

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("part", DataType::Utf8, false)])),
            vec![Arc::new(StringArray::from(vec!["x", "y", "z"]))],
        )?;

        let out = remove_partition_by_columns(&batch, &[("part".to_string(), DataType::Utf8)])?;
        assert_eq!(out.num_columns(), 0);
        assert_eq!(out.num_rows(), 3);

        Ok(())
    }

    /// Generate `count` pseudo-random i64 values using a simple LCG.
    /// These values resist compression (unlike sequential or modular data),
    /// giving more realistic compressed file sizes.
    fn pseudo_random_i64s(count: usize, seed: i64) -> Vec<i64> {
        let mut v = seed;
        (0..count)
            .map(|_| {
                v = v
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                v
            })
            .collect()
    }

    /// Tests file splitting through the full DataFusion pipeline.
    ///
    /// Writes ~62MB of pseudo-random Int64 data (near 1:1 compression ratio)
    /// via COPY TO with a 16MB target file size. Verifies that exactly 4 files
    /// are produced and each file's compressed size is approximately 16MB.
    ///
    /// This exercises the complete write path including the DataFusion demuxer
    /// and VortexSink, unlike a direct `write_stream_to_files` call.
    #[tokio::test]
    async fn test_file_splitting_62mb_into_4_files() -> anyhow::Result<()> {
        use datafusion::datasource::MemTable;
        use datafusion_datasource::file_format::format_as_file_type;

        let ctx = TestSessionContext::default();

        let target_mb = 16_usize;
        let opts = VortexTableOptions {
            target_file_size_mb: target_mb,
            ..Default::default()
        };
        let factory = VortexFormatFactory::new().with_options(opts);

        let batch_rows = 8192_usize;
        let total_elements = 62 * 1024 * 1024 / 8; // ~8,126,464 i64 values ≈ 62MB Arrow memory
        let num_batches = total_elements / batch_rows;
        let expected_total_rows = (num_batches * batch_rows) as i64;

        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        let mut batches = Vec::new();
        for i in 0..num_batches {
            let values = pseudo_random_i64s(batch_rows, (i * batch_rows) as i64);
            batches.push(RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(values))],
            )?);
        }

        let table = MemTable::try_new(schema.clone(), vec![batches])?;
        ctx.session.register_table("source", Arc::new(table))?;

        let source = ctx.session.table("source").await?;
        let logical_plan = LogicalPlanBuilder::copy_to(
            source.logical_plan().clone(),
            "/table/".to_string(),
            format_as_file_type(Arc::new(factory)),
            Default::default(),
            vec![],
        )?
        .build()?;

        ctx.session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        let file_metas = ctx
            .store
            .list(Some(&"/table".into()))
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(
            file_metas.len(),
            4,
            "Expected 4 files for ~62MB data with {target_mb}MB target, got {} (sizes: {:?})",
            file_metas.len(),
            file_metas.iter().map(|m| m.size).collect::<Vec<_>>()
        );

        let target_bytes = (target_mb * 1024 * 1024) as u64;
        for meta in &file_metas {
            assert!(
                meta.size > target_bytes / 2,
                "File {} is {}B, expected at least {}B (target/2)",
                meta.location,
                meta.size,
                target_bytes / 2
            );
        }

        // Verify total row count.
        let result = ctx
            .session
            .sql("SELECT COUNT(*) as cnt FROM '/table/'")
            .await?
            .collect()
            .await?;

        let count = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);

        assert_eq!(count, expected_total_rows, "Total row count mismatch");

        Ok(())
    }

    /// Tests file splitting with compressible data through the full pipeline.
    ///
    /// Uses low-entropy Int64 values (repeating 0..255) which compress ~8:1 in
    /// Vortex. With the current code that compares Arrow memory size against
    /// `target_file_size`, files are split far too early, producing many tiny
    /// compressed files instead of files that are close to the target.
    ///
    /// For ~32MB of Arrow data (~4MB compressed at 8:1) with a 1MB target:
    ///   - **Correct**: 4 files of ~1MB compressed each
    ///   - **Bug**: 32 files of ~0.125MB compressed each
    #[tokio::test]
    async fn test_file_splitting_compressible_data() -> anyhow::Result<()> {
        use datafusion::datasource::MemTable;
        use datafusion_datasource::file_format::format_as_file_type;

        let ctx = TestSessionContext::default();

        let target_mb = 1_usize;
        let opts = VortexTableOptions {
            target_file_size_mb: target_mb,
            ..Default::default()
        };
        let factory = VortexFormatFactory::new().with_options(opts);

        // Generate low-entropy Int64 values: repeating 0..255.
        // Arrow memory: 4M × 8 bytes = 32MB.
        // Vortex compressed: each value only needs ~1 byte → ~4MB total.
        let total_elements = 4_000_000_usize;
        let batch_rows = 8192_usize;
        let num_batches = total_elements / batch_rows;

        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        let mut batches = Vec::new();
        for i in 0..num_batches {
            let values: Vec<i64> = (0..batch_rows)
                .map(|j| ((i * batch_rows + j) % 256) as i64)
                .collect();
            batches.push(RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(values))],
            )?);
        }

        let table = MemTable::try_new(schema.clone(), vec![batches])?;
        ctx.session.register_table("source", Arc::new(table))?;

        let source = ctx.session.table("source").await?;
        let logical_plan = LogicalPlanBuilder::copy_to(
            source.logical_plan().clone(),
            "/table/".to_string(),
            format_as_file_type(Arc::new(factory)),
            Default::default(),
            vec![],
        )?
        .build()?;

        ctx.session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        let file_metas = ctx
            .store
            .list(Some(&"/table".into()))
            .try_collect::<Vec<_>>()
            .await?;

        // With compressible data, there should be few files (not > 10).
        // The buggy code produces many tiny files because it splits on Arrow
        // memory (32MB / 1MB = 32 files) instead of compressed size (~4MB / 1MB = 4 files).
        let total_compressed: u64 = file_metas.iter().map(|m| m.size).sum();
        let target_bytes = (target_mb * 1024 * 1024) as u64;

        // We should have at most ~(total_compressed / target) + 1 files, not
        // ~(arrow_memory / target) files.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "file counts won't exceed usize"
        )]
        let max_expected = (total_compressed / target_bytes + 2) as usize;
        assert!(
            file_metas.len() <= max_expected,
            "Too many files: got {} but total compressed is {}B with {}B target \
             (expected at most {max_expected}). Files are being split on Arrow memory \
             instead of compressed size. Sizes: {:?}",
            file_metas.len(),
            total_compressed,
            target_bytes,
            file_metas.iter().map(|m| m.size).collect::<Vec<_>>()
        );

        // Every file except the first should be reasonably sized. The first
        // file may be smaller because the compression ratio is unknown until
        // the first write completes.
        for meta in file_metas.iter().skip(1) {
            assert!(
                meta.size > target_bytes / 4,
                "File {} is {}B, far below target {}B — splitting on Arrow memory, not compressed size",
                meta.location,
                meta.size,
                target_bytes
            );
        }

        Ok(())
    }

    /// Tests that partitioned writes with 4 partitions (each ~12MB) and
    /// an 8MB target file size produce 2 files per partition (8 total).
    ///
    /// Uses the full DataFusion pipeline: data is registered as a MemTable,
    /// then COPY TO with `PARTITIONED BY` writes through VortexSink.
    #[tokio::test]
    async fn test_file_splitting_partitioned_4_parts_12mb_each() -> anyhow::Result<()> {
        use datafusion::arrow::array::StringArray;
        use datafusion::datasource::MemTable;
        use datafusion_datasource::file_format::format_as_file_type;

        let ctx = TestSessionContext::default();

        let target_mb = 8_usize;
        let opts = VortexTableOptions {
            target_file_size_mb: target_mb,
            ..Default::default()
        };
        let factory = VortexFormatFactory::new().with_options(opts);

        let batch_rows = 8192_usize;
        let per_partition_elements = 12 * 1024 * 1024 / 8; // 1,572,864 i64 values ≈ 12MB
        let batches_per_partition = per_partition_elements / batch_rows; // 192

        let partition_names = ["p0", "p1", "p2", "p3"];

        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Int64, false),
            Field::new("part", DataType::Utf8, false),
        ]));

        // Build many small batches; each batch belongs to exactly one partition.
        let mut batches = Vec::new();
        for (pi, part_name) in partition_names.iter().enumerate() {
            for bi in 0..batches_per_partition {
                let seed = (pi * batches_per_partition + bi) as i64 * batch_rows as i64;
                let values = pseudo_random_i64s(batch_rows, seed);
                let parts: Vec<&str> = vec![*part_name; batch_rows];
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int64Array::from(values)),
                        Arc::new(StringArray::from(parts)),
                    ],
                )?;
                batches.push(batch);
            }
        }

        let expected_total_rows = partition_names.len() * batches_per_partition * batch_rows;

        let table = MemTable::try_new(schema.clone(), vec![batches])?;
        ctx.session.register_table("source", Arc::new(table))?;

        let source = ctx.session.table("source").await?;
        let logical_plan = LogicalPlanBuilder::copy_to(
            source.logical_plan().clone(),
            "/table/".to_string(),
            format_as_file_type(Arc::new(factory)),
            Default::default(),
            vec!["part".to_string()],
        )?
        .build()?;

        ctx.session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        // List all written files.
        let all_files = ctx
            .store
            .list(Some(&"/table".into()))
            .try_collect::<Vec<_>>()
            .await?;

        let target_bytes = (target_mb * 1024 * 1024) as u64;

        // Verify partition directories and file counts.
        for part_name in &partition_names {
            let prefix = object_store::path::Path::from(format!("table/part={part_name}"));
            let partition_files: Vec<_> = all_files
                .iter()
                .filter(|m| m.location.prefix_matches(&prefix))
                .collect();

            assert_eq!(
                partition_files.len(),
                2,
                "Partition '{part_name}' should have 2 files, got {} (files: {:?})",
                partition_files.len(),
                partition_files
                    .iter()
                    .map(|m| format!("{}: {}B", m.location, m.size))
                    .collect::<Vec<_>>()
            );

            for meta in &partition_files {
                assert!(
                    meta.size > target_bytes / 2,
                    "File {} is {}B, expected > {}B (target/2)",
                    meta.location,
                    meta.size,
                    target_bytes / 2
                );
            }
        }

        assert_eq!(
            all_files.len(),
            8,
            "Expected 8 total files (2 per partition × 4 partitions), got {}",
            all_files.len()
        );

        // Verify total row count by reading back.
        let result = ctx
            .session
            .sql("SELECT COUNT(*) as cnt FROM '/table/'")
            .await?
            .collect()
            .await?;

        let count = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);

        assert_eq!(
            count, expected_total_rows as i64,
            "Total row count mismatch"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_write_large_batch_target_file_size_disabled() -> anyhow::Result<()> {
        use datafusion::datasource::MemTable;
        use datafusion_datasource::file_format::format_as_file_type;

        let ctx = TestSessionContext::default();

        let opts = VortexTableOptions {
            // Disable sink-side rolling/splitting.
            target_file_size_mb: 0,
            ..Default::default()
        };
        let factory = VortexFormatFactory::new().with_options(opts);

        let rows_per_partition = 300_000_usize;
        let num_partitions = 8_usize;
        let expected_total_rows = (rows_per_partition * num_partitions) as i64;

        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        let mut partitions: Vec<Vec<RecordBatch>> = Vec::new();
        for p in 0..num_partitions {
            let values = pseudo_random_i64s(rows_per_partition, (p * rows_per_partition) as i64);
            partitions.push(vec![RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(values))],
            )?]);
        }

        let table = MemTable::try_new(schema, partitions)?;
        ctx.session.register_table("source", Arc::new(table))?;

        let source = ctx.session.table("source").await?;
        let logical_plan = LogicalPlanBuilder::copy_to(
            source.logical_plan().clone(),
            "/table/".to_string(),
            format_as_file_type(Arc::new(factory)),
            Default::default(),
            vec![],
        )?
        .build()?;

        ctx.session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        let file_metas = ctx
            .store
            .list(Some(&"/table".into()))
            .try_collect::<Vec<_>>()
            .await?;

        let unique_write_ids: vortex_utils::aliases::hash_set::HashSet<_> = file_metas
            .iter()
            .filter_map(|m| {
                m.location
                    .filename()
                    .and_then(|name| name.split_once('_'))
                    .map(|(prefix, _)| prefix.to_string())
            })
            .collect();

        assert_eq!(
            unique_write_ids.len(),
            1,
            "Expected one write_id with target size disabled; got {:?} from files: {:?}",
            unique_write_ids,
            file_metas
                .iter()
                .map(|m| format!("{}: {}B", m.location, m.size))
                .collect::<Vec<_>>()
        );

        assert_eq!(
            file_metas.len(),
            1,
            "Expected exactly one output file when target size is disabled, got {}",
            file_metas.len()
        );

        let result = ctx
            .session
            .sql("SELECT COUNT(*) as cnt FROM '/table/'")
            .await?
            .collect()
            .await?;

        let count = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);

        assert_eq!(count, expected_total_rows, "Total row count mismatch");

        Ok(())
    }

    #[tokio::test]
    async fn test_target_file_size_uses_single_sink_input_partition() -> anyhow::Result<()> {
        use datafusion::datasource::MemTable;
        use datafusion_datasource::file_format::format_as_file_type;

        let ctx = TestSessionContext::default();

        let opts = VortexTableOptions {
            // Enable sink-side sizing, but make the threshold large enough
            // that all input data should fit in a single file.
            target_file_size_mb: 512,
            ..Default::default()
        };
        let factory = VortexFormatFactory::new().with_options(opts);

        let rows_per_partition = 300_000_usize;
        let num_partitions = 8_usize;
        let expected_total_rows = (rows_per_partition * num_partitions) as i64;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        // Build a MemTable with multiple physical input partitions to mimic
        // DataFusion's parallel writer inputs.
        let mut partitions: Vec<Vec<RecordBatch>> = Vec::new();
        for p in 0..num_partitions {
            let values = pseudo_random_i64s(rows_per_partition, (p * rows_per_partition) as i64);
            partitions.push(vec![RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(values))],
            )?]);
        }

        let table = MemTable::try_new(schema, partitions)?;
        ctx.session.register_table("source", Arc::new(table))?;

        let source = ctx.session.table("source").await?;
        let logical_plan = LogicalPlanBuilder::copy_to(
            source.logical_plan().clone(),
            "/table/".to_string(),
            format_as_file_type(Arc::new(factory)),
            Default::default(),
            vec![],
        )?
        .build()?;

        ctx.session
            .execute_logical_plan(logical_plan)
            .await?
            .collect()
            .await?;

        let file_metas = ctx
            .store
            .list(Some(&"/table".into()))
            .try_collect::<Vec<_>>()
            .await?;

        let unique_write_ids: vortex_utils::aliases::hash_set::HashSet<_> = file_metas
            .iter()
            .filter_map(|m| {
                m.location
                    .filename()
                    .and_then(|name| name.split_once('_'))
                    .map(|(prefix, _)| prefix.to_string())
            })
            .collect();

        assert_eq!(
            unique_write_ids.len(),
            1,
            "Expected one write_id (single sink stream), got {:?} from files: {:?}",
            unique_write_ids,
            file_metas
                .iter()
                .map(|m| format!("{}: {}B", m.location, m.size))
                .collect::<Vec<_>>()
        );

        assert!(
            file_metas.len() < num_partitions,
            "Expected fewer output files than input partitions after coalescing; got {} files for {num_partitions} input partitions",
            file_metas.len()
        );

        let result = ctx
            .session
            .sql("SELECT COUNT(*) AS cnt FROM '/table/'")
            .await?
            .collect()
            .await?;

        let count = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(count, expected_total_rows, "Total row count mismatch");

        Ok(())
    }

    #[tokio::test]
    async fn test_insert_sql_target_size_multi_partition_source_single_write_id()
    -> anyhow::Result<()> {
        use datafusion::datasource::MemTable;

        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                (a BIGINT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/' \
                OPTIONS(target_file_size_mb '64');",
            )
            .await?;

        let rows_per_partition = 300_000_usize;
        let num_partitions = 8_usize;
        let expected_total_rows = (rows_per_partition * num_partitions) as i64;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        let mut partitions: Vec<Vec<RecordBatch>> = Vec::new();
        for p in 0..num_partitions {
            let values = pseudo_random_i64s(rows_per_partition, (p * rows_per_partition) as i64);
            partitions.push(vec![RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(values))],
            )?]);
        }

        let source = MemTable::try_new(schema, partitions)?;
        ctx.session.register_table("source", Arc::new(source))?;

        ctx.session
            .sql("INSERT INTO my_tbl SELECT a FROM source")
            .await?
            .collect()
            .await?;

        let all_files = ctx.store.list(None).try_collect::<Vec<_>>().await?;

        let unique_write_ids: vortex_utils::aliases::hash_set::HashSet<_> = all_files
            .iter()
            .filter_map(|m| {
                m.location
                    .filename()
                    .and_then(|name| name.split_once('_'))
                    .map(|(prefix, _)| prefix.to_string())
            })
            .collect();

        assert_eq!(
            unique_write_ids.len(),
            1,
            "Expected one write_id, got {:?} from files: {:?}",
            unique_write_ids,
            all_files
                .iter()
                .map(|m| format!("{}: {}B", m.location, m.size))
                .collect::<Vec<_>>()
        );
        assert!(
            all_files.len() < num_partitions,
            "Expected fewer files than input partitions; got {} files for {num_partitions} input partitions",
            all_files.len()
        );

        let result = ctx
            .session
            .sql("SELECT COUNT(*) AS cnt FROM my_tbl")
            .await?
            .collect()
            .await?;
        let count = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(count, expected_total_rows, "Total row count mismatch");

        Ok(())
    }

    #[tokio::test]
    async fn test_insert_sql_partitioned_target_size_multi_partition_source_single_write_id()
    -> anyhow::Result<()> {
        use datafusion::arrow::array::StringArray;
        use datafusion::datasource::MemTable;

        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                (part VARCHAR NOT NULL, val BIGINT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/' \
                PARTITIONED BY (part) \
                OPTIONS(target_file_size_mb '64');",
            )
            .await?;

        let rows_per_partition = 150_000_usize;
        let num_partitions = 8_usize;
        let expected_total_rows = (rows_per_partition * num_partitions) as i64;

        let schema = Arc::new(Schema::new(vec![
            Field::new("part", DataType::Utf8, false),
            Field::new("val", DataType::Int64, false),
        ]));

        let mut partitions: Vec<Vec<RecordBatch>> = Vec::new();
        for p in 0..num_partitions {
            let values = pseudo_random_i64s(rows_per_partition, (p * rows_per_partition) as i64);
            let part_name = if p % 2 == 0 { "alpha" } else { "beta" };
            let part_values: Vec<&str> = vec![part_name; rows_per_partition];

            partitions.push(vec![RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(part_values)),
                    Arc::new(Int64Array::from(values)),
                ],
            )?]);
        }

        let source = MemTable::try_new(schema, partitions)?;
        ctx.session.register_table("source", Arc::new(source))?;

        ctx.session
            .sql("INSERT INTO my_tbl (part, val) SELECT part, val FROM source")
            .await?
            .collect()
            .await?;

        let all_files = ctx.store.list(None).try_collect::<Vec<_>>().await?;

        let unique_write_ids: vortex_utils::aliases::hash_set::HashSet<_> = all_files
            .iter()
            .filter_map(|m| {
                m.location
                    .filename()
                    .and_then(|name| name.split_once('_'))
                    .map(|(prefix, _)| prefix.to_string())
            })
            .collect();

        assert_eq!(
            unique_write_ids.len(),
            1,
            "Expected one write_id for partitioned INSERT, got {:?} from files: {:?}",
            unique_write_ids,
            all_files
                .iter()
                .map(|m| format!("{}: {}B", m.location, m.size))
                .collect::<Vec<_>>()
        );

        assert!(
            all_files
                .iter()
                .any(|m| m.location.to_string().contains("part=alpha")),
            "Expected partition directory part=alpha"
        );
        assert!(
            all_files
                .iter()
                .any(|m| m.location.to_string().contains("part=beta")),
            "Expected partition directory part=beta"
        );

        let result = ctx
            .session
            .sql("SELECT COUNT(*) AS cnt FROM my_tbl")
            .await?
            .collect()
            .await?;
        let count = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(count, expected_total_rows, "Total row count mismatch");

        Ok(())
    }

    #[tokio::test]
    async fn test_insert_sql_streaming_source_single_write_id() -> anyhow::Result<()> {
        use arrow_schema::SchemaRef;
        use datafusion::catalog::streaming::StreamingTable;
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
        use datafusion::physical_plan::streaming::PartitionStream;
        use futures::stream;

        #[derive(Debug)]
        struct StaticPartitionStream {
            schema: SchemaRef,
            batch: RecordBatch,
        }

        impl PartitionStream for StaticPartitionStream {
            fn schema(&self) -> &SchemaRef {
                &self.schema
            }

            fn execute(
                &self,
                _ctx: Arc<datafusion::execution::TaskContext>,
            ) -> datafusion::physical_plan::SendableRecordBatchStream {
                let schema = Arc::clone(&self.schema);
                let batch = self.batch.clone();
                Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    stream::iter(vec![Ok(batch)]),
                ))
            }
        }

        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                (a BIGINT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/' \
                OPTIONS(target_file_size_mb '64');",
            )
            .await?;

        let rows_per_partition = 300_000_usize;
        let num_partitions = 8_usize;
        let expected_total_rows = (rows_per_partition * num_partitions) as i64;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        let mut partitions: Vec<Arc<dyn PartitionStream>> = Vec::new();
        for p in 0..num_partitions {
            let values = pseudo_random_i64s(rows_per_partition, (p * rows_per_partition) as i64);
            let batch =
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))])?;

            partitions.push(Arc::new(StaticPartitionStream {
                schema: schema.clone(),
                batch,
            }));
        }

        let source = StreamingTable::try_new(schema, partitions)?;
        ctx.session
            .register_table("source_stream", Arc::new(source))?;

        ctx.session
            .sql("INSERT INTO my_tbl SELECT a FROM source_stream")
            .await?
            .collect()
            .await?;

        let all_files = ctx.store.list(None).try_collect::<Vec<_>>().await?;

        let unique_write_ids: vortex_utils::aliases::hash_set::HashSet<_> = all_files
            .iter()
            .filter_map(|m| {
                m.location
                    .filename()
                    .and_then(|name| name.split_once('_'))
                    .map(|(prefix, _)| prefix.to_string())
            })
            .collect();

        assert_eq!(
            unique_write_ids.len(),
            1,
            "Expected one write_id for streaming source insert, got {:?} from files: {:?}",
            unique_write_ids,
            all_files
                .iter()
                .map(|m| format!("{}: {}B", m.location, m.size))
                .collect::<Vec<_>>()
        );

        let result = ctx
            .session
            .sql("SELECT COUNT(*) AS cnt FROM my_tbl")
            .await?
            .collect()
            .await?;
        let count = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(count, expected_total_rows, "Total row count mismatch");

        Ok(())
    }

    #[tokio::test]
    async fn test_listing_table_direct_insert_into_streaming_exec_single_write_id()
    -> anyhow::Result<()> {
        use arrow_schema::SchemaRef;
        use datafusion::physical_plan::ExecutionPlan;
        use datafusion::physical_plan::collect;
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
        use datafusion::physical_plan::streaming::PartitionStream;
        use datafusion::physical_plan::streaming::StreamingTableExec;
        use datafusion_expr::dml::InsertOp;
        use futures::stream;

        #[derive(Debug)]
        struct StaticPartitionStream {
            schema: SchemaRef,
            batch: RecordBatch,
        }

        impl PartitionStream for StaticPartitionStream {
            fn schema(&self) -> &SchemaRef {
                &self.schema
            }

            fn execute(
                &self,
                _ctx: Arc<datafusion::execution::TaskContext>,
            ) -> datafusion::physical_plan::SendableRecordBatchStream {
                let schema = Arc::clone(&self.schema);
                let batch = self.batch.clone();
                Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    stream::iter(vec![Ok(batch)]),
                ))
            }
        }

        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                (a BIGINT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/' \
                OPTIONS(target_file_size_mb '64');",
            )
            .await?;

        let table_provider = ctx.session.table_provider("my_tbl").await?;

        let rows_per_partition = 300_000_usize;
        let num_partitions = 8_usize;
        let expected_total_rows = (rows_per_partition * num_partitions) as i64;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        let mut partitions: Vec<Arc<dyn PartitionStream>> = Vec::new();
        for p in 0..num_partitions {
            let values = pseudo_random_i64s(rows_per_partition, (p * rows_per_partition) as i64);
            let batch =
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))])?;

            partitions.push(Arc::new(StaticPartitionStream {
                schema: schema.clone(),
                batch,
            }));
        }

        let input = Arc::new(StreamingTableExec::try_new(
            schema,
            partitions,
            None,
            Vec::new(),
            false,
            None,
        )?) as Arc<dyn ExecutionPlan>;

        let plan = table_provider
            .insert_into(&ctx.session.state(), input, InsertOp::Append)
            .await?;
        let _count_batches = collect(plan, ctx.session.task_ctx()).await?;

        let all_files = ctx.store.list(None).try_collect::<Vec<_>>().await?;

        let unique_write_ids: vortex_utils::aliases::hash_set::HashSet<_> = all_files
            .iter()
            .filter_map(|m| {
                m.location
                    .filename()
                    .and_then(|name| name.split_once('_'))
                    .map(|(prefix, _)| prefix.to_string())
            })
            .collect();

        assert_eq!(
            unique_write_ids.len(),
            1,
            "Expected one write_id for direct insert_into streaming exec, got {:?} from files: {:?}",
            unique_write_ids,
            all_files
                .iter()
                .map(|m| format!("{}: {}B", m.location, m.size))
                .collect::<Vec<_>>()
        );

        let result = ctx
            .session
            .sql("SELECT COUNT(*) AS cnt FROM my_tbl")
            .await?
            .collect()
            .await?;
        let count = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(count, expected_total_rows, "Total row count mismatch");

        Ok(())
    }

    #[tokio::test]
    async fn test_listing_table_direct_insert_into_unbounded_streaming_exec_single_write_id()
    -> anyhow::Result<()> {
        use arrow_schema::SchemaRef;
        use datafusion::physical_plan::ExecutionPlan;
        use datafusion::physical_plan::collect;
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
        use datafusion::physical_plan::streaming::PartitionStream;
        use datafusion::physical_plan::streaming::StreamingTableExec;
        use datafusion_expr::dml::InsertOp;
        use futures::stream;

        #[derive(Debug)]
        struct StaticPartitionStream {
            schema: SchemaRef,
            batch: RecordBatch,
        }

        impl PartitionStream for StaticPartitionStream {
            fn schema(&self) -> &SchemaRef {
                &self.schema
            }

            fn execute(
                &self,
                _ctx: Arc<datafusion::execution::TaskContext>,
            ) -> datafusion::physical_plan::SendableRecordBatchStream {
                let schema = Arc::clone(&self.schema);
                let batch = self.batch.clone();
                Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    stream::iter(vec![Ok(batch)]),
                ))
            }
        }

        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                (a BIGINT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/' \
                OPTIONS(target_file_size_mb '64');",
            )
            .await?;

        let table_provider = ctx.session.table_provider("my_tbl").await?;

        let rows_per_partition = 100_000_usize;
        let num_partitions = 8_usize;
        let expected_total_rows = (rows_per_partition * num_partitions) as i64;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));

        let mut partitions: Vec<Arc<dyn PartitionStream>> = Vec::new();
        for p in 0..num_partitions {
            let values = pseudo_random_i64s(rows_per_partition, (p * rows_per_partition) as i64);
            let batch =
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))])?;

            partitions.push(Arc::new(StaticPartitionStream {
                schema: schema.clone(),
                batch,
            }));
        }

        let input = Arc::new(StreamingTableExec::try_new(
            schema,
            partitions,
            None,
            Vec::new(),
            true,
            None,
        )?) as Arc<dyn ExecutionPlan>;

        let plan = table_provider
            .insert_into(&ctx.session.state(), input, InsertOp::Append)
            .await?;
        let _count_batches = collect(plan, ctx.session.task_ctx()).await?;

        let all_files = ctx.store.list(None).try_collect::<Vec<_>>().await?;

        let unique_write_ids: vortex_utils::aliases::hash_set::HashSet<_> = all_files
            .iter()
            .filter_map(|m| {
                m.location
                    .filename()
                    .and_then(|name| name.split_once('_'))
                    .map(|(prefix, _)| prefix.to_string())
            })
            .collect();

        assert_eq!(
            unique_write_ids.len(),
            1,
            "Expected one write_id for unbounded streaming insert, got {:?} from files: {:?}",
            unique_write_ids,
            all_files
                .iter()
                .map(|m| format!("{}: {}B", m.location, m.size))
                .collect::<Vec<_>>()
        );

        let result = ctx
            .session
            .sql("SELECT COUNT(*) AS cnt FROM my_tbl")
            .await?
            .collect()
            .await?;
        let count = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(count, expected_total_rows, "Total row count mismatch");

        Ok(())
    }

    #[tokio::test]
    async fn test_write_partitioned_with_null_partition_values_errors() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                (part VARCHAR, val INT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/' \
                PARTITIONED BY (part);",
            )
            .await?;

        let err = ctx
            .session
            .sql("INSERT INTO my_tbl (part, val) VALUES ('alpha', 1), (NULL, 2), ('alpha', 3), (NULL, 4);")
            .await?
            .collect()
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("contains null values"),
            "Expected NULL partition insert to fail loudly; got: {err}"
        );

        Ok(())
    }
}
