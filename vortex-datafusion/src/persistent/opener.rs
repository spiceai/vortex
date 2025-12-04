// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Range;
use std::pin::Pin;
use std::sync::{Arc, Weak};
use std::task::{Context, Poll};

use arrow_schema::{ArrowError, DataType, Field, SchemaRef};
use datafusion_common::arrow::array::{
    Array, Int32Array, Int64Array, RecordBatch, UInt32Array, UInt64Array,
};
use datafusion_common::pruning::PrunableStatistics;
use datafusion_common::{DataFusionError, Result as DFResult, ScalarValue, Statistics};
use datafusion_datasource::file_meta::FileMeta;
use datafusion_datasource::file_stream::{FileOpenFuture, FileOpener};
use datafusion_datasource::schema_adapter::SchemaAdapterFactory;
use datafusion_datasource::{FileRange, PartitionedFile};
use datafusion_datasource_parquet::EarlyStoppingStream;
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{BinaryExpr, DynamicFilterPhysicalExpr, InListExpr};
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::{PhysicalExpr, PhysicalExprRef, split_conjunction};
use datafusion_physical_expr_adapter::PhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::{
    is_dynamic_physical_expr, snapshot_generation,
};
use datafusion_physical_plan::metrics::Count;
use datafusion_pruning::{
    BoolVecBuilder, FilePruner, PruningStatistics, RequiredColumns, build_statistics_record_batch,
};
use futures::{FutureExt, Stream, StreamExt, TryStreamExt, ready, stream};
use object_store::ObjectStore;
use object_store::path::Path;
use tracing::Instrument;
use vortex::ArrayRef;
use vortex::dtype::FieldName;
use vortex::error::VortexError;
use vortex::expr::{root, select};
use vortex::layout::LayoutReader;
use vortex::metrics::VortexMetrics;
use vortex::scan::ScanBuilder;
use vortex::session::VortexSession;
use vortex_utils::aliases::dash_map::{DashMap, Entry};

use super::cache::VortexFileCache;
use crate::convert::exprs::{can_be_pushed_down, make_vortex_predicate};

#[derive(Clone)]
pub(crate) struct VortexOpener {
    pub session: VortexSession,
    pub object_store: Arc<dyn ObjectStore>,
    /// Projection by index of the file's columns
    pub projection: Option<Arc<[usize]>>,
    /// Filter expression optimized for pushdown into Vortex scan operations.
    /// This may be a subset of file_pruning_predicate containing only expressions
    /// that Vortex can efficiently evaluate.
    pub filter: Option<PhysicalExprRef>,
    /// Filter expression used by DataFusion's FilePruner to eliminate files based on
    /// statistics and partition values without opening them.
    pub file_pruning_predicate: Option<PhysicalExprRef>,
    pub expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    pub schema_adapter_factory: Arc<dyn SchemaAdapterFactory>,
    /// Hive-style partitioning columns
    pub partition_fields: Vec<Arc<Field>>,
    pub file_cache: VortexFileCache,
    /// This is the table's schema without partition columns. It might be different than
    /// the physical schema, and the stream's type will be a projection of it.
    pub logical_schema: SchemaRef,
    pub batch_size: usize,
    pub limit: Option<usize>,
    pub metrics: VortexMetrics,
    pub layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
    /// Whether the query has output ordering specified
    pub has_output_ordering: bool,
}

/// Merges the data types of two fields, preferring the logical type from the
/// table field.
fn merge_field_types(physical_field: &Field, table_field: &Field) -> DataType {
    match (physical_field.data_type(), table_field.data_type()) {
        (DataType::Struct(phys_fields), DataType::Struct(table_fields)) => {
            let merged_fields = merge_fields(phys_fields, table_fields);
            DataType::Struct(merged_fields.into())
        }
        (DataType::List(phys_field), DataType::List(table_field)) => {
            DataType::List(Arc::new(Field::new(
                phys_field.name(),
                merge_field_types(phys_field, table_field),
                phys_field.is_nullable(),
            )))
        }
        (DataType::LargeList(phys_field), DataType::LargeList(table_field)) => {
            DataType::LargeList(Arc::new(Field::new(
                phys_field.name(),
                merge_field_types(phys_field, table_field),
                phys_field.is_nullable(),
            )))
        }
        _ => table_field.data_type().clone(),
    }
}

/// Merges two field collections, using logical types from table_fields where available.
/// Falls back to physical field types when no matching table field is found.
fn merge_fields(
    physical_fields: &arrow_schema::Fields,
    table_fields: &arrow_schema::Fields,
) -> Vec<Field> {
    physical_fields
        .iter()
        .map(|phys_field| {
            table_fields
                .iter()
                .find(|f| f.name() == phys_field.name())
                .map(|table_field| {
                    Field::new(
                        phys_field.name(),
                        merge_field_types(phys_field, table_field),
                        phys_field.is_nullable(),
                    )
                })
                .unwrap_or_else(|| (**phys_field).clone())
        })
        .collect()
}

/// Computes a logical file schema from the physical file schema and the table
/// schema.
///
/// For each field in the physical file schema, looks up the corresponding field
/// in the table schema and uses its logical type.
fn compute_logical_file_schema(
    physical_file_schema: &SchemaRef,
    table_schema: &SchemaRef,
) -> SchemaRef {
    let logical_fields: Vec<Field> = physical_file_schema
        .fields()
        .iter()
        .map(|physical_field| {
            table_schema
                .fields()
                .find(physical_field.name())
                .map(|(_, table_field)| {
                    Field::new(
                        physical_field.name(),
                        merge_field_types(physical_field, table_field),
                        physical_field.is_nullable(),
                    )
                    .with_metadata(physical_field.metadata().clone())
                })
                .unwrap_or_else(|| (**physical_field).clone())
        })
        .collect();

    Arc::new(arrow_schema::Schema::new(logical_fields))
}

impl FileOpener for VortexOpener {
    fn open(&self, file_meta: FileMeta, file: PartitionedFile) -> DFResult<FileOpenFuture> {
        let session = self.session.clone();
        let object_store = self.object_store.clone();
        let projection = self.projection.clone();
        let mut filter = self.filter.clone();
        let file_pruning_predicate = self.file_pruning_predicate.clone();
        let expr_adapter_factory = self.expr_adapter_factory.clone();
        let partition_fields = self.partition_fields.clone();
        let file_cache = self.file_cache.clone();
        let logical_schema = self.logical_schema.clone();
        let batch_size = self.batch_size;
        let limit = self.limit;
        let metrics = self.metrics.clone();
        let layout_reader = self.layout_readers.clone();
        let has_output_ordering = self.has_output_ordering;

        let statistics = file.statistics.clone();

        let projected_schema = match projection.as_ref() {
            None => logical_schema.clone(),
            Some(indices) => Arc::new(logical_schema.project(indices)?),
        };

        let mut predicate_file_schema = logical_schema.clone();

        let schema_adapter = self
            .schema_adapter_factory
            .create(projected_schema, logical_schema.clone());

        Ok(async move {
            // Create FilePruner when we have a predicate and either dynamic expressions
            // or file statistics available. The pruner can eliminate files without
            // opening them based on:
            // - Partition column values (e.g., date=2024-01-01)
            // - File-level statistics (min/max values per column)
            let mut file_pruner = file_pruning_predicate
                .clone()
                .map(|predicate| {
                    // Only create pruner if we have dynamic expressions or file statistics
                    // to work with. Static predicates without stats won't benefit from pruning.
                    Ok::<_, DataFusionError>(
                        (is_dynamic_physical_expr(&predicate) | file.has_statistics()).then_some(
                            FilePruner::new(
                                predicate.clone(),
                                &logical_schema,
                                partition_fields.clone(),
                                file.clone(),
                                Count::default(),
                            )?,
                        ),
                    )
                })
                .transpose()?
                .flatten();

            let cloned_file = file.clone();
            let dynamic_filter_expr =
                file_pruning_predicate.filter(|expr| is_dynamic_physical_expr(expr));
            let statistics = statistics; // re-scope for move

            // Check if this file should be pruned based on statistics/partition values.
            // Returns empty stream if file can be skipped entirely.
            if let Some(file_pruner) = &mut file_pruner
                && file_pruner.should_prune()?
            {
                return Ok(stream::empty().boxed());
            }

            let vxf = file_cache
                .try_get(&file_meta.object_meta, object_store)
                .await
                .map_err(|e| {
                    DataFusionError::Execution(format!("Failed to open Vortex file {e}"))
                })?;

            let physical_file_schema = Arc::new(vxf.dtype().to_arrow_schema().map_err(|e| {
                DataFusionError::Execution(format!("Failed to convert file schema to arrow: {e}"))
            })?);

            if let Some(expr_adapter_factory) = expr_adapter_factory {
                let partition_values = partition_fields
                    .iter()
                    .cloned()
                    .zip(file.partition_values)
                    .collect::<Vec<_>>();

                // The adapter rewrites the expression to the local file schema, allowing
                // for schema evolution and divergence between the table's schema and individual files.
                filter = filter
                    .map(|filter| {
                        let logical_file_schema =
                            compute_logical_file_schema(&physical_file_schema, &logical_schema);

                        let expr = expr_adapter_factory
                            .create(logical_file_schema, physical_file_schema.clone())
                            .with_partition_values(partition_values)
                            .rewrite(filter)?;

                        // Expression might now reference columns that don't exist in the file, so we can give it
                        // another simplification pass.
                        PhysicalExprSimplifier::new(&physical_file_schema).simplify(expr)
                    })
                    .transpose()?;

                predicate_file_schema = physical_file_schema.clone();
            }

            let (schema_mapping, adapted_projections) =
                schema_adapter.map_schema(&physical_file_schema)?;

            let fields = adapted_projections
                .iter()
                .map(|idx| {
                    let field = logical_schema.field(*idx);
                    FieldName::from(field.name().as_str())
                })
                .collect::<Vec<_>>();
            let projection_expr = select(fields, root());

            // We share our layout readers with others partitions in the scan, so we can only need to read each layout in each file once.
            let layout_reader = match layout_reader.entry(file_meta.object_meta.location.clone()) {
                Entry::Occupied(mut occupied_entry) => {
                    if let Some(reader) = occupied_entry.get().upgrade() {
                        log::trace!("reusing layout reader for {}", occupied_entry.key());
                        reader
                    } else {
                        log::trace!("creating layout reader for {}", occupied_entry.key());
                        let reader = vxf.layout_reader().map_err(|e| {
                            DataFusionError::Execution(format!(
                                "Failed to create layout reader: {e}"
                            ))
                        })?;
                        occupied_entry.insert(Arc::downgrade(&reader));
                        reader
                    }
                }
                Entry::Vacant(vacant_entry) => {
                    log::trace!("creating layout reader for {}", vacant_entry.key());
                    let reader = vxf.layout_reader().map_err(|e| {
                        DataFusionError::Execution(format!("Failed to create layout reader: {e}"))
                    })?;
                    vacant_entry.insert(Arc::downgrade(&reader));

                    reader
                }
            };

            let mut scan_builder = ScanBuilder::new(session, layout_reader);
            if let Some(file_range) = file_meta.range {
                scan_builder = apply_byte_range(
                    file_range,
                    file_meta.object_meta.size,
                    vxf.row_count(),
                    scan_builder,
                );
            }

            let filter = filter
                .and_then(|f| {
                    let exprs = split_conjunction(&f)
                        .into_iter()
                        .filter(|expr| can_be_pushed_down(expr, &predicate_file_schema))
                        .collect::<Vec<_>>();

                    make_vortex_predicate(&exprs).transpose()
                })
                .transpose()
                .map_err(|e| DataFusionError::External(e.into()))?;

            if let Some(limit) = limit
                && filter.is_none()
            {
                scan_builder = scan_builder.with_limit(limit);
            }

            let stream = scan_builder
                .with_metrics(metrics)
                .with_projection(projection_expr)
                .with_some_filter(filter)
                .with_ordered(has_output_ordering)
                .map(|chunk| RecordBatch::try_from(chunk.as_ref()))
                .into_stream()
                .map_err(|e| {
                    DataFusionError::Execution(format!("Failed to create Vortex stream: {e}"))
                })?
                .map_ok(move |rb| {
                    // We try and slice the stream into respecting datafusion's configured batch size.
                    stream::iter(
                        (0..rb.num_rows().div_ceil(batch_size * 2))
                            .flat_map(move |block_idx| {
                                let offset = block_idx * batch_size * 2;

                                // If we have less than two batches worth of rows left, we keep them together as a single batch.
                                if rb.num_rows() - offset < 2 * batch_size {
                                    let length = rb.num_rows() - offset;
                                    [Some(rb.slice(offset, length)), None].into_iter()
                                } else {
                                    let first = rb.slice(offset, batch_size);
                                    let second = rb.slice(offset + batch_size, batch_size);
                                    [Some(first), Some(second)].into_iter()
                                }
                            })
                            .flatten()
                            .map(Ok),
                    )
                })
                .map_err(move |e: VortexError| {
                    ArrowError::ExternalError(Box::new(e.with_context(format!(
                        "Failed to read Vortex file: {}",
                        file_meta.object_meta.location
                    ))))
                })
                .try_flatten()
                .map(move |batch| batch.and_then(|b| schema_mapping.map_batch(b)))
                .boxed();

            if let Some(dynamic_filter_expr) = dynamic_filter_expr
                && let Some(statistics) = statistics
            {
                Ok(Box::pin(VortexStoppingStream::new(
                    stream,
                    dynamic_filter_expr,
                    statistics,
                )))
            } else {
                Ok(Box::pin(stream))
            }
        }
        .in_current_span()
        .boxed())
    }
}

// For a given `InListExpr`, calculate the contiguous ranges in the list and return them as `BinaryExpr`'s
// For example, the list `a IN [1,2,3,5,6,8]` would return the expressions:
// - `a >= 1 AND a <= 3`
// - `a >= 5 AND a <= 6`
// - `a >= 8 AND a <= 8`
//
// The `overlap` argument defines allowable overlap between ranges to be considered contiguous.
// For example, with an overlap of 1, the list `a IN [1,2,4,5,7]` would return:
// - `a >= 1 AND a <= 7`
fn contiguous_in_list_ranges(in_list_expr: &InListExpr, overlap: usize) -> Vec<BinaryExpr> {
    let input_expr = in_list_expr.expr();
    let list = in_list_expr.list();

    // only literals are supported
    let mut literals = vec![];
    for value in list.iter() {
        if let Some(literal) = value
            .as_any()
            .downcast_ref::<datafusion_physical_expr::expressions::Literal>()
        {
            match literal.value() {
                ScalarValue::List(list) => {
                    // get the individual values out of the list
                    let inner = list.value(0);
                    match inner {
                        v if v.as_any().downcast_ref::<Int32Array>().is_some() => {
                            let inner_i32 = v.as_any().downcast_ref::<Int32Array>().unwrap();
                            for i in 0..inner_i32.len() {
                                literals.push(ScalarValue::Int32(Some(inner_i32.value(i))));
                            }
                        }
                        v if v.as_any().downcast_ref::<Int64Array>().is_some() => {
                            let inner_i64 = v.as_any().downcast_ref::<Int64Array>().unwrap();
                            for i in 0..inner_i64.len() {
                                literals.push(ScalarValue::Int64(Some(inner_i64.value(i))));
                            }
                        }
                        v if v.as_any().downcast_ref::<UInt32Array>().is_some() => {
                            let inner_u32 = v.as_any().downcast_ref::<UInt32Array>().unwrap();
                            for i in 0..inner_u32.len() {
                                literals.push(ScalarValue::UInt32(Some(inner_u32.value(i))));
                            }
                        }
                        v if v.as_any().downcast_ref::<UInt64Array>().is_some() => {
                            let inner_u64 = v.as_any().downcast_ref::<UInt64Array>().unwrap();
                            for i in 0..inner_u64.len() {
                                literals.push(ScalarValue::UInt64(Some(inner_u64.value(i))));
                            }
                        }
                        _ => {
                            println!(
                                "Data Type not supported for contiguous range calculation: {}",
                                inner.data_type()
                            );
                            return vec![];
                        }
                    };
                }
                ScalarValue::Int32(Some(v)) => {
                    literals.push(ScalarValue::Int32(Some(*v)));
                }
                ScalarValue::Int64(Some(v)) => {
                    literals.push(ScalarValue::Int64(Some(*v)));
                }
                ScalarValue::UInt32(Some(v)) => {
                    literals.push(ScalarValue::UInt32(Some(*v)));
                }
                ScalarValue::UInt64(Some(v)) => {
                    literals.push(ScalarValue::UInt64(Some(*v)));
                }
                _ => {
                    // non-literal found, cannot process
                    println!(
                        "cannot compute contiguous ranges from scalar value: {:?}",
                        literal.value()
                    );
                    return vec![];
                }
            }
        } else {
            // non-literal found, cannot process
            println!(
                "Found non-literal in InListExpr, cannot compute contiguous ranges: {:?}",
                value
            );
            return vec![];
        }
    }

    // sort the scalars
    literals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("Sorted literals: {:?}", literals);

    // group into contiguous ranges
    let mut ranges = vec![];
    let mut start = None;
    let mut end = None;
    for literal in literals {
        if start.is_none() {
            start = Some(literal.clone());
            end = Some(literal.clone());
            continue;
        }

        let end_value = end.as_ref().unwrap();
        let next_value = &literal;

        // check if next_value is contiguous with end_value
        let is_contiguous = match (end_value, next_value) {
            (ScalarValue::Int32(Some(e)), ScalarValue::Int32(Some(n))) => {
                (*n as isize - *e as isize) <= (overlap as isize + 1)
            }
            (ScalarValue::Int64(Some(e)), ScalarValue::Int64(Some(n))) => {
                (*n as isize - *e as isize) <= (overlap as isize + 1)
            }
            (ScalarValue::UInt32(Some(e)), ScalarValue::UInt32(Some(n))) => {
                (*n as isize - *e as isize) <= (overlap as isize + 1)
            }
            (ScalarValue::UInt64(Some(e)), ScalarValue::UInt64(Some(n))) => {
                (*n as isize - *e as isize) <= (overlap as isize + 1)
            }
            _ => false, // unsupported type for this example
        };

        if is_contiguous {
            end = Some(literal.clone());
        } else {
            // finalize current range
            let start_expr = datafusion_physical_expr::expressions::Literal::new(start.unwrap());
            let end_expr = datafusion_physical_expr::expressions::Literal::new(end.unwrap());
            let ge_expr = BinaryExpr::new(input_expr.clone(), Operator::GtEq, Arc::new(start_expr));
            let le_expr = BinaryExpr::new(input_expr.clone(), Operator::LtEq, Arc::new(end_expr));
            let range_expr = BinaryExpr::new(Arc::new(ge_expr), Operator::And, Arc::new(le_expr));
            ranges.push(range_expr);

            // start new range
            start = Some(literal.clone());
            end = Some(literal.clone());
        }
    }

    ranges.push({
        let start_expr = datafusion_physical_expr::expressions::Literal::new(start.unwrap());
        let end_expr = datafusion_physical_expr::expressions::Literal::new(end.unwrap());
        let ge_expr = BinaryExpr::new(input_expr.clone(), Operator::GtEq, Arc::new(start_expr));
        let le_expr = BinaryExpr::new(input_expr.clone(), Operator::LtEq, Arc::new(end_expr));
        BinaryExpr::new(Arc::new(ge_expr), Operator::And, Arc::new(le_expr))
    });

    println!("Contiguous ranges: {:?}", ranges);

    ranges
}

fn min_max_within_in_list(in_list_expr: &InListExpr, min_max: (ScalarValue, ScalarValue)) -> bool {
    let list = in_list_expr.list();

    // only literals are supported
    let mut literals = vec![];
    for value in list.iter() {
        if let Some(literal) = value
            .as_any()
            .downcast_ref::<datafusion_physical_expr::expressions::Literal>()
        {
            match literal.value() {
                ScalarValue::List(list) => {
                    // get the individual values out of the list
                    let inner = list.value(0);
                    match inner {
                        v if v.as_any().downcast_ref::<Int32Array>().is_some() => {
                            let inner_i32 = v.as_any().downcast_ref::<Int32Array>().unwrap();
                            for i in 0..inner_i32.len() {
                                literals.push(ScalarValue::Int32(Some(inner_i32.value(i))));
                            }
                        }
                        v if v.as_any().downcast_ref::<Int64Array>().is_some() => {
                            let inner_i64 = v.as_any().downcast_ref::<Int64Array>().unwrap();
                            for i in 0..inner_i64.len() {
                                literals.push(ScalarValue::Int64(Some(inner_i64.value(i))));
                            }
                        }
                        v if v.as_any().downcast_ref::<UInt32Array>().is_some() => {
                            let inner_u32 = v.as_any().downcast_ref::<UInt32Array>().unwrap();
                            for i in 0..inner_u32.len() {
                                literals.push(ScalarValue::UInt32(Some(inner_u32.value(i))));
                            }
                        }
                        v if v.as_any().downcast_ref::<UInt64Array>().is_some() => {
                            let inner_u64 = v.as_any().downcast_ref::<UInt64Array>().unwrap();
                            for i in 0..inner_u64.len() {
                                literals.push(ScalarValue::UInt64(Some(inner_u64.value(i))));
                            }
                        }
                        _ => {
                            println!(
                                "Data Type not supported for contiguous range calculation: {}",
                                inner.data_type()
                            );
                            return true;
                        }
                    };
                }
                ScalarValue::Int32(Some(v)) => {
                    literals.push(ScalarValue::Int32(Some(*v)));
                }
                ScalarValue::Int64(Some(v)) => {
                    literals.push(ScalarValue::Int64(Some(*v)));
                }
                ScalarValue::UInt32(Some(v)) => {
                    literals.push(ScalarValue::UInt32(Some(*v)));
                }
                ScalarValue::UInt64(Some(v)) => {
                    literals.push(ScalarValue::UInt64(Some(*v)));
                }
                _ => {
                    // non-literal found, cannot process
                    println!(
                        "cannot compute contiguous ranges from scalar value: {:?}",
                        literal.value()
                    );
                    return true;
                }
            }
        } else {
            // non-literal found, cannot process
            println!(
                "Found non-literal in InListExpr, cannot compute contiguous ranges: {:?}",
                value
            );
            return true;
        }
    }

    let Some(first) = literals.first() else {
        return true;
    };

    if min_max.0.data_type() != first.data_type() {
        return true; // cannot compare different data types
    }

    match min_max {
        (ScalarValue::Int32(Some(min)), ScalarValue::Int32(Some(max))) => {
            let search_in_range = min..=max;
            return literals.iter().any(|literal| {
                if let ScalarValue::Int32(Some(v)) = literal {
                    search_in_range.contains(v)
                } else {
                    false
                }
            });
        }
        (ScalarValue::Int64(Some(min)), ScalarValue::Int64(Some(max))) => {
            let search_in_range = min..=max;
            return literals.iter().any(|literal| {
                if let ScalarValue::Int64(Some(v)) = literal {
                    search_in_range.contains(v)
                } else {
                    false
                }
            });
        }
        (ScalarValue::UInt32(Some(min)), ScalarValue::UInt32(Some(max))) => {
            let search_in_range = min..=max;
            return literals.iter().any(|literal| {
                if let ScalarValue::UInt32(Some(v)) = literal {
                    search_in_range.contains(v)
                } else {
                    false
                }
            });
        }
        (ScalarValue::UInt64(Some(min)), ScalarValue::UInt64(Some(max))) => {
            let search_in_range = min..=max;
            return literals.iter().any(|literal| {
                if let ScalarValue::UInt64(Some(v)) = literal {
                    search_in_range.contains(v)
                } else {
                    false
                }
            });
        }
        _ => {
            return true;
        }
    }
}

// compacts a BinaryExpr which contains several `OR`ed `InListExpr`s into a single `InListExpr`
fn in_list_expr_compactor(binary_expr: BinaryExpr) -> Option<InListExpr> {
    if binary_expr.op() != &Operator::Or {
        return None;
    }

    let mut in_list_values = vec![];
    let mut target_left_expr = None;
    if let Some(left_in_list) = binary_expr.left().as_any().downcast_ref::<InListExpr>() {
        in_list_values.extend_from_slice(left_in_list.list());
        target_left_expr = Some(left_in_list.expr().clone());
    } else if let Some(left_binary) = binary_expr.left().as_any().downcast_ref::<BinaryExpr>() {
        if let Some(compacted_left) = in_list_expr_compactor(left_binary.clone()) {
            in_list_values.extend_from_slice(compacted_left.list());
        }
    } else {
        return None;
    }

    if let Some(right_in_list) = binary_expr.right().as_any().downcast_ref::<InListExpr>() {
        in_list_values.extend_from_slice(right_in_list.list());
    } else if let Some(right_binary) = binary_expr.right().as_any().downcast_ref::<BinaryExpr>() {
        if let Some(compacted_right) = in_list_expr_compactor(right_binary.clone()) {
            in_list_values.extend_from_slice(compacted_right.list());
        }
    } else {
        return None;
    }

    if !in_list_values.is_empty()
        && let Some(left_expr) = target_left_expr
    {
        Some(InListExpr::new(left_expr, in_list_values, false, None))
    } else {
        None
    }
}

struct VortexStoppingStream<S> {
    inner: S,
    dynamic_filter_expr: Arc<dyn PhysicalExpr>,
    statistics: Arc<Statistics>,
    done: bool,
    dynamic_filter_generation: Option<u64>,
}

impl<S> VortexStoppingStream<S>
where
    S: Stream<Item = DFResult<RecordBatch>> + Unpin,
{
    pub fn new(
        inner: S,
        dynamic_filter_expr: Arc<dyn PhysicalExpr>,
        statistics: Arc<Statistics>,
    ) -> Self {
        Self {
            inner,
            dynamic_filter_expr,
            statistics,
            done: false,
            dynamic_filter_generation: None,
        }
    }

    fn should_prune(&mut self, batch: &RecordBatch) -> bool {
        let mut hasher = DefaultHasher::default();
        self.dynamic_filter_expr.hash(&mut hasher);
        let new_generation = hasher.finish();

        // let new_generation = snapshot_generation(&self.dynamic_filter_expr);
        // limit dynamic filter expr debug output to 200 characters
        println!(
            "Dynamic filter expr: {}",
            format!("{:?}", self.dynamic_filter_expr)
                .chars()
                .take(200)
                .collect::<String>()
        );
        println!("========== NEW GENERATION: {:?}", new_generation);
        if let Some(current_generation) = self.dynamic_filter_generation.as_mut() {
            if *current_generation == new_generation {
                return false;
            }
            *current_generation = new_generation;
        } else {
            self.dynamic_filter_generation = Some(new_generation);
        }

        println!("========== PROCESSING GENERATION ========== ");

        let dynamic_expr = if let Some(binary_expr) = self
            .dynamic_filter_expr
            .as_any()
            .downcast_ref::<BinaryExpr>()
            && let Some(dynamic_expr) = binary_expr
                .right()
                .as_any()
                .downcast_ref::<DynamicFilterPhysicalExpr>()
        {
            dynamic_expr
        } else if let Some(dynamic_expr) = self
            .dynamic_filter_expr
            .as_any()
            .downcast_ref::<DynamicFilterPhysicalExpr>()
        {
            dynamic_expr
        } else {
            println!("No dynamic filter expression found - not applying file filtering");
            return false;
        };

        println!("Updating dynamic filter generation to {:?}", new_generation);
        println!("Expr: {:?}", dynamic_expr);

        let current_inner_expr = dynamic_expr.current().expect("Should have current expr");

        let in_list_expr = if let Some(in_list_expr) =
            current_inner_expr.as_any().downcast_ref::<InListExpr>()
        {
            println!("Current dynamic filter is InListExpr: {:?}", in_list_expr);
            InListExpr::new(
                in_list_expr.expr().clone(),
                in_list_expr.list().to_vec(),
                in_list_expr.negated(),
                None,
            )
        } else if let Some(binary_expr) = current_inner_expr.as_any().downcast_ref::<BinaryExpr>()
            && let Some(compacted_in_list) = in_list_expr_compactor(binary_expr.clone())
        {
            compacted_in_list
        } else {
            return false;
        };

        let columns = collect_columns(&self.dynamic_filter_expr);
        println!("Required columns for pruning: {:?}", columns);

        let schema = batch.schema();
        println!("Batch schema for pruning: {:?}", schema);

        let mut column_results = vec![];
        for col in columns {
            let prunable_statistics = Box::new(PrunableStatistics::new(
                vec![Arc::clone(&self.statistics)],
                Arc::clone(&schema),
            ));

            let mut required_columns = RequiredColumns::default();
            let field = schema
                .field_with_name(col.name())
                .expect("Field should exist in schema");
            let min_column = required_columns
                .stat_column(&col, field, datafusion_pruning::StatisticsType::Min)
                .expect("should get stat column");
            let max_column = required_columns
                .stat_column(&col, field, datafusion_pruning::StatisticsType::Max)
                .expect("should get stat column");

            println!("Required columns for pruning: {:?}", required_columns);

            let statistics_batch =
                build_statistics_record_batch(prunable_statistics.as_ref(), &required_columns)
                    .expect("Should build statistics record batch");

            // pull the min/max values from the statistics batch
            let min_array = statistics_batch
                .column_by_name(min_column.name())
                .expect("Should get min column");
            let max_array = statistics_batch
                .column_by_name(max_column.name())
                .expect("Should get max column");

            let min_value =
                ScalarValue::try_from_array(min_array, 0).expect("Should get min scalar value");
            let max_value =
                ScalarValue::try_from_array(max_array, 0).expect("Should get max scalar value");

            println!(
                "Column: {}, Min: {:?}, Max: {:?}",
                col.name(),
                min_value,
                max_value
            );

            column_results.push(min_max_within_in_list(
                &in_list_expr,
                (min_value, max_value),
            ));
        }

        if column_results.iter().all(|&r| r) {
            println!("Not pruning file based on dynamic filter");
            false
        } else {
            println!("Pruning file based on dynamic filter");
            true
        }
    }
}

impl<S> Stream for VortexStoppingStream<S>
where
    S: Stream<Item = DFResult<RecordBatch>> + Unpin,
{
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }

        match ready!(self.inner.poll_next_unpin(cx)) {
            None => {
                self.done = true;
                Poll::Ready(None)
            }
            Some(batch) => {
                let batch = batch.unwrap();
                if self.should_prune(&batch) {
                    self.done = true;
                    Poll::Ready(None)
                } else {
                    Poll::Ready(Some(Ok(batch)))
                }
            }
        }
    }
}

/// If the file has a [`FileRange`](datafusion::datasource::listing::FileRange), we translate it into a row range in the file for the scan.
fn apply_byte_range(
    file_range: FileRange,
    total_size: u64,
    row_count: u64,
    scan_builder: ScanBuilder<ArrayRef>,
) -> ScanBuilder<ArrayRef> {
    let row_range = byte_range_to_row_range(
        file_range.start as u64..file_range.end as u64,
        row_count,
        total_size,
    );

    scan_builder.with_row_range(row_range)
}

fn byte_range_to_row_range(byte_range: Range<u64>, row_count: u64, total_size: u64) -> Range<u64> {
    let average_row = total_size / row_count;
    assert!(average_row > 0, "A row must always have at least one byte");

    let start_row = byte_range.start / average_row;
    let end_row = byte_range.end / average_row;

    // We take the min here as `end_row` might overshoot
    start_row..u64::min(row_count, end_row)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use arrow_schema::Fields;
    use chrono::Utc;
    use datafusion::arrow::array::{RecordBatch, StringArray, StructArray};
    use datafusion::arrow::datatypes::{DataType, Schema};
    use datafusion::arrow::util::display::FormatOptions;
    use datafusion::common::record_batch;
    use datafusion::datasource::schema_adapter::DefaultSchemaAdapterFactory;
    use datafusion::logical_expr::{col, lit};
    use datafusion::physical_expr::planner::logical2physical;
    use datafusion::physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
    use datafusion::scalar::ScalarValue;
    use datafusion_physical_expr::expressions::Literal;
    use insta::assert_snapshot;
    use itertools::Itertools;
    use object_store::ObjectMeta;
    use object_store::memory::InMemory;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::arrow::FromArrowArray;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::{ObjectStoreWriter, VortexWrite};
    use vortex::session::VortexSession;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

    #[rstest]
    #[case(0..100, 100, 100, 0..100)]
    #[case(0..105, 100, 105, 0..100)]
    #[case(0..50, 100, 105, 0..50)]
    #[case(50..105, 100, 105, 50..100)]
    #[case(0..1, 4, 8, 0..0)]
    #[case(1..8, 4, 8, 0..4)]
    fn test_range_translation(
        #[case] byte_range: Range<u64>,
        #[case] row_count: u64,
        #[case] total_size: u64,
        #[case] expected: Range<u64>,
    ) {
        assert_eq!(
            byte_range_to_row_range(byte_range, row_count, total_size),
            expected
        );
    }

    #[test]
    fn test_consecutive_ranges() {
        let row_count = 100;
        let total_size = 429;
        let bytes_a = 0..143;
        let bytes_b = 143..286;
        let bytes_c = 286..429;

        let rows_a = byte_range_to_row_range(bytes_a, row_count, total_size);
        let rows_b = byte_range_to_row_range(bytes_b, row_count, total_size);
        let rows_c = byte_range_to_row_range(bytes_c, row_count, total_size);

        assert_eq!(rows_a.end - rows_a.start, 35);
        assert_eq!(rows_b.end - rows_b.start, 36);
        assert_eq!(rows_c.end - rows_c.start, 29);

        assert_eq!(rows_a.start, 0);
        assert_eq!(rows_c.end, 100);
        for (left, right) in [rows_a, rows_b, rows_c].iter().tuple_windows() {
            assert_eq!(left.end, right.start);
        }
    }

    async fn write_arrow_to_vortex(
        object_store: Arc<dyn ObjectStore>,
        path: &str,
        rb: RecordBatch,
    ) -> anyhow::Result<u64> {
        let array = ArrayRef::from_arrow(rb, false);
        let path = Path::parse(path)?;

        let mut write = ObjectStoreWriter::new(object_store, &path).await?;
        let summary = SESSION
            .write_options()
            .write(&mut write, array.to_array_stream())
            .await?;
        write.shutdown().await?;

        Ok(summary.size())
    }

    fn make_meta(path: &str, data_size: u64) -> FileMeta {
        FileMeta {
            object_meta: ObjectMeta {
                location: Path::from(path),
                last_modified: Utc::now(),
                size: data_size,
                e_tag: None,
                version: None,
            },
            range: None,
            extensions: None,
            metadata_size_hint: None,
        }
    }

    #[rstest]
    #[case(Some(Arc::new(DefaultPhysicalExprAdapterFactory) as _), (1, 3), (0, 0))]
    // If we don't have a physical expr adapter, we just drop filters on partition values
    #[case(None, (1, 3), (1, 3))]
    #[tokio::test]
    async fn test_adapter_optimization_partition_column(
        #[case] expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
        #[case] expected_result1: (usize, usize),
        #[case] expected_result2: (usize, usize),
    ) -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "part=1/file.vortex";
        let batch = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let data_size =
            write_arrow_to_vortex(object_store.clone(), file_path, batch.clone()).await?;

        let file_schema = batch.schema();
        let mut file = PartitionedFile::new(file_path.to_string(), data_size);
        file.partition_values = vec![ScalarValue::Int32(Some(1))];

        let table_schema = Arc::new(Schema::new(vec![
            Field::new("part", DataType::Int32, false),
            Field::new("a", DataType::Int32, false),
        ]));

        let make_opener = |filter| VortexOpener {
            session: SESSION.clone(),
            object_store: object_store.clone(),
            projection: Some([0].into()),
            filter: Some(filter),
            file_pruning_predicate: None,
            expr_adapter_factory: expr_adapter_factory.clone(),
            schema_adapter_factory: Arc::new(DefaultSchemaAdapterFactory),
            partition_fields: vec![Arc::new(Field::new("part", DataType::Int32, false))],
            file_cache: VortexFileCache::new(1, 1, SESSION.clone()),
            logical_schema: file_schema.clone(),
            batch_size: 100,
            limit: None,
            metrics: Default::default(),
            layout_readers: Default::default(),
            has_output_ordering: false,
        };

        // filter matches partition value
        let filter = col("part").eq(lit(1));
        let filter = logical2physical(&filter, table_schema.as_ref());

        let opener = make_opener(filter);
        let stream = opener
            .open(make_meta(file_path, data_size), file.clone())
            .unwrap()
            .await
            .unwrap();

        let data = stream.try_collect::<Vec<_>>().await?;
        let num_batches = data.len();
        let num_rows = data.iter().map(|rb| rb.num_rows()).sum::<usize>();

        assert_eq!((num_batches, num_rows), expected_result1);

        // filter doesn't matches partition value
        let filter = col("part").eq(lit(2));
        let filter = logical2physical(&filter, table_schema.as_ref());

        let opener = make_opener(filter);
        let stream = opener
            .open(make_meta(file_path, data_size), file.clone())
            .unwrap()
            .await
            .unwrap();

        let data = stream.try_collect::<Vec<_>>().await?;
        let num_batches = data.len();
        let num_rows = data.iter().map(|rb| rb.num_rows()).sum::<usize>();
        assert_eq!((num_batches, num_rows), expected_result2);

        Ok(())
    }

    #[rstest]
    #[case(Some(Arc::new(DefaultPhysicalExprAdapterFactory) as _))]
    // If we don't have a physical expr adapter, we just drop filters on partition values.
    // This is currently not supported, the work to support it requires to rewrite the predicate with appropriate casts.
    // Seems like datafusion is moving towards having DefaultPhysicalExprAdapterFactory be always provided, which would make it work OOTB.
    // See: https://github.com/apache/datafusion/issues/16800
    // #[case(None)]
    #[tokio::test]
    async fn test_open_files_different_table_schema(
        #[case] expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    ) -> anyhow::Result<()> {
        use datafusion::arrow::util::pretty::pretty_format_batches_with_options;

        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file1_path = "/path/file1.vortex";
        let batch1 = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let data_size1 = write_arrow_to_vortex(object_store.clone(), file1_path, batch1).await?;
        let file1 = PartitionedFile::new(file1_path.to_string(), data_size1);

        let file2_path = "/path/file2.vortex";
        let batch2 = record_batch!(("a", Int16, vec![Some(-1), Some(-2), Some(-3)])).unwrap();
        let data_size2 = write_arrow_to_vortex(object_store.clone(), file2_path, batch2).await?;
        let file2 = PartitionedFile::new(file1_path.to_string(), data_size1);

        // Table schema has can accommodate both files
        let table_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));

        let make_opener = |filter| VortexOpener {
            session: SESSION.clone(),
            object_store: object_store.clone(),
            projection: Some([0].into()),
            filter: Some(filter),
            file_pruning_predicate: None,
            expr_adapter_factory: expr_adapter_factory.clone(),
            schema_adapter_factory: Arc::new(DefaultSchemaAdapterFactory),
            partition_fields: vec![],
            file_cache: VortexFileCache::new(1, 1, SESSION.clone()),
            logical_schema: table_schema.clone(),
            batch_size: 100,
            limit: None,
            metrics: Default::default(),
            layout_readers: Default::default(),
            has_output_ordering: false,
        };

        let filter = col("a").lt(lit(100_i32));
        let filter = logical2physical(&filter, table_schema.as_ref());

        let opener1 = make_opener(filter.clone());
        let stream = opener1
            .open(make_meta(file1_path, data_size1), file1)?
            .await?;

        let format_opts = FormatOptions::new().with_types_info(true);

        let data = stream.try_collect::<Vec<_>>().await?;
        assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
        +-------+
        | a     |
        | Int32 |
        +-------+
        | 1     |
        | 2     |
        | 3     |
        +-------+
        ");

        let opener2 = make_opener(filter.clone());
        let stream = opener2
            .open(make_meta(file2_path, data_size2), file2)?
            .await?;

        let data = stream.try_collect::<Vec<_>>().await?;
        assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
        +-------+
        | a     |
        | Int32 |
        +-------+
        | -1    |
        | -2    |
        | -3    |
        +-------+
        ");

        Ok(())
    }

    #[tokio::test]
    // This test verifies that expression rewriting doesn't fail when there is
    // a nested schema mismatch between the physical file schema and logical
    // table schema.
    async fn test_adapter_logical_physical_struct_mismatch() -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "/path/file.vortex";
        let file_struct_fields = Fields::from(vec![
            Field::new("field1", DataType::Utf8, true),
            Field::new("field2", DataType::Utf8, true),
        ]);
        let struct_array = StructArray::new(
            file_struct_fields.clone(),
            vec![
                Arc::new(StringArray::from(vec!["value1", "value2", "value3"])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
            None,
        );
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "my_struct",
                DataType::Struct(file_struct_fields),
                true,
            )])),
            vec![Arc::new(struct_array)],
        )?;
        let data_size = write_arrow_to_vortex(object_store.clone(), file_path, batch).await?;

        // Table schema has an extra utf8 field.
        let table_schema = Arc::new(Schema::new(vec![Field::new(
            "my_struct",
            DataType::Struct(Fields::from(vec![
                Field::new(
                    "field1",
                    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                    true,
                ),
                Field::new(
                    "field2",
                    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                    true,
                ),
                Field::new("field3", DataType::Utf8, true),
            ])),
            true,
        )]));

        let opener = VortexOpener {
            session: SESSION.clone(),
            object_store: object_store.clone(),
            projection: None,
            filter: Some(logical2physical(
                &col("my_struct").is_not_null(),
                &table_schema,
            )),
            file_pruning_predicate: None,
            expr_adapter_factory: Some(Arc::new(DefaultPhysicalExprAdapterFactory) as _),
            schema_adapter_factory: Arc::new(DefaultSchemaAdapterFactory),
            partition_fields: vec![],
            file_cache: VortexFileCache::new(1, 1, SESSION.clone()),
            logical_schema: table_schema,
            batch_size: 100,
            limit: None,
            metrics: Default::default(),
            layout_readers: Default::default(),
            has_output_ordering: false,
        };

        // The opener should be able to open the file with a filter on the
        // struct column.
        let data = opener
            .open(
                make_meta(file_path, data_size),
                PartitionedFile::new(file_path.to_string(), data_size),
            )?
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(data.len(), 1);
        assert_eq!(data[0].num_rows(), 3);

        Ok(())
    }

    /// Creates file statistics for testing with the given min/max values for column "a"
    fn make_file_statistics(min_value: i32, max_value: i32) -> Arc<Statistics> {
        Arc::new(Statistics {
            num_rows: datafusion_common::stats::Precision::Exact(100),
            total_byte_size: datafusion_common::stats::Precision::Absent,
            column_statistics: vec![datafusion_common::ColumnStatistics {
                null_count: datafusion_common::stats::Precision::Exact(0),
                min_value: datafusion_common::stats::Precision::Exact(ScalarValue::Int32(Some(
                    min_value,
                ))),
                max_value: datafusion_common::stats::Precision::Exact(ScalarValue::Int32(Some(
                    max_value,
                ))),
                sum_value: datafusion_common::stats::Precision::Absent,
                distinct_count: datafusion_common::stats::Precision::Absent,
            }],
        })
    }

    #[test]
    fn test_contiguous_ranges_calculation() {
        let list_array = ScalarValue::new_list(
            &vec![
                ScalarValue::Int32(Some(1)),
                ScalarValue::Int32(Some(2)),
                ScalarValue::Int32(Some(3)),
                ScalarValue::Int32(Some(5)),
                ScalarValue::Int32(Some(6)),
                ScalarValue::Int32(Some(8)),
            ],
            &DataType::Int32,
            false,
        );

        let scalar_list = ScalarValue::List(list_array);

        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        let in_list_expr = InListExpr::new(
            datafusion_physical_expr::expressions::col("a", &schema).unwrap(),
            vec![Arc::new(Literal::new(scalar_list))],
            false,
            None,
        );
        let ranges = contiguous_in_list_ranges(&in_list_expr, 0);
        assert_eq!(ranges.len(), 3, "Should find 3 contiguous ranges");

        assert_eq!(ranges[0].to_string(), "a@0 >= 1 AND a@0 <= 3");
        assert_eq!(ranges[1].to_string(), "a@0 >= 5 AND a@0 <= 6");
        assert_eq!(ranges[2].to_string(), "a@0 >= 8 AND a@0 <= 8");

        let ranges = contiguous_in_list_ranges(&in_list_expr, 1);
        assert_eq!(
            ranges.len(),
            1,
            "Should find 1 contiguous ranges with overlap of 1"
        );

        assert_eq!(ranges[0].to_string(), "a@0 >= 1 AND a@0 <= 8");
    }

    #[test]
    fn test_min_max_within_in_list() {
        let list_array = ScalarValue::new_list(
            &vec![
                ScalarValue::Int32(Some(1)),
                ScalarValue::Int32(Some(2)),
                ScalarValue::Int32(Some(3)),
                ScalarValue::Int32(Some(5)),
                ScalarValue::Int32(Some(6)),
                ScalarValue::Int32(Some(8)),
            ],
            &DataType::Int32,
            false,
        );

        let scalar_list = ScalarValue::List(list_array);

        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        let in_list_expr = InListExpr::new(
            datafusion_physical_expr::expressions::col("a", &schema).unwrap(),
            vec![Arc::new(Literal::new(scalar_list))],
            false,
            None,
        );

        assert!(min_max_within_in_list(
            &in_list_expr,
            (ScalarValue::Int32(Some(1)), ScalarValue::Int32(Some(3)))
        ));
        assert!(!min_max_within_in_list(
            &in_list_expr,
            (ScalarValue::Int32(Some(4)), ScalarValue::Int32(Some(4)))
        ));
        assert!(min_max_within_in_list(
            &in_list_expr,
            (ScalarValue::Int32(Some(5)), ScalarValue::Int32(Some(6)))
        ));
        assert!(!min_max_within_in_list(
            &in_list_expr,
            (ScalarValue::Int32(Some(7)), ScalarValue::Int32(Some(7)))
        ));
        assert!(min_max_within_in_list(
            &in_list_expr,
            (ScalarValue::Int32(Some(8)), ScalarValue::Int32(Some(8)))
        ));
        assert!(min_max_within_in_list(
            &in_list_expr,
            (ScalarValue::Int32(Some(0)), ScalarValue::Int32(Some(8)))
        ));
        assert!(min_max_within_in_list(
            &in_list_expr,
            (ScalarValue::Int32(Some(4)), ScalarValue::Int32(Some(8)))
        ));
        assert!(!min_max_within_in_list(
            &in_list_expr,
            (ScalarValue::Int32(Some(10)), ScalarValue::Int32(Some(14)))
        ));
    }

    #[tokio::test]
    async fn test_vortex_stopping_stream_continues_without_update() {
        // Setup: Create a schema and some test batches
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        // Create test batches with values 1-10
        let batch1 = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let batch2 = record_batch!(("a", Int32, vec![Some(4), Some(5), Some(6)])).unwrap();
        let batch3 = record_batch!(("a", Int32, vec![Some(7), Some(8), Some(9)])).unwrap();

        // Create a stream from the batches
        let inner_stream = stream::iter(vec![Ok(batch1), Ok(batch2), Ok(batch3)]);

        // Create a dynamic filter expression: a > 0 (should always pass)
        // This starts with a simple predicate that doesn't prune anything
        let col_a =
            datafusion_physical_expr::expressions::col("a", &schema).expect("should create column");
        let lit_0 = datafusion_physical_expr::expressions::lit(ScalarValue::Int32(Some(0)));
        let initial_expr =
            Arc::new(BinaryExpr::new(col_a.clone(), Operator::Gt, lit_0)) as Arc<dyn PhysicalExpr>;

        let dynamic_filter = Arc::new(DynamicFilterPhysicalExpr::new(vec![col_a], initial_expr));
        let statistics = make_file_statistics(1, 10);

        // Create the stopping stream
        let stopping_stream =
            VortexStoppingStream::new(inner_stream, dynamic_filter.clone(), statistics);
        futures::pin_mut!(stopping_stream);

        // Without updating the filter, all batches should pass through
        let results: Vec<_> = stopping_stream.try_collect().await.unwrap();
        assert_eq!(results.len(), 3, "All batches should pass through");
    }

    #[tokio::test]
    async fn test_vortex_stopping_stream_stops_after_update() {
        // Setup: Create a schema and some test batches
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        // Create test batches with values 1-10
        let batch1 = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let batch2 = record_batch!(("a", Int32, vec![Some(4), Some(5), Some(6)])).unwrap();
        let batch3 = record_batch!(("a", Int32, vec![Some(7), Some(8), Some(9)])).unwrap();

        // Create a stream from the batches
        let inner_stream = stream::iter(vec![Ok(batch1), Ok(batch2), Ok(batch3)]);

        // default to lit true
        let col_a =
            datafusion_physical_expr::expressions::col("a", &schema).expect("should create column");
        let lit_true = datafusion_physical_expr::expressions::lit(ScalarValue::Boolean(Some(true)));

        let dynamic_filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col_a.clone()],
            lit_true,
        ));

        let statistics = make_file_statistics(1, 10);

        // Create the stopping stream
        let stopping_stream =
            VortexStoppingStream::new(inner_stream, dynamic_filter.clone(), statistics);
        futures::pin_mut!(stopping_stream);

        // Read the first batch
        let first_batch = stopping_stream.next().await.unwrap().unwrap();
        assert_eq!(first_batch.num_rows(), 3, "First batch should pass through");

        let col_a =
            datafusion_physical_expr::expressions::col("a", &schema).expect("should create column");

        let list_array = ScalarValue::new_list(
            &vec![
                ScalarValue::Int32(Some(10)),
                ScalarValue::Int32(Some(13)),
                ScalarValue::Int32(Some(14)),
            ],
            &DataType::Int32,
            false,
        );

        let in_expr = InListExpr::new(
            col_a.clone(),
            vec![Arc::new(Literal::new(ScalarValue::List(list_array)))],
            false,
            None,
        );
        dynamic_filter
            .update(Arc::new(in_expr))
            .expect("should update filter");

        // stream will still continue as 10 is within the max of the file
        let second_batch = stopping_stream.next().await;
        assert!(second_batch.is_some(), "Stream should continue");

        let list_array = ScalarValue::new_list(
            &vec![
                ScalarValue::Int32(Some(12)),
                ScalarValue::Int32(Some(13)),
                ScalarValue::Int32(Some(14)),
            ],
            &DataType::Int32,
            false,
        );

        let in_expr = InListExpr::new(
            col_a.clone(),
            vec![Arc::new(Literal::new(ScalarValue::List(list_array)))],
            false,
            None,
        );
        dynamic_filter
            .update(Arc::new(in_expr))
            .expect("should update filter");

        // stream should stop now as no values overlap with file stats
        let third_batch = stopping_stream.next().await;
        assert!(third_batch.is_none(), "Stream should have stopped");
    }

    fn random_values_in_list(range: Range<i32>, count: usize) -> Vec<ScalarValue> {
        use rand::prelude::*;
        let mut rng = rand::rng();
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            let value = rng.random_range(range.clone());
            values.push(ScalarValue::Int32(Some(value)));
        }
        values
    }

    #[tokio::test]
    async fn test_vortex_stopping_stream_over_wide_range() {
        // Setup: Create a schema and some test batches
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        // Create test batches with values 1-10
        let batch1 = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let batch2 = record_batch!(("a", Int32, vec![Some(4), Some(5), Some(6)])).unwrap();
        let batch3 = record_batch!(("a", Int32, vec![Some(7), Some(8), Some(9)])).unwrap();

        // Create a stream from the batches
        let inner_stream = stream::iter(vec![Ok(batch1), Ok(batch2), Ok(batch3)]);

        let col_a =
            datafusion_physical_expr::expressions::col("a", &schema).expect("should create column");
        // generate a large number of values within the file statistics
        let values = random_values_in_list(1..50000, 30000);

        let list_array = ScalarValue::new_list(&values, &DataType::Int32, false);

        let in_expr = InListExpr::new(
            col_a.clone(),
            vec![Arc::new(Literal::new(ScalarValue::List(list_array)))],
            false,
            None,
        );

        let dynamic_filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col_a.clone()],
            Arc::new(in_expr),
        ));

        let statistics = make_file_statistics(1, 50000);

        // Create the stopping stream
        let stopping_stream =
            VortexStoppingStream::new(inner_stream, dynamic_filter.clone(), statistics);
        futures::pin_mut!(stopping_stream);

        // Read the first batch
        let first_batch = stopping_stream.next().await.unwrap().unwrap();
        assert_eq!(first_batch.num_rows(), 3, "First batch should pass through");

        // generate a large number of values with some overlap outside of the file statistics
        let values = random_values_in_list(25000..75000, 30000);

        let list_array = ScalarValue::new_list(&values, &DataType::Int32, false);

        let in_expr = InListExpr::new(
            col_a.clone(),
            vec![Arc::new(Literal::new(ScalarValue::List(list_array)))],
            false,
            None,
        );
        dynamic_filter
            .update(Arc::new(in_expr))
            .expect("should update filter");

        // stream will still continue as values should overlap with the file stats
        let second_batch = stopping_stream.next().await;
        assert!(second_batch.is_some(), "Stream should continue");

        // all generated values are outside of the file statistics now
        let values = random_values_in_list(55000..100000, 30000);

        let list_array = ScalarValue::new_list(&values, &DataType::Int32, false);

        let in_expr = InListExpr::new(
            col_a.clone(),
            vec![Arc::new(Literal::new(ScalarValue::List(list_array)))],
            false,
            None,
        );
        dynamic_filter
            .update(Arc::new(in_expr))
            .expect("should update filter");

        // stream should stop now as no values overlap with file stats
        let third_batch = stopping_stream.next().await;
        assert!(third_batch.is_none(), "Stream should have stopped");
    }

    // #[tokio::test]
    // async fn test_vortex_stopping_stream_prunes_after_update() {
    //     // Setup: Create a schema and some test batches
    //     let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

    //     // Create test batches
    //     let batch1 = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
    //     let batch2 = record_batch!(("a", Int32, vec![Some(4), Some(5), Some(6)])).unwrap();
    //     let batch3 = record_batch!(("a", Int32, vec![Some(7), Some(8), Some(9)])).unwrap();

    //     // Create a dynamic filter expression that initially passes everything
    //     let col_a =
    //         datafusion_physical_expr::expressions::col("a", &schema).expect("should create column");
    //     let lit_true = datafusion_physical_expr::expressions::lit(ScalarValue::Boolean(Some(true)));

    //     let dynamic_filter = Arc::new(DynamicFilterPhysicalExpr::new(
    //         vec![col_a.clone()],
    //         lit_true,
    //     ));

    //     // Statistics indicate file has values 1-10 (this is what would be pruned against)
    //     let statistics = make_file_statistics(1, 10);

    //     // Create the inner stream with an interleaved update to the dynamic filter
    //     // After the first batch is yielded, we update the filter to prune remaining batches
    //     let dynamic_filter_clone = dynamic_filter.clone();
    //     let schema_clone = schema.clone();
    //     let batches = vec![batch1, batch2, batch3];
    //     let mut batch_iter = batches.into_iter();

    //     let inner_stream = stream::unfold(
    //         (batch_iter, dynamic_filter_clone, schema_clone, 0usize),
    //         |(mut iter, filter, schema, count)| async move {
    //             let batch = iter.next()?;

    //             // After yielding the first batch, update the filter to prune
    //             // The new filter is `a > 1000` which should not match file stats (1-10)
    //             if count == 0 {
    //                 let col_a = datafusion_physical_expr::expressions::col("a", &schema)
    //                     .expect("should create column");
    //                 let lit_1000 =
    //                     datafusion_physical_expr::expressions::lit(ScalarValue::Int32(Some(1000)));
    //                 let new_expr = Arc::new(BinaryExpr::new(
    //                     col_a,
    //                     datafusion_expr::Operator::Gt,
    //                     lit_1000,
    //                 )) as Arc<dyn PhysicalExpr>;
    //                 filter.update(new_expr).expect("should update filter");
    //             }

    //             Some((
    //                 Ok::<_, DataFusionError>(batch),
    //                 (iter, filter, schema, count + 1),
    //             ))
    //         },
    //     );

    //     // Create the stopping stream
    //     let stopping_stream =
    //         VortexStoppingStream::new(inner_stream, dynamic_filter.clone(), statistics);
    //     futures::pin_mut!(stopping_stream);

    //     // The stream should stop after the filter update causes pruning
    //     let results: Vec<_> = stopping_stream.try_collect().await.unwrap();

    //     // We expect to get batch1 (before update), then batch2 should trigger pruning check
    //     // Since the new filter `a > 1000` doesn't overlap with stats min=1, max=10, it should prune
    //     println!("Got {} batches", results.len());
    //     for (i, batch) in results.iter().enumerate() {
    //         println!("Batch {}: {} rows", i, batch.num_rows());
    //     }
    //     // TODO: Once should_prune is fully implemented, this should be:
    //     // assert!(results.len() < 3, "Stream should have been pruned");
    // }

    // #[tokio::test]
    // async fn test_vortex_stopping_stream_generation_tracking() {
    //     // Test that the stream correctly tracks generation changes
    //     let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

    //     let batch = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();

    //     let col_a =
    //         datafusion_physical_expr::expressions::col("a", &schema).expect("should create column");
    //     let lit_true = datafusion_physical_expr::expressions::lit(ScalarValue::Boolean(Some(true)));

    //     let dynamic_filter = Arc::new(DynamicFilterPhysicalExpr::new(
    //         vec![col_a.clone()],
    //         lit_true,
    //     ));

    //     // Verify initial generation
    //     let initial_gen =
    //         datafusion_physical_expr_common::physical_expr::snapshot_generation(&dynamic_filter);
    //     assert_eq!(initial_gen, 1, "Initial generation should be 1");

    //     // Update the filter
    //     let new_lit = datafusion_physical_expr::expressions::lit(ScalarValue::Boolean(Some(false)));
    //     dynamic_filter
    //         .update(new_lit)
    //         .expect("should update filter");

    //     // Verify generation changed
    //     let new_gen =
    //         datafusion_physical_expr_common::physical_expr::snapshot_generation(&dynamic_filter);
    //     assert_eq!(new_gen, 2, "Generation should increment after update");

    //     // Create a stopping stream and verify it tracks generations
    //     let statistics = make_file_statistics(1, 10);
    //     let inner_stream = stream::iter(vec![Ok(batch)]);
    //     let mut stopping_stream =
    //         VortexStoppingStream::new(inner_stream, dynamic_filter.clone(), statistics);

    //     // Initially no generation tracked
    //     assert!(
    //         stopping_stream.dynamic_filter_generation.is_none(),
    //         "No generation tracked initially"
    //     );
    // }
}
