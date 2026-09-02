// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Minimal reproduction for allocation churn in expression optimization and evaluation.
//!
//! Each scenario is run as its own process so that a heap profiler attributes allocations to a
//! single code path:
//!
//! ```text
//! cargo build --profile samply --example expr_alloc
//! ./target/samply/examples/expr_alloc prepare
//! heaptrack ./target/samply/examples/expr_alloc optimize 2000
//! heaptrack ./target/samply/examples/expr_alloc apply 2000
//! heaptrack ./target/samply/examples/expr_alloc execute 2000
//! heaptrack ./target/samply/examples/expr_alloc scan 20
//! ```

#![expect(clippy::unwrap_used)]
#![expect(clippy::unwrap_in_result)]
#![expect(clippy::panic)]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::print_stdout)]

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::ByteBufferMut;
use vortex::dtype::DType;
use vortex::dtype::Nullability::NonNullable;
use vortex::dtype::Nullability::Nullable;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::and;
use vortex::expr::and_collect;
use vortex::expr::cast;
use vortex::expr::eq;
use vortex::expr::get_item;
use vortex::expr::gt;
use vortex::expr::gt_eq;
use vortex::expr::lit;
use vortex::expr::lt;
use vortex::expr::lt_eq;
use vortex::expr::or_collect;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

/// Rows per batch, matching a typical scan batch size.
const BATCH: usize = 8192;
/// Number of batches written to the repro file.
const BATCHES: usize = 64;

fn scope() -> DType {
    DType::Struct(
        StructFields::new(
            [
                "l_orderkey",
                "l_quantity",
                "l_extendedprice",
                "l_discount",
                "l_shipdate",
                "l_returnflag",
                "l_shipmode",
            ]
            .into(),
            vec![
                DType::Primitive(PType::I64, NonNullable),
                DType::Primitive(PType::I64, NonNullable),
                DType::Primitive(PType::F64, NonNullable),
                DType::Primitive(PType::F64, NonNullable),
                DType::Primitive(PType::I32, NonNullable),
                DType::Utf8(NonNullable),
                DType::Utf8(Nullable),
            ],
        ),
        NonNullable,
    )
}

/// A TPC-H Q6-flavoured pushdown filter: a range predicate, two bounds on a float column,
/// a scalar bound, and an `IN` list expanded into an `OR` chain.
fn filter_expr() -> Expression {
    let flags = or_collect(
        ["A", "R", "N"]
            .into_iter()
            .map(|f| eq(get_item("l_returnflag", root()), lit(f))),
    )
    .unwrap();

    and(
        and(
            gt_eq(get_item("l_shipdate", root()), lit(9131i32)),
            lt(get_item("l_shipdate", root()), lit(9496i32)),
        ),
        and(
            and(
                gt_eq(get_item("l_discount", root()), lit(0.05f64)),
                lt_eq(get_item("l_discount", root()), lit(0.07f64)),
            ),
            and(lt(get_item("l_quantity", root()), lit(24i64)), flags),
        ),
    )
}

/// The same filter, but with the explicit casts a query engine inserts when pushing a predicate
/// down to a differently-typed column. `Cast::reduce` calls `ReduceNode::node_dtype`, which is
/// the expensive branch of the reduce path.
fn filter_expr_cast() -> Expression {
    let f64_ = DType::Primitive(PType::F64, NonNullable);
    let i64_ = DType::Primitive(PType::I64, NonNullable);
    let flags = or_collect(
        ["A", "R", "N"]
            .into_iter()
            .map(|f| eq(get_item("l_returnflag", root()), lit(f))),
    )
    .unwrap();

    and(
        and(
            gt_eq(
                cast(get_item("l_shipdate", root()), i64_.clone()),
                lit(9131i64),
            ),
            lt(
                cast(get_item("l_shipdate", root()), i64_.clone()),
                lit(9496i64),
            ),
        ),
        and(
            and(
                gt_eq(
                    cast(get_item("l_discount", root()), f64_.clone()),
                    lit(0.05f64),
                ),
                lt_eq(cast(get_item("l_discount", root()), f64_), lit(0.07f64)),
            ),
            and(
                lt(cast(get_item("l_quantity", root()), i64_), lit(24i64)),
                flags,
            ),
        ),
    )
}

/// An `IN (...)` list expanded into a balanced `OR` chain of `n` casted equality predicates,
/// which is what a query engine produces for a large `IN` list. Used to measure how allocation
/// count scales with expression size.
fn or_chain(n: usize) -> Expression {
    let i64_ = DType::Primitive(PType::I64, NonNullable);
    or_collect((0..n).map(|i| {
        eq(
            cast(get_item("l_quantity", root()), i64_.clone()),
            lit(i as i64),
        )
    }))
    .unwrap()
}

/// An `AND` chain of `n` same-direction range predicates. No pair can ever form a `between`,
/// so this measures what the unconditional `find_between` CNF rebuild costs when it finds nothing.
fn and_chain(n: usize) -> Expression {
    and_collect((0..n).map(|i| gt(get_item("l_orderkey", root()), lit(i as i64)))).unwrap()
}

/// A right-leaning `AND` chain of `n` predicates, as a planner emits before any normalization:
/// `and(p0, and(p1, and(p2, ...)))`. Used to check what happens to spine depth, since
/// `and_collect` rebalances and exists to keep later recursive passes off a long chain.
fn deep_and_chain(n: usize) -> Expression {
    (0..n)
        .rev()
        .map(|i| gt(get_item("l_orderkey", root()), lit(i as i64)))
        .reduce(|acc, p| and(p, acc))
        .unwrap()
}

/// Longest root-to-leaf path through the expression, counting nodes.
fn depth(expr: &Expression) -> usize {
    1 + expr.children().iter().map(depth).max().unwrap_or(0)
}

fn projection_expr() -> Expression {
    select(["l_orderkey", "l_extendedprice", "l_discount"], root())
}

fn batch(seed: usize) -> ArrayRef {
    let base = (seed * BATCH) as i64;
    StructArray::from_fields(&[
        (
            "l_orderkey",
            PrimitiveArray::from_iter((0..BATCH as i64).map(|i| base + i)).into_array(),
        ),
        (
            "l_quantity",
            PrimitiveArray::from_iter((0..BATCH as i64).map(|i| (i % 50) + 1)).into_array(),
        ),
        (
            "l_extendedprice",
            PrimitiveArray::from_iter((0..BATCH).map(|i| 900.0 + (i % 1000) as f64)).into_array(),
        ),
        (
            "l_discount",
            PrimitiveArray::from_iter((0..BATCH).map(|i| (i % 11) as f64 / 100.0)).into_array(),
        ),
        (
            "l_shipdate",
            PrimitiveArray::from_iter((0..BATCH as i32).map(|i| 9000 + (i % 800))).into_array(),
        ),
        (
            "l_returnflag",
            VarBinViewArray::from_iter_str((0..BATCH).map(|i| ["A", "R", "N", "X"][i % 4]))
                .into_array(),
        ),
        (
            "l_shipmode",
            VarBinViewArray::from_iter_str((0..BATCH).map(|i| ["AIR", "RAIL", "SHIP"][i % 3]))
                .into_array(),
        ),
    ])
    .unwrap()
    .into_array()
}

fn repro_path() -> PathBuf {
    env::var_os("VORTEX_ALLOC_REPRO_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| env::temp_dir().join("vortex_expr_alloc_repro.vortex"))
}

/// Run `body` `reps` times over `iters` iterations, reporting the best per-iteration time.
///
/// `EXPR_ALLOC_REPS` defaults to 1 so that heap-profiling runs measure a single pass.
fn timed(
    name: &str,
    iters: usize,
    mut body: impl FnMut() -> VortexResult<usize>,
) -> VortexResult<()> {
    let reps: usize = env::var("EXPR_ALLOC_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let mut sink = 0usize;
    let mut best = f64::INFINITY;
    let mut all = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        for _ in 0..iters {
            sink += body()?;
        }
        let per_iter = start.elapsed().as_secs_f64() * 1e9 / iters as f64;
        all.push(per_iter);
        best = best.min(per_iter);
    }

    let reps_fmt = all
        .iter()
        .map(|ns| format!("{ns:.0}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "{name}: {iters} iters x {reps} reps, best {best:.0} ns/iter [{reps_fmt}], sink={sink}"
    );
    Ok(())
}

fn main() -> VortexResult<()> {
    let mut args = env::args().skip(1);
    let scenario = args.next().unwrap_or_else(|| "optimize".to_string());
    let iters: usize = args
        .next()
        .map(|s| s.parse().unwrap())
        .unwrap_or(match scenario.as_str() {
            "scan" => 20,
            _ => 2000,
        });

    let session = VortexSession::default();
    let _ = &session;

    match scenario.as_str() {
        // Expression simplification only: no arrays involved.
        "optimize" => {
            let scope = scope();
            let expr = filter_expr();
            timed("optimize", iters, || {
                Ok(expr.optimize_recursive(&scope)?.children().len())
            })?;
        }
        "optimize_cast" => {
            let scope = scope();
            let expr = filter_expr_cast();
            timed("optimize_cast", iters, || {
                Ok(expr.optimize_recursive(&scope)?.children().len())
            })?;
        }
        // Constant-time construction of the lazy expression array (includes array-level optimize).
        "chain" => {
            let n: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(32);
            let scope = scope();
            let expr = or_chain(n);
            timed(&format!("chain n={n}"), iters, || {
                Ok(expr.optimize_recursive(&scope)?.children().len())
            })?;
        }
        "and_chain" => {
            let n: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(32);
            let scope = scope();
            let expr = and_chain(n);
            timed(&format!("and_chain n={n}"), iters, || {
                Ok(expr.optimize_recursive(&scope)?.children().len())
            })?;
        }
        "deep_and" => {
            let n: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(512);
            let scope = scope();
            let expr = deep_and_chain(n);
            let optimized = expr.optimize_recursive(&scope)?;
            println!(
                "deep_and: n={n}, input depth {}, optimized depth {}",
                depth(&expr),
                depth(&optimized)
            );
            timed(&format!("deep_and n={n}"), iters, || {
                Ok(expr.optimize_recursive(&scope)?.children().len())
            })?;
        }
        "apply" => {
            let array = batch(0);
            let expr = filter_expr();
            timed("apply", iters, || Ok(array.clone().apply(&expr)?.len()))?;
        }
        // Full per-batch evaluation: apply + execute to canonical.
        "execute" => {
            let array = batch(0);
            let expr = filter_expr();
            let mut ctx = session.create_execution_ctx();
            timed("execute", iters, || {
                Ok(array
                    .clone()
                    .apply(&expr)?
                    .execute::<Canonical>(&mut ctx)?
                    .into_array()
                    .len())
            })?;
        }
        // Write the repro file used by the `scan` scenario.
        "prepare" => {
            let path = repro_path();
            let chunks = ChunkedArray::from_iter((0..BATCHES).map(batch)).into_array();
            let mut buf = ByteBufferMut::empty();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async {
                let session = VortexSession::default().with_tokio();
                session
                    .write_options()
                    .write(&mut buf, chunks.to_array_stream())
                    .await
            })?;
            std::fs::write(&path, buf.as_slice()).unwrap();
            println!(
                "prepare: wrote {} rows to {} ({} bytes)",
                BATCH * BATCHES,
                path.display(),
                buf.len()
            );
        }
        // End-to-end file scan with filter pushdown and projection.
        "scan" => {
            let path = repro_path();
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap();
            let mut sink = 0usize;
            runtime.block_on(async {
                let session = VortexSession::default().with_tokio();
                for _ in 0..iters {
                    let arrays = session
                        .open_options()
                        .open_path(&path)
                        .await?
                        .scan()?
                        .with_filter(filter_expr())
                        .with_projection(projection_expr())
                        .into_array_stream()?
                        .read_all()
                        .await?;
                    sink += arrays.len();
                }
                Ok::<_, vortex::error::VortexError>(())
            })?;
            println!("scan: {iters} iterations, sink={sink}");
        }
        other => panic!("unknown scenario: {other}"),
    }

    Ok(())
}
