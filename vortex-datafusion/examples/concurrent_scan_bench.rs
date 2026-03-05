// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark demonstrating query-time overhead when many Vortex file splits are
//! opened concurrently — the pattern that occurs when a large table is the probe
//! side of a hash join.
//!
//! ## Background
//!
//! Spice (<https://github.com/spiceai/spiceai>) splits large tables into many small
//! Vortex files (~150K rows each). A TPC-H SF1 `lineitem` table produces ~40 files.
//! In hash-join queries (e.g. TPC-H Q5, Q8, Q14), DataFusion opens **all** probe-side
//! splits concurrently after the build side completes.
//!
//! We measured a ~31% total scan-time regression across 16 TPC-H queries between
//! vortex base `5093499e6` and `c536c9aed` (~540 commits), concentrated on the 9
//! queries where lineitem is the hash-join probe side (+10–18ms each).
//!
//! Contributing factors identified via profiling:
//!
//! 1. **`spawn_blocking` in `LazyScanStream`** (PR #5906): `ScanBuilder::prepare()`
//!    moved to `spawn_blocking`. For many concurrent small-file opens, this adds a
//!    thread-pool roundtrip per file. ~23% of the regression.
//!
//! 2. **`execute()` framework overhead** (PRs #5895, #5920, #5922, #5925, #6076,
//!    #6307): The old `RecordBatch::try_from(array)` path was replaced with an
//!    operator execution tree (recursive `execute` → swap child → re-run). For
//!    small arrays, the per-array framework cost is proportionally large.
//!
//! 3. **`DashMap` contention in `VortexSession`** (PR #6000): Every array decode
//!    performs `DashMap` lookups through the session registry. Under concurrent
//!    file opens, this creates lock contention (107 `lock_slow` samples in profiling).
//!
//! ## What this benchmark measures
//!
//! Creates N vortex files, registers them as a DataFusion `ListingTable`, then
//! measures **total wall-clock time** for many repeated queries to push into the
//! multi-second range where noise is negligible:
//!
//! - **Scan-only** (`SELECT SUM(value) FROM probe`): sequential/pipelined file opens
//! - **Hash-join** (`SELECT SUM(p.value) FROM probe JOIN build ...`): concurrent burst
//!   open of all probe files after build side completes
//! - **4× concurrent hash joins**: worst case for thread-pool + lock contention
//!
//! ## Running
//!
//! ```bash
//! cargo run -p vortex-datafusion --release --example concurrent_scan_bench
//! ```

use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::array::{Int32Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use tempfile::tempdir;
use tokio::fs::{self, OpenOptions};
use vortex::array::arrow::FromArrowArray;
use vortex::array::ArrayRef;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;
use vortex_datafusion::VortexFormat;

/// Number of vortex files for the "probe" (large) table.
/// Simulates how Spice splits lineitem into ~40 files at SF1.
const NUM_PROBE_FILES: usize = 40;

/// Rows per probe file. The regression is per-file-open, not per-row, so a
/// moderate count suffices. Production uses ~150K rows/file.
const ROWS_PER_FILE: usize = 50_000;

/// Number of query executions per timed measurement.
/// High enough to push totals into multi-second range.
const QUERY_REPS: usize = 500;

/// Number of rounds for the concurrent test (each round launches 4 queries).
const CONCURRENT_ROUNDS: usize = 100;

/// Number of warm-up measurements (discarded).
const WARMUP_RUNS: usize = 2;

/// Number of timed measurements (median is reported).
const TIMED_RUNS: usize = 6;

/// Build a probe-side RecordBatch (simulating a lineitem-like split).
fn make_probe_batch(file_idx: usize, num_rows: usize) -> RecordBatch {
    let key_start = (file_idx * num_rows) as i32;
    let keys: Vec<i32> = (key_start..key_start + num_rows as i32).collect();
    let join_keys: Vec<i32> = keys.iter().map(|k| k % 1000).collect();
    let values: Vec<i64> = keys.iter().map(|k| (*k as i64) * 100).collect();
    let labels: Vec<String> = keys.iter().map(|k| format!("item_{k}")).collect();

    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("probe_key", DataType::Int32, false),
            Field::new("join_key", DataType::Int32, false),
            Field::new("value", DataType::Int64, false),
            Field::new("label", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int32Array::from(keys)),
            Arc::new(Int32Array::from(join_keys)),
            Arc::new(Int64Array::from(values)),
            Arc::new(StringArray::from(labels)),
        ],
    )
    .unwrap()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

/// Run `reps` sequential query executions, return total wall-clock time in ms.
async fn bench_sequential(ctx: &SessionContext, sql: &str, reps: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..reps {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }
    start.elapsed().as_secs_f64() * 1000.0
}

/// Run `rounds` rounds of 4 concurrent query executions, return total wall-clock time in ms.
async fn bench_concurrent(ctx: &SessionContext, sql: &str, rounds: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..rounds {
        let futs: Vec<_> = (0..4)
            .map(|_| {
                let s = ctx.clone();
                let q = sql.to_string();
                tokio::spawn(async move { s.sql(&q).await.unwrap().collect().await.unwrap() })
            })
            .collect();
        futures::future::join_all(futs).await;
    }
    start.elapsed().as_secs_f64() * 1000.0
}

/// Run warmup + timed measurements, return sorted times in ms.
async fn measure<F, Fut>(warmup: usize, runs: usize, mut f: F) -> Vec<f64>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = f64>,
{
    for _ in 0..warmup {
        f().await;
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        times.push(f().await);
    }
    times
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let session = VortexSession::default().with_tokio();
    let temp_dir = tempdir()?;

    // ── Write probe files ────────────────────────────────────────────────
    let probe_dir = temp_dir.path().join("probe");
    fs::create_dir_all(&probe_dir).await?;

    eprintln!(
        "Writing {NUM_PROBE_FILES} probe files × {ROWS_PER_FILE} rows each ({} total rows)...",
        NUM_PROBE_FILES * ROWS_PER_FILE,
    );
    let write_start = Instant::now();
    for i in 0..NUM_PROBE_FILES {
        let batch = make_probe_batch(i, ROWS_PER_FILE);
        let array = ArrayRef::from_arrow(batch, false);
        let path = probe_dir.join(format!("split_{i:04}.vortex"));
        let mut f = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&path)
            .await?;
        session
            .write_options()
            .write(&mut f, array.to_array_stream())
            .await?;
        if (i + 1) % 10 == 0 {
            eprintln!("  Written {}/{NUM_PROBE_FILES} files...", i + 1);
        }
    }
    eprintln!(
        "  Probe files written in {:.1}s",
        write_start.elapsed().as_secs_f64(),
    );

    // ── Write build table ────────────────────────────────────────────────
    let build_dir = temp_dir.path().join("build");
    fs::create_dir_all(&build_dir).await?;

    let build_keys: Vec<i32> = (0..1000).collect();
    let build_names: Vec<String> = build_keys.iter().map(|k| format!("name_{k}")).collect();
    let build_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("build_key", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int32Array::from(build_keys)),
            Arc::new(StringArray::from(build_names)),
        ],
    )?;
    let build_array = ArrayRef::from_arrow(build_batch, false);
    let build_path = build_dir.join("data.vortex");
    let mut f = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&build_path)
        .await?;
    session
        .write_options()
        .write(&mut f, build_array.to_array_stream())
        .await?;
    eprintln!("  Build table written (1000 rows)");

    // ── Register tables with DataFusion ──────────────────────────────────
    let ctx = SessionContext::new();
    let format = Arc::new(VortexFormat::new(session));

    let probe_url = ListingTableUrl::parse(probe_dir.to_str().unwrap())?;
    let probe_config = ListingTableConfig::new(probe_url)
        .with_listing_options(
            ListingOptions::new(format.clone()).with_session_config_options(ctx.state().config()),
        )
        .infer_schema(&ctx.state())
        .await?;
    ctx.register_table("probe", Arc::new(ListingTable::try_new(probe_config)?))?;

    let build_url = ListingTableUrl::parse(build_path.to_str().unwrap())?;
    let build_config = ListingTableConfig::new(build_url)
        .with_listing_options(
            ListingOptions::new(format).with_session_config_options(ctx.state().config()),
        )
        .infer_schema(&ctx.state())
        .await?;
    ctx.register_table("build", Arc::new(ListingTable::try_new(build_config)?))?;

    // Sanity check
    let count = ctx
        .sql("SELECT COUNT(*) as cnt FROM probe")
        .await?
        .collect()
        .await?;
    let total_rows = count[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(total_rows, (NUM_PROBE_FILES * ROWS_PER_FILE) as i64);
    eprintln!("  Tables registered. Probe: {total_rows} rows across {NUM_PROBE_FILES} files\n");

    let scan_sql = "SELECT SUM(value) FROM probe";
    let join_sql = "\
        SELECT SUM(p.value) \
        FROM probe p \
        JOIN build b ON p.join_key = b.build_key \
        WHERE b.name LIKE 'name_1%'";

    // ── Benchmark 1: Scan-only ───────────────────────────────────────────
    eprintln!(
        "=== Scan-only: {QUERY_REPS}× SELECT SUM(value) FROM probe ===\n\
         (files opened sequentially/pipelined by DataFusion)"
    );
    let mut scan_times = measure(WARMUP_RUNS, TIMED_RUNS, || {
        bench_sequential(&ctx, scan_sql, QUERY_REPS)
    })
    .await;
    let scan_median = median(&mut scan_times);
    eprintln!(
        "  Runs: [{}]",
        scan_times
            .iter()
            .map(|t| format!("{t:.0}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "  Median: {scan_median:.0}ms total ({:.2}ms/query)\n",
        scan_median / QUERY_REPS as f64
    );

    // ── Benchmark 2: Hash join ───────────────────────────────────────────
    eprintln!(
        "=== Hash join: {QUERY_REPS}× probe JOIN build ===\n\
         (all {NUM_PROBE_FILES} probe files opened concurrently after build completes)"
    );
    let mut join_times = measure(WARMUP_RUNS, TIMED_RUNS, || {
        bench_sequential(&ctx, join_sql, QUERY_REPS)
    })
    .await;
    let join_median = median(&mut join_times);
    eprintln!(
        "  Runs: [{}]",
        join_times
            .iter()
            .map(|t| format!("{t:.0}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "  Median: {join_median:.0}ms total ({:.2}ms/query)\n",
        join_median / QUERY_REPS as f64
    );

    // ── Benchmark 3: 4× concurrent hash joins ───────────────────────────
    let total_concurrent_queries = CONCURRENT_ROUNDS * 4;
    eprintln!(
        "=== 4× concurrent hash joins: {CONCURRENT_ROUNDS} rounds × 4 = {total_concurrent_queries} queries ===\n\
         (worst case for thread-pool + DashMap contention)"
    );
    let mut concurrent_times = measure(WARMUP_RUNS, TIMED_RUNS, || {
        bench_concurrent(&ctx, join_sql, CONCURRENT_ROUNDS)
    })
    .await;
    let concurrent_median = median(&mut concurrent_times);
    eprintln!(
        "  Runs: [{}]",
        concurrent_times
            .iter()
            .map(|t| format!("{t:.0}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "  Median: {concurrent_median:.0}ms total ({:.2}ms/query)\n",
        concurrent_median / total_concurrent_queries as f64
    );

    // ── Summary ──────────────────────────────────────────────────────────
    let scan_per_q = scan_median / QUERY_REPS as f64;
    let join_per_q = join_median / QUERY_REPS as f64;
    let concurrent_per_q = concurrent_median / total_concurrent_queries as f64;
    let join_overhead_pct = (join_per_q / scan_per_q - 1.0) * 100.0;

    eprintln!("========================================");
    eprintln!("Summary");
    eprintln!("========================================");
    eprintln!(
        "  {NUM_PROBE_FILES} probe files × {ROWS_PER_FILE} rows = {} total",
        NUM_PROBE_FILES * ROWS_PER_FILE,
    );
    eprintln!();
    eprintln!("  Totals (median of {TIMED_RUNS} runs):");
    eprintln!("    Scan-only ({QUERY_REPS}× seq):       {scan_median:>8.0}ms");
    eprintln!("    Hash-join ({QUERY_REPS}× seq):       {join_median:>8.0}ms");
    eprintln!(
        "    4× concurrent ({total_concurrent_queries} queries): {concurrent_median:>8.0}ms"
    );
    eprintln!();
    eprintln!("  Per-query:");
    eprintln!("    Scan-only:     {scan_per_q:>6.2}ms");
    eprintln!("    Hash-join:     {join_per_q:>6.2}ms  ({join_overhead_pct:+.0}% vs scan)");
    eprintln!("    4× concurrent: {concurrent_per_q:>6.2}ms/query");
    eprintln!();
    eprintln!("  The hash-join forces all {NUM_PROBE_FILES} probe files to open concurrently");
    eprintln!("  after the build side completes. Per-file overhead from:");
    eprintln!("  - spawn_blocking in LazyScanStream (PR #5906)");
    eprintln!("  - execute() framework tree walking (PRs #5925, #6307)");
    eprintln!("  - DashMap contention in VortexSession (PR #6000)");
    eprintln!("  is amplified under concurrent burst opens.");
    eprintln!();
    eprintln!("  In production (TPC-H SF1), this adds ~10–18ms per hash-join query");
    eprintln!("  and ~156ms total across 16 queries (~31% regression).");

    Ok(())
}
