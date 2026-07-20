// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DIAG (cold-stall): steady-state scan-park counters + a side-thread dumper.
//!
//! The cold-tier promotion scan (`LazyScanStream` -> `buffer_unordered` over `handle.spawn`ed
//! split-read tasks) intermittently hangs: the scan returns `Pending` and never re-wakes its
//! consumer (the promotion's SortExec), which holds the table write_lock and wedges ingest.
//!
//! Completion-triggered logging can't see the *frozen steady-state* (once parked, nothing is
//! polled, so no completion fires). This module instead runs a dedicated OS thread that samples the
//! counters every few seconds — so it emits DURING the stall — and distinguishes the two failure
//! modes plus localizes a stuck task body:
//!   * `in_flight = spawned - completed` frozen > 0  => split-read task BODIES are stuck (never
//!     complete). The phase counters say where: stuck before `filter_done` = filter/pruning read;
//!     stuck between `filter_done` and `project_done` = projection read+decode.
//!   * `backlog = completed - yielded` frozen > 0     => tasks complete but the `buffer_unordered`
//!     consumer is never drained => a drain-side lost wake.
//!
//! PER-SCAN (cold-stall): the aggregate globals below sum ALL concurrently-running scans (every
//! table's promotion + query scans), so during a freeze they keep growing (other scans are healthy)
//! and cannot isolate the ONE stuck scan. The per-scan registry (`ScanCounters` / `register_scan`)
//! gives each `LazyScanStream` its own counters; the dumper reports each live scan separately, so
//! the frozen scan is the one whose per-scan counters stop advancing. For that scan:
//!   * `in_flight > 0` frozen  => Mode Y: its split-read task bodies are stuck (never complete).
//!   * `in_flight == 0` && `backlog > 0` frozen => Mode X: tasks completed but the drain wake was
//!     lost (genuine lost-wakeup at the `buffer_unordered`/`FuturesUnordered` -> consumer edge).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use parking_lot::Mutex;

/// Split-read tasks handed to `buffer_unordered` (incremented as each is spawned).
pub(crate) static SPAWNED: AtomicU64 = AtomicU64::new(0);
/// Split-read task bodies that finished (about to send their result on the join channel).
pub(crate) static COMPLETED: AtomicU64 = AtomicU64::new(0);
/// Results drained downstream by the consumer (incremented in the scan stream's terminal map).
pub(crate) static YIELDED: AtomicU64 = AtomicU64::new(0);
/// Task bodies that entered execution.
pub(crate) static TASK_STARTED: AtomicU64 = AtomicU64::new(0);
/// Task bodies past the filter/pruning await.
pub(crate) static TASK_FILTER_DONE: AtomicU64 = AtomicU64::new(0);
/// Task bodies past the projection (read+decode) await.
pub(crate) static TASK_PROJECT_DONE: AtomicU64 = AtomicU64::new(0);

/// DIAG (cold-stall): per-scan counters. One instance per `LazyScanStream`, held by the scan's
/// stream closures and pruned from the registry (via `Weak`) once the stream is dropped. Isolates
/// the single frozen scan from other concurrent scans to resolve Mode X (drain lost-wake) vs Mode Y
/// (stuck task bodies) for THAT scan specifically.
pub(crate) struct ScanCounters {
    pub(crate) id: u64,
    pub(crate) spawned: AtomicU64,
    pub(crate) completed: AtomicU64,
    pub(crate) yielded: AtomicU64,
}

static SCAN_ID: AtomicU64 = AtomicU64::new(0);
static SCAN_REGISTRY: LazyLock<Mutex<Vec<Weak<ScanCounters>>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// Register a new per-scan counter set. The returned `Arc` is held by the scan stream; when the
/// stream is dropped the `Arc` drops and the dumper prunes the now-dead `Weak`.
pub(crate) fn register_scan() -> Arc<ScanCounters> {
    let id = SCAN_ID.fetch_add(1, Ordering::Relaxed);
    let counters = Arc::new(ScanCounters {
        id,
        spawned: AtomicU64::new(0),
        completed: AtomicU64::new(0),
        yielded: AtomicU64::new(0),
    });
    SCAN_REGISTRY.lock().push(Arc::downgrade(&counters));
    counters
}

/// Per-scan sampler state: last `(spawned, completed, yielded)` snapshot + consecutive frozen ticks.
#[derive(Default, Clone, Copy)]
struct PerScanPrev {
    last: (u64, u64, u64),
    stalled_ticks: u64,
    seen: bool,
}

/// Sample every live per-scan counter set, pruning dead ones, and warn for any scan that is frozen
/// with outstanding work (the stuck scan). Reports `in_flight`/`backlog` per scan so Mode X vs Y is
/// read directly off the FROZEN scan rather than the noisy global aggregate.
fn sample_per_scan(prev: &mut HashMap<u64, PerScanPrev>) {
    // Prune dead scans from the registry and collect live ones.
    let live: Vec<Arc<ScanCounters>> = {
        let mut guard = SCAN_REGISTRY.lock();
        guard.retain(|w| w.strong_count() > 0);
        guard.iter().filter_map(Weak::upgrade).collect()
    };
    for entry in prev.values_mut() {
        entry.seen = false;
    }
    for sc in &live {
        let spawned = sc.spawned.load(Ordering::Relaxed);
        let completed = sc.completed.load(Ordering::Relaxed);
        let yielded = sc.yielded.load(Ordering::Relaxed);
        let in_flight = spawned.saturating_sub(completed);
        let backlog = completed.saturating_sub(yielded);
        let cur = (spawned, completed, yielded);
        let outstanding = in_flight > 0 || backlog > 0;
        let e = prev.entry(sc.id).or_default();
        e.seen = true;
        if cur == e.last && outstanding {
            e.stalled_ticks += 1;
            let mode = if in_flight > 0 {
                "Mode-Y stuck task bodies (reads never complete)"
            } else {
                "Mode-X drain lost-wake (tasks done, consumer never re-woken)"
            };
            tracing::warn!(
                target: "vortex::scanpark",
                scan_id = sc.id,
                stalled_ticks = e.stalled_ticks,
                spawned,
                completed,
                yielded,
                in_flight,
                backlog,
                mode,
                "PER-SCAN STALLED: this scan's counters frozen with outstanding work"
            );
        } else if cur != e.last {
            e.stalled_ticks = 0;
        }
        e.last = cur;
    }
    // Forget scans that are no longer live so the map doesn't grow unbounded.
    prev.retain(|_, e| e.seen);
}

/// Start the sampler thread exactly once. Cheap no-op on every call after the first.
pub(crate) fn ensure_dumper() {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("vortex-scanpark-dump".to_string())
        .spawn(|| {
            let mut last = (0u64, 0u64, 0u64);
            let mut stalled_ticks = 0u64;
            let mut per_scan: HashMap<u64, PerScanPrev> = HashMap::new();
            loop {
                std::thread::sleep(Duration::from_secs(5));

                // Per-scan pass first: isolates the single frozen scan (Mode X vs Y) from the
                // noisy global aggregate below.
                sample_per_scan(&mut per_scan);

                let spawned = SPAWNED.load(Ordering::Relaxed);
                let completed = COMPLETED.load(Ordering::Relaxed);
                let yielded = YIELDED.load(Ordering::Relaxed);
                let started = TASK_STARTED.load(Ordering::Relaxed);
                let filter_done = TASK_FILTER_DONE.load(Ordering::Relaxed);
                let project_done = TASK_PROJECT_DONE.load(Ordering::Relaxed);
                let in_flight = spawned.saturating_sub(completed);
                let backlog = completed.saturating_sub(yielded);
                let cur = (spawned, completed, yielded);
                let outstanding = in_flight > 0 || backlog > 0;

                if cur == last && outstanding {
                    stalled_ticks += 1;
                    // Frozen with outstanding work = the stall. `in_flight` vs `backlog`
                    // discriminates stuck-bodies vs drain-lost-wake; the phase gaps
                    // (started/filter_done/project_done) localize a stuck body.
                    // NOTE: this AGGREGATE only stays "frozen" if EVERY scan is frozen; with
                    // concurrent healthy scans the per-scan pass above is authoritative.
                    tracing::warn!(
                        target: "vortex::scanpark",
                        stalled_ticks,
                        spawned,
                        completed,
                        yielded,
                        in_flight,
                        backlog,
                        task_started = started,
                        task_filter_done = filter_done,
                        task_project_done = project_done,
                        stuck_before_filter = started.saturating_sub(filter_done),
                        stuck_before_project = filter_done.saturating_sub(project_done),
                        "SCAN STALLED: counters frozen with outstanding work (aggregate)"
                    );
                } else {
                    if cur != last {
                        stalled_ticks = 0;
                    }
                    if outstanding {
                        tracing::info!(
                            target: "vortex::scanpark",
                            spawned,
                            completed,
                            yielded,
                            in_flight,
                            backlog,
                            "scan progressing (aggregate)"
                        );
                    }
                }
                last = cur;
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(target: "vortex::scanpark", %error, "failed to start scan-park dumper thread");
    }
}
