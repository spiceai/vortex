// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Investigation: does the segment cache account for what a cached segment really retains?
//!
//! [`MokaSegmentCache`]'s weigher charges `ByteBuffer::len()`. That is the length of the *slice*,
//! but a segment resolved through the coalescing read path is a zero-copy slice of the buffer the
//! physical read produced (`CoalescedRequest::resolve` -> `base.slice(start..end)`), and a
//! `Buffer`'s backing `Bytes` keeps that entire allocation alive for as long as any slice of it
//! lives.
//!
//! So the bytes a cached segment retains are not its own length but the length of the whole
//! coalesced read window it was cut from - up to `CoalesceConfig::max_size`, 4MB for a local file
//! and 16MB for object storage. Two consequences the tests below measure:
//!
//! 1. A window is retained in full while *any one* slice of it survives, so eviction cannot
//!    release it. A cache squeezed below the size of its windows overshoots its budget and cannot
//!    evict its way back down.
//! 2. Gap bytes inside a window - the parts no request asked for, up to `CoalesceConfig::distance`
//!    between neighbours - are retained and charged to nobody.
//!
//! Real retained heap is measured with a counting allocator. See the module docs on
//! `footer_size_accounting.rs` for why the measurement is serialised and synchronous.

#![expect(
    clippy::disallowed_types,
    reason = "std locks park without allocating; parking_lot would perturb the counting allocator"
)]
#![expect(
    clippy::tests_outside_test_module,
    reason = "an integration test binary is entirely test code"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "measured byte counts are far inside every cast's range"
)]

use std::alloc::GlobalAlloc;
use std::alloc::Layout as AllocLayout;
use std::alloc::System;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::sync::RwLock;
use std::sync::RwLockWriteGuard;
use std::sync::atomic::AtomicIsize;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::FieldNames;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::memory::MemorySessionExt;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::CoalesceConfig;
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::segments::MokaSegmentCache;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentId;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

struct Counting;

static LIVE: AtomicIsize = AtomicIsize::new(0);

/// [`LIVE`] is process-wide, so no test in this binary may allocate while another is measuring.
static MEASURE: RwLock<()> = RwLock::new(());

fn measuring() -> RwLockWriteGuard<'static, ()> {
    MEASURE.write().unwrap_or_else(PoisonError::into_inner)
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: AllocLayout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(
            new_size as isize - layout.size() as isize,
            Ordering::Relaxed,
        );
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: AllocLayout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn live() -> isize {
    LIVE.load(Ordering::Relaxed)
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    session
});

/// A [`SegmentCache`] that records the buffer handed to every `put` without storing anything, so
/// the recorded slices can be replayed into a real cache under controlled conditions.
#[derive(Default)]
struct Recording {
    puts: Mutex<Vec<(SegmentId, ByteBuffer)>>,
}

#[async_trait]
impl SegmentCache for Recording {
    async fn get(&self, _id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(None)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.puts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((id, buffer));
        Ok(())
    }
}

/// The two admission strategies under comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Admission {
    /// `MokaSegmentCache::new` - copy the segment into its own allocation.
    Compacting,
    /// `MokaSegmentCache::new_sharing_windows` - store the slice as handed over, retaining the
    /// coalesced read window behind it. The pre-fix behaviour.
    SharingWindows,
}

impl Admission {
    fn cache(self, budget: u64) -> MokaSegmentCache {
        match self {
            Self::Compacting => MokaSegmentCache::new(budget),
            Self::SharingWindows => MokaSegmentCache::new_sharing_windows(budget),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Compacting => "compacting    ",
            Self::SharingWindows => "sharing-window",
        }
    }
}

/// A [`VortexReadAt`] that records the length of every physical read, i.e. the size of each
/// coalesced window a cached slice can pin.
struct RecordingReadAt<R> {
    inner: R,
    reads: Mutex<Vec<usize>>,
}

impl<R: VortexReadAt> VortexReadAt for RecordingReadAt<R> {
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.inner.coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.inner.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.inner.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.reads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(length);
        self.inner.read_at(offset, length, alignment)
    }
}

/// Heap moka retains for its own bookkeeping beyond the entries themselves, after a run of
/// inserts. Grows with insert activity, not with the budget; measured at 45-50KB for the 40-insert
/// runs below.
const MOKA_ACTIVITY_SLACK: isize = 128 << 10;

const COLUMNS: usize = 40;
const CHUNKS: usize = 8;
const ROWS: usize = 4096;

fn chunk(seed: u64) -> ArrayRef {
    let column = |c: usize| -> ArrayRef {
        PrimitiveArray::from_iter((0..ROWS).map(|i| (i as i64) * 31 + c as i64 + seed as i64))
            .into_array()
    };
    StructArray::new(
        FieldNames::from_iter((0..COLUMNS).map(|c| format!("column_number_{c}"))),
        (0..COLUMNS).map(column).collect::<Vec<_>>(),
        ROWS,
        Validity::NonNullable,
    )
    .into_array()
}

async fn write_file(path: &std::path::Path) -> VortexResult<()> {
    let chunks: Vec<ArrayRef> = (0..CHUNKS as u64).map(chunk).collect();
    let dtype = chunks[0].dtype().clone();
    let stream = ArrayStreamAdapter::new(dtype, futures::stream::iter(chunks.into_iter().map(Ok)));

    let mut bytes = Vec::new();
    SESSION.write_options().write(&mut bytes, stream).await?;
    std::fs::write(path, &bytes).map_err(|e| vortex_err!("failed to write file: {e}"))?;
    Ok(())
}

/// Scan `columns` from the file at `path` through `cache`, dropping everything the scan produced.
async fn scan_through_cache(
    path: &std::path::Path,
    cache: Arc<dyn SegmentCache>,
    columns: &[&str],
) -> VortexResult<()> {
    let file = SESSION
        .open_options()
        .with_segment_cache(cache)
        .open_path(path)
        .await?;

    let array = file
        .scan()?
        .with_projection(select(columns.to_vec(), root()))
        .into_array_stream()?
        .read_all()
        .await?;

    drop(array);
    drop(file);
    Ok(())
}

fn runtime() -> VortexResult<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| vortex_err!("failed to build runtime: {e}"))
}

/// What one projected scan of `path` offered the segment cache, and the physical reads those
/// offers were sliced out of.
struct Offered {
    /// Every `(id, buffer)` the scan put to the cache, in order.
    puts: Vec<(SegmentId, ByteBuffer)>,
    /// The length of every physical read the scan performed.
    reads: Vec<usize>,
}

fn scan_and_record(path: &std::path::Path, columns: &[&str]) -> VortexResult<Offered> {
    let reader = Arc::new(RecordingReadAt {
        inner: vortex_io::std_file::FileReadAt::open_with_allocator(
            path,
            SESSION.handle(),
            SESSION.allocator(),
        )?,
        reads: Mutex::default(),
    });
    let recording = Arc::new(Recording::default());

    runtime()?.block_on({
        let reader = Arc::clone(&reader);
        let recording = Arc::clone(&recording);
        async move {
            let file = SESSION
                .open_options()
                .with_segment_cache(recording as Arc<dyn SegmentCache>)
                .open(reader as Arc<dyn VortexReadAt>)
                .await?;
            let array = file
                .scan()?
                .with_projection(select(columns.to_vec(), root()))
                .into_array_stream()?
                .read_all()
                .await?;
            drop(array);
            drop(file);
            VortexResult::Ok(())
        }
    })?;

    let puts = std::mem::take(
        &mut *recording
            .puts
            .lock()
            .unwrap_or_else(PoisonError::into_inner),
    );
    let reads = std::mem::take(&mut *reader.reads.lock().unwrap_or_else(PoisonError::into_inner));
    Ok(Offered { puts, reads })
}

/// What the cache thinks it holds, and what it actually retains.
struct Accounting {
    /// Moka's own weighted size after eviction has settled: what the cache believes it holds.
    weighted: u64,
    entries: u64,
    /// Heap released when the cache is dropped: buffers, map entries and moka's own tables.
    retained_total: isize,
    /// The same, less the heap an empty cache of the same budget retains, so this is the entries
    /// alone without moka's fixed structural cost.
    retained_entries: isize,
}

/// Heap released by dropping a settled cache holding `puts`. Takes ownership so that the cache is
/// the only remaining holder of the slices before the measurement.
fn retained_by_cache(
    puts: Vec<(SegmentId, ByteBuffer)>,
    budget: u64,
    admission: Admission,
) -> VortexResult<(isize, u64, u64)> {
    let cache = admission.cache(budget);

    let rt = runtime()?;
    rt.block_on(async {
        for (id, buffer) in puts {
            cache.put(id, buffer).await?;
        }
        // Moka applies queued writes and evictions in bounded batches, so a single
        // `run_pending_tasks` leaves evicted values still queued and alive - measured at 2.86MB
        // still held after one call where six calls settled to 266KB. Drain until the heap stops
        // falling, then confirm with a couple of no-change iterations.
        let mut stable = 0;
        let mut previous = live();
        for _ in 0..64 {
            cache.run_pending_tasks().await;
            let current = live();
            if current >= previous {
                stable += 1;
                if stable == 3 {
                    break;
                }
            } else {
                stable = 0;
            }
            previous = current;
        }
        VortexResult::Ok(())
    })?;

    let weighted = cache.weighted_size();
    let entries = cache.entry_count();

    let before = live();
    drop(cache);
    drop(rt);
    Ok((before - live(), weighted, entries))
}

/// Replay `puts` into a cache with `budget` under `admission`, settle eviction, then measure.
fn account(
    puts: Vec<(SegmentId, ByteBuffer)>,
    budget: u64,
    admission: Admission,
) -> VortexResult<Accounting> {
    let (empty, ..) = retained_by_cache(Vec::new(), budget, admission)?;
    let (retained_total, weighted, entries) = retained_by_cache(puts, budget, admission)?;

    Ok(Accounting {
        weighted,
        entries,
        retained_total,
        retained_entries: retained_total - empty,
    })
}

fn column_names() -> Vec<String> {
    (0..COLUMNS).map(|c| format!("column_number_{c}")).collect()
}

fn report(admission: Admission, budget: u64, a: &Accounting) {
    println!(
        "  {} budget {:>9} | holds {:>3} entries weighing {:>9} B | retains {:>9} B total, \
         {:>9} B in entries | {:>6.2}x budget",
        admission.label(),
        budget,
        a.entries,
        a.weighted,
        a.retained_total,
        a.retained_entries,
        a.retained_entries as f64 / budget as f64,
    );
}

/// The core property the copy restores: a byte budget bounds the memory the cache retains.
///
/// Sharing windows, a cached slice pins the whole coalesced read window it was cut from, so the
/// retained heap is set by the read pattern rather than the budget and eviction cannot release it.
/// Compacting, every entry owns its bytes, so retained heap tracks the budget.
#[test]
fn compacting_admission_keeps_retained_heap_within_budget() -> VortexResult<()> {
    let _m = measuring();

    let dir = tempfile::tempdir().map_err(|e| vortex_err!("tempdir: {e}"))?;
    let path = dir.path().join("eviction.vortex");
    runtime()?.block_on(write_file(&path))?;
    let file_size = std::fs::metadata(&path)
        .map_err(|e| vortex_err!("metadata: {e}"))?
        .len();

    let names = column_names();
    let all: Vec<&str> = names.iter().map(String::as_str).collect();

    {
        // One scan purely to describe the physical reads the cached slices are cut from.
        let offered = scan_and_record(&path, &all)?;
        println!(
            "file {file_size} B, {} segments offered, {} physical reads (largest {} B, total {} B)",
            offered.puts.len(),
            offered.reads.len(),
            offered.reads.iter().max().copied().unwrap_or(0),
            offered.reads.iter().sum::<usize>(),
        );
    }

    // The last budget exceeds the file, so nothing evicts and the ratio isolates the accounting
    // from the eviction behaviour.
    for budget in [256u64 << 10, 512 << 10, 1 << 20, 8 << 20] {
        for admission in [Admission::SharingWindows, Admission::Compacting] {
            // A fresh scan per measurement: recorded slices must not outlive their own measurement.
            let offered = scan_and_record(&path, &all)?;
            let offered_count = offered.puts.len() as u64;
            let a = account(offered.puts, budget, admission)?;
            report(admission, budget, &a);
            let evicted = a.entries < offered_count;

            match admission {
                Admission::Compacting => {
                    // The cache's self-report must be truthful: retained heap is its weighted
                    // size plus moka's own bookkeeping, which grows with insert *activity* rather
                    // than with the budget (measured at 45-50KB for these 40-insert runs).
                    assert!(
                        a.retained_entries <= a.weighted as isize + MOKA_ACTIVITY_SLACK,
                        "charged {} B but retains {} B, beyond the {MOKA_ACTIVITY_SLACK} B slack",
                        a.weighted,
                        a.retained_entries,
                    );
                    // Moka keeps weighted size under the budget, so the above bounds real memory.
                    assert!(a.weighted <= budget, "moka exceeded its own capacity");
                    // And the charge must not be so pessimistic that the budget under-fills.
                    assert!(
                        a.retained_entries as f64 >= a.weighted as f64 * 0.8,
                        "charged {} B but only retains {} B: the weight is too pessimistic",
                        a.weighted,
                        a.retained_entries,
                    );
                }
                // The pre-fix behaviour, asserted so this test fails if the comparison arm ever
                // stops demonstrating the bug. Only overshoots once something has been evicted:
                // with every slice of a window cached, the window is fully charged and the
                // accounting happens to be right - which is why a full scan into a generous cache
                // never exposed this.
                Admission::SharingWindows if evicted => {
                    assert!(
                        a.retained_entries > a.weighted as isize * 2,
                        "sharing windows retained {} B against a charge of {} B: expected the \
                         pinned read window to dominate",
                        a.retained_entries,
                        a.weighted,
                    );
                }
                Admission::SharingWindows => {}
            }
        }
    }

    Ok(())
}

/// Gap accounting: a projection whose columns sit within the coalescing distance of each other,
/// but not adjacent, triggers one physical read spanning the columns in between. Sharing windows,
/// the cache retains those gap bytes for free; compacting, it does not retain them at all.
#[test]
fn compacting_admission_does_not_retain_unrequested_gap_bytes() -> VortexResult<()> {
    let _m = measuring();

    let dir = tempfile::tempdir().map_err(|e| vortex_err!("tempdir: {e}"))?;
    let path = dir.path().join("projection.vortex");
    runtime()?.block_on(write_file(&path))?;

    // Each column is one segment of roughly 70KB, so these projections leave gaps of a few
    // hundred KB - inside the 1MB `CoalesceConfig::file` distance, so they coalesce.
    for columns in [
        vec!["column_number_7"],
        vec!["column_number_7", "column_number_12"],
        vec!["column_number_7", "column_number_12", "column_number_17"],
        vec!["column_number_0", "column_number_20", "column_number_39"],
    ] {
        let descriptor = {
            let offered = scan_and_record(&path, &columns)?;
            format!(
                "{} column(s), {} segments totalling {} B, {} physical reads totalling {} B \
                 (largest {} B)",
                columns.len(),
                offered.puts.len(),
                offered.puts.iter().map(|(_, b)| b.len()).sum::<usize>(),
                offered.reads.len(),
                offered.reads.iter().sum::<usize>(),
                offered.reads.iter().max().copied().unwrap_or(0),
            )
        };
        println!("{descriptor}");

        // A budget large enough that nothing is evicted, so any overshoot is gap bytes alone.
        for admission in [Admission::SharingWindows, Admission::Compacting] {
            let offered = scan_and_record(&path, &columns)?;
            let wanted: usize = offered.puts.iter().map(|(_, b)| b.len()).sum();
            let a = account(offered.puts, 1 << 30, admission)?;
            println!(
                "  {} retains {:>9} B in entries for {:>9} B of wanted segment bytes ({:.2}x)",
                admission.label(),
                a.retained_entries,
                wanted,
                a.retained_entries as f64 / wanted as f64,
            );

            if admission == Admission::Compacting {
                // Own allocation per entry: len + alignment, plus the entry's own bookkeeping.
                assert!(
                    a.retained_entries < (wanted as isize * 3) / 2,
                    "compacting admission retained {} B for {wanted} B of wanted bytes",
                    a.retained_entries,
                );
            }
        }
    }

    Ok(())
}

/// `MultiFileSession`'s footer cache weighs in KB (`approx_byte_size() / 1024`) against a capacity
/// also expressed in KB. Integer division truncates, so a footer under 1KB weighs zero and
/// occupies none of the budget. This checks how small a real footer gets, i.e. whether the
/// truncation is reachable.
#[test]
fn multi_file_footer_cache_kb_weigher_truncation() -> VortexResult<()> {
    // Not measuring heap, but this binary's counting allocator is shared; take the lock so this
    // test's allocations cannot be charged to a measurement running on another thread.
    let _m = measuring();

    let dir = tempfile::tempdir().map_err(|e| vortex_err!("tempdir: {e}"))?;

    for (label, columns, rows) in [("minimal", 1usize, 1usize), ("small", 4, 128)] {
        let path = dir.path().join(format!("{label}.vortex"));
        let array = StructArray::new(
            FieldNames::from_iter((0..columns).map(|c| format!("c{c}"))),
            (0..columns)
                .map(|c| PrimitiveArray::from_iter((0..rows).map(|i| (i + c) as i64)).into_array())
                .collect::<Vec<_>>(),
            rows,
            Validity::NonNullable,
        )
        .into_array();

        let rt = runtime()?;
        rt.block_on(async {
            let dtype = array.dtype().clone();
            let stream =
                ArrayStreamAdapter::new(dtype, futures::stream::iter([VortexResult::Ok(array)]));
            let mut bytes = Vec::new();
            SESSION.write_options().write(&mut bytes, stream).await?;
            std::fs::write(&path, &bytes).map_err(|e| vortex_err!("write: {e}"))?;
            VortexResult::Ok(())
        })?;

        let footer_size = rt.block_on(async {
            let file = SESSION.open_options().open_path(&path).await?;
            VortexResult::Ok(file.footer().approx_byte_size())
        })?;

        // The weigher in `MultiFileSession::default`.
        let weight = footer_size
            .and_then(|bytes| u32::try_from(bytes / 1024).ok())
            .unwrap_or(10);

        println!(
            "{label} footer ({columns} columns, {rows} rows): approx_byte_size {} B \
             -> MultiFileSession weight {weight} KB",
            footer_size.map_or_else(|| "none".to_string(), |b| b.to_string())
        );
    }

    Ok(())
}

/// Calibration for `MokaSegmentCache`'s per-entry overhead charge.
///
/// A compacted value costs its own allocation (`len + *alignment`) plus fixed per-entry heap: the
/// `Bytes` control block, moka's entry record, and its policy bookkeeping. That fixed cost is
/// invisible in `len()` but dominates once segments are small, so the weigher charges a constant
/// for it. This measures what the constant should be, using independently allocated buffers so no
/// window sharing is involved.
#[test]
fn per_entry_overhead_calibration() -> VortexResult<()> {
    let _m = measuring();

    for segment_len in [64usize, 256, 1024, 8192, 65536] {
        let count = 256;
        // Budget high enough that nothing evicts, so every entry is held.
        let puts: Vec<(SegmentId, ByteBuffer)> = (0..count)
            .map(|i| {
                (
                    SegmentId::from(i as u32),
                    ByteBuffer::copy_from(vec![i as u8; segment_len]),
                )
            })
            .collect();

        let a = account(puts, 1 << 30, Admission::Compacting)?;
        let per_entry = a.retained_entries as f64 / count as f64;
        println!(
            "{count} x {segment_len:>6} B segments: charged {:>9} B, retains {:>9} B in entries \
             -> {per_entry:>8.1} B/entry, overhead {:>6.1} B/entry",
            a.weighted,
            a.retained_entries,
            per_entry - segment_len as f64,
        );
    }

    Ok(())
}

/// A [`SegmentCache`] that counts hits and misses on the way through to a real cache.
struct HitCounting {
    inner: MokaSegmentCache,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

#[async_trait]
impl SegmentCache for HitCounting {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        let result = self.inner.get(id).await?;
        if result.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        Ok(result)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.inner.put(id, buffer).await
    }
}

/// Outcome of populating a cache with one scan and then re-scanning through it.
struct Reuse {
    entries: u64,
    hits: usize,
    misses: usize,
    /// Heap the cache retains once eviction has settled, less an empty cache of the same budget.
    retained: isize,
}

/// Populate a cache with one full scan, re-scan through it, then measure what it retains.
fn scan_twice(path: &std::path::Path, budget: u64, admission: Admission) -> VortexResult<Reuse> {
    let (empty, ..) = retained_by_cache(Vec::new(), budget, admission)?;

    let cache = Arc::new(HitCounting {
        inner: admission.cache(budget),
        hits: AtomicUsize::new(0),
        misses: AtomicUsize::new(0),
    });

    let names = column_names();
    let all: Vec<&str> = names.iter().map(String::as_str).collect();
    let rt = runtime()?;

    // First scan populates; counters are reset so the second scan alone is measured.
    rt.block_on(scan_through_cache(
        path,
        Arc::clone(&cache) as Arc<dyn SegmentCache>,
        &all,
    ))?;
    cache.hits.store(0, Ordering::Relaxed);
    cache.misses.store(0, Ordering::Relaxed);

    rt.block_on(scan_through_cache(
        path,
        Arc::clone(&cache) as Arc<dyn SegmentCache>,
        &all,
    ))?;

    let hits = cache.hits.load(Ordering::Relaxed);
    let misses = cache.misses.load(Ordering::Relaxed);

    rt.block_on(async {
        let mut stable = 0;
        let mut previous = live();
        for _ in 0..64 {
            cache.inner.run_pending_tasks().await;
            let current = live();
            if current >= previous {
                stable += 1;
                if stable == 3 {
                    break;
                }
            } else {
                stable = 0;
            }
            previous = current;
        }
    });

    let entries = cache.inner.entry_count();
    let before = live();
    drop(cache);
    drop(rt);
    let retained = before - live() - empty;

    Ok(Reuse {
        entries,
        hits,
        misses,
        retained,
    })
}

/// The practical payoff, stated in the units an operator actually has: for a given amount of real
/// memory, how much of the file does the cache serve on a re-scan?
///
/// Comparing at equal *budget* understates the difference, because both arms then hold the same
/// number of entries - one of them just uses far more memory to do it. Comparing at equal *memory*
/// is the honest question.
#[test]
fn hit_rate_at_equal_memory() -> VortexResult<()> {
    let _m = measuring();

    let dir = tempfile::tempdir().map_err(|e| vortex_err!("tempdir: {e}"))?;
    let path = dir.path().join("reuse.vortex");
    runtime()?.block_on(write_file(&path))?;

    // Undersized budget: the regime where the two arms diverge.
    const BUDGET: u64 = 256 << 10;
    let sharing = scan_twice(&path, BUDGET, Admission::SharingWindows)?;

    // Give the compacting cache the memory the sharing cache actually consumed.
    let equal_memory = sharing.retained.max(1) as u64;
    let compacting = scan_twice(&path, equal_memory, Admission::Compacting)?;

    for (label, budget, r) in [
        ("sharing-window", BUDGET, &sharing),
        ("compacting    ", equal_memory, &compacting),
    ] {
        let total = r.hits + r.misses;
        println!(
            "  {label} budget {budget:>9} B -> retains {:>9} B | {:>3} entries | \
             re-scan {:>3}/{:<3} segments served ({:>5.1}% hit rate)",
            r.retained,
            r.entries,
            r.hits,
            total,
            100.0 * r.hits as f64 / total as f64,
        );
    }

    // Same memory, strictly more of the file served. The budget handed to the compacting arm is
    // the sharing arm's measured footprint, so it may exceed it by its own bookkeeping.
    //
    // Note the sharing arm's footprint is not reproducible run to run - it depends on which slices
    // happen to survive eviction and therefore on how many distinct read windows stay pinned,
    // which moka's lazy eviction makes timing-dependent. Measured between 2.3MB and 5.4MB for this
    // file against the same 256KB budget. That nondeterminism is itself part of the problem.
    assert!(
        compacting.retained <= sharing.retained + MOKA_ACTIVITY_SLACK,
        "compacting used {} B against the {} B the sharing arm was measured at",
        compacting.retained,
        sharing.retained,
    );
    assert!(
        compacting.hits > sharing.hits,
        "for the same memory, compacting served {} segments vs {}",
        compacting.hits,
        sharing.hits,
    );

    Ok(())
}
