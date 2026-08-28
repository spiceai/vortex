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
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::future::BoxFuture;
use moka::future::Cache;
use moka::future::CacheBuilder;
use moka::policy::EvictionPolicy;
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
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::CoalesceConfig;
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
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

/// The cache under investigation, rebuilt here so that moka's own view of what it holds -
/// `weighted_size` and `entry_count`, and `run_pending_tasks` to settle eviction - is observable.
/// Weigher, capacity units and eviction policy are copied from `MokaSegmentCache::new`; only the
/// hasher differs, which does not affect weighing or eviction.
fn mirror_of_moka_segment_cache(max_capacity_bytes: u64) -> Cache<SegmentId, ByteBuffer> {
    CacheBuilder::new(max_capacity_bytes)
        .weigher(|_, buffer: &ByteBuffer| {
            u32::try_from(buffer.len().min(u32::MAX as usize)).vortex_expect("must fit")
        })
        .eviction_policy(EvictionPolicy::tiny_lfu())
        .build()
}

/// A [`SegmentCache`] that records the buffer handed to every `put` without storing anything, so
/// the recorded slices can be replayed into a cache under controlled conditions.
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
    /// `sum(buffer.len())` over every admitted segment: what the weigher charged.
    charged: usize,
    /// Moka's own weighted size after eviction has settled.
    weighted: u64,
    entries: u64,
    /// Heap released when the cache is dropped, less moka's own structural overhead: the segment
    /// bytes the cache was really retaining.
    retained: isize,
}

/// Heap released by dropping a settled cache holding `puts`. Takes ownership so that the cache is
/// the only remaining holder of the slices before the measurement.
fn retained_by_cache(
    puts: Vec<(SegmentId, ByteBuffer)>,
    budget: u64,
) -> VortexResult<(isize, u64, u64)> {
    let cache = mirror_of_moka_segment_cache(budget);

    let rt = runtime()?;
    rt.block_on(async {
        for (id, buffer) in puts {
            cache.insert(id, buffer).await;
        }
        cache.run_pending_tasks().await;
    });

    let weighted = cache.weighted_size();
    let entries = cache.entry_count();

    let before = live();
    drop(cache);
    drop(rt);
    Ok((before - live(), weighted, entries))
}

/// Replay `puts` into a cache with `budget`, settle eviction, then measure. Moka's structural
/// overhead is measured against an empty cache of the same budget and subtracted, so `retained`
/// is segment bytes only.
fn account(puts: Vec<(SegmentId, ByteBuffer)>, budget: u64) -> VortexResult<Accounting> {
    let charged: usize = puts.iter().map(|(_, buffer)| buffer.len()).sum();
    let (overhead, ..) = retained_by_cache(Vec::new(), budget)?;
    let (total, weighted, entries) = retained_by_cache(puts, budget)?;

    Ok(Accounting {
        charged,
        weighted,
        entries,
        retained: total - overhead,
    })
}

fn column_names() -> Vec<String> {
    (0..COLUMNS).map(|c| format!("column_number_{c}")).collect()
}

fn report(label: &str, budget: u64, a: &Accounting) {
    println!(
        "{label}: budget {budget} B | moka holds {} entries weighing {} B | really retains {} B \
         ({:.1}x budget)",
        a.entries,
        a.weighted,
        a.retained,
        a.retained as f64 / budget as f64,
    );
}

/// A cache squeezed below the size of its coalesced read windows evicts down to its budget by its
/// own accounting, yet keeps retaining the windows in full: any one surviving slice pins its whole
/// window, so eviction cannot release the memory.
#[test]
fn evicted_segment_cache_still_retains_whole_coalesced_windows() -> VortexResult<()> {
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
        let mut reads = offered.reads.clone();
        reads.sort_unstable();
        println!(
            "file {file_size} B, {} segments offered to the cache, {} physical reads \
             (largest {} B, total {} B)",
            offered.puts.len(),
            reads.len(),
            reads.last().copied().unwrap_or(0),
            reads.iter().sum::<usize>(),
        );
    }

    for budget in [256u64 << 10, 512 << 10, 1 << 20] {
        // A fresh scan per budget: the recorded slices must not outlive their own measurement.
        let offered = scan_and_record(&path, &all)?;
        let a = account(offered.puts, budget)?;
        report("all columns", budget, &a);
    }

    Ok(())
}

/// Gap accounting: a projection whose columns sit within the coalescing distance of each other,
/// but not adjacent, triggers one physical read spanning the columns in between. The cache is
/// charged for the segments it asked for and retains the gap bytes for free.
#[test]
fn projected_scan_segment_cache_retains_unrequested_gap_bytes() -> VortexResult<()> {
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
        let offered = scan_and_record(&path, &columns)?;
        let read_total: usize = offered.reads.iter().sum();
        let largest_read = offered.reads.iter().max().copied().unwrap_or(0);
        let reads = offered.reads.len();
        let segments = offered.puts.len();

        // A budget large enough that nothing is evicted: the overshoot here is gap bytes alone.
        let a = account(offered.puts, 1 << 30)?;
        println!(
            "{} of {COLUMNS} columns: {segments} segments charged {} B | {reads} physical reads \
             totalling {read_total} B (largest {largest_read} B) | really retains {} B \
             ({:.2}x charged)",
            columns.len(),
            a.charged,
            a.retained,
            a.retained as f64 / a.charged as f64,
        );
        assert_eq!(a.weighted, a.charged as u64, "nothing should have evicted");
        assert_eq!(a.entries, segments as u64);
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
