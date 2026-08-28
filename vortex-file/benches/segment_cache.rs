// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cost of the two [`MokaSegmentCache`] admission strategies.
//!
//! `MokaSegmentCache::new` copies each admitted segment into its own allocation so the cache
//! retains what it charges itself for; `new_sharing_windows` stores the slice as handed over,
//! retaining the whole coalesced read window behind it. The copy buys a real memory bound (see
//! `vortex-file/tests/segment_cache_size_accounting.rs`) and this measures what it costs:
//!
//! * `admission` - the `put` itself, over segment sizes, with no I/O involved.
//! * `scan_cold` - a full scan that populates an empty cache, so every segment pays admission.
//! * `scan_warm` - a full scan served entirely from a populated cache; admission is not on this
//!   path, so the two arms should be indistinguishable.
//! * `scan_bounded` - a full scan against a budget too small to hold the file, the case where the
//!   two arms differ in retained memory. Both do the same I/O; this checks the copy does not make
//!   the churn more expensive.

#![allow(clippy::unwrap_used, reason = "benchmark")]
#![allow(
    clippy::cast_possible_truncation,
    reason = "benchmark fixtures are sized well inside every cast's range"
)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_file::Footer;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::segments::MokaSegmentCache;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentId;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    session
});

const COLUMNS: usize = 40;
const ROWS: usize = 4096;

/// File sizes to scan. The small file's segments stay resident in cache during a scan, so its
/// copies are near-free; the large one pushes them out to DRAM, which is the honest worst case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scale {
    /// ~2.8MB, 40 segments.
    Small,
    /// ~45MB, 640 segments.
    Large,
}

const SCALES: &[Scale] = &[Scale::Small, Scale::Large];

impl Scale {
    fn chunks(self) -> usize {
        match self {
            Self::Small => 8,
            Self::Large => 128,
        }
    }
}

/// Admission strategy under test.
#[derive(Debug, Clone, Copy)]
enum Arm {
    Compacting,
    SharingWindows,
}

const ARMS: &[Arm] = &[Arm::Compacting, Arm::SharingWindows];

impl Arm {
    fn cache(self, budget: u64) -> Arc<dyn SegmentCache> {
        match self {
            Self::Compacting => Arc::new(MokaSegmentCache::new(budget)),
            Self::SharingWindows => Arc::new(MokaSegmentCache::new_sharing_windows(budget)),
        }
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

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

/// A file the scan benchmarks read, written once to a temp dir that outlives the run.
struct Fixture {
    _dir: tempfile::TempDir,
    path: std::path::PathBuf,
    footer: Footer,
    file_size: u64,
}

fn build_fixture(scale: Scale) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.vortex");

    let rt = runtime();
    rt.block_on(async {
        let chunks: Vec<ArrayRef> = (0..scale.chunks() as u64).map(chunk).collect();
        let dtype = chunks[0].dtype().clone();
        let stream =
            ArrayStreamAdapter::new(dtype, futures::stream::iter(chunks.into_iter().map(Ok)));
        let mut bytes = Vec::new();
        SESSION
            .write_options()
            .write(&mut bytes, stream)
            .await
            .unwrap();
        std::fs::write(&path, &bytes).unwrap();
    });

    // Reuse the parsed footer across iterations so the measurement is the segment path, not
    // footer parsing.
    let footer = rt.block_on(async {
        SESSION
            .open_options()
            .open_path(&path)
            .await
            .unwrap()
            .footer()
            .clone()
    });
    let file_size = std::fs::metadata(&path).unwrap().len();

    Fixture {
        _dir: dir,
        path,
        footer,
        file_size,
    }
}

static SMALL: LazyLock<Fixture> = LazyLock::new(|| build_fixture(Scale::Small));
static LARGE: LazyLock<Fixture> = LazyLock::new(|| build_fixture(Scale::Large));

/// Build a fixture before any benchmark enters `block_on`. Its initialiser drives async work on
/// its own runtime, which panics if a runtime is already current on this thread.
fn fixture(scale: Scale) -> &'static Fixture {
    match scale {
        Scale::Small => LazyLock::force(&SMALL),
        Scale::Large => LazyLock::force(&LARGE),
    }
}

/// Scan the whole fixture file through `cache`.
async fn scan(fixture: &'static Fixture, cache: Arc<dyn SegmentCache>) {
    let file = SESSION
        .open_options()
        .with_footer(fixture.footer.clone())
        .with_segment_cache(cache)
        .open_path(&fixture.path)
        .await
        .unwrap();

    let array = file
        .scan()
        .unwrap()
        .into_array_stream()
        .unwrap()
        .read_all()
        .await
        .unwrap();
    divan::black_box(array);
}

/// Budget comfortably larger than either file, so nothing is evicted.
const GENEROUS: u64 = 256 << 20;
/// Budget far below either file, so the cache churns.
const BOUNDED: u64 = 256 << 10;

fn scale_of(label: &str) -> Scale {
    SCALES
        .iter()
        .copied()
        .find(|s| format!("{s:?}").to_lowercase() == label)
        .unwrap()
}

/// Admission cost alone: segments sliced from one shared window, as the read path produces them.
#[divan::bench(args = [1024usize, 8192, 65536, 262144], consts = [0, 1])]
fn admission<const ARM: usize>(bencher: Bencher, segment_len: usize) {
    let arm = ARMS[ARM];
    // One window, sliced per segment - exactly what `CoalescedRequest::resolve` hands the cache.
    let count = 64;
    // Non-zero, non-uniform bytes: a zero-filled source lets the allocator and the copy short-cut
    // work that a real segment would not.
    let window = ByteBuffer::from(
        (0..segment_len * count)
            .map(|i| (i * 31 + 7) as u8)
            .collect::<Vec<u8>>(),
    );
    let slices: Vec<ByteBuffer> = (0..count)
        .map(|i| window.slice(i * segment_len..(i + 1) * segment_len))
        .collect();

    let rt = runtime();
    bencher
        .with_inputs(|| arm.cache(GENEROUS))
        .bench_values(|cache| {
            rt.block_on(async {
                for (i, slice) in slices.iter().enumerate() {
                    cache
                        .put(SegmentId::from(i as u32), slice.clone())
                        .await
                        .unwrap();
                }
            });
            cache
        });
}

/// A scan that populates an empty cache: every segment pays admission.
#[divan::bench(consts = [0, 1], args = ["small", "large"], sample_count = 30)]
fn scan_cold<const ARM: usize>(bencher: Bencher, scale: &str) {
    let arm = ARMS[ARM];
    let fixture = fixture(scale_of(scale));
    let rt = runtime();
    bencher
        .with_inputs(|| arm.cache(GENEROUS))
        .bench_values(|cache| rt.block_on(scan(fixture, cache)));
}

/// A scan served entirely from a populated cache: admission is not on this path.
#[divan::bench(consts = [0, 1], args = ["small", "large"], sample_count = 30)]
fn scan_warm<const ARM: usize>(bencher: Bencher, scale: &str) {
    let arm = ARMS[ARM];
    let fixture = fixture(scale_of(scale));
    let rt = runtime();
    let cache = arm.cache(GENEROUS);
    rt.block_on(scan(fixture, Arc::clone(&cache)));

    bencher.bench(|| rt.block_on(scan(fixture, Arc::clone(&cache))));
}

/// A scan against a budget too small for the file, so the cache admits and evicts throughout -
/// the regime where the two arms differ in retained memory.
#[divan::bench(consts = [0, 1], args = ["small", "large"], sample_count = 30)]
fn scan_bounded<const ARM: usize>(bencher: Bencher, scale: &str) {
    let arm = ARMS[ARM];
    let fixture = fixture(scale_of(scale));
    let rt = runtime();
    let cache = arm.cache(BOUNDED);
    rt.block_on(scan(fixture, Arc::clone(&cache)));

    bencher.bench(|| rt.block_on(scan(fixture, Arc::clone(&cache))));
}

/// No cache at all, as a baseline for what the cache is buying in the first place.
#[divan::bench(args = ["small", "large"], sample_count = 30)]
fn scan_uncached(bencher: Bencher, scale: &str) {
    let fixture = fixture(scale_of(scale));
    let rt = runtime();
    bencher.bench(|| {
        rt.block_on(async {
            let file = SESSION
                .open_options()
                .with_footer(fixture.footer.clone())
                .open_path(&fixture.path)
                .await
                .unwrap();
            let array = file
                .scan()
                .unwrap()
                .into_array_stream()
                .unwrap()
                .read_all()
                .await
                .unwrap();
            divan::black_box(array);
        })
    });
}

/// Print the fixture sizes once so the timings above have context.
#[divan::bench(args = ["small", "large"], sample_count = 1)]
fn describe(bencher: Bencher, scale: &str) {
    let fixture = fixture(scale_of(scale));
    println!(
        "\n  {scale}: {} B on disk, {} segments",
        fixture.file_size,
        fixture.footer.segment_map().len(),
    );
    bencher.bench(|| ());
}
