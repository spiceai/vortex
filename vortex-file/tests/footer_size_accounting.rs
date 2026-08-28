// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Regression tests for [`Footer::approx_byte_size`].
//!
//! Caches bound themselves with that number, so two properties have to hold:
//!
//! 1. It is **stable**. A cache records an entry's size when it admits it and subtracts the same
//!    accessor's value again when it evicts it, so a size that grew in between drives the cache's
//!    accounting negative. A footer's layout tree is built lazily, as scans walk it, into the very
//!    object the cache is holding - so the reported size has to already account for it.
//! 2. It is **close to the truth**. Under-reporting is what makes a bounded cache hold an
//!    unbounded-looking amount of memory; a 50MB budget that really holds 650MB is not a budget.
//!
//! The second test measures real retained heap with a counting allocator. That counter is
//! process-wide, so every test in this binary holds [`MEASURE`] for its whole body: a sibling case
//! allocating on another test thread would otherwise be charged to whichever measurement is in
//! flight. The tests are synchronous for the same reason - each drives its writes on a private
//! current-thread runtime rather than yielding to a shared one.

// `parking_lot`'s fallback path allocates when a lock is contended, which would land inside
// whichever measurement is in flight. The std mutex parks on a futex without allocating.
#![expect(
    clippy::disallowed_types,
    reason = "std Mutex parks without allocating; parking_lot would perturb the counting allocator"
)]

use std::alloc::GlobalAlloc;
use std::alloc::Layout as AllocLayout;
use std::alloc::System;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::AtomicIsize;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::FieldNames;
use vortex_array::stats::PRUNING_STATS;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::Footer;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::LayoutRef;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

/// Tracks live heap bytes so a footer's reported size can be checked against what it really
/// retains.
struct Counting;

static LIVE: AtomicIsize = AtomicIsize::new(0);

/// Serialises this binary's tests. [`LIVE`] is process-wide, so no test here may allocate while
/// another is measuring.
static MEASURE: Mutex<()> = Mutex::new(());

/// Run `f` with no other test in this binary allocating concurrently.
fn measured<T>(f: impl FnOnce() -> T) -> T {
    let _guard = MEASURE.lock().unwrap_or_else(|e| e.into_inner());
    f()
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

/// A schema shape to write and measure.
#[derive(Debug, Clone, Copy)]
struct Shape {
    columns: usize,
    /// Fields per nested struct column, or 0 for flat columns.
    nested: usize,
    chunks: usize,
    rows: usize,
    statistics: bool,
}

fn chunk(shape: Shape, seed: u64) -> ArrayRef {
    let leaf = |field: usize| -> ArrayRef {
        if field % 5 == 4 {
            VarBinViewArray::from_iter_str(
                (0..shape.rows).map(|i| format!("value-{seed}-{}", i % 97)),
            )
            .into_array()
        } else {
            PrimitiveArray::from_iter((0..shape.rows).map(|i| (i as i64) + field as i64))
                .into_array()
        }
    };

    let mut names = Vec::with_capacity(shape.columns);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(shape.columns);
    for c in 0..shape.columns {
        names.push(format!("column_number_{c}"));
        if shape.nested == 0 {
            columns.push(leaf(c));
        } else {
            let inner: Vec<String> = (0..shape.nested).map(|f| format!("field_{f}")).collect();
            columns.push(
                StructArray::new(
                    FieldNames::from_iter(inner.iter().map(String::as_str)),
                    (0..shape.nested).map(leaf).collect::<Vec<_>>(),
                    shape.rows,
                    Validity::NonNullable,
                )
                .into_array(),
            );
        }
    }

    StructArray::new(
        FieldNames::from_iter(names.iter().map(String::as_str)),
        columns,
        shape.rows,
        Validity::NonNullable,
    )
    .into_array()
}

fn write_file(shape: Shape) -> VortexResult<ByteBuffer> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| vortex_err!("failed to build runtime: {e}"))?
        .block_on(write_file_async(shape))
}

async fn write_file_async(shape: Shape) -> VortexResult<ByteBuffer> {
    let chunks: Vec<ArrayRef> = (0..shape.chunks as u64).map(|s| chunk(shape, s)).collect();
    let dtype = chunks[0].dtype().clone();
    let stream = ArrayStreamAdapter::new(dtype, futures::stream::iter(chunks.into_iter().map(Ok)));

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .with_file_statistics(if shape.statistics {
            PRUNING_STATS.to_vec()
        } else {
            vec![]
        })
        .write(&mut bytes, stream)
        .await?;
    Ok(ByteBuffer::from(bytes))
}

/// Materialise every child layout, as scanning the whole file does.
fn materialise(layout: &LayoutRef) -> VortexResult<()> {
    for i in 0..layout.nchildren() {
        materialise(&layout.child(i)?)?;
    }
    Ok(())
}

fn open_footer(file: &ByteBuffer) -> VortexResult<Footer> {
    Ok(SESSION
        .open_options()
        .open_buffer(file.clone())?
        .footer()
        .clone())
}

const FLAT: Shape = Shape {
    columns: 50,
    nested: 0,
    chunks: 8,
    rows: 4096,
    statistics: true,
};
const WIDE: Shape = Shape {
    columns: 200,
    nested: 0,
    chunks: 2,
    rows: 1024,
    statistics: true,
};
const NESTED: Shape = Shape {
    columns: 20,
    nested: 8,
    chunks: 4,
    rows: 1024,
    statistics: true,
};
const NO_STATS: Shape = Shape {
    columns: 50,
    nested: 0,
    chunks: 8,
    rows: 4096,
    statistics: false,
};
const TINY: Shape = Shape {
    columns: 2,
    nested: 0,
    chunks: 2,
    rows: 512,
    statistics: true,
};

/// A cached footer's reported size must not change when a scan materialises its layout tree.
///
/// DataFusion's `DefaultFilesMetadataCache` subtracts `memory_size()` at eviction, having added it
/// at admission; if the value grew in between, its `memory_used` underflows.
#[rstest]
#[case::flat(FLAT)]
#[case::wide(WIDE)]
#[case::nested(NESTED)]
#[case::no_stats(NO_STATS)]
#[case::tiny(TINY)]
fn approx_byte_size_is_stable_across_scans(#[case] shape: Shape) -> VortexResult<()> {
    measured(|| {
        let file = write_file(shape)?;
        let footer = open_footer(&file)?;

        let on_admission = footer.approx_byte_size();
        assert!(on_admission.is_some(), "a parsed footer must report a size");

        // A cache hands the same footer back to every scan, so the tree is built into this object.
        for _ in 0..3 {
            materialise(footer.layout())?;
            assert_eq!(
                footer.approx_byte_size(),
                on_admission,
                "approx_byte_size changed after a scan; caches that record it at admission and \
             subtract it at eviction will underflow",
            );
        }

        Ok(())
    })
}

/// The reported size must be close to the heap the footer really retains, and must not
/// under-report: a cache bounded by an under-reported size holds a multiple of its budget.
#[rstest]
#[case::flat(FLAT)]
#[case::wide(WIDE)]
#[case::nested(NESTED)]
#[case::no_stats(NO_STATS)]
#[case::tiny(TINY)]
fn approx_byte_size_tracks_retained_heap(#[case] shape: Shape) -> VortexResult<()> {
    measured(|| {
        let file = write_file(shape)?;

        // Warm every lazily-initialised registry so it is not charged to the measurement.
        materialise(open_footer(&file)?.layout())?;

        let before = live();
        let footer = open_footer(&file)?;
        materialise(footer.layout())?;
        let retained = live() - before;
        let reported = footer
            .approx_byte_size()
            .expect("a parsed footer must report a size") as isize;
        drop(footer);
        let after_drop = live() - before;

        assert!(
            retained > 0,
            "expected the footer to retain something, measured {retained} bytes",
        );

        // Under-reporting is the failure that matters: it is what let a 50MB cache budget hold an
        // order of magnitude more. Allow a little slack for allocator differences across platforms.
        assert!(
            reported * 5 >= retained * 4,
            "approx_byte_size under-reports: reported {reported} B for {retained} B of retained heap \
         ({:.1}x). {shape:?}",
            retained as f64 / reported as f64,
        );

        // Over-reporting is safe but wasteful, and a large drift means the estimate has stopped
        // tracking what a footer actually holds.
        assert!(
            reported <= retained * 2,
            "approx_byte_size over-reports: reported {reported} B for {retained} B of retained heap \
         ({:.1}x). {shape:?}",
            reported as f64 / retained as f64,
        );

        // Nothing measured above may outlive the footer.
        assert!(
            after_drop < retained / 4,
            "dropping the footer left {after_drop} of {retained} bytes live; the footer graph is \
         holding on to something",
        );

        Ok(())
    })
}
