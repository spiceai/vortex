// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Poll;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Handle;
use vortex_io::runtime::single::SingleThreadRuntime;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::RowSplits;
use crate::SplitRange;
use crate::session::LayoutSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

pub fn new_session() -> VortexSession {
    VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
}

pub fn session_with_handle(handle: Handle) -> VortexSession {
    new_session().with_handle(handle)
}

pub static SCAN_SESSION: LazyLock<VortexSession> = LazyLock::new(new_session);

/// Counts how many splits a scan has registered reads for, and how many of those are still
/// outstanding.
///
/// A split registers its reads when its task is *constructed* (see `scan::tasks::split_exec`),
/// which is what lets a scan's concurrency mean something: only admitted splits should have
/// registered.
#[derive(Debug, Default)]
pub struct SplitProbe {
    registered: AtomicUsize,
    outstanding: AtomicUsize,
    max_outstanding: AtomicUsize,
}

impl SplitProbe {
    /// The number of splits whose reads have been registered.
    pub fn registered(&self) -> usize {
        self.registered.load(Ordering::SeqCst)
    }

    /// The high-water mark of splits registered but not yet completed.
    pub fn max_outstanding(&self) -> usize {
        self.max_outstanding.load(Ordering::SeqCst)
    }

    fn register(&self) {
        self.registered.fetch_add(1, Ordering::SeqCst);
        let outstanding = self.outstanding.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_outstanding
            .fetch_max(outstanding, Ordering::SeqCst);
    }

    fn complete(&self) {
        self.outstanding.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Asserts a scan over one-row splits returned every row in order, registered every split, and
/// never held more than `concurrency` splits registered at once.
pub fn assert_bounded_scan(
    values: &[i32],
    row_count: usize,
    probe: &SplitProbe,
    concurrency: usize,
) {
    let expected: Vec<i32> = (0..row_count)
        .map(|v| i32::try_from(v).unwrap_or(i32::MAX))
        .collect();
    assert_eq!(values, expected.as_slice());
    assert_eq!(probe.registered(), row_count);

    let expected_max = cmp::min(concurrency, row_count);
    assert_eq!(
        probe.max_outstanding(),
        expected_max,
        "expected at most {expected_max} splits registered at once, \
         which requires split construction to be lazy under the concurrency buffer"
    );
}

/// A configurable [`LayoutReader`] test double producing `base + row` for every selected row.
///
/// `split_size` controls the split layout (`None` is a single split), `keep_row` filters rows, and
/// the `fail_*` flags inject filter/projection errors. Every projection records its mask's
/// true-count into `projection_masks`, letting tests assert that a limit is applied before
/// projection, and every filter evaluation records its expression into `filter_exprs`.
#[derive(Debug)]
pub struct TestLayoutReader {
    name: Arc<str>,
    dtype: DType,
    row_count: u64,
    base: i32,
    split_size: Option<u64>,
    keep_row: fn(u64) -> bool,
    fail_first_filter: bool,
    fail_first_projection: bool,
    fail_projection: bool,
    projection_masks: Option<Arc<Mutex<Vec<usize>>>>,
    filter_exprs: Option<Arc<Mutex<Vec<BoundExpression>>>>,
    split_probe: Option<Arc<SplitProbe>>,
}

impl TestLayoutReader {
    pub fn new(row_count: u64) -> Self {
        Self {
            name: Arc::from("test"),
            dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
            row_count,
            base: 0,
            split_size: None,
            keep_row: keep_all,
            fail_first_filter: false,
            fail_first_projection: false,
            fail_projection: false,
            projection_masks: None,
            filter_exprs: None,
            split_probe: None,
        }
    }

    /// Report each projection, which a split registers when its task is constructed, to `probe`.
    pub fn with_split_probe(mut self, probe: Arc<SplitProbe>) -> Self {
        self.split_probe = Some(probe);
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Offset every produced value, so tests can tell which reader a row came from.
    pub fn with_base(mut self, base: i32) -> Self {
        self.base = base;
        self
    }

    pub fn with_split_size(mut self, split_size: u64) -> Self {
        self.split_size = Some(split_size);
        self
    }

    pub fn with_keep_row(mut self, keep_row: fn(u64) -> bool) -> Self {
        self.keep_row = keep_row;
        self
    }

    pub fn with_projection_masks(mut self, projection_masks: Arc<Mutex<Vec<usize>>>) -> Self {
        self.projection_masks = Some(projection_masks);
        self
    }

    pub fn with_filter_exprs(mut self, filter_exprs: Arc<Mutex<Vec<BoundExpression>>>) -> Self {
        self.filter_exprs = Some(filter_exprs);
        self
    }

    pub fn with_fail_first_filter(mut self) -> Self {
        self.fail_first_filter = true;
        self
    }

    pub fn with_fail_first_projection(mut self) -> Self {
        self.fail_first_projection = true;
        self
    }

    pub fn with_projection_error(mut self) -> Self {
        self.fail_projection = true;
        self
    }
}

impl LayoutReader for TestLayoutReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        let row_range = split_range.row_range();
        if let Some(size) = self.split_size {
            let mut boundary = row_range.start + size;
            while boundary < row_range.end {
                splits.push(split_range.row_offset() + boundary);
                boundary += size;
            }
        }
        splits.push(split_range.root_row_range().end);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        if let Some(filter_exprs) = &self.filter_exprs {
            filter_exprs.lock().push(expr.clone());
        }
        if self.fail_first_filter && row_range.start == 0 {
            let len = mask.len();
            return Ok(MaskFuture::new(len, async move {
                Err(vortex_err!("first split filter failed"))
            }));
        }

        let row_range = row_range.clone();
        let keep_row = self.keep_row;
        let row_count = usize::try_from(row_range.end - row_range.start)
            .map_err(|_| vortex_err!("row range must fit in usize"))?;

        Ok(MaskFuture::new(row_count, async move {
            let input_mask = mask.await?;
            Ok(Mask::from_iter(
                (row_range.start..row_range.end)
                    .enumerate()
                    .map(|(idx, row)| input_mask.value(idx) && keep_row(row)),
            ))
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        _expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let row_range = row_range.clone();
        let base = self.base;
        let projection_masks = self.projection_masks.clone();
        let fail = self.fail_projection || (self.fail_first_projection && row_range.start == 0);
        let split_probe = self.split_probe.clone();
        if let Some(probe) = &split_probe {
            probe.register();
        }

        Ok(Box::pin(async move {
            let mask = mask.await;
            if let Some(probe) = &split_probe {
                probe.complete();
            }
            let mask = mask?;
            if let Some(projection_masks) = projection_masks {
                projection_masks.lock().push(mask.true_count());
            }
            if fail {
                return Err(vortex_err!("projection failed"));
            }
            let start = i32::try_from(row_range.start)
                .map_err(|_| vortex_err!("row_range.start must fit in i32"))?;
            let end = i32::try_from(row_range.end)
                .map_err(|_| vortex_err!("row_range.end must fit in i32"))?;
            PrimitiveArray::from_iter((start..end).map(|value| base + value))
                .into_array()
                .filter(mask)
        }))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub fn keep_all(_: u64) -> bool {
    true
}

pub fn keep_odd(row: u64) -> bool {
    row % 2 == 1
}

/// Canonicalize every chunk of a scan into a flat list of `i32` values.
pub fn collect_scan_values<I>(iter: I) -> VortexResult<Vec<i32>>
where
    I: IntoIterator<Item = VortexResult<ArrayRef>>,
{
    let mut ctx = SESSION.create_execution_ctx();
    let mut values = Vec::new();
    for chunk in iter {
        let primitive = chunk?.execute::<PrimitiveArray>(&mut ctx)?;
        values.extend(primitive.into_buffer::<i32>());
    }
    Ok(values)
}

/// Let already-spawned scan tasks run to completion after the stream has been dropped.
pub fn drain_runtime(runtime: &SingleThreadRuntime) {
    for _ in 0..4 {
        let mut yielded = false;
        runtime.block_on(futures::future::poll_fn(move |cx| {
            if yielded {
                Poll::Ready(())
            } else {
                yielded = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }));
    }
}
