// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::cmp;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::runtime::Handle;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::RowSplits;
use crate::SplitRange;
use crate::session::LayoutSession;

pub fn new_session() -> VortexSession {
    vortex_array::array_session()
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

/// A reader that splits one row per split and reports registration through a [`SplitProbe`].
///
/// Each split projects to a single-element `i32` array holding its own row index, so a scan over
/// it yields `0..row_count`.
#[derive(Debug)]
pub struct ProbeLayoutReader {
    name: Arc<str>,
    dtype: DType,
    row_count: u64,
    probe: Arc<SplitProbe>,
}

impl ProbeLayoutReader {
    pub fn new(row_count: usize, probe: Arc<SplitProbe>) -> Self {
        Self {
            name: Arc::from("probe"),
            dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
            row_count: row_count as u64,
            probe,
        }
    }
}

impl LayoutReader for ProbeLayoutReader {
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
        for split in (split_range.row_range().start + 1)..=split_range.row_range().end {
            splits.push(split_range.row_offset() + split);
        }
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        Ok(mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        _expr: &Expression,
        _mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let start = usize::try_from(row_range.start)
            .map_err(|_| vortex_err!("row_range.start must fit in usize"))?;
        let end = usize::try_from(row_range.end)
            .map_err(|_| vortex_err!("row_range.end must fit in usize"))?;

        let values: VortexResult<Vec<i32>> = (start..end)
            .map(|v| i32::try_from(v).map_err(|_| vortex_err!("split value must fit in i32")))
            .collect();
        let array = PrimitiveArray::from_iter(values?).into_array();

        // Registering the projection is what submits this split's reads, so count it here
        // rather than when the returned future is polled.
        self.probe.register();
        let probe = Arc::clone(&self.probe);

        Ok(Box::pin(async move {
            probe.complete();
            Ok(array)
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Asserts that a scan of `row_count` splits yielded `0..row_count` while never holding more
/// than `concurrency` splits registered.
#[expect(
    clippy::cast_possible_truncation,
    reason = "test row counts are far below i32::MAX"
)]
pub fn assert_bounded_scan(
    values: &[i32],
    row_count: usize,
    probe: &SplitProbe,
    concurrency: usize,
) {
    let expected: Vec<i32> = (0..row_count).map(|v| v as i32).collect();
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
