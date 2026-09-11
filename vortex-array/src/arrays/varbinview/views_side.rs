// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`VarBinViewArray`] with its buffers resolved once, for callers that walk
//! every row.

use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;

/// A resolved view over a canonical [`VarBinViewArray`]: the view structs plus borrowed slices of
/// every data buffer, supporting cheap per-lane byte access.
pub(crate) struct ViewsSide<'a> {
    views: &'a [BinaryView],
    buffers: Vec<&'a [u8]>,
}

impl<'a> ViewsSide<'a> {
    pub(crate) fn new(array: &'a VarBinViewArray) -> Self {
        Self {
            views: array.views(),
            buffers: (0..array.data_buffers().len())
                .map(|idx| array.buffer(idx).as_slice())
                .collect(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.views.len()
    }

    /// The view structs themselves, for a caller walking every row in order.
    pub(crate) fn views(&self) -> &'a [BinaryView] {
        self.views
    }

    /// The view at `index` without a bounds check.
    ///
    /// # Safety
    ///
    /// `index` must be strictly less than `self.len()`.
    #[inline]
    pub(crate) unsafe fn view_unchecked(&self, index: usize) -> &'a BinaryView {
        // SAFETY: caller guarantees index < self.views.len().
        unsafe { self.views.get_unchecked(index) }
    }

    /// The full bytes of `view`, which must belong to this side.
    #[inline]
    pub(crate) fn view_bytes(&self, view: &'a BinaryView) -> &'a [u8] {
        if view.is_inlined() {
            view.as_inlined().value()
        } else {
            let view = view.as_view();
            &self.buffers[view.buffer_index as usize][view.as_range()]
        }
    }
}
