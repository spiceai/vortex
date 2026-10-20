// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

/// A future that resolves to a mask.
#[derive(Clone)]
pub struct MaskFuture {
    inner: Shared<BoxFuture<'static, SharedVortexResult<Mask>>>,
    len: usize,
    /// The mask, when it was known before the future was created; see [`Self::known`].
    known: Option<Mask>,
}

impl MaskFuture {
    /// Create a new MaskFuture from a future that returns a mask.
    pub fn new<F>(len: usize, fut: F) -> Self
    where
        F: Future<Output = VortexResult<Mask>> + Send + 'static,
    {
        Self {
            inner: fut
                .inspect(move |r| {
                    if let Ok(mask) = r
                        && mask.len() != len {
                            vortex_panic!("MaskFuture created with future that returned mask of incorrect length (expected {}, got {})", len, mask.len());
                        }
                })
                .map_err(Arc::new)
                .boxed()
                .shared(),
            len,
            known: None,
        }
    }

    /// The mask, if it is known without awaiting: this future was created with [`Self::ready`]
    /// or [`Self::new_true`], or sliced from one that was.
    ///
    /// A reader can use it to skip work for rows the mask excludes before requesting any data.
    pub fn known(&self) -> Option<&Mask> {
        self.known.as_ref()
    }

    /// Returns the length of the mask.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the mask is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Create a MaskFuture from a ready mask.
    pub fn ready(mask: Mask) -> Self {
        let known = mask.clone();
        let mut future = Self::new(mask.len(), async move { Ok(mask) });
        future.known = Some(known);
        future
    }

    /// Create a MaskFuture that resolves to a mask with all values set to true.
    pub fn new_true(row_count: usize) -> Self {
        Self::ready(Mask::new_true(row_count))
    }

    /// Create a MaskFuture that resolves to a slice of the original mask.
    pub fn slice(&self, range: Range<usize>) -> Self {
        // Slicing the whole mask is the identity. Cloning shares the existing future instead of
        // allocating another boxed, shared one that would await it only to hand the mask back.
        if range.start == 0 && range.end == self.len {
            return self.clone();
        }

        if let Some(known) = &self.known {
            return Self::ready(known.slice(range));
        }

        let inner = self.inner.clone();
        Self::new(range.len(), async move { Ok(inner.await?.slice(range)) })
    }

    pub fn inspect(
        self,
        f: impl FnOnce(&SharedVortexResult<Mask>) + 'static + Send + Sync,
    ) -> Self {
        let len = self.len;

        Self {
            inner: self.inner.inspect(f).boxed().shared(),
            len,
            known: self.known,
        }
    }
}

impl Future for MaskFuture {
    type Output = VortexResult<Mask>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.inner.poll_unpin(cx).map_err(VortexError::from)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;

    use super::*;

    /// Slicing resolves to the same mask the equivalent [`Mask::slice`] would produce, for both
    /// the full range (which takes the identity fast path) and a sub-range.
    #[test]
    fn slice_resolves_to_sliced_mask() -> VortexResult<()> {
        futures::executor::block_on(async {
            let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, true, false]));
            let fut = MaskFuture::ready(mask.clone());

            let full = fut.slice(0..mask.len());
            assert_eq!(full.len(), mask.len());
            assert_eq!(full.await?, mask);

            let partial = fut.slice(0..mask.len() - 1);
            assert_eq!(partial.len(), mask.len() - 1);
            assert_eq!(partial.await?, mask.slice(0..mask.len() - 1));
            Ok(())
        })
    }
}

#[cfg(test)]
mod known_tests {
    use vortex_mask::Mask;

    use super::MaskFuture;

    #[test]
    fn a_ready_mask_and_its_slices_are_known() {
        let ready = MaskFuture::ready(Mask::from_indices(10, vec![2, 7]));
        assert_eq!(ready.known(), Some(&Mask::from_indices(10, vec![2, 7])));
        assert_eq!(
            ready.slice(5..10).known(),
            Some(&Mask::from_indices(5, vec![2]))
        );
        assert_eq!(MaskFuture::new_true(4).known(), Some(&Mask::new_true(4)));
    }

    #[test]
    fn a_pending_mask_is_not_known_until_awaited() {
        let pending = MaskFuture::new(4, async { Ok(Mask::new_true(4)) });
        assert!(pending.known().is_none());
        assert!(pending.slice(1..3).known().is_none());
    }
}
