// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use pin_project_lite::pin_project;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;

use crate::scan::split_gate::WaitScope;
use crate::scan::split_gate::current_scope;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// A [`SegmentSource`] that reports a scan split's waits for segment bytes.
///
/// A split that is polled under a scan's split gate and finds a requested segment not yet
/// available is waiting on I/O, which uses no CPU. The gate uses that to let other splits
/// decode in the meantime. Requests made outside a gated split are passed through untouched.
///
/// Wrap the outermost source a reader sees, so that a segment served from a cache or shared
/// with another request still counts as a wait for each request that is pending on it.
pub struct WaitMarkedSegmentSource<S> {
    inner: S,
}

impl<S: SegmentSource> WaitMarkedSegmentSource<S> {
    /// Wraps `inner`.
    pub fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: SegmentSource> SegmentSource for WaitMarkedSegmentSource<S> {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        WaitMarked {
            inner: self.inner.request(id),
            scope: None,
        }
        .boxed()
    }
}

pin_project! {
    struct WaitMarked {
        #[pin]
        inner: SegmentFuture,
        // The split this request is making wait, while it is pending.
        scope: Option<Arc<WaitScope>>,
    }

    impl PinnedDrop for WaitMarked {
        fn drop(this: Pin<&mut Self>) {
            if let Some(scope) = this.project().scope.take() {
                scope.end();
            }
        }
    }
}

impl Future for WaitMarked {
    type Output = VortexResult<BufferHandle>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let polled = this.inner.poll(cx);
        match (&polled, this.scope.is_some()) {
            (Poll::Pending, false) => {
                if let Some(scope) = current_scope() {
                    scope.begin();
                    *this.scope = Some(scope);
                }
            }
            (Poll::Ready(_), true) => {
                if let Some(scope) = this.scope.take() {
                    scope.end();
                }
            }
            _ => {}
        }
        polled
    }
}
