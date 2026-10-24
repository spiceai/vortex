// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use pin_project_lite::pin_project;

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
        mark_pending_wait(self.inner.request(id)).boxed()
    }
}

/// Mark each consumer's pending wait, including consumers of a shared layout future.
/// The underlying segment future may be polled by just one consumer, so marking only that
/// future would leave the other consumers holding decode permits while they wait for it.
pub(crate) fn mark_pending_wait<F: Future>(inner: F) -> impl Future<Output = F::Output> {
    WaitMarked { inner, scope: None }
}

pin_project! {
    struct WaitMarked<F> {
        #[pin]
        inner: F,
        // The split this request is making wait, while it is pending.
        scope: Option<Arc<WaitScope>>,
    }

    impl<F> PinnedDrop for WaitMarked<F> {
        fn drop(this: Pin<&mut Self>) {
            if let Some(scope) = this.project().scope.take() {
                scope.end();
            }
        }
    }
}

impl<F: Future> Future for WaitMarked<F> {
    type Output = F::Output;

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
