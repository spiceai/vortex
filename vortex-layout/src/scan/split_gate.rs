// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lets a scan keep more splits in flight than it decodes at once.
//!
//! A split spends most of its life waiting for segment bytes, during which it uses no CPU. When
//! the scan's concurrency is also the number of splits with reads outstanding, a high-latency
//! source (an object store) is left idle between a split's reads: the scan has to choose between
//! few reads in flight and many concurrent decodes.
//!
//! [`SplitGate`] separates the two. Every split takes a *decode permit* before it is polled and
//! gives it back while it waits on segment I/O marked by
//! [`WaitMarkedSegmentSource`](crate::segments::WaitMarkedSegmentSource). At most as many splits
//! decode as there are permits, however many are in flight, and a finished split keeps its
//! permit until its output leaves the scan stream, so the decoded batches a scan holds stay
//! bounded by the permit count as well.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use parking_lot::Mutex;
use pin_project_lite::pin_project;

/// Counts the marked waits that are in progress within one split.
#[derive(Debug, Default)]
pub(crate) struct WaitScope {
    waits: AtomicUsize,
}

impl WaitScope {
    pub(crate) fn begin(&self) {
        self.waits.fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) fn end(&self) {
        self.waits.fetch_sub(1, Ordering::AcqRel);
    }

    fn waiting(&self) -> bool {
        self.waits.load(Ordering::Acquire) > 0
    }
}

thread_local! {
    /// The split being polled on this thread, if any.
    static CURRENT: RefCell<Option<Arc<WaitScope>>> = const { RefCell::new(None) };
}

/// The split being polled on this thread, if there is one.
pub(crate) fn current_scope() -> Option<Arc<WaitScope>> {
    CURRENT.with(|current| current.borrow().clone())
}

/// Restores the previous scope when dropped, so a panic in the polled future cannot leak one.
struct ScopeGuard(Option<Arc<WaitScope>>);

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        CURRENT.with(|current| *current.borrow_mut() = self.0.take());
    }
}

#[derive(Debug)]
struct GateState {
    available: usize,
    waiters: VecDeque<(u64, Waker)>,
    next_waiter: u64,
}

/// A counting semaphore handing out decode permits.
#[derive(Debug)]
pub(crate) struct SplitGate {
    state: Mutex<GateState>,
}

impl SplitGate {
    pub(crate) fn new(permits: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(GateState {
                available: permits.max(1),
                waiters: VecDeque::new(),
                next_waiter: 0,
            }),
        })
    }

    /// Takes a permit if one is free, otherwise queues `waiter` to be woken when one is.
    fn poll_acquire(
        self: &Arc<Self>,
        waiter: &mut Option<u64>,
        cx: &Context<'_>,
    ) -> Poll<DecodePermit> {
        let mut state = self.state.lock();
        if state.available > 0 {
            state.available -= 1;
            if let Some(id) = waiter.take() {
                state.waiters.retain(|(queued, _)| *queued != id);
            }
            // Several releases can wake the same front waiter before it runs. Pass the wake-up
            // on if another permit is still free after this waiter acquires one.
            let next = (state.available > 0)
                .then(|| state.waiters.front().map(|(_, waker)| waker.clone()))
                .flatten();
            drop(state);
            if let Some(waker) = next {
                waker.wake();
            }
            return Poll::Ready(DecodePermit {
                gate: Arc::clone(self),
            });
        }
        match *waiter {
            Some(id) => {
                if let Some((_, waker)) = state.waiters.iter_mut().find(|(queued, _)| *queued == id)
                {
                    waker.clone_from(cx.waker());
                }
            }
            None => {
                let id = state.next_waiter;
                state.next_waiter += 1;
                state.waiters.push_back((id, cx.waker().clone()));
                *waiter = Some(id);
            }
        }
        Poll::Pending
    }

    /// Removes a waiter that will not poll again, passing on a wake-up it may have consumed.
    fn cancel(&self, id: u64) {
        let next = {
            let mut state = self.state.lock();
            state.waiters.retain(|(queued, _)| *queued != id);
            if state.available > 0 {
                state.waiters.front().map(|(_, waker)| waker.clone())
            } else {
                None
            }
        };
        if let Some(waker) = next {
            waker.wake();
        }
    }

    fn release(&self) {
        let next = {
            let mut state = self.state.lock();
            state.available += 1;
            state.waiters.front().map(|(_, waker)| waker.clone())
        };
        if let Some(waker) = next {
            waker.wake();
        }
    }
}

/// A decode permit; returned to its gate when dropped.
#[derive(Debug)]
pub(crate) struct DecodePermit {
    gate: Arc<SplitGate>,
}

impl Drop for DecodePermit {
    fn drop(&mut self) {
        self.gate.release();
    }
}

pin_project! {
    /// Runs one split future under a [`SplitGate`].
    ///
    /// The split is polled only while it holds a permit. It gives the permit back when a poll
    /// leaves it waiting on marked segment I/O, and takes one again before its next poll. When it
    /// finishes it hands its permit to the caller together with the output, to be dropped when
    /// the output leaves the scan.
    pub(crate) struct GatedSplit<F> {
        #[pin]
        inner: F,
        gate: Arc<SplitGate>,
        scope: Arc<WaitScope>,
        permit: Option<DecodePermit>,
        waiter: Option<u64>,
    }

    impl<F> PinnedDrop for GatedSplit<F> {
        fn drop(this: Pin<&mut Self>) {
            let this = this.project();
            if let Some(id) = this.waiter.take() {
                this.gate.cancel(id);
            }
        }
    }
}

impl<F> GatedSplit<F> {
    pub(crate) fn new(inner: F, gate: Arc<SplitGate>) -> Self {
        Self {
            inner,
            gate,
            scope: Arc::new(WaitScope::default()),
            permit: None,
            waiter: None,
        }
    }
}

impl<F: Future> Future for GatedSplit<F> {
    type Output = (F::Output, Option<DecodePermit>);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        if this.permit.is_none() {
            match this.gate.poll_acquire(this.waiter, cx) {
                Poll::Ready(permit) => *this.permit = Some(permit),
                Poll::Pending => return Poll::Pending,
            }
        }

        let previous = CURRENT.with(|current| current.replace(Some(Arc::clone(this.scope))));
        let polled = {
            let _restore = ScopeGuard(previous);
            this.inner.as_mut().poll(cx)
        };

        match polled {
            Poll::Ready(output) => Poll::Ready((output, this.permit.take())),
            Poll::Pending => {
                if this.scope.waiting() {
                    // Waiting on I/O uses no CPU: let another split decode meanwhile.
                    drop(this.permit.take());
                }
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Wake;
    use std::task::Waker;
    use std::time::Duration;

    use futures::FutureExt;
    use futures::StreamExt;
    use futures::future::BoxFuture;
    use futures::stream;
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::segments::SegmentFuture;
    use crate::segments::SegmentId;
    use crate::segments::SegmentSource;
    use crate::segments::WaitMarkedSegmentSource;
    use crate::segments::mark_pending_wait;

    /// Resolves each request after a pause, standing in for a remote read.
    struct SlowSource;

    impl SegmentSource for SlowSource {
        fn request(&self, _id: SegmentId) -> SegmentFuture {
            async {
                tokio::time::sleep(Duration::from_millis(20)).await;
                Ok(BufferHandle::new_host(ByteBuffer::empty()))
            }
            .boxed()
        }
    }

    /// Tracks how many splits are inside a CPU section and how many are in flight.
    #[derive(Default)]
    struct Probe {
        decoding: AtomicUsize,
        max_decoding: AtomicUsize,
        reading: AtomicUsize,
        max_reading: AtomicUsize,
    }

    fn bump(now: &AtomicUsize, max: &AtomicUsize) {
        let value = now.fetch_add(1, Ordering::SeqCst) + 1;
        max.fetch_max(value, Ordering::SeqCst);
    }

    fn split(
        source: Arc<WaitMarkedSegmentSource<SlowSource>>,
        probe: Arc<Probe>,
    ) -> BoxFuture<'static, VortexResult<()>> {
        async move {
            // CPU section before the read: the permit is held.
            bump(&probe.decoding, &probe.max_decoding);
            probe.decoding.fetch_sub(1, Ordering::SeqCst);

            bump(&probe.reading, &probe.max_reading);
            source.request(SegmentId::from(0)).await?;
            probe.reading.fetch_sub(1, Ordering::SeqCst);

            // CPU section after the read: the permit has been retaken. Yield first so another
            // split can be inside its own CPU section when this one is.
            bump(&probe.decoding, &probe.max_decoding);
            std::thread::sleep(Duration::from_millis(2));
            probe.decoding.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        }
        .boxed()
    }

    async fn run(permits: usize, in_flight: usize, splits: usize) -> Arc<Probe> {
        let probe = Arc::new(Probe::default());
        let source = Arc::new(WaitMarkedSegmentSource::new(SlowSource));
        let gate = SplitGate::new(permits);
        let tasks = (0..splits).map(|_| {
            let gated = GatedSplit::new(
                split(Arc::clone(&source), Arc::clone(&probe)),
                Arc::clone(&gate),
            );
            tokio::spawn(gated)
        });
        let done: Vec<_> = stream::iter(tasks)
            .buffer_unordered(in_flight)
            .map(|result| {
                let (output, permit) = result.expect("split task joins");
                drop(permit);
                output
            })
            .collect()
            .await;
        for output in done {
            output.expect("split succeeds");
        }
        probe
    }

    /// Splits waiting on a segment read hand their permit on, so more splits have reads
    /// outstanding than there are permits, while decoding stays within the permit count.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reads_overlap_beyond_the_permit_count() {
        let probe = run(2, 16, 64).await;
        assert!(
            probe.max_decoding.load(Ordering::SeqCst) <= 2,
            "decoding exceeded the permits: {}",
            probe.max_decoding.load(Ordering::SeqCst)
        );
        assert!(
            probe.max_reading.load(Ordering::SeqCst) > 2,
            "reads never overlapped beyond the permits: {}",
            probe.max_reading.load(Ordering::SeqCst)
        );
    }

    /// Without the gate's hand-off a split holds its permit through its read, so the reads in
    /// flight never exceed the permit count; this is the behaviour the hand-off removes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unmarked_waits_keep_the_permit() {
        let probe = Arc::new(Probe::default());
        let gate = SplitGate::new(2);
        let tasks = (0..32).map(|_| {
            let probe = Arc::clone(&probe);
            let unmarked = async move {
                bump(&probe.reading, &probe.max_reading);
                tokio::time::sleep(Duration::from_millis(5)).await;
                probe.reading.fetch_sub(1, Ordering::SeqCst);
            };
            tokio::spawn(GatedSplit::new(unmarked, Arc::clone(&gate)))
        });
        let done: Vec<_> = stream::iter(tasks)
            .buffer_unordered(16)
            .map(|result| {
                let (output, permit) = result.expect("split task joins");
                drop(permit);
                output
            })
            .collect()
            .await;
        assert_eq!(done.len(), 32);
        assert_eq!(probe.max_reading.load(Ordering::SeqCst), 2);
    }

    /// A finished split keeps its permit until the caller drops it.
    #[tokio::test]
    async fn a_finished_split_holds_its_permit_until_dropped() {
        let gate = SplitGate::new(1);
        let (first, permit) = GatedSplit::new(async { 1 }, Arc::clone(&gate)).await;
        assert_eq!(first, 1);
        assert!(permit.is_some(), "the finished split hands its permit over");
        let mut second = Box::pin(GatedSplit::new(async { 2 }, Arc::clone(&gate)));
        assert!(
            futures::poll!(second.as_mut()).is_pending(),
            "the second split must wait for the first one's permit"
        );
        drop(permit);
        let (value, _permit) = second.await;
        assert_eq!(value, 2);
    }

    #[tokio::test]
    async fn consumers_of_one_pending_array_each_release_their_permit() {
        let gate = SplitGate::new(1);
        // One marked source future is shared by two splits, like a FlatReader array future.
        let shared = mark_pending_wait(futures::future::pending::<usize>())
            .boxed()
            .shared();
        let mut first = Box::pin(GatedSplit::new(
            mark_pending_wait(shared.clone()),
            Arc::clone(&gate),
        ));
        let mut second = Box::pin(GatedSplit::new(
            mark_pending_wait(shared),
            Arc::clone(&gate),
        ));
        assert!(futures::poll!(first.as_mut()).is_pending());
        assert!(futures::poll!(second.as_mut()).is_pending());

        // The second consumer did not poll the shared source. It must still hand off its
        // permit so another split can start a read while the shared array is pending.
        let mut third = Box::pin(GatedSplit::new(async { 3 }, gate));
        assert!(futures::poll!(third.as_mut()).is_ready());
    }

    #[test]
    fn two_releases_wake_two_queued_splits() {
        struct CountWakes(AtomicUsize);
        impl Wake for CountWakes {
            fn wake(self: Arc<Self>) {
                self.wake_by_ref();
            }

            fn wake_by_ref(self: &Arc<Self>) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let gate = SplitGate::new(2);
        let first_waker: Waker = Arc::new(CountWakes(AtomicUsize::new(0))).into();
        let second_wakes = Arc::new(CountWakes(AtomicUsize::new(0)));
        let second_waker: Waker = Arc::clone(&second_wakes).into();
        let first_context = Context::from_waker(&first_waker);
        let second_context = Context::from_waker(&second_waker);

        let mut first_id = None;
        let mut second_id = None;
        let first = match gate.poll_acquire(&mut first_id, &first_context) {
            Poll::Ready(permit) => permit,
            Poll::Pending => panic!("first permit must be available"),
        };
        let second = match gate.poll_acquire(&mut second_id, &second_context) {
            Poll::Ready(permit) => permit,
            Poll::Pending => panic!("second permit must be available"),
        };

        assert!(
            gate.poll_acquire(&mut first_id, &first_context)
                .is_pending()
        );
        assert!(
            gate.poll_acquire(&mut second_id, &second_context)
                .is_pending()
        );
        drop(first);
        drop(second);

        let _first_reacquired = match gate.poll_acquire(&mut first_id, &first_context) {
            Poll::Ready(permit) => permit,
            Poll::Pending => panic!("released permit must be available"),
        };
        assert!(
            second_wakes.0.load(Ordering::SeqCst) > 0,
            "the second queued split must be woken for the remaining permit"
        );
    }
}
