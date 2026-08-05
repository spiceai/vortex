// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cancelling a [`Task`] that has already been polled must not corrupt memory.
//!
//! Dropping a polled `Task` races the spawned future's completion: the task handle's
//! receiver is torn down while the spawned side is delivering its result and waking the
//! waker the poll registered. A channel implementation that releases that waker from
//! inside its own destructor reenters the executor's task teardown, freeing the waker
//! under the sender that is about to call it.
//!
//! This is a stress test, not a deterministic one. It reproduced the fault within a
//! second on every pre-fix run, but it detects the fault by *crashing the test process*
//! (SIGSEGV, or SIGABRT via the allocator's heap checks) rather than by failing an
//! assertion, because the damage is already done by the time control returns.

#![cfg(feature = "tokio")]

use std::future::Future;
use std::task::Poll;
use std::time::Duration;
use std::time::Instant;

use futures::future::poll_fn;
use vortex_io::runtime::tokio::TokioRuntime;

/// Long enough to cover the window reliably; short enough for CI.
const RUN_FOR: Duration = Duration::from_secs(3);
const WORKERS: u64 = 8;

/// Cheap deterministic jitter, so the cancellation lands at varying points relative to
/// the spawned future's completion. A fixed delay would only ever probe one interleaving.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cancelling_a_polled_task_is_sound() {
        let handle = TokioRuntime::current();
        let deadline = Instant::now() + RUN_FOR;

        let mut workers = Vec::new();
        for worker in 0..WORKERS {
            let handle = handle.clone();
            workers.push(tokio::spawn(async move {
                let mut seed = worker.wrapping_mul(7919).wrapping_add(1);
                let mut cancelled = 0u64;
                while Instant::now() < deadline {
                    let r = xorshift(&mut seed);
                    let spin = r % 2048;
                    let yields = (r >> 11) % 4;

                    let mut task = Box::pin(handle.spawn(async move {
                        for _ in 0..yields {
                            tokio::task::yield_now().await;
                        }
                        (0..spin).fold(0u64, |acc, i| acc.wrapping_add(i))
                    }));

                    // Poll once so the task handle registers a waker, then drop it while the
                    // spawned future is still completing. This is what query cancellation does.
                    let first = poll_fn(|cx| Poll::Ready(task.as_mut().poll(cx))).await;
                    if first.is_pending() {
                        for _ in 0..((r >> 24) % 3) {
                            tokio::task::yield_now().await;
                        }
                        drop(task);
                        cancelled += 1;
                    }
                }
                cancelled
            }));
        }

        let mut cancelled = 0u64;
        for worker in workers {
            cancelled += worker.await.expect("stress worker panicked");
        }
        assert!(cancelled > 0, "no cancellation actually raced a completion");
    }
}
