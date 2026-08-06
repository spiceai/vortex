// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Useful utilities for discovering the desired level of parallelism

use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::OnceLock;

/// The one answer this copy of the library will ever give.
///
/// Written exactly once, by whichever of [`set_available_parallelism`] and
/// [`get_available_parallelism`] runs first. Keeping the declared value and the detected one
/// in a single cell is what makes the answer stable: components read it when they are
/// constructed and keep it, so a value that could still change would leave two components in
/// the same process sized against different numbers.
static PARALLELISM: OnceLock<Resolved> = OnceLock::new();

struct Resolved {
    /// `None` when detection failed and nothing was declared.
    value: Option<usize>,
    /// Whether an embedder declared this value, as opposed to it being detected.
    declared: bool,
}

/// Why [`set_available_parallelism`] could not install the requested value.
///
/// The parallelism was already resolved, and it cannot be revised: whatever read it has
/// already sized itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetParallelismError {
    /// The value that was requested, and is *not* in effect.
    pub requested: NonZeroUsize,
    /// The value that is in effect. `None` if detection failed.
    pub installed: Option<usize>,
    /// Whether the installed value was declared by an earlier call — an embedder that
    /// declares two different values — as opposed to detected, which means this call simply
    /// came after something already read the parallelism.
    pub installed_was_declared: bool,
}

impl fmt::Display for SetParallelismError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let requested = self.requested;
        // A failed detection is its own case rather than the word "unknown" substituted into
        // one of the other two, which would read "resolved to the detected unknown".
        // `installed` is only ever `None` on that path — declaring always stores a value — so
        // the `None` arm does not need to distinguish declared from detected.
        match (self.installed_was_declared, self.installed) {
            (true, Some(installed)) => write!(
                f,
                "cannot set the available parallelism to {requested}: {installed} was already \
                 declared"
            ),
            (false, Some(installed)) => write!(
                f,
                "cannot set the available parallelism to {requested}: it was already resolved to \
                 the detected {installed}, so this call came after something read it"
            ),
            (_, None) => write!(
                f,
                "cannot set the available parallelism to {requested}: detection already ran and \
                 failed, so this call came after something read it"
            ),
        }
    }
}

impl Error for SetParallelismError {}

/// Declares the parallelism [`get_available_parallelism`] reports.
///
/// [`get_available_parallelism`] otherwise asks the operating system, which answers for the
/// *machine* rather than for this process. The two differ whenever the host is entitled to
/// less than the machine: a container whose CPU allocation is expressed as a share rather
/// than a quota, a scheduler allocation, or an explicit user setting. Every component that
/// defaults its fan-out from that call then sizes for the whole machine at once. An embedder
/// that has already resolved what it is entitled to declares it here so that they all agree
/// with it.
///
/// This sets a *default fan-out*, not an enforced ceiling. Each component takes the value
/// independently, and some derive a larger number from it — a scan runs several tasks per
/// worker, for instance — so the total concurrency in flight is a multiple of it, not a
/// budget capped by it.
///
/// Call this during start-up, before anything reads the parallelism. Whichever call comes
/// first fixes the answer for good, so a declaration that arrives after any read fails rather
/// than leaving components in one process sized against two different numbers. Declaring the
/// value already in effect succeeds, so an embedder may call this from more than one entry
/// point.
///
/// The value is scoped to this copy of `vortex-utils`. A `cdylib` — the Python, JNI and C
/// bindings each build one — links its own, and must declare into it separately.
///
/// # Errors
///
/// [`SetParallelismError`] if a different value is already in effect, either because it was
/// declared earlier or because it was already detected. The value in effect is unchanged.
pub fn set_available_parallelism(parallelism: NonZeroUsize) -> Result<(), SetParallelismError> {
    let resolved = PARALLELISM.get_or_init(|| Resolved {
        value: Some(parallelism.get()),
        declared: true,
    });

    if resolved.value == Some(parallelism.get()) {
        Ok(())
    } else {
        Err(SetParallelismError {
            requested: parallelism,
            installed: resolved.value,
            installed_was_declared: resolved.declared,
        })
    }
}

/// Estimates the degree of parallelism the program should use, caching the result after the
/// first call.
///
/// Reports the value [`set_available_parallelism`] declared, if it was declared before the
/// first call. Otherwise this is currently implemented using
/// [`std::thread::available_parallelism`], but that might change in the future.
///
/// Returns `None` if nothing was declared and the underlying function fails.
pub fn get_available_parallelism() -> Option<usize> {
    PARALLELISM
        .get_or_init(|| Resolved {
            #[allow(clippy::disallowed_methods)]
            value: std::thread::available_parallelism().ok().map(|n| n.get()),
            declared: false,
        })
        .value
}

#[cfg(test)]
mod tests {
    use super::get_available_parallelism;

    /// The detection path, exercised in a binary where nothing declares a value. The tests
    /// that *do* declare one each live in their own integration-test binary, because the
    /// answer is resolved once per process and cannot be reset.
    #[test]
    fn detects_parallelism_when_nothing_is_declared() {
        let detected = get_available_parallelism()
            .expect("available_parallelism is expected to resolve on a test host");
        assert!(
            detected >= 1,
            "parallelism must be positive, got {detected}"
        );

        // Repeated calls read the resolved value rather than re-probing, so they cannot
        // disagree.
        assert_eq!(get_available_parallelism(), Some(detected));
    }
}
