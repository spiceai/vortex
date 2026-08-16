// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Useful utilities for discovering the desired level of parallelism

use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::OnceLock;

/// The resolved parallelism, written once by whichever of [`set_available_parallelism`] and
/// [`get_available_parallelism`] runs first. A single cell keeps every reader and any later
/// declaration in agreement.
static PARALLELISM: OnceLock<Resolved> = OnceLock::new();

struct Resolved {
    /// `None` when detection failed and nothing was declared.
    value: Option<usize>,
    /// Whether an embedder declared this value, as opposed to it being detected.
    declared: bool,
}

/// The parallelism was already resolved to a different value, so the requested one was not
/// installed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetParallelismError {
    /// The value that was requested, and is *not* in effect.
    pub requested: NonZeroUsize,
    /// The value that is in effect. `None` if detection failed.
    pub installed: Option<usize>,
    /// Whether the installed value was declared by an earlier call, as opposed to detected.
    pub installed_was_declared: bool,
}

impl fmt::Display for SetParallelismError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let requested = self.requested;
        // `installed` is `None` only when detection ran and failed, so the `None` arm needs
        // no declared/detected distinction.
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
/// The operating system answers for the machine; an embedder running under a narrower CPU
/// entitlement (a container share, a scheduler allocation) declares that entitlement here so
/// every component sizes from it. It is a default fan-out, not an enforced ceiling: some
/// components derive a larger number from it.
///
/// Call during start-up, before anything reads the parallelism: whichever of the setter and
/// getter runs first fixes the value for the process. Declaring the value already in effect
/// succeeds. Each `cdylib` links its own copy of `vortex-utils` and must declare separately.
///
/// # Errors
///
/// [`SetParallelismError`] if a different value is already in effect (declared or detected).
/// The value in effect is unchanged.
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

    /// The detection path, in a binary where nothing declares a value. The declaration
    /// orderings live in their own integration-test binaries: the value resolves once per
    /// process.
    #[test]
    fn detects_parallelism_when_nothing_is_declared() {
        let detected = get_available_parallelism()
            .expect("available_parallelism is expected to resolve on a test host");
        assert!(
            detected >= 1,
            "parallelism must be positive, got {detected}"
        );

        // Repeated calls read the resolved value.
        assert_eq!(get_available_parallelism(), Some(detected));
    }
}
