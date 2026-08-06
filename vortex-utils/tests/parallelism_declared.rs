// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A declared parallelism is what [`get_available_parallelism`] reports, and it cannot be
//! revised afterwards.
//!
//! The value is resolved once per process and cannot be reset, so this binary holds exactly
//! one test: a second test in the same file would race it for the single resolution. The
//! companion binary `parallelism_declared_too_late` covers the opposite ordering.

use std::num::NonZeroUsize;

use vortex_utils::parallelism::get_available_parallelism;
use vortex_utils::parallelism::set_available_parallelism;

/// Deliberately not a plausible core count: if the operating system's answer were still being
/// reported, the assertions below could not pass by coincidence.
const DECLARED: usize = 7919;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_declared_parallelism_is_reported_and_cannot_be_revised() {
        let requested = NonZeroUsize::new(DECLARED).expect("the declared value is not zero");

        set_available_parallelism(requested).expect("the first declaration must succeed");
        assert_eq!(get_available_parallelism(), Some(DECLARED));

        // Idempotent, so an embedder may declare its value from more than one entry point.
        set_available_parallelism(requested).expect("re-declaring the same value must succeed");
        assert_eq!(get_available_parallelism(), Some(DECLARED));

        // A conflicting declaration is refused, and reports what stays in effect rather than
        // handing back what was rejected. Anything already sized from the first answer would
        // not be resized, so the two would disagree.
        let conflicting = NonZeroUsize::new(4).expect("4 is not zero");
        let err = set_available_parallelism(conflicting)
            .expect_err("a second, different declaration must be refused");
        assert_eq!(err.requested, conflicting);
        assert_eq!(err.installed, Some(DECLARED));
        assert!(
            err.installed_was_declared,
            "the value in effect was declared, not detected"
        );
        assert_eq!(get_available_parallelism(), Some(DECLARED));
    }
}
