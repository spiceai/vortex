// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Declare-first ordering: the declared value is reported and cannot be revised.
//!
//! The value resolves once per process, so each ordering needs its own test binary;
//! `parallelism_declared_too_late` covers read-first.

use std::num::NonZeroUsize;

use vortex_utils::parallelism::get_available_parallelism;
use vortex_utils::parallelism::set_available_parallelism;

/// Not a plausible core count, so a detected value cannot satisfy the assertions by
/// coincidence.
const DECLARED: usize = 7919;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_declared_parallelism_is_reported_and_cannot_be_revised() {
        let requested = NonZeroUsize::new(DECLARED).expect("the declared value is not zero");

        set_available_parallelism(requested).expect("the first declaration must succeed");
        assert_eq!(get_available_parallelism(), Some(DECLARED));

        // Re-declaring the same value succeeds.
        set_available_parallelism(requested).expect("re-declaring the same value must succeed");
        assert_eq!(get_available_parallelism(), Some(DECLARED));

        // A conflicting declaration is refused; the first value stays in effect.
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
