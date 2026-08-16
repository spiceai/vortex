// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Read-first ordering: a declaration arriving after the parallelism has been read is
//! refused, and the detected value stays in effect.
//!
//! One test per binary — see `parallelism_declared`.

use std::num::NonZeroUsize;

use vortex_utils::parallelism::get_available_parallelism;
use vortex_utils::parallelism::set_available_parallelism;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_declaration_after_the_first_read_is_refused() {
        let detected = get_available_parallelism()
            .expect("available_parallelism is expected to resolve on a test host");

        // One more than the detected value, so requested and installed differ on any host.
        let requested = NonZeroUsize::new(detected + 1).expect("detected + 1 is not zero");

        let err = set_available_parallelism(requested)
            .expect_err("declaring after the first read must be refused");
        assert_eq!(err.requested, requested);
        assert_eq!(err.installed, Some(detected));
        assert!(
            !err.installed_was_declared,
            "the value in effect came from detection, not from an earlier declaration"
        );

        // The detected value is still what readers see.
        assert_eq!(get_available_parallelism(), Some(detected));
    }
}
