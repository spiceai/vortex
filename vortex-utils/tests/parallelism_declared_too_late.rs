// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A declaration that arrives after something has already read the parallelism is refused,
//! and the value already in effect keeps being reported.
//!
//! This is the ordering that actually bites an embedder, and the reason the declared value
//! and the detected one share a single cell. Were they separate, a late declaration would
//! change what the getter reports while every component built before it kept the machine's
//! core count — one process sized against two different numbers, with nothing to show for it.
//! Failing loudly instead gives the embedder something it can log.
//!
//! One test per binary, for the reason given in `parallelism_declared`.

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

        // One more than whatever this host reports, so the assertions below distinguish the
        // requested value from the resolved one on any machine.
        let requested = NonZeroUsize::new(detected + 1).expect("detected + 1 is not zero");

        let err = set_available_parallelism(requested)
            .expect_err("declaring after the first read must be refused");
        assert_eq!(err.requested, requested);
        assert_eq!(err.installed, Some(detected));
        assert!(
            !err.installed_was_declared,
            "the value in effect came from detection, not from an earlier declaration"
        );

        // The refusal is not cosmetic: the detected value is still what everything reads.
        assert_eq!(get_available_parallelism(), Some(detected));
    }
}
