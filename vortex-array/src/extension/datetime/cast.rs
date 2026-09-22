// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Value conversion between temporal extension types.
//!
//! A `vortex.date` value is a count of `source_unit` since the epoch and a `vortex.timestamp`
//! value is a count of `target_unit` since the same epoch, so converting between them is a
//! rescale of the storage value. Both the array kernel and the scalar cast need it, and they
//! need to agree: a scan falsifies `cast(col as timestamp) > lit` into
//! `cast(max(col) as timestamp) <= lit`, binds `max(col)` to a literal and casts *that*, so a
//! file is pruned on the scalar conversion and its surviving rows are filtered on the array
//! one. Two conversions that disagree drop rows that match.
//!
//! [`DateToTimestamp`] is the one implementation of it, so neither caller can accept a value
//! the other refuses.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;

/// Converts `vortex.date` values into `vortex.timestamp` values of another unit.
///
/// Built once per conversion, so the per-value work is arithmetic and two comparisons.
pub(crate) struct DateToTimestamp {
    /// The scale to apply. Exactly one of these is ever other than 1, so a value is multiplied
    /// and then divided without losing the low bits of a widening conversion.
    multiply: i64,
    divide: i64,
    /// The instants the target `vortex.timestamp` can hold, in the target's own unit.
    min: i64,
    max: i64,
}

impl DateToTimestamp {
    /// Prepare the conversion from `source_unit` dates to `target_unit` timestamps.
    pub(crate) fn new(source_unit: TimeUnit, target_unit: TimeUnit) -> VortexResult<Self> {
        let source_ns = to_nanoseconds(source_unit)?;
        let target_ns = to_nanoseconds(target_unit)?;

        let (multiply, divide) = if source_ns >= target_ns {
            (source_ns / target_ns, 1)
        } else {
            (1, target_ns / source_ns)
        };

        let (min, max) = Timestamp::storage_range(target_unit)?;
        Ok(Self {
            multiply,
            divide,
            min,
            max,
        })
    }

    /// Rescale one date value into the target timestamp's unit.
    ///
    /// Refuses rather than rounds or wraps: a value the target unit cannot hold exactly, one
    /// that leaves the `i64` range, and one that lands outside the instants a
    /// `vortex.timestamp` can represent are all errors, because a silently moved instant is a
    /// wrong comparison.
    pub(crate) fn convert(&self, value: i64) -> VortexResult<i64> {
        let mut scaled = i128::from(value)
            .checked_mul(i128::from(self.multiply))
            .ok_or_else(|| {
                vortex_error::vortex_err!(
                    Compute: "Date value {value} overflows while scaling to timestamp"
                )
            })?;

        if self.divide != 1 {
            let divisor = i128::from(self.divide);
            if scaled % divisor != 0 {
                vortex_bail!(
                    Compute: "Date value {value} cannot be represented exactly in target timestamp unit"
                );
            }
            scaled /= divisor;
        }

        let scaled = i64::try_from(scaled).map_err(|_| {
            vortex_error::vortex_err!(Compute: "Date value {value} overflows target timestamp range")
        })?;

        // The `i64` range is wider than the instants a timestamp scalar accepts, and this is
        // where the array and the scalar would otherwise part company: an array can hold the
        // value, and a scalar built from it is rejected by `Timestamp`'s validation, so a
        // file's statistics fail to cast while the rows they gate convert happily. The bounds
        // are `Timestamp`'s own, so there is one range rather than two that must agree.
        if scaled < self.min || scaled > self.max {
            vortex_bail!(
                Compute: "Date value {value} is outside the range a timestamp can represent"
            );
        }

        Ok(scaled)
    }
}

/// The number of nanoseconds in one `unit`.
fn to_nanoseconds(unit: TimeUnit) -> VortexResult<i64> {
    match unit {
        TimeUnit::Nanoseconds => Ok(1),
        TimeUnit::Microseconds => Ok(1_000),
        TimeUnit::Milliseconds => Ok(1_000_000),
        TimeUnit::Seconds => Ok(1_000_000_000),
        TimeUnit::Days => Ok(86_400_000_000_000),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn days_scale_up_to_nanoseconds() -> VortexResult<()> {
        let conversion = DateToTimestamp::new(TimeUnit::Days, TimeUnit::Nanoseconds)?;
        assert_eq!(conversion.convert(19_783)?, 1_709_251_200_000_000_000);
        Ok(())
    }

    #[test]
    fn milliseconds_scale_up_to_nanoseconds() -> VortexResult<()> {
        let conversion = DateToTimestamp::new(TimeUnit::Milliseconds, TimeUnit::Nanoseconds)?;
        assert_eq!(
            conversion.convert(1_709_251_200_000)?,
            1_709_251_200_000_000_000
        );
        Ok(())
    }

    #[test]
    fn a_value_the_target_unit_cannot_hold_exactly_is_refused() -> VortexResult<()> {
        // One millisecond, asked for in seconds: the conversion divides, and 1 is not a whole
        // number of seconds. Rounding it would move the instant.
        let conversion = DateToTimestamp::new(TimeUnit::Milliseconds, TimeUnit::Seconds)?;
        assert!(conversion.convert(1).is_err());
        // A whole second still converts.
        assert_eq!(conversion.convert(2_000)?, 2);
        Ok(())
    }

    #[test]
    fn a_value_that_leaves_the_i64_range_is_refused() -> VortexResult<()> {
        let conversion = DateToTimestamp::new(TimeUnit::Days, TimeUnit::Nanoseconds)?;
        assert!(conversion.convert(i64::MAX).is_err());
        Ok(())
    }

    /// A date past the last instant a timestamp can hold has to be refused.
    ///
    /// The `i64` range does not catch these: `i32::MAX` days is a valid `Date32` value and
    /// scales into `i64` seconds with room to spare, but no timestamp scalar can be built from
    /// the result. Converting it anyway is what puts the array and the scalar out of step —
    /// the array holds it, and the scalar a file's statistics go through does not.
    #[rstest]
    #[case(TimeUnit::Seconds)]
    #[case(TimeUnit::Milliseconds)]
    #[case(TimeUnit::Microseconds)]
    fn a_date_beyond_the_last_representable_instant_is_refused(
        #[case] target_unit: TimeUnit,
    ) -> VortexResult<()> {
        let conversion = DateToTimestamp::new(TimeUnit::Days, target_unit)?;

        assert!(
            conversion.convert(i64::from(i32::MAX)).is_err(),
            "the largest Date32 value is far past the last instant"
        );
        // Jiff stops at the end of year 9999; 2_932_896 days is 10000-01-01.
        assert!(
            conversion.convert(2_932_896).is_err(),
            "the first day past the end of the range is refused"
        );
        assert!(
            conversion.convert(2_932_895).is_ok(),
            "the last day inside the range still converts"
        );
        Ok(())
    }

    /// The range is the target's, not the source's.
    #[test]
    fn the_representable_range_narrows_with_the_target_unit() -> VortexResult<()> {
        // 9999-12-31 fits in seconds through microseconds; nanoseconds run out in 2262.
        let last_day = 2_932_895;
        assert!(
            DateToTimestamp::new(TimeUnit::Days, TimeUnit::Seconds)?
                .convert(last_day)
                .is_ok()
        );
        assert!(
            DateToTimestamp::new(TimeUnit::Days, TimeUnit::Nanoseconds)?
                .convert(last_day)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn a_timestamp_in_days_has_no_conversion() {
        assert!(DateToTimestamp::new(TimeUnit::Days, TimeUnit::Days).is_err());
    }
}
