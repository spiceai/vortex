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

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::extension::datetime::TimeUnit;

/// The `(multiply, divide)` pair that rescales a value from `source_unit` to `target_unit`.
///
/// Exactly one of the two is ever a value other than 1, so a caller multiplies and then
/// divides without losing the low bits of a widening conversion.
pub(crate) fn date_to_timestamp_scale(
    source_unit: TimeUnit,
    target_unit: TimeUnit,
) -> VortexResult<(i64, i64)> {
    let source_ns = to_nanoseconds(source_unit)?;
    let target_ns = to_nanoseconds(target_unit)?;

    if source_ns >= target_ns {
        let multiply = source_ns / target_ns;
        return Ok((multiply, 1));
    }

    let divide = target_ns / source_ns;
    Ok((1, divide))
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

/// Rescale one temporal value by the pair from [`date_to_timestamp_scale`].
///
/// Refuses rather than rounds: a value the target unit cannot hold exactly, or one that leaves
/// the `i64` range, is an error, because a silently rounded instant is a wrong comparison.
pub(crate) fn convert_temporal_value(value: i64, multiply: i64, divide: i64) -> VortexResult<i64> {
    let mut scaled = i128::from(value)
        .checked_mul(i128::from(multiply))
        .ok_or_else(|| {
            vortex_error::vortex_err!(
                Compute: "Date value {value} overflows while scaling to timestamp"
            )
        })?;

    if divide != 1 {
        let divisor = i128::from(divide);
        if scaled % divisor != 0 {
            vortex_bail!(
                Compute: "Date value {value} cannot be represented exactly in target timestamp unit"
            );
        }
        scaled /= divisor;
    }

    if scaled < i128::from(i64::MIN) || scaled > i128::from(i64::MAX) {
        vortex_bail!(Compute: "Date value {value} overflows target timestamp range");
    }

    i64::try_from(scaled)
        .map_err(|_| vortex_error::vortex_err!(Compute: "Date value {value} overflows target timestamp range"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_scale_up_to_nanoseconds() -> VortexResult<()> {
        let (multiply, divide) = date_to_timestamp_scale(TimeUnit::Days, TimeUnit::Nanoseconds)?;
        assert_eq!((multiply, divide), (86_400_000_000_000, 1));
        assert_eq!(
            convert_temporal_value(19_783, multiply, divide)?,
            1_709_251_200_000_000_000
        );
        Ok(())
    }

    #[test]
    fn milliseconds_scale_up_to_nanoseconds() -> VortexResult<()> {
        let (multiply, divide) =
            date_to_timestamp_scale(TimeUnit::Milliseconds, TimeUnit::Nanoseconds)?;
        assert_eq!((multiply, divide), (1_000_000, 1));
        assert_eq!(
            convert_temporal_value(1_709_251_200_000, multiply, divide)?,
            1_709_251_200_000_000_000
        );
        Ok(())
    }

    #[test]
    fn a_value_the_target_unit_cannot_hold_exactly_is_refused() -> VortexResult<()> {
        // One millisecond, asked for in seconds: the conversion divides, and 1 is not a whole
        // number of seconds. Rounding it would move the instant.
        let (multiply, divide) =
            date_to_timestamp_scale(TimeUnit::Milliseconds, TimeUnit::Seconds)?;
        assert_eq!((multiply, divide), (1, 1_000));
        assert!(convert_temporal_value(1, multiply, divide).is_err());
        // A whole second still converts.
        assert_eq!(convert_temporal_value(2_000, multiply, divide)?, 2);
        Ok(())
    }

    #[test]
    fn a_value_that_leaves_the_i64_range_is_refused() -> VortexResult<()> {
        let (multiply, divide) = date_to_timestamp_scale(TimeUnit::Days, TimeUnit::Nanoseconds)?;
        assert!(convert_temporal_value(i64::MAX, multiply, divide).is_err());
        Ok(())
    }
}
