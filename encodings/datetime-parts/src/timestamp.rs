// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::extension::datetime::TimeUnit;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

pub const SECONDS_PER_DAY: i64 = 86_400; // 24 * 60 * 60

/// Parts of a Unix timestamp (time since 1970-01-01 00:00:00 UTC).
///
/// Broken down into:
///  * Days since epoch (an `i32`; a timestamp whose day count does not fit cannot be split,
///    see [`days_fit_i32`])
///  * Seconds within day (0-86399)
///  * Subseconds (range depends on `TimeUnit`)
pub struct TimestampParts {
    pub days: i32,
    pub seconds: i32,
    pub subseconds: i32,
}

/// Splits a Unix timestamp into its component parts.
///
/// # Arguments
///
/// * `timestamp` - Number of time units since the Unix epoch
/// * `time_unit` - Precision of the timestamp (ns, μs, ms, or s)
///
/// # Errors
///
/// Returns an error if `time_unit` is days, which cannot be split, or if the timestamp's day
/// count does not fit in an `i32`.
pub fn split(timestamp: i64, time_unit: TimeUnit) -> VortexResult<TimestampParts> {
    if !days_fit_i32(timestamp, time_unit)? {
        vortex_bail!("Timestamp {timestamp} {time_unit} has a day count that does not fit in i32");
    }
    Ok(match time_unit {
        TimeUnit::Nanoseconds => split_with_divisor::<1_000_000_000>(timestamp),
        TimeUnit::Microseconds => split_with_divisor::<1_000_000>(timestamp),
        TimeUnit::Milliseconds => split_with_divisor::<1_000>(timestamp),
        TimeUnit::Seconds => split_with_divisor::<1>(timestamp),
        TimeUnit::Days => vortex_bail!("Cannot handle day-level data"),
    })
}

/// Whether `timestamp`'s day count fits the `i32` that [`TimestampParts::days`] stores.
///
/// A `vortex.timestamp` accepts every `i64`, but in seconds or milliseconds the day count of
/// one past about ±5.8 million years leaves `i32`. Splitting it anyway truncates the days and
/// silently moves the instant, so callers check first and keep such values unsplit.
///
/// # Errors
///
/// Returns an error if `time_unit` is days, which cannot be split.
pub fn days_fit_i32(timestamp: i64, time_unit: TimeUnit) -> VortexResult<bool> {
    let ticks_per_day = match time_unit {
        TimeUnit::Nanoseconds => SECONDS_PER_DAY * 1_000_000_000,
        TimeUnit::Microseconds => SECONDS_PER_DAY * 1_000_000,
        TimeUnit::Milliseconds => SECONDS_PER_DAY * 1_000,
        TimeUnit::Seconds => SECONDS_PER_DAY,
        TimeUnit::Days => vortex_bail!("Cannot handle day-level data"),
    };
    Ok(i32::try_from(timestamp / ticks_per_day).is_ok())
}

/// Split a Unix timestamp into parts using a constant DIVISOR.
///
/// The caller guarantees the day count fits in `i32` (see [`days_fit_i32`]).
#[inline]
pub(crate) fn split_with_divisor<const DIVISOR: i64>(timestamp: i64) -> TimestampParts {
    let days = timestamp / (SECONDS_PER_DAY * DIVISOR);
    let total_seconds = timestamp / DIVISOR;
    let seconds = total_seconds - days * SECONDS_PER_DAY;
    let subseconds = timestamp - total_seconds * DIVISOR;
    TimestampParts {
        days: days.as_(),
        seconds: seconds.as_(),
        subseconds: subseconds.as_(),
    }
}

/// Combines timestamp parts back into a Unix timestamp.
///
/// # Arguments
///
/// * `parts` - Component parts of the timestamp (days, seconds, subseconds)
/// * `time_unit` - Precision of the timestamp (ns, μs, ms, or s)
///
/// # Errors
///
/// Returns an error if `time_unit` is days, which cannot be combined.
pub fn combine(ts_parts: TimestampParts, time_unit: TimeUnit) -> i64 {
    let divisor = match time_unit {
        TimeUnit::Nanoseconds => 1_000_000_000,
        TimeUnit::Microseconds => 1_000_000,
        TimeUnit::Milliseconds => 1_000,
        TimeUnit::Seconds => 1,
        TimeUnit::Days => vortex_panic!("Cannot handle day-level data"),
    };

    ts_parts.days as i64 * SECONDS_PER_DAY * divisor
        + ts_parts.seconds as i64 * divisor
        + ts_parts.subseconds as i64
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(TimeUnit::Seconds, 1)]
    #[case(TimeUnit::Milliseconds, 1_000)]
    #[case(TimeUnit::Microseconds, 1_000_000)]
    #[case(TimeUnit::Nanoseconds, 1_000_000_000)]
    fn test_split_divmod(#[case] unit: TimeUnit, #[case] divisor: i64) -> VortexResult<()> {
        let ticks_per_day = SECONDS_PER_DAY * divisor;
        for ts in [
            0,
            1,
            -1,
            divisor - 1,
            3723 * divisor + divisor / 2,
            -(3723 * divisor + divisor / 2),
            ticks_per_day - 1,
            ticks_per_day,
            -ticks_per_day,
        ] {
            let parts = split(ts, unit)?;

            let days: i32 = (ts / ticks_per_day).as_();
            let seconds: i32 = ((ts % ticks_per_day) / divisor).as_();
            let subseconds: i32 = ((ts % ticks_per_day) % divisor).as_();
            assert_eq!(parts.days, days);
            assert_eq!(parts.seconds, seconds);
            assert_eq!(parts.subseconds, subseconds);
            assert_eq!(combine(parts, unit), ts);
        }
        Ok(())
    }

    #[test]
    fn test_split_seconds() {
        // 1970-01-02 01:02:03 UTC
        let ts = SECONDS_PER_DAY + 3723; // 1 day + (1*3600 + 2*60 + 3) seconds
        let parts = split(ts, TimeUnit::Seconds).unwrap();
        assert_eq!(parts.days, 1);
        assert_eq!(parts.seconds, 3723);
        assert_eq!(parts.subseconds, 0);
    }

    #[test]
    fn test_split_milliseconds() {
        // 1970-01-02 01:02:03.456 UTC
        let ts = (SECONDS_PER_DAY + 3723) * 1000 + 456;
        let parts = split(ts, TimeUnit::Milliseconds).unwrap();
        assert_eq!(parts.days, 1);
        assert_eq!(parts.seconds, 3723);
        assert_eq!(parts.subseconds, 456);
    }

    #[test]
    fn test_split_microseconds() {
        // 1970-01-02 01:02:03.456789 UTC
        let ts = (SECONDS_PER_DAY + 3723) * 1_000_000 + 456789;
        let parts = split(ts, TimeUnit::Microseconds).unwrap();
        assert_eq!(parts.days, 1);
        assert_eq!(parts.seconds, 3723);
        assert_eq!(parts.subseconds, 456789);
    }

    #[test]
    fn test_split_nanoseconds() {
        // 1970-01-02 01:02:03.456789123 UTC
        let ts = (SECONDS_PER_DAY + 3723) * 1_000_000_000 + 456789123;
        let parts = split(ts, TimeUnit::Nanoseconds).unwrap();
        assert_eq!(parts.days, 1);
        assert_eq!(parts.seconds, 3723);
        assert_eq!(parts.subseconds, 456789123);
    }

    #[test]
    fn test_split_epoch() {
        let ts = 0;
        for unit in [
            TimeUnit::Seconds,
            TimeUnit::Milliseconds,
            TimeUnit::Microseconds,
            TimeUnit::Nanoseconds,
        ] {
            let parts = split(ts, unit).unwrap();
            assert_eq!(parts.days, 0);
            assert_eq!(parts.seconds, 0);
            assert_eq!(parts.subseconds, 0);
        }
    }

    #[test]
    fn test_split_days_error() {
        assert!(split(1, TimeUnit::Days).is_err());
    }

    #[test]
    fn test_combine_seconds() {
        // 1970-01-02 01:02:03 UTC
        let parts = TimestampParts {
            days: 1,
            seconds: 3723, // 1*3600 + 2*60 + 3
            subseconds: 0,
        };
        let ts = combine(parts, TimeUnit::Seconds);
        assert_eq!(ts, SECONDS_PER_DAY + 3723);
    }

    #[test]
    fn test_combine_milliseconds() {
        // 1970-01-02 01:02:03.456 UTC
        let parts = TimestampParts {
            days: 1,
            seconds: 3723,
            subseconds: 456,
        };
        let ts = combine(parts, TimeUnit::Milliseconds);
        assert_eq!(ts, (SECONDS_PER_DAY + 3723) * 1000 + 456);
    }

    #[test]
    fn test_combine_microseconds() {
        // 1970-01-02 01:02:03.456789 UTC
        let parts = TimestampParts {
            days: 1,
            seconds: 3723,
            subseconds: 456789,
        };
        let ts = combine(parts, TimeUnit::Microseconds);
        assert_eq!(ts, (SECONDS_PER_DAY + 3723) * 1_000_000 + 456789);
    }

    #[test]
    fn test_combine_nanoseconds() {
        // 1970-01-02 01:02:03.456789123 UTC
        let parts = TimestampParts {
            days: 1,
            seconds: 3723,
            subseconds: 456789123,
        };
        let ts = combine(parts, TimeUnit::Nanoseconds);
        assert_eq!(ts, (SECONDS_PER_DAY + 3723) * 1_000_000_000 + 456789123);
    }

    #[test]
    #[should_panic(expected = "Cannot handle day-level data")]
    fn test_combine_days_error() {
        let parts = TimestampParts {
            days: 1,
            seconds: 0,
            subseconds: 0,
        };
        combine(parts, TimeUnit::Days);
    }

    /// A day count past `i32` is refused rather than truncated, in the units it can happen in.
    #[rstest]
    #[case(TimeUnit::Seconds, i64::MAX)]
    #[case(TimeUnit::Seconds, i64::MIN)]
    #[case(TimeUnit::Seconds, 371_085_174_374_400)] // 2^32 days, which truncates to 0
    #[case(TimeUnit::Milliseconds, i64::MAX)]
    #[case(TimeUnit::Milliseconds, i64::MIN)]
    fn a_day_count_past_i32_is_refused(#[case] unit: TimeUnit, #[case] ts: i64) {
        assert!(!days_fit_i32(ts, unit).unwrap());
        assert!(split(ts, unit).is_err());
    }

    /// Every microsecond and nanosecond timestamp, and the 9999-12-31 sentinel in every unit,
    /// still splits and recombines exactly.
    #[rstest]
    #[case(TimeUnit::Microseconds, i64::MAX)]
    #[case(TimeUnit::Microseconds, i64::MIN)]
    #[case(TimeUnit::Nanoseconds, i64::MAX)]
    #[case(TimeUnit::Nanoseconds, i64::MIN)]
    #[case(TimeUnit::Seconds, 253_402_214_400)]
    #[case(TimeUnit::Milliseconds, 253_402_214_400_000)]
    #[case(TimeUnit::Microseconds, 253_402_214_400_000_000)]
    #[case(TimeUnit::Seconds, -377_705_116_800)]
    fn a_day_count_inside_i32_round_trips(#[case] unit: TimeUnit, #[case] ts: i64) {
        assert!(days_fit_i32(ts, unit).unwrap());
        assert_eq!(combine(split(ts, unit).unwrap(), unit), ts);
    }
}
