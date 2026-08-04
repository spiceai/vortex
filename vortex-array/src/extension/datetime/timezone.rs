// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Resolution of Arrow timezone strings into `jiff` time zones.

use jiff::tz::Offset;
use jiff::tz::TimeZone;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

/// Resolves an Arrow timezone string into a [`TimeZone`].
///
/// Arrow permits the timezone of a timestamp to be either an IANA time zone name
/// (`America/New_York`) or a fixed offset from UTC (`+00:00`). `jiff` only resolves the former,
/// so fixed offsets are parsed directly. Producers that emit the offset form are common: Iceberg
/// maps every `timestamptz` column to `+00:00`.
///
/// # Examples
///
/// ```
/// use vortex_array::extension::datetime::resolve_timezone;
///
/// assert_eq!(resolve_timezone("+00:00")?, jiff::tz::TimeZone::UTC);
/// assert_eq!(resolve_timezone("-05:30")?.to_fixed_offset()?.seconds(), -19_800);
/// assert!(resolve_timezone("Not/A/Timezone").is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn resolve_timezone(timezone: &str) -> VortexResult<TimeZone> {
    if let Ok(tz) = TimeZone::get(timezone) {
        return Ok(tz);
    }

    let Some(seconds) = parse_utc_offset_seconds(timezone) else {
        vortex_bail!("unknown timezone: {timezone}");
    };
    let offset = Offset::from_seconds(seconds)
        .map_err(|e| vortex_err!("timezone offset out of range: {timezone}: {e}"))?;

    Ok(TimeZone::fixed(offset))
}

/// Whether an Arrow timezone string denotes UTC.
///
/// UTC has more than one Arrow spelling: any zero fixed offset such as `+00:00` (which is what
/// Iceberg and `arrow-rs` emit), and the IANA names that are zero-offset for all time, such as
/// `UTC` and `Etc/UTC`.
///
/// # Examples
///
/// ```
/// use vortex_array::extension::datetime::is_utc_timezone;
///
/// assert!(is_utc_timezone("UTC"));
/// assert!(is_utc_timezone("Etc/UTC"));
/// assert!(is_utc_timezone("+00:00"));
/// assert!(is_utc_timezone("-00:00"));
/// assert!(!is_utc_timezone("+09:00"));
/// assert!(!is_utc_timezone("America/New_York"));
/// ```
pub fn is_utc_timezone(timezone: &str) -> bool {
    resolve_timezone(timezone).is_ok_and(|tz| match tz.to_fixed_offset() {
        Ok(offset) => offset == Offset::UTC,
        // A named IANA zone has no single offset, so it denotes UTC only if it is zero-offset with
        // no transitions in either direction.
        Err(_) => {
            let epoch = jiff::Timestamp::UNIX_EPOCH;
            tz.to_offset(epoch) == Offset::UTC
                && tz.following(epoch).next().is_none()
                && tz.preceding(epoch).next().is_none()
        }
    })
}

/// Parses the fixed UTC offset forms Arrow allows — `±HH:MM`, `±HHMM` and `±HH` — into a signed
/// number of seconds. Returns `None` for anything else, including offsets whose minutes component
/// is not a valid minute.
fn parse_utc_offset_seconds(timezone: &str) -> Option<i32> {
    let bytes = timezone.as_bytes();

    let digits = match bytes.len() {
        // ±HH:MM
        6 if bytes[3] == b':' => [bytes[1], bytes[2], bytes[4], bytes[5]],
        // ±HHMM
        5 => [bytes[1], bytes[2], bytes[3], bytes[4]],
        // ±HH
        3 => [bytes[1], bytes[2], b'0', b'0'],
        _ => return None,
    };

    let mut values = [0i32; 4];
    for (value, digit) in values.iter_mut().zip(digits) {
        *value = i32::from(digit.checked_sub(b'0').filter(|d| *d < 10)?);
    }

    let minutes = values[2] * 10 + values[3];
    if minutes > 59 {
        return None;
    }
    // Hours are not bounded here; `Offset::from_seconds` rejects anything beyond ±25:59:59.
    let seconds = (values[0] * 10 + values[1]) * 60 * 60 + minutes * 60;

    match bytes[0] {
        b'+' => Some(seconds),
        b'-' => Some(-seconds),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use jiff::tz::TimeZone;
    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::extension::datetime::is_utc_timezone;
    use crate::extension::datetime::resolve_timezone;

    #[rstest]
    // Iceberg emits this form for every `timestamptz` column.
    #[case("+00:00", 0)]
    #[case("-00:00", 0)]
    #[case("+09:00", 9 * 60 * 60)]
    #[case("-05:30", -(5 * 60 * 60 + 30 * 60))]
    #[case("+14:00", 14 * 60 * 60)]
    // The abbreviated forms Arrow also permits.
    #[case("+0930", 9 * 60 * 60 + 30 * 60)]
    #[case("-08", -(8 * 60 * 60))]
    fn resolves_fixed_offset(
        #[case] timezone: &str,
        #[case] expected_seconds: i32,
    ) -> VortexResult<()> {
        let tz = resolve_timezone(timezone)?;
        assert_eq!(tz.to_fixed_offset()?.seconds(), expected_seconds);
        Ok(())
    }

    #[test]
    fn resolves_iana_name() -> VortexResult<()> {
        // IANA lookup still takes precedence over offset parsing.
        let tz = resolve_timezone("America/New_York")?;
        assert_eq!(tz.iana_name(), Some("America/New_York"));
        assert_eq!(resolve_timezone("UTC")?, TimeZone::UTC);
        Ok(())
    }

    #[rstest]
    #[case("Not/A/Timezone")]
    #[case("")]
    // Missing sign.
    #[case("00:00")]
    // Wrong separator, and a bare separator in the abbreviated position.
    #[case("+00-00")]
    #[case("+0:000")]
    // Non-digits where digits are required.
    #[case("+aa:bb")]
    #[case("+ab")]
    // Minutes out of range.
    #[case("+00:60")]
    #[case("+0099")]
    // Hours beyond the ±25:59:59 offset range.
    #[case("+26:00")]
    #[case("-99")]
    fn rejects_invalid_timezone(#[case] timezone: &str) {
        assert!(
            resolve_timezone(timezone).is_err(),
            "expected {timezone} to be rejected"
        );
    }

    #[rstest]
    // Every spelling of a zero offset denotes UTC.
    #[case("+00:00", true)]
    #[case("-00:00", true)]
    #[case("+0000", true)]
    #[case("+00", true)]
    // So do the IANA names that are zero-offset for all time.
    #[case("UTC", true)]
    #[case("Etc/UTC", true)]
    #[case("Etc/GMT", true)]
    #[case("Zulu", true)]
    #[case("Universal", true)]
    // Non-zero offsets and zones that are not UTC do not.
    #[case("+09:00", false)]
    #[case("-05:30", false)]
    #[case("+00:01", false)]
    #[case("Etc/GMT+5", false)]
    // A zone that is zero-offset only for part of its history is not UTC.
    #[case("Europe/London", false)]
    #[case("America/New_York", false)]
    // Neither is anything unresolvable.
    #[case("Not/A/Timezone", false)]
    #[case("", false)]
    fn detects_utc_timezone(#[case] timezone: &str, #[case] expected: bool) {
        assert_eq!(is_utc_timezone(timezone), expected, "for {timezone}");
    }
}
