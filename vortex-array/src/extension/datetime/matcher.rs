// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtVTable;
use crate::dtype::extension::Matcher;
use crate::extension::datetime::Date;
use crate::extension::datetime::Time;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::extension::datetime::resolve_timezone;

/// Matcher for temporal extension data types.
pub struct AnyTemporal;

impl Matcher for AnyTemporal {
    type Match<'a> = TemporalMetadata<'a>;

    fn try_match<'a>(item: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if let Some(opts) = item.metadata_opt::<Timestamp>() {
            return Some(TemporalMetadata::Timestamp(&opts.unit, &opts.tz));
        }
        if let Some(opts) = item.metadata_opt::<Date>() {
            return Some(TemporalMetadata::Date(opts));
        }
        if let Some(opts) = item.metadata_opt::<Time>() {
            return Some(TemporalMetadata::Time(opts));
        }
        None
    }
}

/// Metadata for temporal extension data types.
#[derive(Debug, PartialEq, Eq)]
pub enum TemporalMetadata<'a> {
    /// Metadata for Timestamp dtypes, a tuple of time unit and optional timezone.
    Timestamp(&'a TimeUnit, &'a Option<Arc<str>>),
    /// Metadata for Date dtypes
    Date(&'a <Date as ExtVTable>::Metadata),
    /// Metadata for Time dtypes
    Time(&'a <Time as ExtVTable>::Metadata),
}

// TODO(ngates): remove this logic in favor of having an ExtScalarVTable in vortex_array::scalar.
//  Currently this is used largely to implement scalar display hacks.
impl TemporalMetadata<'_> {
    /// Get the time unit of the temporal dtype.
    pub fn time_unit(&self) -> TimeUnit {
        match self {
            TemporalMetadata::Time(unit) => **unit,
            TemporalMetadata::Date(unit) => **unit,
            TemporalMetadata::Timestamp(unit, _tz) => **unit,
        }
    }

    /// Convert a timestamp value to a Jiff value.
    pub fn to_jiff(&self, v: i64) -> VortexResult<TemporalJiff> {
        match self {
            TemporalMetadata::Time(unit) => Ok(TemporalJiff::Time(
                jiff::civil::Time::MIN.checked_add(unit.to_jiff_span(v)?)?,
            )),
            TemporalMetadata::Date(unit) => Ok(TemporalJiff::Date(
                jiff::civil::Date::new(1970, 1, 1)?.checked_add(unit.to_jiff_span(v)?)?,
            )),
            TemporalMetadata::Timestamp(unit, tz) => match tz {
                None => Ok(TemporalJiff::Unzoned(
                    jiff::civil::DateTime::new(1970, 1, 1, 0, 0, 0, 0)?
                        .checked_add(unit.to_jiff_span(v)?)?,
                )),
                Some(tz) => Ok(TemporalJiff::Zoned(
                    jiff::Timestamp::UNIX_EPOCH
                        .checked_add(unit.to_jiff_span(v)?)?
                        .to_zoned(resolve_timezone(tz.as_ref())?),
                )),
            },
        }
    }
}

/// A Jiff representation of a temporal value.
pub enum TemporalJiff {
    /// A time value.
    Time(jiff::civil::Time),
    /// A date value.
    Date(jiff::civil::Date),
    /// A zone-naive timestamp value.
    Unzoned(jiff::civil::DateTime),
    /// A zoned timestamp value.
    Zoned(jiff::Zoned),
}

impl Display for TemporalJiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemporalJiff::Time(t) => write!(f, "{t}"),
            TemporalJiff::Date(d) => write!(f, "{d}"),
            TemporalJiff::Unzoned(dt) => write!(f, "{dt}"),
            TemporalJiff::Zoned(z) => write!(f, "{z}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::extension::datetime::TemporalMetadata;
    use crate::extension::datetime::TimeUnit;

    /// Arrow allows a fixed UTC offset in place of an IANA name, and Iceberg maps every
    /// `timestamptz` column to `+00:00`.
    #[rstest]
    // A zero offset resolves to UTC, which `jiff` annotates by name.
    #[case("+00:00", "1970-01-01T00:00:01+00:00[UTC]")]
    #[case("-05:30", "1969-12-31T18:30:01-05:30[-05:30]")]
    #[case("+09:00", "1970-01-01T09:00:01+09:00[+09:00]")]
    #[case("America/New_York", "1969-12-31T19:00:01-05:00[America/New_York]")]
    fn to_jiff_resolves_timezone(
        #[case] timezone: &str,
        #[case] expected: &str,
    ) -> VortexResult<()> {
        let tz = Some(Arc::from(timezone));
        let metadata = TemporalMetadata::Timestamp(&TimeUnit::Seconds, &tz);
        assert_eq!(metadata.to_jiff(1)?.to_string(), expected);
        Ok(())
    }

    #[test]
    fn to_jiff_rejects_unknown_timezone() {
        let tz = Some(Arc::from("Not/A/Timezone"));
        let metadata = TemporalMetadata::Timestamp(&TimeUnit::Seconds, &tz);
        assert!(metadata.to_jiff(1).is_err());
    }
}
