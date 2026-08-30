// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporal extension data types.

use std::fmt;
use std::sync::Arc;

use jiff::Span;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::resolve_timezone;
use crate::scalar::ScalarValue;

/// Timestamp DType.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Timestamp;

impl Timestamp {
    /// Creates a new Timestamp extension =dtype with the given time unit and nullability.
    pub fn new(time_unit: TimeUnit, nullability: Nullability) -> ExtDType<Self> {
        Self::new_with_tz(time_unit, None, nullability)
    }

    /// Creates a new Timestamp extension dtype with the given time unit, timezone, and nullability.
    pub fn new_with_tz(
        time_unit: TimeUnit,
        timezone: Option<Arc<str>>,
        nullability: Nullability,
    ) -> ExtDType<Self> {
        ExtDType::try_new(
            TimestampOptions {
                unit: time_unit,
                tz: timezone,
            },
            DType::Primitive(PType::I64, nullability),
        )
        .vortex_expect("failed to create timestamp dtype")
    }

    /// Creates a new `Timestamp` extension dtype with the given options and nullability.
    pub fn new_with_options(options: TimestampOptions, nullability: Nullability) -> ExtDType<Self> {
        ExtDType::try_new(options, DType::Primitive(PType::I64, nullability))
            .vortex_expect("failed to create timestamp dtype")
    }

    /// The inclusive range of storage values a `vortex.timestamp` in `unit` can hold.
    ///
    /// Taken from Jiff's own limits, and the one definition of them: `unpack_native` refuses a
    /// scalar outside this range and `DateToTimestamp` refuses to convert a date into one, so
    /// a converted date can always be carried by a scalar. A scan prunes a file on the scalar
    /// conversion and filters the rows it kept on the array one, so the two ranges parting
    /// company is a file dropped by a comparison the row filter would never have made.
    ///
    /// Nanoseconds are the whole of `i64`: its floor is 1677-09-21 and its ceiling 2262-04-11,
    /// both far inside Jiff's range, so the conversion below clamps rather than narrowing.
    pub(crate) fn storage_range(unit: TimeUnit) -> VortexResult<(i64, i64)> {
        Ok(match unit {
            TimeUnit::Nanoseconds => (
                i64::try_from(jiff::Timestamp::MIN.as_nanosecond()).unwrap_or(i64::MIN),
                i64::try_from(jiff::Timestamp::MAX.as_nanosecond()).unwrap_or(i64::MAX),
            ),
            TimeUnit::Microseconds => (
                jiff::Timestamp::MIN.as_microsecond(),
                jiff::Timestamp::MAX.as_microsecond(),
            ),
            TimeUnit::Milliseconds => (
                jiff::Timestamp::MIN.as_millisecond(),
                jiff::Timestamp::MAX.as_millisecond(),
            ),
            TimeUnit::Seconds => (
                jiff::Timestamp::MIN.as_second(),
                jiff::Timestamp::MAX.as_second(),
            ),
            TimeUnit::Days => vortex_bail!("Timestamp does not support Days time unit"),
        })
    }
}

/// Options for the Timestamp DType.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TimestampOptions {
    /// The time unit of the timestamp.
    pub unit: TimeUnit,
    /// The timezone of the timestamp, if any.
    pub tz: Option<Arc<str>>,
}

impl fmt::Display for TimestampOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.tz {
            Some(tz) => write!(f, "{}, tz={}", self.unit, tz),
            None => write!(f, "{}", self.unit),
        }
    }
}

/// Unpacked value of a [`Timestamp`] extension scalar.
///
/// Each variant carries the raw storage value and an optional timezone.
pub enum TimestampValue<'a> {
    /// Seconds since the Unix epoch.
    Seconds(i64, Option<&'a Arc<str>>),
    /// Milliseconds since the Unix epoch.
    Milliseconds(i64, Option<&'a Arc<str>>),
    /// Microseconds since the Unix epoch.
    Microseconds(i64, Option<&'a Arc<str>>),
    /// Nanoseconds since the Unix epoch.
    Nanoseconds(i64, Option<&'a Arc<str>>),
}

impl fmt::Display for TimestampValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (span, tz) = match self {
            TimestampValue::Seconds(v, tz) => (Span::new().seconds(*v), *tz),
            TimestampValue::Milliseconds(v, tz) => (Span::new().milliseconds(*v), *tz),
            TimestampValue::Microseconds(v, tz) => (Span::new().microseconds(*v), *tz),
            TimestampValue::Nanoseconds(v, tz) => (Span::new().nanoseconds(*v), *tz),
        };
        let ts = jiff::Timestamp::UNIX_EPOCH + span;

        match tz {
            None => write!(f, "{ts}"),
            // A timezone that does not resolve must not abort a `Display` impl, which has no way to
            // report the failure. Render the underlying UTC timestamp instead.
            Some(tz) => match resolve_timezone(tz.as_ref()) {
                Ok(zone) => write!(f, "{}", ts.to_zoned(zone)),
                Err(_) => write!(f, "{ts}"),
            },
        }
    }
}

impl ExtVTable for Timestamp {
    type Metadata = TimestampOptions;

    type NativeValue<'a> = TimestampValue<'a>;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.timestamp");
        *ID
    }

    // NOTE(ngates): unfortunately we're stuck with this hand-rolled serialization format for
    //  backwards compatibility.
    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        let mut bytes = Vec::with_capacity(4);
        let unit_tag: u8 = metadata.unit.into();

        bytes.push(unit_tag);

        // Encode time_zone as u16 length followed by utf8 bytes.
        match &metadata.tz {
            None => bytes.extend_from_slice(0u16.to_le_bytes().as_slice()),
            Some(tz) => {
                let tz_bytes = tz.as_bytes();
                let tz_len = u16::try_from(tz_bytes.len())
                    .unwrap_or_else(|err| vortex_panic!("tz did not fit in u16: {}", err));
                bytes.extend_from_slice(tz_len.to_le_bytes().as_slice());
                bytes.extend_from_slice(tz_bytes);
            }
        }

        Ok(bytes)
    }

    fn deserialize_metadata(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        vortex_ensure!(
            data.len() >= 3,
            "Timestamp metadata must have at least 3 bytes, got {}",
            data.len()
        );

        let tag = data[0];
        let time_unit = TimeUnit::try_from(tag)?;
        let tz_len_bytes: [u8; 2] = data[1..3]
            .try_into()
            .ok()
            .vortex_expect("Verified to have two bytes");
        let tz_len = u16::from_le_bytes(tz_len_bytes) as usize;
        if tz_len == 0 {
            return Ok(TimestampOptions {
                unit: time_unit,
                tz: None,
            });
        }

        // Attempt to load from len-prefixed bytes
        vortex_ensure!(
            data.len() >= 3 + tz_len,
            "Timestamp metadata is truncated: declared timezone length {} but only {} bytes available",
            tz_len,
            data.len() - 3
        );
        let tz_bytes = &data[3..3 + tz_len];
        let tz: Arc<str> = str::from_utf8(tz_bytes)
            .map_err(|e| vortex_err!("timezone is not valid utf8 string: {e}"))?
            .to_string()
            .into();
        Ok(TimestampOptions {
            unit: time_unit,
            tz: Some(tz),
        })
    }

    fn can_coerce_from(ext_dtype: &ExtDType<Self>, other: &DType) -> bool {
        let DType::Extension(other_ext) = other else {
            return false;
        };
        let Some(other_opts) = other_ext.metadata_opt::<Timestamp>() else {
            return false;
        };
        let our_opts = ext_dtype.metadata();
        our_opts.tz == other_opts.tz
            && our_opts.unit <= other_opts.unit
            && (ext_dtype.storage_dtype().is_nullable() || !other.is_nullable())
    }

    fn least_supertype(ext_dtype: &ExtDType<Self>, other: &DType) -> Option<DType> {
        let DType::Extension(other_ext) = other else {
            return None;
        };
        let other_opts = other_ext.metadata_opt::<Timestamp>()?;
        let our_opts = ext_dtype.metadata();
        if our_opts.tz != other_opts.tz {
            return None;
        }
        let finest = our_opts.unit.min(other_opts.unit);
        let union_null = ext_dtype.storage_dtype().nullability() | other.nullability();
        Some(DType::Extension(
            Timestamp::new_with_tz(finest, our_opts.tz.clone(), union_null).erased(),
        ))
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure!(
            matches!(ext_dtype.storage_dtype(), DType::Primitive(PType::I64, _)),
            "Timestamp storage dtype must be i64"
        );
        Ok(())
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let metadata = ext_dtype.metadata();
        let ts_value = storage_value.as_primitive().cast::<i64>()?;
        let tz = metadata.tz.as_ref();

        let value = match metadata.unit {
            TimeUnit::Nanoseconds => TimestampValue::Nanoseconds(ts_value, tz),
            TimeUnit::Microseconds => TimestampValue::Microseconds(ts_value, tz),
            TimeUnit::Milliseconds => TimestampValue::Milliseconds(ts_value, tz),
            TimeUnit::Seconds => TimestampValue::Seconds(ts_value, tz),
            TimeUnit::Days => vortex_bail!("Timestamp does not support Days time unit"),
        };

        // Compare against the timestamp's own range rather than routing the value through a
        // Jiff `Span`. A span's limits are not a timestamp's at either end: they stop one
        // short of `i64::MIN` nanoseconds, which is a valid 1677 instant, and they run past
        // the last instant everywhere else, where the unchecked constructors abort rather
        // than report. This range is also what `DateToTimestamp` converts into, so a scalar
        // accepts exactly the values the array kernel produces.
        let (min, max) = Self::storage_range(metadata.unit)?;
        vortex_ensure!(
            (min..=max).contains(&ts_value),
            "Invalid timestamp scalar: {ts_value} {} is outside the instants a timestamp can represent ({min}..={max})",
            metadata.unit
        );

        // Validate the timezone resolves, accepting both IANA names and fixed UTC offsets.
        if let Some(tz) = tz {
            resolve_timezone(tz.as_ref())
                .map_err(|e| vortex_err!("Invalid timezone for timestamp scalar: {}", e))?;
        }

        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability::Nullable;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::extension::datetime::TimestampValue;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    #[test]
    fn validate_timestamp_scalar() -> VortexResult<()> {
        let dtype = DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullable).erased());
        Scalar::try_new(dtype, Some(ScalarValue::Primitive(PValue::I64(0))))?;

        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn reject_timestamp_with_invalid_timezone() {
        let dtype = DType::Extension(
            Timestamp::new_with_tz(
                TimeUnit::Seconds,
                Some(Arc::from("Not/A/Timezone")),
                Nullable,
            )
            .erased(),
        );
        let result = Scalar::try_new(dtype, Some(ScalarValue::Primitive(PValue::I64(0))));
        assert!(result.is_err());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn display_timestamp_scalar() {
        // Local (no timezone) timestamp.
        let local_dtype = DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullable).erased());
        let scalar = Scalar::new(local_dtype, Some(ScalarValue::Primitive(PValue::I64(0))));
        assert_eq!(format!("{}", scalar.as_extension()), "1970-01-01T00:00:00Z");

        // Zoned timestamp.
        let zoned_dtype = DType::Extension(
            Timestamp::new_with_tz(
                TimeUnit::Seconds,
                Some(Arc::from("America/New_York")),
                Nullable,
            )
            .erased(),
        );
        let scalar = Scalar::new(zoned_dtype, Some(ScalarValue::Primitive(PValue::I64(0))));
        assert_eq!(
            format!("{}", scalar.as_extension()),
            "1969-12-31T19:00:00-05:00[America/New_York]"
        );
    }

    #[test]
    fn least_supertype_timestamp_units() {
        use crate::dtype::Nullability::NonNullable;

        let secs = DType::Extension(Timestamp::new(TimeUnit::Seconds, NonNullable).erased());
        let ns = DType::Extension(Timestamp::new(TimeUnit::Nanoseconds, NonNullable).erased());
        let expected =
            DType::Extension(Timestamp::new(TimeUnit::Nanoseconds, NonNullable).erased());
        assert_eq!(secs.least_supertype(&ns).unwrap(), expected);
        assert_eq!(ns.least_supertype(&secs).unwrap(), expected);
    }

    #[test]
    fn least_supertype_timestamp_tz_mismatch() {
        use crate::dtype::Nullability::NonNullable;

        let utc = DType::Extension(
            Timestamp::new_with_tz(TimeUnit::Seconds, Some(Arc::from("UTC")), NonNullable).erased(),
        );
        let none = DType::Extension(Timestamp::new(TimeUnit::Seconds, NonNullable).erased());
        assert!(utc.least_supertype(&none).is_none());
    }

    #[test]
    fn least_supertype_timestamp_same_tz() {
        use crate::dtype::Nullability::NonNullable;

        let utc_s = DType::Extension(
            Timestamp::new_with_tz(TimeUnit::Seconds, Some(Arc::from("UTC")), NonNullable).erased(),
        );
        let utc_ns = DType::Extension(
            Timestamp::new_with_tz(TimeUnit::Nanoseconds, Some(Arc::from("UTC")), NonNullable)
                .erased(),
        );
        let expected = DType::Extension(
            Timestamp::new_with_tz(TimeUnit::Nanoseconds, Some(Arc::from("UTC")), NonNullable)
                .erased(),
        );
        assert_eq!(utc_s.least_supertype(&utc_ns).unwrap(), expected);
    }

    #[test]
    fn can_coerce_from_timestamp_tz() {
        use crate::dtype::Nullability::NonNullable;

        let utc = DType::Extension(
            Timestamp::new_with_tz(TimeUnit::Nanoseconds, Some(Arc::from("UTC")), NonNullable)
                .erased(),
        );
        let utc_s = DType::Extension(
            Timestamp::new_with_tz(TimeUnit::Seconds, Some(Arc::from("UTC")), NonNullable).erased(),
        );
        let none = DType::Extension(Timestamp::new(TimeUnit::Nanoseconds, NonNullable).erased());
        assert!(utc.can_coerce_from(&utc_s));
        assert!(!utc.can_coerce_from(&none));
    }

    #[test]
    fn deserialize_empty_metadata_returns_error() {
        use crate::dtype::extension::ExtVTable;

        let vtable = Timestamp;
        assert!(vtable.deserialize_metadata(&[]).is_err());
    }

    #[test]
    fn deserialize_too_short_metadata_returns_error() {
        use crate::dtype::extension::ExtVTable;

        let vtable = Timestamp;
        // Only 2 bytes - too short for the required 3-byte header.
        assert!(vtable.deserialize_metadata(&[0x00, 0x01]).is_err());
    }

    #[test]
    fn deserialize_truncated_timezone_returns_error() {
        use crate::dtype::extension::ExtVTable;

        let vtable = Timestamp;
        // Valid tag (0x00 = Nanoseconds), tz_len = 10 (little-endian: [0x0A, 0x00]),
        // but only 3 bytes of timezone data instead of the declared 10.
        let data = [0x00u8, 0x0A, 0x00, b'U', b'T', b'C'];
        assert!(vtable.deserialize_metadata(&data).is_err());
    }

    /// A fixed UTC offset renders at that offset. Iceberg emits `+00:00` for every `timestamptz`.
    #[test]
    fn display_renders_fixed_offset() {
        let utc = Arc::from("+00:00");
        assert_eq!(
            TimestampValue::Seconds(1, Some(&utc)).to_string(),
            "1970-01-01T00:00:01+00:00[UTC]"
        );

        let tokyo = Arc::from("+09:00");
        assert_eq!(
            TimestampValue::Seconds(1, Some(&tokyo)).to_string(),
            "1970-01-01T09:00:01+09:00[+09:00]"
        );
    }

    /// `Display` has no way to report a failure, so a timezone that does not resolve falls back to
    /// rendering the underlying UTC timestamp rather than aborting the process.
    #[test]
    fn display_falls_back_on_unresolvable_timezone() {
        let bad = Arc::from("Not/A/Timezone");
        let unzoned = TimestampValue::Seconds(1, None).to_string();
        assert_eq!(TimestampValue::Seconds(1, Some(&bad)).to_string(), unzoned);
    }

    /// A storage value outside the timestamp range has to be reported, not panicked on.
    ///
    /// The hazard is Jiff's `Span`, whose limits are wider than a timestamp's and whose
    /// unchecked constructors abort outside them: validating a storage value by building a
    /// span from it takes the process down on the integer extrema below. `storage_range` is
    /// total over `i64`, so every value here is reported. Each is one a `vortex.timestamp`
    /// array can hold, so any read of one reaches this path.
    ///
    /// Nanoseconds are absent because every `i64` is a valid nanosecond timestamp — see
    /// `every_i64_is_a_valid_nanosecond_timestamp`.
    #[rstest::rstest]
    #[case(TimeUnit::Seconds, i64::MAX)]
    #[case(TimeUnit::Seconds, i64::MIN)]
    #[case(TimeUnit::Milliseconds, i64::MAX)]
    #[case(TimeUnit::Microseconds, i64::MAX)]
    // Just past the last instant, and well short of the span limit — the half of the range
    // that a span reports on, pinned alongside the half it aborts on.
    #[case(TimeUnit::Seconds, 253_402_300_800)]
    fn an_out_of_range_storage_value_is_an_error_not_a_panic(
        #[case] unit: TimeUnit,
        #[case] storage_value: i64,
    ) {
        let dtype = DType::Extension(Timestamp::new(unit, Nullable).erased());
        let err = Scalar::try_new(dtype, Some(storage_value.into()))
            .expect_err("a value past the last instant is not a timestamp");
        assert!(
            err.to_string().contains("Invalid timestamp scalar"),
            "the error has to name the problem, got: {err}"
        );
    }

    /// Every `i64` is a valid nanosecond timestamp, its floor included.
    ///
    /// `i64::MIN` nanoseconds is 1677-09-21T00:12:43.145224192Z and `i64::MAX` is
    /// 2262-04-11T23:47:16.854775807Z, both far inside Jiff's range. A Jiff `Span` stops one
    /// short of `i64::MIN`, so validating through one refuses a value that a
    /// `vortex.timestamp[ns]` array holds and that `DateToTimestamp` converts into — the
    /// scalar and the array disagreeing at exactly the boundary this module exists to align.
    #[rstest::rstest]
    #[case(i64::MIN)]
    #[case(i64::MIN + 1)]
    #[case(0)]
    #[case(i64::MAX)]
    fn every_i64_is_a_valid_nanosecond_timestamp(#[case] storage_value: i64) {
        let dtype = DType::Extension(Timestamp::new(TimeUnit::Nanoseconds, Nullable).erased());
        Scalar::try_new(dtype, Some(storage_value.into()))
            .expect("every i64 nanosecond count is an instant a timestamp can represent");
    }

    /// The range a date converts into is exactly the range a scalar accepts.
    ///
    /// `DateToTimestamp` refuses to produce a value outside `Timestamp::storage_range` so that
    /// a converted date can always be carried by a scalar. That only holds while this is the
    /// same range the scalar itself enforces, so pin the two together at both ends — a scan
    /// prunes a file on the scalar conversion and filters the rows it kept on the array one.
    #[rstest::rstest]
    #[case(TimeUnit::Seconds)]
    #[case(TimeUnit::Milliseconds)]
    #[case(TimeUnit::Microseconds)]
    #[case(TimeUnit::Nanoseconds)]
    fn the_storage_range_is_exactly_what_a_scalar_accepts(#[case] unit: TimeUnit) {
        let (min, max) = Timestamp::storage_range(unit).expect("a range for every stored unit");
        let scalar = |v: i64| {
            Scalar::try_new(
                DType::Extension(Timestamp::new(unit, Nullable).erased()),
                Some(v.into()),
            )
        };

        assert!(
            scalar(min).is_ok(),
            "{unit}: the floor of the range is an instant"
        );
        assert!(
            scalar(max).is_ok(),
            "{unit}: the ceiling of the range is an instant"
        );
        // Guarded, because the nanosecond range is the whole of `i64` and has no outside.
        if min > i64::MIN {
            assert!(
                scalar(min - 1).is_err(),
                "{unit}: one below the floor is not"
            );
        }
        if max < i64::MAX {
            assert!(
                scalar(max + 1).is_err(),
                "{unit}: one above the ceiling is not"
            );
        }
    }

    /// A timestamp is never stored in days, so it has no range of storage values.
    #[test]
    fn a_timestamp_in_days_has_no_storage_range() {
        assert!(Timestamp::storage_range(TimeUnit::Days).is_err());
    }
}
