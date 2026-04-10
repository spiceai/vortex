// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;

use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::array::ArrayView;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::extension::datetime::AnyTemporal;
use crate::extension::datetime::TemporalMetadata;
use crate::extension::datetime::TimeUnit;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Extension {
    fn cast(array: ArrayView<'_, Extension>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let DType::Extension(ext_dtype) = dtype else {
            return Ok(None);
        };

        if array.ext_dtype().eq_ignore_nullability(ext_dtype) {
            let new_storage = match array
                .storage_array()
                .cast(ext_dtype.storage_dtype().clone())
                .and_then(|a| a.to_canonical().map(|c| c.into_array()))
            {
                Ok(arr) => arr,
                Err(e) => {
                    tracing::warn!("Failed to cast storage array: {e}");
                    return Ok(None);
                }
            };

            return Ok(Some(
                ExtensionArray::new(ext_dtype.clone(), new_storage).into_array(),
            ));
        }

        if let Some(new_storage) = cast_temporal_date_to_timestamp(&array, dtype)? {
            return Ok(Some(
                ExtensionArray::new(ext_dtype.clone(), new_storage).into_array(),
            ));
        }

        Ok(None)
    }
}

fn cast_temporal_date_to_timestamp(
    array: &ArrayView<'_, Extension>,
    target_dtype: &DType,
) -> VortexResult<Option<ArrayRef>> {
    let DType::Extension(target_ext_dtype) = target_dtype else {
        return Ok(None);
    };

    let Some(source_temporal) = array.ext_dtype().metadata_opt::<AnyTemporal>() else {
        return Ok(None);
    };
    let Some(target_temporal) = target_ext_dtype.metadata_opt::<AnyTemporal>() else {
        return Ok(None);
    };

    let (TemporalMetadata::Date(source_unit), TemporalMetadata::Timestamp(target_unit, _)) =
        (source_temporal, target_temporal)
    else {
        return Ok(None);
    };

    let source_i64 = array
        .storage_array()
        .cast(DType::Primitive(PType::I64, array.dtype().nullability()))?;
    let source_i64 = source_i64.to_primitive();

    let converted = cast_date_values_to_timestamp(&source_i64, *source_unit, *target_unit)?;

    converted
        .into_array()
        .cast(target_ext_dtype.storage_dtype().clone())
        .map(Some)
}

fn cast_date_values_to_timestamp(
    values: &PrimitiveArray,
    source_unit: TimeUnit,
    target_unit: TimeUnit,
) -> VortexResult<PrimitiveArray> {
    let (multiply, divide) = date_to_timestamp_scale(source_unit, target_unit)?;

    let input = values.as_slice::<i64>();
    let mut output = BufferMut::with_capacity(input.len());
    match values.validity_mask()?.bit_buffer() {
        AllOr::All => {
            for &value in input {
                // SAFETY: output has sufficient capacity for all pushed values.
                unsafe { output.push_unchecked(convert_temporal_value(value, multiply, divide)?) };
            }
        }
        AllOr::None => {
            for _ in 0..input.len() {
                // SAFETY: output has sufficient capacity for all pushed values.
                unsafe { output.push_unchecked(0i64) };
            }
        }
        AllOr::Some(bits) => {
            for (&value, valid) in input.iter().zip(bits.iter()) {
                if valid {
                    // SAFETY: output has sufficient capacity for all pushed values.
                    unsafe {
                        output.push_unchecked(convert_temporal_value(value, multiply, divide)?)
                    };
                } else {
                    // SAFETY: output has sufficient capacity for all pushed values.
                    unsafe { output.push_unchecked(0i64) };
                }
            }
        }
    }

    Ok(PrimitiveArray::new(output.freeze(), values.validity()?))
}

fn date_to_timestamp_scale(
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

fn to_nanoseconds(unit: TimeUnit) -> VortexResult<i64> {
    match unit {
        TimeUnit::Nanoseconds => Ok(1),
        TimeUnit::Microseconds => Ok(1_000),
        TimeUnit::Milliseconds => Ok(1_000_000),
        TimeUnit::Seconds => Ok(1_000_000_000),
        TimeUnit::Days => Ok(86_400_000_000_000),
    }
}

fn convert_temporal_value(value: i64, multiply: i64, divide: i64) -> VortexResult<i64> {
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

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::Nullability;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;

    #[test]
    fn cast_same_ext_dtype() {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let storage = Buffer::<i64>::empty().into_array();

        let arr = ExtensionArray::new(ext_dtype.clone(), storage);

        let output = arr
            .clone()
            .into_array()
            .cast(DType::Extension(ext_dtype.clone()))
            .unwrap();
        assert_eq!(arr.len(), output.len());
        assert_eq!(arr.dtype(), output.dtype());
        assert_eq!(output.dtype(), &DType::Extension(ext_dtype));
    }

    #[test]
    fn cast_same_ext_dtype_differet_nullability() {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let storage = Buffer::<i64>::empty().into_array();

        let arr = ExtensionArray::new(ext_dtype.clone(), storage);
        assert!(!arr.dtype().is_nullable());

        let new_dtype = DType::Extension(ext_dtype).with_nullability(Nullability::Nullable);

        let output = arr.clone().into_array().cast(new_dtype.clone()).unwrap();
        assert_eq!(arr.len(), output.len());
        assert!(arr.dtype().eq_ignore_nullability(output.dtype()));
        assert_eq!(output.dtype(), &new_dtype);
    }

    #[test]
    fn cast_date_days_to_timestamp_nanoseconds() {
        let source_dtype = Date::new(TimeUnit::Days, Nullability::NonNullable).erased();
        let target_dtype = Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable).erased();

        let arr = ExtensionArray::new(source_dtype, buffer![0i32, 1, -1].into_array());
        let output = arr
            .into_array()
            .cast(DType::Extension(target_dtype.clone()))
            .unwrap()
            .to_extension();

        assert_eq!(output.dtype(), &DType::Extension(target_dtype));

        let storage = output.storage_array().to_primitive();
        assert_eq!(
            storage.as_slice::<i64>(),
            &[0, 86_400_000_000_000, -86_400_000_000_000]
        );
    }

    #[test]
    fn cast_date_days_to_timestamp_seconds_nullable() {
        let source_dtype = Date::new(TimeUnit::Days, Nullability::Nullable).erased();
        let target_dtype = Timestamp::new(TimeUnit::Seconds, Nullability::Nullable).erased();

        let arr = ExtensionArray::new(
            source_dtype,
            PrimitiveArray::from_option_iter([Some(0i32), None, Some(2)]).into_array(),
        );

        let output = arr
            .into_array()
            .cast(DType::Extension(target_dtype.clone()))
            .unwrap()
            .to_extension();

        assert_eq!(output.dtype(), &DType::Extension(target_dtype));

        let storage = output.storage_array().to_primitive();
        assert_eq!(
            storage.scalar_at(0).unwrap().as_primitive().as_::<i64>(),
            Some(0)
        );
        assert!(storage.scalar_at(1).unwrap().is_null());
        assert_eq!(
            storage.scalar_at(2).unwrap().as_primitive().as_::<i64>(),
            Some(172_800)
        );
    }

    #[test]
    fn cast_different_ext_dtype() {
        let original_dtype =
            Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        // Note NS here instead of MS
        let target_dtype = Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable).erased();

        let storage = buffer![1i64].into_array();
        let arr = ExtensionArray::new(original_dtype, storage);

        assert!(
            arr.into_array()
                .cast(DType::Extension(target_dtype))
                .and_then(|a| a.to_canonical().map(|c| c.into_array()))
                .is_err()
        );
    }

    #[rstest]
    #[case(create_timestamp_array(TimeUnit::Milliseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Microseconds, true))]
    #[case(create_timestamp_array(TimeUnit::Nanoseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Seconds, true))]
    fn test_cast_extension_conformance(#[case] array: ExtensionArray) {
        test_cast_conformance(&array.into_array());
    }

    fn create_timestamp_array(time_unit: TimeUnit, nullable: bool) -> ExtensionArray {
        let ext_dtype =
            Timestamp::new_with_tz(time_unit, Some("UTC".into()), nullable.into()).erased();

        let storage = if nullable {
            PrimitiveArray::from_option_iter([
                Some(1_000_000i64), // 1 second in microseconds
                None,
                Some(2_000_000),
                Some(3_000_000),
                None,
            ])
            .into_array()
        } else {
            buffer![1_000_000i64, 2_000_000, 3_000_000, 4_000_000, 5_000_000].into_array()
        };

        ExtensionArray::new(ext_dtype, storage)
    }
}
