// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
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
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::validity::Validity;

impl CastReduce for Extension {
    fn cast(array: ArrayView<'_, Extension>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let DType::Extension(ext_dtype) = dtype else {
            // Target is not an extension type.
            // Delegate to the storage array's cast.
            return Ok(Some(array.storage_array().cast(dtype.clone())?));
        };

        if array.ext_dtype().eq_ignore_nullability(ext_dtype) {
            // Same extension type: restructure by casting the storage array. This is buffer-free
            // (`cast` returns a lazy cast expression), so it stays in the reduce phase.
            let new_storage = match array
                .storage_array()
                .cast(ext_dtype.storage_dtype().clone())
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

        // A different extension type (e.g. Date -> Timestamp) requires per-value conversion that
        // reads buffers, so it is deferred to the execute phase in `CastKernel`, where an
        // `ExecutionCtx` is available.
        Ok(None)
    }
}

impl CastKernel for Extension {
    fn cast(
        array: ArrayView<'_, Extension>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let DType::Extension(ext_dtype) = dtype else {
            // Non-extension targets are restructured buffer-free by `CastReduce`.
            return Ok(None);
        };

        if let Some(new_storage) = cast_temporal_date_to_timestamp(&array, dtype, ctx)? {
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
    ctx: &mut ExecutionCtx,
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
        .cast(DType::Primitive(PType::I64, array.dtype().nullability()))?
        .execute::<PrimitiveArray>(ctx)?;

    let converted = cast_date_values_to_timestamp(&source_i64, *source_unit, *target_unit, ctx)?;

    let new_storage = converted
        .into_array()
        .cast(target_ext_dtype.storage_dtype().clone())?
        .execute::<PrimitiveArray>(ctx)?
        .into_array();

    Ok(Some(new_storage))
}

fn cast_date_values_to_timestamp(
    values: &PrimitiveArray,
    source_unit: TimeUnit,
    target_unit: TimeUnit,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let (multiply, divide) = date_to_timestamp_scale(source_unit, target_unit)?;

    let input = values.as_slice::<i64>();
    let validity = values.validity()?;
    let mut output = BufferMut::with_capacity(input.len());

    match &validity {
        Validity::NonNullable | Validity::AllValid => {
            for &value in input {
                // SAFETY: output has sufficient capacity for all pushed values.
                unsafe { output.push_unchecked(convert_temporal_value(value, multiply, divide)?) };
            }
        }
        Validity::AllInvalid => {
            for _ in 0..input.len() {
                // SAFETY: output has sufficient capacity for all pushed values.
                unsafe { output.push_unchecked(0i64) };
            }
        }
        Validity::Array(_) => {
            // Resolve validity to a boolean mask once. Null slots keep a placeholder 0 so a garbage
            // source value in a null slot cannot trip the overflow check in
            // `convert_temporal_value`; the output re-uses `validity`, so those slots stay null.
            let mask = validity.execute_mask(input.len(), ctx)?;
            for (i, &value) in input.iter().enumerate() {
                let converted = if mask.value(i) {
                    convert_temporal_value(value, multiply, divide)?
                } else {
                    0i64
                };
                // SAFETY: output has sufficient capacity for all pushed values.
                unsafe { output.push_unchecked(converted) };
            }
        }
    }

    Ok(PrimitiveArray::new(output.freeze(), validity))
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
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::executor::VortexSessionExecute;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

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
    fn cast_date_days_to_timestamp_nanoseconds() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source_dtype = Date::new(TimeUnit::Days, Nullability::NonNullable).erased();
        let target_dtype = Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable).erased();

        let arr = ExtensionArray::new(source_dtype, buffer![0i32, 1, -1].into_array());
        let output = arr
            .into_array()
            .cast(DType::Extension(target_dtype.clone()))?;

        let expected = ExtensionArray::new(
            target_dtype,
            buffer![0i64, 86_400_000_000_000, -86_400_000_000_000].into_array(),
        )
        .into_array();
        assert_arrays_eq!(output, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn cast_date_days_to_timestamp_seconds_nullable() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source_dtype = Date::new(TimeUnit::Days, Nullability::Nullable).erased();
        let target_dtype = Timestamp::new(TimeUnit::Seconds, Nullability::Nullable).erased();

        let arr = ExtensionArray::new(
            source_dtype,
            PrimitiveArray::from_option_iter([Some(0i32), None, Some(2)]).into_array(),
        );

        let output = arr
            .into_array()
            .cast(DType::Extension(target_dtype.clone()))?;

        let expected = ExtensionArray::new(
            target_dtype,
            PrimitiveArray::from_option_iter([Some(0i64), None, Some(172_800)]).into_array(),
        )
        .into_array();
        assert_arrays_eq!(output, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn cast_different_ext_dtype() {
        let original_dtype =
            Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        // Note NS here instead of MS
        let target_dtype = Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable).erased();

        let storage = buffer![1i64].into_array();
        let arr = ExtensionArray::new(original_dtype, storage);

        let result = arr
            .into_array()
            .cast(DType::Extension(target_dtype))
            .and_then(|a| {
                a.execute::<ExtensionArray>(&mut SESSION.create_execution_ctx())
                    .map(|c| c.into_array())
            });
        assert!(result.is_err());
    }

    #[test]
    fn cast_timestamp_to_i64() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ext_dtype = Timestamp::new_with_tz(
            TimeUnit::Nanoseconds,
            Some("UTC".into()),
            Nullability::NonNullable,
        )
        .erased();
        let storage = buffer![1i64, 2, 3].into_array();
        let arr = ExtensionArray::new(ext_dtype, storage).into_array();

        let result = arr.cast(DType::Primitive(PType::I64, Nullability::NonNullable))?;
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_arrays_eq!(result, buffer![1i64, 2, 3].into_array(), &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(create_timestamp_array(TimeUnit::Milliseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Microseconds, true))]
    #[case(create_timestamp_array(TimeUnit::Nanoseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Seconds, true))]
    fn test_cast_extension_conformance(#[case] array: ExtensionArray) {
        test_cast_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
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
