// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::primitive::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::NativePType;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Decimal {
    fn cast(array: ArrayView<'_, Decimal>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Only nullability changes within the same decimal dtype are reducible without execution.
        // Precision/scale changes need the kernel.
        let DType::Decimal(to_decimal_dtype, to_nullability) = dtype else {
            return Ok(None);
        };
        let DType::Decimal(from_decimal_dtype, _) = array.dtype() else {
            vortex_panic!(
                "DecimalArray must have decimal dtype, got {:?}",
                array.dtype()
            );
        };

        if from_decimal_dtype != to_decimal_dtype {
            return Ok(None);
        }

        let Some(new_validity) = array
            .validity()?
            .trivial_cast_nullability(*to_nullability, array.len())?
        else {
            return Ok(None);
        };

        // SAFETY: validity has the same length, only its nullability tag changes.
        unsafe {
            Ok(Some(
                DecimalArray::new_unchecked_handle(
                    array.buffer_handle().clone(),
                    array.values_type(),
                    *to_decimal_dtype,
                    new_validity,
                )
                .into_array(),
            ))
        }
    }
}

impl CastKernel for Decimal {
    fn cast(
        array: ArrayView<'_, Decimal>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let DType::Decimal(from_decimal_dtype, _) = array.dtype() else {
            vortex_panic!(
                "DecimalArray must have decimal dtype, got {:?}",
                array.dtype()
            );
        };

        if let DType::Primitive(to_ptype, to_nullability) = dtype {
            let validity = array.validity()?;
            let new_validity =
                validity
                    .clone()
                    .cast_nullability(*to_nullability, array.len(), ctx)?;
            let mask = validity.execute_mask(array.len(), ctx)?;

            return Ok(Some(match_each_native_ptype!(*to_ptype, |T| {
                match_each_decimal_value_type!(array.values_type(), |F| {
                    PrimitiveArray::new(
                        cast_decimal_buffer_to_primitive::<F, T>(
                            array.buffer::<F>(),
                            from_decimal_dtype.scale(),
                            mask,
                        )?,
                        new_validity,
                    )
                    .into_array()
                })
            })));
        }

        let DType::Decimal(to_decimal_dtype, to_nullability) = dtype else {
            return Ok(None);
        };

        // Scale changes are not yet supported
        if from_decimal_dtype.scale() != to_decimal_dtype.scale() {
            vortex_bail!(
                "Casting decimal with scale {} to scale {} not yet implemented",
                from_decimal_dtype.scale(),
                to_decimal_dtype.scale()
            );
        }

        // Downcasting precision is not yet supported
        if to_decimal_dtype.precision() < from_decimal_dtype.precision() {
            vortex_bail!(
                "Downcasting decimal from precision {} to {} not yet implemented",
                from_decimal_dtype.precision(),
                to_decimal_dtype.precision()
            );
        }

        // If the dtype is exactly the same, return self
        if array.dtype() == dtype {
            return Ok(Some(array.array().clone()));
        }

        // Cast the validity to the new nullability
        let new_validity = array
            .validity()?
            .cast_nullability(*to_nullability, array.len(), ctx)?;

        // If the target needs a wider physical type, upcast the values
        let target_values_type = DecimalType::smallest_decimal_value_type(to_decimal_dtype);
        let array = if target_values_type > array.values_type() {
            upcast_decimal_values(array, target_values_type)?
        } else {
            array.array().as_::<Decimal>().into_owned()
        };

        // SAFETY: new_validity same length as previous validity, just cast
        unsafe {
            Ok(Some(
                DecimalArray::new_unchecked_handle(
                    array.buffer_handle().clone(),
                    array.values_type(),
                    *to_decimal_dtype,
                    new_validity,
                )
                .into_array(),
            ))
        }
    }
}

/// Upcast a DecimalArray to a wider physical representation (e.g., i32 -> i64) while keeping
/// the same precision and scale.
///
/// This is useful when you need to widen the underlying storage type to accommodate operations
/// that might overflow the current representation, or to match the physical type expected by
/// downstream consumers.
///
/// # Errors
///
/// Returns an error if `to_values_type` is narrower than the array's current values type.
/// Only upcasting (widening) is supported.
pub fn upcast_decimal_values(
    array: ArrayView<'_, Decimal>,
    to_values_type: DecimalType,
) -> VortexResult<DecimalArray> {
    let from_values_type = array.values_type();

    // If already the target type, just clone
    if from_values_type == to_values_type {
        return Ok(array.array().as_::<Decimal>().into_owned());
    }

    // Only allow upcasting (widening)
    if to_values_type < from_values_type {
        vortex_bail!(
            "Cannot downcast decimal values from {:?} to {:?}. Only upcasting is supported.",
            from_values_type,
            to_values_type
        );
    }

    let decimal_dtype = array.decimal_dtype();
    let validity = array.validity()?;

    // Use match_each_decimal_value_type to dispatch based on source and target types
    match_each_decimal_value_type!(from_values_type, |F| {
        let from_buffer = array.buffer::<F>();
        match_each_decimal_value_type!(to_values_type, |T| {
            let to_buffer = upcast_decimal_buffer::<F, T>(from_buffer);
            Ok(DecimalArray::new(to_buffer, decimal_dtype, validity))
        })
    })
}

/// Upcast a buffer of decimal values from type F to type T.
/// Since T is wider than F, this conversion never fails.
fn upcast_decimal_buffer<F: NativeDecimalType, T: NativeDecimalType>(from: Buffer<F>) -> Buffer<T> {
    from.iter()
        .map(|&v| T::from(v).vortex_expect("upcast should never fail"))
        .collect()
}

fn cast_decimal_buffer_to_primitive<F, T>(
    from: Buffer<F>,
    scale: i8,
    mask: Mask,
) -> VortexResult<Buffer<T>>
where
    F: NativeDecimalType,
    T: NativePType,
{
    let scale_factor = 10_f64.powi(i32::from(scale));

    match mask.bit_buffer() {
        AllOr::All => {
            let mut buffer = BufferMut::<T>::with_capacity(from.len());
            for value in from {
                let value = cast_decimal_value_to_primitive::<F, T>(value, scale_factor)?;
                buffer.push(value);
            }
            Ok(buffer.freeze())
        }
        AllOr::None => Ok(Buffer::zeroed(from.len())),
        AllOr::Some(validity) => {
            let mut buffer = BufferMut::<T>::with_capacity(from.len());
            for (value, valid) in from.iter().zip(validity.iter()) {
                if valid {
                    let value = cast_decimal_value_to_primitive::<F, T>(*value, scale_factor)?;
                    buffer.push(value);
                } else {
                    buffer.push(T::default());
                }
            }
            Ok(buffer.freeze())
        }
    }
}

fn cast_decimal_value_to_primitive<F, T>(value: F, scale_factor: f64) -> VortexResult<T>
where
    F: NativeDecimalType,
    T: NativePType,
{
    let value = value
        .to_f64()
        .ok_or_else(|| vortex_err!(Compute: "Failed to cast decimal value {value} to f64"))?
        / scale_factor;

    T::from(value).ok_or_else(
        || vortex_err!(Compute: "Failed to cast decimal value {value} to {:?}", T::PTYPE),
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use super::upcast_decimal_values;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::DecimalArray;
    use crate::builtins::ArrayBuiltins;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    #[test]
    fn cast_decimal_to_nullable() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::new(
            buffer![100i32, 200, 300],
            decimal_dtype,
            Validity::NonNullable,
        );

        // Cast to nullable
        let nullable_dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        #[expect(deprecated)]
        let casted = array
            .into_array()
            .cast(nullable_dtype.clone())
            .unwrap()
            .to_decimal();

        assert_eq!(casted.dtype(), &nullable_dtype);
        assert!(matches!(casted.validity(), Ok(Validity::AllValid)));
        assert_eq!(casted.len(), 3);
    }

    #[test]
    fn cast_nullable_to_non_nullable() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with no nulls
        let array = DecimalArray::new(buffer![100i32, 200, 300], decimal_dtype, Validity::AllValid);

        // Cast to non-nullable
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        #[expect(deprecated)]
        let casted = array
            .into_array()
            .cast(non_nullable_dtype.clone())
            .unwrap()
            .to_decimal();

        assert_eq!(casted.dtype(), &non_nullable_dtype);
        assert!(matches!(casted.validity(), Ok(Validity::NonNullable)));
    }

    #[test]
    #[should_panic(expected = "Cannot cast array with invalid values to non-nullable type")]
    fn cast_nullable_with_nulls_to_non_nullable_fails() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with nulls
        let array = DecimalArray::from_option_iter([Some(100i32), None, Some(300)], decimal_dtype);

        // Attempt to cast to non-nullable should fail
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        #[expect(deprecated)]
        let result = array
            .into_array()
            .cast(non_nullable_dtype)
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));
        result.unwrap();
    }

    #[test]
    fn cast_different_scale_fails() {
        let array = DecimalArray::new(
            buffer![100i32],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );

        // Try to cast to different scale - not supported
        let different_dtype = DType::Decimal(DecimalDType::new(15, 3), Nullability::NonNullable);
        #[expect(deprecated)]
        let result = array
            .into_array()
            .cast(different_dtype)
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Casting decimal with scale 2 to scale 3 not yet implemented")
        );
    }

    #[test]
    fn cast_downcast_precision_fails() {
        let array = DecimalArray::new(
            buffer![100i64],
            DecimalDType::new(18, 2),
            Validity::NonNullable,
        );

        // Try to downcast precision - not supported
        let smaller_dtype = DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable);
        #[expect(deprecated)]
        let result = array
            .into_array()
            .cast(smaller_dtype)
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Downcasting decimal from precision 18 to 10 not yet implemented")
        );
    }

    #[test]
    fn cast_upcast_precision_succeeds() {
        let array = DecimalArray::new(
            buffer![100i32, 200, 300],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );

        // Cast to higher precision with same scale - should succeed
        let wider_dtype = DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable);
        #[expect(deprecated)]
        let casted = array.into_array().cast(wider_dtype).unwrap().to_decimal();

        assert_eq!(casted.precision(), 38);
        assert_eq!(casted.scale(), 2);
        assert_eq!(casted.len(), 3);
        // Should be stored in i128 now (precision 38 requires i128)
        assert_eq!(casted.values_type(), DecimalType::I128);
    }

    #[test]
    fn cast_decimal_to_f64_applies_scale() {
        let array = DecimalArray::new(
            buffer![12345i64, -50, 0],
            DecimalDType::new(15, 2),
            Validity::NonNullable,
        );
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);

        #[expect(deprecated)]
        let casted = array
            .into_array()
            .cast(dtype.clone())
            .unwrap()
            .to_primitive();

        assert_eq!(casted.as_ref().dtype(), &dtype);
        assert!(matches!(
            casted.as_ref().validity(),
            Ok(Validity::NonNullable)
        ));
        let values = casted.as_slice::<f64>();
        assert!((values[0] - 123.45).abs() < 0.000000000001);
        assert_eq!(values[1], -0.5);
        assert_eq!(values[2], 0.0);
    }

    #[test]
    fn cast_nullable_decimal_to_nullable_f64_preserves_validity() {
        let array = DecimalArray::from_option_iter(
            [Some(12345i64), None, Some(-50)],
            DecimalDType::new(15, 2),
        );
        let dtype = DType::Primitive(PType::F64, Nullability::Nullable);

        #[expect(deprecated)]
        let casted = array
            .into_array()
            .cast(dtype.clone())
            .unwrap()
            .to_primitive();

        assert_eq!(casted.as_ref().dtype(), &dtype);
        let mask = casted
            .as_ref()
            .validity()
            .unwrap()
            .execute_mask(casted.len(), &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        assert!(mask.value(0));
        assert!(!mask.value(1));
        assert!(mask.value(2));
        let values = casted.as_slice::<f64>();
        assert!((values[0] - 123.45).abs() < 0.000000000001);
        assert_eq!(values[2], -0.5);
    }

    #[test]
    fn cast_to_non_decimal_returns_err() {
        let array = DecimalArray::new(
            buffer![100i32],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );

        // Try to cast to non-decimal type - should fail since no kernel can handle it
        #[expect(deprecated)]
        let result = array
            .into_array()
            .cast(DType::Utf8(Nullability::NonNullable))
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No CastKernel to cast canonical array")
        );
    }

    #[rstest]
    #[case(DecimalArray::new(buffer![100i32, 200, 300], DecimalDType::new(10, 2), Validity::NonNullable))]
    #[case(DecimalArray::new(buffer![10000i64, 20000, 30000], DecimalDType::new(18, 4), Validity::NonNullable))]
    #[case(DecimalArray::from_option_iter([Some(100i32), None, Some(300)], DecimalDType::new(10, 2)))]
    #[case(DecimalArray::new(buffer![42i32], DecimalDType::new(5, 1), Validity::NonNullable))]
    fn test_cast_decimal_conformance(#[case] array: DecimalArray) {
        test_cast_conformance(&array.into_array());
    }

    #[test]
    fn upcast_decimal_values_i32_to_i64() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::new(
            buffer![100i32, 200, 300],
            decimal_dtype,
            Validity::NonNullable,
        );

        assert_eq!(array.values_type(), DecimalType::I32);

        let array = array.as_view();
        let casted = upcast_decimal_values(array, DecimalType::I64).unwrap();

        assert_eq!(casted.values_type(), DecimalType::I64);
        assert_eq!(casted.decimal_dtype(), decimal_dtype);
        assert_eq!(casted.len(), 3);

        // Verify values are preserved
        let buffer = casted.buffer::<i64>();
        assert_eq!(buffer.as_ref(), &[100i64, 200, 300]);
    }

    #[test]
    fn upcast_decimal_values_i64_to_i128() {
        let decimal_dtype = DecimalDType::new(18, 4);
        let array = DecimalArray::new(
            buffer![10000i64, 20000, 30000],
            decimal_dtype,
            Validity::NonNullable,
        );

        let array = array.as_view();
        let casted = upcast_decimal_values(array, DecimalType::I128).unwrap();

        assert_eq!(casted.values_type(), DecimalType::I128);
        assert_eq!(casted.decimal_dtype(), decimal_dtype);

        let buffer = casted.buffer::<i128>();
        assert_eq!(buffer.as_ref(), &[10000i128, 20000, 30000]);
    }

    #[test]
    fn upcast_decimal_values_same_type_returns_clone() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::new(
            buffer![100i32, 200, 300],
            decimal_dtype,
            Validity::NonNullable,
        );

        let array = array.as_view();
        let casted = upcast_decimal_values(array, DecimalType::I32).unwrap();

        assert_eq!(casted.values_type(), DecimalType::I32);
        assert_eq!(casted.decimal_dtype(), decimal_dtype);
    }

    #[test]
    fn upcast_decimal_values_with_nulls() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::from_option_iter([Some(100i32), None, Some(300)], decimal_dtype);

        let array = array.as_view();
        let casted = upcast_decimal_values(array, DecimalType::I64).unwrap();

        assert_eq!(casted.values_type(), DecimalType::I64);
        assert_eq!(casted.len(), 3);

        // Check validity is preserved
        let mask = casted
            .as_ref()
            .validity()
            .unwrap()
            .execute_mask(
                casted.as_ref().len(),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap();
        assert!(mask.value(0));
        assert!(!mask.value(1));
        assert!(mask.value(2));

        // Check non-null values
        let buffer = casted.buffer::<i64>();
        assert_eq!(buffer[0], 100);
        assert_eq!(buffer[2], 300);
    }

    #[test]
    fn upcast_decimal_values_downcast_fails() {
        let decimal_dtype = DecimalDType::new(18, 4);
        let array = DecimalArray::new(
            buffer![10000i64, 20000, 30000],
            decimal_dtype,
            Validity::NonNullable,
        );

        // Attempt to downcast from i64 to i32 should fail
        let array = array.as_view();
        let result = upcast_decimal_values(array, DecimalType::I32);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot downcast decimal values")
        );
    }
}
