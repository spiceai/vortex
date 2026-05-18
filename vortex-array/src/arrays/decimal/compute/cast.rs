// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::DType;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_decimal_value_type;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::arrays::primitive::PrimitiveArray;
use crate::compute::CastKernel;
use crate::compute::CastKernelAdapter;
use crate::register_kernel;
use crate::stats::ArrayStats;
use crate::vtable::ValidityHelper;

impl CastKernel for DecimalVTable {
    fn cast(&self, array: &DecimalArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let DType::Decimal(from_precision_scale, _) = array.dtype() else {
            vortex_panic!(
                "DecimalArray must have decimal dtype, got {:?}",
                array.dtype()
            );
        };

        if let DType::Primitive(to_ptype, to_nullability) = dtype {
            let new_validity = array
                .validity()
                .clone()
                .cast_nullability(*to_nullability, array.len())?;
            let mask = array.validity_mask();

            return Ok(Some(match_each_native_ptype!(*to_ptype, |T| {
                match_each_decimal_value_type!(array.values_type(), |F| {
                    PrimitiveArray::new(
                        cast_decimal_buffer_to_primitive::<F, T>(
                            array.buffer::<F>(),
                            from_precision_scale.scale(),
                            mask,
                        )?,
                        new_validity,
                    )
                    .into_array()
                })
            })));
        }

        let DType::Decimal(to_precision_scale, to_nullability) = dtype else {
            return Ok(None);
        };

        // We only support casting to the same decimal type with different nullability
        if from_precision_scale != to_precision_scale {
            vortex_bail!(
                "Cannot cast decimal({},{}) to decimal({},{})",
                from_precision_scale.precision(),
                from_precision_scale.scale(),
                to_precision_scale.precision(),
                to_precision_scale.scale()
            );
        }

        // If the dtype is exactly the same, return self
        if array.dtype() == dtype {
            return Ok(Some(array.to_array()));
        }

        // Cast the validity to the new nullability
        let new_validity = array
            .validity()
            .clone()
            .cast_nullability(*to_nullability, array.len())?;

        // Construct DecimalArray directly since we can't use new() without knowing the concrete type
        Ok(Some(
            DecimalArray {
                dtype: DType::Decimal(*from_precision_scale, *to_nullability),
                values: array.byte_buffer(),
                values_type: array.values_type(),
                validity: new_validity,
                stats_set: ArrayStats::default(),
            }
            .to_array(),
        ))
    }
}

register_kernel!(CastKernelAdapter(DecimalVTable).lift());

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
        .ok_or_else(|| vortex_err!(ComputeError: "Failed to cast decimal value {value} to f64"))?
        / scale_factor;

    T::from(value).ok_or_else(
        || vortex_err!(ComputeError: "Failed to cast decimal value {value} to {:?}", T::PTYPE),
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::DecimalDType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;

    use crate::arrays::DecimalArray;
    use crate::canonical::ToCanonical;
    use crate::compute::cast;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;

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
        let casted = cast(array.as_ref(), &nullable_dtype).unwrap().to_decimal();

        assert_eq!(casted.dtype(), &nullable_dtype);
        assert_eq!(casted.validity(), &Validity::AllValid);
        assert_eq!(casted.len(), 3);
    }

    #[test]
    fn cast_nullable_to_non_nullable() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with no nulls
        let array = DecimalArray::new(buffer![100i32, 200, 300], decimal_dtype, Validity::AllValid);

        // Cast to non-nullable
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        let casted = cast(array.as_ref(), &non_nullable_dtype)
            .unwrap()
            .to_decimal();

        assert_eq!(casted.dtype(), &non_nullable_dtype);
        assert_eq!(casted.validity(), &Validity::NonNullable);
    }

    #[test]
    #[should_panic(expected = "Cannot cast array with invalid values to non-nullable type")]
    fn cast_nullable_with_nulls_to_non_nullable_fails() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with nulls
        let array = DecimalArray::from_option_iter([Some(100i32), None, Some(300)], decimal_dtype);

        // Attempt to cast to non-nullable should fail
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        cast(array.as_ref(), &non_nullable_dtype).unwrap();
    }

    #[test]
    fn cast_different_precision_fails() {
        let array = DecimalArray::new(
            buffer![100i32],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );

        // Try to cast to different precision
        let different_dtype = DType::Decimal(DecimalDType::new(15, 3), Nullability::NonNullable);
        let result = cast(array.as_ref(), &different_dtype);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot cast decimal(10,2) to decimal(15,3)")
        );
    }

    #[test]
    fn cast_decimal_to_f64_applies_scale() {
        let array = DecimalArray::new(
            buffer![12345i64, -50, 0],
            DecimalDType::new(15, 2),
            Validity::NonNullable,
        );
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);

        let casted = cast(array.as_ref(), &dtype).unwrap().to_primitive();

        assert_eq!(casted.dtype(), &dtype);
        assert_eq!(casted.validity(), &Validity::NonNullable);
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

        let casted = cast(array.as_ref(), &dtype).unwrap().to_primitive();

        assert_eq!(casted.dtype(), &dtype);
        assert_eq!(casted.validity(), array.validity());
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
        let result = cast(array.as_ref(), &DType::Utf8(Nullability::NonNullable));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No compute kernel to cast")
        );
    }

    #[rstest]
    #[case(DecimalArray::new(buffer![100i32, 200, 300], DecimalDType::new(10, 2), Validity::NonNullable))]
    #[case(DecimalArray::new(buffer![10000i64, 20000, 30000], DecimalDType::new(18, 4), Validity::NonNullable))]
    #[case(DecimalArray::from_option_iter([Some(100i32), None, Some(300)], DecimalDType::new(10, 2)))]
    #[case(DecimalArray::new(buffer![42i32], DecimalDType::new(5, 1), Validity::NonNullable))]
    fn test_cast_decimal_conformance(#[case] array: DecimalArray) {
        test_cast_conformance(array.as_ref());
    }
}
