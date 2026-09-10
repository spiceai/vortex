// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Extension;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::scalar::Scalar;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::list_contains::ListContainsElementKernel;

/// Answers `IN` for an extension needle through its storage.
///
/// An extension value *is* its storage value — `ScalarValue` has no extension
/// variant — so membership over a timestamp, date or any user-defined extension
/// is membership over the integers or bytes underneath it. `Operator::Eq`
/// already compares extensions this way, so unwrapping both sides gives the same
/// answers while letting the storage type's own set probe do the work.
impl ListContainsElementKernel for Extension {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(list_constant) = list.as_opt::<Constant>() else {
            return Ok(None);
        };
        let list_scalar = list_constant.scalar().as_list();
        let Some(elements) = list_scalar.element_values() else {
            return Ok(None);
        };

        // Two extensions are only comparable when they agree on the whole
        // extension dtype — a timestamp in milliseconds must not be matched
        // against one in seconds. Executing as a parent kernel skips the check
        // `ListContains` does for itself, so decline a mismatch here and let the
        // generic path raise the error.
        if !list_scalar
            .element_dtype()
            .eq_ignore_nullability(element.dtype())
        {
            return Ok(None);
        }

        let storage = element.storage_array().clone();
        let storage_dtype = storage.dtype().clone();
        let storage_list = Scalar::list(
            storage_dtype,
            elements
                .iter()
                .map(|value| {
                    // SAFETY: these are the values of a validated extension list
                    // scalar, so each is a valid value of the storage dtype.
                    unsafe { Scalar::new_unchecked(storage.dtype().clone(), value.clone()) }
                })
                .collect(),
            list_constant.scalar().dtype().nullability(),
        );
        let len = storage.len();

        ListContains
            .try_new_array(
                len,
                EmptyOptions,
                [ConstantArray::new(storage_list, len).into_array(), storage],
            )?
            .execute(ctx)
            .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::bool::BoolArrayExt;
    use crate::arrays::scalar_fn::ScalarFnFactoryExt;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::Scalar;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::fns::list_contains::ListContains;

    /// Membership over a timestamp column, whose storage is the `i64` beneath it.
    fn timestamps_in(list_values: &[i64], unit: TimeUnit) -> Vec<bool> {
        let mut ctx = array_session().create_execution_ctx();
        let ext_dtype = Timestamp::new(unit, Nullability::NonNullable).erased();
        let needles = ExtensionArray::new(
            ext_dtype.clone(),
            buffer![10i64, 20, 30, 40, 50].into_array(),
        )
        .into_array();

        let element_dtype = DType::Extension(ext_dtype);
        let list = Scalar::list(
            Arc::new(element_dtype.clone()),
            list_values
                .iter()
                .map(|v| {
                    Scalar::primitive(*v, Nullability::NonNullable)
                        .cast(&element_dtype)
                        .expect("i64 into the timestamp it stores")
                })
                .collect(),
            Nullability::NonNullable,
        );

        let len = needles.len();
        let result = ListContains
            .try_new_array(
                len,
                EmptyOptions,
                [ConstantArray::new(list, len).into_array(), needles],
            )
            .expect("build")
            .execute::<BoolArray>(&mut ctx)
            .expect("execute");
        let bits = result.bit_buffer_view();
        (0..bits.len()).map(|i| bits.value(i)).collect()
    }

    #[test]
    fn a_timestamp_list_is_answered_through_its_storage() {
        // Long enough to take the set probe rather than OR-of-equalities.
        assert_eq!(
            timestamps_in(&[10, 30, 50, 70], TimeUnit::Milliseconds),
            vec![true, false, true, false, true]
        );
        // Below the probe threshold, so the equality form answers it; both must
        // agree.
        assert_eq!(
            timestamps_in(&[20, 40], TimeUnit::Milliseconds),
            vec![false, true, false, true, false]
        );
    }

    #[test]
    fn a_different_time_unit_is_not_matched_against_raw_storage() {
        // The list is in seconds and the column in milliseconds. Their storage
        // integers would compare equal, but the values do not.
        let mut ctx = array_session().create_execution_ctx();
        let needles = ExtensionArray::new(
            Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
            buffer![10i64, 20, 30].into_array(),
        )
        .into_array();
        let seconds =
            DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullability::NonNullable).erased());
        let list = Scalar::list(
            Arc::new(seconds.clone()),
            vec![
                Scalar::primitive(10i64, Nullability::NonNullable)
                    .cast(&seconds)
                    .expect("i64 into the timestamp it stores"),
            ],
            Nullability::NonNullable,
        );
        let len = needles.len();
        let result = ListContains
            .try_new_array(
                len,
                EmptyOptions,
                [ConstantArray::new(list, len).into_array(), needles],
            )
            .expect("build")
            .execute::<BoolArray>(&mut ctx);
        assert!(
            result.is_err(),
            "a seconds list must not match a milliseconds column"
        );
    }
}
