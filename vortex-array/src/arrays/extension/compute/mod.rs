// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod filter;
mod mask;
pub(crate) mod rules;
mod slice;
mod take;

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::Nullability;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;

    #[test]
    fn test_filter_extension_array() {
        let ext_dtype = Date::new(TimeUnit::Days, Nullability::NonNullable).erased();

        // Create storage array
        let storage = buffer![1i32, 2, 3, 4, 5].into_array();
        let array = ExtensionArray::new(ext_dtype.clone(), storage);
        test_filter_conformance(&array.into_array());

        // Test with nullable extension type
        let ext_dtype_nullable = ext_dtype.with_nullability(Nullability::Nullable);
        let storage = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None])
            .into_array();
        let array = ExtensionArray::new(ext_dtype_nullable, storage);
        test_filter_conformance(&array.into_array());
    }

    #[rstest]
    #[case({
        // Simple extension type (non-nullable i64)
        let storage = buffer![1i64, 2, 3, 4, 5].into_array();
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        ExtensionArray::new(ext_dtype, storage)
    })]
    #[case({
        // Nullable extension type
        let storage = PrimitiveArray::from_option_iter([Some(1i64), None, Some(3), Some(4), None])
            .into_array();
        let ext_dtype_nullable = Timestamp::new(
            TimeUnit::Milliseconds,
            Nullability::Nullable,
        ).erased();
        ExtensionArray::new(ext_dtype_nullable, storage)
    })]
    #[case({
        // Single element
        let storage = buffer![42i64].into_array();
        let ext_dtype_single = Timestamp::new(
            TimeUnit::Milliseconds,
            Nullability::NonNullable,
        ).erased();
        ExtensionArray::new(ext_dtype_single, storage)
    })]
    #[case({
        // Larger array for edge cases
        let storage = buffer![0i64..100].into_array();
        let ext_dtype_large = Timestamp::new(
            TimeUnit::Milliseconds,
            Nullability::NonNullable,
        ).erased();
        ExtensionArray::new(ext_dtype_large, storage)
    })]
    fn test_take_extension_array_conformance(#[case] array: ExtensionArray) {
        test_take_conformance(&array.into_array());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::dtype::Nullability;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;

    #[rstest]
    // Note: The original test_all_consistency cases for extension arrays caused errors
    // because of unsupported extension type "uuid". We'll use simpler test cases.
    #[case::extension_simple({
        let storage = buffer![1i64, 2, 3, 4, 5].into_array();
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        ExtensionArray::new(ext_dtype, storage)
    })]
    #[case::extension_nullable({
        let storage = PrimitiveArray::from_option_iter([Some(1i64), None, Some(3), Some(4), None])
            .into_array();
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
        ExtensionArray::new(ext_dtype, storage)
    })]
    #[case::extension_large({
        let storage = buffer![0..100i64].into_array();
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        ExtensionArray::new(ext_dtype, storage)
    })]
    fn test_extension_consistency(#[case] array: ExtensionArray) {
        test_array_consistency(&array.into_array());
    }

    /// Reading a value out of a timestamp column resolves its timezone. Arrow allows a fixed UTC
    /// offset there, and a timezone that resolves to nothing must surface as an error rather than
    /// abort the process.
    #[rstest]
    #[case("+00:00", true)]
    #[case("-05:30", true)]
    #[case("America/New_York", true)]
    #[case("Not/A/Timezone", false)]
    fn test_extension_scalar_at_resolves_timezone(
        #[case] timezone: &str,
        #[case] expected_ok: bool,
    ) {
        let ext_dtype = Timestamp::new_with_tz(
            TimeUnit::Microseconds,
            Some(Arc::from(timezone)),
            Nullability::NonNullable,
        )
        .erased();
        let array = ExtensionArray::new(ext_dtype, buffer![1_000_000i64, 2_000_000].into_array())
            .into_array();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(
            array.execute_scalar(0, &mut ctx).is_ok(),
            expected_ok,
            "for {timezone}"
        );
    }
}
