// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::EmptyMetadata;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct TestI32Ext;
impl ExtVTable for TestI32Ext {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = &'a str;

    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    fn id(&self) -> ExtId {
        ExtId::new("test_ext")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![])
    }

    fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        _storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok("")
    }
}

impl TestI32Ext {
    fn new_non_nullable() -> ExtDType<TestI32Ext> {
        ExtDType::try_new(
            EmptyMetadata,
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
        .unwrap()
    }
}

#[test]
fn test_ext_scalar_equality() {
    let scalar1 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );
    let scalar2 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );
    let scalar3 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(43i32, Nullability::NonNullable),
    );

    let ext1 = scalar1.as_extension();
    let ext2 = scalar2.as_extension();
    let ext3 = scalar3.as_extension();

    assert_eq!(ext1, ext2);
    assert_ne!(ext1, ext3);
}

#[test]
fn test_ext_scalar_partial_ord() {
    let scalar1 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(10i32, Nullability::NonNullable),
    );
    let scalar2 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(20i32, Nullability::NonNullable),
    );

    let ext1 = scalar1.as_extension();
    let ext2 = scalar2.as_extension();

    assert!(ext1 < ext2);
    assert!(ext2 > ext1);
}

#[test]
fn test_ext_scalar_partial_ord_different_types() {
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct TestExt2;
    impl ExtVTable for TestExt2 {
        type Metadata = EmptyMetadata;
        type NativeValue<'a> = &'a str;

        #[expect(clippy::disallowed_methods, reason = "test-only id")]
        fn id(&self) -> ExtId {
            ExtId::new("test_ext_2")
        }

        fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            Ok(vec![])
        }

        fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
            Ok(EmptyMetadata)
        }

        fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
            Ok(())
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            _storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok("")
        }
    }

    let scalar1 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(10i32, Nullability::NonNullable),
    );
    let scalar2 = Scalar::extension::<TestExt2>(
        EmptyMetadata,
        Scalar::primitive(20i32, Nullability::NonNullable),
    );

    let ext1 = scalar1.as_extension();
    let ext2 = scalar2.as_extension();

    // Different extension types should not be comparable
    assert_eq!(ext1.partial_cmp(&ext2), None);
}

#[test]
fn test_ext_scalar_hash() {
    use vortex_utils::aliases::hash_set::HashSet;

    let scalar1 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );
    let scalar2 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let mut set = HashSet::new();
    set.insert(scalar2);
    set.insert(scalar1);

    // Same value should hash the same
    assert_eq!(set.len(), 1);

    // Different value should hash differently
    let scalar3 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(43i32, Nullability::NonNullable),
    );
    set.insert(scalar3);
    assert_eq!(set.len(), 2);
}

#[test]
fn test_ext_scalar_storage() {
    let storage_scalar = Scalar::primitive(42i32, Nullability::NonNullable);
    let ext_scalar = Scalar::extension::<TestI32Ext>(EmptyMetadata, storage_scalar.clone());

    let ext = ext_scalar.as_extension();
    assert_eq!(ext.to_storage_scalar(), storage_scalar);
}

#[test]
fn test_ext_scalar_ext_dtype() {
    let ext_dtype = TestI32Ext::new_non_nullable();
    let scalar = Scalar::extension::<TestI32Ext>(
        EmptyMetadata.clone(),
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let ext = scalar.as_extension();
    assert_eq!(ext.ext_dtype().id(), ext_dtype.id());
    assert_eq!(ext.ext_dtype(), &ext_dtype.erased());
}

#[test]
fn test_ext_scalar_cast_to_storage() {
    let scalar = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let ext = scalar.as_extension();

    // Cast to storage type
    let casted = ext
        .cast(&DType::Primitive(PType::I32, Nullability::NonNullable))
        .unwrap();
    assert_eq!(
        casted.dtype(),
        &DType::Primitive(PType::I32, Nullability::NonNullable)
    );
    assert_eq!(casted.as_primitive().typed_value::<i32>(), Some(42));

    // Cast to nullable storage type
    let casted_nullable = ext
        .cast(&DType::Primitive(PType::I32, Nullability::Nullable))
        .unwrap();
    assert_eq!(
        casted_nullable.dtype(),
        &DType::Primitive(PType::I32, Nullability::Nullable)
    );
    assert_eq!(
        casted_nullable.as_primitive().typed_value::<i32>(),
        Some(42)
    );
}

#[test]
fn test_ext_scalar_cast_to_self() {
    let ext_dtype = TestI32Ext::new_non_nullable();

    let scalar = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let ext = scalar.as_extension();
    let ext_dtype = ext_dtype.erased();

    // Cast to same extension type
    let casted = ext.cast(&DType::Extension(ext_dtype.clone())).unwrap();
    assert_eq!(casted.dtype(), &DType::Extension(ext_dtype.clone()));

    // Cast to nullable version of same extension type
    let nullable_ext = DType::Extension(ext_dtype).as_nullable();
    let casted_nullable = ext.cast(&nullable_ext).unwrap();
    assert_eq!(casted_nullable.dtype(), &nullable_ext);
}

#[test]
fn test_ext_scalar_cast_incompatible() {
    let scalar = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let ext = scalar.as_extension();

    // Cast to incompatible type should fail
    let result = ext.cast(&DType::Utf8(Nullability::NonNullable));
    assert!(result.is_err());
}

#[test]
fn test_ext_scalar_cast_null_to_non_nullable() {
    let scalar = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
    );

    let ext = scalar.as_extension();

    // Cast null to non-nullable should fail
    let result = ext.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
    assert!(result.is_err());
}

#[test]
fn test_ext_scalar_with_metadata() {
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct TestExtMetadata;
    impl ExtVTable for TestExtMetadata {
        type Metadata = usize;
        type NativeValue<'a> = &'a str;

        #[expect(clippy::disallowed_methods, reason = "test-only id")]
        fn id(&self) -> ExtId {
            ExtId::new("test_ext_metadata")
        }

        fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            vortex_bail!("not implemented")
        }

        fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
            vortex_bail!("not implemented")
        }

        fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
            Ok(())
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            _storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok("")
        }
    }

    let scalar = Scalar::extension::<TestExtMetadata>(
        1234,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let ext = scalar.as_extension();
    assert_eq!(ext.ext_dtype().metadata::<TestExtMetadata>(), &1234);
}

/// A `vortex.date` scalar has to convert into a `vortex.timestamp` one, not be re-labelled.
///
/// A scan falsifies `cast(col as timestamp) > lit` into `cast(max(col) as timestamp) <= lit`
/// and binds `max(col)` to a literal, so this cast decides whether a file is read at all.
/// Casting through the target's storage type instead — which is what happens when an
/// extension source is not given the chance to convert itself — hands back the date's own
/// number under the timestamp's meaning, and the two share `i64` storage for `Date64`, so it
/// does not even fail: the file is pruned and its matching rows never load.
#[test]
fn test_ext_scalar_cast_date_to_timestamp() {
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;

    // 2024-03-01, as days and as milliseconds since the epoch.
    const DAYS: i32 = 19_783;
    const MILLIS: i64 = 1_709_251_200_000;
    const NANOS: i64 = 1_709_251_200_000_000_000;

    let nanos_dtype =
        DType::Extension(Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable).erased());

    let from_days = Scalar::try_new(
        DType::Extension(Date::new(TimeUnit::Days, Nullability::NonNullable).erased()),
        Some(DAYS.into()),
    )
    .unwrap();
    let casted = from_days.cast(&nanos_dtype).unwrap();
    assert_eq!(casted.dtype(), &nanos_dtype);
    assert_eq!(
        casted.as_extension().to_storage_scalar(),
        Scalar::primitive(NANOS, Nullability::NonNullable),
        "a date in days has to scale into the target's unit"
    );

    let from_millis = Scalar::try_new(
        DType::Extension(Date::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()),
        Some(MILLIS.into()),
    )
    .unwrap();
    let casted = from_millis.cast(&nanos_dtype).unwrap();
    assert_eq!(casted.dtype(), &nanos_dtype);
    assert_eq!(
        casted.as_extension().to_storage_scalar(),
        Scalar::primitive(NANOS, Nullability::NonNullable),
        "sharing i64 storage with the target is not a reason to skip the conversion"
    );
}

/// The scalar cast has to land on the same instant as the array kernel.
///
/// One of the two is applied to a file's statistics and the other to its rows, so a
/// disagreement between them is not a type error — it is a file pruned by a comparison the
/// row filter would never have made, or a scan that fails on a statistic whose rows convert
/// fine.
///
/// Each value is cast on its own rather than as a batch: an array cast fails as a whole, so a
/// batch containing one unconvertible value would say nothing about the ones beside it.
#[test]
fn test_ext_scalar_cast_date_to_timestamp_matches_the_array_kernel() {
    use crate::IntoArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::builtins::ArrayBuiltins;
    use crate::executor::VortexSessionExecute;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;

    let session = crate::array_session();
    let mut ctx = session.create_execution_ctx();

    // Ordinary dates, then the boundaries: the largest `Date32` value, the first day no
    // timestamp can represent (10000-01-01) and the last one that can. The boundaries are
    // where a range check applied to one path and not the other shows up — the array holds
    // the converted value and the scalar's validation rejects it.
    for (source_unit, values) in [
        (
            TimeUnit::Days,
            vec![0i64, 19_783, -1, i64::from(i32::MAX), 2_932_896, 2_932_895],
        ),
        (
            TimeUnit::Milliseconds,
            vec![
                0i64,
                1_709_251_200_000,
                -86_400_000,
                i64::from(i32::MAX) * 86_400_000,
                253_402_300_800_000,
                253_402_300_799_000,
            ],
        ),
    ] {
        for target_unit in [
            TimeUnit::Seconds,
            TimeUnit::Milliseconds,
            TimeUnit::Microseconds,
            TimeUnit::Nanoseconds,
        ] {
            let source_dtype = Date::new(source_unit, Nullability::NonNullable).erased();
            let target_dtype =
                DType::Extension(Timestamp::new(target_unit, Nullability::NonNullable).erased());

            for value in &values {
                let (storage, scalar_value) = if source_unit == TimeUnit::Days {
                    let Ok(days) = i32::try_from(*value) else {
                        continue;
                    };
                    (PrimitiveArray::from_iter([days]).into_array(), days.into())
                } else {
                    (
                        PrimitiveArray::from_iter([*value]).into_array(),
                        (*value).into(),
                    )
                };

                let array_cast = ExtensionArray::new(source_dtype.clone(), storage)
                    .into_array()
                    .cast(target_dtype.clone())
                    .and_then(|a| a.execute::<ExtensionArray>(&mut ctx));

                let scalar =
                    Scalar::try_new(DType::Extension(source_dtype.clone()), Some(scalar_value))
                        .unwrap();
                let scalar_cast = scalar.cast(&target_dtype);

                match array_cast {
                    Ok(array) => {
                        let expected = array.into_array().execute_scalar(0, &mut ctx).unwrap();
                        assert_eq!(
                            scalar_cast.unwrap(),
                            expected,
                            "{source_unit} -> {target_unit} disagrees on {value}"
                        );
                    }
                    Err(_) => assert!(
                        scalar_cast.is_err(),
                        "{source_unit} -> {target_unit} converts {value}, which the array kernel refuses"
                    ),
                }
            }
        }
    }
}

/// A pair of extension types with no defined conversion has to be refused, not re-labelled.
///
/// `vortex.timestamp[ms]` and `vortex.timestamp[ns]` both store `i64`, so re-labelling one as
/// the other returns an instant a million times too small without any error. The array kernel
/// refuses this pair, and so must the scalar cast.
#[test]
fn test_ext_scalar_cast_between_timestamp_units_is_refused() {
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;

    let millis = Scalar::try_new(
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()),
        Some(1_709_251_200_000i64.into()),
    )
    .unwrap();

    let nanos =
        DType::Extension(Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable).erased());
    assert!(millis.cast(&nanos).is_err());
}
