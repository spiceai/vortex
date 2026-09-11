// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::fns::list_contains::ListContains;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::FSST;
use crate::FSSTArrayExt;

/// Answers `IN` over compressed strings without decompressing any of them.
///
/// FSST compression under a fixed symbol table is deterministic and lossless, so
/// two values are equal exactly when their codes are — the same property
/// `compare_fsst_constant` relies on to answer `Eq` in compressed space.
/// Compressing the list once therefore turns membership into a `Binary`
/// membership test over the codes, which is a few bytes per row against the
/// decompressed string the generic path would have to materialize.
/// Rows per list element below which this kernel declines.
///
/// It compresses the whole list once per batch and saves decompressing every
/// row, so both sides scale and which one wins is a ratio rather than a size.
/// Measured on an 8192-row batch: a column of short words — the least
/// compressible shape, and so the one with the least decompression to save —
/// breaks even near 256 elements, while a column of URLs stays ahead to around
/// 768. The lower crossover is the one to take: declining early costs a missed
/// speedup, accepting late costs a regression.
const MIN_ROWS_PER_LIST_ELEMENT: usize = 32;

impl ListContainsElementKernel for FSST {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // The list is compressed once and probed by every row, so it has to be
        // the same list for every row.
        let Some(list_constant) = list.as_opt::<Constant>() else {
            return Ok(None);
        };
        let list_scalar = list_constant.scalar().as_list();
        let Some(elements) = list_scalar.element_values() else {
            return Ok(None);
        };

        // Executing as a parent kernel bypasses `ListContains`'s own execution,
        // and with it the check that the list's elements and the needle are the
        // same type. Compressing both sides to `Binary` codes would answer a
        // mismatch that the definition rejects, so decline it here and let the
        // generic path raise the error.
        if !list_scalar
            .element_dtype()
            .eq_ignore_nullability(element.dtype())
        {
            return Ok(None);
        }

        if elements.len().saturating_mul(MIN_ROWS_PER_LIST_ELEMENT) > element.len() {
            return Ok(None);
        }

        let compressor = element.compressor();
        let code_dtype = DType::Binary(Nullability::NonNullable);
        let mut codes = Vec::with_capacity(elements.len());
        for value in elements {
            let bytes = match value {
                Some(ScalarValue::Utf8(value)) => value.as_bytes(),
                Some(ScalarValue::Binary(value)) => value.as_slice(),
                // A null element has no compressed form. The generic path
                // answers those, so hand the whole list back to it.
                _ => return Ok(None),
            };
            codes.push(Scalar::binary(
                ByteBuffer::from(compressor.compress(bytes)),
                Nullability::NonNullable,
            ));
        }

        let code_list = Scalar::list(
            Arc::new(code_dtype),
            codes,
            list_constant.scalar().dtype().nullability(),
        );
        let needles = element.codes().into_array();
        let len = needles.len();

        ListContains
            .try_new_array(
                len,
                EmptyOptions,
                [ConstantArray::new(code_list, len).into_array(), needles],
            )?
            .execute(ctx)
            .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::bool::BoolArrayExt;
    use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::fns::list_contains::ListContains;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    /// The result row by row, with `None` for an invalid slot.
    fn bool_answers(
        result: &vortex_array::Array<vortex_array::arrays::Bool>,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> Vec<Option<bool>> {
        let bits = result.bit_buffer_view();
        let validity = result
            .validity()
            .expect("validity")
            .execute_mask(bits.len(), ctx)
            .expect("validity mask");
        (0..bits.len())
            .map(|i| validity.value(i).then(|| bits.value(i)))
            .collect()
    }

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    /// The values the tests below compress. Deliberately includes an empty
    /// string, a value sharing a long prefix with another, and nulls.
    const VALUES: [Option<&str>; 8] = [
        Some("hello"),
        None,
        Some("world"),
        Some(""),
        Some("a value long enough to need a buffer 1"),
        Some("a value long enough to need a buffer 2"),
        Some("hello"),
        None,
    ];

    fn answers(
        list: Vec<&str>,
        needle_nullability: Nullability,
    ) -> VortexResult<Vec<Option<bool>>> {
        let mut ctx = SESSION.create_execution_ctx();
        let plain = VarBinArray::from_iter(VALUES, DType::Utf8(needle_nullability)).into_array();
        let compressor = fsst_train_compressor(&plain, &mut ctx)?;
        let compressed = fsst_compress(&plain, &compressor, &mut ctx)?.into_array();

        let list = Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            list.into_iter()
                .map(|value| Scalar::utf8(value, Nullability::NonNullable))
                .collect(),
            Nullability::NonNullable,
        );

        let mut results = Vec::new();
        for needles in [plain, compressed] {
            let len = needles.len();
            let result = ListContains
                .try_new_array(
                    len,
                    EmptyOptions,
                    [ConstantArray::new(list.clone(), len).into_array(), needles],
                )?
                .execute::<BoolArray>(&mut ctx)?;
            results.push(bool_answers(&result, &mut ctx));
        }

        // The compressed answer must equal the uncompressed one; the tests then
        // only have to assert the latter.
        assert_eq!(results[0], results[1], "FSST answer diverged from VarBin");
        Ok(results.remove(1))
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn probes_compressed_codes() -> VortexResult<()> {
        // Long enough to take the set probe rather than OR-of-equalities.
        assert_eq!(
            answers(
                vec!["hello", "world", "absent", "also absent", "and another"],
                Nullability::Nullable
            )?,
            vec![
                Some(true),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(false)
            ]
        );
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn empty_string_is_a_code_like_any_other() -> VortexResult<()> {
        assert_eq!(
            answers(vec!["", "a", "b", "c"], Nullability::Nullable)?,
            vec![
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(false)
            ]
        );
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_shared_prefix_does_not_collide() -> VortexResult<()> {
        // The two long values differ only in their final byte, so a code
        // comparison that stopped at a prefix would answer both alike.
        assert_eq!(
            answers(
                vec![
                    "a value long enough to need a buffer 1",
                    "q",
                    "qq",
                    "qqq",
                    "qqqq"
                ],
                Nullability::Nullable
            )?,
            vec![
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                Some(false)
            ]
        );
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn short_list_below_the_probe_threshold() -> VortexResult<()> {
        assert_eq!(
            answers(vec!["hello", ""], Nullability::Nullable)?,
            vec![
                Some(true),
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                Some(true),
                Some(false)
            ]
        );
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn empty_list_matches_nothing() -> VortexResult<()> {
        assert_eq!(
            answers(vec![], Nullability::Nullable)?,
            vec![Some(false); VALUES.len()]
        );
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn non_nullable_needles() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let plain = VarBinArray::from_iter(
            ["a", "b", "c", "d"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let compressor = fsst_train_compressor(&plain, &mut ctx)?;
        let compressed = fsst_compress(&plain, &compressor, &mut ctx)?.into_array();

        let list = Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            ["a", "c", "e", "f"]
                .map(|v| Scalar::utf8(v, Nullability::NonNullable))
                .to_vec(),
            Nullability::NonNullable,
        );
        let len = compressed.len();
        let result = ListContains
            .try_new_array(
                len,
                EmptyOptions,
                [ConstantArray::new(list, len).into_array(), compressed],
            )?
            .execute::<BoolArray>(&mut ctx)?;

        assert_eq!(
            bool_answers(&result, &mut ctx),
            vec![Some(true), Some(false), Some(true), Some(false)]
        );
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn mismatched_element_type_is_rejected() -> VortexResult<()> {
        // A `List<Binary>` against `Utf8` needles is not a valid `list_contains`:
        // the generic path rejects it. The FSST kernel must not answer it.
        let mut ctx = SESSION.create_execution_ctx();
        let plain =
            VarBinArray::from_iter(["a", "b"].map(Some), DType::Utf8(Nullability::NonNullable))
                .into_array();
        let compressor = fsst_train_compressor(&plain, &mut ctx)?;
        let compressed = fsst_compress(&plain, &compressor, &mut ctx)?.into_array();

        let binary_list = Scalar::list(
            Arc::new(DType::Binary(Nullability::NonNullable)),
            vec![
                Scalar::binary(
                    vortex_buffer::ByteBuffer::copy_from(b"a"),
                    Nullability::NonNullable,
                ),
                Scalar::binary(
                    vortex_buffer::ByteBuffer::copy_from(b"c"),
                    Nullability::NonNullable,
                ),
            ],
            Nullability::NonNullable,
        );

        let run = |needles: vortex_array::ArrayRef| {
            let len = needles.len();
            ListContains
                .try_new_array(
                    len,
                    EmptyOptions,
                    [
                        ConstantArray::new(binary_list.clone(), len).into_array(),
                        needles,
                    ],
                )
                .and_then(|a| a.execute::<BoolArray>(&mut SESSION.create_execution_ctx()))
        };

        let varbin = run(plain);
        let fsst = run(compressed);
        assert!(
            varbin.is_err(),
            "generic path should reject the type mismatch"
        );
        assert!(
            fsst.is_err(),
            "FSST kernel answered a type mismatch the generic path rejects"
        );
        Ok(())
    }
}
