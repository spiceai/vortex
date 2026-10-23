// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BtrBlocks-specific compressor wrapping the generic [`CascadingCompressor`].

use std::ops::Deref;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use crate::BtrBlocksCompressorBuilder;
use crate::CascadingCompressor;

/// The BtrBlocks-style compressor with all built-in schemes pre-registered.
///
/// This is a thin wrapper around [`CascadingCompressor`] that provides a default set of
/// compression schemes via [`BtrBlocksCompressorBuilder`].
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressor, BtrBlocksCompressorBuilder, Scheme, SchemeExt};
/// use vortex_btrblocks::schemes::integer::IntDictScheme;
///
/// // Default compressor - all schemes allowed.
/// let compressor = BtrBlocksCompressor::default();
///
/// // Remove specific schemes using the builder.
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .exclude_schemes([IntDictScheme.id()])
///     .build();
/// ```
#[derive(Clone)]
pub struct BtrBlocksCompressor(
    /// The underlying cascading compressor.
    pub CascadingCompressor,
);

impl BtrBlocksCompressor {
    /// Compresses an array using BtrBlocks-inspired compression.
    pub fn compress(&self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.0.compress(array, ctx)
    }
}

impl Deref for BtrBlocksCompressor {
    type Target = CascadingCompressor;

    fn deref(&self) -> &CascadingCompressor {
        &self.0
    }
}

impl std::fmt::Debug for BtrBlocksCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BtrBlocksCompressor")
            .field("compressor", &self.0)
            .finish()
    }
}

impl Default for BtrBlocksCompressor {
    fn default() -> Self {
        BtrBlocksCompressorBuilder::default().build()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    #[cfg(feature = "zstd")]
    use vortex_array::ArrayId;
    #[cfg(feature = "zstd")]
    use vortex_array::ArrayPlugin;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::List;
    use vortex_array::arrays::ListView;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    #[cfg(feature = "zstd")]
    use vortex_utils::aliases::hash_set::HashSet;

    use crate::BtrBlocksCompressor;
    #[cfg(feature = "zstd")]
    use crate::BtrBlocksCompressorBuilder;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    #[rstest]
    #[case::zctl(
        unsafe {
            ListViewArray::new_unchecked(
                buffer![1i32, 2, 3, 4, 5].into_array(),
                buffer![0i32, 3].into_array(),
                buffer![3i32, 2].into_array(),
                Validity::NonNullable,
            ).with_zero_copy_to_list(true)
        },
        true,
    )]
    #[case::overlapping(
        ListViewArray::new(
            buffer![1i32, 2, 3].into_array(),
            buffer![0i32, 0, 0].into_array(),
            buffer![3i32, 3, 3].into_array(),
            Validity::NonNullable,
        ),
        false,
    )]
    fn listview_compress_roundtrip(
        #[case] input: ListViewArray,
        #[case] expect_list: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array_ref = input.clone().into_array();
        let result = BtrBlocksCompressor::default()
            .compress(&array_ref, &mut SESSION.create_execution_ctx())?;
        if expect_list {
            assert!(result.as_opt::<List>().is_some());
        } else {
            assert!(result.as_opt::<ListView>().is_some());
        }
        assert_arrays_eq!(result, input, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_constant_all_true() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = BoolArray::new(BitBuffer::from(vec![true; 100]), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(
            &array.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(compressed.is::<Constant>());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_constant_all_false() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = BoolArray::new(BitBuffer::from(vec![false; 100]), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(
            &array.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(compressed.is::<Constant>());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_nullable_all_valid_compressed() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = BoolArray::new(
            BitBuffer::from(vec![true; 100]),
            Validity::from(BitBuffer::from(vec![true; 100])),
        );
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(
            &array.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(compressed.is::<Constant>());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_nullable_with_nulls_not_compressed() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let validity = Validity::from(BitBuffer::from_iter((0..100).map(|i| i % 3 != 0)));
        let array = BoolArray::new(BitBuffer::from(vec![true; 100]), validity);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(
            &array.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(!compressed.is::<Constant>());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_mixed_not_constant() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = BoolArray::new(
            BitBuffer::from(vec![true, false, true, false, true]),
            Validity::NonNullable,
        );
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(
            &array.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(!compressed.is::<Constant>());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_binary_constant_compressed() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = vec![Some(b"constant-bytes".as_slice()); 100];
        let array = VarBinViewArray::from_iter(values, DType::Binary(Nullability::NonNullable));
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(
            &array.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(compressed.is::<Constant>());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_binary_dict_compressed() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let distinct_values: [&[u8]; 3] = [b"alpha", b"beta", b"gamma"];
        let values = (0..1000)
            .map(|idx| Some(distinct_values[idx % distinct_values.len()]))
            .collect::<Vec<_>>();
        let array = VarBinViewArray::from_iter(values, DType::Binary(Nullability::NonNullable));
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(
            &array.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(compressed.is::<Dict>());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn test_compact_binary_zstd_compressed() -> VortexResult<()> {
        let values = (0..1024)
            .map(|idx| {
                let mut value = Vec::from(&b"common binary payload prefix "[..]);
                value.extend_from_slice(&(idx as u32).to_le_bytes());
                value.extend_from_slice(&[b'x'; 96]);
                value
            })
            .collect::<Vec<_>>();
        let array = VarBinViewArray::from_iter(
            values.iter().map(|value| Some(value.as_slice())),
            DType::Binary(Nullability::NonNullable),
        );

        let compressor = BtrBlocksCompressorBuilder::default().with_compact().build();
        let mut ctx = SESSION.create_execution_ctx();
        let compressed = compressor.compress(&array.clone().into_array(), &mut ctx)?;

        assert!(
            compressed.is::<vortex_zstd::Zstd>(),
            "expected Zstd, got {}",
            compressed.encoding_id()
        );
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[cfg(feature = "zstd")]
    #[rstest]
    #[case::array_level(vortex_zstd::Zstd.id())]
    #[case::buffer_level(vortex_zstd::ZstdBuffers.id())]
    fn test_cuda_compatible_binary_zstd_follows_editions(
        #[case] allowed: ArrayId,
    ) -> VortexResult<()> {
        let values = (0..1024)
            .map(|idx| {
                let mut value = Vec::from(&b"common binary payload prefix "[..]);
                value.extend_from_slice(&(idx as u32).to_le_bytes());
                value.extend_from_slice(&[b'x'; 96]);
                value
            })
            .collect::<Vec<_>>();
        let array = VarBinViewArray::from_iter(
            values.iter().map(|value| Some(value.as_slice())),
            DType::Binary(Nullability::NonNullable),
        );

        // The CUDA preset carries both Zstd schemes; the edition filter decides which one
        // survives.
        let compressor = BtrBlocksCompressorBuilder::default()
            .only_cuda_compatible()
            .retain_allowed_encodings(&HashSet::from([allowed]))
            .build();
        let mut ctx = SESSION.create_execution_ctx();
        let compressed = compressor.compress(&array.clone().into_array(), &mut ctx)?;

        assert_eq!(compressed.encoding_id(), allowed);
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    /// Strings shaped like JSON documents, with a seed so two calls draw different data.
    fn json_strings(rows: usize, seed: u64) -> VarBinViewArray {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        VarBinViewArray::from_iter_str((0..rows).map(|row| {
            let mut s = format!("{{\"id\":{row},\"items\":[");
            for _ in 0..(next() % 8 + 1) {
                let v = next();
                s.push_str(&format!(
                    "{{\"sku\":\"SKU-{:08x}\",\"qty\":{}}},",
                    v >> 32,
                    (v >> 40) % 50
                ));
            }
            s.push_str("]}");
            s
        }))
    }

    #[test]
    fn replaying_a_plan_reproduces_the_search_on_the_same_array() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = json_strings(4096, 7).into_array();
        let compressor = BtrBlocksCompressor::default();

        let searched = compressor.compress(&input, &mut ctx)?;
        let (recorded, plan) = compressor.compress_recording(&input, &mut ctx)?;
        let replayed = compressor.compress_with_plan(&input, &plan, &mut ctx)?;

        assert!(!plan.is_empty(), "recording decided nothing");
        let tree = searched.display_tree_encodings_only().to_string();
        assert_eq!(recorded.display_tree_encodings_only().to_string(), tree);
        assert_eq!(replayed.display_tree_encodings_only().to_string(), tree);
        assert_eq!(replayed.nbytes(), searched.nbytes());
        assert_arrays_eq!(replayed, input, &mut ctx);
        Ok(())
    }

    #[test]
    fn a_replayed_plan_roundtrips_other_data_of_the_same_dtype() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let compressor = BtrBlocksCompressor::default();
        let (recorded, plan) =
            compressor.compress_recording(&json_strings(1024, 7).into_array(), &mut ctx)?;

        // Different rows, and short repeated strings the recorded data never had.
        let other = json_strings(1024, 11).into_array();
        let replayed = compressor.compress_with_plan(&other, &plan, &mut ctx)?;
        assert_arrays_eq!(replayed, other, &mut ctx);

        let repeated = VarBinViewArray::from_iter_str((0..1024).map(|i| ["a", "bb", "ccc"][i % 3]))
            .into_array();
        let replayed = compressor.compress_with_plan(&repeated, &plan, &mut ctx)?;
        assert_arrays_eq!(replayed, repeated, &mut ctx);
        // The replay used the recorded scheme rather than searching again: a search on these
        // strings picks a different one.
        let searched = compressor.compress(&repeated, &mut ctx)?;
        assert_ne!(searched.encoding_id(), recorded.encoding_id());
        assert_eq!(replayed.encoding_id(), recorded.encoding_id());
        Ok(())
    }

    #[test]
    fn an_empty_plan_falls_back_to_the_search() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = json_strings(2048, 3).into_array();
        let compressor = BtrBlocksCompressor::default();

        let searched = compressor.compress(&input, &mut ctx)?;
        let replayed =
            compressor.compress_with_plan(&input, &crate::CompressionPlan::default(), &mut ctx)?;

        assert_eq!(
            replayed.display_tree_encodings_only().to_string(),
            searched.display_tree_encodings_only().to_string()
        );
        assert_eq!(replayed.nbytes(), searched.nbytes());
        Ok(())
    }

    /// A plan is replayed on other data of the same dtype; a scheme that cannot encode that data
    /// must not fail the write.
    #[test]
    fn a_replayed_plan_survives_data_its_schemes_cannot_encode() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let compressor = BtrBlocksCompressor::default();
        let sequential = vortex_array::arrays::PrimitiveArray::from_iter(0..4096_i64).into_array();
        let (recorded, plan) = compressor.compress_recording(&sequential, &mut ctx)?;
        // The premise: the plan holds a scheme that random integers cannot use.
        assert_eq!(recorded.encoding_id().to_string(), "vortex.sequence");

        let mut state = 0x9e37_79b9_u64;
        let random = vortex_array::arrays::PrimitiveArray::from_iter((0..4096).map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 1_000_000) as i64
        }))
        .into_array();
        let replayed = compressor.compress_with_plan(&random, &plan, &mut ctx)?;
        assert_eq!(
            replayed.encoding_id(),
            compressor.compress(&random, &mut ctx)?.encoding_id()
        );
        assert_arrays_eq!(replayed, random, &mut ctx);
        Ok(())
    }
}
