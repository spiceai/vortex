// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression round trips and regressions involving the built-in schemes.
//!
//! These tests cover scheme composition and configuration. Individual encoding kernels and the
//! generic selection algorithm are tested in their owning crates.

use std::iter;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
#[cfg(feature = "zstd")]
use vortex_array::ArrayId;
#[cfg(feature = "zstd")]
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::validity::Validity;
use vortex_datetime_parts::DateTimeParts;
#[cfg(feature = "zstd")]
use vortex_edition::EDITION_DECLARATIONS;
#[cfg(feature = "zstd")]
use vortex_edition::EDITION_FAMILIES;
#[cfg(feature = "zstd")]
use vortex_edition::EditionId;
#[cfg(feature = "zstd")]
use vortex_edition::EditionSessionExt;
#[cfg(feature = "zstd")]
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexResult;
use vortex_fastlanes::Delta;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;
use crate::BtrBlocksCompressorBuilder;
use crate::SESSION;

#[track_caller]
fn assert_roundtrip(compressor: &BtrBlocksCompressor, input: &ArrayRef) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(input, &mut ctx)?;
    assert_arrays_eq!(&compressed, input, &mut ctx);

    Ok(compressed)
}

#[test]
fn test_default_compressor_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let integers = PrimitiveArray::from_option_iter(
        (0..2048i32).map(|i| (i % 7 != 0).then_some(1_000_000 + (i * 37) % 100)),
    );
    let floats = PrimitiveArray::from_iter((0..2048).map(|i| f64::from(i) / 100.0));
    let strings = VarBinViewArray::from_iter(
        (0..2048).map(|i| (i % 7 != 0).then(|| format!("https://example.com/products/{i}"))),
        DType::Utf8(Nullability::Nullable),
    );
    let binary = VarBinViewArray::from_iter(
        (0..2048).map(|i| (i % 7 != 0).then(|| format!("PREFIX_{i:09}").into_bytes())),
        DType::Binary(Nullability::Nullable),
    );
    let decimals = DecimalArray::new(
        (0..2048i64).map(|i| i * 123).collect(),
        DecimalDType::new(12, 2),
        Validity::NonNullable,
    );
    let timestamps = TemporalArray::new_timestamp(
        PrimitiveArray::from_iter((0..2048i64).map(|i| 1_700_000_000_000_000 + i * 123_456))
            .into_array(),
        TimeUnit::Microseconds,
        None,
    );
    let elements = StructArray::from_fields(&[
        ("integer", integers.into_array()),     //
        ("float", floats.into_array()),         //
        ("string", strings.into_array()),       //
        ("binary", binary.into_array()),        //
        ("decimal", decimals.into_array()),     //
        ("timestamp", timestamps.into_array()), //
    ])?;
    let offsets = PrimitiveArray::from_iter((0..=2048i32).step_by(4));
    let input = ListArray::try_new(
        elements.into_array(),
        offsets.into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();

    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.nbytes() < input.nbytes());

    Ok(())
}

#[rstest]
#[case::seconds(TimeUnit::Seconds)]
#[case::milliseconds(TimeUnit::Milliseconds)]
fn test_extreme_timestamps_roundtrip(#[case] unit: TimeUnit) -> VortexResult<()> {
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let input = TemporalArray::new_timestamp(
        PrimitiveArray::from_iter((0..2048i64).map(|i| match i % 3 {
            0 => i64::MAX - i,
            1 => i64::MIN + i,
            _ => 1_700_000_000 + i,
        }))
        .into_array(),
        unit,
        None,
    )
    .into_array();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.is::<DateTimeParts>());

    Ok(())
}

#[rstest]
#[case::unaligned({
    let mut rng = StdRng::seed_from_u64(7);
    let mut value = 500_000i32;
    PrimitiveArray::from_iter((0..1025).map(|_| {
        value += 1 + (rng.next_u32() % 6) as i32;
        value
    }))
})]
#[case::nullable_unaligned(
    PrimitiveArray::from_option_iter(iter::once(None).chain((1i32..=100_000).map(Some)))
)]
fn test_delta_unaligned_roundtrip(#[case] input: PrimitiveArray) -> VortexResult<()> {
    // Zero-padding the final chunk used to inflate the delta span and reject this scheme.
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let input = input.into_array();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.is::<Delta>());

    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(sum(&compressed, &mut ctx)?, sum(&input, &mut ctx)?);

    Ok(())
}

#[cfg(feature = "zstd")]
#[rstest]
#[case::array_level(CORE_2026_08_3, vortex_zstd::Zstd.id())]
#[case::buffer_level(vortex_zstd::editions::ZSTD_2026_02, vortex_zstd::ZstdBuffers.id())]
fn test_cuda_binary_zstd_follows_editions(
    #[case] edition: EditionId,
    #[case] expected: ArrayId,
) -> VortexResult<()> {
    let values: Vec<_> = (0..1024u32)
        .map(|i| {
            let mut value = Vec::from(&b"common binary payload prefix "[..]);
            value.extend_from_slice(&i.to_le_bytes());
            value.extend_from_slice(&[b'x'; 96]);
            value
        })
        .collect();
    let input = VarBinViewArray::from_iter_bin(values.iter().map(Vec::as_slice)).into_array();

    // The CUDA preset enables both Zstd schemes; the edition filter must choose between them.
    let session = VortexSession::empty();
    vortex_zstd::initialize(&session);
    for family in EDITION_FAMILIES {
        session.editions().declare_family(family)?;
    }
    for declaration in EDITION_DECLARATIONS {
        session.register_edition(declaration)?;
    }
    session.enable_edition(edition)?;
    let compressor = BtrBlocksCompressorBuilder::from_session(&session)
        .only_cuda_compatible()
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert_eq!(compressed.encoding_id(), expected);

    Ok(())
}

/// Deterministic JSON-shaped strings, the wide values byte-capped blocks are split from.
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

/// A plan recorded on one block is replayed on the blocks split from it, which hold other rows
/// of the same dtype; the replay must round-trip them.
#[test]
fn a_replayed_plan_roundtrips_other_data_of_the_same_dtype() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let (recorded, plan) =
        compressor.compress_recording(&json_strings(1024, 7).into_array(), &mut ctx)?;

    // Different rows, and short repeated strings the recorded data never had.
    let other = json_strings(1024, 11).into_array();
    let replayed = compressor.compress_with_plan(&other, &plan, &mut ctx)?;
    assert_arrays_eq!(replayed, other, &mut ctx);

    let repeated =
        VarBinViewArray::from_iter_str((0..1024).map(|i| ["a", "bb", "ccc"][i % 3])).into_array();
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
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();

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
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let sequential = PrimitiveArray::from_iter(0..4096_i64).into_array();
    let (recorded, plan) = compressor.compress_recording(&sequential, &mut ctx)?;
    // The premise: the plan holds a scheme that random integers cannot use.
    assert_eq!(recorded.encoding_id().to_string(), "vortex.sequence");

    let mut state = 0x9e37_79b9_u64;
    let random = PrimitiveArray::from_iter((0..4096).map(|_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        i64::try_from(state % 1_000_000).unwrap_or(0)
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
