// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::tests_outside_test_module)]

use std::sync::Arc;
use std::sync::LazyLock;

use futures::StreamExt;
use futures::pin_mut;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::FieldNames;
use vortex_array::field_path;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::ByteBuffer;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

mod common;

use common::enable_all_registered_array_encodings;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();

    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);

    session
});

#[tokio::test]
async fn test_file_roundtrip() {
    let mut ctx = SESSION.create_execution_ctx();
    // Create a simple roundtrip
    let nums = PrimitiveArray::from_iter((0..1024).cycle().take(16_384)).into_array();

    let a_array = StructArray::new(
        FieldNames::from(["raw", "compressed"]),
        vec![nums.clone(), nums.clone()],
        16_384,
        Validity::NonNullable,
    )
    .into_array();

    let b_array = PrimitiveArray::from_iter((1024..2048).cycle().take(16_384)).into_array();

    let data = StructArray::new(
        FieldNames::from(["a", "b"]),
        vec![a_array, b_array],
        16_384,
        Validity::NonNullable,
    )
    .into_array();

    // Create a writer which by default uses the BtrBlocks compressor for a.compressed, but leaves
    // the b and the a.raw columns uncompressed.
    let default_strategy = Arc::new(CompressingStrategy::new(
        FlatLayoutStrategy::default(),
        BtrBlocksCompressor::default(),
    ));

    let writer = Arc::new(
        TableStrategy::new(Arc::new(FlatLayoutStrategy::default()), default_strategy)
            .with_field_writer(field_path!(a.raw), Arc::new(FlatLayoutStrategy::default()))
            .with_field_writer(field_path!(b), Arc::new(FlatLayoutStrategy::default())),
    );

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .with_strategy(writer)
        .write(&mut bytes, data.to_array_stream())
        .await
        .expect("write");

    let bytes = ByteBuffer::from(bytes);
    let vxf = SESSION.open_options().open_buffer(bytes).expect("open");

    // Read the data back
    let stream = vxf
        .scan()
        .expect("scan")
        .into_stream()
        .expect("into_stream");

    pin_mut!(stream);

    while let Some(next) = stream.next().await {
        let next = next.expect("next");
        let next = next.execute::<StructArray>(&mut ctx).unwrap();
        let a = next
            .unmasked_field_by_name("a")
            .unwrap()
            .clone()
            .execute::<StructArray>(&mut ctx)
            .unwrap();
        let b = next.unmasked_field_by_name("b").unwrap();

        let raw = a.unmasked_field_by_name("raw").unwrap();
        let compressed = a.unmasked_field_by_name("compressed").unwrap();

        assert!(raw.is_canonical());
        assert!(!compressed.is_canonical());

        assert!(b.is_canonical());
        assert!(raw.nbytes() > compressed.nbytes());
    }
}

/// Regression test: writing a Dict<ListView> where the list has
/// Validity::Array(BoolArray) and the dict codes are nullable used to fail
/// with "Array vortex.fill_null does not support serialization".
#[tokio::test]
async fn test_dict_listview_validity_roundtrip() {
    let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4, 5]).into_array();
    let offsets = PrimitiveArray::from_iter(vec![0u32, 2, 4]).into_array();
    let sizes = PrimitiveArray::from_iter(vec![2u32, 2, 1]).into_array();
    let list_validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
    let listview = ListViewArray::new(elements, offsets, sizes, list_validity).into_array();

    let codes = PrimitiveArray::new(
        vortex_buffer::buffer![0u32, 0, 1, 0, 2],
        Validity::from_iter(vec![true, false, true, true, true]),
    )
    .into_array();

    let dict = DictArray::new(codes, listview).into_array();

    let data = StructArray::from_fields(&[("col", dict)])
        .expect("from_fields")
        .into_array();

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .write(&mut bytes, data.to_array_stream())
        .await
        .expect("write should not fail with fill_null serialization error");

    let bytes = ByteBuffer::from(bytes);
    let vxf = SESSION.open_options().open_buffer(bytes).expect("open");

    let stream = vxf
        .scan()
        .expect("scan")
        .into_stream()
        .expect("into_stream");
    pin_mut!(stream);

    let chunk = stream
        .next()
        .await
        .unwrap()
        .expect("read back should succeed");
    vortex_array::assert_arrays_eq!(data, chunk, &mut SESSION.create_execution_ctx());
    assert!(stream.next().await.is_none(), "expected a single chunk");
}

/// Regression test: writing a zero-row table with a nullable struct column used to
/// panic in `CollectStrategy` ("must have visited at least one chunk").
#[tokio::test]
async fn test_write_empty_nullable_struct_column() {
    let inner = StructArray::new(
        FieldNames::from(["a"]),
        vec![PrimitiveArray::from_iter(Vec::<i32>::new()).into_array()],
        0,
        Validity::AllValid,
    )
    .into_array();

    let data = StructArray::new(
        FieldNames::from(["c0"]),
        vec![inner],
        0,
        Validity::NonNullable,
    )
    .into_array();

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .write(&mut bytes, data.to_array_stream())
        .await
        .expect("writing an empty nullable struct column should not panic");

    let bytes = ByteBuffer::from(bytes);
    let vxf = SESSION.open_options().open_buffer(bytes).expect("open");
    let stream = vxf
        .scan()
        .expect("scan")
        .into_stream()
        .expect("into_stream");
    pin_mut!(stream);

    let mut rows = 0usize;
    while let Some(next) = stream.next().await {
        rows += next.expect("read back should succeed").len();
    }
    assert_eq!(rows, 0);
}

/// Writes `data` with the default strategy and an optional data block maximum, and returns the
/// file read back with its segment count.
async fn roundtrip_with_block_max(
    data: vortex_array::ArrayRef,
    max_bytes: Option<u64>,
) -> vortex_error::VortexResult<(vortex_array::ArrayRef, usize)> {
    let strategy = vortex_file::WriteStrategyBuilder::default()
        .with_data_block_max_bytes(max_bytes)
        .build();
    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .with_strategy(strategy)
        .write(&mut bytes, data.to_array_stream())
        .await?;
    let vxf = SESSION
        .open_options()
        .open_buffer(ByteBuffer::from(bytes))?;
    let segments = vxf.footer().segment_map().len();
    let read = vxf.scan()?.into_array_stream()?.read_all().await?;
    Ok((read, segments))
}

/// A data block maximum splits the blocks of a wide string column and changes no value: every
/// row, including nulls and rows of other columns, reads back as written.
#[tokio::test]
async fn test_data_block_max_bytes_roundtrip() -> vortex_error::VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let rows = 20_000;
    let wide = vortex_array::arrays::VarBinViewArray::from_iter_nullable_str(
        (0..rows)
            .map(|i| (i % 97 != 0).then(|| format!("{i:08}-{}", "payload ".repeat(70 + i % 13)))),
    )
    .into_array();
    let ids = PrimitiveArray::from_iter(0..rows as i64).into_array();
    let data = StructArray::new(
        FieldNames::from(["id", "wide"]),
        vec![ids, wide],
        rows,
        Validity::NonNullable,
    )
    .into_array();

    let (default_read, default_segments) = roundtrip_with_block_max(data.clone(), None).await?;
    let (split_read, split_segments) =
        roundtrip_with_block_max(data.clone(), Some(256 * 1024)).await?;

    vortex_array::assert_arrays_eq!(default_read, data, &mut ctx);
    vortex_array::assert_arrays_eq!(split_read, data, &mut ctx);
    // ~600-byte rows: a 256 KiB maximum splits each 8,192-row block of `wide` into 256-row
    // blocks, so the file holds many more segments than without it.
    assert!(
        split_segments > default_segments * 4,
        "expected the maximum to split `wide`: {split_segments} segments vs {default_segments}"
    );
    Ok(())
}

/// Random bytes, `width` per row: no compression scheme shrinks them.
fn random_binary(rows: usize, width: usize, seed: u64) -> vortex_array::ArrayRef {
    let mut state = seed;
    let values: Vec<Vec<u8>> = (0..rows)
        .map(|_| {
            (0..width)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 56) as u8
                })
                .collect()
        })
        .collect();
    vortex_array::arrays::VarBinViewArray::from_iter_bin(values.iter().map(Vec::as_slice))
        .into_array()
}

/// Writes `data` with `max`, through BtrBlocks or a compressor that stores chunks as they are,
/// and returns the file size and its largest segment, checking the values read back.
async fn file_and_largest_segment(
    data: &vortex_array::ArrayRef,
    max: Option<u64>,
    pass_through: bool,
) -> vortex_error::VortexResult<(usize, u32)> {
    let mut builder = vortex_file::WriteStrategyBuilder::default().with_data_block_max_bytes(max);
    if pass_through {
        builder = builder.with_compressor(
            |chunk: &vortex_array::ArrayRef, _: &mut vortex_array::ExecutionCtx| Ok(chunk.clone()),
        );
    }
    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .with_strategy(builder.build())
        .write(&mut bytes, data.to_array_stream())
        .await?;
    let file_len = bytes.len();
    let vxf = SESSION
        .open_options()
        .open_buffer(ByteBuffer::from(bytes))?;
    let largest = vxf
        .footer()
        .segment_map()
        .iter()
        .map(|s| s.length)
        .max()
        .unwrap_or(0);
    let read = vxf.scan()?.into_array_stream()?.read_all().await?;
    vortex_array::assert_arrays_eq!(read, data.clone(), &mut SESSION.create_execution_ctx());
    Ok((file_len, largest))
}

/// Splitting a block must not store its whole data buffer in every piece: when nothing shrinks
/// the bytes (random binary, or a compressor that stores chunks as they are), each piece's
/// segment holds only its own rows. Covers a block split as it arrives and one rebuilt by
/// coalescing two separately allocated chunks.
#[tokio::test]
async fn split_blocks_of_incompressible_binary_store_only_their_rows()
-> vortex_error::VortexResult<()> {
    let one_block = random_binary(8192, 1000, 7);
    // 100-byte rows (~0.9 MiB, under the coalescing minimum), then 480-byte rows (~3.9 MiB).
    let coalesced = vortex_array::arrays::ChunkedArray::try_new(
        vec![random_binary(8192, 100, 11), random_binary(8192, 480, 13)],
        one_block.dtype().clone(),
    )?
    .into_array();
    let maximum: u64 = 4 << 20;
    for (name, data) in [("one block", &one_block), ("coalesced", &coalesced)] {
        for pass_through in [false, true] {
            let (plain, _) = file_and_largest_segment(data, None, pass_through).await?;
            let (split, largest) =
                file_and_largest_segment(data, Some(maximum), pass_through).await?;
            assert!(
                split <= plain + plain / 50,
                "{name}, pass-through {pass_through}: split file {split} bytes vs {plain}"
            );
            assert!(
                u64::from(largest) <= maximum,
                "{name}, pass-through {pass_through}: largest split segment {largest} bytes"
            );
        }
    }
    Ok(())
}
