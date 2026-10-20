// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt as _;
use futures::pin_mut;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

#[derive(Clone)]
pub struct RepartitionWriterOptions {
    /// The minimum uncompressed size in bytes for a block.
    pub block_size_minimum: u64,
    /// The multiple of the number of rows in each block.
    pub block_len_multiple: usize,
    /// Optional target uncompressed size in bytes for a block.
    ///
    /// The repartition writer attempts to produce partitions with this uncompressed size. This is
    /// only a best effort attempt: the partitions may be arbitrarily larger or smaller. Reasons for
    /// this include:
    ///
    /// 1. The size of one element may not perfectly divide the target size, resulting in blocks
    ///    that are either too large or too small.
    ///
    /// 2. Variable length types are expensive to pack due to the need to read each element length.
    ///
    /// 3. View types are expensive to pack due to each view sharing an arbitrary slice of data.
    pub block_size_target: Option<u64>,
    pub canonicalize: bool,
}

impl RepartitionWriterOptions {
    /// Compute the effective block length for a given [`DType`].
    ///
    /// For fixed-width types where [`DType::element_size`] is known and large enough that
    /// `element_size * block_len_multiple` would exceed `block_size_target`, this reduces the
    /// block length so each block stays close to the target byte size.
    fn effective_block_len(&self, dtype: &DType) -> usize {
        let Some(block_size_target) = self.block_size_target else {
            return self.block_len_multiple;
        };
        match dtype.element_size() {
            Some(elem_size) if elem_size > 0 => {
                // `div_ceil` ensures we overshoot the block_size_target; therefore preventing
                // `write_stream` from combining adjacent 0.9 MiB chunks into one 1.8 MiB chunk.
                let max_rows = usize::try_from(block_size_target.div_ceil(elem_size as u64))
                    .unwrap_or(usize::MAX);
                self.block_len_multiple.min(max_rows).max(1)
            }
            _ => self.block_len_multiple,
        }
    }
}

/// Repartition a stream of arrays into blocks.
///
/// Each emitted block (except the last) is at least `block_size_minimum` bytes and contains a
/// multiple of `block_len_multiple` rows.
#[derive(Clone)]
pub struct RepartitionStrategy {
    child: Arc<dyn LayoutStrategy>,
    options: RepartitionWriterOptions,
    block_size_maximum: Option<u64>,
}

impl RepartitionStrategy {
    pub fn new<S: LayoutStrategy>(child: S, options: RepartitionWriterOptions) -> Self {
        Self {
            child: Arc::new(child),
            options,
            block_size_maximum: None,
        }
    }

    /// Split a canonical string or binary chunk whose rows reference more than `maximum`
    /// bytes into blocks of at most about `maximum` bytes.
    ///
    /// [`RepartitionWriterOptions::block_size_target`] sizes blocks of fixed-width types by
    /// their element size; a variable-width type has none, so a block of a wide string column
    /// holds `block_len_multiple` rows whatever their width, and a reader that wants one row
    /// reads all of them. A split block holds a power-of-two number of rows, which divides
    /// any power-of-two `block_len_multiple`, so its boundaries line up with the blocks of
    /// every other column. Chunks at or under `maximum` are repartitioned as without it.
    pub fn with_block_size_maximum(mut self, maximum: Option<u64>) -> Self {
        self.block_size_maximum = maximum;
        self
    }
}

/// The number of rows per block that keeps each block of `chunk` at or under `maximum` bytes,
/// as a power of two no larger than `max_rows`, or `None` when `chunk` needs no split.
///
/// Only a canonical [`VarBinView`] chunk is measured, by the bytes its rows reference (each
/// view plus the bytes it points at), not by its buffers: rows sliced from a larger array share
/// that array's buffers.
fn split_rows(chunk: &ArrayRef, maximum: u64, max_rows: usize) -> Option<usize> {
    let views = chunk.as_opt::<VarBinView>()?;
    let len = chunk.len();
    if len <= 1 {
        return None;
    }
    let referenced: u64 = views
        .views()
        .iter()
        .map(|view| u64::from(view.len()))
        .sum::<u64>()
        + (len * size_of::<BinaryView>()) as u64;
    if referenced <= maximum {
        return None;
    }
    let fit = usize::try_from(u128::from(maximum) * len as u128 / u128::from(referenced))
        .unwrap_or(usize::MAX)
        .clamp(1, max_rows.max(1));
    let rows = 1_usize << (usize::BITS - 1 - fit.leading_zeros());
    (rows < len).then_some(rows)
}

#[async_trait]
impl LayoutStrategy for RepartitionStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        // TODO(os): spawn stream below like:
        // canon_stream = stream.map(async {to_canonical}).map(spawn).buffered(parallelism)
        let dtype = stream.dtype().clone();
        let stream = if self.options.canonicalize {
            let canonicalize_session = session.clone();
            SequentialStreamAdapter::new(
                dtype.clone(),
                stream.map(move |chunk| {
                    let (sequence_id, chunk) = chunk?;
                    let mut ctx = canonicalize_session.create_execution_ctx();
                    let canonical = chunk.execute::<Canonical>(&mut ctx)?.into_array();
                    VortexResult::Ok((sequence_id, canonical))
                }),
            )
            .sendable()
        } else {
            stream
        };

        let dtype_clone = dtype.clone();
        let options = self.options.clone();

        // For fixed-width types with large per-element sizes, reduce the block_len_multiple
        // so that each block targets block_size_target bytes rather than producing oversized
        // segments.
        let block_len = options.effective_block_len(&dtype);
        let block_size_minimum = options.block_size_minimum;
        let block_size_maximum = self.block_size_maximum;
        let repartition_session = session.clone();

        let repartitioned_stream = try_stream! {
            let canonical_stream = stream.peekable();
            pin_mut!(canonical_stream);

            let mut ctx = repartition_session.create_execution_ctx();
            let mut chunks = ChunksBuffer::new(block_size_minimum, block_len);
            while let Some(chunk) = canonical_stream.as_mut().next().await {
                let (sequence_id, chunk) = chunk?;
                let mut sequence_pointer = sequence_id.descend();
                if let Some(rows) = block_size_maximum
                    .and_then(|maximum| split_rows(&chunk, maximum, block_len))
                {
                    // Blocks buffered from earlier chunks come first, as they are.
                    let pending = chunks.take_all();
                    if !pending.is_empty() {
                        let pending = ChunkedArray::try_new(pending, dtype_clone.clone())?;
                        yield (
                            sequence_pointer.advance(),
                            pending.into_array().execute::<Canonical>(&mut ctx)?.into_array(),
                        );
                    }
                    let mut offset = 0;
                    while offset < chunk.len() {
                        let end = (offset + rows).min(chunk.len());
                        yield (sequence_pointer.advance(), chunk.slice(offset..end)?);
                        offset = end;
                    }
                    continue;
                }
                let mut offset = 0;
                while offset < chunk.len() {
                    let end = (offset + block_len).min(chunk.len());
                    let sliced = chunk.slice(offset..end)?;
                    chunks.push_back(sliced);
                    offset = end;

                    if chunks.have_enough() {
                        let output_chunks = chunks.collect_exact_blocks()?;
                        assert!(!output_chunks.is_empty());
                        let chunked =
                            ChunkedArray::try_new(output_chunks, dtype_clone.clone())?;
                        if !chunked.is_empty() {
                            let canonical = chunked.into_array().execute::<Canonical>(&mut ctx)?.into_array();
                            yield (
                                sequence_pointer.advance(),
                                canonical,
                            )
                        }
                    }
                }
                if canonical_stream.as_mut().peek().await.is_none() {
                    let to_flush = ChunkedArray::try_new(
                        chunks.data.drain(..).map(|(arr, _)| arr),
                        dtype_clone.clone(),
                    )?;
                    if !to_flush.is_empty() {
                        let canonical = to_flush.into_array().execute::<Canonical>(&mut ctx)?.into_array();
                        yield (
                            sequence_pointer.advance(),
                            canonical,
                        )
                    }
                }
            }
        };

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, repartitioned_stream).sendable(),
                eof,
                session,
            )
            .await
    }
}

struct ChunksBuffer {
    /// Each entry stores the chunk and the `nbytes()` snapshot taken at push time.
    /// This avoids accounting mismatches when interior-mutable arrays (e.g. `SharedArray`)
    /// change their reported size after being pushed.
    data: VecDeque<(ArrayRef, u64)>,
    row_count: usize,
    nbytes: u64,
    block_size_minimum: u64,
    block_len_multiple: usize,
}

impl ChunksBuffer {
    fn new(block_size_minimum: u64, block_len_multiple: usize) -> Self {
        Self {
            data: Default::default(),
            row_count: 0,
            nbytes: 0,
            block_size_minimum,
            block_len_multiple,
        }
    }

    fn have_enough(&self) -> bool {
        self.nbytes >= self.block_size_minimum && self.row_count >= self.block_len_multiple
    }

    fn collect_exact_blocks(&mut self) -> VortexResult<Vec<ArrayRef>> {
        let nblocks = self.row_count / self.block_len_multiple;
        let mut res = Vec::with_capacity(self.data.len());
        let mut remaining = nblocks * self.block_len_multiple;
        while remaining > 0 {
            let (chunk, _) = self
                .pop_front()
                .vortex_expect("must have at least one chunk");
            let len = chunk.len();

            if len > remaining {
                let left = chunk.slice(0..remaining)?;
                let right = chunk.slice(remaining..len)?;
                self.push_front(right);
                res.push(left);
                remaining = 0;
            } else {
                res.push(chunk);
                remaining -= len;
            }
        }
        Ok(res)
    }

    fn push_back(&mut self, chunk: ArrayRef) {
        let nb = chunk.nbytes();
        self.row_count += chunk.len();
        self.nbytes += nb;
        self.data.push_back((chunk, nb));
    }

    fn push_front(&mut self, chunk: ArrayRef) {
        let nb = chunk.nbytes();
        self.row_count += chunk.len();
        self.nbytes += nb;
        self.data.push_front((chunk, nb));
    }

    /// Removes every buffered chunk, in order.
    fn take_all(&mut self) -> Vec<ArrayRef> {
        self.row_count = 0;
        self.nbytes = 0;
        self.data.drain(..).map(|(chunk, _)| chunk).collect()
    }

    fn pop_front(&mut self) -> Option<(ArrayRef, u64)> {
        let res = self.data.pop_front();
        if let Some((chunk, nb)) = res.as_ref() {
            self.row_count -= chunk.len();
            self.nbytes -= nb;
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SharedArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;

    use super::*;
    use crate::LayoutStrategy;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::new_session;

    const ONE_MEG: u64 = 1 << 20;

    #[test]
    fn effective_block_len_small_elements() {
        // f64 = 8 bytes/element. 8192 * 8 = 64 KiB << 1 MiB, so no reduction.
        let dtype = DType::Primitive(PType::F64, NonNullable);
        let options = RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 8192,
            block_size_target: Some(ONE_MEG),
            canonicalize: false,
        };
        assert_eq!(options.effective_block_len(&dtype), 8192);
    }

    #[test]
    fn effective_block_len_large_elements() {
        // FixedSizeList(f64, 1000) = 8000 bytes/element.
        // div_ceil(1 MiB, 8000) = 132, so effective block len = min(8192, 132) = 132.
        let dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F64, NonNullable)),
            1000,
            NonNullable,
        );
        let options = RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 8192,
            block_size_target: Some(ONE_MEG),
            canonicalize: false,
        };
        assert_eq!(options.effective_block_len(&dtype), 132);
    }

    #[test]
    fn effective_block_len_variable_width() {
        // Utf8 has no known element_size, so block_len_multiple is unchanged.
        let dtype = DType::Utf8(NonNullable);
        let options = RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 8192,
            block_size_target: Some(ONE_MEG),
            canonicalize: false,
        };
        assert_eq!(options.effective_block_len(&dtype), 8192);
    }

    #[test]
    fn effective_block_len_very_large_elements() {
        // FixedSizeList(f64, 1_000_000) = 8_000_000 bytes/element.
        // 1 MiB / 8_000_000 = 0, clamped to max(1) = 1.
        let dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F64, NonNullable)),
            1_000_000,
            NonNullable,
        );
        let options = RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 8192,
            block_size_target: Some(ONE_MEG),
            canonicalize: false,
        };
        assert_eq!(options.effective_block_len(&dtype), 1);
    }

    #[test]
    fn repartition_large_element_type_produces_small_blocks() -> VortexResult<()> {
        // Create a FixedSizeList(f64, 1000) array with 1000 lists.
        // Each list is 8000 bytes, so 1000 lists = 8 MiB total.
        // With block_size_target = 1 MiB, effective block_len = 133.
        // We expect the repartition to produce blocks of 132 rows each.
        let list_size: u32 = 1000;
        let num_lists: usize = 1000;
        let total_elements = list_size as usize * num_lists;

        let elements = PrimitiveArray::from_iter((0..total_elements).map(|i| i as f64));
        let fsl = FixedSizeListArray::new(
            elements.into_array(),
            list_size,
            Validity::NonNullable,
            num_lists,
        );

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        let child = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default());
        let strategy = RepartitionStrategy::new(
            child,
            RepartitionWriterOptions {
                block_size_minimum: 0,
                block_len_multiple: 8192,
                block_size_target: Some(ONE_MEG),
                canonicalize: false,
            },
        );

        let stream = fsl.into_array().to_array_stream().sequenced(ptr);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    stream,
                    eof,
                    &session,
                )
                .await
        })?;

        // The layout should be a ChunkedLayout with multiple children.
        // With 1000 rows and effective block_len = 132:
        //   - 7 full blocks of 132 rows = 924 rows
        //   - 1 remainder block of 76 rows
        //   - Total: 8 blocks, 1000 rows
        assert_eq!(layout.row_count(), num_lists as u64);

        // All non-last children should have 131 rows.
        let nchildren = layout.nchildren();
        assert!(nchildren > 1, "expected multiple chunks, got {nchildren}");

        for i in 0..nchildren - 1 {
            let child = layout.slot(i)?.vortex_expect("chunk slot present");
            assert_eq!(
                child.row_count(),
                132,
                "chunk {i} has {} rows, expected 131",
                child.row_count()
            );
        }

        // Last child gets the remainder.
        let last = layout
            .slot(nchildren - 1)?
            .vortex_expect("chunk slot present");
        assert_eq!(last.row_count(), 1000 - 132 * (nchildren as u64 - 1));

        Ok(())
    }

    #[test]
    fn repartition_small_element_type_unchanged() -> VortexResult<()> {
        // For f64 (8 bytes/element), effective block_len stays at 8192.
        // With 10000 elements and block_size_minimum=0, we get one block of 8192
        // and one remainder of 1808.
        let num_elements: usize = 10000;
        let elements = PrimitiveArray::from_iter((0..num_elements).map(|i| i as f64));

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        let child = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default());
        let strategy = RepartitionStrategy::new(
            child,
            RepartitionWriterOptions {
                block_size_minimum: 0,
                block_len_multiple: 8192,
                block_size_target: Some(ONE_MEG),
                canonicalize: false,
            },
        );

        let stream = elements.into_array().to_array_stream().sequenced(ptr);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    stream,
                    eof,
                    &session,
                )
                .await
        })?;

        assert_eq!(layout.row_count(), num_elements as u64);
        assert_eq!(layout.nchildren(), 2);
        assert_eq!(
            layout
                .slot(0)?
                .vortex_expect("chunk slot present")
                .row_count(),
            8192
        );
        assert_eq!(
            layout
                .slot(1)?
                .vortex_expect("chunk slot present")
                .row_count(),
            1808
        );

        Ok(())
    }

    /// Regression test: `SharedArray` slices sharing an `Arc<Mutex<SharedState>>` can
    /// transition from Source to Cached when any one of them is canonicalized. This caused
    /// `pop_front` to panic with `attempt to subtract with overflow` because the buffer's
    /// running `nbytes` total was accumulated with the smaller Source-era values while
    /// `pop_front` subtracted the larger Cached-era values.
    #[test]
    fn chunks_buffer_pop_front_no_panic_after_shared_execution() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let n = 20_000usize;
        let block_len = 10_000usize;

        let constant = ConstantArray::new(42i64, n);
        let shared = SharedArray::new(constant.into_array());
        let shared_handle = shared.clone();
        let arr = shared.into_array();

        let s1 = arr.slice(0..block_len)?;
        let s2 = arr.slice(block_len..n)?;

        let mut buf = ChunksBuffer::new(0, block_len);
        buf.push_back(s1);
        buf.push_back(s2);

        let _output = buf.pop_front().unwrap();

        // Transition SharedState from Source to Cached for ALL slices sharing this Arc.
        use vortex_array::arrays::shared::SharedArrayExt;
        let _canonical =
            shared_handle.get_or_compute(|source| source.clone().execute::<Canonical>(&mut ctx))?;

        // Before the fix this panicked with "attempt to subtract with overflow".
        let _s2 = buf.pop_front().unwrap();
        assert_eq!(buf.nbytes, 0);
        assert_eq!(buf.row_count, 0);

        Ok(())
    }

    /// Writes `array` through a coalescing repartition with a block size maximum and returns
    /// the row count of each block written.
    fn block_rows_with_maximum(array: ArrayRef, maximum: Option<u64>) -> VortexResult<Vec<u64>> {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = RepartitionStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            RepartitionWriterOptions {
                block_size_minimum: ONE_MEG,
                block_len_multiple: 8192,
                block_size_target: Some(ONE_MEG),
                canonicalize: true,
            },
        )
        .with_block_size_maximum(maximum);
        let stream = array.to_array_stream().sequenced(ptr);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(ctx.into(), segments, stream, eof, &session)
                .await
        })?;
        (0..layout.nchildren())
            .map(|i| {
                Ok(layout
                    .slot(i)?
                    .vortex_expect("chunk slot present")
                    .row_count())
            })
            .collect()
    }

    fn strings(rows: usize, width: usize) -> ArrayRef {
        VarBinViewArray::from_iter_str((0..rows).map(|row| format!("{row:0width$}"))).into_array()
    }

    #[test]
    fn a_block_size_maximum_splits_wide_strings_into_power_of_two_blocks() -> VortexResult<()> {
        // 16,384 rows of 1,000 bytes reference 1,016 bytes each (with their views): an 8,192-row
        // block is ~8 MiB, so a 1 MiB maximum fits 1,032 rows, rounded down to 1,024.
        let rows = block_rows_with_maximum(strings(16_384, 1000), Some(ONE_MEG))?;
        assert_eq!(rows, vec![1024; 16]);
        Ok(())
    }

    #[test]
    fn without_a_maximum_wide_strings_keep_the_row_multiple() -> VortexResult<()> {
        let rows = block_rows_with_maximum(strings(16_384, 1000), None)?;
        assert_eq!(rows, vec![8192, 8192]);
        Ok(())
    }

    #[test]
    fn narrow_strings_under_the_maximum_coalesce_as_without_it() -> VortexResult<()> {
        // 20-byte strings reference 8192 * 36 bytes, about 288 KiB, per 8,192-row block, far
        // under the maximum, so the blocks are whatever coalescing makes of them without it.
        // (Coalescing weighs a block by `nbytes()`, which counts the buffer it shares with the
        // other slices of the input, so here no two blocks merge.)
        let without = block_rows_with_maximum(strings(65_536, 20), None)?;
        let with = block_rows_with_maximum(strings(65_536, 20), Some(4 * ONE_MEG))?;
        assert_eq!(with, without);
        assert_eq!(with.iter().sum::<u64>(), 65_536);
        Ok(())
    }

    #[test]
    fn split_rows_measures_referenced_bytes_not_buffers() {
        // Rows sliced from a larger array share its buffers; only the bytes they point at count.
        let wide = strings(8192, 1000);
        let slice = wide.slice(0..64).vortex_expect("slice");
        assert_eq!(
            split_rows(&slice, ONE_MEG, 8192),
            None,
            "64 rows reference ~64 KiB"
        );
        assert_eq!(split_rows(&wide, ONE_MEG, 8192), Some(1024));
        // A maximum smaller than one row still yields blocks of one row.
        assert_eq!(split_rows(&wide, 100, 8192), Some(1));
        // Non-string chunks are never split.
        let ints = PrimitiveArray::from_iter(0..8192_i64).into_array();
        assert_eq!(split_rows(&ints, 1, 8192), None);
    }
}
