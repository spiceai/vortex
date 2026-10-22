// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use futures::StreamExt as _;
use futures::stream;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::expr::stats::Stat;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// A boxed compressor function from arrays into compressed arrays.
///
/// API consumers are free to implement this trait to provide new plugin compressors.
pub trait CompressorPlugin: Send + Sync + 'static {
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>;

    /// Compresses consecutive chunks of one column, returning one compressed chunk per input.
    ///
    /// The chunks are drawn from the same data, so a compressor may choose its encodings once
    /// and reuse them across the group. The default compresses each chunk on its own.
    ///
    /// A wrapper must forward this method to the compressor it wraps: inheriting the default
    /// silently drops whatever reuse the inner compressor does.
    fn compress_chunks(
        &self,
        chunks: &[ArrayRef],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<ArrayRef>> {
        chunks
            .iter()
            .map(|chunk| self.compress_chunk(chunk, ctx))
            .collect()
    }
}

impl CompressorPlugin for Arc<dyn CompressorPlugin> {
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.as_ref().compress_chunk(chunk, ctx)
    }

    fn compress_chunks(
        &self,
        chunks: &[ArrayRef],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<ArrayRef>> {
        self.as_ref().compress_chunks(chunks, ctx)
    }
}

impl<F> CompressorPlugin for F
where
    F: Fn(&ArrayRef, &mut ExecutionCtx) -> VortexResult<ArrayRef> + Send + Sync + 'static,
{
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self(chunk, ctx)
    }
}

impl CompressorPlugin for BtrBlocksCompressor {
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.compress(chunk, ctx)
    }

    /// Searches for encodings once for the whole group and applies them to every chunk.
    ///
    /// Choosing an encoding compresses a sample once per candidate, and a chunk of at most
    /// 1,024 rows is its own sample, so searching every small chunk costs several full
    /// compressions each. The search runs on the sample the compressor would draw from the
    /// group as one array, so each chunk is encoded as the unsplit group would have been. A
    /// search on the first chunk alone fits only that chunk: when it differs from the rest
    /// (incompressible rows ahead of JSON, say), the rest are encoded to suit it.
    fn compress_chunks(
        &self,
        chunks: &[ArrayRef],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<ArrayRef>> {
        let [first, ..] = chunks else {
            return Ok(Vec::new());
        };
        if chunks.len() == 1 {
            return Ok(vec![self.compress(first, ctx)?]);
        }
        let group = ChunkedArray::try_new(chunks.to_vec(), first.dtype().clone())?.into_array();
        let plan = self.plan_for(&group, ctx)?;
        chunks
            .iter()
            .map(|chunk| self.compress_with_plan(chunk, &plan, ctx))
            .collect()
    }
}

/// A layout writer that compresses chunks.
#[derive(Clone)]
pub struct CompressingStrategy {
    child: Arc<dyn LayoutStrategy>,
    compressor: Arc<dyn CompressorPlugin>,
    stats: Arc<[Stat]>,
    concurrency: usize,
    /// Consecutive chunks are compressed together until they hold this many rows; see
    /// [`Self::with_group_rows`].
    group_rows: usize,
}

impl CompressingStrategy {
    /// Create a new compressing layout strategy with the given child strategy and compressor.
    pub fn new<S: LayoutStrategy, C: CompressorPlugin>(child: S, compressor: C) -> Self {
        Self {
            child: Arc::new(child),
            compressor: Arc::new(compressor),
            stats: Stat::all().collect(),
            concurrency: get_available_parallelism().unwrap_or(1),
            group_rows: 0,
        }
    }

    /// Compress consecutive chunks together, through [`CompressorPlugin::compress_chunks`],
    /// until a group holds at least `rows` rows.
    ///
    /// A chunk that holds `rows` or more on its own is compressed alone, as without grouping.
    /// The default, 0, compresses every chunk alone.
    pub fn with_group_rows(mut self, rows: usize) -> Self {
        self.group_rows = rows;
        self
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Override the set of statistics computed on each chunk before compression.
    /// Defaults to `Stat::all()`.
    pub fn with_stats(mut self, stats: &[Stat]) -> Self {
        self.stats = stats.into();
        self
    }
}

#[async_trait]
impl LayoutStrategy for CompressingStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let compressor = Arc::clone(&self.compressor);
        let stats = Arc::clone(&self.stats);
        let session = session.clone();
        let compute_session = session.clone();

        let handle = session.handle();
        let stream = group_chunks(stream, self.group_rows)
            .map(move |group| {
                let compressor = Arc::clone(&compressor);
                let stats = Arc::clone(&stats);
                let session = compute_session.clone();
                handle.spawn_cpu(move || {
                    let group = group?;
                    let mut ctx = session.create_execution_ctx();
                    let (sequence_ids, chunks): (Vec<_>, Vec<_>) = group.into_iter().unzip();
                    // Compute the stats for each chunk prior to compression
                    for chunk in &chunks {
                        chunk.statistics().compute_all(&stats, &mut ctx)?;
                    }
                    let compressed = compressor.compress_chunks(&chunks, &mut ctx)?;
                    vortex_ensure!(
                        compressed.len() == chunks.len(),
                        "compressor returned {} chunks for {} inputs",
                        compressed.len(),
                        chunks.len()
                    );
                    Ok(sequence_ids.into_iter().zip(compressed).collect::<Vec<_>>())
                })
            })
            .buffered(self.concurrency)
            .flat_map(|group: VortexResult<Vec<_>>| {
                stream::iter(match group {
                    Ok(chunks) => chunks.into_iter().map(Ok).collect::<Vec<_>>(),
                    Err(err) => vec![Err(err)],
                })
            });

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, stream).sendable(),
                eof,
                &session,
            )
            .await
    }
}

/// Groups consecutive chunks of `stream` that each hold fewer than `rows` rows, until a group
/// holds at least `rows` rows. A chunk of `rows` or more rows is always a group of its own, so
/// grouping never replays encodings onto a chunk large enough to search on its own. With `rows`
/// of 0 or 1 every chunk is its own group. An error ends the group in progress, which is
/// emitted first.
fn group_chunks(
    stream: SendableSequentialStream,
    rows: usize,
) -> impl Stream<Item = VortexResult<Vec<(SequenceId, ArrayRef)>>> + Send {
    type Item = VortexResult<(SequenceId, ArrayRef)>;
    struct State {
        stream: SendableSequentialStream,
        group: Vec<(SequenceId, ArrayRef)>,
        group_rows: usize,
        /// An item read past the end of the group that was emitted, to start the next one.
        carry: Option<Item>,
    }
    stream::unfold(
        State {
            stream,
            group: Vec::new(),
            group_rows: 0,
            carry: None,
        },
        move |mut state| async move {
            loop {
                let next = match state.carry.take() {
                    Some(item) => Some(item),
                    None => state.stream.next().await,
                };
                match next {
                    Some(Ok((sequence_id, chunk))) => {
                        if chunk.len() >= rows && !state.group.is_empty() {
                            state.carry = Some(Ok((sequence_id, chunk)));
                            state.group_rows = 0;
                            return Some((Ok(std::mem::take(&mut state.group)), state));
                        }
                        state.group_rows += chunk.len();
                        state.group.push((sequence_id, chunk));
                        if state.group_rows >= rows {
                            state.group_rows = 0;
                            return Some((Ok(std::mem::take(&mut state.group)), state));
                        }
                    }
                    Some(Err(err)) => {
                        if state.group.is_empty() {
                            return Some((Err(err), state));
                        }
                        state.carry = Some(Err(err));
                        state.group_rows = 0;
                        return Some((Ok(std::mem::take(&mut state.group)), state));
                    }
                    None if state.group.is_empty() => return None,
                    None => {
                        state.group_rows = 0;
                        return Some((Ok(std::mem::take(&mut state.group)), state));
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_error::VortexExpect;
    use vortex_error::vortex_err;
    use vortex_io::runtime::single::block_on;

    use super::*;
    use crate::sequence::SequentialArrayStreamExt;

    fn group_lengths(lengths: &[usize], rows: usize) -> Vec<Vec<usize>> {
        let chunks = lengths
            .iter()
            .map(|&len| PrimitiveArray::from_iter(0..len as i64).into_array())
            .collect::<Vec<_>>();
        let array = ChunkedArray::try_new(chunks, DType::Primitive(PType::I64, NonNullable))
            .vortex_expect("chunked")
            .into_array();
        let (ptr, _eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        block_on(|_| async move {
            group_chunks(stream, rows)
                .map(|group| {
                    group
                        .vortex_expect("group")
                        .iter()
                        .map(|(_, chunk)| chunk.len())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
                .await
        })
    }

    #[test]
    fn small_chunks_group_until_the_row_count_and_large_chunks_stay_alone() {
        assert_eq!(
            group_lengths(&[100, 100, 9000, 50, 8000, 8192, 4096, 4096, 10], 8192),
            vec![
                vec![100, 100],
                vec![9000],
                vec![50, 8000],
                vec![8192],
                vec![4096, 4096],
                vec![10]
            ],
        );
    }

    #[test]
    fn without_a_row_count_every_chunk_is_its_own_group() {
        assert_eq!(
            group_lengths(&[100, 100, 9000], 0),
            vec![vec![100], vec![100], vec![9000]]
        );
    }

    #[test]
    fn an_error_ends_the_group_and_is_emitted_after_it() {
        let dtype = DType::Primitive(PType::I64, NonNullable);
        let (mut ptr, _eof) = SequenceId::root().split();
        let items: Vec<VortexResult<(SequenceId, ArrayRef)>> = vec![
            Ok((
                ptr.advance(),
                PrimitiveArray::from_iter(0..10_i64).into_array(),
            )),
            Err(vortex_err!("read failed")),
            Ok((
                ptr.advance(),
                PrimitiveArray::from_iter(0..10_i64).into_array(),
            )),
        ];
        let stream = SequentialStreamAdapter::new(dtype, stream::iter(items)).sendable();
        let groups = block_on(|_| async move {
            group_chunks(stream, 8192)
                .map(|group| group.map(|g| g.len()).map_err(|e| e.to_string()))
                .collect::<Vec<_>>()
                .await
        });
        assert_eq!(groups.len(), 3, "{groups:?}");
        assert_eq!(groups[0], Ok(1));
        assert!(
            groups[1].as_ref().is_err_and(|e| e.contains("read failed")),
            "{groups:?}"
        );
        assert_eq!(groups[2], Ok(1));
    }

    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// ~1.5 KB rows of hex digits of random words: no dictionary or symbol table built on
    /// them fits other data.
    fn hex_rows(rows: usize, seed: u64) -> ArrayRef {
        let mut state = seed;
        vortex_array::arrays::VarBinViewArray::from_iter_str((0..rows).map(|_| {
            (0..96)
                .map(|_| format!("{:016x}", next(&mut state)))
                .collect::<String>()
        }))
        .into_array()
    }

    /// ~1.5 KB JSON documents of repeated keys and words with varying numbers.
    fn documents(rows: usize, seed: u64) -> ArrayRef {
        const WORDS: [&str; 8] = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        ];
        let mut state = seed;
        vortex_array::arrays::VarBinViewArray::from_iter_str((0..rows).map(|i| {
            let mut s = format!("{{\"id\":{i},\"items\":[");
            while s.len() < 1_500 {
                let v = next(&mut state);
                s.push_str(&format!(
                    "{{\"k\":\"{}-{}\",\"n\":{},\"t\":\"{} {}\"}},",
                    WORDS[(v % 8) as usize],
                    (v >> 8) % 10_000,
                    (v >> 24) % 1_000_000,
                    WORDS[((v >> 44) % 8) as usize],
                    (v >> 50) % 97
                ));
            }
            s.push_str("]}");
            s
        }))
        .into_array()
    }

    /// A group whose first chunk is unlike the rest is encoded as the unsplit group would be:
    /// the schemes are chosen on a sample of the whole group, not on its first chunk. Here the
    /// first chunk alone picks FSST, and replaying that on the documents instead of the
    /// group's own choice leaves them larger and slower to filter.
    #[test]
    fn a_group_is_encoded_as_its_unsplit_data_would_be() -> VortexResult<()> {
        let mut ctx = crate::test::SESSION.create_execution_ctx();
        let mut chunks = vec![hex_rows(1024, 7)];
        chunks.extend((1..8).map(|i| documents(1024, i)));
        let compressor = BtrBlocksCompressor::default();

        let unsplit =
            ChunkedArray::try_new(chunks.clone(), chunks[0].dtype().clone())?.into_array();
        let expected = compressor.compress(&unsplit, &mut ctx)?.encoding_id();
        // The premise: the first chunk on its own chooses differently.
        let (first, plan) = compressor.compress_recording(&chunks[0], &mut ctx)?;
        assert_ne!(first.encoding_id(), expected);
        assert_eq!(
            compressor
                .compress_with_plan(&chunks[1], &plan, &mut ctx)?
                .encoding_id(),
            first.encoding_id()
        );

        let grouped = compressor.compress_chunks(&chunks, &mut ctx)?;
        for (compressed, chunk) in grouped.iter().zip(&chunks) {
            assert_eq!(compressed.encoding_id(), expected);
            vortex_array::assert_arrays_eq!(compressed.clone(), chunk.clone(), &mut ctx);
        }
        Ok(())
    }
}
