// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::iter;
use std::ops::Range;
use std::sync::Arc;

use futures::Stream;
use futures::future::BoxFuture;
use futures::future::Either as EitherFuture;
use itertools::Either;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;
use crate::scan::scan_builder::SplitConcurrency;
use crate::scan::splits::Splits;
use crate::scan::tasks::TaskContext;
use crate::scan::tasks::TaskFuture;
use crate::scan::tasks::split_exec;

/// A projected subset (by indices, range, and filter) of rows from a Vortex data source.
///
/// The method of this struct enable, possibly concurrent, scanning of multiple row ranges of this
/// data source.
pub struct RepeatedScan<A: 'static + Send> {
    session: VortexSession,
    layout_reader: LayoutReaderRef,
    projection: Expression,
    filter: Option<Expression>,
    ordered: bool,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    selection: Selection,
    /// The natural splits of the file.
    splits: Splits,
    /// How many splits to make progress on concurrently.
    concurrency: SplitConcurrency,
    /// Function to apply to each [`ArrayRef`] within the spawned split tasks.
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    /// Maximal number of rows to read (after filtering)
    limit: Option<u64>,
    /// The dtype of the projected arrays.
    dtype: DType,
}

impl RepeatedScan<ArrayRef> {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn execute_array_iter<B: BlockingRuntime>(
        &self,
        row_range: Option<Range<u64>>,
        runtime: &B,
    ) -> VortexResult<impl ArrayIterator + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream(row_range)?;
        let iter = runtime.block_on_stream(stream);
        Ok(ArrayIteratorAdapter::new(dtype, iter))
    }

    pub fn execute_array_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl ArrayStream + Send + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream(row_range)?;
        Ok(ArrayStreamAdapter::new(dtype, stream))
    }
}

impl<A: 'static + Send> RepeatedScan<A> {
    /// Constructor just to allow `scan_builder` to create a `RepeatedScan`.
    #[expect(
        clippy::too_many_arguments,
        reason = "all arguments are needed for scan construction"
    )]
    pub fn new(
        session: VortexSession,
        layout_reader: LayoutReaderRef,
        projection: Expression,
        filter: Option<Expression>,
        ordered: bool,
        row_range: Option<Range<u64>>,
        selection: Selection,
        splits: Splits,
        concurrency: SplitConcurrency,
        map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
        limit: Option<u64>,
        dtype: DType,
    ) -> Self {
        Self {
            session,
            layout_reader,
            projection,
            filter,
            ordered,
            row_range,
            selection,
            splits,
            concurrency,
            map_fn,
            limit,
            dtype,
        }
    }

    /// Constructs a task per row split of the scan, returned as a vector of futures.
    ///
    /// Note that this registers the reads of *every* split up-front. Prefer
    /// [`Self::execute_stream`], which builds the same tasks lazily so that the configured
    /// concurrency bounds how many splits have reads in flight.
    pub fn execute(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        self.split_tasks(row_range).collect()
    }

    /// The row range of each split this scan will read, in file order.
    ///
    /// Computing every range up-front is cheap — it only intersects ranges — unlike
    /// constructing the split tasks, which registers I/O. See [`Self::split_tasks`].
    fn split_ranges(&self, row_range: Option<Range<u64>>) -> Vec<Range<u64>> {
        let selection_range: Option<Range<u64>> = match &self.selection {
            Selection::IncludeByIndex(buf) if !buf.is_empty() => {
                Some(buf[0]..buf[buf.len() - 1] + 1)
            }
            Selection::IncludeRoaring(roaring) if !roaring.is_empty() => {
                Some(roaring.min().vortex_expect("empty")..roaring.max().vortex_expect("empty") + 1)
            }
            _ => None,
        };
        let row_range = intersect_ranges(self.row_range.as_ref(), row_range);
        let row_range = intersect_ranges(row_range.as_ref(), selection_range);

        let ranges = match &self.splits {
            Splits::Natural(vec) => {
                debug_assert!(vec.is_sorted());
                let splits_iter = match row_range {
                    None => Either::Left(vec.iter().copied()),
                    Some(range) => {
                        if range.is_empty() {
                            return Vec::new();
                        }
                        let lo = vec.partition_point(|&x| x < range.start);
                        let hi = vec.partition_point(|&x| x < range.end);
                        Either::Right(
                            iter::once(range.start)
                                .chain(vec[lo..hi].iter().copied())
                                .chain(iter::once(range.end)),
                        )
                    }
                };

                Either::Left(splits_iter.tuple_windows().map(|(start, end)| start..end))
            }
            Splits::Ranges(ranges) => Either::Right(match row_range {
                None => Either::Left(ranges.iter().cloned()),
                Some(range) => {
                    if range.is_empty() {
                        return Vec::new();
                    }
                    Either::Right(ranges.iter().filter_map(move |r| {
                        let start = cmp::max(r.start, range.start);
                        let end = cmp::min(r.end, range.end);
                        (start < end).then_some(start..end)
                    }))
                }
            }),
        };

        ranges.collect()
    }

    /// Builds one split task at a time, in file order.
    ///
    /// Constructing a task registers its reads with the I/O system — that eager registration is
    /// deliberate prefetch, see [`split_exec`] — so this iterator must stay lazy. Whatever
    /// buffers it is what bounds the splits with reads in flight: [`Self::execute_stream`]
    /// buffers by the scan's [`SplitConcurrency`], while [`Self::execute`] collects it and so
    /// registers every split at once.
    ///
    /// A split whose task fails to construct yields the error and ends the iterator.
    fn split_tasks(
        &self,
        row_range: Option<Range<u64>>,
    ) -> impl Iterator<Item = VortexResult<TaskFuture<Option<A>>>> + Send + 'static + use<A> {
        let mut ranges = self.split_ranges(row_range).into_iter();
        let selection = self.selection.clone();
        let ctx = Arc::new(TaskContext {
            filter: self.filter.clone().map(|f| Arc::new(FilterExpr::new(f))),
            reader: Arc::clone(&self.layout_reader),
            projection: self.projection.clone(),
            mapper: Arc::clone(&self.map_fn),
        });

        let mut limit = self.limit;
        let mut finished = false;

        iter::from_fn(move || {
            loop {
                if finished {
                    return None;
                }

                let range = ranges.next()?;
                let row_mask = selection.row_mask(&range);
                if row_mask.mask().all_false() {
                    continue;
                }

                let task = split_exec(Arc::clone(&ctx), row_mask, limit.as_mut());
                if task.is_err() || limit.is_some_and(|l| l == 0) {
                    finished = true;
                }
                return Some(task);
            }
        })
    }

    /// Streams the scan, holding at most [`SplitConcurrency::effective`] splits in flight.
    ///
    /// Splits are constructed lazily under that bound, so a split's reads are not registered
    /// until it is admitted.
    pub fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        use futures::StreamExt;
        let concurrency = self.concurrency.effective();
        let handle = self.session.handle();

        let stream =
            futures::stream::iter(self.split_tasks(row_range)).map(move |task| match task {
                Ok(task) => EitherFuture::Left(handle.spawn(task)),
                Err(err) => EitherFuture::Right(futures::future::ready(Err(err))),
            });

        let stream = if self.ordered {
            stream.buffered(concurrency).boxed()
        } else {
            stream.buffer_unordered(concurrency).boxed()
        };

        Ok(stream.filter_map(|chunk| async move { chunk.transpose() }))
    }
}

fn intersect_ranges(left: Option<&Range<u64>>, right: Option<Range<u64>>) -> Option<Range<u64>> {
    match (left, right) {
        (None, None) => None,
        (None, Some(r)) => Some(r),
        (Some(l), None) => Some(l.clone()),
        (Some(l), Some(r)) => Some(cmp::max(l.start, r.start)..cmp::min(l.end, r.end)),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_error::VortexResult;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::single::SingleThreadRuntime;

    use crate::scan::scan_builder::ScanBuilder;
    use crate::scan::test::ProbeLayoutReader;
    use crate::scan::test::SplitProbe;
    use crate::scan::test::assert_bounded_scan;
    use crate::scan::test::session_with_handle;

    /// The configured concurrency must bound the splits whose reads have been registered, so
    /// split tasks have to be constructed lazily under the buffer rather than all up-front.
    /// `1` is genuinely serial.
    #[rstest]
    #[case::serial(1, true)]
    #[case::serial_unordered(1, false)]
    #[case::bounded(3, true)]
    #[case::bounded_unordered(3, false)]
    fn absolute_concurrency_bounds_registered_splits(
        #[case] concurrency: usize,
        #[case] ordered: bool,
    ) -> VortexResult<()> {
        const ROW_COUNT: usize = 8;

        let mut ctx = array_session().create_execution_ctx();
        let probe = Arc::new(SplitProbe::default());
        let reader = Arc::new(ProbeLayoutReader::new(ROW_COUNT, Arc::clone(&probe)));

        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());

        let scan = ScanBuilder::new(session, reader)
            .with_absolute_concurrency(concurrency)
            .with_ordered(ordered)
            .prepare()?;

        let mut values = Vec::new();
        for (yielded, chunk) in runtime
            .block_on_stream(scan.execute_stream(None)?)
            .enumerate()
        {
            let prim = chunk?.execute::<PrimitiveArray>(&mut ctx)?;
            values.extend(prim.into_buffer::<i32>().iter().copied());

            let admitted = yielded + 1 + concurrency;
            assert!(
                probe.registered() <= admitted,
                "{} splits registered after {} yielded, expected at most {admitted}",
                probe.registered(),
                yielded + 1,
            );
        }

        if !ordered {
            values.sort_unstable();
        }
        assert_bounded_scan(&values, ROW_COUNT, &probe, concurrency);

        Ok(())
    }
}
