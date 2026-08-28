// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use moka::future::Cache;
use moka::future::CacheBuilder;
use moka::policy::EvictionPolicy;
use rustc_hash::FxBuildHasher;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_metrics::Counter;
use vortex_metrics::Label;
use vortex_metrics::MetricBuilder;
use vortex_metrics::MetricsRegistry;

use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Cache for individual segment byte buffers.
///
/// Caches are optional and operate above a [`SegmentSource`]. They should only store host buffers:
/// device buffers and other non-host handles should be passed through uncached.
#[async_trait]
pub trait SegmentCache: Send + Sync {
    /// Return a cached segment, or `None` on cache miss.
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>>;
    /// Store a segment in the cache.
    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()>;
}

/// Segment cache implementation that never stores anything.
pub struct NoOpSegmentCache;

#[async_trait]
impl SegmentCache for NoOpSegmentCache {
    async fn get(&self, _id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(None)
    }

    async fn put(&self, _id: SegmentId, _buffer: ByteBuffer) -> VortexResult<()> {
        Ok(())
    }
}

/// Bytes charged per entry on top of its buffer, covering the `Bytes` control block, the map
/// entry, and moka's per-entry policy bookkeeping.
///
/// Without this a cache of small segments overshoots badly: the cost is fixed per entry, so it
/// dominates once segments are a few hundred bytes. Measured at a flat 483 B/entry against real
/// retained heap, independent of segment size, by `per_entry_overhead_calibration` in
/// `vortex-file/tests/segment_cache_size_accounting.rs`; rounded up so the charge never
/// under-reports.
const ENTRY_OVERHEAD: usize = 512;

/// Heap bytes an admitted segment costs the cache: its own allocation plus per-entry overhead.
///
/// [`compact`] gives every value an allocation of `len + *alignment` bytes (see
/// [`vortex_buffer::BufferMut::with_capacity_preferred_aligned`]), so this is exact rather than an
/// estimate - which is the whole point of copying on admission.
fn charged_bytes(buffer: &ByteBuffer) -> usize {
    buffer.len() + *buffer.alignment() + ENTRY_OVERHEAD
}

/// Copy `buffer` into an allocation it exclusively owns, preserving its reported alignment.
///
/// Segments arrive as zero-copy slices of a coalesced read window: `CoalescedRequest::resolve`
/// cuts one physical read into a slice per segment, and `Buffer`'s backing `Bytes` keeps that
/// whole window alive while any slice of it lives. A cache storing such a slice therefore retains
/// the entire window - up to `CoalesceConfig::max_size`, 4MB for a local file and 16MB for object
/// storage - while `len()` reports only the slice.
///
/// That gap cannot be closed by weighing more accurately. `Buffer` maintains
/// `length * size_of::<T>() == bytes.len()`, and `Bytes` exposes no capacity, so the size of the
/// allocation a slice pins is not observable from the slice. Copying is what makes the weight
/// truthful: after this, the value owns its bytes and nothing else is retained on its behalf.
///
/// The copy is unconditional for the same reason - there is no way to ask whether a given buffer
/// is already the sole occupant of its allocation.
fn compact(buffer: &ByteBuffer) -> ByteBuffer {
    // `preferred_alignment: None` keeps the allocation at `len + *alignment` instead of
    // over-aligning to `Alignment::DEFAULT_ALIGNMENT` and charging 256 bytes of slack per entry.
    ByteBuffer::copy_from_preferred_aligned(buffer, buffer.alignment(), None)
}

/// A [`SegmentCache`] based around an in-memory Moka cache.
///
/// Segments are copied into their own allocation on admission; see [`compact`] for why, and
/// [`Self::new_sharing_windows`] for the variant that does not.
pub struct MokaSegmentCache {
    cache: Cache<SegmentId, ByteBuffer, FxBuildHasher>,
    compact_on_put: bool,
}

impl MokaSegmentCache {
    /// Construct a Moka-backed cache capped by total buffer bytes.
    ///
    /// Admitted segments are copied into their own allocation so that the cache retains exactly
    /// what it charges itself for. See [`compact`].
    pub fn new(max_capacity_bytes: u64) -> Self {
        Self {
            cache: Self::build(max_capacity_bytes),
            compact_on_put: true,
        }
    }

    /// Construct a Moka-backed cache that stores admitted segments as-is.
    ///
    /// Cheaper on admission - no copy - but a cached segment then retains the whole coalesced read
    /// window it was sliced from, so the cache's byte capacity stops bounding its memory. Provided
    /// for benchmarking the two admission strategies against each other; prefer [`Self::new`].
    pub fn new_sharing_windows(max_capacity_bytes: u64) -> Self {
        Self {
            cache: Self::build(max_capacity_bytes),
            compact_on_put: false,
        }
    }

    fn build(max_capacity_bytes: u64) -> Cache<SegmentId, ByteBuffer, FxBuildHasher> {
        CacheBuilder::new(max_capacity_bytes)
            .name("vortex-segment-cache")
            .weigher(|_, buffer: &ByteBuffer| {
                u32::try_from(charged_bytes(buffer).min(u32::MAX as usize))
                    .vortex_expect("must fit")
            })
            // We configure LFU (vs LRU) since the cache is mostly used when re-reading the
            // same file - it is _not_ used when reading the same segments during a single
            // scan.
            .eviction_policy(EvictionPolicy::tiny_lfu())
            .build_with_hasher(FxBuildHasher)
    }

    /// Total weight of the entries currently held, in bytes, once eviction has settled.
    pub fn weighted_size(&self) -> u64 {
        self.cache.weighted_size()
    }

    /// Number of entries currently held, once eviction has settled.
    pub fn entry_count(&self) -> u64 {
        self.cache.entry_count()
    }

    /// Apply any pending admission and eviction work.
    ///
    /// Moka performs that work in the background, so [`Self::weighted_size`] and
    /// [`Self::entry_count`] only reflect a settled cache after this resolves.
    pub async fn run_pending_tasks(&self) {
        self.cache.run_pending_tasks().await;
    }
}

#[async_trait]
impl SegmentCache for MokaSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.cache.get(&id).await)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        let buffer = if self.compact_on_put {
            compact(&buffer)
        } else {
            buffer
        };
        self.cache.insert(id, buffer).await;
        Ok(())
    }
}

/// Wrapper for [`SegmentCache`] that tracks its hit rate.
pub struct InstrumentedSegmentCache<C> {
    segment_cache: C,

    hits: Counter,
    misses: Counter,
    stores: Counter,
}

impl<C: SegmentCache> InstrumentedSegmentCache<C> {
    /// Wrap a segment cache and record hit/miss/store metrics with the supplied labels.
    pub fn new(
        segment_cache: C,
        metrics_registry: &dyn MetricsRegistry,
        labels: Vec<Label>,
    ) -> Self {
        Self {
            segment_cache,
            hits: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("vortex.file.segments.cache.hits"),
            misses: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("vortex.file.segments.cache.misses"),
            stores: MetricBuilder::new(metrics_registry)
                .add_labels(labels)
                .counter("vortex.file.segments.cache.stores"),
        }
    }
}

#[async_trait]
impl<C: SegmentCache> SegmentCache for InstrumentedSegmentCache<C> {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        let result = self.segment_cache.get(id).await?;
        if result.is_some() {
            self.hits.add(1);
        } else {
            self.misses.add(1);
        }
        Ok(result)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.segment_cache.put(id, buffer).await?;
        self.stores.add(1);
        Ok(())
    }
}

/// [`SegmentSource`] wrapper that consults a [`SegmentCache`] before the underlying source.
pub struct SegmentCacheSourceAdapter {
    cache: Arc<dyn SegmentCache>,
    source: Arc<dyn SegmentSource>,
}

impl SegmentCacheSourceAdapter {
    /// Construct a cache-fronted source.
    pub fn new(cache: Arc<dyn SegmentCache>, source: Arc<dyn SegmentSource>) -> Self {
        Self { cache, source }
    }
}

impl SegmentSource for SegmentCacheSourceAdapter {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let cache = Arc::clone(&self.cache);
        let delegate = self.source.request(id);

        async move {
            if let Ok(Some(segment)) = cache.get(id).await {
                tracing::debug!("Resolved segment {} from cache", id);
                return Ok(BufferHandle::new_host(segment));
            }
            let result = delegate.await?;
            // Cache only CPU buffers; device buffers are not cached.
            if let Some(buffer) = result.as_host_opt()
                && let Err(e) = cache.put(id, buffer.clone()).await
            {
                tracing::warn!("Failed to store segment {} in cache: {}", id, e);
            }
            Ok(result)
        }
        .boxed()
    }
}
