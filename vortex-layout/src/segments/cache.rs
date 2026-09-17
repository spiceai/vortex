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

/// A [`SegmentCache`] based around an in-memory Moka cache.
pub struct MokaSegmentCache(Cache<SegmentId, ByteBuffer, FxBuildHasher>);

impl MokaSegmentCache {
    /// Construct a Moka-backed cache capped by total buffer bytes.
    pub fn new(max_capacity_bytes: u64) -> Self {
        Self(
            CacheBuilder::new(max_capacity_bytes)
                .name("vortex-segment-cache")
                // Weight each segment by the number of bytes in the buffer.
                .weigher(|_, buffer: &ByteBuffer| {
                    u32::try_from(buffer.len().min(u32::MAX as usize)).vortex_expect("must fit")
                })
                // We configure LFU (vs LRU) since the cache is mostly used when re-reading the
                // same file - it is _not_ used when reading the same segments during a single
                // scan.
                .eviction_policy(EvictionPolicy::tiny_lfu())
                .build_with_hasher(FxBuildHasher),
        )
    }
}

#[async_trait]
impl SegmentCache for MokaSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.0.get(&id).await)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.0.insert(id, buffer).await;
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
        let source = Arc::clone(&self.source);

        async move {
            if let Ok(Some(segment)) = cache.get(id).await {
                tracing::debug!("Resolved segment {} from cache", id);
                return Ok(BufferHandle::new_host(segment));
            }
            let result = source.request(id).await?;
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_error::vortex_err;

    use super::*;

    #[derive(Default)]
    struct CountingSource {
        requests: AtomicUsize,
    }

    impl SegmentSource for CountingSource {
        fn request(&self, _id: SegmentId) -> SegmentFuture {
            self.requests.fetch_add(1, Ordering::SeqCst);
            futures::future::ready(Ok(BufferHandle::new_host(ByteBuffer::from(vec![
                1u8, 2, 3,
            ]))))
            .boxed()
        }
    }

    #[tokio::test]
    async fn cache_hit_does_not_request_the_source() -> VortexResult<()> {
        let cache: Arc<dyn SegmentCache> = Arc::new(MokaSegmentCache::new(1024));
        let id = SegmentId::from(0);
        cache.put(id, ByteBuffer::from(vec![9u8, 8, 7])).await?;
        let source = Arc::new(CountingSource::default());
        let adapter = SegmentCacheSourceAdapter::new(cache, Arc::<CountingSource>::clone(&source));

        let request = adapter.request(id);
        assert_eq!(source.requests.load(Ordering::SeqCst), 0);
        let result = request.await?.to_host().await;
        assert_eq!(result.as_slice(), &[9, 8, 7]);
        assert_eq!(source.requests.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn miss_reads_and_populates_the_cache() -> VortexResult<()> {
        let cache: Arc<dyn SegmentCache> = Arc::new(MokaSegmentCache::new(1024));
        let source = Arc::new(CountingSource::default());
        let adapter = SegmentCacheSourceAdapter::new(cache, Arc::<CountingSource>::clone(&source));
        let id = SegmentId::from(0);

        for _ in 0..2 {
            assert_eq!(
                adapter.request(id).await?.to_host().await.as_slice(),
                &[1, 2, 3]
            );
        }
        assert_eq!(source.requests.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn dropping_an_unpolled_request_does_not_request_the_source() {
        let source = Arc::new(CountingSource::default());
        let adapter = SegmentCacheSourceAdapter::new(
            Arc::new(NoOpSegmentCache),
            Arc::<CountingSource>::clone(&source),
        );
        drop(adapter.request(SegmentId::from(0)));
        assert_eq!(source.requests.load(Ordering::SeqCst), 0);
    }

    struct UnavailableCache;

    #[async_trait]
    impl SegmentCache for UnavailableCache {
        async fn get(&self, _id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
            Err(vortex_err!("cache unavailable"))
        }

        async fn put(&self, _id: SegmentId, _buffer: ByteBuffer) -> VortexResult<()> {
            Err(vortex_err!("cache unavailable"))
        }
    }

    #[tokio::test]
    async fn unavailable_cache_preserves_source_results() -> VortexResult<()> {
        let source = Arc::new(CountingSource::default());
        let adapter = SegmentCacheSourceAdapter::new(
            Arc::new(UnavailableCache),
            Arc::<CountingSource>::clone(&source),
        );
        let result = adapter.request(SegmentId::from(0)).await?.to_host().await;
        assert_eq!(result.as_slice(), &[1, 2, 3]);
        assert_eq!(source.requests.load(Ordering::SeqCst), 1);
        Ok(())
    }
}
