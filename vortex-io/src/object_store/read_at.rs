// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use bytes::Bytes;
use futures::FutureExt;
use futures::SinkExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResult;
use object_store::GetResultPayload;
use object_store::ObjectMeta;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::BufferAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::CoalesceConfig;
use crate::ReadAtRequest;
use crate::ReadAtStream;
use crate::VortexReadAt;
use crate::runtime::Handle;
#[cfg(not(target_arch = "wasm32"))]
use crate::std_file::read_exact_at;

/// Default number of concurrent requests to allow.
pub const DEFAULT_CONCURRENCY: usize = 192;

/// Marks waits on a remote object store without coupling Vortex to a query engine.
///
/// The callback receives the network request and its response byte stream separately. It must
/// preserve their outputs. In particular, it should not mark work that decodes a Vortex array or
/// copies bytes after a stream item is ready. Applications can leave the callback unset for local
/// and in-memory stores.
pub trait ObjectStoreIoWait: Send + Sync + 'static {
    /// Wrap a remote object metadata request.
    fn head(
        &self,
        future: BoxFuture<'static, object_store::Result<ObjectMeta>>,
    ) -> BoxFuture<'static, object_store::Result<ObjectMeta>>;

    /// Wrap a remote object range request.
    fn get(
        &self,
        future: BoxFuture<'static, object_store::Result<GetResult>>,
    ) -> BoxFuture<'static, object_store::Result<GetResult>>;

    /// Wrap each pending wait for the next bytes from a remote response.
    fn stream(
        &self,
        stream: BoxStream<'static, object_store::Result<Bytes>>,
    ) -> BoxStream<'static, object_store::Result<Bytes>>;
}

/// An object store backed I/O source.
pub struct ObjectStoreReadAt {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    uri: Arc<str>,
    handle: Handle,
    allocator: BufferAllocatorRef,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
    io_wait: Option<Arc<dyn ObjectStoreIoWait>>,
}

impl ObjectStoreReadAt {
    /// Create a new object store source.
    pub fn new(store: Arc<dyn ObjectStore>, path: ObjectPath, handle: Handle) -> Self {
        Self::new_with_allocator(
            store,
            path,
            handle,
            BufferAllocatorRef::statically_allocated(),
        )
    }

    /// Create a new object store source with a custom writable buffer allocator.
    pub fn new_with_allocator(
        store: Arc<dyn ObjectStore>,
        path: ObjectPath,
        handle: Handle,
        allocator: BufferAllocatorRef,
    ) -> Self {
        let uri = Arc::from(path.to_string());
        Self {
            store,
            path,
            uri,
            handle,
            allocator,
            concurrency: DEFAULT_CONCURRENCY,
            coalesce_config: Some(CoalesceConfig::object_storage()),
            io_wait: None,
        }
    }

    /// Mark remote object-store waits using the application's I/O-wait contract.
    /// Leave this unset for local and in-memory object stores.
    pub fn with_io_wait(mut self, io_wait: Arc<dyn ObjectStoreIoWait>) -> Self {
        self.io_wait = Some(io_wait);
        self
    }

    /// Set the concurrency for this source.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Set the coalesce config for this source.
    pub fn with_coalesce_config(mut self, config: CoalesceConfig) -> Self {
        self.coalesce_config = Some(config);
        self
    }
}

async fn read_object_store_range(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    io_handle: Handle,
    allocator: BufferAllocatorRef,
    request: ReadAtRequest,
    io_wait: Option<Arc<dyn ObjectStoreIoWait>>,
) -> VortexResult<BufferHandle> {
    let ReadAtRequest {
        offset,
        length,
        alignment,
    } = request;
    let range = offset..(offset + length as u64);
    let mut buffer = allocator.with_capacity_aligned::<u8>(length, alignment);
    // SAFETY: each return path checks that every byte was initialized.
    unsafe { buffer.set_len(length) };

    let get_range = range.clone();
    let get = async move {
        store
            .get_opts(
                &path,
                GetOptions {
                    range: Some(GetRange::Bounded(get_range)),
                    ..Default::default()
                },
            )
            .await
    };
    let response = match &io_wait {
        Some(marker) => marker.get(get.boxed()).await?,
        None => get.await?,
    };

    let buffer = match response.payload {
        #[cfg(not(target_arch = "wasm32"))]
        GetResultPayload::File(file, _) => io_handle
            .spawn_blocking(move || {
                read_exact_at(&file, buffer.as_mut_slice(), range.start)?;
                Ok::<_, io::Error>(buffer)
            })
            .await
            .map_err(io::Error::other)?,
        #[cfg(target_arch = "wasm32")]
        GetResultPayload::File(..) => {
            unreachable!("File payload not supported on wasm32")
        }
        GetResultPayload::Stream(byte_stream) => {
            let mut byte_stream = match &io_wait {
                Some(marker) => marker.stream(byte_stream),
                None => byte_stream,
            };
            let mut written = 0usize;
            while let Some(bytes) = byte_stream.next().await {
                let bytes = bytes?;
                let end = written + bytes.len();
                vortex_ensure!(
                    end <= length,
                    "Object store stream returned too many bytes: {} > expected {} (range: {:?})",
                    end,
                    length,
                    range
                );
                buffer.as_mut_slice()[written..end].copy_from_slice(&bytes);
                written = end;
            }

            vortex_ensure!(
                written == length,
                "Object store stream returned {} bytes but expected {} bytes (range: {:?})",
                written,
                length,
                range
            );

            buffer
        }
    };

    Ok(BufferHandle::new_host(buffer.freeze()))
}

impl VortexReadAt for ObjectStoreReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce_config
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let io_wait = self.io_wait.clone();
        async move {
            let head = async move { store.head(&path).await };
            let meta = match io_wait {
                Some(marker) => marker.head(head.boxed()).await?,
                None => head.await?,
            };
            Ok(meta.size)
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let handle = self.handle.clone();
        let allocator = self.allocator.clone();
        let io_handle = handle.clone();
        let io_wait = self.io_wait.clone();
        handle
            .spawn_io(read_object_store_range(
                store,
                path,
                io_handle,
                allocator,
                ReadAtRequest::new(offset, length, alignment),
                io_wait,
            ))
            .boxed()
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        if requests.is_empty() {
            return stream::empty().boxed();
        }

        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let handle = self.handle.clone();
        let allocator = self.allocator.clone();
        let io_wait = self.io_wait.clone();
        let concurrency = self.concurrency.max(1);
        let (mut send, recv) = mpsc::channel(concurrency);
        let io_handle = handle.clone();

        // A single runtime task drives all GETs, avoiding one spawn per range. Do not use
        // ObjectStore::get_ranges here: it returns one Vec after every range completes, whereas
        // VortexReadAt::read_ranges must expose each result as soon as it is ready.
        let task = handle.spawn_io(async move {
            let reads = requests.iter().copied().map(|request| {
                let store = Arc::clone(&store);
                let path = path.clone();
                let io_handle = io_handle.clone();
                let allocator = allocator.clone();
                let io_wait = io_wait.clone();
                async move {
                    let result = read_object_store_range(
                        store, path, io_handle, allocator, request, io_wait,
                    )
                    .await;
                    (request, result)
                }
            });

            let mut reads = stream::iter(reads).buffer_unordered(concurrency);
            while let Some(result) = reads.next().await {
                if send.send(result).await.is_err() {
                    break;
                }
            }
        });

        async_stream::stream! {
            let mut recv = recv;
            while let Some(result) = recv.next().await {
                yield result;
            }
            task.await;
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {

    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use object_store::PutPayload;
    use object_store::memory::InMemory;

    use super::*;
    use crate::runtime::AbortHandle;
    use crate::runtime::AbortHandleRef;
    use crate::runtime::Executor;

    const TEST_DATA: &[u8] = b"object store test data";

    #[derive(Default)]
    struct CountingExecutor {
        spawn_count: AtomicUsize,
        spawn_io_count: AtomicUsize,
    }

    #[derive(Default)]
    struct CountingIoWait {
        heads: AtomicUsize,
        gets: AtomicUsize,
        streams: AtomicUsize,
    }

    impl ObjectStoreIoWait for CountingIoWait {
        fn head(
            &self,
            future: BoxFuture<'static, object_store::Result<ObjectMeta>>,
        ) -> BoxFuture<'static, object_store::Result<ObjectMeta>> {
            self.heads.fetch_add(1, Ordering::SeqCst);
            future
        }

        fn get(
            &self,
            future: BoxFuture<'static, object_store::Result<GetResult>>,
        ) -> BoxFuture<'static, object_store::Result<GetResult>> {
            self.gets.fetch_add(1, Ordering::SeqCst);
            future
        }

        fn stream(
            &self,
            stream: BoxStream<'static, object_store::Result<Bytes>>,
        ) -> BoxStream<'static, object_store::Result<Bytes>> {
            self.streams.fetch_add(1, Ordering::SeqCst);
            stream
        }
    }

    impl Executor for CountingExecutor {
        fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
            self.spawn_count.fetch_add(1, Ordering::SeqCst);
            TokioAbortHandle::new_handle(tokio::spawn(fut).abort_handle())
        }

        fn spawn_io(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
            self.spawn_io_count.fetch_add(1, Ordering::SeqCst);
            TokioAbortHandle::new_handle(tokio::spawn(fut).abort_handle())
        }

        fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            TokioAbortHandle::new_handle(tokio::spawn(async move { task() }).abort_handle())
        }

        fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            TokioAbortHandle::new_handle(tokio::task::spawn_blocking(task).abort_handle())
        }
    }

    struct TokioAbortHandle(tokio::task::AbortHandle);

    impl TokioAbortHandle {
        fn new_handle(handle: tokio::task::AbortHandle) -> AbortHandleRef {
            Box::new(Self(handle))
        }
    }

    impl AbortHandle for TokioAbortHandle {
        fn abort(self: Box<Self>) {
            self.0.abort();
        }
    }

    #[tokio::test]
    async fn read_at_uses_spawn_io() -> anyhow::Result<()> {
        let executor = Arc::new(CountingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));

        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test.bin");
        store.put(&path, PutPayload::from_static(TEST_DATA)).await?;

        let reader = ObjectStoreReadAt::new(store, path, handle);
        let buffer = reader.read_at(7, 5, Alignment::new(1)).await?;

        assert_eq!(buffer.to_host().await.as_slice(), b"store");
        assert_eq!(executor.spawn_io_count.load(Ordering::SeqCst), 1);
        assert_eq!(executor.spawn_count.load(Ordering::SeqCst), 0);

        Ok(())
    }

    #[tokio::test]
    async fn read_ranges_uses_one_io_task() -> anyhow::Result<()> {
        let executor = Arc::new(CountingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));

        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test.bin");
        store.put(&path, PutPayload::from_static(TEST_DATA)).await?;

        let reader = ObjectStoreReadAt::new(store, path, handle);
        let requests: Arc<[ReadAtRequest]> = Arc::from([
            ReadAtRequest::new(0, 6, Alignment::new(1)),
            ReadAtRequest::new(7, 5, Alignment::new(1)),
            ReadAtRequest::new(18, 4, Alignment::new(1)),
        ]);
        let results = reader.read_ranges(requests).collect::<Vec<_>>().await;

        assert_eq!(results.len(), 3);
        for (request, result) in results {
            let buffer = result?;
            let offset = usize::try_from(request.offset)?;
            assert_eq!(buffer.len(), request.length);
            assert_eq!(
                buffer.to_host().await.as_slice(),
                &TEST_DATA[offset..offset + request.length]
            );
        }
        assert_eq!(executor.spawn_io_count.load(Ordering::SeqCst), 1);
        assert_eq!(executor.spawn_count.load(Ordering::SeqCst), 0);

        Ok(())
    }

    #[tokio::test]
    async fn io_wait_marker_covers_head_and_both_range_read_paths() -> anyhow::Result<()> {
        let executor = Arc::new(CountingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test.bin");
        store.put(&path, PutPayload::from_static(TEST_DATA)).await?;

        let marker = Arc::new(CountingIoWait::default());
        let reader = ObjectStoreReadAt::new(store, path, handle)
            .with_io_wait(Arc::clone(&marker) as Arc<dyn ObjectStoreIoWait>);
        assert_eq!(reader.size().await?, TEST_DATA.len() as u64);
        assert_eq!(reader.read_at(7, 5, Alignment::new(1)).await?.len(), 5);
        let requests: Arc<[ReadAtRequest]> = Arc::from([
            ReadAtRequest::new(0, 6, Alignment::new(1)),
            ReadAtRequest::new(7, 5, Alignment::new(1)),
        ]);
        let results = reader.read_ranges(requests).collect::<Vec<_>>().await;
        assert_eq!(results.len(), 2);
        for (_, result) in results {
            result?;
        }

        assert_eq!(marker.heads.load(Ordering::SeqCst), 1);
        assert_eq!(marker.gets.load(Ordering::SeqCst), 3);
        assert_eq!(marker.streams.load(Ordering::SeqCst), 3);
        Ok(())
    }
}
