// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex file footer metadata.
//!
//! A footer contains the root layout, file-level statistics, the segment map, and the read contexts
//! needed to resolve array/layout encoding ids during deserialization.
//!
//! The byte-level footer and postscript layout is part of the file-format spec; this module exposes
//! the structured Rust representation and serializer/deserializer state machine.
mod field_sizes;
mod file_layout;
mod file_statistics;
mod postscript;
mod segment;

use std::sync::Arc;

mod serializer;
pub use serializer::*;
mod deserializer;
pub use deserializer::*;
pub use field_sizes::CompressedFieldSizes;
pub use file_statistics::FileStatistics;
use flatbuffers::root;
use itertools::Itertools;
pub use segment::*;
use vortex_array::ArrayId;
use vortex_array::dtype::DType;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::footer as fb;
use vortex_layout::LayoutEncodingId;
use vortex_layout::LayoutRef;
use vortex_layout::layout_from_flatbuffer_with_options;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

/// Bytes of slack a [`ByteBuffer`] allocation carries beyond its payload, from the default
/// over-alignment applied by
/// [`BufferMut::with_capacity_preferred_aligned`](vortex_buffer::BufferMut::with_capacity_preferred_aligned).
fn buffer_alloc_slack() -> usize {
    *Alignment::DEFAULT_ALIGNMENT
}

/// Captures the layout information of a Vortex file.
#[derive(Debug, Clone)]
pub struct Footer {
    root_layout: LayoutRef,
    segments: Arc<[SegmentSpec]>,
    statistics: Option<FileStatistics>,
    // The specific arrays used within the file, in the order they were registered.
    array_read_ctx: ReadContext,
    // The approximate size of the footer in bytes, used for caching and memory management.
    approx_byte_size: Option<usize>,
}

impl Footer {
    pub fn new(
        root_layout: LayoutRef,
        segments: Arc<[SegmentSpec]>,
        statistics: Option<FileStatistics>,
        array_read_ctx: ReadContext,
    ) -> Self {
        Self {
            root_layout,
            segments,
            statistics,
            array_read_ctx,
            approx_byte_size: None,
        }
    }

    pub(crate) fn with_approx_byte_size(mut self, approx_byte_size: usize) -> Self {
        self.approx_byte_size = Some(approx_byte_size);
        self
    }

    /// Record [`Self::approx_byte_size`] from the sizes only the caller knows: the serialized
    /// footer segments it wrote, and the layout tree those segments describe.
    pub(crate) fn with_approx_retained_size(
        self,
        serialized_bytes: usize,
        layout_tree_bytes: usize,
    ) -> Self {
        let size = self.approx_retained_size(serialized_bytes, layout_tree_bytes);
        self.with_approx_byte_size(size)
    }

    /// Read the [`Footer`] from a flatbuffer.
    pub(crate) fn from_flatbuffer(
        footer_bytes: FlatBuffer,
        layout_bytes: FlatBuffer,
        dtype_bytes_len: usize,
        dtype: DType,
        statistics: Option<FileStatistics>,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        // The footer keeps a private copy of each flatbuffer segment it parsed, and every copy
        // carries the over-alignment slack `BufferMut` adds. A file written with
        // `exclude_dtype` has no dtype segment.
        let dtype_buffers = usize::from(dtype_bytes_len > 0);
        let serialized_bytes = footer_bytes.len()
            + layout_bytes.len()
            + dtype_bytes_len
            + (2 + dtype_buffers) * buffer_alloc_slack();
        let fb_footer = root::<fb::Footer>(&footer_bytes)?;

        // Create a LayoutContext from the registry.
        let layout_specs = fb_footer.layout_specs();
        #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
        let layout_ids: Arc<[_]> = layout_specs
            .iter()
            .flat_map(|e| e.iter())
            .map(|encoding| LayoutEncodingId::new(encoding.id()))
            .collect();
        let layout_read_ctx = ReadContext::new(layout_ids);

        // Create an ArrayContext from the registry.
        let array_specs = fb_footer.array_specs();
        #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
        let array_ids: Arc<[_]> = array_specs
            .iter()
            .flat_map(|e| e.iter())
            .map(|encoding| ArrayId::new(encoding.id()))
            .collect();
        let array_read_ctx = ReadContext::new(array_ids);

        // `layout_tree_bytes` budgets for the layout tree, which is built lazily but must be
        // charged up front - see `Self::approx_byte_size`.
        let (root_layout, layout_tree_bytes) = layout_from_flatbuffer_with_options(
            layout_bytes,
            &dtype,
            &layout_read_ctx,
            &array_read_ctx,
            session,
            session.allows_unknown(),
        )?;

        let segments: Arc<[SegmentSpec]> = fb_footer
            .segment_specs()
            .ok_or_else(|| vortex_err!("FileLayout missing segment specs"))?
            .iter()
            .map(SegmentSpec::try_from)
            .try_collect()?;

        // Note this assertion is `<=` since we allow zero-length segments
        if !segments.is_sorted_by_key(|segment| segment.offset) {
            vortex_bail!("Segment offsets are not ordered");
        }

        let footer = Self {
            root_layout,
            segments,
            statistics,
            array_read_ctx,
            approx_byte_size: None,
        };
        let approx_byte_size = footer.approx_retained_size(serialized_bytes, layout_tree_bytes);
        Ok(footer.with_approx_byte_size(approx_byte_size))
    }

    /// Approximate heap bytes this footer retains once its layout tree is fully materialised.
    ///
    /// `serialized_bytes` is the size of the flatbuffer copies the footer holds on to, and
    /// `layout_tree_bytes` the estimated cost of the materialised layout tree - see
    /// [`approx_layout_tree_size`].
    ///
    /// The layout tree is deliberately charged up front. A [`Footer`] is handed to caches that
    /// record its size once, at admission, and subtract that same accessor's value at eviction; a
    /// size that grew in between would drive their accounting negative. See
    /// [`Self::approx_byte_size`].
    fn approx_retained_size(&self, serialized_bytes: usize, layout_tree_bytes: usize) -> usize {
        serialized_bytes
            + layout_tree_bytes
            + self.dtype().approx_heap_size()
            + self
                .statistics
                .as_ref()
                .map_or(0, FileStatistics::approx_heap_size)
            + self.segments.len() * size_of::<SegmentSpec>()
            + size_of_val(self.array_read_ctx.ids())
    }

    /// Returns the root [`LayoutRef`] of the file.
    pub fn layout(&self) -> &LayoutRef {
        &self.root_layout
    }

    /// Returns the segment map of the file.
    pub fn segment_map(&self) -> &Arc<[SegmentSpec]> {
        &self.segments
    }

    /// Returns the statistics of the file.
    pub fn statistics(&self) -> Option<&FileStatistics> {
        self.statistics.as_ref()
    }

    /// Computes the compressed size in bytes of every field in the file, keyed by field path.
    ///
    /// Sizes are derived by attributing each segment in the [segment map][Self::segment_map] to a
    /// field in the [layout tree][Self::layout]; see [`CompressedFieldSizes`] for the exact
    /// attribution semantics. No IO is performed.
    pub fn compressed_field_sizes(&self) -> VortexResult<CompressedFieldSizes> {
        CompressedFieldSizes::try_new(&self.root_layout, &self.segments)
    }

    /// Returns the [`DType`] of the file.
    pub fn dtype(&self) -> &DType {
        self.root_layout.dtype()
    }

    /// Approximate heap bytes this footer retains, for cache admission and memory accounting.
    ///
    /// This covers everything the footer keeps alive: the flatbuffer copies, the parsed dtype,
    /// the file statistics, the segment map, the encoding read contexts, and the layout tree.
    ///
    /// The layout tree is built lazily, as scans walk it, *into the footer a cache is already
    /// holding*. The value reported here therefore budgets for the fully materialised tree from
    /// the start, and is stable for the lifetime of the footer. Caches rely on that: they record
    /// an entry's size at admission and subtract this accessor's value again at eviction, so a
    /// size that grew in between would drive their accounting negative.
    pub fn approx_byte_size(&self) -> Option<usize> {
        self.approx_byte_size
    }

    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.root_layout.row_count()
    }

    /// Returns a serializer for this footer.
    pub fn into_serializer(self) -> FooterSerializer {
        FooterSerializer::new(self)
    }

    /// Create a deserializer for a Vortex file footer.
    pub fn deserializer(eof_buffer: ByteBuffer, session: VortexSession) -> FooterDeserializer {
        FooterDeserializer::new(eof_buffer, session)
    }
}
