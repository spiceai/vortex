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

/// Maximum number of user-defined metadata segments. Keeps postscript bookkeeping small so the
/// footer and required segments still fit the initial tail read.
pub(crate) const MAX_METADATA_SEGMENTS: usize = 16;

/// Maximum length, in UTF-8 bytes, of a user-defined metadata key (keys live in the postscript).
///
/// 64 bytes covers reverse-DNS query-engine keys, not just short Iceberg-style keys: e.g.
/// `org.apache.spark.sql.parquet.row.metadata` (41 bytes), which Spark writes into every Parquet
/// file. With [`MAX_METADATA_SEGMENTS`] keys this bounds the postscript key budget at 1 KiB.
pub(crate) const MAX_METADATA_KEY_BYTES: usize = 64;

/// User-defined metadata segment locators stored as `(key, locator)` pairs.
pub(crate) type MetadataSegments = Arc<[(String, SegmentSpec)]>;

/// Captures the layout information of a Vortex file.
#[derive(Debug, Clone)]
pub struct Footer {
    root_layout: LayoutRef,
    segments: Arc<[SegmentSpec]>,
    statistics: Option<FileStatistics>,
    metadata: Arc<[(String, SegmentSpec)]>,
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
            metadata: Arc::from([]),
            array_read_ctx,
            approx_byte_size: None,
        }
    }

    /// Record [`Self::approx_byte_size`] from the sizes only the caller knows: the serialized
    /// footer segments, and the layout tree those segments describe.
    ///
    /// The layout tree is charged up front even though it is built lazily. A [`Footer`] is handed
    /// to caches that record its size once, at admission, and subtract that same accessor's value
    /// at eviction; a size that grew in between would drive their accounting negative.
    pub(crate) fn set_approx_retained_size(
        &mut self,
        serialized_bytes: usize,
        layout_tree_bytes: usize,
    ) {
        self.approx_byte_size = Some(
            serialized_bytes
                + layout_tree_bytes
                + self.dtype().approx_heap_size()
                + self
                    .statistics
                    .as_ref()
                    .map_or(0, FileStatistics::approx_heap_size)
                + size_of_val(&*self.segments)
                + size_of_val(&*self.metadata)
                + self
                    .metadata
                    .iter()
                    .map(|(key, _segment)| key.len())
                    .sum::<usize>()
                + size_of_val(self.array_read_ctx.ids()),
        );
    }

    /// Read the [`Footer`] from a flatbuffer.
    pub(crate) fn from_flatbuffer(
        footer_bytes: &[u8],
        layout_bytes: FlatBuffer,
        dtype_bytes_len: usize,
        dtype: DType,
        statistics: Option<FileStatistics>,
        metadata: Arc<[(String, SegmentSpec)]>,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        // The footer keeps a private copy of the layout flatbuffer segment, and of the dtype
        // segment unless the file was written with `exclude_dtype`. `BufferMut` over-aligns each
        // copy to `Alignment::DEFAULT_ALIGNMENT`, so each carries up to that much slack. The
        // footer segment is only borrowed for the duration of the parse, so it is not retained.
        let retained_segments = 1 + usize::from(dtype_bytes_len > 0);
        let serialized_bytes = layout_bytes.len()
            + dtype_bytes_len
            + retained_segments * Alignment::DEFAULT_ALIGNMENT.as_usize();
        let fb_footer = root::<fb::Footer>(footer_bytes)?;

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

        let mut footer = Self::new(root_layout, segments, statistics, array_read_ctx)
            .with_metadata_segments(metadata);
        footer.set_approx_retained_size(serialized_bytes, layout_tree_bytes);
        Ok(footer)
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

    /// Returns the user-defined metadata segment locators stored in the postscript.
    pub fn metadata_segments(&self) -> impl Iterator<Item = (&str, &SegmentSpec)> {
        self.metadata
            .iter()
            .map(|(key, segment)| (key.as_str(), segment))
    }

    /// Returns the user-defined metadata segment locator for the given key.
    pub fn metadata_segment(&self, key: &str) -> Option<&SegmentSpec> {
        self.metadata
            .iter()
            .find_map(|(candidate, segment)| (candidate == key).then_some(segment))
    }

    pub(crate) fn with_metadata_segments(mut self, metadata: Arc<[(String, SegmentSpec)]>) -> Self {
        self.metadata = metadata;
        self
    }

    pub(crate) fn segment_specs_with_metadata(&self) -> Arc<[SegmentSpec]> {
        self.segments
            .iter()
            .copied()
            .chain(self.metadata.iter().map(|(_, segment)| *segment))
            .collect()
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

    /// Validate that every segment declared in the footer lies within a file of `file_size` bytes.
    pub(crate) fn validate_file_size(&self, file_size: u64) -> VortexResult<()> {
        validate_segments_within_file(&self.segments, file_size)
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

/// Validate that every segment declared in the footer lies within a file of `file_size` bytes.
///
/// A corrupt or malicious file can declare a segment whose offset or length extends past the end
/// of the file. Rejecting such files up front ensures that later slicing of the backing buffer
/// returns a [`VortexError`](vortex_error::VortexError) rather than panicking (see issue #8819).
fn validate_segments_within_file(segments: &[SegmentSpec], file_size: u64) -> VortexResult<()> {
    for segment in segments {
        let within_file = segment
            .offset
            .checked_add(segment.length as u64)
            .is_some_and(|end| end <= file_size);
        if !within_file {
            vortex_bail!(
                "Segment at offset {} with length {} extends past the end of the \
                 {file_size}-byte file",
                segment.offset,
                segment.length,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Alignment;

    use super::*;

    fn segment(offset: u64, length: u32) -> SegmentSpec {
        SegmentSpec {
            offset,
            length,
            alignment: Alignment::none(),
        }
    }

    #[test]
    fn accepts_segments_within_file() -> VortexResult<()> {
        validate_segments_within_file(&[segment(0, 100), segment(100, 50)], 150)?;
        Ok(())
    }

    #[test]
    fn rejects_segment_extending_past_end_of_file() {
        let err =
            validate_segments_within_file(&[segment(0, 100), segment(100, 51)], 150).unwrap_err();
        assert!(err.to_string().contains("past the end"), "{err}");
    }

    #[test]
    fn rejects_segment_offset_length_overflow() {
        let err = validate_segments_within_file(&[segment(u64::MAX, 1)], u64::MAX).unwrap_err();
        assert!(err.to_string().contains("past the end"), "{err}");
    }
}
