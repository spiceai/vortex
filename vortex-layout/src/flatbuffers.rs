// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::sync::LazyLock;

use flatbuffers::FlatBufferBuilder;
use flatbuffers::VerifierOptions;
use flatbuffers::WIPOffset;
use flatbuffers::root_with_opts;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::layout;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::Layout;
use crate::LayoutBuildContext;
use crate::LayoutContext;
use crate::LayoutRef;
use crate::children::ViewedLayoutChildren;
use crate::layouts::foreign::new_foreign_layout;
use crate::segments::SegmentId;
use crate::session::LayoutSessionExt;

static LAYOUT_VERIFIER: LazyLock<VerifierOptions> = LazyLock::new(|| {
    VerifierOptions {
        // Overridden
        max_tables: env::var("VORTEX_MAX_LAYOUT_TABLES")
            .ok()
            .and_then(|lmt| lmt.parse::<usize>().ok())
            .unwrap_or(1000000),
        max_depth: env::var("VORTEX_MAX_LAYOUT_DEPTH")
            .ok()
            .and_then(|lmt| lmt.parse::<usize>().ok())
            .unwrap_or(64),
        // Defaults from flatbuffers
        max_apparent_size: 1 << 31,
        ignore_missing_null_terminator: false,
    }
});

/// Approximate heap bytes retained by one materialised layout tree node.
///
/// Materialising a node allocates an `Arc<dyn Layout>` for the node itself, an
/// `Arc<dyn LayoutChildren>` view over its serialized children, the `OnceCell` array that
/// memoises those children, a clone of the node's [`DType`] (a freshly parsed
/// [`StructFields`](vortex_array::dtype::StructFields) for a struct node) and its segment id
/// list. The exact total is encoding- and schema-dependent, so this is a single conservative
/// constant, chosen above the largest per-node cost measured across flat, wide, chunked and
/// nested schemas. `footer_size_accounting` in `vortex-file/tests` is the regression test that
/// keeps it honest.
pub const APPROX_LAYOUT_NODE_BYTES: usize = 480;

/// Approximate heap bytes a fully materialised layout tree will retain.
///
/// Computed from the serialized layout alone: the tree is walked in its flatbuffer form, so
/// nothing is materialised and no [`LayoutRef`] is allocated. Callers use this to budget for a
/// [`LayoutRef`] whose children are built lazily, and whose retained size therefore grows after
/// it has been handed to a cache.
pub fn approx_layout_tree_size(flatbuffer: impl AsRef<[u8]>) -> VortexResult<usize> {
    let fb_layout = root_with_opts::<layout::Layout>(&LAYOUT_VERIFIER, flatbuffer.as_ref())?;
    Ok(approx_tree_size_of(fb_layout))
}

/// [`approx_layout_tree_size`] over an already validated flatbuffer root.
///
/// Validation dominates the cost of the walk, so callers that have already validated the buffer
/// go through here rather than paying for it twice.
fn approx_tree_size_of(fb_layout: layout::Layout<'_>) -> usize {
    fn count(layout: layout::Layout<'_>) -> usize {
        1 + layout
            .children()
            .unwrap_or_default()
            .iter()
            .map(count)
            .sum::<usize>()
    }

    count(fb_layout) * APPROX_LAYOUT_NODE_BYTES
}

/// Parse a [`LayoutRef`] from a layout flatbuffer.
pub fn layout_from_flatbuffer(
    flatbuffer: FlatBuffer,
    dtype: &DType,
    layout_ctx: &ReadContext,
    ctx: &ReadContext,
    session: &VortexSession,
) -> VortexResult<LayoutRef> {
    Ok(layout_from_flatbuffer_with_options(flatbuffer, dtype, layout_ctx, ctx, session, false)?.0)
}

/// Parse a [`LayoutRef`] from a layout flatbuffer with unknown-encoding behavior control.
///
/// Also returns [`approx_layout_tree_size`] for the parsed layout. The two are computed together
/// because both need a validated flatbuffer, and validating it twice costs more than the walk
/// itself.
pub fn layout_from_flatbuffer_with_options(
    flatbuffer: FlatBuffer,
    dtype: &DType,
    layout_ctx: &ReadContext,
    ctx: &ReadContext,
    session: &VortexSession,
    allow_unknown: bool,
) -> VortexResult<(LayoutRef, usize)> {
    let layout_session = session.layouts();
    let layouts = layout_session.registry();
    let fb_layout = root_with_opts::<layout::Layout>(&LAYOUT_VERIFIER, &flatbuffer)?;
    let approx_tree_size = approx_tree_size_of(fb_layout);
    let encoding_id = layout_ctx
        .resolve(fb_layout.encoding())
        .ok_or_else(|| vortex_err!("Invalid encoding ID: {}", fb_layout.encoding()))?;
    let encoding = layouts.find(&encoding_id);

    if encoding.is_none() && allow_unknown {
        return Ok((
            foreign_layout_from_fb(fb_layout, dtype, layout_ctx)?,
            approx_tree_size,
        ));
    }
    let encoding =
        encoding.ok_or_else(|| vortex_err!("Invalid encoding ID: {}", fb_layout.encoding()))?;

    // SAFETY: we validate the flatbuffer above in the `root` call, and extract a loc.
    let viewed_children = unsafe {
        ViewedLayoutChildren::new_unchecked(
            flatbuffer.clone(),
            fb_layout._tab.loc(),
            ctx.clone(),
            layout_ctx.clone(),
            layouts.clone(),
            allow_unknown,
            session.clone(),
        )
    };

    let build_ctx = LayoutBuildContext {
        session,
        array_read_ctx: ctx,
    };
    let layout = encoding.build(
        dtype,
        fb_layout.row_count(),
        fb_layout
            .metadata()
            .map(|m| m.bytes())
            .unwrap_or_else(|| &[]),
        fb_layout
            .segments()
            .unwrap_or_default()
            .iter()
            .map(SegmentId::from)
            .collect(),
        &viewed_children,
        &build_ctx,
    )?;

    Ok((layout, approx_tree_size))
}

fn foreign_layout_from_fb(
    fb_layout: layout::Layout<'_>,
    dtype: &DType,
    layout_ctx: &ReadContext,
) -> VortexResult<LayoutRef> {
    let encoding_id = layout_ctx
        .resolve(fb_layout.encoding())
        .ok_or_else(|| vortex_err!("Invalid encoding ID: {}", fb_layout.encoding()))?;

    let children = fb_layout
        .children()
        .unwrap_or_default()
        .iter()
        .map(|child| foreign_layout_from_fb(child, dtype, layout_ctx))
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(new_foreign_layout(
        encoding_id,
        dtype.clone(),
        fb_layout.row_count(),
        fb_layout
            .metadata()
            .map(|m| m.bytes().to_vec())
            .unwrap_or_default(),
        fb_layout
            .segments()
            .unwrap_or_default()
            .iter()
            .map(SegmentId::from)
            .collect(),
        children,
    ))
}

impl dyn Layout + '_ {
    /// Serialize the layout into a [`FlatBufferBuilder`].
    pub fn flatbuffer_writer<'a>(
        &'a self,
        ctx: &'a LayoutContext,
    ) -> impl WriteFlatBuffer<Target<'a> = layout::Layout<'a>> + FlatBufferRoot + 'a {
        LayoutFlatBufferWriter { layout: self, ctx }
    }
}

/// An adapter struct for writing a layout to a FlatBuffer.
struct LayoutFlatBufferWriter<'a> {
    layout: &'a dyn Layout,
    ctx: &'a LayoutContext,
}

impl FlatBufferRoot for LayoutFlatBufferWriter<'_> {}

impl WriteFlatBuffer for LayoutFlatBufferWriter<'_> {
    type Target<'fb> = layout::Layout<'fb>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        // First we recurse into the children and write them out
        let child_layouts = self.layout.children()?;
        let children = child_layouts
            .iter()
            .map(|layout| {
                LayoutFlatBufferWriter {
                    layout: layout.as_ref(),
                    ctx: self.ctx,
                }
                .write_flatbuffer(fbb)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let children = (!children.is_empty()).then(|| fbb.create_vector(&children));

        // Next we write out the metadata if it's non-empty.
        let metadata = self.layout.metadata();
        let metadata = (!metadata.is_empty()).then(|| fbb.create_vector(&metadata));

        let segments = self
            .layout
            .segment_ids()
            .into_iter()
            .map(|s| *s)
            .collect::<Vec<_>>();
        let segments = (!segments.is_empty()).then(|| fbb.create_vector(&segments));

        // Dictionary-encode the layout ID
        let encoding = self.ctx.intern(&self.layout.encoding_id()).ok_or_else(|| {
            vortex_err!(
                "Failed to intern layout encoding ID: {}",
                self.layout.encoding_id()
            )
        })?;

        Ok(layout::Layout::create(
            fbb,
            &layout::LayoutArgs {
                encoding,
                row_count: self.layout.row_count(),
                metadata,
                children,
                segments,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use flatbuffers::FlatBufferBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_flatbuffers::layout as fbl;
    use vortex_session::registry::ReadContext;

    use super::APPROX_LAYOUT_NODE_BYTES;
    use super::layout_from_flatbuffer_with_options;
    use crate::LayoutEncodingId;
    use crate::session::LayoutSession;

    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    #[test]
    fn unknown_layout_encoding_allow_unknown() {
        let mut fbb = FlatBufferBuilder::new();

        let child_metadata = fbb.create_vector(&[9u8]);
        let child = fbl::Layout::create(
            &mut fbb,
            &fbl::LayoutArgs {
                encoding: 1,
                row_count: 3,
                metadata: Some(child_metadata),
                children: None,
                segments: None,
            },
        );

        let children = fbb.create_vector(&[child]);
        let metadata = fbb.create_vector(&[1u8, 2, 3]);
        let segments = fbb.create_vector(&[7u32]);
        let root = fbl::Layout::create(
            &mut fbb,
            &fbl::LayoutArgs {
                encoding: 0,
                row_count: 10,
                metadata: Some(metadata),
                children: Some(children),
                segments: Some(segments),
            },
        );
        fbb.finish_minimal(root);
        let (buf, start) = fbb.collapse();
        let layout_buffer = vortex_flatbuffers::FlatBuffer::align_from(
            vortex_buffer::ByteBuffer::from(buf).slice(start..),
        );

        let layout_ctx = ReadContext::new([
            LayoutEncodingId::new("vortex.test.foreign_layout"),
            LayoutEncodingId::new("vortex.test.foreign_child_layout"),
        ]);
        let array_ctx = ReadContext::new([]);
        let session = vortex_array::array_session().with::<LayoutSession>();

        let (layout, approx_tree_size) = layout_from_flatbuffer_with_options(
            layout_buffer,
            &DType::Variant(Nullability::Nullable),
            &layout_ctx,
            &array_ctx,
            &session,
            true,
        )
        .unwrap();

        assert_eq!(
            approx_tree_size,
            2 * APPROX_LAYOUT_NODE_BYTES,
            "the foreign layout and its single child"
        );
        assert_eq!(layout.encoding_id().as_ref(), "vortex.test.foreign_layout");
        assert_eq!(layout.row_count(), 10);
        assert_eq!(layout.metadata(), vec![1, 2, 3]);
        assert_eq!(layout.segment_ids().len(), 1);
        assert_eq!(*layout.segment_ids()[0], 7);
        assert_eq!(layout.nchildren(), 1);

        let child = layout.child(0).unwrap();
        assert_eq!(
            child.encoding_id().as_ref(),
            "vortex.test.foreign_child_layout"
        );
        assert_eq!(child.metadata(), vec![9]);
    }
}
