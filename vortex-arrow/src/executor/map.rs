// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericListArray;
use arrow_array::MapArray as ArrowMapArray;
use arrow_array::StructArray as ArrowStructArray;
use arrow_schema::FieldRef;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::executor::list::to_arrow_list;

/// Convert a Vortex `List<Struct<key, value>>` array into an Arrow [`MapArray`](ArrowMapArray).
///
/// Arrow's `Map` has no Vortex `DType` of its own; it is aliased to `List<Struct<key, value>>`
/// on the way in (see [`crate::dtype`]), so restoring the `Map` on the way out is a matter of
/// re-attaching the entries field and the `ordered` flag to the same offsets and entries.
pub(super) fn to_arrow_map(
    array: ArrayRef,
    entries_field: &FieldRef,
    ordered: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // Map offsets are always i32.
    let list_array = to_arrow_list::<i32>(array, entries_field, ctx)?;

    let Some(list_array) = list_array.as_any().downcast_ref::<GenericListArray<i32>>() else {
        vortex_bail!("to_arrow_list returned a non-ListArray when building a MapArray");
    };
    let (_list_field, offsets, entries, nulls) = list_array.clone().into_parts();

    let Some(entries_struct) = entries.as_any().downcast_ref::<ArrowStructArray>() else {
        vortex_bail!(
            "Map entries must be a StructArray, got {}",
            entries.data_type()
        );
    };

    Ok(Arc::new(ArrowMapArray::try_new(
        Arc::clone(entries_field),
        offsets,
        entries_struct.clone(),
        nulls,
        ordered,
    )?))
}
