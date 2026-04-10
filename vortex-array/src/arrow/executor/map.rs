// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::MapArray as ArrowMapArray;
use arrow_array::StructArray as ArrowStructArray;
use arrow_schema::FieldRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrow::executor::list::to_arrow_list;

/// Convert a Vortex List<Struct<key, value>> array into an Arrow MapArray.
pub(super) fn to_arrow_map(
    array: ArrayRef,
    entries_field: &FieldRef,
    ordered: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // First, convert to Arrow ListArray<i32> since Map uses i32 offsets.
    let list_array = to_arrow_list::<i32>(array, entries_field, ctx)?;

    // Downcast to GenericListArray<i32> to extract its components.
    let Some(list_array) = list_array
        .as_any()
        .downcast_ref::<arrow_array::GenericListArray<i32>>()
    else {
        vortex_bail!("to_arrow_list returned a non-ListArray when building a MapArray");
    };

    // Extract components from the ListArray.
    let (_list_field, offsets, entries, nulls) = list_array.clone().into_parts();

    // The entries should be a StructArray. Downcast it.
    let Some(entries_struct) = entries.as_any().downcast_ref::<ArrowStructArray>() else {
        vortex_bail!("Map entries must be a StructArray");
    };
    let entries_struct = entries_struct.clone();

    // Build the MapArray from the components.
    let map_array = ArrowMapArray::try_new(
        Arc::clone(entries_field),
        offsets,
        entries_struct,
        nulls,
        ordered,
    )?;

    Ok(Arc::new(map_array))
}
