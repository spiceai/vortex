// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::MinMaxPartial;
use super::MinMaxResult;
use super::min_max;
use crate::ExecutionCtx;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

pub(super) fn accumulate_extension(
    partial: &mut MinMaxPartial,
    array: &ExtensionArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let non_nullable_ext_dtype = array.ext_dtype().with_nullability(Nullability::NonNullable);
    // Build the extension scalars fallibly: an extension type whose metadata the storage value
    // cannot satisfy (an unresolvable timezone, say) must fail the aggregation rather than abort
    // the process, since this runs on the write path when building zone maps.
    let local = min_max(array.storage_array(), ctx)?
        .map(|MinMaxResult { min, max }| -> VortexResult<MinMaxResult> {
            Ok(MinMaxResult {
                min: Scalar::try_extension_ref(non_nullable_ext_dtype.clone(), min)?,
                max: Scalar::try_extension_ref(non_nullable_ext_dtype, max)?,
            })
        })
        .transpose()?;
    partial.merge(local);
    Ok(())
}
