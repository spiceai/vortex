// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use datafusion_common::Result as DFResult;
use vortex::error::VortexResult;

pub(crate) mod exprs;
mod scalars;
pub(crate) mod schema;
pub(crate) mod stats;

/// First-party trait for implementing conversion from DataFusion types to Vortex types.
pub(crate) trait FromDataFusion<D: ?Sized>: Sized {
    fn from_df(df: &D) -> Self;
}

/// First-party trait for fallible conversion from DataFusion to Vortex types.
pub(crate) trait TryFromDataFusion<D: ?Sized>: Sized {
    fn try_from_df(df: &D) -> DFResult<Self>;
}

/// First-party trait for implementing conversion from Vortex to DataFusion types.
pub(crate) trait TryToDataFusion<D> {
    fn try_to_df(&self) -> VortexResult<D>;
}
