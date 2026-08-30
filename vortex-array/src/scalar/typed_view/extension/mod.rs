// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ExtScalar`] typed view implementation.

use std::fmt;
use std::hash::Hash;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::dtype::DType;
use crate::dtype::extension::ExtDTypeRef;
use crate::extension::datetime::AnyTemporal;
use crate::extension::datetime::DateToTimestamp;
use crate::extension::datetime::TemporalMetadata;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A scalar value representing an extension type.
///
/// Extension types allow wrapping a storage type with custom semantics.
#[derive(Debug, Clone, Copy)]
pub struct ExtScalar<'a> {
    /// A reference to the `DType` of the extension type. This **must** be the [`DType::Extension`]
    /// variant.
    dtype: &'a DType,

    /// The extension data type reference.
    ///
    /// We store this here as a convenience so that we do not need to unwrap the dtype every time.
    ext_dtype: &'a ExtDTypeRef,

    /// The underlying scalar value, or [`None`] if null.
    value: Option<&'a ScalarValue>,
}

impl fmt::Display for ExtScalar<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(value) = self.value else {
            return write!(f, "null");
        };

        self.ext_dtype.fmt_storage_value(f, value)
    }
}

impl<'a> ExtScalar<'a> {
    /// Creates a new extension scalar from a data type and scalar value.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the dtype is an extension type and that the scalar value has
    /// been verified to be valid for the extension type.
    pub(crate) fn new_unchecked(dtype: &'a DType, value: Option<&'a ScalarValue>) -> Self {
        let DType::Extension(ext_dtype) = dtype else {
            vortex_panic!("Expected extension scalar, found {}", dtype)
        };

        Self {
            dtype,
            ext_dtype,
            value,
        }
    }

    /// Return the [`DType`] of the extension scalar.
    pub fn dtype(&self) -> &DType {
        self.dtype
    }

    /// Returns the extension data type.
    pub fn ext_dtype(&self) -> &'a ExtDTypeRef {
        self.ext_dtype
    }

    /// Returns the storage scalar of the extension scalar.
    pub fn to_storage_scalar(&self) -> Scalar {
        Scalar::try_new(self.ext_dtype.storage_dtype().clone(), self.value.cloned())
            .vortex_expect("ExtScalar is invalid")
    }

    /// Casts this scalar to the given `dtype`.
    pub(crate) fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if self.value.is_none() && !target_dtype.is_nullable() {
            vortex_bail!(
                "cannot cast extension dtype with id {} and storage type {} to {}",
                self.ext_dtype.id(),
                self.ext_dtype.storage_dtype(),
                target_dtype
            );
        }

        if self
            .ext_dtype
            .storage_dtype()
            .eq_ignore_nullability(target_dtype)
        {
            // Casting from an extension type to the underlying storage type is OK.
            return Scalar::try_new(target_dtype.clone(), self.value.cloned());
        }

        if let DType::Extension(ext_dtype) = target_dtype {
            // The same extension dtype: the value already means what the target says it does.
            if self.ext_dtype.eq_ignore_nullability(ext_dtype) {
                return Scalar::try_new(target_dtype.clone(), self.value.cloned());
            }

            // A different one needs the value converted, which is defined only for the pairs
            // `Extension`'s `CastKernel` converts at the array level.
            if let Some(scalar) = self.cast_date_to_timestamp(ext_dtype, target_dtype)? {
                return Ok(scalar);
            }
        }

        vortex_bail!(
            "cannot cast extension dtype with id {} and storage type {} to {}",
            self.ext_dtype.id(),
            self.ext_dtype.storage_dtype(),
            target_dtype
        );
    }

    /// Convert a `vortex.date` scalar to a `vortex.timestamp` one, rescaling the value.
    ///
    /// Returns `Ok(None)` for any other pair of extension types, leaving the caller to refuse
    /// the cast. The conversion mirrors `Extension`'s `CastKernel`, which is what a scan
    /// applies to the rows of a file whose statistics were compared through here.
    fn cast_date_to_timestamp(
        &self,
        target_ext_dtype: &ExtDTypeRef,
        target_dtype: &DType,
    ) -> VortexResult<Option<Scalar>> {
        let (Some(source_temporal), Some(target_temporal)) = (
            self.ext_dtype.metadata_opt::<AnyTemporal>(),
            target_ext_dtype.metadata_opt::<AnyTemporal>(),
        ) else {
            return Ok(None);
        };

        let (TemporalMetadata::Date(source_unit), TemporalMetadata::Timestamp(target_unit, _)) =
            (source_temporal, target_temporal)
        else {
            return Ok(None);
        };

        // Null is handled by `Scalar::cast` before it reaches an extension scalar, and by the
        // nullability check above; a null that gets this far still converts to a null.
        let Some(value) = self.to_storage_scalar().as_primitive().as_opt::<i64>() else {
            return Ok(None);
        };
        let Some(value) = value else {
            return Ok(Some(Scalar::try_new(target_dtype.clone(), None)?));
        };

        let converted = DateToTimestamp::new(*source_unit, *target_unit)?.convert(value)?;

        let storage_value = Scalar::primitive(converted, target_dtype.nullability())
            .cast(target_ext_dtype.storage_dtype())?
            .into_value();
        Ok(Some(Scalar::try_new(target_dtype.clone(), storage_value)?))
    }
}

// TODO(connor): In the future we may want to allow implementors to customize this behavior.

impl PartialEq for ExtScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.ext_dtype.eq_ignore_nullability(other.ext_dtype)
            && self.to_storage_scalar() == other.to_storage_scalar()
    }
}

impl Eq for ExtScalar<'_> {}

// Ord is not implemented since it's undefined for different Extension DTypes
impl PartialOrd for ExtScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if !self.ext_dtype.eq_ignore_nullability(other.ext_dtype) {
            return None;
        }
        self.to_storage_scalar()
            .partial_cmp(&other.to_storage_scalar())
    }
}

impl Hash for ExtScalar<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ext_dtype.hash(state);
        self.to_storage_scalar().hash(state);
    }
}

#[cfg(test)]
mod tests;
