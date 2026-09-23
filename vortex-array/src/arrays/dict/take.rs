// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::SmallVec;
use vortex_error::VortexResult;

use super::Dict;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::ConstantArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::expr::stats::StatsProviderExt;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar::Scalar;
use crate::stats::StatsSet;
use crate::validity::Validity;

pub trait TakeReduce: VTable {
    /// Take elements from an array at the given indices without reading buffers.
    ///
    /// This trait is for take implementations that can operate purely on array metadata and
    /// structure without needing to read or execute on the underlying buffers. Implementations
    /// should return `None` if taking requires buffer access.
    ///
    /// # Preconditions
    ///
    /// The indices are guaranteed to be non-empty.
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>>;
}

pub trait TakeExecute: VTable {
    /// Take elements from an array at the given indices, potentially reading buffers.
    ///
    /// Unlike [`TakeReduce`], this trait is for take implementations that may need to read
    /// and execute on the underlying buffers to produce the result.
    ///
    /// # Preconditions
    ///
    /// The indices are guaranteed to be non-empty.
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Short-circuits take for the inputs that need no encoding-specific work.
///
/// Returns `Some(result)` when the answer is already known, or `None` when take must proceed
/// normally.
fn short_circuit<V: VTable>(array: ArrayView<'_, V>, indices: &ArrayRef) -> Option<ArrayRef> {
    // Fast-path for empty indices.
    if indices.is_empty() {
        let result_dtype = array
            .dtype()
            .clone()
            .union_nullability(indices.dtype().nullability());
        return Some(Canonical::empty(&result_dtype).into_array());
    }

    // Fast-path for empty arrays: all indices must be null, return all-invalid result.
    if array.is_empty() {
        return Some(
            ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                .into_array(),
        );
    }

    None
}

#[derive(Default, Debug)]
pub struct TakeReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for TakeReduceAdaptor<V>
where
    V: TakeReduce,
{
    type Parent = Dict;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ArrayView<'_, Dict>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle the values child (index 1), not the codes child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        if let Some(result) = short_circuit::<V>(array, parent.codes()) {
            return Ok(Some(result));
        }
        let result = <V as TakeReduce>::take(array, parent.codes())?;
        if let Some(taken) = &result {
            propagate_take_stats(array.array(), taken, parent.codes())?;
        }
        Ok(result)
    }
}

#[derive(Default, Debug)]
pub struct TakeExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for TakeExecuteAdaptor<V>
where
    V: TakeExecute,
{
    type Parent = Dict;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle the values child (index 1), not the codes child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        if let Some(result) = short_circuit::<V>(array, parent.codes()) {
            return Ok(Some(result));
        }
        let result = <V as TakeExecute>::take(array, parent.codes(), ctx)?;
        if let Some(taken) = &result {
            propagate_take_stats(array.array(), taken, parent.codes())?;
        }
        Ok(result)
    }
}

pub(crate) fn propagate_take_stats(
    source: &ArrayRef,
    target: &ArrayRef,
    indices: &ArrayRef,
) -> VortexResult<()> {
    let indices_all_valid = matches!(
        indices.validity()?,
        Validity::NonNullable | Validity::AllValid
    );
    target.statistics().with_mut_typed_stats_set(|mut st| {
        if indices_all_valid {
            let is_constant = source.statistics().get_as::<bool>(Stat::IsConstant);
            if matches!(is_constant, Precision::Exact(true)) {
                // Any combination of elements from a constant array is still const
                st.set(Stat::IsConstant, Precision::exact(true));
            }
        }
        let inexact_min_max = [
            Stat::Min,
            Stat::Max,
            Stat::UncompressedSizeInBytes,
            Stat::IsConstant,
        ]
        .into_iter()
        .filter_map(|stat| match source.statistics().get(stat).into_inexact() {
            Precision::Exact(scalar) | Precision::Inexact(scalar) => {
                scalar.into_value().map(|sv| (stat, Precision::Inexact(sv)))
            }
            Precision::Absent => None,
        })
        .collect::<SmallVec<_>>();
        st.combine_sets(
            &(unsafe { StatsSet::new_unchecked(inexact_min_max) }).as_typed_ref(source.dtype()),
        )
    })
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use vortex_error::VortexExpect;

    use super::*;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    /// `UncompressedSizeInBytes` has to survive a dict take, inexactly, the same way `Min`/`Max`
    /// already do: the optimizer sizes joins from it, and a value's absence after a take is read
    /// as "unknown" rather than "zero", so losing it silently degrades plans over dictionary-
    /// encoded columns rather than failing loudly.
    #[test]
    fn uncompressed_size_in_bytes_propagates_through_take_as_inexact() -> VortexResult<()> {
        let source = buffer![1i32, 2, 3, 4].into_array();
        source.statistics().set(
            Stat::UncompressedSizeInBytes,
            Precision::exact(
                Scalar::primitive(16u64, Nullability::NonNullable)
                    .into_value()
                    .vortex_expect("non-null size"),
            ),
        );

        let target = buffer![1i32, 3].into_array();
        let indices = buffer![0u64, 2].into_array();
        propagate_take_stats(&source, &target, &indices)?;

        assert_eq!(
            target
                .statistics()
                .get_as::<u64>(Stat::UncompressedSizeInBytes),
            Precision::Inexact(16),
            "the source's size should carry forward, downgraded to inexact"
        );
        Ok(())
    }

    /// `IsConstant` is handled twice: exactly when every index is valid (the block above the
    /// one this row extends), and inexactly here otherwise. A null index breaks the "every
    /// output value equals the source's constant" guarantee, so the exact path must not fire,
    /// but the source's own `IsConstant` reading (whatever it is) is still a reasonable inexact
    /// carry-forward rather than dropping the stat entirely.
    #[test]
    fn is_constant_propagates_inexactly_through_take_with_a_null_index() -> VortexResult<()> {
        let source = ConstantArray::new(Scalar::primitive(7i32, Nullability::NonNullable), 4)
            .into_array();
        source.statistics().set(
            Stat::IsConstant,
            Precision::exact(
                Scalar::from(true)
                    .into_value()
                    .vortex_expect("non-null bool"),
            ),
        );

        let target = ConstantArray::new(Scalar::primitive(7i32, Nullability::NonNullable), 2)
            .into_array();
        // A nullable index array with one null: `indices_all_valid` is false, so the exact
        // IsConstant path above must not run.
        let indices = crate::arrays::PrimitiveArray::from_option_iter([Some(0u64), None])
            .into_array();
        propagate_take_stats(&source, &target, &indices)?;

        assert_eq!(
            target.statistics().get_as::<bool>(Stat::IsConstant),
            Precision::Inexact(true),
            "IsConstant should still carry forward inexactly when it can't be proven exact"
        );
        Ok(())
    }
}
