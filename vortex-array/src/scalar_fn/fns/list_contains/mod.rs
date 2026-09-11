// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::hash::Hash;
use std::ops::BitOr;

use arrow_buffer::bit_iterator::BitIndexIterator;
pub use kernel::*;
use num_traits::AsPrimitive;
use num_traits::Zero;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_set::HashSet;
use vortex_utils::iter::ReduceBalancedIterExt;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinViewArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::primitive::NativeValue;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrays::varbinview::ViewsSide;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::ListScalar;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

#[derive(Clone)]
pub struct ListContains;

impl ScalarFnVTable for ListContains {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.contains");
        *ID
    }

    fn serialize(&self, _instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("list"),
            1 => ChildName::from("needle"),
            _ => unreachable!(
                "Invalid child index {} for ListContains expression",
                child_idx
            ),
        }
    }
    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let list_dtype = &arg_dtypes[0];
        let needle_dtype = &arg_dtypes[1];

        let nullability = match list_dtype {
            DType::List(_, list_nullability) => list_nullability,
            _ => {
                vortex_bail!(
                    "First argument to ListContains must be a List, got {:?}",
                    list_dtype
                );
            }
        }
        .bitor(needle_dtype.nullability());

        Ok(DType::Bool(nullability))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let list_array = args.get(0)?;
        let value_array = args.get(1)?;

        // The needle is tested first: for an `IN` list it is an array, so this
        // fails on a downcast, whereas `list_array.as_constant()` deep-clones
        // every element of the list before the chain can reject it.
        if let Some(value_scalar) = value_array.as_constant()
            && let Some(list_constant) = list_array.as_opt::<Constant>()
        {
            let result = compute_contains_scalar(list_constant.scalar(), &value_scalar)?;
            return Ok(ConstantArray::new(result, args.row_count()).into_array());
        }

        compute_list_contains(&list_array, &value_array, ctx)
    }

    // Nullability matters for contains([], x) where x is false.
    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

fn compute_contains_scalar(list: &Scalar, needle: &Scalar) -> VortexResult<Scalar> {
    let nullability = list.dtype().nullability() | needle.dtype().nullability();

    // Handle null list or null needle
    if list.is_null() || needle.is_null() {
        return Ok(Scalar::null(DType::Bool(nullability)));
    }

    let list_scalar = list.as_list();
    let elements = list_scalar
        .elements()
        .ok_or_else(|| vortex_err!("Expected non-null list"))?;

    let contains = elements.iter().any(|elem| elem == needle);
    Ok(Scalar::bool(contains, nullability))
}

fn compute_list_contains(
    array: &ArrayRef,
    value: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let DType::List(elem_dtype, _) = array.dtype() else {
        vortex_bail!("Array must be of List type");
    };
    if !elem_dtype.as_ref().eq_ignore_nullability(value.dtype()) {
        vortex_bail!(
            "Element type {} of list does not match search value {}",
            elem_dtype,
            value.dtype(),
        );
    }

    if value.all_invalid(ctx)? || array.all_invalid(ctx)? {
        return Ok(ConstantArray::new(
            Scalar::null(DType::Bool(Nullability::Nullable)),
            array.len(),
        )
        .into_array());
    }

    let nullability = array.dtype().nullability() | value.dtype().nullability();

    if let Some(value_scalar) = value.as_constant() {
        list_contains_scalar(array, &value_scalar, nullability, ctx)
    } else if let Some(list_constant) = array.as_opt::<Constant>() {
        constant_list_scalar_contains(&list_constant.scalar().as_list(), value, nullability, ctx)
    } else {
        todo!("unsupported list contains with list and element as arrays")
    }
}

/// List length past which membership is answered by probing a set instead of by
/// OR-ing one equality per element.
///
/// The equality form costs one full-length comparison per element, so it grows
/// with the list, while the probe is built once per batch and then answers each
/// row in constant time. Where they cross depends on how expensive one
/// comparison is, which is a property of the column rather than of its type.
/// Measured on an 8192-row batch: `i64` needles run the equality form at 3.46,
/// 7.29 and 11.42us for one, two and three elements against a probe at 17.25us
/// for four, crossing just above four; `Utf8` needles run it at 10.92, 26.04 and
/// 38.96us against a probe at 37.08us, crossing just below three. Four sits
/// between them, within about 13% of the equality form at its worst point and
/// ahead of it everywhere after.
const HASH_PROBE_MIN_ELEMENTS: usize = 4;

/// Slots per element to size the probe set with.
///
/// Sizing it at exactly the element count leaves the table at its maximum load
/// factor, where the resulting probe chains cost 16-57% more than this across
/// four to eight thousand elements. Four slots an element buys a further 7-9%
/// over two for the mid-range sizes, for at most a few tens of kilobytes.
const PROBE_SET_HEADROOM: usize = 4;

/// There is a constant list scalar (haystack) being compared to an array of needles.
fn constant_list_scalar_contains(
    list_scalar: &ListScalar<'_>,
    values: &ArrayRef,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Borrowed rather than materialized as `Vec<Scalar>`: this runs once per
    // batch, and cloning every element of the list back out of the scalar costs
    // more than the set built from them.
    let element_values = list_scalar.element_values().vortex_expect("non null");

    if element_values.len() >= HASH_PROBE_MIN_ELEMENTS
        && let Some(probed) = hash_probe_contains(element_values, values, nullability, ctx)?
    {
        return Ok(probed);
    }

    let elements = list_scalar.elements().vortex_expect("non null");
    let len = values.len();
    let false_scalar = Scalar::bool(false, nullability);

    let result = elements
        .iter()
        .map(|element| {
            Binary
                .try_new_array(
                    len,
                    Operator::Eq,
                    [
                        ConstantArray::new(element.clone(), len).into_array(),
                        values.clone(),
                    ],
                )?
                .fill_null(false_scalar.clone())
        })
        .collect::<VortexResult<Vec<_>>>()?
        .into_iter()
        .try_reduce_balanced(|acc, res| acc.binary(res, Operator::Or))?;

    Ok(result.unwrap_or_else(|| ConstantArray::new(false_scalar, len).into_array()))
}

/// Answers membership by building a set from the list once and probing it in a
/// single pass over the needles.
///
/// Covers primitive, `Utf8` and `Binary` needles. Returns `None` for anything
/// else, and for a list holding a null element, leaving the caller on the
/// OR-of-equalities form, which is the definition of the operation.
///
/// Validity follows the equality form exactly: it fills a null comparison with
/// `false`, so a null needle is `false` rather than null, and the result carries
/// no invalid slots.
fn hash_probe_contains(
    elements: &[Option<ScalarValue>],
    values: &ArrayRef,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let len = values.len();
    let ptype = match values.dtype() {
        DType::Primitive(ptype, _) => *ptype,
        // Strings and byte strings key on their bytes; see `bytes_probe_contains`.
        DType::Utf8(_) | DType::Binary(_) => {
            return bytes_probe_contains(elements, values, nullability, ctx);
        }
        _ => return Ok(None),
    };

    let needles = values.clone().execute::<PrimitiveArray>(ctx)?;
    let validity = needles.validity()?.execute_mask(len, ctx)?;

    let bits = match_each_native_ptype!(ptype, |T| {
        let Some(set) = primitive_key_set::<T>(elements) else {
            return Ok(None);
        };
        let slice = needles.as_slice::<T>();
        probe_rows(len, &validity, |idx| set.contains(&NativeValue(slice[idx])))
    });

    Ok(Some(BoolArray::new(bits, nullability.into()).into_array()))
}

/// Keys the list on its primitive values, or `None` if any element is not a
/// non-null primitive of `T`.
///
/// [`NativeValue`] is the key rather than the bare value because its equality is
/// the one the kernel answers with: `NativePType::is_eq` compares floats by
/// their bits, so `NaN` matches itself and `-0.0` does not match `0.0`, and a
/// set keyed on the value would disagree with `Operator::Eq` on both.
///
/// A null element is not a key, and the equality form maps it to `false`
/// through its own null fill, so such a list goes back to that form rather than
/// being answered with a key missing.
fn primitive_key_set<T: NativePType>(
    elements: &[Option<ScalarValue>],
) -> Option<HashSet<NativeValue<T>>>
where
    NativeValue<T>: Hash + Eq,
{
    let mut set = HashSet::with_capacity(elements.len() * PROBE_SET_HEADROOM);
    for element in elements {
        let ScalarValue::Primitive(pvalue) = element.as_ref()? else {
            return None;
        };
        // The list's element dtype was checked against the needle dtype before
        // dispatch, so this holds; `cast` would otherwise convert a value of
        // some other width and key it under a number it never equals.
        if !pvalue.is_instance_of(&T::PTYPE) {
            return None;
        }
        set.insert(NativeValue(pvalue.cast::<T>().ok()?));
    }
    Some(set)
}

/// Membership for `Utf8`/`Binary` needles, keyed on the element bytes.
///
/// The keys borrow from `elements`, which outlives the set, so building it
/// copies no string data.
fn bytes_probe_contains(
    elements: &[Option<ScalarValue>],
    values: &ArrayRef,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let mut set: HashSet<&[u8]> = HashSet::with_capacity(elements.len() * PROBE_SET_HEADROOM);
    for element in elements {
        // A null element is not a key; the whole list goes back to the
        // equality form rather than being answered with a key missing.
        let bytes = match element {
            Some(ScalarValue::Utf8(value)) => value.as_bytes(),
            Some(ScalarValue::Binary(value)) => value.as_slice(),
            _ => return Ok(None),
        };
        set.insert(bytes);
    }

    let len = values.len();

    // `VarBin` already holds what the probe needs — a byte buffer and a run of
    // offsets — but it is not canonical, so executing it would first build a
    // `VarBinView`: sixteen bytes of view per row, to reach bytes that are
    // already contiguous. For short values that view costs more to construct
    // than the whole comparison it serves, and FSST's codes arrive in exactly
    // this shape.
    let bits = if let Some(varbin) = values.as_opt::<VarBin>() {
        let validity = varbin.varbin_validity().execute_mask(len, ctx)?;
        let bytes = varbin.bytes().as_slice();
        let offsets = varbin.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(offsets.ptype(), |O| {
            // One offset per row plus a final end, so adjacent pairs are the
            // rows in order.
            let offsets = offsets.as_slice::<O>();
            probe_rows(len, &validity, |idx| {
                let (start, end): (usize, usize) = (offsets[idx].as_(), offsets[idx + 1].as_());
                set.contains(&bytes[start..end])
            })
        })
    } else {
        let needles = values.clone().execute::<VarBinViewArray>(ctx)?;
        let validity = needles.validity()?.execute_mask(len, ctx)?;
        // Resolved once: reaching a row's bytes through the array re-derives the
        // views slice and re-checks the buffer index on every row.
        let side = ViewsSide::new(&needles);
        let views = side.views();
        probe_rows(len, &validity, |idx| {
            set.contains(side.view_bytes(&views[idx]))
        })
    };

    Ok(Some(BoolArray::new(bits, nullability.into()).into_array()))
}

/// One pass over the needles, setting the bit for each one the set holds.
///
/// An invalid needle is `false`, which is what the OR-of-equalities form's null
/// fill produces, so the result carries no invalid slots. `hit` is asked about a
/// row only when that row is valid.
fn probe_rows(len: usize, validity: &Mask, hit: impl Fn(usize) -> bool) -> BitBuffer {
    match validity {
        Mask::AllTrue(_) => (0..len).map(hit).collect(),
        Mask::AllFalse(_) => BitBuffer::new_unset(len),
        Mask::Values(valid) => {
            // Walking the valid rows a word at a time skips runs of nulls
            // wholesale, where testing validity per row pays a branch for every
            // one of them.
            let mut bits = BitBufferMut::new_unset(len);
            valid.bit_buffer().for_each_set_index(|idx| {
                if hit(idx) {
                    bits.set(idx);
                }
            });
            bits.freeze()
        }
    }
}

/// Returns a [`BoolArray`] where each bit represents if a list contains the scalar.
fn list_contains_scalar(
    array: &ArrayRef,
    value: &Scalar,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // If the list array is constant, we perform a single comparison.
    if array.len() > 1 && array.is::<Constant>() {
        let contains = list_contains_scalar(&array.slice(0..1)?, value, nullability, ctx)?;
        return Ok(ConstantArray::new(contains.execute_scalar(0, ctx)?, array.len()).into_array());
    }

    let list_array = array.clone().execute::<ListViewArray>(ctx)?;

    let elems = list_array.elements();
    if elems.is_empty() {
        // Must return false when a list is empty (but valid), or null when the list itself is null.
        return list_false_or_null(&list_array, nullability);
    }

    let rhs = ConstantArray::new(value.clone(), elems.len());
    let matching_elements = Binary.try_new_array(
        elems.len(),
        Operator::Eq,
        &[elems.clone(), rhs.clone().into_array()],
    )?;

    // TODO(ngates): we should execute this into a Columnar and check for constant.
    let matches = matching_elements.execute::<BoolArray>(ctx)?;

    // Fast path: no elements match.
    if let Some(pred) = matches.as_constant() {
        return match pred.as_bool().value() {
            // All comparisons are invalid (result in `null`), and search is not null because
            // we already checked for null above.
            None => {
                assert!(
                    !rhs.scalar().is_null(),
                    "Search value must not be null here"
                );
                // False, unless the list itself is null in which case we return null.
                list_false_or_null(&list_array, nullability)
            }
            // No elements match, and all comparisons are valid (result in `false`).
            Some(false) => {
                // False, but match the nullability to the input list array.
                Ok(
                    ConstantArray::new(Scalar::bool(false, nullability), list_array.len())
                        .into_array(),
                )
            }
            // All elements match, and all comparisons are valid (result in `true`).
            Some(true) => {
                // True, unless the list itself is empty or NULL.
                list_is_not_empty(&list_array, nullability, ctx)
            }
        };
    }

    // Get the offsets and sizes as primitive arrays. They are non-negative, so reinterpret to
    // unsigned and dispatch over the 4 unsigned widths each (4x4 instead of 8x8).
    let offsets = list_array
        .offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
    let sizes = list_array.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let sizes = sizes.reinterpret_cast(sizes.ptype().to_unsigned());

    // Process based on the offset and size types.
    let list_matches = match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
        match_each_unsigned_integer_ptype!(sizes.ptype(), |S| {
            process_matches::<O, S>(matches, list_array.len(), offsets, sizes)
        })
    });

    Ok(BoolArray::new(
        list_matches,
        list_array.validity()?.union_nullability(nullability),
    )
    .into_array())
}

/// Returns a [`BitBuffer`] where each bit represents if a list contains the scalar, derived from a
/// [`BoolArray`] of matches on the child elements array.
fn process_matches<O, S>(
    matches: BoolArray,
    list_array_len: usize,
    offsets: PrimitiveArray,
    sizes: PrimitiveArray,
) -> BitBuffer
where
    O: IntegerPType,
    S: IntegerPType,
{
    let offsets_slice = offsets.as_slice::<O>();
    let sizes_slice = sizes.as_slice::<S>();
    let bits = matches.bit_buffer_view();

    (0..list_array_len)
        .map(|i| {
            let offset = offsets_slice[i].as_();
            let size = sizes_slice[i].as_();

            // BitIndexIterator yields indices of true bits only. If `.next()` returns
            // `Some(_)`, at least one element in this list's range matches.
            let mut set_bits = BitIndexIterator::new(bits.inner(), offset, size);
            set_bits.next().is_some()
        })
        .collect::<BitBuffer>()
}

/// Returns a `Bool` array with `false` for lists that are valid,
/// or `NULL` if the list itself is null.
fn list_false_or_null(
    list_array: &ListViewArray,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    match list_array.validity()? {
        Validity::NonNullable => {
            // All false.
            Ok(ConstantArray::new(Scalar::bool(false, nullability), list_array.len()).into_array())
        }
        Validity::AllValid => {
            // All false, but nullable.
            Ok(
                ConstantArray::new(Scalar::bool(false, Nullability::Nullable), list_array.len())
                    .into_array(),
            )
        }
        Validity::AllInvalid => {
            // All nulls, must be nullable result.
            Ok(ConstantArray::new(
                Scalar::null(DType::Bool(Nullability::Nullable)),
                list_array.len(),
            )
            .into_array())
        }
        Validity::Array(validity_array) => {
            // Create a new bool array with false, and the provided nulls
            let buffer = BitBuffer::new_unset(list_array.len());
            Ok(BoolArray::new(buffer, Validity::Array(validity_array)).into_array())
        }
    }
}

/// Returns a `Bool` array with `true` for lists which are NOT empty, or `false` if they are empty,
/// or `NULL` if the list itself is null.
fn list_is_not_empty(
    list_array: &ListViewArray,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Short-circuit for all invalid.
    if list_array.validity()?.definitely_all_null() {
        return Ok(ConstantArray::new(
            Scalar::null(DType::Bool(Nullability::Nullable)),
            list_array.len(),
        )
        .into_array());
    }

    let sizes = list_array.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let buffer = match_each_integer_ptype!(sizes.ptype(), |S| {
        BitBuffer::from_iter(sizes.as_slice::<S>().iter().map(|&size| size != S::zero()))
    });

    // Copy over the validity mask from the input.
    Ok(BoolArray::new(
        buffer,
        list_array.validity()?.union_nullability(nullability),
    )
    .into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use itertools::Itertools;
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ListArray;
    use crate::arrays::VarBinArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType::I32;
    use crate::dtype::StructFields;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::gt;
    use crate::expr::list_contains;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::or;
    use crate::expr::root;
    use crate::expr::stats::Stat;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::list_contains::BoolArray;
    use crate::scalar_fn::fns::list_contains::ConstantArray;
    use crate::scalar_fn::fns::list_contains::ListViewArray;
    use crate::scalar_fn::fns::list_contains::PrimitiveArray;
    use crate::stats::StatsSession;
    use crate::stats::stat as stat_expr;
    use crate::validity::Validity;

    static STATS_SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<StatsSession>());

    fn stat(expr: Expression, stat: Stat) -> Expression {
        stat_expr(expr, stat.aggregate_fn().unwrap())
    }

    fn test_array() -> ArrayRef {
        ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2, 2, 2, 3, 3, 3]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 10]).into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array()
    }

    #[test]
    pub fn test_one() {
        let arr = test_array();

        let expr = list_contains(root(), lit(1));
        let item = arr.apply(&expr).unwrap();

        assert_eq!(
            item.execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_all() {
        let arr = test_array();

        let expr = list_contains(root(), lit(2));
        let item = arr.apply(&expr).unwrap();

        assert_eq!(
            item.execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_none() {
        let arr = test_array();

        let expr = list_contains(root(), lit(4));
        let item = arr.apply(&expr).unwrap();

        assert_eq!(
            item.execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
        assert_eq!(
            item.execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_empty() {
        let arr = ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 5]).into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array();

        let expr = list_contains(root(), lit(2));
        let item = arr.apply(&expr).unwrap();

        assert_eq!(
            item.execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_nullable() {
        let arr = ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 5]).into_array(),
            Validity::Array(BoolArray::from(BitBuffer::from(vec![true, false])).into_array()),
        )
        .unwrap()
        .into_array();

        let expr = list_contains(root(), lit(2));
        let item = arr.apply(&expr).unwrap();

        assert_eq!(
            item.execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert!(
            !item
                .is_valid(1, &mut array_session().create_execution_ctx())
                .unwrap()
        );
    }

    #[test]
    pub fn test_return_type() {
        let scope = DType::Struct(
            StructFields::new(
                ["array"].into(),
                vec![DType::List(
                    Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                    Nullability::Nullable,
                )],
            ),
            Nullability::NonNullable,
        );

        let expr = list_contains(get_item("array", root()), lit(2));

        // Expect nullable, although scope is non-nullable
        assert_eq!(
            expr.return_dtype(&scope).unwrap(),
            DType::Bool(Nullability::Nullable)
        );
    }

    #[test]
    pub fn list_falsification() -> VortexResult<()> {
        let expr = list_contains(
            lit(Scalar::list(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                vec![1.into(), 2.into(), 3.into()],
                Nullability::NonNullable,
            )),
            col("a"),
        );
        let scope = DType::Struct(
            StructFields::new(
                ["a"].into(),
                vec![DType::Primitive(I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );

        // The falsifier describes the list as intervals: the scope lies wholly
        // outside the list's range, or wholly inside a gap between two adjacent
        // list values. For a list this dense the gaps cover every element, so
        // it proves exactly what a term per element would.
        assert_eq!(
            expr.falsify(&scope, &STATS_SESSION)?,
            Some(or(
                or(
                    lt(stat(col("a"), Stat::Max), lit(1i32)),
                    gt(stat(col("a"), Stat::Min), lit(3i32)),
                ),
                or(
                    and(
                        gt(stat(col("a"), Stat::Min), lit(1i32)),
                        lt(stat(col("a"), Stat::Max), lit(2i32)),
                    ),
                    and(
                        gt(stat(col("a"), Stat::Min), lit(2i32)),
                        lt(stat(col("a"), Stat::Max), lit(3i32)),
                    ),
                ),
            ))
        );
        Ok(())
    }

    #[test]
    pub fn test_display() {
        let expr = list_contains(get_item("tags", root()), lit("urgent"));
        assert_eq!(expr.to_string(), "vortex.list.contains($.tags, \"urgent\")");

        let expr2 = list_contains(root(), lit(42));
        assert_eq!(expr2.to_string(), "vortex.list.contains($, 42i32)");
    }

    #[test]
    pub fn test_constant_scalars() {
        let arr = test_array();

        // Both list and needle are constants - should use scalar optimization
        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
            vec![1.into(), 2.into(), 3.into()],
            Nullability::NonNullable,
        );

        // Test contains true
        let expr = list_contains(lit(list_scalar.clone()), lit(2i32));
        let result = arr.clone().apply(&expr).unwrap();
        assert_eq!(
            result
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::NonNullable)
        );

        // Test contains false
        let expr = list_contains(lit(list_scalar), lit(42i32));
        let result = arr.apply(&expr).unwrap();
        assert_eq!(
            result
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::NonNullable)
        );
    }

    // -- Tests migrated from compute/list_contains.rs --

    fn nonnull_strings(values: Vec<Vec<&str>>) -> ArrayRef {
        let mut ctx = array_session().create_execution_ctx();

        ListArray::from_iter_slow::<u64, _>(values, Arc::new(DType::Utf8(Nullability::NonNullable)))
            .unwrap()
            .into_array()
            .execute::<ListViewArray>(&mut ctx)
            .vortex_expect("failed to convert to listview")
            .into_array()
    }

    fn null_strings(values: Vec<Vec<Option<&str>>>) -> ArrayRef {
        let elements = values.iter().flatten().cloned().collect_vec();

        let mut offsets = values
            .iter()
            .scan(0u64, |st, v| {
                *st += v.len() as u64;
                Some(*st)
            })
            .collect_vec();
        offsets.insert(0, 0u64);
        let offsets = Buffer::from_iter(offsets).into_array();

        let elements =
            VarBinArray::from_iter(elements, DType::Utf8(Nullability::Nullable)).into_array();

        let mut ctx = array_session().create_execution_ctx();

        ListArray::try_new(elements, offsets, Validity::NonNullable)
            .unwrap()
            .as_array()
            .clone()
            .execute::<ListViewArray>(&mut ctx)
            .vortex_expect("failed to convert to listview")
            .into_array()
    }

    fn bool_array(values: Vec<bool>, validity: Validity) -> BoolArray {
        BoolArray::new(values.into_iter().collect(), validity)
    }

    #[rstest]
    #[case(
        nonnull_strings(vec![vec![], vec!["a"], vec!["a", "b"]]),
        Some("a"),
        bool_array(vec![false, true, true], Validity::NonNullable)
    )]
    #[case(
        null_strings(vec![vec![], vec![Some("a"), None], vec![Some("a"), None, Some("b")]]),
        Some("a"),
        bool_array(vec![false, true, true], Validity::AllValid)
    )]
    #[case(
        null_strings(vec![vec![], vec![Some("a"), None], vec![Some("b"), None, None]]),
        Some("a"),
        bool_array(vec![false, true, false], Validity::AllValid)
    )]
    #[case(
        nonnull_strings(vec![vec![], vec!["a"], vec!["a"]]),
        Some("a"),
        bool_array(vec![false, true, true], Validity::NonNullable)
    )]
    #[case(
        nonnull_strings(vec![vec![], vec![], vec![]]),
        Some("a"),
        bool_array(vec![false, false, false], Validity::NonNullable)
    )]
    #[case(
        nonnull_strings(vec![vec!["b"], vec![], vec!["b"]]),
        Some("a"),
        bool_array(vec![false, false, false], Validity::NonNullable)
    )]
    #[case(
        null_strings(vec![vec![], vec![None, None], vec![None, None, None]]),
        None,
        bool_array(vec![false, true, true], Validity::AllInvalid)
    )]
    #[case(
        null_strings(vec![vec![], vec![None, None], vec![None, None, None]]),
        Some("a"),
        bool_array(vec![false, false, false], Validity::AllValid)
    )]
    fn test_contains_nullable(
        #[case] list_array: ArrayRef,
        #[case] value: Option<&str>,
        #[case] expected: BoolArray,
    ) {
        let mut ctx = array_session().create_execution_ctx();
        let element_nullability = list_array
            .dtype()
            .as_list_element_opt()
            .unwrap()
            .nullability();
        let scalar = match value {
            None => Scalar::null(DType::Utf8(Nullability::Nullable)),
            Some(v) => Scalar::utf8(v, element_nullability),
        };
        let elem = ConstantArray::new(scalar, list_array.len());
        let expr = list_contains(root(), lit(elem.scalar().clone()));
        let result = list_array.apply(&expr).unwrap();
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_constant_list() {
        let mut ctx = array_session().create_execution_ctx();
        let list_array = ConstantArray::new(
            Scalar::list(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                vec![1i32.into(), 2i32.into(), 3i32.into()],
                Nullability::NonNullable,
            ),
            2,
        )
        .into_array();

        let expr = list_contains(root(), lit(2i32));
        let contains = list_array.apply(&expr).unwrap();
        let expected = BoolArray::from_iter([true, true]);
        assert_arrays_eq!(contains, expected, &mut ctx);
    }

    #[test]
    fn test_all_nulls() {
        let mut ctx = array_session().create_execution_ctx();
        let list_array = ConstantArray::new(
            Scalar::null(DType::List(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                Nullability::Nullable,
            )),
            5,
        )
        .into_array();

        let expr = list_contains(root(), lit(2i32));
        let contains = list_array.apply(&expr).unwrap();

        let expected = BoolArray::new(
            [false, false, false, false, false].into_iter().collect(),
            Validity::AllInvalid,
        );
        assert_arrays_eq!(contains, expected, &mut ctx);
    }

    #[test]
    fn test_list_array_element() {
        let mut ctx = array_session().create_execution_ctx();
        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
            vec![1.into(), 3.into(), 6.into()],
            Nullability::NonNullable,
        );

        let arr = (0..7).collect::<PrimitiveArray>().into_array();
        let expr = list_contains(lit(list_scalar), root());
        let contains = arr.apply(&expr).unwrap();

        let expected = BoolArray::from_iter([false, true, false, true, false, false, true]);
        assert_arrays_eq!(contains, expected, &mut ctx);
    }

    #[test]
    fn test_list_contains_empty_listview() {
        let mut ctx = array_session().create_execution_ctx();
        let empty_elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
        let offsets = Buffer::from_iter([0u32, 0, 0, 0]).into_array();
        let sizes = Buffer::from_iter([0u32, 0, 0, 0]).into_array();

        let list_array = unsafe {
            ListViewArray::new_unchecked(
                empty_elements.into_array(),
                offsets,
                sizes,
                Validity::NonNullable,
            )
            .with_zero_copy_to_list(true)
        };

        let expr = list_contains(root(), lit(42i32));
        let result = list_array.into_array().apply(&expr).unwrap();

        let expected = BoolArray::from_iter([false, false, false, false]);
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_list_contains_all_null_elements() {
        let mut ctx = array_session().create_execution_ctx();
        let elements = PrimitiveArray::from_option_iter::<i32, _>([None, None, None, None, None]);
        let offsets = Buffer::from_iter([0u32, 2, 4]).into_array();
        let sizes = Buffer::from_iter([2u32, 2, 1]).into_array();

        let list_array = unsafe {
            ListViewArray::new_unchecked(
                elements.into_array(),
                offsets,
                sizes,
                Validity::NonNullable,
            )
            .with_zero_copy_to_list(true)
        };

        // Searching for null
        let null_scalar = Scalar::null(DType::Primitive(I32, Nullability::Nullable));
        let expr = list_contains(root(), lit(null_scalar));
        let result = list_array.clone().into_array().apply(&expr).unwrap();

        let expected = BoolArray::new(
            [false, false, false].into_iter().collect(),
            Validity::AllInvalid,
        );
        assert_arrays_eq!(result, expected, &mut ctx);

        // Searching for non-null
        let expr2 = list_contains(root(), lit(42i32));
        let result2 = list_array.into_array().apply(&expr2).unwrap();

        let expected2 = BoolArray::from_iter([false, false, false]);
        assert_arrays_eq!(result2, expected2, &mut ctx);
    }

    #[test]
    fn test_list_contains_large_offsets() {
        let mut ctx = array_session().create_execution_ctx();
        let elements = Buffer::from_iter([1i32, 2, 3, 4, 5]).into_array();

        let offsets = Buffer::from_iter([0u32, 1, 4, 0]).into_array();
        let sizes = Buffer::from_iter([1u32, 2, 1, 0]).into_array();

        let list_array =
            ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

        let expr = list_contains(root(), lit(2i32));
        let result = list_array.clone().into_array().apply(&expr).unwrap();

        let expected = BoolArray::from_iter([false, true, false, false]);
        assert_arrays_eq!(result, expected, &mut ctx);

        let expr5 = list_contains(root(), lit(5i32));
        let result5 = list_array.into_array().apply(&expr5).unwrap();

        let expected5 = BoolArray::from_iter([false, false, true, false]);
        assert_arrays_eq!(result5, expected5, &mut ctx);
    }

    #[test]
    fn test_list_contains_offset_size_boundary() {
        let mut ctx = array_session().create_execution_ctx();
        let elements = Buffer::from_iter(0..256).into_array();
        let offsets = Buffer::from_iter([0u8, 100, 200, 254]).into_array();
        let sizes = Buffer::from_iter([50u8, 50, 54, 2]).into_array();

        let list_array =
            ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

        let expr = list_contains(root(), lit(255i32));
        let result = list_array.clone().into_array().apply(&expr).unwrap();

        let expected = BoolArray::from_iter([false, false, false, true]);
        assert_arrays_eq!(result, expected, &mut ctx);

        let expr_zero = list_contains(root(), lit(0i32));
        let result_zero = list_array.into_array().apply(&expr_zero).unwrap();

        let expected_zero = BoolArray::from_iter([true, false, false, false]);
        assert_arrays_eq!(result_zero, expected_zero, &mut ctx);
    }
}
