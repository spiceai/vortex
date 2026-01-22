// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CASE WHEN expression for conditional value selection.
//!
//! This expression evaluates a series of WHEN conditions and returns the corresponding
//! THEN value for the first condition that evaluates to true. If no conditions match
//! and an ELSE clause is provided, the ELSE value is returned; otherwise, NULL is returned.
//!
//! # Structure
//!
//! The expression has children in the following order:
//! - pairs of (condition, value) for each WHEN/THEN clause
//! - optionally, a final ELSE value
//!
//! For example, `CASE WHEN a THEN 1 WHEN b THEN 2 ELSE 3 END` has children:
//! `[a, 1, b, 2, 3]`

use std::fmt;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use prost::Message;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_scalar::Scalar;
use vortex_vector::Datum;
use vortex_vector::VectorOps;

use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::compute::zip;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::expression::Expression;

/// Options for the CaseWhen expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CaseWhenOptions {
    /// Number of WHEN/THEN pairs (each pair contributes 2 children)
    pub num_when_then_pairs: u32,
    /// Whether an ELSE clause is present (contributes 1 child at the end)
    pub has_else: bool,
}

impl fmt::Display for CaseWhenOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "case_when(pairs={}, else={})",
            self.num_when_then_pairs, self.has_else
        )
    }
}

/// A CASE WHEN expression.
///
/// Evaluates conditions in order and returns the value corresponding to the
/// first matching condition.
pub struct CaseWhen;

impl VTable for CaseWhen {
    type Options = CaseWhenOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.case_when")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::CaseWhenOpts {
                num_when_then_pairs: options.num_when_then_pairs,
                has_else: options.has_else,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Options> {
        let opts = pb::CaseWhenOpts::decode(metadata)?;
        Ok(CaseWhenOptions {
            num_when_then_pairs: opts.num_when_then_pairs,
            has_else: opts.has_else,
        })
    }

    fn arity(&self, options: &Self::Options) -> Arity {
        let num_children =
            options.num_when_then_pairs as usize * 2 + if options.has_else { 1 } else { 0 };
        Arity::Exact(num_children)
    }

    fn child_name(&self, options: &Self::Options, child_idx: usize) -> ChildName {
        let pair_count = options.num_when_then_pairs as usize;
        let num_when_then_children = pair_count * 2;

        if child_idx < num_when_then_children {
            let pair_idx = child_idx / 2;
            if child_idx % 2 == 0 {
                ChildName::from(Arc::from(format!("when_{}", pair_idx)))
            } else {
                ChildName::from(Arc::from(format!("then_{}", pair_idx)))
            }
        } else if options.has_else && child_idx == num_when_then_children {
            ChildName::from("else")
        } else {
            unreachable!(
                "Invalid child index {} for CaseWhen expression with {} pairs",
                child_idx, pair_count
            )
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "CASE")?;
        for i in 0..options.num_when_then_pairs as usize {
            write!(
                f,
                " WHEN {} THEN {}",
                expr.child(i * 2),
                expr.child(i * 2 + 1)
            )?;
        }
        if options.has_else {
            let else_idx = options.num_when_then_pairs as usize * 2;
            write!(f, " ELSE {}", expr.child(else_idx))?;
        }
        write!(f, " END")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        // The return dtype is based on the THEN expressions
        if options.num_when_then_pairs == 0 {
            vortex_bail!("CaseWhen must have at least one WHEN/THEN pair");
        }

        // Get the first THEN expression's dtype (index 1)
        let first_then_dtype = &arg_dtypes[1];

        // If there's no ELSE, the result is always nullable (unmatched rows are NULL)
        if !options.has_else {
            Ok(first_then_dtype.as_nullable())
        } else {
            Ok(first_then_dtype.clone())
        }
    }

    fn evaluate(
        &self,
        options: &Self::Options,
        expr: &Expression,
        scope: &ArrayRef,
    ) -> VortexResult<ArrayRef> {
        use vortex_buffer::BitBuffer;
        use vortex_mask::Mask;

        use crate::compute::filter;

        let len = scope.len();

        // Determine output dtype from first THEN expression
        let output_dtype = expr.child(1).return_dtype(scope.dtype())?;
        let else_idx = options.num_when_then_pairs as usize * 2;

        // Track which rows have matched a condition (using BitBuffer for boolean ops)
        let mut matched_bits = BitBuffer::new_unset(len);

        // Start with null result - we'll fill in values as conditions match
        let mut result: ArrayRef =
            ConstantArray::new(Scalar::null(output_dtype.as_nullable()), len).into_array();

        // Process when/then pairs in order (first match wins)
        for i in 0..options.num_when_then_pairs as usize {
            // Evaluate condition
            let cond = expr.child(i * 2).evaluate(scope)?;
            let cond_bool = cond.to_bool();
            let cond_mask = cond_bool.to_mask_fill_null_false();
            let cond_bits = cond_mask.to_bit_buffer();

            // Compute which rows match THIS condition AND haven't matched a previous one
            // effective_cond = cond AND NOT(already_matched)
            let effective_bits = &cond_bits & &(!&matched_bits);
            let effective_mask = Mask::from_buffer(effective_bits.clone());

            // Short-circuit: skip THEN evaluation if no rows match this condition
            if effective_mask.all_false() {
                continue;
            }

            // Evaluate THEN expression
            let then_val = if effective_mask.all_true() {
                // All rows match - safe to evaluate on full scope
                expr.child(i * 2 + 1).evaluate(scope)?
            } else {
                // Filter scope to only matching rows, evaluate, then scatter back
                let filtered_scope = filter(scope, &effective_mask)?;
                let filtered_result = expr.child(i * 2 + 1).evaluate(&filtered_scope)?;

                // Scatter the filtered result back using builder
                scatter_with_mask(&filtered_result, &effective_mask, &output_dtype, len)?
            };

            // Merge into result: use zip to overlay then_val where effective_mask is true
            result = zip(&then_val, &result, &effective_mask)?;

            // Update matched_bits
            matched_bits = &matched_bits | &effective_bits;

            // Short-circuit: if all rows have matched, we're done
            if matched_bits.true_count() == len {
                break;
            }
        }

        // Handle ELSE clause for unmatched rows
        let unmatched_bits = !&matched_bits;
        let unmatched_mask = Mask::from_buffer(unmatched_bits);

        if !unmatched_mask.all_false() {
            let else_val = if options.has_else {
                // Evaluate ELSE for unmatched rows
                if unmatched_mask.all_true() {
                    expr.child(else_idx).evaluate(scope)?
                } else {
                    let filtered_scope = filter(scope, &unmatched_mask)?;
                    let filtered_else = expr.child(else_idx).evaluate(&filtered_scope)?;
                    scatter_with_mask(&filtered_else, &unmatched_mask, &output_dtype, len)?
                }
            } else {
                // No ELSE - unmatched rows stay null (already set in result)
                return Ok(result);
            };

            result = zip(&else_val, &result, &unmatched_mask)?;
        }

        Ok(result)
    }

    fn execute(&self, options: &Self::Options, args: ExecutionArgs) -> VortexResult<Datum> {
        let row_count = args.row_count;
        let mut datums = args.datums;

        // Check if all inputs are scalars (for returning scalar result)
        let all_scalars = datums.iter().all(|d| matches!(d, Datum::Scalar(_)));

        // Collect when/then pairs from datums
        let mut when_then_pairs = Vec::with_capacity(options.num_when_then_pairs as usize);
        for i in 0..options.num_when_then_pairs as usize {
            let cond = datums[i * 2].clone();
            let then_val = datums[i * 2 + 1].clone();
            when_then_pairs.push((cond, then_val));
        }

        // Get the else value if present
        let else_value = options.has_else.then(|| {
            let else_idx = options.num_when_then_pairs as usize * 2;
            datums.remove(else_idx)
        });

        // Determine output dtype from return_dtype
        let output_dtype = args.return_dtype;

        // Create the result by starting from the else value or null
        let mut result: Datum = if let Some(else_val) = else_value {
            else_val
        } else {
            // Create a null scalar of the output dtype, which will be repeated as needed
            use vortex_vector::Scalar as VScalar;
            Datum::Scalar(VScalar::null(&output_dtype))
        };

        // Process when/then pairs in reverse order
        // For each (condition, then_value), we select from then_value where condition is true
        for (cond, then_val) in when_then_pairs.into_iter().rev() {
            result = execute_zip(then_val, result, cond, row_count, &output_dtype)?;
        }

        // If all inputs were scalars and result is still length 1, return as scalar
        if all_scalars
            && let Datum::Vector(v) = &result
            && v.len() == 1
        {
            return Ok(Datum::Scalar(v.scalar_at(0)));
        }

        Ok(result)
    }
}

/// Helper function to perform zip operation on Datum values.
/// Selects from `if_true` where `condition` is true, otherwise from `if_false`.
fn execute_zip(
    if_true: Datum,
    if_false: Datum,
    condition: Datum,
    row_count: usize,
    output_dtype: &DType,
) -> VortexResult<Datum> {
    use vortex_mask::Mask;
    use vortex_vector::BoolDatum;

    use crate::LEGACY_SESSION;
    use crate::VectorExecutor;
    use crate::vectors::VectorIntoArray;

    let cond_bool = condition.into_bool();

    // Convert condition to Mask using the same pattern as evaluate()
    let mask = match cond_bool {
        BoolDatum::Scalar(s) => {
            let value = s.value().unwrap_or(false); // NULL treated as false
            Mask::new(row_count, value)
        }
        BoolDatum::Vector(v) => {
            // Convert to BoolArray and use to_mask_fill_null_false() for DRY
            let bool_dtype = DType::Bool(Nullability::Nullable);
            let bool_array: BoolArray = v.into_array(&bool_dtype);
            bool_array.to_mask_fill_null_false()
        }
    };

    // Short-circuit: if mask is all true, return if_true; if all false, return if_false
    if mask.all_true() {
        return Ok(if_true);
    }
    if mask.all_false() {
        return Ok(if_false);
    }

    // Convert datums to vectors for zip
    let true_vector = if_true.unwrap_into_vector(row_count);
    let false_vector = if_false.unwrap_into_vector(row_count);

    // Convert vectors to arrays for zip operation
    let true_array = true_vector.into_array(output_dtype);
    let false_array = false_vector.into_array(output_dtype);

    // Perform zip
    let result_array = zip(&true_array, &false_array, &mask)?;

    // Convert back to vector
    let result_vector = result_array.execute_vector(&LEGACY_SESSION)?;

    Ok(Datum::Vector(result_vector))
}

/// Creates a CASE WHEN expression with an ELSE clause.
///
/// The children should be provided as: condition1, then1, condition2, then2, ..., else_value
///
/// # Example
/// ```ignore
/// // CASE WHEN x > 0 THEN 'positive' WHEN x < 0 THEN 'negative' ELSE 'zero' END
/// case_when(vec![
///     gt(col("x"), lit(0)), lit("positive"),
///     lt(col("x"), lit(0)), lit("negative"),
///     lit("zero"),
/// ])
/// ```
pub fn case_when<I: IntoIterator<Item = Expression>>(children: I) -> Expression {
    let children: Vec<_> = children.into_iter().collect();
    let num_children = children.len();

    // Must have odd number of children (pairs + else)
    assert!(
        num_children >= 3 && num_children % 2 == 1,
        "case_when requires at least one when/then pair and an else: got {} children",
        num_children
    );

    #[allow(clippy::cast_possible_truncation)]
    let num_when_then_pairs = ((num_children - 1) / 2) as u32;
    let options = CaseWhenOptions {
        num_when_then_pairs,
        has_else: true,
    };

    CaseWhen.new_expr(options, children)
}

/// Creates a CASE WHEN expression without an ELSE clause (returns NULL when no conditions match).
///
/// The children should be provided as: condition1, then1, condition2, then2, ...
///
/// # Example
/// ```ignore
/// // CASE WHEN x > 0 THEN 'positive' WHEN x < 0 THEN 'negative' END
/// // (returns NULL when x = 0)
/// case_when_no_else(vec![
///     gt(col("x"), lit(0)), lit("positive"),
///     lt(col("x"), lit(0)), lit("negative"),
/// ])
/// ```
pub fn case_when_no_else<I: IntoIterator<Item = Expression>>(children: I) -> Expression {
    let children: Vec<_> = children.into_iter().collect();
    let num_children = children.len();

    // Must have even number of children (pairs only)
    assert!(
        num_children >= 2 && num_children % 2 == 0,
        "case_when_no_else requires at least one when/then pair: got {} children",
        num_children
    );

    #[allow(clippy::cast_possible_truncation)]
    let num_when_then_pairs = (num_children / 2) as u32;
    let options = CaseWhenOptions {
        num_when_then_pairs,
        has_else: false,
    };

    CaseWhen.new_expr(options, children)
}

/// Scatter values from a filtered (shorter) array back to their original positions.
/// The mask indicates which positions in the output should receive values from the source.
/// Positions where mask is false will be null.
fn scatter_with_mask(
    source: &ArrayRef,
    mask: &vortex_mask::Mask,
    dtype: &DType,
    output_len: usize,
) -> VortexResult<ArrayRef> {
    use crate::builders::builder_with_capacity;

    let nullable_dtype = dtype.as_nullable();
    let mut builder = builder_with_capacity(&nullable_dtype, output_len);
    let mut source_idx = 0;

    for i in 0..output_len {
        if mask.value(i) {
            // Copy value from source, casting to nullable if needed
            let scalar = source.scalar_at(source_idx);
            let nullable_scalar = scalar.cast(&nullable_dtype)?;
            builder.append_scalar(&nullable_scalar)?;
            source_idx += 1;
        } else {
            // Insert null
            builder.append_null();
        }
    }

    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_error::VortexExpect as _;
    use vortex_scalar::Scalar;

    use super::*;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::expr::exprs::binary::eq;
    use crate::expr::exprs::binary::gt;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::root::root;
    use crate::expr::test_harness;

    // ==================== Serialization Tests ====================

    #[test]
    fn test_serialization_roundtrip() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 2,
            has_else: true,
        };

        let serialized = CaseWhen.serialize(&options).unwrap().unwrap();
        let deserialized = CaseWhen.deserialize(&serialized).unwrap();

        assert_eq!(options, deserialized);
    }

    #[test]
    fn test_serialization_no_else() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 3,
            has_else: false,
        };

        let serialized = CaseWhen.serialize(&options).unwrap().unwrap();
        let deserialized = CaseWhen.deserialize(&serialized).unwrap();

        assert_eq!(options, deserialized);
    }

    // ==================== Display Tests ====================

    #[test]
    fn test_display_with_else() {
        // CASE WHEN col > 0 THEN 100 ELSE 0 END
        let condition = gt(col("value"), lit(0i32));
        let then_val = lit(100i32);
        let else_val = lit(0i32);

        let expr = case_when([condition, then_val, else_val]);
        let display = format!("{}", expr);
        assert!(display.contains("CASE"));
        assert!(display.contains("WHEN"));
        assert!(display.contains("THEN"));
        assert!(display.contains("ELSE"));
        assert!(display.contains("END"));
    }

    #[test]
    fn test_display_no_else() {
        // CASE WHEN col > 0 THEN 100 END
        let condition = gt(col("value"), lit(0i32));
        let then_val = lit(100i32);

        let expr = case_when_no_else([condition, then_val]);
        let display = format!("{}", expr);
        assert!(display.contains("CASE"));
        assert!(display.contains("WHEN"));
        assert!(display.contains("THEN"));
        assert!(!display.contains("ELSE"));
        assert!(display.contains("END"));
    }

    #[test]
    fn test_display_multiple_conditions() {
        let expr = case_when([
            gt(col("x"), lit(10i32)),
            lit("high"),
            gt(col("x"), lit(5i32)),
            lit("medium"),
            lit("low"),
        ]);
        let display = format!("{}", expr);
        // Should contain two WHEN clauses
        assert_eq!(display.matches("WHEN").count(), 2);
        assert_eq!(display.matches("THEN").count(), 2);
    }

    // ==================== DType Tests ====================

    #[test]
    fn test_return_dtype_with_else() {
        let condition = lit(true);
        let then_val = lit(100i32);
        let else_val = lit(0i32);

        let expr = case_when([condition, then_val, else_val]);
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result_dtype = expr.return_dtype(&input_dtype).unwrap();
        // With else, result dtype matches the then expression
        assert_eq!(
            result_dtype,
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_return_dtype_without_else_is_nullable() {
        let condition = lit(true);
        let then_val = lit(100i32);

        let expr = case_when_no_else([condition, then_val]);
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result_dtype = expr.return_dtype(&input_dtype).unwrap();
        // Without else, result is always nullable
        assert_eq!(
            result_dtype,
            DType::Primitive(PType::I32, Nullability::Nullable)
        );
    }

    #[test]
    fn test_return_dtype_with_struct_input() {
        let dtype = test_harness::struct_dtype();

        // CASE WHEN $.col1 > 10 THEN 100 ELSE 0 END
        let expr = case_when([
            gt(get_item("col1", root()), lit(10u16)),
            lit(100i32),
            lit(0i32),
        ]);

        let result_dtype = expr.return_dtype(&dtype).unwrap();
        assert_eq!(
            result_dtype,
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    // ==================== Arity Tests ====================

    #[test]
    fn test_arity_with_else() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 2,
            has_else: true,
        };
        // 2 pairs (4 children) + 1 else = 5 children
        assert_eq!(CaseWhen.arity(&options), Arity::Exact(5));
    }

    #[test]
    fn test_arity_without_else() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 2,
            has_else: false,
        };
        // 2 pairs (4 children) = 4 children
        assert_eq!(CaseWhen.arity(&options), Arity::Exact(4));
    }

    #[test]
    fn test_arity_single_condition() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };
        // 1 pair (2 children) + 1 else = 3 children
        assert_eq!(CaseWhen.arity(&options), Arity::Exact(3));
    }

    // ==================== Child Name Tests ====================

    #[test]
    fn test_child_names() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 2,
            has_else: true,
        };

        assert_eq!(CaseWhen.child_name(&options, 0).to_string(), "when_0");
        assert_eq!(CaseWhen.child_name(&options, 1).to_string(), "then_0");
        assert_eq!(CaseWhen.child_name(&options, 2).to_string(), "when_1");
        assert_eq!(CaseWhen.child_name(&options, 3).to_string(), "then_1");
        assert_eq!(CaseWhen.child_name(&options, 4).to_string(), "else");
    }

    // ==================== Expression Manipulation Tests ====================

    #[test]
    fn test_replace_children() {
        let expr = case_when([lit(true), lit(1i32), lit(0i32)]);
        expr.with_children([lit(false), lit(2i32), lit(3i32)])
            .vortex_expect("operation should succeed in test");
    }

    // ==================== Evaluate Tests ====================

    #[test]
    fn test_evaluate_simple_condition() {
        // Test: CASE WHEN value > 2 THEN 100 ELSE 0 END
        // Input: [1, 2, 3, 4, 5]
        // Expected: [0, 0, 100, 100, 100]
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when([
            gt(get_item("value", root()), lit(2i32)),
            lit(100i32),
            lit(0i32),
        ]);

        let result = expr.evaluate(&test_array).unwrap().to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[0, 0, 100, 100, 100]);
    }

    #[test]
    fn test_evaluate_multiple_conditions() {
        // Test: CASE WHEN value == 1 THEN 10 WHEN value == 3 THEN 30 ELSE 0 END
        // Input: [1, 2, 3, 4, 5]
        // Expected: [10, 0, 30, 0, 0]
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when([
            eq(get_item("value", root()), lit(1i32)),
            lit(10i32),
            eq(get_item("value", root()), lit(3i32)),
            lit(30i32),
            lit(0i32),
        ]);

        let result = expr.evaluate(&test_array).unwrap().to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[10, 0, 30, 0, 0]);
    }

    #[test]
    fn test_evaluate_first_match_wins() {
        // Test: CASE WHEN value > 2 THEN 100 WHEN value > 3 THEN 200 ELSE 0 END
        // Input: [1, 2, 3, 4, 5]
        // Expected: [0, 0, 100, 100, 100] - first condition wins for values 3, 4, 5
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when([
            gt(get_item("value", root()), lit(2i32)),
            lit(100i32),
            gt(get_item("value", root()), lit(3i32)),
            lit(200i32),
            lit(0i32),
        ]);

        let result = expr.evaluate(&test_array).unwrap().to_primitive();
        // First match wins: 3, 4, 5 all get 100 (from first condition)
        assert_eq!(result.as_slice::<i32>(), &[0, 0, 100, 100, 100]);
    }

    #[test]
    fn test_evaluate_no_else_returns_null() {
        // Test: CASE WHEN value > 3 THEN 100 END
        // Input: [1, 2, 3, 4, 5]
        // Expected: [null, null, null, 100, 100]
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when_no_else([gt(get_item("value", root()), lit(3i32)), lit(100i32)]);

        let result = expr.evaluate(&test_array).unwrap();

        // Check the dtype is nullable
        assert!(result.dtype().is_nullable());

        // Positions 0, 1, 2 should be null, 3, 4 should be 100
        assert_eq!(result.scalar_at(0), Scalar::null(result.dtype().clone()));
        assert_eq!(result.scalar_at(1), Scalar::null(result.dtype().clone()));
        assert_eq!(result.scalar_at(2), Scalar::null(result.dtype().clone()));
        assert_eq!(
            result.scalar_at(3),
            Scalar::from(100i32).cast(result.dtype()).unwrap()
        );
        assert_eq!(
            result.scalar_at(4),
            Scalar::from(100i32).cast(result.dtype()).unwrap()
        );
    }

    #[test]
    fn test_evaluate_all_conditions_false() {
        // Test: CASE WHEN value > 100 THEN 1 ELSE 0 END
        // Input: [1, 2, 3, 4, 5]
        // Expected: [0, 0, 0, 0, 0] - no conditions match
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when([
            gt(get_item("value", root()), lit(100i32)),
            lit(1i32),
            lit(0i32),
        ]);

        let result = expr.evaluate(&test_array).unwrap().to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_evaluate_all_conditions_true() {
        // Test: CASE WHEN value > 0 THEN 100 ELSE 0 END
        // Input: [1, 2, 3, 4, 5]
        // Expected: [100, 100, 100, 100, 100] - all match
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when([
            gt(get_item("value", root()), lit(0i32)),
            lit(100i32),
            lit(0i32),
        ]);

        let result = expr.evaluate(&test_array).unwrap().to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[100, 100, 100, 100, 100]);
    }

    #[test]
    fn test_evaluate_with_literal_condition() {
        // Test: CASE WHEN true THEN 100 ELSE 0 END (constant true condition)
        let test_array = buffer![1i32, 2, 3].into_array();

        let expr = case_when([lit(true), lit(100i32), lit(0i32)]);

        let result = expr.evaluate(&test_array).unwrap();
        // Constant folding should produce a constant array
        if let Some(constant) = result.as_constant() {
            assert_eq!(constant, Scalar::from(100i32));
        } else {
            let prim = result.to_primitive();
            assert_eq!(prim.as_slice::<i32>(), &[100, 100, 100]);
        }
    }

    #[test]
    fn test_evaluate_with_bool_column_result() {
        // Test: CASE WHEN value > 2 THEN true ELSE false END
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when([
            gt(get_item("value", root()), lit(2i32)),
            lit(true),
            lit(false),
        ]);

        let result = expr.evaluate(&test_array).unwrap().to_bool();
        assert_eq!(
            result.bit_buffer().iter().collect::<Vec<_>>(),
            vec![false, false, true, true, true]
        );
    }

    #[test]
    fn test_evaluate_with_nullable_condition() {
        // Test: CASE WHEN nullable_bool THEN 100 ELSE 0 END
        // Where the condition has null values - nulls should be treated as false
        let test_array = StructArray::from_fields(&[(
            "cond",
            BoolArray::from_iter([Some(true), None, Some(false), None, Some(true)]).into_array(),
        )])
        .unwrap()
        .into_array();

        let expr = case_when([get_item("cond", root()), lit(100i32), lit(0i32)]);

        let result = expr.evaluate(&test_array).unwrap().to_primitive();
        // true -> 100, null -> 0 (treated as false), false -> 0
        assert_eq!(result.as_slice::<i32>(), &[100, 0, 0, 0, 100]);
    }

    #[test]
    fn test_evaluate_with_nullable_result_values() {
        // Test: CASE WHEN value > 2 THEN nullable_value ELSE 0 END
        let test_array = StructArray::from_fields(&[
            ("value", buffer![1i32, 2, 3, 4, 5].into_array()),
            (
                "result",
                PrimitiveArray::from_option_iter([Some(10), None, Some(30), Some(40), Some(50)])
                    .into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        let expr = case_when([
            gt(get_item("value", root()), lit(2i32)),
            get_item("result", root()),
            lit(0i32),
        ]);

        let result = expr.evaluate(&test_array).unwrap();
        let prim = result.to_primitive();

        // Values 1, 2 don't match -> 0
        // Value 3 matches -> 30 (from result column)
        // Value 4 matches -> 40
        // Value 5 matches -> 50
        assert_eq!(prim.as_slice::<i32>(), &[0, 0, 30, 40, 50]);
    }

    #[test]
    fn test_evaluate_with_all_null_condition() {
        // Test: CASE WHEN all_nulls THEN 100 ELSE 0 END
        // All null conditions should be treated as false
        let test_array = StructArray::from_fields(&[(
            "cond",
            BoolArray::from_iter([None, None, None]).into_array(),
        )])
        .unwrap()
        .into_array();

        let expr = case_when([get_item("cond", root()), lit(100i32), lit(0i32)]);

        let result = expr.evaluate(&test_array).unwrap().to_primitive();
        // All null -> treated as false -> else value
        assert_eq!(result.as_slice::<i32>(), &[0, 0, 0]);
    }

    // ==================== Execute Tests ====================

    #[test]
    fn test_execute_with_scalar_inputs() {
        use vortex_dtype::PTypeDowncast;
        use vortex_vector::Scalar as VScalar;
        use vortex_vector::bool::BoolScalar;
        use vortex_vector::primitive::PScalar;

        // CASE WHEN true THEN 100 ELSE 0 END with all scalars
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };

        let cond_dtype = DType::Bool(Nullability::NonNullable);
        let then_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let else_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let return_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let datums = vec![
            Datum::Scalar(VScalar::from(BoolScalar::new(Some(true)))),
            Datum::Scalar(VScalar::from(PScalar::new(Some(100i32)))),
            Datum::Scalar(VScalar::from(PScalar::new(Some(0i32)))),
        ];

        let args = ExecutionArgs {
            datums,
            dtypes: vec![cond_dtype, then_dtype, else_dtype],
            row_count: 1,
            return_dtype,
        };

        let result = CaseWhen.execute(&options, args).unwrap();

        // Should return scalar since all inputs were scalars
        match result {
            Datum::Scalar(s) => {
                let prim = s.into_primitive().into_i32();
                assert_eq!(prim.value(), Some(100));
            }
            Datum::Vector(v) => {
                // Also acceptable: a length-1 vector
                assert_eq!(v.len(), 1);
            }
        }
    }

    #[test]
    fn test_execute_with_scalar_false_condition() {
        use vortex_dtype::PTypeDowncast;
        use vortex_vector::Scalar as VScalar;
        use vortex_vector::bool::BoolScalar;
        use vortex_vector::primitive::PScalar;

        // CASE WHEN false THEN 100 ELSE 42 END
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };

        let cond_dtype = DType::Bool(Nullability::NonNullable);
        let then_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let else_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let return_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let datums = vec![
            Datum::Scalar(VScalar::from(BoolScalar::new(Some(false)))),
            Datum::Scalar(VScalar::from(PScalar::new(Some(100i32)))),
            Datum::Scalar(VScalar::from(PScalar::new(Some(42i32)))),
        ];

        let args = ExecutionArgs {
            datums,
            dtypes: vec![cond_dtype, then_dtype, else_dtype],
            row_count: 1,
            return_dtype,
        };

        let result = CaseWhen.execute(&options, args).unwrap();

        match result {
            Datum::Scalar(s) => {
                let prim = s.into_primitive().into_i32();
                assert_eq!(prim.value(), Some(42));
            }
            Datum::Vector(v) => {
                assert_eq!(v.len(), 1);
                let prim = v.into_primitive().into_i32();
                assert_eq!(prim.get(0).copied(), Some(42));
            }
        }
    }

    #[test]
    fn test_execute_with_vector_condition() {
        use vortex_dtype::PTypeDowncast;
        use vortex_vector::Scalar as VScalar;
        use vortex_vector::bool::BoolVector;
        use vortex_vector::primitive::PScalar;

        // CASE WHEN [true, false, true] THEN 100 ELSE 0 END
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };

        let cond_dtype = DType::Bool(Nullability::NonNullable);
        let then_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let else_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let return_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let cond_vector = BoolVector::from_iter([true, false, true]);
        let datums = vec![
            Datum::Vector(cond_vector.into()),
            Datum::Scalar(VScalar::from(PScalar::new(Some(100i32)))),
            Datum::Scalar(VScalar::from(PScalar::new(Some(0i32)))),
        ];

        let args = ExecutionArgs {
            datums,
            dtypes: vec![cond_dtype, then_dtype, else_dtype],
            row_count: 3,
            return_dtype,
        };

        let result = CaseWhen.execute(&options, args).unwrap();

        match result {
            Datum::Vector(v) => {
                let prim = v.into_primitive().into_i32();
                assert_eq!(prim.get(0).copied(), Some(100));
                assert_eq!(prim.get(1).copied(), Some(0));
                assert_eq!(prim.get(2).copied(), Some(100));
            }
            Datum::Scalar(_) => panic!("Expected vector result"),
        }
    }

    #[test]
    fn test_execute_with_nullable_condition() {
        use vortex_dtype::PTypeDowncast;
        use vortex_mask::Mask;
        use vortex_vector::Scalar as VScalar;
        use vortex_vector::bool::BoolVector;
        use vortex_vector::primitive::PScalar;

        // CASE WHEN [true, NULL, false, NULL] THEN 100 ELSE 0 END
        // NULL should be treated as false
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };

        let cond_dtype = DType::Bool(Nullability::Nullable);
        let then_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let else_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let return_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let bits = vortex_buffer::BitBuffer::from_iter([true, true, false, false]);
        let validity = Mask::from_iter([true, false, true, false]); // positions 1, 3 are NULL
        let cond_vector = BoolVector::new(bits, validity);

        let datums = vec![
            Datum::Vector(cond_vector.into()),
            Datum::Scalar(VScalar::from(PScalar::new(Some(100i32)))),
            Datum::Scalar(VScalar::from(PScalar::new(Some(0i32)))),
        ];

        let args = ExecutionArgs {
            datums,
            dtypes: vec![cond_dtype, then_dtype, else_dtype],
            row_count: 4,
            return_dtype,
        };

        let result = CaseWhen.execute(&options, args).unwrap();

        match result {
            Datum::Vector(v) => {
                let prim = v.into_primitive().into_i32();
                assert_eq!(prim.get(0).copied(), Some(100)); // true -> 100
                assert_eq!(prim.get(1).copied(), Some(0)); // NULL -> 0 (treated as false)
                assert_eq!(prim.get(2).copied(), Some(0)); // false -> 0
                assert_eq!(prim.get(3).copied(), Some(0)); // NULL -> 0 (treated as false)
            }
            Datum::Scalar(_) => panic!("Expected vector result"),
        }
    }

    #[test]
    fn test_execute_without_else() {
        use vortex_dtype::PTypeDowncast;
        use vortex_vector::Scalar as VScalar;
        use vortex_vector::bool::BoolVector;
        use vortex_vector::primitive::PScalar;

        // CASE WHEN [true, false, true] THEN 100 END (no else)
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: false,
        };

        let cond_dtype = DType::Bool(Nullability::NonNullable);
        let then_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let return_dtype = DType::Primitive(PType::I32, Nullability::Nullable);

        let cond_vector = BoolVector::from_iter([true, false, true]);
        let datums = vec![
            Datum::Vector(cond_vector.into()),
            Datum::Scalar(VScalar::from(PScalar::new(Some(100i32)))),
        ];

        let args = ExecutionArgs {
            datums,
            dtypes: vec![cond_dtype, then_dtype],
            row_count: 3,
            return_dtype,
        };

        let result = CaseWhen.execute(&options, args).unwrap();

        match result {
            Datum::Vector(v) => {
                let prim = v.into_primitive().into_i32();
                assert_eq!(prim.get(0).copied(), Some(100)); // true -> 100
                assert_eq!(prim.get(1), None); // false -> NULL
                assert_eq!(prim.get(2).copied(), Some(100)); // true -> 100
            }
            Datum::Scalar(_) => panic!("Expected vector result"),
        }
    }

    #[test]
    fn test_execute_multiple_conditions() {
        use vortex_dtype::PTypeDowncast;
        use vortex_vector::Scalar as VScalar;
        use vortex_vector::bool::BoolVector;
        use vortex_vector::primitive::PScalar;

        // CASE WHEN [true, false, false] THEN 10
        //      WHEN [false, true, false] THEN 20
        //      ELSE 0 END
        // Expected: [10, 20, 0]
        let options = CaseWhenOptions {
            num_when_then_pairs: 2,
            has_else: true,
        };

        let cond_dtype = DType::Bool(Nullability::NonNullable);
        let then_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let return_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let cond1 = BoolVector::from_iter([true, false, false]);
        let cond2 = BoolVector::from_iter([false, true, false]);

        let datums = vec![
            Datum::Vector(cond1.into()),
            Datum::Scalar(VScalar::from(PScalar::new(Some(10i32)))),
            Datum::Vector(cond2.into()),
            Datum::Scalar(VScalar::from(PScalar::new(Some(20i32)))),
            Datum::Scalar(VScalar::from(PScalar::new(Some(0i32)))),
        ];

        let args = ExecutionArgs {
            datums,
            dtypes: vec![
                cond_dtype.clone(),
                then_dtype.clone(),
                cond_dtype,
                then_dtype.clone(),
                then_dtype,
            ],
            row_count: 3,
            return_dtype,
        };

        let result = CaseWhen.execute(&options, args).unwrap();

        match result {
            Datum::Vector(v) => {
                let prim = v.into_primitive().into_i32();
                assert_eq!(prim.get(0).copied(), Some(10));
                assert_eq!(prim.get(1).copied(), Some(20));
                assert_eq!(prim.get(2).copied(), Some(0));
            }
            Datum::Scalar(_) => panic!("Expected vector result"),
        }
    }

    #[test]
    fn test_execute_all_true_short_circuit() {
        use vortex_dtype::PTypeDowncast;
        use vortex_vector::Scalar as VScalar;
        use vortex_vector::bool::BoolVector;
        use vortex_vector::primitive::PScalar;

        // CASE WHEN [true, true, true] THEN 100 ELSE 0 END
        // Should short-circuit and return the then value
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };

        let cond_dtype = DType::Bool(Nullability::NonNullable);
        let then_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let else_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let return_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let cond_vector = BoolVector::from_iter([true, true, true]);
        let datums = vec![
            Datum::Vector(cond_vector.into()),
            Datum::Scalar(VScalar::from(PScalar::new(Some(100i32)))),
            Datum::Scalar(VScalar::from(PScalar::new(Some(0i32)))),
        ];

        let args = ExecutionArgs {
            datums,
            dtypes: vec![cond_dtype, then_dtype, else_dtype],
            row_count: 3,
            return_dtype,
        };

        let result = CaseWhen.execute(&options, args).unwrap();

        // Could be scalar (from short-circuit) or vector
        match result {
            Datum::Vector(v) => {
                let prim = v.into_primitive().into_i32();
                assert_eq!(prim.get(0).copied(), Some(100));
                assert_eq!(prim.get(1).copied(), Some(100));
                assert_eq!(prim.get(2).copied(), Some(100));
            }
            Datum::Scalar(s) => {
                let prim = s.into_primitive().into_i32();
                assert_eq!(prim.value(), Some(100));
            }
        }
    }

    #[test]
    fn test_execute_all_false_short_circuit() {
        use vortex_dtype::PTypeDowncast;
        use vortex_vector::Scalar as VScalar;
        use vortex_vector::bool::BoolVector;
        use vortex_vector::primitive::PScalar;

        // CASE WHEN [false, false, false] THEN 100 ELSE 42 END
        // Should short-circuit and return the else value
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };

        let cond_dtype = DType::Bool(Nullability::NonNullable);
        let then_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let else_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let return_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let cond_vector = BoolVector::from_iter([false, false, false]);
        let datums = vec![
            Datum::Vector(cond_vector.into()),
            Datum::Scalar(VScalar::from(PScalar::new(Some(100i32)))),
            Datum::Scalar(VScalar::from(PScalar::new(Some(42i32)))),
        ];

        let args = ExecutionArgs {
            datums,
            dtypes: vec![cond_dtype, then_dtype, else_dtype],
            row_count: 3,
            return_dtype,
        };

        let result = CaseWhen.execute(&options, args).unwrap();

        // Could be scalar (from short-circuit) or vector
        match result {
            Datum::Vector(v) => {
                let prim = v.into_primitive().into_i32();
                assert_eq!(prim.get(0).copied(), Some(42));
                assert_eq!(prim.get(1).copied(), Some(42));
                assert_eq!(prim.get(2).copied(), Some(42));
            }
            Datum::Scalar(s) => {
                let prim = s.into_primitive().into_i32();
                assert_eq!(prim.value(), Some(42));
            }
        }
    }

    #[test]
    fn test_execute_with_null_scalar_condition() {
        use vortex_dtype::PTypeDowncast;
        use vortex_vector::Scalar as VScalar;
        use vortex_vector::bool::BoolScalar;
        use vortex_vector::primitive::PScalar;

        // CASE WHEN NULL THEN 100 ELSE 42 END
        // NULL condition should be treated as false
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };

        let cond_dtype = DType::Bool(Nullability::Nullable);
        let then_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let else_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let return_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let null_bool = BoolScalar::null();
        let datums = vec![
            Datum::Scalar(VScalar::from(null_bool)),
            Datum::Scalar(VScalar::from(PScalar::new(Some(100i32)))),
            Datum::Scalar(VScalar::from(PScalar::new(Some(42i32)))),
        ];

        let args = ExecutionArgs {
            datums,
            dtypes: vec![cond_dtype, then_dtype, else_dtype],
            row_count: 1,
            return_dtype,
        };

        let result = CaseWhen.execute(&options, args).unwrap();

        // NULL condition -> treated as false -> else value
        match result {
            Datum::Scalar(s) => {
                let prim = s.into_primitive().into_i32();
                assert_eq!(prim.value(), Some(42));
            }
            Datum::Vector(v) => {
                let prim = v.into_primitive().into_i32();
                assert_eq!(prim.get(0).copied(), Some(42));
            }
        }
    }

    #[test]
    fn test_evaluate_divide_by_zero_protected_by_case_when() {
        // This test verifies that CASE WHEN properly short-circuits evaluation
        // to avoid divide-by-zero errors.
        // Pattern: CASE WHEN denominator > 0 THEN numerator/denominator ELSE NULL END
        // With input where some denominators are 0, the division should NOT be evaluated
        // for those rows.

        use vortex_buffer::buffer;
        use vortex_dtype::PType;

        use crate::arrays::StructArray;
        use crate::expr::VTableExt;
        use crate::expr::exprs::binary::Binary;
        use crate::expr::exprs::operators::Operator;
        use crate::expr::get_item;
        use crate::expr::gt;
        use crate::expr::lit;
        use crate::expr::root;

        // Create test data: numerator=[10, 20, 30], denominator=[2, 0, 5]
        // Expected: CASE WHEN denominator > 0 THEN numerator/denominator ELSE NULL END
        //         = [5, NULL, 6]
        let test_array = StructArray::from_fields(&[
            ("numerator", buffer![10i32, 20, 30].into_array()),
            ("denominator", buffer![2i32, 0, 5].into_array()),
        ])
        .unwrap()
        .into_array();

        // Build: CASE WHEN $.denominator > 0 THEN $.numerator / $.denominator ELSE null END
        let condition = gt(get_item("denominator", root()), lit(0i32));
        let division = Binary
            .try_new_expr(
                Operator::Div,
                [
                    get_item("numerator", root()),
                    get_item("denominator", root()),
                ],
            )
            .unwrap();
        let null_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let null_val = lit(Scalar::null(null_dtype));

        let expr = case_when([condition, division, null_val]);

        // This should NOT panic with divide-by-zero
        let result = expr.evaluate(&test_array).unwrap();

        // Verify results
        assert_eq!(result.len(), 3);

        // Row 0: 10/2 = 5
        assert_eq!(
            result.scalar_at(0),
            Scalar::from(5i32).cast(result.dtype()).unwrap()
        );

        // Row 1: denominator=0, so result is NULL (division was NOT evaluated)
        assert_eq!(result.scalar_at(1), Scalar::null(result.dtype().clone()));

        // Row 2: 30/5 = 6
        assert_eq!(
            result.scalar_at(2),
            Scalar::from(6i32).cast(result.dtype()).unwrap()
        );
    }
}
