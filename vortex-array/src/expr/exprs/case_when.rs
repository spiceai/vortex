// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! N-ary CASE WHEN expression for conditional value selection.
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
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_scalar::Scalar;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::compute::zip;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExecutionResult;
use crate::expr::ExprId;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::expression::Expression;

/// Options for the N-ary CaseWhen expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CaseWhenOptions {
    /// Number of WHEN/THEN pairs (each pair contributes 2 children)
    pub num_when_then_pairs: u32,
    /// Whether an ELSE clause is present (contributes 1 child at the end).
    /// If false, unmatched rows return NULL.
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

/// An N-ary CASE WHEN expression.
///
/// Evaluates conditions in order and returns the value corresponding to the
/// first matching condition.
///
/// Children are in order: [when_0, then_0, when_1, then_1, ..., else?]
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

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
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
            write!(f, " WHEN ")?;
            expr.child(i * 2).fmt_sql(f)?;
            write!(f, " THEN ")?;
            expr.child(i * 2 + 1).fmt_sql(f)?;
        }
        if options.has_else {
            let else_idx = options.num_when_then_pairs as usize * 2;
            write!(f, " ELSE ")?;
            expr.child(else_idx).fmt_sql(f)?;
        }
        write!(f, " END")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        if options.num_when_then_pairs == 0 {
            vortex_bail!("CaseWhen must have at least one WHEN/THEN pair");
        }

        // The return dtype is based on the first THEN expression (index 1)
        let then_dtype = &arg_dtypes[1];

        // If there's no ELSE, the result is always nullable (unmatched rows are NULL)
        if !options.has_else {
            Ok(then_dtype.as_nullable())
        } else {
            Ok(then_dtype.clone())
        }
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: ExecutionArgs,
    ) -> VortexResult<ExecutionResult> {
        let row_count = args.row_count;
        let num_pairs = options.num_when_then_pairs as usize;

        // Single pair case: use efficient binary implementation
        if num_pairs == 1 {
            return execute_binary_case_when(options.has_else, args);
        }

        // N-ary case: evaluate from right to left (innermost first)
        // CASE WHEN a THEN x WHEN b THEN y ELSE z END
        // evaluates as: CASE WHEN a THEN x ELSE (CASE WHEN b THEN y ELSE z END) END
        //
        // We iterate from the last pair backwards, building up the result.

        // Start with the else value (or null if no else)
        let mut result: ArrayRef = if options.has_else {
            let else_idx = num_pairs * 2;
            args.inputs[else_idx].clone()
        } else {
            // Need to determine the output dtype from the first THEN value
            let first_then = &args.inputs[1];
            let then_dtype = first_then.dtype().as_nullable();
            ConstantArray::new(Scalar::null(then_dtype), row_count).into_array()
        };

        // Process pairs from right to left
        for i in (0..num_pairs).rev() {
            let cond_idx = i * 2;
            let then_idx = i * 2 + 1;

            let condition = &args.inputs[cond_idx];
            let then_value = &args.inputs[then_idx];

            // Execute condition to get a BoolArray
            let cond_bool = condition.clone().execute::<BoolArray>(args.ctx)?;
            // SQL semantics: NULL condition is treated as FALSE (i.e., we take the ELSE branch)
            let mask = cond_bool.to_mask_fill_null_false();

            // Short-circuit: all true -> just return THEN value for this pair
            if mask.all_true() {
                result = then_value.clone();
                continue;
            }

            // Short-circuit: all false -> keep the current result (skip this pair)
            if mask.all_false() {
                continue;
            }

            // Use zip to select: where mask is true, take then_value; else take result
            result = zip(then_value.as_ref(), result.as_ref(), &mask)?;
        }

        result.execute::<ExecutionResult>(args.ctx)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        // CaseWhen is null-sensitive because NULL conditions are treated as false
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Efficient implementation for binary CASE WHEN (single when/then pair)
fn execute_binary_case_when(_has_else: bool, args: ExecutionArgs) -> VortexResult<ExecutionResult> {
    let row_count = args.row_count;

    // Extract inputs based on arity: [condition, then_value] or [condition, then_value, else_value]
    let (condition, then_value, else_value) = match args.inputs.len() {
        2 => {
            let [condition, then_value]: [ArrayRef; 2] = args
                .inputs
                .try_into()
                .map_err(|_| vortex_error::vortex_err!("Expected 2 inputs"))?;
            (condition, then_value, None)
        }
        3 => {
            let [condition, then_value, else_value]: [ArrayRef; 3] = args
                .inputs
                .try_into()
                .map_err(|_| vortex_error::vortex_err!("Expected 3 inputs"))?;
            (condition, then_value, Some(else_value))
        }
        n => vortex_bail!("Binary CaseWhen expects 2 or 3 inputs, got {}", n),
    };

    // Execute condition to get a BoolArray
    let cond_bool = condition.execute::<BoolArray>(args.ctx)?;
    // SQL semantics: NULL condition is treated as FALSE (i.e., we take the ELSE branch)
    let mask = cond_bool.to_mask_fill_null_false();

    // Short-circuit: all true -> just return THEN value
    if mask.all_true() {
        return then_value.execute::<ExecutionResult>(args.ctx);
    }

    // Short-circuit: all false -> return ELSE value or NULL
    if mask.all_false() {
        return match else_value {
            Some(else_value) => else_value.execute::<ExecutionResult>(args.ctx),
            None => {
                // Create NULL constant of appropriate type
                let then_dtype = then_value.dtype().as_nullable();
                Ok(ExecutionResult::constant(
                    Scalar::null(then_dtype),
                    row_count,
                ))
            }
        };
    }

    // Get else value for zip (create NULL constant if no else clause)
    let else_value = else_value.unwrap_or_else(|| {
        let then_dtype = then_value.dtype().as_nullable();
        ConstantArray::new(Scalar::null(then_dtype), row_count).into_array()
    });

    // Use zip to select: where mask is true, take then_value; else take else_value
    let result = zip(then_value.as_ref(), else_value.as_ref(), &mask)?;

    result.execute::<ExecutionResult>(args.ctx)
}

/// Creates an N-ary CASE WHEN expression from a flat list of children.
///
/// # Arguments
/// - `children`: Iterator of expressions in order: [when_0, then_0, when_1, then_1, ..., else]
///
/// The last element is always treated as the ELSE clause.
///
/// # Panics
/// Panics if children has fewer than 3 elements (at least one when/then pair + else required).
///
/// # Example
/// ```ignore
/// // CASE WHEN x > 10 THEN 'high' WHEN x > 5 THEN 'medium' ELSE 'low' END
/// case_when([
///     gt(col("x"), lit(10)), lit("high"),
///     gt(col("x"), lit(5)), lit("medium"),
///     lit("low"),
/// ])
/// ```
#[allow(clippy::cast_possible_truncation)]
pub fn case_when<I: IntoIterator<Item = Expression>>(children: I) -> Expression {
    let children: Vec<_> = children.into_iter().collect();
    assert!(
        children.len() >= 3,
        "case_when requires at least 3 children (one when/then pair + else)"
    );
    assert!(
        children.len() % 2 == 1,
        "case_when with else must have odd number of children"
    );

    let num_when_then_pairs = (children.len() - 1) / 2;
    let options = CaseWhenOptions {
        num_when_then_pairs: num_when_then_pairs as u32,
        has_else: true,
    };
    CaseWhen.new_expr(options, children)
}

/// Creates an N-ary CASE WHEN expression without an ELSE clause.
///
/// Returns NULL when no condition matches.
///
/// # Arguments
/// - `children`: Iterator of expressions in order: [when_0, then_0, when_1, then_1, ...]
///
/// # Panics
/// Panics if children has fewer than 2 elements (at least one when/then pair required).
///
/// # Example
/// ```ignore
/// // CASE WHEN x > 10 THEN 'high' WHEN x > 5 THEN 'medium' END
/// case_when_no_else([
///     gt(col("x"), lit(10)), lit("high"),
///     gt(col("x"), lit(5)), lit("medium"),
/// ])
/// ```
#[allow(clippy::cast_possible_truncation)]
pub fn case_when_no_else<I: IntoIterator<Item = Expression>>(children: I) -> Expression {
    let children: Vec<_> = children.into_iter().collect();
    assert!(
        children.len() >= 2,
        "case_when_no_else requires at least 2 children (one when/then pair)"
    );
    assert!(
        children.len() % 2 == 0,
        "case_when_no_else must have even number of children"
    );

    let num_when_then_pairs = children.len() / 2;
    let options = CaseWhenOptions {
        num_when_then_pairs: num_when_then_pairs as u32,
        has_else: false,
    };
    CaseWhen.new_expr(options, children)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_error::VortexExpect as _;
    use vortex_scalar::Scalar;
    use vortex_session::VortexSession;

    use super::*;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::VortexSessionExecute as _;
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
    use crate::session::ArraySession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    /// Helper to evaluate an expression using the apply+execute pattern
    fn evaluate_expr(expr: &Expression, array: &ArrayRef) -> ArrayRef {
        let mut ctx = SESSION.create_execution_ctx();
        array
            .apply(expr)
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_array()
    }

    // ==================== Serialization Tests ====================

    #[test]
    fn test_serialization_roundtrip() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 2,
            has_else: true,
        };
        let serialized = CaseWhen.serialize(&options).unwrap().unwrap();
        let deserialized = CaseWhen
            .deserialize(&serialized, &VortexSession::empty())
            .unwrap();
        assert_eq!(options, deserialized);
    }

    #[test]
    fn test_serialization_no_else() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: false,
        };
        let serialized = CaseWhen.serialize(&options).unwrap().unwrap();
        let deserialized = CaseWhen
            .deserialize(&serialized, &VortexSession::empty())
            .unwrap();
        assert_eq!(options, deserialized);
    }

    // ==================== Display Tests ====================

    #[test]
    fn test_display_with_else() {
        let expr = case_when([gt(col("value"), lit(0i32)), lit(100i32), lit(0i32)]);
        let display = format!("{}", expr);
        assert!(display.contains("CASE"));
        assert!(display.contains("WHEN"));
        assert!(display.contains("THEN"));
        assert!(display.contains("ELSE"));
        assert!(display.contains("END"));
    }

    #[test]
    fn test_display_no_else() {
        let expr = case_when_no_else([gt(col("value"), lit(0i32)), lit(100i32)]);
        let display = format!("{}", expr);
        assert!(display.contains("CASE"));
        assert!(display.contains("WHEN"));
        assert!(display.contains("THEN"));
        assert!(!display.contains("ELSE"));
        assert!(display.contains("END"));
    }

    #[test]
    fn test_display_nary() {
        // CASE WHEN x > 10 THEN 'high' WHEN x > 5 THEN 'medium' ELSE 'low' END
        let expr = case_when([
            gt(col("x"), lit(10i32)),
            lit("high"),
            gt(col("x"), lit(5i32)),
            lit("medium"),
            lit("low"),
        ]);
        let display = format!("{}", expr);
        // Should contain both WHEN clauses
        assert_eq!(display.matches("WHEN").count(), 2);
        assert_eq!(display.matches("THEN").count(), 2);
        assert!(display.contains("ELSE"));
    }

    // ==================== DType Tests ====================

    #[test]
    fn test_return_dtype_with_else() {
        let expr = case_when([lit(true), lit(100i32), lit(0i32)]);
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result_dtype = expr.return_dtype(&input_dtype).unwrap();
        assert_eq!(
            result_dtype,
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_return_dtype_without_else_is_nullable() {
        let expr = case_when_no_else([lit(true), lit(100i32)]);
        let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result_dtype = expr.return_dtype(&input_dtype).unwrap();
        assert_eq!(
            result_dtype,
            DType::Primitive(PType::I32, Nullability::Nullable)
        );
    }

    #[test]
    fn test_return_dtype_with_struct_input() {
        let dtype = test_harness::struct_dtype();
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
    fn test_arity_single_pair_with_else() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };
        assert_eq!(CaseWhen.arity(&options), Arity::Exact(3));
    }

    #[test]
    fn test_arity_single_pair_without_else() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: false,
        };
        assert_eq!(CaseWhen.arity(&options), Arity::Exact(2));
    }

    #[test]
    fn test_arity_multiple_pairs() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 3,
            has_else: true,
        };
        // 3 pairs * 2 children + 1 else = 7
        assert_eq!(CaseWhen.arity(&options), Arity::Exact(7));
    }

    // ==================== Child Name Tests ====================

    #[test]
    fn test_child_names_single_pair() {
        let options = CaseWhenOptions {
            num_when_then_pairs: 1,
            has_else: true,
        };
        assert_eq!(CaseWhen.child_name(&options, 0).to_string(), "when_0");
        assert_eq!(CaseWhen.child_name(&options, 1).to_string(), "then_0");
        assert_eq!(CaseWhen.child_name(&options, 2).to_string(), "else");
    }

    #[test]
    fn test_child_names_multiple_pairs() {
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
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when([
            gt(get_item("value", root()), lit(2i32)),
            lit(100i32),
            lit(0i32),
        ]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[0, 0, 100, 100, 100]);
    }

    #[test]
    fn test_evaluate_nary_multiple_conditions() {
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

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[10, 0, 30, 0, 0]);
    }

    #[test]
    fn test_evaluate_nary_first_match_wins() {
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        // Both conditions match for values > 3, but first one wins
        let expr = case_when([
            gt(get_item("value", root()), lit(2i32)),
            lit(100i32),
            gt(get_item("value", root()), lit(3i32)),
            lit(200i32),
            lit(0i32),
        ]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[0, 0, 100, 100, 100]);
    }

    #[test]
    fn test_evaluate_no_else_returns_null() {
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when_no_else([gt(get_item("value", root()), lit(3i32)), lit(100i32)]);

        let result = evaluate_expr(&expr, &test_array);
        assert!(result.dtype().is_nullable());

        assert_eq!(
            result.scalar_at(0).unwrap(),
            Scalar::null(result.dtype().clone())
        );
        assert_eq!(
            result.scalar_at(1).unwrap(),
            Scalar::null(result.dtype().clone())
        );
        assert_eq!(
            result.scalar_at(2).unwrap(),
            Scalar::null(result.dtype().clone())
        );
        assert_eq!(
            result.scalar_at(3).unwrap(),
            Scalar::from(100i32).cast(result.dtype()).unwrap()
        );
        assert_eq!(
            result.scalar_at(4).unwrap(),
            Scalar::from(100i32).cast(result.dtype()).unwrap()
        );
    }

    #[test]
    fn test_evaluate_all_conditions_false() {
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when([
            gt(get_item("value", root()), lit(100i32)),
            lit(1i32),
            lit(0i32),
        ]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_evaluate_all_conditions_true() {
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when([
            gt(get_item("value", root()), lit(0i32)),
            lit(100i32),
            lit(0i32),
        ]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[100, 100, 100, 100, 100]);
    }

    #[test]
    fn test_evaluate_with_literal_condition() {
        let test_array = buffer![1i32, 2, 3].into_array();
        let expr = case_when([lit(true), lit(100i32), lit(0i32)]);
        let result = evaluate_expr(&expr, &test_array);

        if let Some(constant) = result.as_constant() {
            assert_eq!(constant, Scalar::from(100i32));
        } else {
            let prim = result.to_primitive();
            assert_eq!(prim.as_slice::<i32>(), &[100, 100, 100]);
        }
    }

    #[test]
    fn test_evaluate_with_bool_column_result() {
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when([
            gt(get_item("value", root()), lit(2i32)),
            lit(true),
            lit(false),
        ]);

        let result = evaluate_expr(&expr, &test_array).to_bool();
        assert_eq!(
            result.to_bit_buffer().iter().collect::<Vec<_>>(),
            vec![false, false, true, true, true]
        );
    }

    #[test]
    fn test_evaluate_with_nullable_condition() {
        let test_array = StructArray::from_fields(&[(
            "cond",
            BoolArray::from_iter([Some(true), None, Some(false), None, Some(true)]).into_array(),
        )])
        .unwrap()
        .into_array();

        let expr = case_when([get_item("cond", root()), lit(100i32), lit(0i32)]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[100, 0, 0, 0, 100]);
    }

    #[test]
    fn test_evaluate_with_nullable_result_values() {
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

        let result = evaluate_expr(&expr, &test_array);
        let prim = result.to_primitive();
        assert_eq!(prim.as_slice::<i32>(), &[0, 0, 30, 40, 50]);
    }

    #[test]
    fn test_evaluate_with_all_null_condition() {
        let test_array = StructArray::from_fields(&[(
            "cond",
            BoolArray::from_iter([None, None, None]).into_array(),
        )])
        .unwrap()
        .into_array();

        let expr = case_when([get_item("cond", root()), lit(100i32), lit(0i32)]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[0, 0, 0]);
    }

    #[test]
    fn test_nary_no_else() {
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        // CASE WHEN value = 1 THEN 10 WHEN value = 3 THEN 30 END (no else)
        let expr = case_when_no_else([
            eq(get_item("value", root()), lit(1i32)),
            lit(10i32),
            eq(get_item("value", root()), lit(3i32)),
            lit(30i32),
        ]);

        let result = evaluate_expr(&expr, &test_array);
        assert!(result.dtype().is_nullable());

        // Values 1 -> 10, 3 -> 30, others -> NULL
        assert_eq!(
            result.scalar_at(0).unwrap(),
            Scalar::from(10i32).cast(result.dtype()).unwrap()
        );
        assert_eq!(
            result.scalar_at(1).unwrap(),
            Scalar::null(result.dtype().clone())
        );
        assert_eq!(
            result.scalar_at(2).unwrap(),
            Scalar::from(30i32).cast(result.dtype()).unwrap()
        );
        assert_eq!(
            result.scalar_at(3).unwrap(),
            Scalar::null(result.dtype().clone())
        );
        assert_eq!(
            result.scalar_at(4).unwrap(),
            Scalar::null(result.dtype().clone())
        );
    }

    // ==================== Advanced N-ary Tests ====================

    #[test]
    fn test_nary_5_conditions() {
        // Test with 5 when/then pairs to stress the n-ary implementation
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array())])
                .unwrap()
                .into_array();

        // CASE WHEN value=1 THEN 100 WHEN value=3 THEN 300 WHEN value=5 THEN 500
        //      WHEN value=7 THEN 700 WHEN value=9 THEN 900 ELSE 0 END
        let expr = case_when([
            eq(get_item("value", root()), lit(1i32)),
            lit(100i32),
            eq(get_item("value", root()), lit(3i32)),
            lit(300i32),
            eq(get_item("value", root()), lit(5i32)),
            lit(500i32),
            eq(get_item("value", root()), lit(7i32)),
            lit(700i32),
            eq(get_item("value", root()), lit(9i32)),
            lit(900i32),
            lit(0i32),
        ]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[100, 0, 300, 0, 500, 0, 700, 0, 900, 0]);
    }

    #[test]
    fn test_nary_all_conditions_short_circuit_true() {
        // First condition matches all rows
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when([
            gt(get_item("value", root()), lit(0i32)), // Always true
            lit(100i32),
            gt(get_item("value", root()), lit(3i32)), // Would match some
            lit(200i32),
            lit(0i32),
        ]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        // All rows should match first condition
        assert_eq!(result.as_slice::<i32>(), &[100, 100, 100, 100, 100]);
    }

    #[test]
    fn test_nary_all_conditions_false() {
        // No conditions match, should return else value
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when([
            gt(get_item("value", root()), lit(100i32)), // Never true
            lit(100i32),
            gt(get_item("value", root()), lit(200i32)), // Never true
            lit(200i32),
            lit(999i32), // Else
        ]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[999, 999, 999, 999, 999]);
    }

    #[test]
    fn test_nary_cascading_conditions() {
        // Test cascading conditions where later conditions catch what earlier ones miss
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 5, 10, 15, 20].into_array())])
                .unwrap()
                .into_array();

        // CASE WHEN value > 15 THEN 4 WHEN value > 10 THEN 3 WHEN value > 5 THEN 2 WHEN value > 1 THEN 1 ELSE 0 END
        let expr = case_when([
            gt(get_item("value", root()), lit(15i32)),
            lit(4i32),
            gt(get_item("value", root()), lit(10i32)),
            lit(3i32),
            gt(get_item("value", root()), lit(5i32)),
            lit(2i32),
            gt(get_item("value", root()), lit(1i32)),
            lit(1i32),
            lit(0i32),
        ]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        // value=1 -> 0, value=5 -> 1, value=10 -> 2, value=15 -> 3, value=20 -> 4
        assert_eq!(result.as_slice::<i32>(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_nary_with_nullable_conditions() {
        // Test n-ary with nullable conditions in the middle
        let test_array = StructArray::from_fields(&[
            ("value", buffer![1i32, 2, 3, 4, 5].into_array()),
            (
                "cond1",
                BoolArray::from_iter([Some(true), None, Some(false), None, Some(true)]).into_array(),
            ),
            (
                "cond2",
                BoolArray::from_iter([Some(false), Some(true), Some(true), Some(false), Some(false)])
                    .into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        // CASE WHEN cond1 THEN 100 WHEN cond2 THEN 200 ELSE 0 END
        let expr = case_when([
            get_item("cond1", root()),
            lit(100i32),
            get_item("cond2", root()),
            lit(200i32),
            lit(0i32),
        ]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        // row 0: cond1=true -> 100
        // row 1: cond1=null(false) -> cond2=true -> 200
        // row 2: cond1=false -> cond2=true -> 200
        // row 3: cond1=null(false) -> cond2=false -> 0
        // row 4: cond1=true -> 100
        assert_eq!(result.as_slice::<i32>(), &[100, 200, 200, 0, 100]);
    }

    #[test]
    fn test_nary_no_else_all_unmatched() {
        // N-ary without else where no conditions match
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3].into_array())])
                .unwrap()
                .into_array();

        let expr = case_when_no_else([
            gt(get_item("value", root()), lit(100i32)), // Never true
            lit(100i32),
            gt(get_item("value", root()), lit(200i32)), // Never true
            lit(200i32),
        ]);

        let result = evaluate_expr(&expr, &test_array);
        assert!(result.dtype().is_nullable());

        // All values should be NULL
        for i in 0..3 {
            assert_eq!(
                result.scalar_at(i).unwrap(),
                Scalar::null(result.dtype().clone())
            );
        }
    }

    #[test]
    fn test_large_array() {
        // Test with a larger array to verify performance characteristics
        let size: i32 = 10000;
        let data: Vec<i32> = (0..size).collect();
        let test_array = StructArray::from_fields(&[("value", Buffer::from(data).into_array())])
            .unwrap()
            .into_array();

        // CASE WHEN value > 7500 THEN 3 WHEN value > 5000 THEN 2 WHEN value > 2500 THEN 1 ELSE 0 END
        let expr = case_when([
            gt(get_item("value", root()), lit(7500i32)),
            lit(3i32),
            gt(get_item("value", root()), lit(5000i32)),
            lit(2i32),
            gt(get_item("value", root()), lit(2500i32)),
            lit(1i32),
            lit(0i32),
        ]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        let result_slice = result.as_slice::<i32>();

        // Verify correctness at boundaries
        assert_eq!(result_slice[0], 0); // value=0 -> 0
        assert_eq!(result_slice[2500], 0); // value=2500 -> 0
        assert_eq!(result_slice[2501], 1); // value=2501 -> 1
        assert_eq!(result_slice[5000], 1); // value=5000 -> 1
        assert_eq!(result_slice[5001], 2); // value=5001 -> 2
        assert_eq!(result_slice[7500], 2); // value=7500 -> 2
        assert_eq!(result_slice[7501], 3); // value=7501 -> 3
        assert_eq!(result_slice[9999], 3); // value=9999 -> 3
    }

    #[test]
    fn test_nary_with_column_results() {
        // Test n-ary where THEN values come from columns, not literals
        let test_array = StructArray::from_fields(&[
            ("value", buffer![1i32, 2, 3, 4, 5].into_array()),
            ("result_a", buffer![10i32, 20, 30, 40, 50].into_array()),
            ("result_b", buffer![100i32, 200, 300, 400, 500].into_array()),
        ])
        .unwrap()
        .into_array();

        // CASE WHEN value < 3 THEN result_a WHEN value > 3 THEN result_b ELSE 0 END
        let expr = case_when([
            gt(lit(3i32), get_item("value", root())), // value < 3
            get_item("result_a", root()),
            gt(get_item("value", root()), lit(3i32)), // value > 3
            get_item("result_b", root()),
            lit(0i32),
        ]);

        let result = evaluate_expr(&expr, &test_array).to_primitive();
        // value=1 -> result_a=10, value=2 -> result_a=20, value=3 -> else=0
        // value=4 -> result_b=400, value=5 -> result_b=500
        assert_eq!(result.as_slice::<i32>(), &[10, 20, 0, 400, 500]);
    }

    #[test]
    fn test_string_results() {
        let test_array =
            StructArray::from_fields(&[("value", buffer![1i32, 2, 3, 4, 5].into_array())])
                .unwrap()
                .into_array();

        // CASE WHEN value < 2 THEN 'low' WHEN value < 4 THEN 'medium' ELSE 'high' END
        let expr = case_when([
            gt(lit(2i32), get_item("value", root())),
            lit("low"),
            gt(lit(4i32), get_item("value", root())),
            lit("medium"),
            lit("high"),
        ]);

        let result = evaluate_expr(&expr, &test_array);
        let varbinview = result.to_varbinview();

        assert_eq!(varbinview.scalar_at(0).unwrap().as_utf8().value().unwrap().as_str(), "low");
        assert_eq!(varbinview.scalar_at(1).unwrap().as_utf8().value().unwrap().as_str(), "medium");
        assert_eq!(varbinview.scalar_at(2).unwrap().as_utf8().value().unwrap().as_str(), "medium");
        assert_eq!(varbinview.scalar_at(3).unwrap().as_utf8().value().unwrap().as_str(), "high");
        assert_eq!(varbinview.scalar_at(4).unwrap().as_utf8().value().unwrap().as_str(), "high");
    }
}
