// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::Expression;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;

/// Converting an expression to a conjunctive normal form can lead to a large number of expression
/// nodes.
/// For now, we will just extract the conjuncts from the expression, and return a vector of conjuncts.
/// We could look at try cnf with a size cap and otherwise return the original conjuncts.
pub fn conjuncts(expr: &Expression) -> Vec<Expression> {
    conjuncts_with_spine_depth(expr).0
}

/// Extract the conjuncts of `expr` along with the depth of the `And` spine they were found under.
///
/// The depth lets a caller distinguish a spine that is already at most as shallow as the balanced
/// tree [`and_collect`](crate::expr::and_collect) builds from one that is still a long chain,
/// without allocating that tree to find out. Compare it against [`balanced_spine_depth`].
pub(crate) fn conjuncts_with_spine_depth(expr: &Expression) -> (Vec<Expression>, usize) {
    let mut conjuncts = vec![];
    let depth = conjuncts_impl(expr, &mut conjuncts, 0);
    if conjuncts.is_empty() {
        conjuncts.push(expr.clone());
    }
    (conjuncts, depth)
}

/// The `And` nesting depth of the balanced tree [`and_collect`] builds over `n` conjuncts.
///
/// [`and_collect`]: crate::expr::and_collect
pub(crate) fn balanced_spine_depth(n: usize) -> usize {
    n.max(1).next_power_of_two().ilog2() as usize
}

/// Collects the conjuncts below `expr`, returning the deepest `And` nesting reached.
fn conjuncts_impl(expr: &Expression, conjuncts: &mut Vec<Expression>, depth: usize) -> usize {
    if let Some(operator) = expr.as_opt::<Binary>()
        && *operator == Operator::And
    {
        let lhs = conjuncts_impl(expr.child(0), conjuncts, depth + 1);
        let rhs = conjuncts_impl(expr.child(1), conjuncts, depth + 1);
        lhs.max(rhs)
    } else {
        conjuncts.push(expr.clone());
        depth
    }
}
