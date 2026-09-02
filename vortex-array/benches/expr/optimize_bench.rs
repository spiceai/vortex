// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::and_collect;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::or;
use vortex_array::expr::root;

fn main() {
    divan::main();
}

fn struct_scope() -> DType {
    DType::Struct(
        StructFields::new(
            ["x"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
        ),
        Nullability::NonNullable,
    )
}

fn build_or_chain(n: usize) -> Expression {
    let base = eq(get_item("x", root()), lit(0i32));
    (1..n).fold(base, |acc, i| {
        or(acc, eq(get_item("x", root()), lit(i as i32)))
    })
}

#[divan::bench(args = [200])]
fn optimize_or_chain(bencher: Bencher, n: usize) {
    let expr = build_or_chain(n);
    let scope = struct_scope();
    bencher.bench(|| expr.optimize_recursive(&scope).unwrap());
}

/// An `AND` chain of same-direction bounds, so no pair can ever form a `between`. This is the
/// shape of most multi-predicate filters, and the case where a rewrite pass that always rebuilds
/// the conjunct spine does the most wasted work.
fn build_and_chain(n: usize) -> Expression {
    and_collect((0..n).map(|i| gt(get_item("x", root()), lit(i as i32)))).unwrap()
}

#[divan::bench(args = [32, 200])]
fn optimize_and_chain(bencher: Bencher, n: usize) {
    let expr = build_and_chain(n);
    let scope = struct_scope();
    bencher.bench(|| expr.optimize_recursive(&scope).unwrap());
}
