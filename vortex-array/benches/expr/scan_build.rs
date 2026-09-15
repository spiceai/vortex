// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The expression work `ScanBuilder::build` performs once per scan split.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_utils::aliases::dash_map::DashMap;

fn main() {
    divan::main();
}

/// A TPC-H `orders` file dtype: nine columns of mixed types.
fn orders_dtype() -> DType {
    DType::Struct(
        StructFields::new(
            [
                "o_orderkey",
                "o_custkey",
                "o_orderstatus",
                "o_totalprice",
                "o_orderdate",
                "o_orderpriority",
                "o_clerk",
                "o_shippriority",
                "o_comment",
            ]
            .into(),
            vec![
                DType::Primitive(PType::I64, Nullability::Nullable),
                DType::Primitive(PType::I64, Nullability::Nullable),
                DType::Utf8(Nullability::Nullable),
                DType::Primitive(PType::F64, Nullability::Nullable),
                DType::Primitive(PType::I32, Nullability::Nullable),
                DType::Utf8(Nullability::Nullable),
                DType::Utf8(Nullability::Nullable),
                DType::Primitive(PType::I32, Nullability::Nullable),
                DType::Utf8(Nullability::Nullable),
            ],
        ),
        Nullability::NonNullable,
    )
}

/// The projection a `SELECT` of `n` of those columns pushes into the scan.
fn projection(n: usize) -> Expression {
    let DType::Struct(fields, _) = orders_dtype() else {
        unreachable!()
    };
    let children: Vec<(FieldName, Expression)> = fields
        .names()
        .iter()
        .take(n)
        .map(|name| (name.clone(), get_item(name.clone(), root())))
        .collect();
    pack(children, Nullability::NonNullable)
}

/// The filter a point lookup pushes into the scan.
fn point_filter() -> Expression {
    eq(get_item("o_orderkey", root()), lit(1_234_567i64))
}

#[divan::bench(args = [1, 3, 7, 9])]
fn optimize_projection(bencher: Bencher, n: usize) {
    let expr = projection(n);
    let scope = orders_dtype();
    bencher.bench(|| expr.optimize_recursive(&scope).unwrap());
}

#[divan::bench]
fn optimize_point_filter(bencher: Bencher) {
    let expr = point_filter();
    let scope = orders_dtype();
    bencher.bench(|| expr.optimize_recursive(&scope).unwrap());
}

/// What a cache hit costs: hashing the whole expression, comparing it, cloning both.
#[divan::bench(args = [1, 3, 7, 9])]
fn cache_hit_projection(bencher: Bencher, n: usize) {
    let expr = projection(n);
    let scope = orders_dtype();
    let map: DashMap<(Expression, DType), Expression> = DashMap::default();
    map.insert(
        (expr.clone(), scope.clone()),
        expr.optimize_recursive(&scope).unwrap(),
    );
    bencher.bench(|| {
        let hit = map.get(&(expr.clone(), scope.clone())).unwrap();
        hit.clone()
    });
}

/// What hashing the key alone costs, with no map involved.
#[divan::bench(args = [1, 7])]
fn hash_projection_key(bencher: Bencher, n: usize) {
    use std::hash::BuildHasher;
    let expr = projection(n);
    let scope = orders_dtype();
    let hasher = std::collections::hash_map::RandomState::new();
    bencher.bench(|| hasher.hash_one((&expr, &scope)));
}
