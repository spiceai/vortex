// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Differential oracle for `list_contains` over a constant list.
//!
//! Every answer below was recorded by running this test against the commit
//! before the set probe was introduced, so the table pins the previous
//! implementation's results rather than the current one's. A change that makes
//! `list_contains` faster leaves it untouched; a change that makes it answer
//! differently fails here.
//!
//! The matrix is built around what the probe covers and what it declines. List
//! lengths straddle the threshold at which membership stops being answered by
//! OR-ing one equality per element, and run past it to the sizes the
//! performance claims rest on. Each covered needle type appears on both sides of
//! that threshold, because the two forms have to agree; the null-in-list rows
//! are what the probe refuses, so they exercise the fallback that has to keep
//! answering them.
//!
//! The values are chosen for the cases where a set keyed on anything but the
//! kernel's own equality would answer differently: `NaN`, which matches itself,
//! the two signed zeros, which do not match each other, strings long enough to
//! live in a buffer rather than inline in their view, and lists holding
//! duplicates.
//!
//! To regenerate: `git stash`, run against the parent commit, paste the
//! printed table into `EXPECTED`, then unstash.

#![expect(clippy::expect_used, clippy::tests_outside_test_module)]

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::list_contains;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;

/// One row per case: the answers `list_contains` gave, as `T` (true), `.`
/// (false) and `?` (invalid).
fn render(case: &str, list: Scalar, needles: ArrayRef) -> String {
    let session = vortex_array::array_session();
    let mut ctx = session.create_execution_ctx();

    let result = needles
        .apply(&list_contains(lit(list), root()))
        .expect("apply list_contains")
        .execute::<BoolArray>(&mut ctx)
        .expect("execute list_contains");

    let bits = result.bit_buffer_view();
    let validity = result
        .validity()
        .expect("validity")
        .execute_mask(bits.len(), &mut ctx)
        .expect("validity mask");
    let answers: String = (0..bits.len())
        .map(|i| match (validity.value(i), bits.value(i)) {
            (false, _) => '?',
            (true, true) => 'T',
            (true, false) => '.',
        })
        .collect();
    format!("{case:34} {answers}")
}

/// Integer needles: values in the list, values absent, and nulls landing on
/// both.
fn int_needles() -> ArrayRef {
    PrimitiveArray::new(
        buffer![0i64, 1, 2, 3, 4, 5, 7, 9, 15, 16, 63, 64, -1, 1_000, 0, 8],
        Validity::from_iter((0..16).map(|i| i % 3 != 0)),
    )
    .into_array()
}

fn int_list(len: usize, nullability: Nullability) -> Scalar {
    Scalar::list(
        Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
        (0..len as i64).map(|v| Scalar::from(v * 2)).collect(),
        nullability,
    )
}

fn main_table() -> String {
    let mut rows = Vec::new();

    // Integer lists across the threshold and up to the sizes the performance
    // claims are measured at.
    for len in [1usize, 2, 3, 4, 5, 8, 64, 2048] {
        for nullability in [Nullability::NonNullable, Nullability::Nullable] {
            let tag = if nullability == Nullability::Nullable {
                "nullable"
            } else {
                "nonnull"
            };
            rows.push(render(
                &format!("i64 m={len} list={tag}"),
                int_list(len, nullability),
                int_needles(),
            ));
        }
    }

    // A null element: not a set key, so the whole list falls back.
    let nullable_i64 = DType::Primitive(PType::I64, Nullability::Nullable);
    let mut with_null: Vec<Scalar> = (0..8i64)
        .map(|v| {
            Scalar::from(v * 2)
                .cast(&nullable_i64)
                .expect("i64 into nullable i64")
        })
        .collect();
    with_null.push(Scalar::null(nullable_i64.clone()));
    rows.push(render(
        "i64 m=9 one-null-element",
        Scalar::list(Arc::new(nullable_i64), with_null, Nullability::Nullable),
        int_needles(),
    ));

    // Floats key on their bits, which is what `Operator::Eq` compares: `NaN`
    // matches itself and `-0.0` does not match `0.0`.
    let float_needles = || {
        PrimitiveArray::new(
            buffer![0.0f64, 1.0, 2.0, f64::NAN, 4.0, -0.0, 8.0, 1e9],
            Validity::AllValid,
        )
        .into_array()
    };
    for len in [1usize, 2, 3, 4, 5, 64] {
        rows.push(render(
            &format!("f64 m={len}"),
            Scalar::list(
                Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)),
                (0..len).map(|v| Scalar::from(v as f64 * 2.0)).collect(),
                Nullability::NonNullable,
            ),
            float_needles(),
        ));
    }
    let mut nan_list: Vec<Scalar> = (0..8).map(|v| Scalar::from(v as f64 * 2.0)).collect();
    nan_list.push(Scalar::from(f64::NAN));
    rows.push(render(
        "f64 m=9 nan-in-list",
        Scalar::list(
            Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)),
            nan_list,
            Nullability::NonNullable,
        ),
        float_needles(),
    ));

    // A signed zero on both sides: the two are distinct values, and each of the
    // needles below is in exactly one of the two lists.
    for (tag, zero) in [("neg", -0.0f64), ("pos", 0.0f64)] {
        rows.push(render(
            &format!("f64 m=4 {tag}-zero-in-list"),
            Scalar::list(
                Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)),
                [zero, 3.0, 5.0, 7.0].map(Scalar::from).to_vec(),
                Nullability::NonNullable,
            ),
            float_needles(),
        ));
    }

    // Strings key on their bytes. The needles are short enough to sit inline in
    // their view; `utf8 long` below uses values too long for that, which resolve
    // through a data buffer instead.
    let string_needles = || {
        VarBinViewArray::from_iter_str(["k000", "k002", "k004", "zzz", "k126", "k010"]).into_array()
    };
    for len in [1usize, 2, 3, 4, 5, 64] {
        rows.push(render(
            &format!("utf8 m={len}"),
            Scalar::list(
                Arc::new(DType::Utf8(Nullability::NonNullable)),
                (0..len)
                    .map(|v| Scalar::from(format!("k{:03}", v * 2)))
                    .collect(),
                Nullability::NonNullable,
            ),
            string_needles(),
        ));
    }

    // Values past the 12 bytes a view holds inline, so the probe has to reach
    // the data buffer for every one of them.
    let long = |v: usize| format!("a string too long to inline {v:04}");
    rows.push(render(
        "utf8 m=8 long",
        Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            (0..8).map(|v| Scalar::from(long(v * 2))).collect(),
            Nullability::NonNullable,
        ),
        VarBinViewArray::from_iter_str([long(0), long(1), long(4), long(99), long(6)]).into_array(),
    ));

    // Nullable needles, including one that is null, against a list of the same
    // strings: a null needle answers false rather than null.
    rows.push(render(
        "utf8 m=4 nullable-needles",
        Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            (0..4)
                .map(|v| Scalar::from(format!("k{:03}", v * 2)))
                .collect(),
            Nullability::NonNullable,
        ),
        VarBinViewArray::from_iter(
            [Some("k000"), None, Some("k004"), None, Some("zzz")],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array(),
    ));

    // Byte strings take the same path as `Utf8`, keyed on the same bytes.
    rows.push(render(
        "binary m=4",
        Scalar::list(
            Arc::new(DType::Binary(Nullability::NonNullable)),
            (0..4u8)
                .map(|v| {
                    Scalar::binary(
                        ByteBuffer::copy_from([v * 2, 0xff]),
                        Nullability::NonNullable,
                    )
                })
                .collect(),
            Nullability::NonNullable,
        ),
        VarBinViewArray::from_iter(
            [
                Some([0u8, 0xff].as_slice()),
                Some(&[2, 0xff]),
                Some(&[1, 0xff]),
                None,
                Some(&[6, 0xff]),
            ],
            DType::Binary(Nullability::Nullable),
        )
        .into_array(),
    ));

    // Integer widths other than the i64 above, including an unsigned one: the
    // set is keyed by the needle's own width, so each has to be built for it.
    rows.push(render(
        "i32 m=8",
        Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            (0..8i32).map(|v| Scalar::from(v * 2)).collect(),
            Nullability::NonNullable,
        ),
        PrimitiveArray::new(
            buffer![0i32, 3, 4, -1, 14, 15],
            Validity::from_iter([true, true, true, false, true, true]),
        )
        .into_array(),
    ));
    rows.push(render(
        "u8 m=8",
        Scalar::list(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            (0..8u8).map(|v| Scalar::from(v * 2)).collect(),
            Nullability::NonNullable,
        ),
        PrimitiveArray::new(buffer![0u8, 3, 4, 255, 14, 15], Validity::AllValid).into_array(),
    ));

    // A list that repeats a value, and one that is empty: both answer exactly
    // what the set they key does.
    rows.push(render(
        "i64 m=8 duplicates",
        Scalar::list(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            [0i64, 0, 2, 2, 2, 4, 4, 4].map(Scalar::from).to_vec(),
            Nullability::NonNullable,
        ),
        int_needles(),
    ));
    rows.push(render(
        "i64 m=0 empty",
        Scalar::list_empty(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            Nullability::NonNullable,
        ),
        int_needles(),
    ));

    rows.join("\n")
}

/// Recorded from the implementation before the set probe. See the module doc.
const EXPECTED: &str = "\
i64 m=1 list=nonnull               ..............T.\n\
i64 m=1 list=nullable              ..............T.\n\
i64 m=2 list=nonnull               ..T...........T.\n\
i64 m=2 list=nullable              ..T...........T.\n\
i64 m=3 list=nonnull               ..T.T.........T.\n\
i64 m=3 list=nullable              ..T.T.........T.\n\
i64 m=4 list=nonnull               ..T.T.........T.\n\
i64 m=4 list=nullable              ..T.T.........T.\n\
i64 m=5 list=nonnull               ..T.T.........T.\n\
i64 m=5 list=nullable              ..T.T.........T.\n\
i64 m=8 list=nonnull               ..T.T.........T.\n\
i64 m=8 list=nullable              ..T.T.........T.\n\
i64 m=64 list=nonnull              ..T.T......T..T.\n\
i64 m=64 list=nullable             ..T.T......T..T.\n\
i64 m=2048 list=nonnull            ..T.T......T.TT.\n\
i64 m=2048 list=nullable           ..T.T......T.TT.\n\
i64 m=9 one-null-element           ..T.T.........T.\n\
f64 m=1                            T.......\n\
f64 m=2                            T.T.....\n\
f64 m=3                            T.T.T...\n\
f64 m=4                            T.T.T...\n\
f64 m=5                            T.T.T.T.\n\
f64 m=64                           T.T.T.T.\n\
f64 m=9 nan-in-list                T.TTT.T.\n\
f64 m=4 neg-zero-in-list           .....T..\n\
f64 m=4 pos-zero-in-list           T.......\n\
utf8 m=1                           T.....\n\
utf8 m=2                           TT....\n\
utf8 m=3                           TTT...\n\
utf8 m=4                           TTT...\n\
utf8 m=5                           TTT...\n\
utf8 m=64                          TTT.TT\n\
utf8 m=8 long                      T.T.T\n\
utf8 m=4 nullable-needles          T.T..\n\
binary m=4                         TT..T\n\
i32 m=8                            T.T.T.\n\
u8 m=8                             T.T.T.\n\
i64 m=8 duplicates                 ..T.T.........T.\n\
i64 m=0 empty                      ................";

#[test]
fn constant_list_contains_answers_are_unchanged() {
    let actual = main_table();
    assert_eq!(
        actual, EXPECTED,
        "\nlist_contains changed an answer. Actual table:\n{actual}\n"
    );
}
