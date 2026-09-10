// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `list_contains` falsifier must only ever claim that a zone holds no
//! matching row.
//!
//! Deriving the predicate is cheap to check on its own, and the unit tests
//! beside the falsifier do. What they cannot see is whether that predicate,
//! once evaluated against a zone's statistics, excludes a zone that does hold a
//! matching row — which is how an `IN` query comes back missing rows. So this
//! checks the property directly: every case states its zones, derives the
//! statistics from them, prunes, and requires that each excluded zone contains
//! zero matches.
//!
//! The falsifier describes the list as intervals rather than as one equality
//! per element, so it depends on the sorted order of the list agreeing with the
//! order the statistics comparison uses. These cases are built around the
//! places that could disagree: floats (where the order is total, so `NaN` and
//! the two signed zeros have positions), truncated string bounds (which widen a
//! zone rather than narrow it), duplicate elements, and the list lengths either
//! side of the cap on how many interior gaps are emitted.

#![expect(clippy::expect_used, clippy::panic, clippy::tests_outside_test_module)]

use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::list_contains;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_io::session::RuntimeSession;
use vortex_layout::layouts::zoned::zone_map::ZoneMap;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
});

/// Rows per zone. Every case states its zones as equal-length groups so that
/// the nominal zone length and the row count agree.
const ZONE_LEN: usize = 4;

/// Prunes `stats` with the falsifier derived for `list`, and fails if a zone it
/// excluded holds a matching row.
///
/// Returns how many zones were excluded, so a caller can also require that a
/// case prunes at all — a case that prunes nothing satisfies the soundness
/// property without having tested it.
fn zones_pruned(
    case: &str,
    column_dtype: &DType,
    stats: Vec<(AggregateFnRef, ArrayRef)>,
    list: Scalar,
    zone_has_match: &[bool],
) -> usize {
    let zone_count = zone_has_match.len();
    let fields = stats
        .iter()
        .map(|(agg, values)| (agg.to_string(), values.clone()))
        .collect::<Vec<_>>();
    let zone_map = ZoneMap::try_new(
        column_dtype.clone(),
        StructArray::from_fields(&fields).expect("stats table"),
        stats.iter().map(|(agg, _)| agg.clone()).collect(),
        ZONE_LEN as u64,
        (zone_count * ZONE_LEN) as u64,
    )
    .expect("zone map");

    let predicate = list_contains(lit(list), root())
        .falsify(column_dtype, &SESSION)
        .expect("falsify")
        .unwrap_or_else(|| panic!("{case}: no falsifier derived"));

    let mask = zone_map.prune(&predicate, &SESSION).expect("prune");

    let mut pruned = 0;
    for zone in 0..zone_count {
        if !mask.value(zone) {
            continue;
        }
        pruned += 1;
        assert!(
            !zone_has_match[zone],
            "{case}: zone {zone} was pruned but holds a matching row",
        );
    }
    pruned
}

/// Statistics for primitive zones, as the writer computes them: the minimum and
/// maximum under the total order skipping NaN, plus the NaN count that a float
/// column's bounds are only usable alongside.
fn primitive_stats<T: NativePType>(zones: &[Vec<T>]) -> Vec<(AggregateFnRef, ArrayRef)> {
    let mut mins = Vec::with_capacity(zones.len());
    let mut maxs = Vec::with_capacity(zones.len());
    let mut nan_counts = Vec::with_capacity(zones.len());
    for zone in zones {
        let non_nan = || zone.iter().copied().filter(|v| !v.is_nan());
        // Every case keeps at least one ordinary value in each zone, so the
        // bounds are always defined.
        mins.push(
            non_nan()
                .min_by(|a, b| a.total_compare(*b))
                .expect("zone has a non-NaN value"),
        );
        maxs.push(
            non_nan()
                .max_by(|a, b| a.total_compare(*b))
                .expect("zone has a non-NaN value"),
        );
        nan_counts.push(zone.iter().filter(|v| v.is_nan()).count() as u64);
    }

    let mut stats = vec![
        (
            Min.bind(NumericalAggregateOpts::skip_nans()),
            PrimitiveArray::new(Buffer::<T>::copy_from(&mins), Validity::AllValid).into_array(),
        ),
        (
            Max.bind(NumericalAggregateOpts::skip_nans()),
            PrimitiveArray::new(Buffer::<T>::copy_from(&maxs), Validity::AllValid).into_array(),
        ),
    ];
    // A float column's bounds are only usable alongside its NaN count, and only
    // a float column has one to store.
    if T::PTYPE.is_float() {
        stats.push((
            NanCount.bind(EmptyOptions),
            PrimitiveArray::new(Buffer::<u64>::copy_from(&nan_counts), Validity::AllValid)
                .into_array(),
        ));
    }
    stats
}

/// Statistics for string zones, taken verbatim so a case can state bounds that
/// are wider than the zone's own values — which is what the writer stores when
/// it truncates a long bound.
fn utf8_stats(mins: &[&str], maxs: &[&str]) -> Vec<(AggregateFnRef, ArrayRef)> {
    let column = |values: &[&str]| {
        VarBinViewArray::from_iter(
            values.iter().map(|value| Some(*value)),
            DType::Utf8(Nullability::Nullable),
        )
        .into_array()
    };
    vec![
        (Min.bind(NumericalAggregateOpts::skip_nans()), column(mins)),
        (Max.bind(NumericalAggregateOpts::skip_nans()), column(maxs)),
    ]
}

fn int_list(values: &[i64]) -> Scalar {
    Scalar::list(
        Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
        values.iter().copied().map(Scalar::from).collect(),
        Nullability::NonNullable,
    )
}

fn float_list(values: &[f64]) -> Scalar {
    Scalar::list(
        Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)),
        values.iter().copied().map(Scalar::from).collect(),
        Nullability::NonNullable,
    )
}

fn utf8_list(values: &[&str]) -> Scalar {
    Scalar::list(
        Arc::new(DType::Utf8(Nullability::NonNullable)),
        values
            .iter()
            .map(|v| Scalar::utf8(*v, Nullability::NonNullable))
            .collect(),
        Nullability::NonNullable,
    )
}

/// Whether each zone holds a value of `list`, under integer equality.
fn int_matches(zones: &[Vec<i64>], list: &[i64]) -> Vec<bool> {
    zones
        .iter()
        .map(|zone| zone.iter().any(|v| list.contains(v)))
        .collect()
}

#[test]
fn a_zone_inside_a_gap_is_pruned_and_one_holding_a_value_is_not() {
    // The list is two clusters far apart, which is the shape the interior gaps
    // exist for: every zone lies inside the list's overall range, so the outer
    // bounds alone prove nothing.
    let list: Vec<i64> = (0..8).chain(1_000..1_008).collect();
    let zones: Vec<Vec<i64>> = vec![
        vec![2, 3, 4, 5],           // holds list values
        vec![100, 200, 300, 400],   // wholly inside the gap
        vec![900, 950, 999, 1_000], // straddles the gap's upper end, holds 1000
        vec![1_004, 1_005, 1_006, 1_007],
        vec![20, 30, 40, 50], // inside the gap, nowhere near a value
    ];
    let matches = int_matches(&zones, &list);
    assert_eq!(matches, vec![true, false, true, true, false]);

    let pruned = zones_pruned(
        "two clusters",
        &DType::Primitive(PType::I64, Nullability::NonNullable),
        primitive_stats(&zones),
        int_list(&list),
        &matches,
    );
    assert_eq!(pruned, 2, "both gap-interior zones should prune");
}

#[test]
fn duplicate_elements_do_not_open_a_gap() {
    // A duplicated value has nothing between it and itself. Treating the pair
    // as a gap would exclude every zone whose bounds sit on that value, which
    // is exactly the zone that matches.
    let list = vec![5i64, 5, 5, 5, 5, 5, 5, 5];
    let zones: Vec<Vec<i64>> = vec![vec![5, 5, 5, 5], vec![1, 2, 3, 4], vec![6, 7, 8, 9]];
    let matches = int_matches(&zones, &list);
    assert_eq!(matches, vec![true, false, false]);

    let pruned = zones_pruned(
        "all duplicates",
        &DType::Primitive(PType::I64, Nullability::NonNullable),
        primitive_stats(&zones),
        int_list(&list),
        &matches,
    );
    assert_eq!(pruned, 2, "the zones either side of the value should prune");
}

#[test]
fn every_list_length_around_the_gap_cap_stays_sound() {
    // Only so many interior gaps are emitted, and which boundaries they fall on
    // changes with the list length. Sweeping the lengths either side of that cap
    // covers the change of regime; the values are spread so that most zones sit
    // inside some gap and pruning is actually exercised.
    let mut total_pruned = 0;
    for len in [1usize, 2, 3, 4, 5, 30, 31, 32, 33, 34, 35, 64, 129] {
        let list: Vec<i64> = (0..len as i64).map(|i| i * 100).collect();
        let zones: Vec<Vec<i64>> = (0..16)
            .map(|z| {
                let base = z * 61;
                vec![base, base + 7, base + 13, base + 29]
            })
            .collect();
        let matches = int_matches(&zones, &list);
        total_pruned += zones_pruned(
            &format!("m={len}"),
            &DType::Primitive(PType::I64, Nullability::NonNullable),
            primitive_stats(&zones),
            int_list(&list),
            &matches,
        );
    }
    assert!(
        total_pruned > 0,
        "the sweep pruned nothing, so proved nothing"
    );
}

#[test]
fn floats_order_nan_and_signed_zero_the_way_the_statistics_do() {
    // Float comparison here is total: `NaN` equals itself and sorts above every
    // number, and `-0.0` is distinct from and below `0.0`. The falsifier sorts
    // the list with the same order the statistics comparison uses, so these
    // three values have positions rather than being unordered — this is the case
    // that goes wrong if the two orders ever disagree.
    let list = vec![-1.0f64, -0.0, 0.0, 1.0, 100.0, f64::NAN];
    let zones: Vec<Vec<f64>> = vec![
        vec![-0.0, 0.0, 0.5, 0.75],          // holds both zeros
        vec![10.0, 20.0, 30.0, 40.0],        // inside the gap between 1 and 100
        vec![1.0, 2.0, 3.0, 4.0],            // holds 1.0
        vec![f64::NAN, 200.0, 300.0, 400.0], // holds NaN, which is in the list
        vec![-100.0, -50.0, -20.0, -10.0],   // below the list's lowest value
    ];
    let matches: Vec<bool> = zones
        .iter()
        .map(|zone| {
            zone.iter()
                .any(|v| list.iter().any(|l| l.to_bits() == v.to_bits()))
        })
        .collect();
    assert_eq!(matches, vec![true, false, true, true, false]);

    let pruned = zones_pruned(
        "floats with NaN and signed zero",
        &DType::Primitive(PType::F64, Nullability::NonNullable),
        primitive_stats(&zones),
        float_list(&list),
        &matches,
    );
    assert!(pruned > 0, "no float zone pruned, so nothing was tested");
}

#[test]
fn truncated_string_bounds_cannot_create_a_false_exclusion() {
    // A zone whose extreme value is too long to store is written with a bound
    // that is looser than the truth: a minimum below the real minimum and a
    // maximum above the real maximum. That widens a zone, which can only stop it
    // being pruned, never cause it to be pruned wrongly — this states the case
    // in which it would, if the direction were ever reversed.
    let list = ["aa", "abzz", "ac", "zz"];
    // Zone 0 holds "abzz", stored truncated to two bytes as min "ab", max "ac".
    // Zone 1 holds only values inside the gap between "ac" and "zz".
    // Zone 2 holds "zz" with an exact bound.
    let mins = ["ab", "b", "zz"];
    let maxs = ["ac", "c", "zz"];
    let matches = vec![true, false, true];

    let pruned = zones_pruned(
        "truncated utf8 bounds",
        &DType::Utf8(Nullability::NonNullable),
        utf8_stats(&mins, &maxs),
        utf8_list(&list),
        &matches,
    );
    assert_eq!(pruned, 1, "only the gap-interior zone should prune");
}

#[test]
fn exhaustive_over_a_small_domain() {
    // Every list of up to five values from a small domain, against every
    // partition of that domain into three zones. This is the case the
    // hand-written ones above cannot cover: it makes no assumption about which
    // shape of list or zone is interesting, so a boundary that only misbehaves
    // for one arrangement still has to show up.
    const DOMAIN: i64 = 7;
    let mut checked = 0usize;
    let mut pruned_total = 0usize;

    for mask in 1u32..(1 << DOMAIN) {
        let list: Vec<i64> = (0..DOMAIN).filter(|i| mask & (1 << i) != 0).collect();
        if list.len() > 5 {
            continue;
        }
        for shift in 0..DOMAIN {
            let zones: Vec<Vec<i64>> = (0..3)
                .map(|z| {
                    (0..ZONE_LEN as i64)
                        .map(|r| (shift + z * 2 + r) % DOMAIN)
                        .collect()
                })
                .collect();
            let matches = int_matches(&zones, &list);
            pruned_total += zones_pruned(
                &format!("list={list:?} shift={shift}"),
                &DType::Primitive(PType::I64, Nullability::NonNullable),
                primitive_stats(&zones),
                int_list(&list),
                &matches,
            );
            checked += 1;
        }
    }

    assert!(checked > 500, "expected a wide sweep, ran {checked} cases");
    assert!(
        pruned_total > 0,
        "the sweep pruned nothing, so proved nothing"
    );
}

#[test]
fn a_run_of_duplicates_does_not_consume_every_boundary() {
    // Only so many boundaries are emitted, and they divide the sorted list
    // evenly. A list that is mostly one repeated value has exactly one real gap;
    // choosing boundaries before collapsing the duplicates spends all of them
    // inside the run and misses it, which prunes nothing at all.
    let mut list = vec![0i64; 32];
    list.push(1_000);
    let zones: Vec<Vec<i64>> = vec![
        vec![500, 501, 502, 503],   // wholly inside the only gap
        vec![0, 1, 2, 3],           // holds 0
        vec![997, 998, 999, 1_000], // holds 1000
    ];
    let matches = int_matches(&zones, &list);
    assert_eq!(matches, vec![false, true, true]);

    let pruned = zones_pruned(
        "a run of duplicates",
        &DType::Primitive(PType::I64, Nullability::NonNullable),
        primitive_stats(&zones),
        int_list(&list),
        &matches,
    );
    assert_eq!(pruned, 1, "the zone inside the only gap should prune");
}

#[test]
fn a_null_element_does_not_cost_the_whole_list_its_pruning() {
    // A null element cannot make `list_contains` true, so it is not a value a
    // zone has to be checked against. Abandoning the list because one is present
    // would give an ordinary parameterized `IN` list no pruning whatsoever.
    let nullable = DType::Primitive(PType::I64, Nullability::Nullable);
    let mut elements: Vec<Scalar> = [0i64, 1, 2, 1_000]
        .iter()
        .map(|v| {
            Scalar::from(*v)
                .cast(&nullable)
                .expect("i64 into nullable i64")
        })
        .collect();
    elements.push(Scalar::null(nullable.clone()));
    let list = Scalar::list(Arc::new(nullable), elements, Nullability::Nullable);

    let zones: Vec<Vec<i64>> = vec![
        vec![500, 501, 502, 503],   // inside the gap between 2 and 1000
        vec![0, 1, 2, 3],           // holds list values
        vec![997, 998, 999, 1_000], // holds 1000
    ];
    let matches = int_matches(&zones, &[0, 1, 2, 1_000]);
    assert_eq!(matches, vec![false, true, true]);

    let pruned = zones_pruned(
        "a null element",
        &DType::Primitive(PType::I64, Nullability::NonNullable),
        primitive_stats(&zones),
        list,
        &matches,
    );
    assert_eq!(pruned, 1, "the gap-interior zone should still prune");
}
