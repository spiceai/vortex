# Spice patches carried on this fork

Spice-specific changes to Vortex exist only as commits on this fork's branches. When the
fork is re-cut for a new upstream version (`spiceai-52` → `53` → `54` → …), any patch that
is not deliberately carried forward is lost silently: nothing fails, the code reverts to
upstream behaviour, and the bug returns in the next release.

That has already happened, twice — see `spiceai-54`'s history for the reentrant-waker and
Arrow `Map` incidents. This file is the list that upgrade work checks against.

## This branch: `spiceai-54-vortex-0.85.0`

Re-cut from upstream tag `0.85.0` (`afb005379dd9a1b7dc7ae2cb49f4f465d91cc2b3`), replacing
`spiceai-54` (based on `0.79.0`). Still DataFusion 54 / Arrow 58.3 — this is a Vortex-only
pre-upgrade ahead of the DataFusion 55 bump (spiceai/spiceai#13570), landed independently so
it can be validated against the existing DF54 baseline first.

**Every row below was re-verified against `0.85.0` by diffing the patched files and/or
attempting a real `git cherry-pick`, not by title match.** Where a cherry-pick is cited as
"applied cleanly", that means the mechanical merge produced no conflicts; it does not by
itself mean the behavior was re-tested — see each row's Verify command for what was actually
run.

## Convention

Follow what the DataFusion fork does:

- Create a **`spiceai-<version>-patches` branch** from each new version branch.
- Cherry-pick every patch below as an **individual commit — never squashed or batched**, so
  the next upgrade can enumerate exactly what needs porting.
- Perform a **patch audit** at every upgrade: for each patch, is it still needed, or did it
  land upstream?

(As before, `spiceai-54` never got a separate `-patches` branch — patches landed directly on
it. This re-cut carries that same shape for now: patches are directly on
`spiceai-54-vortex-0.85.0`.)

## Applied to this branch

| # | Patch | New commit | Old origin | Verify |
|---|---|---|---|---|
| A1 | Flat reader subrange decode reuse (untracked by the previous ledger — a real fix, not noticed until this audit) | `dafa0960a` | `e09ec32c4` | Compiles; no dedicated test found. **GAP** — add one if this is load-bearing. |
| 13a | Let an embedder declare available parallelism | `f8587c4c6` | `579c9962d` | `cargo test -p vortex-utils --test parallelism_declared --test parallelism_declared_too_late` — **ran, both pass** |
| 13b | Give a failed parallelism detection its own message, stop allocating in `Display` | `dbf3a4d46` | `a70767284` | same as above |
| 13c | Tighten the parallelism comments to behavior | `fc3ee4c22` | `92284354f` | same as above |

Row 13 (`set_available_parallelism`, spiceai/spiceai#12328) is fully carried forward and
behaviorally verified on this branch.

Note: all three cherry-picks initially conflicted **only** on `SPICE_PATCHES.md` itself
(this file didn't exist at `0.85.0`) — that conflict was resolved by dropping the old file
content in favor of this rewrite, not by a code-level resolution. The actual code
(`vortex-utils/src/parallelism.rs` and its tests) applied with zero conflicts each time.

## Dropped — confirmed upstreamed or out of scope

| # | Patch | Old origin | Evidence |
|---|---|---|---|
| 1 | Tokio one-shot in `single.rs` | `b27e89af5` (#9) | Upstream PR vortex-data/vortex#9360 ("replace `faern::oneshot` with `futures::oneshot`", merged 2026-08-18, ancestor of `0.85.0`) touches this exact file. **The upstream issue this PR closes (vortex-data/vortex#6221) is Spice's own bug report** — it links `spiceai/spiceai#8830` and this fork's PR #9. `0.85.0:vortex-io/src/runtime/single.rs` now uses `futures::channel::oneshot`. |
| 2 | Tokio one-shot for the spawned `Task` result channel | `53b08c10c` (#82) | Same PR #9360, same file family (`vortex-io/src/runtime/handle.rs`). |
| 3 | Tokio one-shot for the segment-read result channel | (this branch, part of #82) | Same PR #9360 (`vortex-file/src/segments/source.rs`). |
| 8 (old #12) | N-ary `CASE WHEN` expression | `4bfa4331b` | `0.85.0:vortex-array/src/scalar_fn/fns/case_when.rs` already has `num_when_then_pairs: u32` with the same n-ary arity computation. |
| 9 | Pushdown TRUE-bubbling / empty `IN` list | `8044a8470` (#8) | Both touched files are in `vortex-datafusion`, which Spice vendors directly into `crates/vortex` in the main repo (`vortex-datafusion = { path = "crates/vortex" }`) and does not take from this fork at all. Not fork state. |
| 10 (partial) | `UncompressedSizeInBytes` in `vortex-datafusion/src/persistent/format.rs` | `6712e9ffa` (#3) | Same as row 9 — vendored, out of scope for this fork. |
| 5 | Balanced `list_contains` OR tree | `d694abda6` (#37) | The function this commit added (`or_arrays_balanced`/`array_depth`) no longer exists anywhere in `spiceai-54`'s own later history — superseded internally by the untracked Sept-11 series (see below), which itself converges with `0.85.0`'s own `vortex_utils::iter::ReduceBalancedIterExt`. Dead code; nothing to carry. |
| — | `ea3890528` "duckdb 1.5.5 (#8895)" | — | Not a Spice patch: identical to upstream's own `0.80.0` release commit (`037b1a8778db`), already superseded by landing on `0.85.0`. |
| — | `68badd689` "TableStrategy hides panic... (#8672)" | — | Not a Spice patch: `(cherry picked from commit 4abe3d04f...)`, authored by a Vortex/spiraldb maintainer, cherry-picked onto `spiceai-54` ahead of a release. Already upstream. |
| — | `b5498e8e1` "get CI green" (#81), `8d2efe7a7` style import sort | — | CI/lint housekeeping for subsystems Spice doesn't build (java/, python/, ffi/) or trivial reformatting of files already changed upstream. Not carried; not fork state Spice depends on. |
| — | `ab4f0b177` "restore load_full + HashMap import dropped in rebase" | — | `vortex-array/src/arc_swap_map.rs` doesn't exist at `0.85.0` at all (not moved, gone), and `grep -rn "load_full"` across `vortex-array/src` finds zero matches. Spice's own fix for a Spice-only prior rebase mistake in a file structure that no longer exists. Nothing to port. |
| 6 | Avoid session lock re-entry in writer init | `c536c9aed` (#29) | The hazard was calling the `SessionGuard`-returning `session.arrays()` twice within one statement (`ArrayContext::new(self.session.arrays()...).with_registry(self.session.arrays()...)`), so the two guards overlapped for the whole statement instead of the first being dropped before the second was taken. `vortex-file/src/writer.rs`'s array-context construction was rewritten into `new_array_context`, which calls the lock-free, `Vec`-returning `session.enabled_component_ids(...)` exactly once and reuses the result — the guard-returning call this row exists to guard against isn't made here at all anymore. Swept every `.arrays()` call site across `vortex-file`, `vortex-array`, `vortex-layout`, `vortex-io`, `vortex-scan`, `vortex-btrblocks`, `vortex-utils`, `vortex-session` (`grep -rn "session\.arrays()"`, 10 hits): every one either calls it once or binds the guard to a variable and reuses that variable, never re-invoking the accessor within a live guard's scope. No code change; the hazard's precondition doesn't exist in the crates Spice consumes. Still a **GAP** in the sense that no regression test pins this — if a future edit reintroduces a double call, nothing here would catch it. |
| 14 | Arrow Map alias — `DType` alias + map-entry recursion in the session importer | `840e746a9`, `3e7fa40d3` (superseded the stale `1a6dc54f1` origin, then this) | **Superseded by a full upstream rewrite, not just a fix.** Upstream 0.85.0 replaced the `List<Struct<key,value>>` alias both Spice's patch and the pre-split `vortex-array/src/arrow` code used with a **native `Map` array/dtype**: `vortex-array/src/dtype/map.rs` (`MapDType`) and `vortex-array/src/arrays/map/` (`Array<Map>`, backed by a `ListView<Struct<key,value>>`). Both `vortex-arrow/src/convert.rs` (import) and `vortex-arrow/src/executor/mod.rs` (export) dispatch `DataType::Map` to this native path, and `vortex-arrow/src/session.rs` recurses into map key/value fields for extension-type metadata exactly as Spice's `3e7fa40d3` did. **Ran `cargo test -p vortex-arrow --lib map` on this branch: 17 tests pass**, including `executor::map::tests::map_roundtrip_preserves_nested_uuid_fields` (the exact extension-recursion scenario `3e7fa40d3` fixed) and `session::tests::schema_roundtrip_preserves_map_uuid_fields`. Upstream's own coverage here is broader than Spice's two patches were. Attempting to cherry-pick the old alias-based patch onto this would be a regression, not a port — the P0 defect (spiceai/spiceai#13524) is closed by a better mechanism than the one Spice shipped. **Remaining gap**: this proves the fix at the `vortex-arrow` crate level; the Spice main repo's own guard (`crates/vortex/src/persistent/mod.rs::map_column_roundtrips_through_a_vortex_file`, per `docs/dev/fork_patches.md`) still needs re-running once this branch is pinned, since that test exercises the full write path through the vendored `vortex-datafusion` sink, which this fork does not build. |

## Needs re-porting (real conflicts against `0.85.0` — deferred, not dropped)

Each of these was attempted as a `git cherry-pick` against this branch and produced a real
content conflict (not just the spurious `SPICE_PATCHES.md` one above). Landing them requires
reading the new upstream code and re-implementing the fix, not resolving conflict markers.
**Do not consider any of these dropped** — they are open work.

| # | Patch | Old origin | What changed upstream |
|---|---|---|---|
| 11 | Intra-file decode parallelism | `26b274c72` (#62) | Real conflicts in `vortex-file/Cargo.toml`, `vortex-file/src/tests.rs`; `vortex-layout/src/scan/split_by.rs` also changed upstream (small diff, not yet compared line-by-line). Perf-only; still a ledger **GAP** (no guard). |
| — | Cache `ArrayKernels` as a per-`ExecutionCtx` snapshot | `ccaa55627` | Real conflict in `vortex-array/src/executor.rs`. Adds a `KernelSnapshot` type and `ArrayKernels::snapshot()` (confirmed absent at `0.85.0`) into the hot `execute_until` path on the central `ExecutionCtx` struct — higher blast radius than the other items here if ported carelessly. Perf-only. Not previously in the ledger. |
| — | Restore lint checks on forks | `bb80c537b` | Conflicts across `.github/workflows/ci.yml`, `cast.rs`, `vortex-datafusion/src/persistent/sink.rs` (the last is out of scope). Low priority — CI config for this fork, not behavior Spice depends on. |

## Applied — re-ported as fresh implementations (not clean cherry-picks)

These landed as new commits against `0.85.0` rather than cherry-picks, since the code they touch was restructured. Each is behaviorally verified, not just compiled.

| # | Patch | New commit | Old origin | Verify |
|---|---|---|---|---|
| 4 | Fixed-offset timezone resolution (`resolve_timezone`, threaded through scalar validation, `Display`, and `TemporalMetadata::to_jiff`; `Scalar::try_extension_ref` through the 4 sites that built extension scalars infallibly on a read path) | `dc9892190` | `6cdea73d6` (#75), `5b4bee108` (#78) | `cargo test -p vortex-array --lib extension::datetime::timezone extension::datetime::matcher` — **ran, 42 pass** |
| 7, 15, 16 | `vortex.date`→`vortex.timestamp` array + scalar casts via a shared `DateToTimestamp`, `Timestamp::storage_range` replacing Jiff-`Span`-based validation | `dc9892190` (supersedes this branch's earlier `a98a78c1b`, which used a `legacy_session()` workaround instead of the proper `CastKernel` split found in this row's own later Spice history) | `7e5b08151` (#28), this branch's `3c5246867` (#93) | `cargo test -p vortex-array --lib arrays::extension::compute::cast scalar::typed_view::extension` — **ran, all pass**; full crate: `cargo test -p vortex-array -p vortex-datetime-parts --lib` — **3428 + 72 pass, 0 failed** |
| 10 (real fix) | `UncompressedSizeInBytes` propagation through `vortex-array/src/arrays/dict/take.rs`, and inclusion in `stats::PRUNING_STATS` | `1672b7328` | `a9ef29dea` (previously **not in the ledger at all** — the old ledger cited only the vendored, out-of-scope `6712e9ffa`) | New tests added (none existed before): `cargo test -p vortex-array --lib arrays::dict::take` — **ran, 2 pass** (`uncompressed_size_in_bytes_propagates_through_take_as_inexact`, `is_constant_propagates_inexactly_through_take_with_a_null_index`); full crate `cargo test -p vortex-array --lib` — **3430 pass, 0 failed**. Not subsumed by the `AggregateFnVTable` redesign — that changes how the stat is computed per array type, not whether `PRUNING_STATS` includes it or `take` propagates it. |

Not ported: the DuckDB `TIMESTAMP_TZ` dtype/scalar `is_utc_timezone` gate (`vortex-duckdb/src/convert/{dtype,scalar}.rs`) — `vortex-duckdb` is not a crate Spice's `Cargo.toml` consumes, directly or transitively. Fork hygiene, not something Spice's re-pin needs; left open.

## Untracked in the previous ledger — needs its own audit

Six commits landed on `spiceai-54` on 2026-09-11 (author Ben Chambers) that were never added
to this file, so this re-cut is also the first time they've been evaluated:

`6c9ffc507`, `2f1a22ada`, `af1c5b301`, `f73241661`, `95d40c8bb`, `aff66352b` — a rewrite of
`IN`-list / `list_contains` handling, overlapping row 5's file
(`vortex-array/src/scalar_fn/fns/list_contains/mod.rs`) and, per the audit, written against a
`list_contains` dispatch shape `0.85.0` has since restructured around a generic
`process_matches::<O, S>` with offset reinterpret-casting. `0.85.0` independently converged on
the same `vortex_utils::iter::ReduceBalancedIterExt` utility this series uses, which is *why*
row 5's original patch is dead code.

**Re-read in full and re-scoped**: this series is Spice's own *performance* work, not a fix
for a pre-existing upstream bug — every commit in it is `perf(...)` except one, and that one
(`af1c5b301`, "decline a null-bearing extension list instead of panicking") fixes a panic in
an extension-type `list_contains` kernel that **`2f1a22ada`, three commits later the same day,
deletes outright** ("removes the `Scalar::new_unchecked` that retagged each element... Moving
[extension handling] there drops the per-batch cost to nothing"). By the end of the series,
Spice's own code no longer has an extension-specific `list_contains` kernel at all — matching
`0.85.0`, which never had one either (`grep` finds no `extension` reference anywhere in
`vortex-array/src/scalar_fn/fns/list_contains/mod.rs`). The panic's precondition — a
Spice-only kernel calling `Scalar::new_unchecked` on a null extension element — exists on
neither `spiceai-54`'s current tip nor `0.85.0`. **Not independently reproduced against
0.85.0** (no repro attempted; this is a reading of the commit sequence, not a run) — call this
unverified rather than confirmed, but it is not a live-bug row regardless: at worst it is
moot, not open.

What the series actually adds beyond that: a hash-set membership probe for constant `IN`
lists instead of one `Eq` array + OR per element, an interval-based statistics-pruning
rewrite (a single top-level `OR` of gaps, replacing an `AND`-of-equalities that the optimizer
compared pairwise — quadratic in list length), an FSST-level `ListContainsElementKernel` that
answers `IN` over compressed strings without decompressing, and a word-at-a-time null-mask
probe. All four are `perf(...)`-labeled, real, and substantial (the largest single commit
here, `f73241661`, is also the one with the most test evidence: 39 recorded answers in
`vortex-array/tests/in_list_differential.rs`, described as failing under 5 of 6 mutations to
the pruning logic in `vortex-layout/tests/list_contains_pruning.rs` — neither test file exists
at `0.85.0`). Re-implementing this is a **feature port against a since-rewritten 944-line
file**, not a patch cherry-pick — deprioritized below the smaller items in this ledger for
that reason, not because it lacks value. Budget a dedicated session that reads
`process_matches::<O, S>` first.

Also unresolved from the previous audit and not re-checked here: `4e2d62654` (#87, "Absolute
split concurrency, stop deriving concurrency defaults from host parallelism") — thematically
related to row 13 but touches a different file set (`vortex-file/src/read/{driver,request}.rs`,
`vortex-file/src/segments/source.rs`, `vortex-layout/src/scan/*`) and conflicts for real at
`0.85.0`. Needs its own ledger row once re-ported.

## Known remaining exposure

The `oneshot` crate (or its `futures::channel::oneshot` upstream replacement) hazard —
a receiver polled and then dropped while its sender completes — was never confirmed closed
everywhere by this audit. Before trusting rows 1-3 as fully dropped, run the ledger's own
mechanical guard on this branch:

```
cargo test -p vortex-io --features tokio --test cancel_stress
```

This was **not run** in this pass (no `cancel_stress` test target found at `0.85.0` under
that invocation as of this writing — confirm the test still exists under the upstream
replacement before treating rows 1-3 as closed).

## Upgrade checklist

1. Carry this file forward first, then create the next version's branch from it.
2. Cherry-pick every tracked patch as an individual commit. Never squash.
3. Run each Verify command against the new branch. A missing patch must be a recorded
   decision, not an omission.
4. For each patch, check whether it landed upstream in the new release; if so, drop it and
   note that here, with the upstream commit/PR as evidence — not a title guess.
5. Update the table: new SHAs, new upstream status.

## Keeping the list honest

- A patch is tracked only once it has a `Verify` command that matches code rather than
  prose. Comments survive a bad re-cut; code does not.
- When adding a patch to this fork, add its row here in the same change.
