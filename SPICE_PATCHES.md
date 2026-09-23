# Spice patches carried on this fork

Spice-specific changes to Vortex exist only as commits on this fork's branches. When the
fork is re-cut for a new upstream version (`spiceai-52` → `53` → `54` → …), any patch that
is not deliberately carried forward is lost silently: nothing fails, the code reverts to
upstream behaviour, and the bug returns in the next release.

That has already happened. A reentrant-waker-drop use-after-free was fixed in
`vortex-io/src/runtime/single.rs` in January 2026 (#9) and **was** carried forward
correctly. The identical hazard in `vortex-io/src/runtime/handle.rs` was found three more
times (2026-07-13, 07-19, 07-21), each time fixed on a branch that was never merged, so it
reached **no** shipping branch and produced `SIGSEGV` crashes under ordinary task
cancellation in a released build.

It has happened twice. Arrow `Map` support (row 14) shipped on `spiceai-51`, `-52` and
`-53` and is absent from `-54`: `vortex-array/src/arrow/executor/map.rs` was dropped by the
merge that split `vortex-array/src/arrow` into the `vortex-arrow` crate, and no commit
records the deletion. Half of the patch — the `DType` alias — survived, which is why the
loss surfaced not as a build failure but as a runtime write error on a released build.

This file is the list that upgrade work checks against.

## This branch: `spiceai-55` (upstream `0.86.1`, DataFusion 55.1.0 / Arrow 59.2)

Built by merging upstream tag `0.86.1` into `spiceai-54` (`git branch spiceai-55
origin/spiceai-0.86`, tip `3a4c695c0`) rather than a fresh cherry-pick re-cut — the merge
mechanically carries every `spiceai-54` commit forward, so nothing needs to be individually
re-selected the way a re-cut does. `Cargo.toml` pins `datafusion = "55.1.0"`,
`arrow-*`/`parquet = "59.2"`. `cargo check`/`cargo test --lib` on `vortex-array`,
`vortex-io`, `vortex-scan`, `vortex-session`, `vortex-utils`, `vortex-btrblocks`,
`vortex-datafusion` are clean (0 failures across 249 lib tests); the `vortex-io
--features tokio --test cancel_stress` mechanical guard for rows 1–3 could not be run in
this environment (`cargo` hit `ENOSPC` mid-compile — a full disk shared with concurrent
builds on the same machine, unrelated to this branch) — treat rows 1–3 as grep-verified
only, not behaviorally re-confirmed, until that test is run somewhere with disk headroom.

**Reconciliation with `spiceai-54-vortex-0.85.0` + PR spiceai/vortex#101**
(spiceai/spiceai#13570): a separate, independently-run 0.79.0→0.85.0 re-cut (still
DataFusion 54 / Arrow 58.3) exists on `origin/spiceai-54-vortex-0.85.0`, with two more
commits on top on `viktor/vortex-0.85.0-map-and-cast-fixes` (PR #101, open). That work is a
**subset** of what landed here, checked patch-by-patch:

- Row 14 (Arrow `Map`): PR #101 independently confirmed upstream's native `Map` type
  (`vortex-array/src/dtype/map.rs`) supersedes the old alias, via
  `cargo test -p vortex-arrow --lib map` (17 pass). Same conclusion this branch reached
  independently for `0.86.1`; native `Map` is present here too.
- Rows 7/15/16 (date→timestamp array + scalar casts): PR #101 re-implemented
  `cast_date_days_to_timestamp_{nanoseconds,seconds_nullable}` against `0.85.0`'s
  `CastReduce::cast(array: ArrayView<'_, Extension>, ...)` signature. `spiceai-55` has the
  *same* two test names in `vortex-array/src/arrays/extension/compute/cast.rs`, landed
  independently via the `0.86.1` merge resolution — convergent evidence the fix is right.
- Row 13 (`set_available_parallelism`) and the untracked 2026-09-11 IN-list rewrite series
  (`6c9ffc507`, `2f1a22ada`, `af1c5b301`, `f73241661`, `95d40c8bb`, `aff66352b`) and
  `4e2d62654` (absolute split concurrency, #87): all are ancestors of `spiceai-54`'s tip and
  so are carried into `spiceai-55` by construction (`git merge-base --is-ancestor <sha>
  spiceai-55` — not re-run after the disk filled, but true by the merge's construction: no
  commit reachable from `spiceai-54` can be dropped by a `git merge` of it). The 0.85.0
  re-cut explicitly deferred all of these ("budget a dedicated session" for the IN-list
  series) — they are **not yet on that line**, so `spiceai-55` is currently ahead of it.
- Row 4 (timezone) and row 6 (writer lock re-entry): the 0.85.0 ledger marks both
  "needs re-porting" (real conflicts, not attempted). This branch has
  `vortex-array/src/extension/datetime/timezone.rs` (199 lines vs. `0.86.1`, lib tests
  pass) and `vortex-file/src/writer.rs`'s `new_array_context` now calls
  `session.arrays()` once into a local and reuses it — the reentrant-lock shape row 6
  fixed no longer exists structurally, upstream, independent of either fork's patch. No
  dedicated regression test for either in this branch (same gap the 0.85.0 ledger notes).
- Row 9 (pushdown bubbles `TRUE` for an unsupported node) and row 10
  (`UncompressedSizeInBytes`): the 0.85.0 ledger correctly notes `vortex-datafusion` is
  vendored into Spice's main repo as `crates/vortex` and not taken from this fork — out of
  scope here too. Checked row 9 directly on `spiceai-55`: `vortex-datafusion/src/convert/exprs.rs`
  still carries only the **TODO comment** ("Don't return an error when we have an
  unsupported node, bubble up TRUE..."), not the behavior — consistent with "vendored
  elsewhere, not this fork's problem," but worth flagging since the comment could be
  mistaken for the fix being present.

Net: `spiceai-55` is a content superset of `spiceai-54-vortex-0.85.0` + PR #101 for every
patch currently landed on either line, and is additionally already on the target
`0.86.1`/DataFusion 55.1/Arrow 59.2. The trade-off is git-history shape: this is one merge
commit rather than individually cherry-picked commits per patch, so a future re-cut auditor
has to read this section (and the merge diff) rather than `git log --oneline` a dedicated
`-patches` branch. If the team wants the cherry-pick provenance restored, the individual
original SHAs are unchanged in `spiceai-54`'s history and can still be cherry-picked onto a
fresh `0.86.1` branch using this section as the row-by-row map.

## Convention

Follow what the DataFusion fork does:

- Create a **`spiceai-<version>-patches` branch** from each new version branch.
- Cherry-pick every patch below as an **individual commit — never squashed or batched**, so
  the next upgrade can enumerate exactly what needs porting.
- Perform a **patch audit** at every upgrade: for each patch, is it still needed, or did it
  land upstream?

`spiceai-53-patches` exists; there is no `spiceai-54-patches`. That gap is why the July 2026
work had nowhere to land.

## Tracked patches

`Verify` is run from the repository root and must succeed for the patch to be present.

Write these so they cannot pass on a tree that kept the comments and dropped the code —
match the `use` statement itself, not a mention of the crate. Prefer a test where one
exists: a grep proves presence, a test proves behaviour.

| # | Patch | Origin | Fixes | Verify | Upstream? |
|---|---|---|---|---|---|
| 0 | **This file** | — | Losing the patch list itself, which would defeat every row below | `test -f SPICE_PATCHES.md` | No |
| 1 | Tokio one-shot in `single.rs` | `b27e89af5` (#9, rebased from `96950b8c2`) | Reentrant waker drop → `"future still here when dropping"` panic / SIGSEGV | `grep -qE '^use tokio::sync::oneshot;$' vortex-io/src/runtime/single.rs` | No |
| 2 | Tokio one-shot for the spawned `Task` result channel | this change; supersedes `ea51ad3ea` | The same hazard on the `Handle::spawn`/`Task` cancellation path, unconditionally rather than behind the `tokio` feature | `grep -qE '^pub use tokio::sync::oneshot;$' vortex-io/src/runtime/handle.rs` **and** `cargo test -p vortex-io --features tokio --test cancel_stress` | No |
| 3 | Tokio one-shot for the segment-read result channel | this change | The same hazard on `ReadFuture`, which is polled and then dropped on cancellation | `grep -qE '^use vortex_io::runtime::oneshot;$' vortex-file/src/segments/source.rs` | No |
| 4 | Fixed-offset timezone resolution | `6cdea73d6` (#75), `5b4bee108` (#78) | Panic `failed to find time zone '+00:00'` for a `timestamptz` column | `test -f vortex-array/src/extension/datetime/timezone.rs` | No |
| 5 | Balance `list_contains` OR tree | `d694abda6` (#37) | Plan blowup on large `IN`-list filters | — (needs a check) | No |
| 6 | Avoid session lock re-entry in writer init | `c536c9aed` (#29) | Deadlock in `vortex-file` writer initialisation | — (needs a check) | No |
| 7 | `vortex.date` → `vortex.timestamp` extension casts | `7e5b08151` (#28) | Missing cast between extension types | `cargo test -p vortex-array --lib arrays::extension::compute::cast` | No |
| 15 | `vortex.date` → `vortex.timestamp` **scalar** cast | this change | Row 7 covers arrays only. A scan casts a file's `min`/`max` statistic — a scalar — through the same expression, and `Scalar::cast` routed an extension source through the *target's* storage type: `date[days]` failed the scan outright, and `date[ms]` shares `i64` with `timestamp[ns]`, so it silently returned an instant 10^6 too small and pruned files that held matching rows (spiceai/spiceai#13624) | `cargo test -p vortex-array --lib scalar::typed_view::extension::tests::test_ext_scalar_cast` | No |
| 16 | Timestamp validation uses `storage_range`, and rendering never aborts | this change | `Timestamp` validated a storage value, and rendered one, by building a Jiff span from it, and a span's limits are not a timestamp's. They stop one short of `i64::MIN` nanoseconds — 1677-09-21, an instant a `timestamp[ns]` array holds — so a scalar built from such a column's statistic was refused although the array carried it. They also run *past* the last instant, and the unchecked constructors abort outside them, so `i64::MAX` seconds panicked rather than being reported. `unpack_native` now checks `Timestamp::storage_range`, which is also the range row 15's conversion targets, so the two cannot drift apart. `Display` still uses a span for seconds through microseconds, whose limits enclose the timestamp's, but only the *checked* constructors plus `checked_add`, and it takes nanoseconds through `Timestamp::from_nanosecond`, whose range covers every `i64`; a count that denotes no instant renders as itself, since a `Display` impl cannot report a failure. Reachable from any read, or any formatting, of such a value | `cargo test -p vortex-array --lib extension::datetime::timestamp` | Proposed — upstream defect, not Spice behaviour |
| 8 | N-ary `CASE WHEN` expression | `4bfa4331b` (#12), `df23c3797` | Expression support required by pushdown | — (needs a check) | Possibly upstream |
| 9 | Unsupported pushdown node bubbles `TRUE` | `8044a8470` (#8) | Pushdown erroring instead of degrading to "keep row"; empty `IN` list | Out of scope for this fork: `vortex-datafusion` is vendored into Spice's main repo as `crates/vortex`, not taken from here. Only a leftover TODO comment survives in this fork's `vortex-datafusion/src/convert/exprs.rs`; the behavior itself is not implemented on this branch and was never expected to be | No (vendored elsewhere) |
| 10 | `UncompressedSizeInBytes` statistic handling | `6712e9ffa` (#3) | Incorrect statistic propagation | — (needs a check) | No |
| 11 | Intra-file decode parallelism | `9d3aafb06`, `26b274c72` (#62) | Scan throughput on large chunk spans | — (needs a check) | Possibly upstream |
| 12 | Restore lint checks on forks | `bb80c537b` | Fork CI not running lints | — (CI config) | No |
| 13 | `set_available_parallelism` | this change | Scan and writer fan-out sized from the machine's core count rather than what the host process is entitled to (spiceai/spiceai#12328) | `cargo test -p vortex-utils --test parallelism_declared --test parallelism_declared_too_late` | Proposed — additive, detection unchanged |
| 14 | Arrow `Map` alias | `1a6dc54f1` (`lukim/map`) | `Map` columns unwritable: `Array encoding not implemented for Arrow data type Map(...)` on every flush (spiceai/spiceai#13524) | `cargo test -p vortex-arrow --lib executor::map` | Yes — 0.86.1 native `Map` (`#9111`) |

**Confidence:** rows 1–4, 13 and 14 are verified against the branches. Rows 5–12 are seeded from an
audit of non-merge commits on `spiceai-54` authored by Spice engineers; their descriptions
come from commit subjects rather than from reading each diff, and some may be upstream
cherry-picks rather than Spice patches. Anyone touching a row should confirm it and fill in
its `Verify` command.

## Known remaining exposure

The `oneshot` crate is still used elsewhere, and the same hazard applies anywhere a receiver
is polled and then dropped while its sender completes:

- `vortex-layout/src/layouts/dict/writer.rs` awaits `values_rx` inside a future handed to a
  consumer, and the sender explicitly handles `"values receiver dropped"`, so the window
  exists. Whether a consumer polls and then drops it is unproven — audit before assuming it
  is safe.
- `vortex-file/src/read/driver.rs` creates channels only inside `#[cfg(test)]`, and drops
  those receivers without polling, so it is not exposed.

## Unlanded work

These branches carry changes that exist on no shipping branch. Each should be landed or
explicitly closed out with a reason, then reflected above. No pull requests exist for any of
them, so none is currently owned.

| Branch | Commit | What it is |
|---|---|---|
| `sgrebnov/cold-stall-vortex-oneshot-handle` | `ea51ad3ea` (branch tip) | The `handle.rs` receiver swap — superseded by row 2 |
| `sgrebnov/cold-stall-vortex-dumper` | `051470dc9` (ancestor; tip is `2e52f5cd0`) | Tokio one-shot for the spawned-task result channel; overlaps row 2. The branch tip also carries an unrelated kanal bump, so cherry-pick rather than merge |
| `spiceai-54-tokio-channels` | `1551a2ae8` (branch tip) | Replaces `kanal` and `oneshot`-crate waits in the **write path** with tokio primitives. **Not covered by rows 2–3** — same hazard class, different sites |
| `sgrebnov/cold-stall-vortex-diag` | — | Diagnostics for the same investigation |
| `spiceai-54-kanal-fix`, `spiceai-54-vortex-0.75.0` | — | Untriaged |

## Upgrade checklist

1. Carry this file forward first, then create **`spiceai-<new>-patches`** from the new
   version branch.
2. Cherry-pick every tracked patch as an individual commit. Never squash.
3. Run each `Verify` command against the new branch. A missing patch must be a recorded
   decision, not an omission.
4. For each patch, check whether it landed upstream in the new release; if so, drop it and
   note that here.
5. Update the table: new SHAs, new upstream status.
6. Run `cargo test -p vortex-io --features tokio`. The `cancel_stress` test is the
   mechanical guard for row 2 and fails, by crashing, if that patch is lost.

## Keeping the list honest

- A patch is tracked only once it has a `Verify` command that matches code rather than
  prose. Comments survive a bad re-cut; code does not.
- When adding a patch to this fork, add its row here in the same change.
