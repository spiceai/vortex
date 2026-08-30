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
| 16 | Timestamp validation uses the timestamp range, not a span's | this change | `Timestamp::unpack_native` validated a storage value by building a Jiff span from it, and a span's limits are not a timestamp's at either end. They run *past* the last instant, and the unchecked constructors abort outside them, so `i64::MAX` seconds panicked instead of being reported. They also stop one short of `i64::MIN` nanoseconds, which is 1677-09-21 — an instant a `timestamp[ns]` array holds and row 15's conversion produces, so the scalar refused a value the array accepted. Both ends now come from `Timestamp::storage_range`, the single range row 15 converts into. Reachable from any read of such a value | `cargo test -p vortex-array --lib extension::datetime::timestamp` | Proposed — upstream defect, not Spice behaviour |
| 8 | N-ary `CASE WHEN` expression | `4bfa4331b` (#12), `df23c3797` | Expression support required by pushdown | — (needs a check) | Possibly upstream |
| 9 | Unsupported pushdown node bubbles `TRUE` | `8044a8470` (#8) | Pushdown erroring instead of degrading to "keep row"; empty `IN` list | — (needs a check) | No |
| 10 | `UncompressedSizeInBytes` statistic handling | `6712e9ffa` (#3) | Incorrect statistic propagation | — (needs a check) | No |
| 11 | Intra-file decode parallelism | `9d3aafb06`, `26b274c72` (#62) | Scan throughput on large chunk spans | — (needs a check) | Possibly upstream |
| 12 | Restore lint checks on forks | `bb80c537b` | Fork CI not running lints | — (CI config) | No |
| 13 | `set_available_parallelism` | this change | Scan and writer fan-out sized from the machine's core count rather than what the host process is entitled to (spiceai/spiceai#12328) | `cargo test -p vortex-utils --test parallelism_declared --test parallelism_declared_too_late` | Proposed — additive, detection unchanged |
| 14 | Arrow `Map` alias | `1a6dc54f1` (`lukim/map`) | `Map` columns unwritable: `Array encoding not implemented for Arrow data type Map(...)` on every flush (spiceai/spiceai#13524) | `cargo test -p vortex-arrow --lib map_array_roundtrips_through_the_list_alias` | No |

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
