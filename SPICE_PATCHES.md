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
| 4 | Fixed-offset timezone resolution | `6cdea73d6` (#75), `5b4bee108` (#78) | Panic `failed to find time zone '+00:00'` for a `timestamptz` column | `test -f vortex-array/src/extension/datetime/timezone.rs` **and** `cargo test -p vortex-array --lib extension::datetime::timezone` | Partly — 0.93 resolves fixed offsets itself (`resolve_time_zone`) and refuses an unresolvable zone when the dtype is built. On `spiceai-55` 0.93 the upstream resolver delegates to this file's `resolve_timezone`, so `Display`, `unpack_native`, `is_utc_timezone` and dtype validation accept the same zones |
| 5 | Balance `list_contains` OR tree | `d694abda6` (#37) | Plan blowup on large `IN`-list filters | — | **Dropped at 0.93** — upstream rewrote `list_contains` around a prepared set (`PreparedSet`), so a constant list builds no OR tree. Guarded in spiceai/spiceai by `test_large_in_list_filter_pushdown_stays_evaluable` |
| 6 | Avoid session lock re-entry in writer init | `c536c9aed` (#29) | Deadlock in `vortex-file` writer initialisation | — | **Dropped** — upstream rewrote writer initialisation to read the session once, and #8919 made the session registry lock-free |
| 7 | `vortex.date` → `vortex.timestamp` extension casts | `7e5b08151` (#28) | Missing cast between extension types | `cargo test -p vortex-array --lib arrays::extension::compute::cast` | No — 0.93 still refuses the cast |
| 15 | `vortex.date` → `vortex.timestamp` **scalar** cast | `3c5246867` (#93) | Row 7 covers arrays only. A scan casts a file's `min`/`max` statistic — a scalar — through the same expression, and `Scalar::cast` routed an extension source through the *target's* storage type: `date[days]` failed the scan outright, and `date[ms]` shares `i64` with `timestamp[ns]`, so it silently returned an instant 10^6 too small and pruned files that held matching rows (spiceai/spiceai#13624) | `cargo test -p vortex-array --lib scalar::typed_view::extension::tests::test_ext_scalar_cast` | No — 0.93's `ExtScalar::cast` only casts to the storage type |
| 16 | Timestamp validation uses `storage_range`, and rendering never aborts | `3c5246867` (#93), `4539a2fa6` (#108) | `Timestamp` validated a storage value, and rendered one, by building a Jiff span from it, and a span's limits are not a timestamp's. They stop one short of `i64::MIN` nanoseconds — 1677-09-21, an instant a `timestamp[ns]` array holds — so a scalar built from such a column's statistic was refused although the array carried it. They also run *past* the last instant, and the unchecked constructors abort outside them, so `i64::MAX` seconds panicked rather than being reported. `unpack_native` checks `Timestamp::storage_range` (every `i64` since #108), which is also the range row 15's conversion targets. `Display` takes nanoseconds through `Timestamp::from_nanosecond`, whose range covers every `i64`, renders a count that denotes no instant as itself, and renders the UTC instant when a timezone does not resolve | `cargo test -p vortex-array --lib extension::datetime::timestamp` | Partly — 0.93 (#10953) also accepts every `i64` and falls back to the raw count, but renders `i64::MIN` nanoseconds as a count and aborts on an unresolvable zone |
| 8 | N-ary `CASE WHEN` expression | `4bfa4331b` (#12), `df23c3797` | Expression support required by pushdown | — | Yes — upstream `case_when` |
| 9 | Unsupported pushdown node bubbles `TRUE` | `8044a8470` (#8) | Pushdown erroring instead of degrading to "keep row"; empty `IN` list | — | **Not fork state** — `vortex-datafusion` is vendored into spiceai/spiceai as `crates/vortex`; from 0.93 the fork carries upstream's `vortex-datafusion` unmodified |
| 10 | `UncompressedSizeInBytes` statistic handling | `6712e9ffa` (#3) | Incorrect statistic propagation | — | Yes — 0.93 records `UncompressedSizeInBytes` in the default file statistics when the editions allow it (#10923) |
| 11 | Intra-file decode parallelism | `9d3aafb06`, `26b274c72` (#62) | Scan throughput on large chunk spans | `grep -q 'LayoutSubSplitting { max_rows: u64 }' vortex-layout/src/scan/split_by.rs` | No. Spice's scan uses `SplitBy::Layout` and does not select `LayoutSubSplitting` |
| 12 | Restore lint checks on forks | `bb80c537b` | Fork CI not running lints | — (CI config) | Mostly — 0.93's workflows fall back to `ubuntu-latest` off upstream's runner pool. The fork still carries the `publish-dry-runs` fallback, the instrumented-test timeout, and a flatbuffer back-compat check against the base branch rather than `develop` |
| 13 | `set_available_parallelism` | this change | Scan and writer fan-out sized from the machine's core count rather than what the host process is entitled to (spiceai/spiceai#12328) | `cargo test -p vortex-utils --test parallelism_declared --test parallelism_declared_too_late` | Proposed — additive, detection unchanged |
| 14 | Arrow `Map` alias | `1a6dc54f1` (`lukim/map`) | `Map` columns unwritable: `Array encoding not implemented for Arrow data type Map(...)` on every flush (spiceai/spiceai#13524) | `cargo test -p vortex-arrow --lib executor::map` | Yes — 0.86.1 native `Map` (`#9111`) |
| 17 | Byte-capped data blocks for string/binary columns, with encodings chosen once per row block | `e5566746d` (#106) | A string or binary column is written in blocks of 8,192 rows whatever its row width, because `effective_block_len` sizes blocks only for fixed-width types; a ~3 KB/row JSON column gets ~12.5 MiB blocks, and a one-row lookup reads and copies a whole block (spiceai/spiceai#15323). `WriteStrategyBuilder::with_data_block_max_bytes` splits such a block into power-of-two row counts under the cap. Searching encodings on each small block costs several full compressions per block (a block of at most 1,024 rows is its own sample), so `CompressingStrategy::with_group_rows` and `CascadingCompressor::plan_for`/`compress_with_plan` choose encodings once per row block, on the sample the compressor would draw from the unsplit block, and apply them to its split blocks (a choice made on the first split block alone fits only it: incompressible rows ahead of JSON picked FSST for the JSON, +15% size and 2.4x `LIKE` CPU) | `cargo test -p vortex-layout --lib repartition::tests::a_block_size_maximum_splits_wide_strings_into_power_of_two_blocks` **and** `cargo test -p vortex-btrblocks --lib tests::a_replayed_plan_roundtrips_other_data_of_the_same_dtype` **and** `cargo test -p vortex-layout --lib compressed::tests::a_group_is_encoded_as_its_unsplit_data_would_be` **and** `cargo test -p vortex-file --test test_write_table test_data_block_max_bytes_roundtrip` | Proposed — additive; off unless the cap is set |
| 18 | Chunked reads skip chunks a known mask excludes | `e5566746d` (#106) | A point read of one row of a range fetched every capped block of it (spiceai/spiceai#15323) | `cargo test -p vortex-layout --lib chunked::reader::test::test_known_mask_skips_unselected_chunks` | **Yes at 0.93** — upstream `MaskFuture::peek` and `ChunkedReader` skip the chunks a completed mask excludes; the fork's `MaskFuture::known` is gone |
| 19 | Constant `IN` lists: FSST answers membership on its codes, and a zone is falsified by the list's range and the gaps between its values | `f73241661`, `2f1a22ada`, `af1c5b301`, `6c9ffc507` (#95) | `IN` over an FSST column decompressed every row to compare it; a long sparse `IN` list falsified zones only by its range, so zones lying between list values were read | `cargo test -p vortex-fsst --lib compute::list_contains` **and** `cargo test -p vortex-expr --lib stats::rewrite::builtins::tests::rewrites_list_contains_falsifier` **and** `cargo test -p vortex-layout --test list_contains_pruning` **and** `cargo test -p vortex-array --test in_list_differential` | Partly — 0.93's prepared set (`PreparedSet`) replaces the fork's hash probe, and its falsifier tests each element up to 16 and only the range past that. The fork keeps the FSST kernel (ported to `ListContainsOptions` and prepared sets) and the gap falsifier, which leaves out gaps between consecutive integers. The differential table was re-recorded at 0.93: a null needle answers null (SQL's `NULL IN (...)`), where it answered `false` |
| 20 | Absolute split concurrency, admitted lazily | `4e2d62654` (#87) | A scan's configured concurrency bounded execution but not registered reads — every split's reads were registered up front — and the per-worker multiplier read the machine's cores, so a caller sizing a memory reservation per in-flight split under-reserved | `cargo test -p vortex-layout --lib scan::repeated_scan::tests::absolute_concurrency_bounds_registered_splits` **and** `cargo test -p vortex-layout --lib scan::scan_builder::test::into_stream_bounds_registered_splits` | No — 0.93 builds every split eagerly except for a filtered limit. Re-implemented at 0.93 on upstream's row-limit scan: splits are built in CPU tasks as the buffer admits them (off the poller, as upstream requires), a limit is reserved in split order, and preparation primes the first admitted wave so its I/O starts early |
| 21 | Flat reader reuses one segment request across subrange decodes | `e09ec32c4` | Overlapping subrange reads of one flat segment requested it again per read | `cargo test -p vortex-layout --lib flat::reader::test::flat_subrange_regressed_query_patterns_reuse_segment_request` | No |

**0.93 re-cut (`spiceai-55` + `0.93.0`):** every row was checked against the merged tree, and its `Verify` command run. Also dropped: the session-stored `WriteStrategyBuilder` (`9b4bde99d`, #6) — 0.93 builds the strategy from the session's compressors, and spiceai/spiceai passes its strategy to the writer directly — and the per-`ExecutionCtx` kernel snapshot (`ccaa55627`), which 0.93 implements itself (`execute_parent_snapshot`).

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
