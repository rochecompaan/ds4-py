# System-Prefix Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache each rendered system prefix as a native KV payload so later sessions with the same model and prefix skip its initial prefill without changing generated tokens.

**Architecture:** Compute one SHA-256 key from the engine's model id, routed quantization, configured context size, and exact rendered prefix token ids, then store the payload at `<DS4PY_HOME>/sysprompt/<sha256>.kv`. The existing chat-start transaction will try a guarded native payload load, preserve the already-rendered transcript segments, and otherwise use ordinary sync/prefill before atomically publishing a replacement. The existing in-memory `system` checkpoint remains the final step on every path.

**Tech Stack:** Python 3.14, existing `EngineWorker` jobs, `SessionStore` KV header and atomic streaming writer, `FakeNative`, pytest, ds4py agent CLI.

**Spec:**
- `docs/specs/2026-10-21-ds4-py-design.md`: “KV reuse and `Plan`”, “The system-prefix cache”, and the M4 completion criteria.
- `docs/specs/2026-11-10-m3-harness-integration-design.md`: sections 10.3, 10.4, and 14.2.

## Global Constraints

- Use `<DS4PY_HOME>/sysprompt/<sha256>.kv`, defaulting to `~/.ds4py` through `SessionStore` when `DS4PY_HOME` is unset or empty.
- Key the file by model id, routed quantization, context size, and the complete ordered prefix token ids; do not key by prompt text.
- Use the existing `KV_MAGIC`, `KV_VERSION`, and payload representation used by `session.kv`; no compatibility with another file format is required.
- Probe `Session.file_payloads_supported()` and native `restore_enabled` before every cache load/save. Either being false means no cache read, no cache write, and the current rendering/prefill behavior.
- Missing, unreadable, corrupt, truncated, extra-data, and token-mismatched cache files must fall back to ordinary prefill and be atomically replaced after a successful prefill.
- Never accept loaded native tokens unless they exactly equal the rendered prefix, including length and order. Preserve the staged transcript and its original system segment boundaries on a cache hit.
- Publish through the existing destination-directory temporary-file-plus-rename implementation; concurrent writers must never expose a partial final file.
- Cache save failure is non-fatal: ordinary prefill, the live transcript, and the in-memory `system` checkpoint remain authoritative. Only `StoreError` from publication is suppressed; native engine failures are not hidden.
- Route direct chat initialization, harness request binding, system-prompt rebuilds, `/new`, and rendered restores through the same `BindingTransaction.start_chat()` implementation.
- A rendered restore never loads the session's own `session.kv`. With file-payload support it may load only the separately keyed and exactly validated system-prefix file, leaving the restored message tail to render/prefill. Without file-payload support it retains a full rebuild prefill.
- Preserve the existing `EngineError("chat start was interrupted")` behavior when ordinary system prefill is interrupted.
- Do not change sampling, rendering, decode, generated token ids, or CLI output. Greedy output with the same seed must match with and without the cache.
- Do not add startup warming, busy/progress UI, new dependencies, new public options, or downstream-specific names.

## Review Focus

- **Valid native payload for the wrong tokens:** reset native state, prefill the expected tokens, and replace the file; `test_mismatched_cache_is_rebuilt_and_replaced` pins this.
- **Corruption after an otherwise valid header:** leave the staged transcript intact and worker reusable; parameterized corrupt/truncated/unreadable cases pin this.
- **Rendered restore:** supported engines reuse only the exact separate system-prefix file; unsupported engines still rebuild everything; updated `tests/agent/test_restore.py` cases pin both and assert no `session.kv` load.
- **System rewrites after `/new` or changed instructions/tools:** old checkpoints must not make a different prefix reusable; direct session and real runtime tests pin both.
- **Cache I/O failure and concurrent cold starts:** save failure must not abort chat, only complete KV bytes may appear at the final path, and temporary files must be cleaned up; cache and streaming-writer fault tests pin these.

---

### Task 1: Test and implement the complete cache lifecycle

**Files:**
- Create: `src/ds4py/engine/sysprompt.py`
- Modify: `src/ds4py/engine/history.py` (`SessionHistory._sync_chat`)
- Modify: `src/ds4py/engine/worker.py` (`LoadPayloadJob` and `_load_payload`)
- Modify: `src/ds4py/engine/session.py` (shared internal payload-job drains)
- Modify: `src/ds4py/errors.py` (narrow payload load/mismatch errors)
- Modify: `src/ds4py/store.py` (store-owned `sysprompt/` layout)
- Modify: `docs/specs/2026-11-10-m3-harness-integration-design.md` (section 10.3)
- Modify: `tests/conftest.py` (isolate test cache writes)
- Create: `tests/engine/test_system_prefix_cache.py`
- Create: `tests/agent/test_agent_system_prefix_cache.py`
- Modify: `tests/agent/test_restore.py`
- Modify: `tests/agent/test_compaction_persistence.py`
- Modify: `tests/agent/test_store_failures.py`
- Modify: `tests/agent/test_store.py`

**Interfaces:**
- Produces: `system_prefix_path(store: SessionStore, engine: Engine, tokens: Sequence[int]) -> Path`, returning `store.sysprompt_path(digest)`. The digest input is the ASCII model id, routed quant bits, context size, and `token_digest(tokens)`, with unambiguous separators.
- Produces: `sync_system_prefix(session: Session, tokens: tuple[int, ...]) -> None`, called only inside the binding transaction. Pending cancellation and disabled native restore use ordinary prefill without cache I/O. On a miss/fallback it submits the existing `SyncJob` and raises `EngineError("chat start was interrupted")` for an `Interrupted` event. After supported successful prefill it publishes best-effort through `SessionStore.write_kv_stream_atomic`. Public payload methods and the cache share `Session._save_payload_events` / `Session._load_payload_events`; `SessionHistory._sync_chat` retains sole responsibility for `_checkpoint_native(SYSTEM)`.
- Consumes: `LoadPayloadJob(file: BinaryIO, payload_bytes: int, transcript: Transcript | None, expected_tokens: tuple[int, ...] | None = None)`. Existing public session restore passes the mutable transcript and no expected tokens, preserving its behavior. A cache load passes no transcript and exact expected tokens.
- Worker behavior: native load failure on the verified-cache path resets native state and raises `PayloadLoadError`; token mismatch resets and raises `PayloadMismatch`. On success, close stale native checkpoints, emit `Loaded`, and modify the transcript only when one was supplied. The shared internal Session load drain clears history checkpoint/pause bookkeeping once for both public and cache callers.
- Local test helpers: define `fake_engine` and event-capture helpers in the new test modules; do not refer to nonexistent shared helpers.

- [ ] **Step 1: Isolate test cache state**

Add an autouse pytest fixture that points `DS4PY_HOME` at a per-test temporary directory. Existing tests that explicitly set or delete the variable retain their own overrides; production path resolution remains unchanged.

- [ ] **Step 2: Write the engine cache tests before implementation**

In `tests/engine/test_system_prefix_cache.py`, add focused tests with these observable assertions:

1. `test_second_startup_loads_same_prefix_without_prefill`: cold start writes one valid `.kv` with `payload_bytes == len(prefix) * 1024`; warm start has identical tokens and original system segments, `system` checkpoint length `len(prefix)`, `Plan("extend", len(prefix), 0, True, None)`, a `Loaded` event, and no positive-work system `Prefill`.
2. `test_new_session_reuses_cache_and_changed_prefix_gets_new_file`: `session.new()` plus `start_chat()` hits the same file; a different tool definition and instructions creates a second filename and cold-prefills. Separate key assertions vary model id, quant bits, context size, and token order, and change `DS4PY_HOME` to prove root resolution.
3. Parameterized invalid-file cases: missing, directory/unreadable path, bad magic, bad version, short header, short payload, and extra bytes. Every case completes ordinary prefill, preserves exact prefix/segments, and leaves a valid replacement except missing (which creates it). A following startup must load it without positive-work system prefill.
4. `test_mismatched_cache_is_rebuilt_and_replaced`: copy valid prefix A's payload to prefix B's expected filename; B's next sync must be a rebuild from cached position zero, not an extension. Verify replacement and a subsequent warm hit.
5. `test_unsupported_file_payloads_prefill_without_touching_cache`: with `FakeNative(payload_support=False)`, a pre-existing would-be cache file remains byte-identical and native load/stage are never called; with a fresh home, no `sysprompt` directory is created.
6. `test_cache_write_failure_does_not_abort_chat_start`: force `write_kv_stream_atomic` to raise `StoreError`; assert ordinary prefill, exact transcript, and the `system` checkpoint still succeed.
7. `test_interrupted_system_prefill_keeps_existing_error_contract`: cancel before cold start and assert `EngineError("chat start was interrupted")` with no published cache file.
8. `test_concurrent_cold_starts_publish_one_complete_file`: two independent FakeNative engines race the same home/key; exactly one valid final `.kv` and no `.tmp` remain, and both prefixes warm-load successfully.
9. `test_greedy_tokens_match_without_cache_and_with_cold_and_warm_cache`: with `SamplingPolicy(temperature=0, seed=1)`, compare all generated `Token.id` values and final transcripts for unsupported-payload, cold-cache, and warm-cache runs. This portable test proves orchestration/transcript equality; the real-model check in Task 2 proves native KV output equivalence.

When attributing prefill events, ignore zero-work `Prefill(total=0)` emitted by `SavePayloadJob`'s confirming sync. Require cold positive-work prefill and warm zero positive-work system prefill rather than assuming identical backend event counts.

- [ ] **Step 3: Write agent, rendered-restore, and streaming-writer tests before implementation**

In `tests/agent/test_agent_system_prefix_cache.py`, construct the real `DS4Model`/`AgentRuntime` with FakeNative. Run one prompt, call `await runtime.new()`, then run the next prompt. Assert the post-`/new` system initialization loaded the same cache with no positive-work system `Prefill`, retained the `system` checkpoint, and returned the scripted answer.

Update `test_stripped_payload_switch_renders_fresh_with_full_prefill` in `tests/agent/test_restore.py` to reflect the approved supported-engine semantics: system-prefix cache load is allowed, the first sync extends from the exact cached system prefix, and only the restored tail is prefilled. Record native payload-load filenames and assert no load comes from `session.kv`. Add a separate unsupported-engine stripped/rendered restore case with `payload_support=False` that retains `plan.kind == "rebuild"` and `prefilled == pos` and performs no cache/session KV load.

In `tests/agent/test_store_failures.py`, parameterize `fsync` and `replace` failure for `write_kv_stream_atomic`; assert `StoreError`, prior bytes intact, and no temporary file. This characterizes the reused atomic primitive rather than changing it.

Apply the same supported rendered-restore expectation to the missing/zero-payload cases in `tests/agent/test_compaction_persistence.py`: one tail sync extends from the cached preamble. Keep the explicit `payload_support=False` compaction restore test unchanged as full-prefill coverage.

- [ ] **Step 4: Run all focused tests and verify RED**

Run:

```bash
uv run --frozen --extra agent pytest tests/engine/test_system_prefix_cache.py tests/agent/test_agent_system_prefix_cache.py tests/agent/test_restore.py tests/agent/test_store_failures.py -q
```

Expected: cache lifecycle, changed-key, invalid-file, mismatch, unsupported no-I/O, `/new`, and supported rendered-restore expectations fail because caching does not exist. The unsupported rendered-restore full-prefill test and existing atomic-writer characterization may pass because they pin preserved behavior; no production code is written until the feature tests have been watched failing for the missing cache.

- [ ] **Step 5: Implement the key, guarded load, ordinary fallback, and best-effort atomic save**

Create `sysprompt.py` with the interfaces above. Reuse the existing KV constants, events, jobs, and `SessionStore`; do not duplicate the KV format, atomic writer, or Session's payload-job drains. Probe payload support and native restore before path I/O. Catch only `StoreError`, `OSError`, `PayloadLoadError`, and `PayloadMismatch` around the attempted cache load. After any load failure, run ordinary prefill and attempt replacement; suppress only `StoreError` from publication. Do not catch ordinary-prefill or fatal worker errors.

Extend `LoadPayloadJob` as specified. On native load failure or token mismatch use the worker's existing reset path so the following sync genuinely rebuilds. On successful cache load preserve the staged transcript and clear checkpoint/pause ownership through the shared Session helper. Ordinary `Session.load_payload()` behavior remains unchanged.

Change `_sync_chat` to call `sync_system_prefix()` and then `_checkpoint_native(SYSTEM)` exactly once on every successful path.

- [ ] **Step 6: Amend M3 §10.3 with the approved rendered-restore semantics**

State that rendered restore never calls `load_payload` for the session's own `session.kv`. With `file_payloads_supported=True`, its normal chat initialization may load only the separate system-prefix cache after exact token validation, so the restored tail rather than the system prefix is prefilled. Missing/invalid/mismatched cache and `False` support retain ordinary rendering and full rebuild prefill. Record in the execution ledger that the owner approved this refinement under §14.2 and that the PR must call it out for Roché.

- [ ] **Step 7: Run focused and neighboring regression tests**

Run:

```bash
uv run --frozen --extra agent pytest tests/engine/test_system_prefix_cache.py tests/agent/test_agent_system_prefix_cache.py tests/agent/test_restore.py tests/agent/test_store_failures.py tests/engine/test_payload_jobs.py tests/engine/test_session.py tests/engine/test_session_chat.py tests/engine/test_binding_transaction.py tests/engine/test_fatal_state.py -q
```

Expected: all pass. Cold paths have positive system prefill, warm paths load exact tokens, invalid files are replaced, unsupported paths perform no cache I/O, and rendered restore obeys both support cases.

- [ ] **Step 8: Run the complete portable regression suite**

Run:

```bash
HOME=$(mktemp -d) uv run --frozen --extra agent pytest -m "not build and not live" -q
```

Expected: all portable tests pass; record exact passed/skipped/deselected counts. Investigate rather than weaken any existing failure.

- [ ] **Step 9: Commit the implementation, tests, plan, and approved spec refinement**

```bash
git add docs/plans/2026-11-20-system-prefix-cache.md docs/specs/2026-11-10-m3-harness-integration-design.md src/ds4py/engine/sysprompt.py src/ds4py/engine/history.py src/ds4py/engine/worker.py src/ds4py/engine/session.py src/ds4py/errors.py src/ds4py/store.py tests/conftest.py tests/engine/test_system_prefix_cache.py tests/agent/test_agent_system_prefix_cache.py tests/agent/test_restore.py tests/agent/test_compaction_persistence.py tests/agent/test_store_failures.py tests/agent/test_store.py
git commit -m "feat: cache rendered system prefixes" -m "Part of #38"
```

Final whole-branch review produced five additional RED regression tests: disabled native restore, failed native load reset, warm interruption, saved token count, and store-owned cache layout. The owner approved the `restore_enabled` guard and contained shared-Session-helper refactor. All five passed after the corrections, followed by the focused and complete gates. Cache eviction and unrelated test-helper restructuring remain outside this feature.

### Task 2: Real-model measurements, static gates, and delivery evidence

**Files:**
- No planned production changes. Any correction returns to Task 1's RED-GREEN cycle with a regression test first.
- Update the PR body with measured results; do not commit generated caches, traces, model output, or build products.

**Interfaces:**
- Consumes the completed implementation and the provided Metal `libds4.dylib` and Qwen GGUF model.
- Produces reproducible timing, size, output-equivalence, gate, commit, PR, and review evidence.

- [ ] **Step 1: Establish the live command and exclusive model access**

Before every model-run batch, execute `pgrep -fl "ds4|dogfood|satyrn-agent"`. If another model process is present, stop and ask the owner. Check the provided library with `ds4py check --lib <lib>`. If this branch rejects its manifest, rebuild only as requested with `ds4py build --src <provided-source-tree> --backend metal --out <directory under ~/work>` and check the rebuilt library.

Use the ds4py agent CLI in non-interactive mode with `--model ~/models/Qwen3.8-27B-Q4_K_M.gguf --temp 0 --seed 1 --lib <lib>` and a fixed first prompt. Never use the raw prompt command.

- [ ] **Step 2: Measure cold and warm first-token latency, n=3 each**

Use separate isolated homes for cold and warm groups. For each cold repetition, remove only that group's `sysprompt/` cache before launching the agent. Prime one warm cache, then launch three fresh agent processes against it without deleting the file. Capture timestamped `run_start`, system `Prefill`/`Synced` or `Loaded`, and first `Token` events from the real agent run record, and pair them with a monotonic external wall-clock measurement. Do not use final-answer completion time as first-token latency.

Record for every repetition: cold/warm, exact command, first-token seconds, system prefill token count/rate where present, generated greedy token ids/text, cache path, cache file size, and cache-load duration. Measure load duration with a temporary external wrapper around the agent CLI that times only the cache `LoadPayloadJob`; do not add product logging solely for the benchmark.

Repeat one live run with a valid-header/wrong-token cache and one with a corrupt cache. Both must fall back to a full rebuild and produce the same greedy token ids as clean cold. These are the real-backend proofs of mismatch reset and corrupt-KV output equivalence; FakeNative's scripted logits cannot prove KV fidelity.

Expected: cold performs ordinary system prefill and creates one complete cache; warm has no positive-work system `Prefill`, loads the same tokens, and greedy output matches every cold run. Report all n=3 values and means, not only the best result.

- [ ] **Step 3: Run the required static gate and final portable suite**

```bash
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
HOME=$(mktemp -d) uv run --frozen --extra agent pytest -m "not build and not live" -q
```

Expected: every static hook passes and the portable suite is green. Record exact counts and warnings accurately.

- [ ] **Step 4: Review the diff and deliver**

Inspect `git diff origin/main...HEAD` and `git status --short`; ensure only planned source/tests/spec/plan files are tracked and no cache, build, model, trace, or local environment artifacts are included. Push `feat/system-prefix-cache` and open a PR whose short body states what is cached and where, all fallbacks, live commands and n=3 measurements, output equivalence, and gate results. Include a **Human reviewer callout for Roché** describing the approved §10.3 rendered-restore refinement. Use `Part of #38` in commit messages, with no co-author or tool trailers.

- [ ] **Step 5: Run the required PR review loop**

Load and follow `.agents/skills/pr-review-loop/SKILL.md`, alternating fresh reviewers between the two models requested by the owner. Ask the owner for each ruling before accepting a scope/design change or declining a concrete finding. Fix correctness findings through TDD, rerun affected gates after pushed changes, and record every finding and disposition in the PR's review-evidence section.
