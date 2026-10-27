# Chat Decode Review Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve review findings 1, 2, 3, 4, and 6 without weakening the M2 chat, parser, hook, and resume contracts.

**Architecture:** Resolve one `ControlSpan` at the session boundary and share it with the stream and job. Remove unreachable pause/EOS abstractions, keep public immutable parser snapshots but construct them only at observation boundaries, share the GLM/Qwen delimiter lifecycle in one internal core, and make sampled-token tuples lazy. Keep reserve-continuation resynchronization unchanged for future history compaction.

**Tech Stack:** Python 3.14, pytest, existing `ds4py.engine` native-worker abstractions; no new dependencies or native bindings.

**Spec:** [`docs/specs/2026-10-21-ds4-py-design.md`](../specs/2026-10-21-ds4-py-design.md), sections 5–10; [`docs/plans/2026-10-21-m2-chat-decode.md`](2026-10-21-m2-chat-decode.md), Tasks 10–12; review findings 1–4 and 6 in the current request.

## Global Constraints

- Preserve eval → transcript → `Token` → parser/display events; uncommitted drafts remain invisible.
- Preserve public `feed(bytes) -> Snapshot` and `finish() -> Snapshot` semantics, canonical per-call raw text, and chunk independence.
- A reserve continuation still submits `SyncJob` for the current transcript, as required by spec section 10 and M2 Task 12. Review finding 5 is rejected because removing that sync would bypass future compaction/history replacement.
- Do not add native ABI bindings, dependencies, speculative EOS replay, or a second control/snapshot mechanism.
- New behavior and performance regressions use TDD. Pure dead-code removal and parser refactoring use the existing behavior suites instead of tests that merely assert source structure.

## Review Focus

- Changed stop sequences during a paused raw or chat round must raise without destroying the paused controller; the original sequences must still resume it (Task 1 public regression).
- Multi-byte tokens inside an active DSML value must retain literal/control behavior while constructing one snapshot per observation (Task 3 instrumented regression plus existing stream tests).
- GLM/Qwen one-shot, every-split, bytewise, fuzz, and canonical-raw behavior must remain unchanged after lifecycle extraction (Task 4 existing parser suites).
- Hook history must remain committed-only and must become a tuple only when accessed (Task 5 lazy-materialization regression plus existing hook-state tests).
- Context-reserve resume must retain its initial sync, one assistant segment/prefix, remaining budget, and repeat-pause behavior (Task 1 and final `test_context_reserve.py` verification; finding 5 remains out of scope).

---

### Task 1: Resolve Stop-Sequence Control Before Constructing Jobs

**Files:**
- Modify: `src/ds4py/engine/pause.py`
- Modify: `src/ds4py/engine/decode.py`
- Modify: `src/ds4py/engine/chat.py`
- Modify: `src/ds4py/engine/session.py`
- Test: `tests/engine/test_stop_sequences.py`
- Test: `tests/engine/test_session.py`
- Test: `tests/engine/test_decode_tools.py`
- Test: `tests/engine/test_multi_token_step.py`
- Test: `tests/engine/test_logits.py`

**Interfaces:**
- Produces: `resolve_control(current: ControlSpan | None, sequences: Iterable[bytes]) -> ControlSpan`.
- Produces: `GenerateJob(..., control: ControlSpan = field(default_factory=ControlSpan), resume: bool = False)`; removes `stop_sequences` and all control-replacement reconciliation.
- Requires: a job with a stream receives the identical `stream.control` object or raises `ValueError` before generation.

- [ ] **Step 1: Write failing public and constructor regressions**

Add raw and chat tests that stop on `b"stop"`, call the next `run_round(stop_sequences=[b"other"])`, and require `ValueError("stop_sequences cannot change while resuming a round")`. Assert no new token was sampled and a retry with `b"stop"` still resumes successfully. Add a direct test that mismatched `GenerateJob(stream=stream, control=other)` is rejected instead of mutating `stream.control`.

- [ ] **Step 2: Run the regressions and verify RED**

Run: `uv run pytest -q tests/engine/test_stop_sequences.py tests/engine/test_session.py::test_public_round_forwards_stop_sequences`

Expected: the new tests fail because changed sequences are ignored and an explicit job control currently replaces the stream controller.

- [ ] **Step 3: Implement single-controller resolution**

Make fresh raw/chat rounds construct one controller through `resolve_control(None, sequences)`. Make resumed rounds require canonical sequence equality and reuse the paused controller. Construct fresh `RoundStream(..., control=control)`, pass that same object to every continuation job, remove `GenerateJob.stop_sequences`, and retain only an identity guard for stream/job mismatches. Update direct test job construction to pass `control=stream.control` where needed and update the public forwarding test to inspect `job.control.sequences`. Document on `Session.run_round` that a paused round resumes with unchanged stop sequences.

- [ ] **Step 4: Run focused verification**

Run: `uv run pytest -q tests/engine/test_stop_sequences.py tests/engine/test_session.py tests/engine/test_context_reserve.py tests/engine/test_decode.py tests/engine/test_decode_tools.py`

Expected: PASS, including unchanged reserve resync behavior.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/engine/pause.py src/ds4py/engine/decode.py src/ds4py/engine/chat.py src/ds4py/engine/session.py tests/engine
git commit -m "fix(engine): resolve stop-sequence ownership before decode"
```

---

### Task 2: Remove Speculative Pause and EOS Replay State

**Files:**
- Modify: `src/ds4py/engine/pause.py`
- Modify: `src/ds4py/engine/stream.py`
- Modify: `src/ds4py/engine/decode.py`
- Test: `tests/engine/test_pause.py`
- Test: `tests/engine/test_stop_sequences.py`

**Interfaces:**
- Retains: `StopSequenceMatcher.feed/pending/active_sequences`, `ControlSpan.feed/stop/resume`, and `RoundStream.resume()`.
- Removes: `Pause`, `PausedAfterEos`, `Resume`, `StreamCheckpoint`, `TextRecord`, clone/inherit APIs, record/override state, mixed-case aliases, region-state storage with no production caller, EOS replay fields, and checkpoint text collection.

- [ ] **Step 1: Establish the behavior baseline**

Run: `uv run pytest -q tests/engine/test_pause.py tests/engine/test_stop_sequences.py tests/engine/test_decode.py`

Expected: PASS before the behavior-preserving cleanup. Do not add tests that assert dead APIs are absent; the Testing Value Gate does not justify source-shape tests.

- [ ] **Step 2: Delete unreachable state and retain the used matcher/resume path**

Remove EOS marking from stop-token acceptance and reduce stream resume to clearing the matched-stop state and finalization flag. Replace production `Feed` calls with `feed`. Delete tests whose only purpose is manually constructing unreachable EOS/checkpoint, clone/inherit, record, alias, or region-gating states; retain meaningful UTF-8, overlap, bounded-state, stop, and resume coverage.

- [ ] **Step 3: Verify no production behavior changed**

Run: `uv run pytest -q tests/engine/test_pause.py tests/engine/test_stop_sequences.py tests/engine/test_decode.py tests/engine/test_decode_tools.py tests/engine/test_context_reserve.py`

Expected: PASS without EOS/checkpoint-only tests.

- [ ] **Step 4: Commit**

```bash
git add src/ds4py/engine/pause.py src/ds4py/engine/stream.py src/ds4py/engine/decode.py tests/engine/test_pause.py tests/engine/test_stop_sequences.py
git commit -m "refactor(engine): remove speculative pause replay state"
```

---

### Task 3: Construct Parser Snapshots at Observation Boundaries

**Files:**
- Modify: `src/ds4py/engine/tool_syntax/base.py`
- Modify: `src/ds4py/engine/stream.py`
- Test: `tests/engine/test_decode_tools.py`

**Interfaces:**
- Retains: public `ToolParser.feed(data) -> Snapshot` and `finish() -> Snapshot`.
- Produces internally: `ToolParser._feed_byte(byte: int) -> None`; `RoundStream` calls `_snapshot()` once after consuming each observed token.

- [ ] **Step 1: Write the failing snapshot-boundary regression**

Start a real DSML stream through an open parameter value, instrument the real parser's `_snapshot`, then observe a three-byte value token. Assert one new snapshot and unchanged parser/display results. The test names the performance break it catches: per-byte whole-buffer snapshot copying.

- [ ] **Step 2: Run the regression and verify RED**

Run: `uv run pytest -q tests/engine/test_decode_tools.py::test_stream_observe_snapshots_once_per_token`

Expected: FAIL with three snapshot constructions instead of one.

- [ ] **Step 3: Separate byte mutation from snapshot construction**

Move the existing per-byte feed transition into `_feed_byte`; keep `feed` returning the same immutable snapshot. During control observation, read `parser.state` and `parser.in_think` for byte transitions, then refresh `RoundStream.snapshot` once after the token's bytes are consumed.

- [ ] **Step 4: Run parser and decode verification**

Run: `uv run pytest -q tests/engine/tool_syntax tests/engine/test_decode_tools.py tests/engine/test_logits.py tests/engine/test_stop_sequences.py`

Expected: PASS, including every split, bytewise input, greedy sampling, and hook parser state.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/engine/tool_syntax/base.py src/ds4py/engine/stream.py tests/engine/test_decode_tools.py
git commit -m "perf(engine): snapshot parsers at token boundaries"
```

---

### Task 4: Share the GLM/Qwen Delimiter Lifecycle

**Files:**
- Create: `src/ds4py/engine/tool_syntax/delimiter.py`
- Modify: `src/ds4py/engine/tool_syntax/glm.py`
- Modify: `src/ds4py/engine/tool_syntax/qwen.py`
- Test: existing `tests/engine/tool_syntax/`

**Interfaces:**
- Produces internally: `DelimiterParser(ToolParser)`, owning buffer/call lifecycle, common parse loop, decoding, call publication, after-call handling, and finalization.
- Requires subclasses to define `_call_close`, `_parameter_close`, `_family`, `_parse_call()`, `_parse_parameter()`, and `_parse_parameter_value()`; Qwen alone overrides compound function-close handling.

- [ ] **Step 1: Establish the parser behavior baseline**

Run: `uv run pytest -q tests/engine/tool_syntax`

Expected: PASS. This is a behavior-preserving extraction, so no source-shape or duplicate-inheritance tests are added.

- [ ] **Step 2: Extract the shared lifecycle without changing DSML**

Move only duplicated GLM/Qwen machinery into `DelimiterParser`. Keep GLM name-first/string parsing and Qwen function/parameter/JSON inference in their family modules. Preserve error messages, canonical raw boundaries, trailing-space completion, and parser terminal behavior.

- [ ] **Step 3: Run complete parser verification**

Run: `uv run pytest -q tests/engine/tool_syntax tests/engine/test_decode_tools.py tests/engine/test_tool_prompt.py`

Expected: PASS for all fixtures, every split, bytewise/random chunks, raw stanzas, and public tool-call events.

- [ ] **Step 4: Commit**

```bash
git add src/ds4py/engine/tool_syntax/delimiter.py src/ds4py/engine/tool_syntax/glm.py src/ds4py/engine/tool_syntax/qwen.py
git commit -m "refactor(engine): share delimiter parser lifecycle"
```

---

### Task 5: Materialize Sampled-Token Tuples Lazily

**Files:**
- Modify: `src/ds4py/engine/logits.py`
- Modify: `src/ds4py/engine/decode.py`
- Test: `tests/engine/test_logits.py`

**Interfaces:**
- Retains: `LogitsView.sampled_tokens -> tuple[int, ...]` as an immutable snapshot at access time.
- Changes internally: retain the committed sequence and a per-observation cached tuple; do not copy history in `_update` or when no hook reads the property.

- [ ] **Step 1: Write the failing lazy-materialization regression**

Use a real `LogitsView` and a `Sequence` that counts iteration. Call `_update` with that sequence and assert zero iterations; read `sampled_tokens` twice and assert the expected tuple with one iteration total. Keep the existing committed-token/history behavior tests unchanged.

- [ ] **Step 2: Run the regression and verify RED**

Run: `uv run pytest -q tests/engine/test_logits.py::test_sampled_tokens_materialize_only_on_access`

Expected: FAIL because `_update` eagerly iterates the sequence.

- [ ] **Step 3: Implement lazy snapshots and canonical history retention**

Store the source sequence in `LogitsView`, invalidate its tuple cache in `_update`, and materialize on first property access. For chat hooks, retain `stream.sampled_tokens` directly rather than copying it into a second list; append in `DecodeContext` only for raw rounds without a stream.

- [ ] **Step 4: Run hook and decode verification**

Run: `uv run pytest -q tests/engine/test_logits.py tests/engine/test_multi_token_step.py tests/engine/test_decode.py tests/engine/test_context_reserve.py`

Expected: PASS, including actual bias, committed-only history, multiple hooks, reserve continuation, and `PlainStep` fallback.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/engine/logits.py src/ds4py/engine/decode.py tests/engine/test_logits.py
git commit -m "perf(engine): lazily snapshot sampled token history"
```

---

## Final Verification

- [ ] Run `uv run pytest -m "not live and not build" -q`; require all selected tests to pass.
- [ ] Run the captured Qwen CUDA replay with the existing M2 live environment; require one passing replay.
- [ ] Run `git diff --check main...HEAD` and inspect `git status --short`; require a clean tracked tree.
- [ ] Report the accepted five repairs, the reason finding 5 remains unchanged, exact test results, and final commits without claiming unconfigured lint checks.
