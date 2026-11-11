# Tentative Decode Transactions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic per-block staging transaction that lets multi-token decode steps observe canonical plus tentative Python state while publishing only the natively committed prefix.

**Architecture:** Add explicit immutable capture and restore state for every streaming component, then fork that state into a block-local overlay owned by `DecodeContext`. Keep `PlainStep.commit()` on its current direct path; multi-token steps use `begin_staging()`, `stage()`, `publish()`, and `discard()` around the existing native eval/rewind operations.

**Tech Stack:** Python 3.14+, pytest, existing ds4-py engine and FakeNative/FakeMultiTokenStep test infrastructure.

**Spec:** [satyrn-agent agent package design, sections 8.2 and 8.4](https://github.com/satyrn-ai/satyrn-agent/blob/docs/agent-package-spec/docs/superpowers/specs/2026-10-29-agent-package-design.md#82-tentative-decode-transactions)

## Global Constraints

- Implement only section 8.2; do not change native structs/bindings or add DFlash/Satyrn identifiers or policy to ds4-py.
- Only the existing worker may call native functions. The decode step retains responsibility for native eval, rewind, and commit ordering.
- Never mutate or publish canonical transcript, stream, parser, decoder, control, or sampled-history state while staging.
- Use explicit complete state objects; do not use ad hoc `copy.copy()` or `copy.deepcopy()` of selected object fields.
- Keep `PlainStep` direct and byte-for-byte compatible, with no staging allocation.
- Write each focused test first, watch it fail for the missing behavior, implement the smallest change, and rerun it.
- Run all project commands inside `devenv shell`; use `uv run --frozen --extra agent` and the isolated-HOME portable-suite convention from `AGENTS.md`.

## Review Focus

- Split UTF-8 codepoints, think markers, tool openers, and parameter delimiters must survive a fork without duplicate or missing display/control bytes; Task 1 pins each with interrupted-reference comparisons.
- A staged Qwen tool opener must make the next verify row structurally greedy before that row is committed; Task 3 pins the sampling policy seen by the fake native session.
- A metadata-sensitive hook must see exact tentative position, parser state, think count, and sampled IDs for every row; Task 3 pins the complete sequence.
- A terminal token inside a staged block must publish its token and all parser/display/tool events only after native commit; Tasks 2 and 3 pin ordering and native/transcript positions.
- Cancellation, mismatch, stop-before, hook/native failure, and abandoned blocks must publish exactly the valid kept prefix or discard the whole overlay as appropriate; Tasks 2 and 3 pin transcript, stream, sampled history, events, exceptions, and native position.

---

### Task 1: Complete streaming-state capture and fork primitives

**Files:**
- Modify: `src/ds4py/engine/tool_syntax/base.py`
- Modify: `src/ds4py/engine/tool_syntax/dsml.py`
- Modify: `src/ds4py/engine/tool_syntax/delimiter.py`
- Modify: `src/ds4py/engine/decoder.py`
- Modify: `src/ds4py/engine/pause.py`
- Modify: `src/ds4py/engine/stream.py`
- Test: `tests/engine/test_stream_state.py`
- Test: existing `tests/engine/tool_syntax/**`

**Interfaces:**
- Consumes: existing `ToolParser`, `Decoder`, `ControlSpan`, and `RoundStream` mutable state.
- Produces:
  - `ToolParser.capture() -> ToolParserState`, `restore(state: ToolParserState) -> None`, and `fork() -> ToolParser`.
  - `Decoder.capture() -> DecoderState`, `restore(state: DecoderState) -> None`, and `fork() -> Decoder`.
  - `ControlSpan.capture() -> ControlSpanState`, `restore(state: ControlSpanState) -> None`, and `fork() -> ControlSpan`.
  - `RoundStream.capture() -> RoundStreamState`, `restore(state: RoundStreamState) -> None`, and `fork() -> RoundStream`.

- [ ] **Step 1: Write failing parser-fork tests**

In `test_stream_state.py`, parameterize `Dsml(v41=False)`, `Dsml(v41=True)`, `Glm()`, and `Qwen()`. At partial opener, structural, and parameter-value boundaries, assert that a fork fed the remaining chunks has the same `Snapshot` and `wants_greedy_sampling()` as an uninterrupted parser. Feed a different suffix to the original, restore the captured state, and assert that it again matches the uninterrupted reference. Include a completed call followed by a partial next call to cover `_calls` and delimiter state, plus nested JSON arguments to prove fork/restore independence.

- [ ] **Step 2: Run parser tests to verify RED**

Run: `uv run --frozen --extra agent pytest tests/engine/test_stream_state.py -q`
Expected: FAIL because the capture/fork/restore methods and state types do not exist.

- [ ] **Step 3: Implement complete parser state capture**

Define an immutable `ToolParserState` containing all base mutable fields: public state/raw bytes/parse position/error/flags, calls, active flag, pending opener bytes, marker tail, and one family-specific immutable value. `Dsml` captures its call/name/args/parameter positions and string flag; `DelimiterParser` captures its after-call/call/name/args/parameter positions. Add explicit freeze/thaw helpers for nested list/dict/scalar values in both in-progress `ToolArgument` values and completed `ToolCall` arguments, so no mutable JSON value is shared by a snapshot and a parser. Restore installs thawed containers rather than mutating captured ones. `fork()` uses the same concrete constructor (`Dsml` preserves `v41`) and restores the complete state; recompute the public `Snapshot` from restored parser state.

- [ ] **Step 4: Add failing decoder, control, and round-stream fork tests**

For every component, capture a partial state, mutate the original through a distinguishable suffix, restore it, and compare subsequent behavior and finalization with an uninterrupted reference. For `Decoder`, split a multibyte UTF-8 codepoint and `<think>`/`</think>` markers around capture and assert identical `TextDelta` sequences. For `ControlSpan`, cover overlapping partial stop sequences and a matched stop followed by `resume()`. For `RoundStream`, assert that `sampled_tokens`, parser snapshot, think count, last published state, pending control bytes, finalization, and call-emitted flag match the reference while the source remains unchanged. Parameterize round-stream forks over every parser-family flag.

- [ ] **Step 5: Implement immutable component and aggregate states**

`DecoderState` includes `thinking`, pending marker bytes, and the incremental UTF-8 decoder's `getstate()` value. `ControlSpanState` includes matcher states, matched stop, and active sequences; configured sequences/tables are immutable and shared. `RoundStreamState` aggregates those states plus sampled IDs, think count, finalized flag, control pending bytes, last state/thinking, and call-emitted flag. `RoundStream.fork()` uses a private `__new__` construction path that forks the existing parser/decoder/control directly rather than calling an `__init__` that lacks the original engine info. Restore derives `RoundStream.snapshot` from the restored parser, preserves the canonical control-object and sampled-list identities, and slice-replaces list contents instead of rebinding them.

- [ ] **Step 6: Verify and commit the state primitives**

Run the new test plus `pytest tests/engine/tool_syntax tests/engine/test_stop_sequences.py -q`. Expected: all pass. Commit only Task 1 files with `feat: add streaming state fork primitives`.

### Task 2: Block-local DecodeContext staging transactions

**Files:**
- Modify: `src/ds4py/engine/decode.py`
- Test: `tests/engine/test_tentative_decode.py`

**Interfaces:**
- Consumes: Task 1 `fork()`/`capture()`/`restore()` methods and existing `Verdict`, `DecodeContext.accept()`, `commit()`, and `finalize()`.
- Produces:
  - `DecodeContext.begin_staging() -> None`
  - `DecodeContext.stage(token: int, verdict: Verdict | None = None) -> Verdict`
  - `DecodeContext.publish() -> None`
  - `DecodeContext.discard() -> None`
  - `DecodeContext.staging: bool` as a read-only activity property.

- [ ] **Step 1: Write failing raw and chat transaction tests**

Build real `DecodeContext` instances with the existing scripted native doubles. For raw and chat, begin staging and stage two tokens. Assert that transcript, canonical stream/control/sampled history, and emitted events remain at their prefixes while `before_sample()` and `sampling()` observe the staged suffix. Call `publish()` and assert the exact tokens/events appear once in their original order, canonical list/object identities remain intact, and a later hook sees prior plus newly published IDs exactly once. On a second block, call `discard()` and assert no state or event changes. Assert nested `begin_staging()` and inactive `stage()`/`publish()`/`discard()` raise `EngineError`; while staging, direct `commit()`, `resume()`, and `finalize()` also raise before any native/canonical mutation.

- [ ] **Step 2: Run transaction tests to verify RED**

Run: `uv run --frozen --extra agent pytest tests/engine/test_tentative_decode.py -q`
Expected: FAIL because the staging API does not exist.

- [ ] **Step 3: Implement the private overlay and staged reads**

Add a private `_DecodeOverlay` holding staged token IDs, buffered events, `block_position=session.pos()`, forked `RoundStream | None`, forked raw `ControlSpan`, a copied raw sampled-ID list only when no stream exists, and block-start `_accepted`, `reason`, `finalized`, `_finish_parser`, and `_verdict` values; `_committed` is not captured because staging never changes it. Chat sampled history is single-sourced by `overlay.stream.sampled_tokens`. Route `_publish()` and `sampling()` through the overlay only while active. During staging, `before_sample()` reports `position=overlay.block_position + len(overlay.token_ids)` and the overlay's parser/think/sampled metadata, never the changing live native position. Pin both stage-before-eval and eval-before-stage doubles. `begin_staging()` forks state without allocating anything for direct `PlainStep` calls. Retain the existing `buffer_events()`/`flush_events()` test-double contract until Task 3 converts it.

- [ ] **Step 4: Implement stage, publish, and discard without changing direct commit**

Extract shared token verdict/terminal handling without changing `PlainStep.commit()` behavior; `stage()` reuses the exact `finish()` calls, including `finish_parser=False` for tool completion and the existing `None` behavior for max tokens. `stage()` updates only overlay state, buffers `Token`/thinking/text/syntax events, and detects control/stop results. `publish()` accepts an empty kept prefix and applies the same resume-aware first-append/subsequent-extend and `_committed` logic as direct commit. It uses private direct-state helpers and restores the existing canonical stream/control objects in place, slice-replaces the existing sampled-ID lists, preserves `job.stream.control is job.control`, emits buffered events in order, and clears the overlay while retaining the resulting verdict and terminal state. `discard()` clears the overlay and restores block-start counters/verdict/terminal state. Both leave native eval/rewind to the caller.

- [ ] **Step 5: Pin terminal finalization and abandoned-block cleanup**

Add tests where a staged token selects `max_tokens`, `stop_sequence`, `tool_call_complete`, and `parser_error`. Before `publish()`, assert no events or canonical terminal state; after native commit and `publish()`, assert `finalize()` produces the same ordered events, `_finish_parser` behavior, and reason as direct serial decode. Assert the complete sample/native call log so no later row is sampled, staged, evaluated, or selected after a terminal verdict. A step that returns with an active overlay must make `generate()` discard it and raise `EngineError("decode returned with an active staging block")`, with no canonical or emitted changes.

- [ ] **Step 6: Verify direct parity and commit**

Run `pytest tests/engine/test_tentative_decode.py tests/engine/test_decode.py tests/engine/test_stop_sequences.py tests/engine/test_logits.py -q`. Expected: all pass, including unchanged direct/default behavior. Commit `feat: add tentative decode transactions`.

### Task 3: Multi-token integration, family parity, and final gates

**Files:**
- Modify: `src/ds4py/engine/decode.py`
- Modify: `tests/engine/fake_steps.py`
- Modify: `tests/engine/test_multi_token_step.py`
- Modify: `tests/engine/test_tentative_decode.py`
- Test: existing `tests/engine/**`

**Interfaces:**
- Consumes: Task 2's four staging methods and `staging` property.
- Produces: FakeMultiTokenStep coverage of the same publish/discard contract required from downstream multi-token steps; no new production API.

- [ ] **Step 1: Write failing multi-token semantic tests**

Add tests proving that a Qwen opener in one staged row changes the next row's `SamplingPolicy` to `temperature=0.0, top_k=0, top_p=1.0, min_p=0.0`; a hook sees `position == base + staged_count` and exact parser/think/sampled metadata for every verify row; and a terminal staged token emits its complete event sequence only after native commit, with no later-row operations. Parameterize serial-versus-staged parity over DSML, DSML v4.1, GLM, and Qwen, with chunk-boundary, think/text, stop-sequence, malformed-parser, and tool-call cases. Compare token IDs, transcript/segments, all event types and order, terminal reason, sampled history, and native position. Compare RNG state through the hooked verify path where every row is sampled; anchor-plus-scripted-draft blocks instead assert the explicitly derived anchor-only RNG state. RNG rollback remains decode-step-owned and is not added to streaming-state capture.

- [ ] **Step 2: Run semantic tests to verify RED**

Run the named new tests with pytest. Expected: current block buffering fails staged metadata/sampling and publish-order assertions.

- [ ] **Step 3: Convert FakeMultiTokenStep to the transaction contract**

Wrap each block in `begin_staging()` and use `stage()` for each accepted row. On `STOP_BEFORE`, do not evaluate/stage the stop token; rewind native state to the accepted prefix and `publish()` that prefix, including an empty prefix, retaining `stop_token`. On cancellation or hook failure after an accepted prefix advances native state, commit/rewind to that prefix, publish it, then return interrupted or re-raise. On mismatch, publish the accepted prefix and preserve the existing sampled-replacement semantics. Use `discard()` only when abandoning the entire overlay before native advancement or after rewinding the whole block. For native eval/commit or rewind/cleanup failure with uncertain state, propagate the exact error without publishing uncertain tokens or permitting another native operation on that state. Add fault-injection tests for hook failure before/after native advancement, native failure before/after a partial side effect, and rewind failure; assert the kept/discarded prefix, events, sampled history, transcript/native alignment where usable, and exception identity. Remove `buffer_events()`, `flush_events()`, and `_event_buffer` along with their superseded fake-step calls. Do not add fallback to `PlainStep` after a block starts.

- [ ] **Step 4: Verify focused and complete regression suites**

Run:

```bash
uv run --frozen --extra agent pytest tests/engine/test_tentative_decode.py tests/engine/test_multi_token_step.py tests/engine/test_decode.py tests/engine/test_stop_sequences.py tests/engine/test_logits.py -q
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest -m "not build and not live" -q
git diff --check
```

Expected: all focused tests, static checks, and the complete portable suite pass; the diff contains only section 8.2 files. Record counts and any pre-existing skips/warnings without treating them as new failures.

- [ ] **Step 5: Commit and hand off for real-engine acceptance**

Commit `test: exercise tentative decode parity` if only test infrastructure changed; otherwise include the minimal production correction justified by the preceding RED test. Request independent review, fix valid findings through RED-GREEN, and keep the branch available for the downstream patched-libds4/real-GGUF acceptance suite. Do not claim live parity from FakeNative results.
