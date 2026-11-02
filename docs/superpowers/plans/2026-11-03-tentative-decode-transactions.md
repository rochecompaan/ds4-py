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
- Cancellation, mismatch, stop-before, and abandoned blocks must discard every uncommitted token/event without losing the canonical prefix; Tasks 2 and 3 pin transcript, stream, sampled history, events, and native position.

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

In `test_stream_state.py`, parameterize `Dsml(v41=False)`, `Dsml(v41=True)`, `Glm()`, and `Qwen()`. At partial opener, structural, and parameter-value boundaries, assert that a fork fed the remaining chunks has the same `Snapshot` and `wants_greedy_sampling()` as an uninterrupted parser. Feed a different suffix to the original and assert that the fork is unchanged. Include a completed call followed by a partial next call to cover `_calls` and delimiter state.

- [ ] **Step 2: Run parser tests to verify RED**

Run: `uv run --frozen --extra agent pytest tests/engine/test_stream_state.py -q`
Expected: FAIL because the capture/fork/restore methods and state types do not exist.

- [ ] **Step 3: Implement complete parser state capture**

Define an immutable `ToolParserState` containing all base mutable fields: public state/raw position/error/flags, calls, active flag, pending opener bytes, marker tail, and one family-specific immutable value. `Dsml` captures its call/name/args/parameter positions and string flag; `DelimiterParser` captures its after-call/call/name/args/parameter positions. Restore replaces containers rather than mutating captured ones. `fork()` uses the same concrete constructor (`Dsml` preserves `v41`) and restores the complete state; recompute the public `Snapshot` from restored parser state.

- [ ] **Step 4: Add failing decoder, control, and round-stream fork tests**

For `Decoder`, split a multibyte UTF-8 codepoint and `<think>`/`</think>` markers around capture and assert identical `TextDelta` sequences and finalization. For `ControlSpan`, cover overlapping partial stop sequences and a matched stop followed by `resume()`. For `RoundStream`, assert that `sampled_tokens`, parser snapshot, think count, last published state, pending control bytes, finalization, and call-emitted flag match an uninterrupted reference while the source remains unchanged.

- [ ] **Step 5: Implement immutable component and aggregate states**

`DecoderState` includes `thinking`, pending marker bytes, and the incremental UTF-8 decoder's `getstate()` value. `ControlSpanState` includes matcher states, matched stop, and active sequences; configured sequences/tables are immutable and shared. `RoundStreamState` aggregates those states plus sampled IDs, think count, finalized flag, control pending bytes, last state/thinking, and call-emitted flag. Restore derives `RoundStream.snapshot` from the restored parser instead of storing a second parser-state copy.

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

Build real `DecodeContext` instances with the existing scripted native doubles. For raw and chat, begin staging and stage two tokens. Assert that transcript, canonical stream/control/sampled history, and emitted events remain at their prefixes while `before_sample()` and `sampling()` observe the staged suffix. Call `publish()` and assert the exact tokens/events appear once in their original order. On a second block, call `discard()` and assert no state or event changes. Also assert nested `begin_staging()` and `stage()`/`publish()`/`discard()` without an active block raise `EngineError` before native or canonical mutation.

- [ ] **Step 2: Run transaction tests to verify RED**

Run: `uv run --frozen --extra agent pytest tests/engine/test_tentative_decode.py -q`
Expected: FAIL because the staging API does not exist.

- [ ] **Step 3: Implement the private overlay and staged reads**

Add a private `_DecodeOverlay` holding staged token IDs, buffered events, forked `RoundStream | None`, forked raw `ControlSpan`, copied sampled IDs, and the block-start accepted/terminal/verdict state. Route `_publish()`, `before_sample()`, and `sampling()` through the overlay only while active. `begin_staging()` forks state without allocating anything for direct `PlainStep` calls. Retain the existing `buffer_events()`/`flush_events()` test-double contract until Task 3 converts it.

- [ ] **Step 4: Implement stage, publish, and discard without changing direct commit**

Extract the shared token verdict/terminal handling without changing `PlainStep.commit()` behavior. `stage()` updates only overlay state and buffers `Token`, thinking, text, syntax, and control results. `publish()` appends/extends the exact staged token prefix, restores canonical stream/control/sampled history from the overlay, emits buffered events in order, and clears the overlay while retaining the resulting verdict and terminal reason. `discard()` clears the overlay and restores block-start counters/verdict/terminal state. Both leave native eval/rewind to the caller.

- [ ] **Step 5: Pin terminal finalization and abandoned-block cleanup**

Add tests where a staged token selects `max_tokens`, `stop_sequence`, `tool_call_complete`, and `parser_error`. Before `publish()`, assert no events or canonical terminal state; after native commit and `publish()`, assert `finalize()` produces the same ordered events and reason as direct serial decode. A step that returns with an active overlay must make `generate()` discard it and raise `EngineError("decode returned with an active staging block")`, with no canonical or emitted changes.

- [ ] **Step 6: Verify direct parity and commit**

Run `pytest tests/engine/test_tentative_decode.py tests/engine/test_decode.py tests/engine/test_stop_sequences.py tests/engine/test_logits.py -q`. Expected: all pass, including unchanged direct/default behavior. Commit `feat: add tentative decode transactions`.

### Task 3: Multi-token integration, family parity, and final gates

**Files:**
- Modify: `tests/engine/fake_steps.py`
- Modify: `tests/engine/test_multi_token_step.py`
- Modify: `tests/engine/test_tentative_decode.py`
- Test: existing `tests/engine/**`

**Interfaces:**
- Consumes: Task 2's four staging methods and `staging` property.
- Produces: FakeMultiTokenStep coverage of the same publish/discard contract required from downstream multi-token steps; no new production API.

- [ ] **Step 1: Write failing multi-token semantic tests**

Add tests proving that a Qwen opener in one staged row changes the next row's `SamplingPolicy` to `temperature=0.0, top_k=0, top_p=1.0, min_p=0.0`; a hook sees exact canonical-plus-staged metadata for every verify row; and a terminal staged token emits its complete event sequence only after native commit. Parameterize serial-versus-staged parity over DSML, DSML v4.1, GLM, and Qwen, with chunk-boundary, think/text, stop-sequence, malformed-parser, and tool-call cases. Compare token IDs, transcript, `TextDelta`, `Thinking`, `ToolSyntax`, `ToolCalls`, terminal reason, sampled history, RNG state, and native position.

- [ ] **Step 2: Run semantic tests to verify RED**

Run the named new tests with pytest. Expected: current block buffering fails staged metadata/sampling and publish-order assertions.

- [ ] **Step 3: Convert FakeMultiTokenStep to the transaction contract**

Wrap each block in `begin_staging()`. Use `stage()` for each accepted row. After native success, rewind any rejected native suffix and `publish()` exactly the kept prefix; on cancellation, stop-before, mismatch discard, hook failure, or native failure, rewind as required and `discard()`. Remove the superseded `buffer_events()`/`flush_events()` test-double calls. Do not catch or translate native errors in `DecodeContext`, and do not add fallback to `PlainStep` after a block starts.

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
