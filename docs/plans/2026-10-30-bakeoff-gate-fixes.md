# Bake-off Gate Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the M3 bake-off use a practical default response budget, reject incomplete model responses, account for canonical tool events, and replay byte-arbitrary C traces.

**Architecture:** Keep each fix at its existing boundary: CLI configuration, `DS4StreamedResponse`, `summarize_events`, and `read_replay`. Do not add a continuation protocol or rewrite the C trace format. Restore and interruption fixes already exist in the M3 baseline.

**Tech Stack:** Python 3.14, Pydantic AI Harness 0.59.0, pytest, pyright, Ruff, uv.

**Spec:** `docs/specs/2026-10-30-bakeoff-gate-fixes-design.md`

## Global Constraints

- Work only in `.worktrees/m3-harness-agent` on `m3-harness-agent`.
- The default bake-off response budget is **8192 tokens for both agents**; an explicit `--tokens` value remains authoritative.
- `max_tokens` and `parser_error` must fail the affected agent response; never continue automatically after either terminal reason.
- Raw engine and `Session.run_round` behavior remains unchanged.
- Keep engine `ToolCalls` as the parsed-call source; pair canonical harness calls/results by `part.tool_call_id`.
- Preserve legacy `ToolResultEvent` behavior and all existing public signatures unless a task explicitly adds a type.
- Do not change the external C writer or claim live-model verification from fake-based tests.
- Commit each task separately after its verification passes.

## Review Focus

- An explicit low `--tokens` value must not be replaced by the new default; Task 1 checks this directly.
- A terminal failure after earlier tool side effects must retain an error run record and loadable snapshot; Task 2 runs the existing failed-run snapshot regression alongside its new tests.
- Canonical results may resolve calls in a different order; Task 3 proves ID pairing rather than positional pairing.
- Canonical retries, denials, unknown IDs, duplicates, and missing outcomes must not be invented as successful executions; Task 3 covers each.
- Valid existing C and ds4-py traces must remain readable; Task 4 runs the complete replay test module in addition to its invalid-byte fixture.

### Task 1: Shared Response Budget

**Files:**
- Modify: `bench/run_bakeoff.py:437-448`
- Modify: `bench/README.md:19-24,47-63`
- Verify: `tests/bench/test_runner.py`

**Interfaces:**
- Consumes: existing `_parser() -> argparse.ArgumentParser` and `AgentConfig.tokens`.
- Produces: default `--tokens=8192`; no new API.

- [ ] **Step 1: Verify the current default**

Run: `uv run --extra agent python -c 'from bench.run_bakeoff import _parser; print(_parser().parse_args(["--results", "r", "--model", "m", "--agent", "a"]).tokens)'`
Expected: `512`.

- [ ] **Step 2: Change the default and documentation**

Set the parser default to `8192`. Document that thinking consumes the per-response budget, both agents receive the same value, and an explicit larger value may still be needed. Do not add a test that merely restates static CLI configuration; the approved Testing Value Gate calls for direct verification.

- [ ] **Step 3: Verify default and explicit-budget behavior**

Run the Step 1 command again and a variant with `--tokens 2048`.
Expected: `8192` and `2048`, respectively.

Run: `uv run --extra agent pytest tests/bench/test_runner.py -q`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add bench/run_bakeoff.py bench/README.md
git commit -m "fix(bench): raise shared response budget"
```

### Task 2: Terminal Response Failures

**Files:**
- Modify: `src/ds4py/model.py:40-48,448-471`
- Test: `tests/agent/test_model.py:185-198`
- Test: `tests/agent/test_runtime.py`

**Interfaces:**
- Consumes: existing `Stopped.reason` values and `AgentRuntime.run` exception handling.
- Produces: `ResponseTerminalError(EngineError)` with public `reason: str`; `DS4StreamedResponse` raises it for `max_tokens` and `parser_error` after terminal dispatch, with normal `_finish()` cleanup.

- [ ] **Step 1: Write failing response and lifecycle tests**

Replace the two tests that currently accept `length`/`error` responses with tests asserting `ResponseTerminalError.reason == "max_tokens"` and `"parser_error"`. Use the existing partial-text and malformed-tool scripts.

Add `test_response_terminal_failures_have_error_records_and_nonzero_exit` in `tests/agent/test_runtime.py`, parameterized over those two cases. Drive a real `AgentRuntime` through `run_noninteractive` with `StringIO`; assert exit code `1`, the reason in stderr, `runtime.status == "error"`, and a final `run_end` record with `status == "error"`.

- [ ] **Step 2: Watch the new tests fail**

Run the two model tests and the new runtime test with `pytest -q`.
Expected: FAIL because partial/malformed responses return successfully instead of raising `ResponseTerminalError`.

- [ ] **Step 3: Implement terminal rejection**

Add `ResponseTerminalError(EngineError)` in `src/ds4py/model.py`. In `_get_event_iterator`, after dispatching a `Stopped` event and preserving the existing context-reserve branch, raise `ResponseTerminalError(event.reason)` when the reason is `max_tokens` or `parser_error`. Do not alter `_stopped` finish-reason mapping or other terminal reasons.

- [ ] **Step 4: Verify the fix and neighboring lifecycle behavior**

Run the focused model/runtime tests plus `test_failed_run_reports_error_and_keeps_a_loadable_snapshot`.
Expected: all pass; error records and snapshot recovery remain intact.

Run: `uv run --extra agent pytest -q`
Expected: full suite passes with no new warnings.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/model.py tests/agent/test_model.py tests/agent/test_runtime.py
git commit -m "fix(agent): reject incomplete model responses"
```

### Task 3: Canonical Tool Outcome Accounting

**Files:**
- Modify: `bench/metrics.py:88-174`
- Test: `tests/bench/test_metrics.py`

**Interfaces:**
- Consumes: `FunctionToolCallEvent.payload.part.tool_call_id` and `FunctionToolResultEvent.payload.part`, plus existing engine `ToolCalls` and legacy `ToolResultEvent` records.
- Produces: unchanged `summarize_events(events, *, wall_seconds) -> EventSummary`; canonical calls are registered and resolved by ID, with `MetricError` for inconsistent canonical evidence.

- [ ] **Step 1: Write failing canonical-event tests**

Add fixtures matching the live JSON shape. Cover:
- two calls resolved in reverse order, one `success` and one `denied`; expect `parsed == 2`, `executed == 1`, `refused == 1`, and `complete`;
- a `retry-prompt` result increments `retried`, not `executed`;
- duplicate calls, duplicate results, unknown result IDs, missing/empty IDs, and a registered call without a result each raise `MetricError`;
- existing legacy `ToolResultEvent` tests remain unchanged and green.

- [ ] **Step 2: Watch canonical tests fail**

Run: `uv run --extra agent pytest tests/bench/test_metrics.py -q`
Expected: FAIL because canonical outcomes remain unaccounted and malformed canonical evidence is not rejected.

- [ ] **Step 3: Implement ID-based accounting**

In `summarize_events`, validate canonical call/result parts and track registered versus resolved IDs. Share outcome classification with the legacy result path so `success`, `denied`, `failed`, `interrupted`, and `retry-prompt` retain one meaning. Do not increment `parsed` for `FunctionToolCallEvent`; engine `ToolCalls` owns that count. Raise `MetricError` for unresolved canonical calls after all records are processed.

- [ ] **Step 4: Verify synthetic and captured evidence**

Run: `uv run --extra agent pytest tests/bench/test_metrics.py -q`
Expected: all pass.

Parse the 18 captured ds4-py event streams under `/home/roche/projects/satyrn/ds4-py-results/tokens-2048` with `summarize_events` and a positive fixed wall time.
Expected: all 250 canonical calls have matching outcomes; no schema or pairing error; no parsed-call double counting.

Run: `uv run --extra agent pytest -q`
Expected: full suite passes with no new warnings.

- [ ] **Step 5: Commit**

```bash
git add bench/metrics.py tests/bench/test_metrics.py
git commit -m "fix(bench): account for canonical tool outcomes"
```

### Task 4: Byte-Safe C Trace Reading

**Files:**
- Modify: `src/ds4py/engine/replay.py:61-67,85-88`
- Test: `tests/engine/test_replay.py`
- Fixture: `tests/fixtures/replay/basic.trace` (read as bytes by the new test; do not rewrite the shared fixture)

**Interfaces:**
- Consumes: arbitrary bytes in C token display fields and existing decimal token IDs/metadata.
- Produces: unchanged `read_replay(path: str | Path) -> ReplayTrace`; lossless UTF-8 `surrogateescape` decoding at both existing read entry points.

- [ ] **Step 1: Write a failing byte-level replay test**

Create a temporary copy of `basic.trace` as bytes. Replace sampled token display text with the observed raw prefixes `b"\xe2\x99"`, `b"\xaf"`, `b"\xe1\xbb"`, and `b"\x9d"` without changing token IDs or structural newlines. Assert `read_replay` succeeds and every round's `prompt` and `sampled` IDs exactly match those from `read(BASIC)`.

- [ ] **Step 2: Watch the fixture fail**

Run the new test with `pytest -q`.
Expected: FAIL with `UnicodeDecodeError` under strict UTF-8 decoding.

- [ ] **Step 3: Implement lossless decoding**

At both `read_replay` and `_Reader.read` file-entry points, use `path.read_bytes().decode("utf-8", errors="surrogateescape")`. Keep the `str` parser and all replay validation unchanged.

- [ ] **Step 4: Verify replay compatibility**

Run: `uv run --extra agent pytest tests/engine/test_replay.py -q`
Expected: all pass, including the invalid-byte fixture and all existing valid traces.

Read the two captured traces named in the report and compare their replay IDs with a byte-level invocation of `read_replay`.
Expected: both parse without `UnicodeDecodeError`; no replacement-character conversion is used.

Run: `uv run --extra agent pytest -q`
Expected: full suite passes with no new warnings.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/engine/replay.py tests/engine/test_replay.py
git commit -m "fix(replay): read arbitrary C token bytes"
```

## Final Verification and Review

- [ ] Run `uv run --extra agent pytest -m "not build and not live" -q`.
- [ ] Run `uv run --extra agent pyright`.
- [ ] Run `uv run --extra agent ruff check src bench tests`.
- [ ] Run `git diff --check` against the M3 starting commit and inspect the complete diff.
- [ ] Dispatch one fresh-context whole-branch reviewer with the spec, plan, review focus, diff, and ledger rulings.
- [ ] Keep the performance todo open pending a clean live bake-off; do not present local tests as live gate evidence.
