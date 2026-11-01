# M3 Review Lifecycle, Restore, and Benchmark Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all seven whole-branch review findings without weakening lifecycle, restore, telemetry, or hidden-grading contracts.

**Architecture:** Make existing owners authoritative: the stream context closes its native round, save provenance carries rendered notes, canonical history contains the bound request view, the runtime publishes harness events, typed benchmark models decode reports, graders stay outside agent workspaces, and session guards admit public operations.

**Tech Stack:** Python 3.14, pytest, pydantic-ai-harness, stdlib benchmark tooling, Ruff, Pyright.

**Spec:** `docs/specs/2026-10-21-ds4-py-design.md` plus the seven accepted findings in the 2026-11-01 whole-branch review.

## Global Constraints

- Work only in `.worktrees/m3-harness-agent` on `m3-harness-agent`.
- Write and watch each behavioral regression fail before changing production code.
- Preserve existing public imports where practical; do not require a live model for verification.
- Existing KV saves remain loadable; old rendered saves that lack note text must not silently invent different note content.
- Restore must retain canonical messages that are absent from the bound model-request view.
- Restore rechecks default to 8,192 tokens and request limit 100, with explicit CLI overrides.
- Hidden test source and bytecode must never be mounted into the mutable agent workspace.
- Commit each finding separately after its focused tests pass; run the complete non-live suite after implementation.

## Review Focus

- Exiting a stream before iteration, or while retaining its iterator, must release the native round; Task 1 tests a following request.
- KV restore and stripped rendering must use identical note bytes; Task 2 compares token prefixes under a later restore clock.
- Bound keys may be a non-contiguous ordered subsequence of canonical history; Task 3 includes omitted middle history.
- Harness events must be published and recorded exactly once without feedback through the recorder; Task 4 checks subscribers and JSONL.
- Agent-controlled startup files and normal bytecode caching must not bypass grading or leak tests; Task 6 runs real pytest.
- Every public mutation or generation entry point must reject a closed session before touching native state; Task 7 parameterizes those entry points.

---

### Task 1: Stream context owns native-round cleanup

**Files:**
- Modify: `src/ds4py/model.py`
- Test: `tests/agent/test_model_stream.py` or `tests/agent/test_model_cancel.py`

**Interfaces:** `_open_stream()` always awaits the existing idempotent `DS4StreamedResponse._finish()` on context exit.

- [ ] Add a regression that consumes one event, retains the iterator, exits `request_stream()`, then performs another request or session mutation.
- [ ] Run the test and confirm it fails with `EngineError: a round is open`.
- [ ] Add `try/finally` around the context yield and await `_finish()`; do not mark normally completed responses interrupted.
- [ ] Run focused model stream/cancellation tests and commit `fix(agent): close native rounds at stream context exit`.

### Task 2: Persist exact note text and placement

**Files:**
- Modify: `src/ds4py/store.py`, `src/ds4py/engine/chat.py`, `src/ds4py/binding.py`, `src/ds4py/runtime.py`, `src/ds4py/runtime_restore.py`
- Test: `tests/agent/test_restore.py`, focused store/system-note tests as needed

**Interfaces:** Note provenance carries optional text for legacy KV reads; new saves always include note text. Rendered restore schedules saved notes at their original message boundaries instead of clearing note state and regenerating them.

- [ ] Add a later-clock stripped-restore regression asserting the rebuilt saved prefix equals the original tokens and the date note is not regenerated.
- [ ] Confirm the current render path diverges; add save-format coverage proving note text round-trips.
- [ ] Persist note text in transcript/save provenance, restore the saved note schedule for rendering, and reject rendered legacy note records whose exact text is unavailable.
- [ ] Run restore, binding, store, and system-note tests and commit `fix(restore): persist rendered system notes`.

### Task 3: Restore a bound view within canonical history

**Files:**
- Modify: `src/ds4py/binding.py`
- Test: `tests/agent/test_binding.py`, `tests/agent/test_restore.py`

**Interfaces:** Restore matches ordered bound keys to a subsequence of canonical snapshot messages and binds only those matched messages; runtime canonical history remains complete.

- [ ] Add a save/restore regression with four canonical messages and two non-contiguous bound messages.
- [ ] Confirm validation fails on the count mismatch.
- [ ] Replace positional/equal-count matching with ordered key lookup and content-hash validation; reject missing, duplicate, or reordered bound keys.
- [ ] Run binding and restore tests and commit `fix(restore): separate canonical history from bound views`.

### Task 4: Runtime owns harness-event publication

**Files:**
- Modify: `src/ds4py/runtime.py`, `src/ds4py/capabilities/run_record.py`
- Test: `tests/agent/test_footer.py`, `tests/agent/test_run_record.py`

**Interfaces:** A runtime harness-event bridge publishes `agent_event` and shell-service events. `DS4RunRecord` subscribes and writes those events without emitting them.

- [ ] Add regressions proving event subscribers receive shell and agent events without recorder publication and the recorder writes one copy from the bus.
- [ ] Confirm the current tests show recorder-owned publication or fail the new ownership assertions.
- [ ] Move publication into the runtime bridge, install it for every run, and remove recorder emit/filter feedback while preserving JSONL event kinds and run metadata.
- [ ] Run footer, run-record, runtime, and shell-job tests and commit `fix(runtime): own harness event publication`.

### Task 5: Explicit typed benchmark report contracts

**Files:**
- Create: `bench/models.py`
- Modify: `bench/run_bakeoff.py`, `bench/metrics.py`, `bench/report.py`, `bench/restore_check.py`, `bench/replay_c_traces.py`
- Test: `tests/bench/test_report_commands.py`, existing runner/metrics tests

**Interfaces:** Dependency-neutral dataclasses own `AgentConfig`, `EventSummary`, `ToolAccounting`, and `TaskRunResult.from_dict()`. Existing imports may re-export them. Both commands deserialize actual task results and construct complete configs without runtime casts.

- [ ] Add command regressions using real report JSON; capture the deserialized tasks and restore settings, including tokens and request limit.
- [ ] Confirm the advertised check/replay paths raise `NameError` before doing their work.
- [ ] Extract the models, implement explicit nested deserialization, add `--tokens`/`--request-limit` recheck options, and remove report-shaped casts and `SimpleNamespace` contracts.
- [ ] Run benchmark tests and commit `fix(bench): deserialize reports through typed models`.

### Task 6: Hidden grading stays outside agent workspaces

**Files:**
- Modify: `bench/workspaces.py`
- Test: `tests/bench/test_workspaces.py`

**Interfaces:** `grade_workspace()` passes hidden paths under `grader_dir`, visible paths under `agent_dir`, and inserts explicit agent root/src import paths only after importing pytest.

- [ ] Add real-pytest regressions for direct grader paths and absence of hidden source/`__pycache__` in the agent workspace after grading and between attempts.
- [ ] Confirm current mounting fails the new assertions with normal bytecode caching.
- [ ] Remove test mounting and run hidden tests from grader-owned storage while preserving startup/conftest isolation.
- [ ] Run workspace and runner tests and commit `fix(bench): keep hidden grading artifacts out of workspaces`.

### Task 7: Shared closed-session admission guards

**Files:**
- Modify: `src/ds4py/engine/session.py`
- Test: focused engine session/lifecycle test module

**Interfaces:** Public mutation, generation, and checkpoint operations use `_require_idle()` (or `_require_open()` where active-round behavior is intentionally separate); cancellation and event subscription retain their documented semantics.

- [ ] Add parameterized after-close regressions for append, truncation, chat mutation, hook registration, generation, and checkpoint operations.
- [ ] Confirm current append/generation paths reach native work instead of raising `session is closed`.
- [ ] Replace selective active-round checks with the shared guards; audit all public methods rather than fixing only the reproduction.
- [ ] Run engine session/chat/scoped-state and agent model tests and commit `fix(engine): enforce closed session guards`.

### Final verification

- [ ] Run `uv run pytest -m 'not live and not build'` and require a clean pass.
- [ ] Run `uv run ruff check src bench tests` and `uv run pyright src bench tests` if these are the branch's established checks; otherwise run the repository equivalents.
- [ ] Review `git diff main...HEAD --stat`, per-fix commits, and `git status --short`; report verification and any live-only limits without claiming M3 live acceptance.
