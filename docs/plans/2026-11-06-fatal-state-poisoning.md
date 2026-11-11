# Fatal-State Poisoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a decode step permanently poison its owning worker after uncertain native failure while rejecting all later native-capable work and preserving exact exceptions and owner-driven cleanup.

**Architecture:** Inject a private worker poisoning callback through `generate()` into `DecodeContext`; `poison(error)` records the error and re-raises it unchanged. Gate both worker dispatch variants on the stored error, using public `WorkerPoisonedError` with explicit exception chaining, while leaving `_STOP` and existing `ExitStack` cleanup available.

**Tech Stack:** Python 3.14+, pytest, existing `EngineWorker`, `DecodeContext`, tentative overlays, and instrumented `FakeNative` doubles.

**Spec:** [Fatal-State Poisoning Design](../specs/2026-11-06-fatal-state-poisoning-design.md), implementing agent package design sections 8.2 and 8.5.

## Global Constraints

- Keep the existing worker as the only native owner; add no worker, thread, native struct, binding, or native API.
- Do not change direct `PlainStep`, allocate staging state for it, or fall back to it after a block starts or a native failure.
- Keep native eval, rewind, commit, and cleanup ordering in the decode step. After `poison()`, the supported step exit path and `generate()` perform only Python-local bookkeeping and overlay discard.
- Preserve exact initial exception identity and normal original/cleanup chaining. Do not publish uncertain staged tokens or events.
- `poison()` is terminal and single-use for a decode step. The worker gate is not a containment claim for deliberately non-cooperative code holding a borrowed raw handle.
- Reject queued and newly submitted ordinary calls/jobs before their callback or native handler; always complete their Future/event channel.
- `Engine.close()` and `_STOP` remain available. Lifecycle identity operations never clear poisoning.
- Follow strict RED→GREEN TDD before each runtime change and run all project commands from the worktree root inside `devenv shell` with frozen uv dependencies.
- Keep design and implementation in one branch/PR. Do not merge or push without explicit approval.

## Review Focus

- A native failure after a partial side effect must publish no staged prefix and must not trigger another native cleanup call from the poisoned job; Task 1 pins exception identity, native logs, and Python-only `finally` execution.
- Original plus cleanup failure must remain traceable from both the initial job and every later rejection; Tasks 1 and 2 pin `__context__`, `__cause__`, object identity, and distinct rejection wrappers.
- Work admitted before poisoning but dequeued afterward must never invoke its callback/handler and must finish its waiters; Task 2 controls queue order with events and bounded waits.
- `Session.new()` must retain its existing failed-transition ordering without recovery, while switch/fork identity changes must not clear poison; Task 2 pins IDs, events, scoped-state closure, and later rejection.
- Poisoned shutdown must close retained snapshots, session, and engine exactly once on the owner thread even when close follows rejected queued work; Tasks 2 and 3 pin resource logs and thread termination.

## PR Handoff

The PR description must explicitly ask Roché to review these design decisions:

1. Rejection is centralized at the `EngineWorker` dispatch boundary for both calls and submitted jobs, rather than duplicated across facades.
2. Later operations raise public `WorkerPoisonedError(EngineError)` with message `engine worker is poisoned`; its `__cause__` is the recorded fatal exception, preserving access to the original native error through normal cleanup-error chaining.

Do not create, push, or merge the PR without the repository owner's explicit approval.

---

### Task 1: Decode poisoning operation and abnormal multi-token exit

**Files:**
- Modify: `src/ds4py/engine/decode.py`
- Modify: `src/ds4py/engine/worker.py`
- Create: `tests/engine/test_fatal_state.py`

**Interfaces:**
- Consumes: existing `GenerateJob.decode_step`, `DecodeContext` staging, `generate()`, and worker job exception transfer.
- Produces:
  - `DecodeContext.poison(error: BaseException) -> Never`.
  - Internal `generate(..., poison: Callable[[BaseException], None] | None = None)` and matching optional `DecodeContext` callback for worker-free unit contexts.
  - Internal `EngineWorker._poison(error: BaseException) -> None` and stored fatal exception for Task 2.

- [ ] **Step 1: Write the failing abnormal-path tests**

Create a real worker-backed multi-token step that begins staging, accepts a token, and injects native `eval` failure before and after a partial side effect. On the uncertain path it calls `ctx.poison(error)`; its `finally` performs only Python bookkeeping. Assert the exact exception reaches the submitted job, the Python `finally` runs once, no token/transcript/event is published, no native eval/rewind/commit/cleanup occurs after poisoning, and the step never falls back to `PlainStep`.

Add a second case where native `rewind` cleanup raises while the original eval error is being handled and the cleanup error is passed to `poison()`. Assert the initial error is the exact cleanup object and `cleanup.__context__ is original`, with no canonical publication.

- [ ] **Step 2: Run the new tests to verify RED**

Run: `uv run --frozen --extra agent pytest tests/engine/test_fatal_state.py -q`
Expected: FAIL because `DecodeContext.poison` does not exist; no production code has been changed.

- [ ] **Step 3: Implement the minimal poisoning path**

Add the optional private callback to `DecodeContext` and `generate()`. `DecodeContext.poison(error)` invokes the callback when present and then raises the same `error` object. `EngineWorker` owns the recorded `BaseException | None`, passes `_poison` only for its `GenerateJob` execution, and records the supplied exception without native calls. Keep the existing `generate()` exception path limited to overlay discard and re-raise. Do not add rejection behavior yet.

- [ ] **Step 4: Verify GREEN and focused decode regressions**

Run: `uv run --frozen --extra agent pytest tests/engine/test_fatal_state.py tests/engine/test_tentative_decode.py tests/engine/test_multi_token_step.py tests/engine/test_decode.py -q`
Expected: PASS, including unchanged direct and tentative decode behavior.

- [ ] **Step 5: Commit the decode operation**

Commit the Task 1 files with `feat: add decode fatal-state poisoning`.

### Task 2: Permanent worker rejection, routing, lifecycle, and waiter completion

**Files:**
- Modify: `src/ds4py/errors.py`
- Modify: `src/ds4py/__init__.py`
- Modify: `src/ds4py/engine/worker.py`
- Modify: `tests/engine/test_fatal_state.py`

**Interfaces:**
- Consumes: Task 1's recorded worker fatal exception and poisoned multi-token fixture.
- Produces:
  - Public `WorkerPoisonedError(EngineError)`, exported from `ds4py`.
  - Worker dispatch rejection with exact message `engine worker is poisoned` and explicit `raise ... from recorded_error` for every ordinary `_Call` and `_Submission`.

- [ ] **Step 1: Write failing rejection and waiter tests**

After poisoning, parameterize direct `call`, `plan`, submitted `SyncJob`, `ResetJob`, `CheckpointJob` create/drop, `GenerateJob`, payload save/load, and callbacks that would otherwise eval, rewind, save/load a snapshot, or invoke an extension. Assert each callback/handler and native-double call log remains untouched. Assert `WorkerPoisonedError`, exact message, recorded `__cause__`, and distinct wrapper objects across two rejections.

For calls, exercise both `Future.result()` and `Future.exception()` with bounded waits. For submissions, assert iteration raises and `events.wait()` completes through a bounded helper wait. Include a blocked poisoning job with calls/jobs/close admitted behind it; release it and prove already queued work is rejected without execution and close still terminates.

- [ ] **Step 2: Write failing public-facade and lifecycle tests**

Poison through `Session.set_decode_step()` plus `run_round()`, then exercise public engine tokenization/token text, session `ctx_used`, `plan`, `set_think`, `reconcile_round`, later rounds, checkpoints, payload jobs, `on_worker`, and extension-callback routing. Assert every native-capable operation raises before entering the double.

Attempt `new()`, reset, switch with native restore, and fork. Assert no operation clears poison or selects plain decode. For failed `new()`, pin unchanged old ID, closed scoped state, emitted `session_end`, and absent `session_start`. Pure identity switch/fork may finish, but later native work receives the same poison contract.

- [ ] **Step 3: Run routing tests to verify RED**

Run: `uv run --frozen --extra agent pytest tests/engine/test_fatal_state.py -q`
Expected: FAIL because `WorkerPoisonedError` and dispatch rejection do not exist; Task 1 abnormal-path tests remain valid.

- [ ] **Step 4: Implement the central dispatch gate**

Add and export `WorkerPoisonedError`. In the worker loop, handle `_STOP` before the poison check. For `_Call`, retain `set_running_or_notify_cancel()` and check inside the existing `try` so the Future receives the error. For `_Submission`, check inside the existing `try/finally` so the error, `_END`, and `finished` are always published. Raise a fresh typed error from the recorded fatal exception. Do not add facade-specific guards or alter close methods.

- [ ] **Step 5: Verify GREEN and focused routing regressions**

Run: `uv run --frozen --extra agent pytest tests/engine/test_fatal_state.py tests/engine/test_worker.py tests/engine/test_lifecycle.py tests/engine/test_payload_jobs.py tests/engine/test_worker_interrupt.py tests/engine/test_session_interrupt.py -q`
Expected: PASS; all rejected waiters complete and existing interruption/lifecycle behavior remains intact.

- [ ] **Step 6: Commit worker rejection**

Commit the Task 2 files with `feat: reject native work after fatal decode failure`.

### Task 3: Exactly-once poisoned shutdown and complete verification

**Files:**
- Modify: `tests/engine/test_fatal_state.py`
- Modify only if a failing test demonstrates a defect: existing worker/session cleanup files from Task 1 or 2.

**Interfaces:**
- Consumes: Tasks 1–2 poisoning and rejection behavior plus existing `Engine.close()`, `Session.close()`, worker `_STOP`, and `ExitStack` ownership.
- Produces: regression evidence for exactly-once resource release; no new runtime API.

- [ ] **Step 1: Add the poisoned-shutdown regression test**

Create and retain a snapshot before poisoning, instrument snapshot/session/engine close and owner IDs, then poison the worker and call public `Engine.close()` and worker close repeatedly. Assert one close per resource in snapshot/session/engine order on the worker thread, exactly one terminated owner thread, and no extra native operation from rejected work or repeated close. Run it before any cleanup edit; if it passes, retain it as characterization coverage and make no runtime change. If it fails, use systematic debugging and a focused RED→GREEN fix.

- [ ] **Step 2: Run all focused subsystem suites**

Run:

```bash
uv run --frozen --extra agent pytest tests/engine/test_fatal_state.py tests/engine/test_worker.py tests/engine/test_decode.py tests/engine/test_multi_token_step.py tests/engine/test_tentative_decode.py tests/engine/test_lifecycle.py tests/engine/test_worker_interrupt.py tests/engine/test_session_interrupt.py tests/engine/test_payload_jobs.py tests/engine/test_scoped_state.py tests/engine/test_decode_step_selection.py -q
```

Expected: PASS. This covers worker, decode, lifecycle, interruption, checkpoint/snapshot, payload, extension routing, and tentative decode.

- [ ] **Step 3: Run the complete required gates**

Run from the worktree root:

```bash
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest -m "not build and not live" -q
git diff --check
```

Expected: all static checks and portable tests pass with no diff errors. Record exact counts and pre-existing skips. Do not claim live parity from fakes.

- [ ] **Step 4: Review and commit the completed branch**

Review the full diff against sections 8.2/8.5 and the design, including the two Roché decision points. Commit any remaining test/docs changes with a focused message such as `test: verify fatal-state poisoning cleanup`. Keep the branch available for downstream real-engine acceptance; do not merge or push.
