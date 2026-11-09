# Gate 4 Interruption Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Gate 4 deliver Ctrl+C during real generation and prove interrupted-run, follow-up, and clean-exit behavior using structured session events.

**Architecture:** Move PTY process control into a focused `bench/pty.py` module while preserving the existing generic driver. Add a dedicated interruption driver that watches `events.jsonl` and correlates lifecycle events by `run_id`; `run_bakeoff.py` supplies the command and acceptance inputs. A final integration test exercises the real TUI and runtime with the fake engine before any production cancellation change is considered.

**Tech Stack:** Python 3.14, `pty`, `select`, append-only JSONL, pytest, prompt-toolkit, existing fake native engine.

**Spec:** `docs/specs/2026-11-02-gate-4-interruption-driver-design.md`

## Global Constraints

- Send Ctrl+C through the real PTY after the active run's first `Token` event.
- Require `Stopped(reason=interrupted)` and `run_end` for that same run before the follow-up.
- Require the follow-up's `run_end(status=success)` and clean process exit.
- Do not change core cancellation unless the real-TUI integration test exposes a separate defect.
- Do not accept visible output, footer text, or an uninterrupted successful run as interruption evidence.
- Do not run a live bake-off before the other reported gate repairs have local verification.

## Review Focus

- A partially appended JSON line: defer it until complete rather than failing or skipping the next valid event; pinned by Task 2's fragmented-event test.
- Stale or unrelated `run_id` events: never trigger Ctrl+C or the follow-up; pinned by Task 2's run-correlation test.
- Visible `streaming`/`idle` text without structured generation: time out with a useful lifecycle error; pinned by Task 2's output-only regression.
- Repeated token events: send exactly one Ctrl+C; pinned by Task 2's successful child-script exchange.
- Child exit or timeout while waiting for an event: kill/reap as needed and return a bounded diagnostic; pinned by Task 2's early-exit test.

---

### Task 1: Isolate PTY process control

**Files:**
- Create: `bench/pty.py`
- Modify: `bench/run_bakeoff.py`
- Test: `tests/bench/test_bakeoff_restore.py`

**Interfaces:**
- Consumes: existing `drive_pty` implementation and its callers.
- Produces: `bench.pty.drive_pty(command: Sequence[str], *, steps: Iterable[tuple[str, str | None]], cwd: str | Path | None = None, env: Mapping[str, str] | None = None, timeout: float = 30) -> CheckResult`, re-exported through `bench.run_bakeoff` for existing imports.

- [ ] **Step 1: Run the existing PTY regression before refactoring**

Run: `uv run --frozen --extra agent pytest tests/bench/test_bakeoff_restore.py::test_pty_interrupt_resumes -q`
Expected: PASS.

- [ ] **Step 2: Move the generic driver without changing behavior**

Move `drive_pty` and only its PTY-specific imports/helpers into `bench/pty.py`. Import it in `bench/run_bakeoff.py` so existing `from bench.run_bakeoff import drive_pty` callers continue to work. Do not change `interruption_acceptance` in this task.

- [ ] **Step 3: Verify the extraction**

Run: `uv run --frozen --extra agent pytest tests/bench/test_bakeoff_restore.py::test_pty_interrupt_resumes tests/bench/test_bakeoff_restore.py::test_interruption_acceptance_uses_tui_idle_lifecycle -q`
Expected: both PASS.

- [ ] **Step 4: Commit the focused refactor**

```bash
git add bench/pty.py bench/run_bakeoff.py
git commit -m "refactor(bench): isolate PTY process control"
```

### Task 2: Drive interruption from structured session events

**Files:**
- Modify: `bench/pty.py`
- Modify: `bench/run_bakeoff.py`
- Create: `tests/bench/test_interruption_driver.py`

**Interfaces:**
- Consumes: `drive_pty` process-management behavior from Task 1; session event objects with top-level `run_id`, `kind`, and `payload`.
- Produces: `drive_interruption_pty(command: Sequence[str], *, home: str | Path, follow_up: str, cwd: str | Path | None = None, env: Mapping[str, str] | None = None, timeout: float = 30) -> CheckResult`.
- `interruption_acceptance` calls this driver with `home=root / "home"`, the existing follow-up sentence, and `timeout=180`.

- [ ] **Step 1: Write the failing output-only regression**

Create a child script that prints `streaming` and `idle` but never creates a `Token` event. Call `drive_interruption_pty` with a timeout below one second. Assert `result.passed` is false, the detail names the missing initial `Token`, and the follow-up was never received.

- [ ] **Step 2: Write the failing fragmented-event and run-correlation tests**

Use a child script that writes `home/sessions/test-session/events.jsonl` incrementally. Cover split JSON writes and unrelated `run_id` events before the active run. The active exchange is: `run_start` → two `Token` events → one Ctrl+C → `Stopped(reason=interrupted)` → `run_end` → follow-up → second `run_end(status=success)` → `/exit`. Assert exit status 0, exactly one Ctrl+C, and correct follow-up ordering.

- [ ] **Step 3: Write the failing early-exit diagnostic test**

Have the child exit before creating an event stream. Assert a prompt, bounded `CheckResult` failure naming the expected initial `Token`; the driver must reap the child and must not hang.

- [ ] **Step 4: Run the new tests to verify RED**

Run: `uv run --frozen --extra agent pytest tests/bench/test_interruption_driver.py -q`
Expected: FAIL because `drive_interruption_pty` does not exist; no test may fail from fixture syntax or an import unrelated to the new interface.

- [ ] **Step 5: Implement the structured interruption driver**

In `bench/pty.py`, share PTY read/write/reap mechanics with the generic driver. Discover the single session stream under `home/sessions/*/events.jsonl`, read it incrementally, retain an incomplete trailing line, and ignore unrelated runs. Trigger once on the active run's first `Token`; accept only its interrupted `Stopped` and subsequent non-error `run_end` before sending `follow_up + "\n"`. Require the next run's `run_end(status=success)`, send `/exit\n`, and require process status 0. On failure, kill/reap if needed and retain the existing bounded PTY detail plus the awaited lifecycle state.

Update `interruption_acceptance` to call this interface instead of regex `streaming`/`idle` steps.

- [ ] **Step 6: Verify the driver and existing PTY behavior**

Run: `uv run --frozen --extra agent pytest tests/bench/test_interruption_driver.py tests/bench/test_bakeoff_restore.py::test_pty_interrupt_resumes -q`
Expected: all PASS.

- [ ] **Step 7: Commit the driver repair**

```bash
git add bench/pty.py bench/run_bakeoff.py tests/bench/test_interruption_driver.py
git commit -m "fix(bench): drive Gate 4 interruption from session events"
```

### Task 3: Prove the real TUI interruption path without a model

**Files:**
- Create: `tests/agent/test_tui_interruption.py`
- Modify only if a reproduced defect requires it after renewed diagnosis: `src/ds4py/tui.py`, `src/ds4py/runtime.py`, or the owning engine module.

**Interfaces:**
- Consumes: `AgentRuntime`, `TUI`, `create_pipe_input`, `DummyOutput`, and the existing fake engine/model helpers.
- Produces: non-live evidence that Ctrl+C interrupts an actual token-producing run and a follow-up can finish cleanly.

- [ ] **Step 1: Add the real-TUI integration test**

Start `TUI.run` with an initial prompt and a fake native script that generates beyond two tokens. After the second generated token, send `\x03` through `create_pipe_input`; then submit a follow-up whose fake response terminates normally and `/exit`. Assert one `Stopped` event with reason `interrupted`, a subsequent `run_end(status=success)` for the follow-up, and `TUI.run` returning 0 within five seconds.

- [ ] **Step 2: Run the integration test**

Run: `uv run --frozen --extra agent pytest tests/agent/test_tui_interruption.py -q`
Expected: PASS without production changes. If it FAILS, stop this plan and diagnose the reproduced cancellation defect before changing production code or relaxing the assertion.

- [ ] **Step 3: Commit the integration regression**

```bash
git add tests/agent/test_tui_interruption.py
git commit -m "test(agent): cover TUI interrupt and follow-up"
```

- [ ] **Step 4: Run complete non-live verification**

Run: `uv run --frozen --extra agent pytest -m 'not live and not build' -q`
Expected: all tests PASS; record the exact count.

Run: `uv run --frozen --extra agent prek run --all-files --show-diff-on-failure`
Expected: every hook PASS with a clean worktree.

- [ ] **Step 5: Record readiness and leave live acceptance pending**

Update `TODO-ec34b362` with the commits, test evidence, and remaining live Gate 4 acceptance. Do not provision a host or run the full bake-off in this plan.
