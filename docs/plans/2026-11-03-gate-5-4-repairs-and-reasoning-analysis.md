# Gate 5 and Gate 4 Repairs and Reasoning Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair restored-workspace attachment and terminal interrupt recovery, then explain the retained long reasoning rounds without another GPU run.

**Architecture:** Keep restore in a disposable workspace and change only its local continuation metadata to name that copy; `ModelResponse.workspace_ref` is not sent to the model, so the saved prompt and KV prefix remain unchanged. Treat `KeyboardInterrupt` from `prompt_async()` like the existing Ctrl+C key binding: cancel the active run and recreate the prompt instead of closing the TUI. Analyze the retained task evidence only after both fixes pass locally.

**Tech Stack:** Python 3.14, pytest, Pydantic AI workspaces, prompt-toolkit, existing bake-off JSON/JSONL evidence.

**Spec:** `docs/specs/2026-11-02-gate-4-interruption-driver-design.md`, `docs/specs/2026-10-30-bakeoff-gate-fixes-design.md`, and the new failure evidence in `docs/reports/2026-11-03-gate-repairs-targeted-bakeoff.md`.

## Global Constraints

- Keep restore writes in disposable workspace copies.
- Do not change saved prompt bytes or normal `/switch` workspace behavior.
- Do not relax Gate 4: it still requires an interrupted run, a successful follow-up, and clean exit.
- Keep `asyncio.CancelledError` and EOF behavior unchanged.
- Do not start another live bake-off in this task.
- Base reasoning findings on retained C and ds4-py evidence; separate observations from recommendations.

## Review Focus

- A relocated restore must still extend the saved KV prefix; the Gate 5 test asserts `followup_plan == "extend"`.
- Restoring in the original workspace must keep working; parameterize the existing Gate 5 test to cover both locations.
- A prompt-level `KeyboardInterrupt` must not lose the follow-up or close the process; parameterize the real TUI test to cover both Ctrl+C delivery forms.
- Reasoning analysis must not confuse output volume with engine throughput; Task 3 cross-checks token counts, wall times, prefill, and decode rates.

---

### Task 1: Gate 5 restored-workspace attachment

**Files:**
- Modify: `bench/restore_check.py`
- Test: `tests/bench/test_bakeoff_restore.py`

**Interfaces:**
- Consumes: existing `switch_via_command(runtime, session_id)` and `AgentRuntime.messages`.
- Produces: `_relocate_local_workspace(runtime: AgentRuntime) -> None`, called only by the restore driver after `/switch` and before the follow-up.

- [ ] **Step 1: Write the failing regression**

Parameterize `test_restore_driver_preserves_noninteractive_prompt_composition` to run once in the original project and once from a copied project. In both cases, require status 0 and `followup_plan == "extend"`; retain the existing noninteractive composition assertion.

- [ ] **Step 2: Run the regression and confirm the original failure**

Run: `uv run --extra agent pytest tests/bench/test_bakeoff_restore.py::test_restore_driver_preserves_noninteractive_prompt_composition -q`

Expected: the original-workspace case passes; the copied-workspace case fails with `Skills needs a workspace` naming the original `local:` path.

- [ ] **Step 3: Implement explicit local-workspace relocation**

Add `_relocate_local_workspace(runtime)`. Find the last `ModelResponse`; when its `workspace_ref.provider` is `local`, replace that mutable response's reference with `WorkspaceRef(provider="local", id=str(runtime.project_dir.resolve()))`. Leave missing and non-local references unchanged. Call it in `_restore_operation` immediately after `switch_via_command`.

- [ ] **Step 4: Verify both restore locations and write isolation**

Run: `uv run --extra agent pytest tests/bench/test_bakeoff_restore.py::test_restore_driver_preserves_noninteractive_prompt_composition tests/bench/test_bakeoff_restore.py::test_restore_processes_use_disposable_workspace_copies -q`

Expected: 3 tests pass. The copied restore extends the same prompt, and process restores still use distinct disposable copies without writing to the source workspace.

- [ ] **Step 5: Commit the Gate 5 repair**

```bash
git add bench/restore_check.py tests/bench/test_bakeoff_restore.py docs/plans/2026-11-03-gate-5-4-repairs-and-reasoning-analysis.md
git commit -m "fix(bench): relocate restored workspace metadata"
```

### Task 2: Gate 4 prompt-level interrupt recovery

**Files:**
- Modify: `src/ds4py/tui.py`
- Test: `tests/agent/test_tui_interruption.py`

**Interfaces:**
- Consumes: `PromptSession.prompt_async()` and `AgentRuntime.cancel()`.
- Produces: the existing `_ensure_prompt() -> asyncio.Task[str]` contract, now resilient to prompt-level `KeyboardInterrupt`.

- [ ] **Step 1: Write the failing regression**

Parameterize `test_tui_ctrl_c_interrupts_generation_and_allows_follow_up` with normal Ctrl+C input and a one-shot `KeyboardInterrupt` raised from `PromptSession.prompt_async` after generation starts. In the exception case, call `runtime.cancel()` as the terminal signal path would and then return to the real prompt implementation. Both cases must produce one interrupted round, a successful follow-up, and clean `/exit`.

- [ ] **Step 2: Run the regression and confirm the original failure**

Run: `uv run --extra agent pytest tests/agent/test_tui_interruption.py -q`

Expected: the normal Ctrl+C case passes; the prompt-exception case fails by raising `KeyboardInterrupt` before the follow-up.

- [ ] **Step 3: Keep the prompt alive across terminal interrupts**

In the prompt coroutine inside `_ensure_prompt`, catch only `KeyboardInterrupt` around `self.session.prompt_async("ds4py> ")`. Call `self.runtime.cancel()` and loop back to a fresh `prompt_async()` call. Preserve text validation, EOF, cancellation, and all other exceptions.

- [ ] **Step 4: Run focused and full verification**

Run: `uv run --extra agent pytest tests/agent/test_tui_interruption.py tests/agent/test_tui.py tests/bench/test_pty_driver.py -q`

Expected: all focused tests pass, including both interrupt delivery forms.

Run: `uv run --extra agent pytest -q`

Expected: the full suite passes with no new warnings.

Run: `uv run prek run --all-files`

Expected: all hooks pass.

- [ ] **Step 5: Commit the Gate 4 repair**

```bash
git add src/ds4py/tui.py tests/agent/test_tui_interruption.py
git commit -m "fix(agent): recover prompts after terminal interrupts"
```

### Task 3: Analyze the retained long reasoning rounds

**Files:**
- Create: `docs/reports/2026-11-03-long-reasoning-analysis.md`
- Read: `docs/reports/2026-11-03-gate-repairs-targeted-bakeoff.md`
- Read: retained evidence under `/home/roche/projects/satyrn/ds4-py-evidence/full/gate-repairs-0ec1288/`

**Interfaces:**
- Consumes: `bakeoff.json`, per-attempt `attempt.json`, `interactive.log`, `trace.jsonl`, and C attempt metrics.
- Produces: an evidence-backed explanation and ranked next experiments; no production interface changes.

- [ ] **Step 1: Derive comparable round and timing summaries**

For `scale-generator`, `affine-cipher`, `tree-building`, and `zipper`, derive per-arm attempt counts, token totals, wall times, prefill/decode rates, tool outcomes, terminal reasons, and the longest model rounds. Inspect the full text of the long ds4-py rounds and compare them with the C path through the same task.

- [ ] **Step 2: Write the analysis**

Document why `scale-generator` exhausted 8,192 tokens, whether the other long rounds show repetition or useful work, and why token counts alone do not explain Gate 3. Distinguish agent behavior, prompt/context effects, and engine throughput. Rank small next experiments by expected evidence value; do not claim a performance repair without live verification.

- [ ] **Step 3: Verify and commit the report**

Cross-check every quoted count and timing against the retained JSON/JSONL, run `uv run prek run --all-files`, and inspect `git diff --check`. Per the Testing Value Gate, use these direct checks rather than adding tests for documentation text.

```bash
git add docs/reports/2026-11-03-long-reasoning-analysis.md
git commit -m "docs: analyze long bake-off reasoning rounds"
```
