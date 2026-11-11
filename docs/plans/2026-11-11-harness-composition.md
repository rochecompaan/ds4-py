# Generic Harness Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit generic composition boundary for agent CLI extensions, including typed open options, generic `--top-k`, post-builtin commands, and identity-change hooks, without changing the default agent path.

**Architecture:** A small `agent_composition` module owns the public typed options, configured-session context manager, extension protocol, and default extension. `agent_cli` constructs one options value and delegates configured-session creation while retaining model/runtime construction and command orchestration. `AgentRuntime` receives optional synchronous before/after identity callables; its no-hook branches retain the merged implementation's ordering exactly, including run-record relocation.

**Tech Stack:** Python 3.14, `dataclasses`, `typing.Protocol`, `argparse`, `contextlib`, pytest, existing fake native engine and agent fixtures.

**Spec:** `docs/specs/2026-11-10-m3-harness-integration-design.md`, especially §§5–8, §12.1, and §14.1 item 1. Payload support is explicitly out of scope.

## Global Constraints

- Branch from `origin/main` at `770ace6df7da162f0a68c97382b7165658485357` or later on `feat/harness-composition`.
- Add no downstream product identifiers or policy to production code or tests.
- Add no payload probe, stripped save, rendered-switch fallback, or persistence-schema change.
- Preserve default engine open, sampling, runtime arguments, question wiring, tracing, close order, diagnostics, identity behavior, and ADR 0002 behavior.
- Preserve explicit zero values and computed think modes; use direct typed fields, never downstream `Namespace` inspection or `getattr` fallback.
- Construct `DS4Model` and `AgentRuntime` in `open_runtime()` exactly once. A custom configured session supplies the exact sampling object; do not merge it again.
- Close in this order: configured-session context enter → model/runtime construction → `AgentRuntime.close()` → configured-session context exit/wrapper close.
- Run parser configuration after generic options and command configuration after built-in commands. Duplicate names continue to fail through argparse and `SlashCommandRegistry`.
- Identity hooks are synchronous runtime orchestration, never session-event listeners; `fork()` invokes neither hook.
- Keep mechanical formatting separate from behavioral edits. Make one local commit only after the requested final review and gates.

## Review Focus

- Explicit `0` sampling values and `think=0`: prove they reach the same typed options and resulting policy rather than being treated as missing (Tasks 1 and 2).
- Default switch telemetry: prove the no-hook path retains its current pre-`Session.switch()` run-record relocation while the extension path relocates inside restore (Task 3). With hooks configured, `switching`/`session_end` remain in the old session's `events.jsonl`; extension tests must not assume target-file attribution.
- Before/after hook failures: prove before failure is atomic, after failure retains the adopted identity and sets `status="error"`, and restore failure keeps target telemetry (Task 3).
- Custom sampling ownership: prove the exact extension-provided policy reaches `DS4Model` without a second `SamplingPolicy.for_engine()` call (Task 2).
- Interactive versus noninteractive composition: prove both open through the extension, but only interactive construction invokes command configuration (Task 2).

---

### Task 1: Typed Composition Contracts and Generic Top-K

**Files:**
- Create: `src/ds4py/agent_composition.py`
- Modify: `src/ds4py/agent_cli.py`
- Test: `tests/agent/test_noninteractive.py`
- Test: `tests/agent/test_agent_composition.py`

**Interfaces:**
- Produces frozen `EngineOpenOptions` with `model: str`, `ctx: int`, `backend: str | None`, `think: int`, `seed: int | None`, `threads: int`, `lib: Path | None`, `tokens: int`, `interactive: bool`, `temperature: float | None`, `top_k: int | None`, `top_p: float | None`, and `min_p: float | None`.
- Produces frozen `ConfiguredSession(session: Session, sampling: SamplingPolicy, close: Callable[[], None])`, a context manager whose exit calls `close()` exactly once per exit.
- Produces `AgentExtension(Protocol)` with `program`, `configure_parser()`, `resolve_extension_config()`, `open_session()`, `configure_commands()`, `before_identity_change()`, and `after_identity_change()` signatures from spec §5.
- Produces a stateless default extension with `program == "ds4py"`, no-op configuration/identity methods, and the existing engine/session/sampling open flow.

- [ ] **Step 1: Write failing generic parser and default-sampling tests**

Add parametrized coverage for `--top-k` unset, `--top-k 0`, and `--top-k 40`. Drive the real default `open_runtime()` with the existing controlled engine/model/runtime doubles and assert `SamplingPolicy.top_k == 0`, `0`, and `40`, respectively. Extend the existing default CLI mapping test with `--lib` and all generic sampling values so explicit zero remains visible.

- [ ] **Step 2: Run the tests and verify RED**

Run: `HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest tests/agent/test_noninteractive.py tests/agent/test_agent_composition.py -q`
Expected: FAIL because `--top-k` is unrecognized and the composition module/contracts do not exist.

- [ ] **Step 3: Implement the contracts, default open flow, and generic parser option**

Create the interfaces above. Move the existing default `ds4py.open()`, `engine.session()`, and one `SamplingPolicy.for_engine()` flow into the default extension's `open_session(options, project_dir, extension_config)`, reading only typed fields and passing `top_k=options.top_k`. Keep `from ds4py import open as open_engine` function-local so the existing monkeypatch seam stays live; keep harness imports in `agent_composition` `TYPE_CHECKING`-only. Use `ExitStack.enter_context(open_engine(...))` and expose `stack.close` as `ConfiguredSession.close`, closing the stack if session/sampling setup fails; retain the existing close-order test as no-hook proof. Add `parser(extension: AgentExtension | None = None)`, set `prog` from the extension, add `--top-k` as `type=int`, then invoke `configure_parser()` after all generic options. Wire `open_runtime()` through the configured-session context while preserving its existing signature when no extension is supplied and retaining model/runtime construction in `agent_cli`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest tests/agent/test_noninteractive.py tests/agent/test_agent_composition.py -q`
Expected: PASS, including all pre-existing default-path tests.

### Task 2: Explicit Extension Composition, Commands, and Labels

**Files:**
- Modify: `src/ds4py/agent_cli.py`
- Modify: `src/ds4py/noninteractive.py`
- Modify: `src/ds4py/tui.py`
- Test: `tests/agent/test_agent_composition.py`
- Test: `tests/agent/test_noninteractive.py`
- Test: `tests/agent/test_tui_commands.py`

**Interfaces:**
- Consumes Task 1's `EngineOpenOptions`, `ConfiguredSession`, `AgentExtension`, and default extension.
- `main(argv: list[str], *, extension: AgentExtension | None = None) -> int` computes `project_dir = Path.cwd()` once, resolves extension config once after parsing/CLI validation, and passes both through `command()` and `open_runtime()`.
- `command(args, *, extension=None, extension_config=None, project_dir: Path | None = None)` preserves direct callers by defaulting `project_dir` to `Path.cwd()`.
- `open_runtime(args, project_dir, *, read_only=False, extension=None, extension_config=None)` forwards the exact extension/config/project values to `open_session()`; existing `bench/restore_plan.py` callers need no change.
- `_run_tui(runtime, prompt, *, configure_commands: Callable[[SlashCommandRegistry, AgentRuntime], None] | None = None)` forwards the optional configurator.
- `TUI(..., configure_commands: Callable[[SlashCommandRegistry, AgentRuntime], None] | None = None)` invokes it immediately after `install_builtin_commands()`.
- `run_noninteractive(..., program: str = "ds4py")` and `_report(err, message, program)` use `program` only for diagnostic prefixes; the default-label assertions stay unchanged. Update the existing single-signature `command` and `run_noninteractive` test doubles for the forwarded keywords.

- [ ] **Step 1: Write failing typed-extension and close-order tests**

Use a recording extension whose `open_session()` returns a real `ConfiguredSession` around a controlled session and a distinct sampling object. Parametrize `--think-level 25` and `--nothink`, include every generic engine/model field plus zero temperature/top-k/top-p/min-p, and assert the single captured options value has `think == 1025` or `0`, the right `interactive` value, and every explicit zero. Assert extension config is the exact resolved object, the model receives the exact session and sampling object, and events are `open → model → runtime → runtime close → configured close`. Make `SamplingPolicy.for_engine()` fail if the custom path calls it.

- [ ] **Step 2: Write failing parser, command, path, and label tests**

Cover: generic parser options exist before extension configuration; duplicate parser options and duplicate slash commands fail; command configuration observes built-ins and registers a usable extension command; interactive and noninteractive paths both call `open_session()`, while only interactive calls `configure_commands()`; parser `prog`, top-level command errors, and noninteractive run/save diagnostics use the extension's `program`. Keep existing `--yolo` rejection and no-approval-runtime assertions unchanged. Add a default command test proving trace enter/exit and prefix/question behavior remain in their current order.

- [ ] **Step 3: Run the new tests and verify RED**

Run: `HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest tests/agent/test_agent_composition.py tests/agent/test_noninteractive.py tests/agent/test_tui_commands.py -q`
Expected: FAIL because `main()` does not accept an extension, config/commands are not delegated, and noninteractive labels are fixed.

- [ ] **Step 4: Implement extension propagation and post-builtin commands**

Resolve extension config once in `main()` with the same computed `project_dir`, pass it unchanged to `open_session()`, and use `extension.program` for parser/top-level/noninteractive labels. Build `EngineOpenOptions` once in `open_runtime()` using `_think_mode(args)` and `THINK_LEVEL_BASE` only in core. Set `hooks_configured = extension is not None` before installing the default; pass both custom identity methods to `AgentRuntime` when that flag is true and explicit `None` values otherwise (Task 3 consumes them), preserving its exact no-hook branch with one construction site. Forward command configuration only to interactive TUI construction after built-ins. Leave prefix loading, tracing, question wiring, prompt handling, and runtime close in the existing command path.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest tests/agent/test_agent_composition.py tests/agent/test_noninteractive.py tests/agent/test_tui_commands.py -q`
Expected: PASS.

### Task 3: Runtime Identity-Change Orchestration

**Files:**
- Modify: `src/ds4py/runtime.py`
- Test: `tests/agent/test_runtime.py`
- Test: `tests/agent/test_restore.py`

**Interfaces:**
- Consumes optional `before_identity_change: Callable[[], None] | None` and `after_identity_change: Callable[[], None] | None` constructor keywords supplied by Task 2.
- No hooks means the merged code's behavior and event-file ordering remain unchanged.
- With either hook configured, `new()` and `switch()` use the spec's before → session operation → runtime bookkeeping → after ordering; `fork()` uses no hooks.

- [ ] **Step 1: Write failing success-order and fork tests**

For `new()`, record before/session-start/after order and assert runtime messages, title, notes/binder reset, timestamps, and run-record relocation are complete when after runs. For `switch()`, assert before runs after all validation, the old record is still active before the restore callback, relocation occurs at callback entry, target bookkeeping is complete before after, and idle is set only after after returns. Assert `fork()` calls neither hook.

- [ ] **Step 2: Write failing failure-atomicity and default-regression tests**

Cover before-hook failure for new/switch: old identity, messages, status, and record location remain unchanged and no session operation occurs. Cover after-hook failure for new/switch: the adopted identity and target record remain, `status == "error"`, and the error propagates. In the before hook install a real generic `PlainStep`; assert it remains installed after after-hook failure without adding any native verify-mode policy. Cover restore failure after callback entry: after is skipped, target identity/error state and record location remain, and error `session_start` telemetry is written to the target. Add a no-hook characterization test proving current switch record relocation/event ordering is unchanged.

- [ ] **Step 3: Run identity tests and verify RED**

Run: `HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest tests/agent/test_runtime.py tests/agent/test_restore.py -q`
Expected: FAIL because runtime identity callables and orchestration do not exist.

- [ ] **Step 4: Implement the minimal hook orchestration**

Store optional callables and a flag recording whether either was supplied; `open_runtime()` supplies either both custom methods or neither. For configured `new()`, invoke before after `_stop_active()`, retain all existing bookkeeping and the current mid-bookkeeping `_status = "idle"` assignment, then invoke after last; on after error set error and propagate. For configured `switch()`, invoke before only after record/snapshot/plan/provenance validation, relocate at restore callback entry, invoke after after `Session.switch()` returns, then set idle; on after error set error and propagate. Keep the existing no-hook relocation and status assignment positions. Do not invoke hooks from `fork()` or any session listener.

- [ ] **Step 5: Run identity and companion tests and verify GREEN**

Run: `HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest tests/agent/test_runtime.py tests/agent/test_restore.py tests/agent/test_agent_composition.py tests/agent/test_noninteractive.py tests/agent/test_tui_commands.py -q`
Expected: PASS.

### Task 4: Full Verification, Review, and Local Commit

**Files:**
- Review the complete diff from `770ace6df7da162f0a68c97382b7165658485357`.
- Modify only files needed to fix evidence-backed review findings within the approved scope.

- [ ] **Step 1: Run the complete static gate**

Run: `HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent prek run --all-files --show-diff-on-failure`
Expected: all hooks pass with no blanket suppressions or unsafe fixes.

- [ ] **Step 2: Run the complete portable suite with isolated HOME**

Run: `HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest -m "not build and not live" -q`
Expected: all tests pass; only pre-existing documented skips remain. Also run `git diff --check` and verify production/test diff contains no downstream identifiers or payload-support changes.

- [ ] **Step 3: Run `/review-loop` on the complete diff**

Use one fresh reviewer with the `thermo-nuclear-code-quality-review` skill for structure/maintainability and another fresh reviewer for correctness/tests. Supply the spec §§5–8, §12.1, and §14, this plan, the base/head diff, and the no-default-change constraint. Fix P0 findings and evidence-backed P1 findings worth fixing now using RED→GREEN; ask before any restructuring beyond this plan. Re-review only material fixes.

- [ ] **Step 4: Re-run affected gates and commit locally**

Re-run the complete static gate and portable suite after material fixes. Stage only the plan, focused production files, and tests; commit on `feat/harness-composition` with message `feat: add generic agent harness composition`. Do not push or open a PR.
