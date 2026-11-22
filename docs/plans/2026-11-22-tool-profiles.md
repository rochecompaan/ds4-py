# Tool Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add generic `auto | native | pi` agent tool profiles, with four Pi adapters over the existing filesystem and shell toolsets.

**Architecture:** Thread a typed profile through CLI/runtime/composition. A Pi capability wraps existing tool execution with exact schemas and argument conversion, while native composition remains unchanged. Resolve Hermes `auto` to Pi only as a declared development hypothesis.

**Tech Stack:** Python 3.14, pydantic-ai, pydantic-ai-harness 0.59.0, pytest, devenv, uv, prek, GitHub stacked PRs.

**Spec:** `docs/specs/2026-11-22-tool-profiles-design.md`

## Global Constraints

- Base is PR A `934ee5b917fb69f301a90e5f0dc4b2404192331d`; PR B stacks on `feat/hermes-tool-calls`.
- Existing families and explicit-native behavior remain unchanged.
- Pi exposes exactly `read`, `bash`, `edit`, `write`, in that order.
- Hermes `auto -> pi` is a development hypothesis awaiting comparison, never a measured result.
- No downstream product identifier enters `src/` or `tests/`.
- No live GGUF, benchmark, or result-text parity claim.
- Use devenv + `uv --frozen --extra agent`, focused failing tests first, and separate mechanical formatting commits.
- Reuse workspace containment, read-only policy, events, retries, output limits, and existing atomic replacement execution.

## Review Focus

- **Default and explicit profile plumbing:** a value must reach composition unchanged rather than being reset by CLI/runtime defaults. Task 7 pins the full path.
- **Schema drift:** Pi descriptions, argument names, order, and the two approved wording changes must match the reference. Task 8 uses an independent fixture.
- **Offset/default conversion:** 0 and 1 both read line 1; omitted limit is 2000; omitted bash timeout is 30 seconds. Task 8 pins boundary values.
- **Batch edit safety:** ambiguity, overlap, and a later invalid replacement must leave original bytes untouched. Task 8 pins atomicity.
- **Profile selection and policy:** explicit native/Pi work across families, auto preserves existing families, and workspace/read-only/events still apply. Tasks 8 and 9 pin these paths.

---

### Task 7: `tool_profile` plumbing

**Files:**
- Modify: `src/ds4py/agent_composition.py`
- Modify: `src/ds4py/agent_cli.py`
- Modify: `src/ds4py/runtime.py`
- Modify: `src/ds4py/agent.py`
- Test: `tests/agent/test_agent_composition.py`, `tests/agent/test_agent_factory.py`

**Interfaces:**
- Produces: `type ToolProfile = Literal["auto", "native", "pi"]`; `EngineOpenOptions.tool_profile: ToolProfile = "auto"` appended last; `AgentRuntime(..., tool_profile: ToolProfile = "auto")`; `build_agent_components(..., tool_profile: ToolProfile = "auto")`.

- [ ] **Step 1: Write failing parser and propagation tests**

Assert the CLI default is `auto` and each explicit choice is accepted. Extend the typed options/runtime test to pass `--tool-profile pi` and require the same value in `EngineOpenOptions`, the runtime keyword arguments, and the composition call. Assert an omitted option produces `auto` and native tool output remains the current set.

- [ ] **Step 2: Run and verify RED**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/agent/test_agent_composition.py -q`
Expected: FAIL because `--tool-profile` and the composition/runtime parameters do not exist.

- [ ] **Step 3: Implement the plumbing**

Add the PEP 695 alias, parser choice, defaulted engine field, runtime keyword, and composition argument. Pass the value unchanged at every boundary. Do not change tool selection in this task beyond accepting the parameter.

- [ ] **Step 4: Run composition/factory regressions and commit**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/agent/test_agent_composition.py tests/agent/test_agent_factory.py -q`
Expected: PASS and existing native behavior unchanged.

```bash
git add src/ds4py/agent_composition.py src/ds4py/agent_cli.py src/ds4py/runtime.py src/ds4py/agent.py tests/agent/test_agent_composition.py tests/agent/test_agent_factory.py
git commit -m "feat(agent): thread a generic tool profile through composition"
```

### Task 8: Four Pi adapters

**Files:**
- Create: `src/ds4py/capabilities/pi_profile.py`
- Create: `src/ds4py/agent_prompt_pi.py`
- Create: `tests/agent/test_pi_profile.py`
- Create: `tests/fixtures/pi_profile.json`
- Modify: `src/ds4py/agent.py`

**Interfaces:**
- Consumes: `FileSystemToolset`, `ShellToolset`, `Replacement`, `ToolProfile`, and the existing workspace/run context.
- Produces: `PiProfile` with one wrapped toolset exposing `read`, `bash`, `edit`, `write`; `PI_CODING_AGENT_INSTRUCTIONS`; explicit `tool_profile="pi"` composition.

**Reference contracts:**
- Definitions equal `ds4_agent_mellum_prompt.inc` except read says `50,000 characters` and bash says the configured ds4-py 30-second default.
- `read(path, offset=1, limit=2000)`: 0 and 1 both map to zero-based 0; positive N maps to N-1.
- `edit(path, edits[{oldText,newText}])`: match all items against the original file, never incrementally; reject ambiguity/overlap and submit original→final as one existing atomic whole-file replacement.
- `bash(command, timeout?)`: forward explicit `timeout_seconds`; omission reaches the existing 30-second default.
- `write(path, content)`: no expected-hash argument.

- [ ] **Step 1: Write exact-definition and failing behavior tests**

Create an independent fixture containing all four expected definitions. Test names, descriptions, parameter order/schemas, strict flags, no `tool_kind`, and the two wording changes. Add real workspace tests for read/write/edit/bash round trips, offset/default-limit conversion, explicit and omitted timeout, multi-edit atomicity, ambiguity/overlap refusal, and the cross-referencing pair `a→b`, `b→z` on original `a\nb\nc\n` (expected `b\nz\nc\n`). Pin read-only hiding of edit/write, workspace enforcement, existing result text, and ctx-bearing filesystem/shell events. Assert Pi instructions contain no native tool names.

- [ ] **Step 2: Run and verify RED**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/agent/test_pi_profile.py -q`
Expected: FAIL because the Pi capability, definitions, and instructions do not exist.

- [ ] **Step 3: Implement the wrapper and adapters**

Build a `WrapperToolset` over the existing filesystem/shell toolsets. Prepare exact typed `ToolDefinition`s and validators in reference order with no `tool_kind`, map calls back to original names, and call `wrapped.call_tool(original_name, converted_args, ctx, underlying_tool)` using the ctx-bearing tool from `wrapped.get_tools(ctx)`. Filter edit/write when `ctx.workspace.read_only`. Convert only the documented arguments. For edit, resolve every oldText against original decoded bytes, reject non-unique/overlapping ranges, compute final content with one non-overlapping splice, and call the existing ctx-bearing edit once with a whole-original to final `Replacement`. This preserves non-incremental matching and atomic events/hash/approval behavior.

In Pi composition, register the Pi capability instead of native filesystem/shell/job/skills/question tools, with filesystem read defaults of 2000 lines/50,000 characters and shell timeout 30. Omit `ToolOutputLimits` so exactly four tools are exposed; its spill/read_tool_result capability is a declared Pi-mode departure. Keep the remaining non-tool capabilities. Use Pi instructions unless `system` overrides them.

- [ ] **Step 4: Run adapter tests and commit**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/agent/test_pi_profile.py tests/agent/test_agent_factory.py -q`
Expected: PASS, including real round trips and native unchanged behavior.

```bash
git add src/ds4py/capabilities/pi_profile.py src/ds4py/agent_prompt_pi.py src/ds4py/agent.py tests/agent/test_pi_profile.py tests/fixtures/pi_profile.json
git commit -m "feat(agent): expose read, bash, edit and write for the pi profile"
```

### Task 9: Profile selection and generic regression gate

**Files:**
- Modify: `src/ds4py/capabilities/pi_profile.py`
- Modify: `src/ds4py/agent.py`
- Test: `tests/agent/test_pi_profile.py`

**Interfaces:**
- Produces: `resolve_tool_profile(requested: ToolProfile, *, family: str) -> Literal["native", "pi"]`; explicit values pass through, Hermes auto selects Pi, every other auto selects native.

- [ ] **Step 1: Write failing selection tests**

Assert Hermes auto exposes exactly `read`, `bash`, `edit`, `write`; Qwen and DeepSeek auto keep the native names; explicit native on Hermes keeps native; explicit Pi on Qwen selects Pi. Cover an engine-less fake model with auto→native. Assert all auto-mapping text describes a hypothesis pending comparison, not an accepted result.

- [ ] **Step 2: Run and verify RED**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/agent/test_pi_profile.py -q -k 'auto or explicit'`
Expected: FAIL because auto has no family-aware selection.

- [ ] **Step 3: Implement selection**

Add the small resolver and apply it once in `build_agent_components`. Use `model.session.engine.info.family` when present and native for engine-less test doubles. Do not infer profile from tool names or change explicit selections. Document that comparison runs select native/Pi explicitly and the dependency is re-pinned if the final ruling changes auto.

- [ ] **Step 4: Run the complete static gate**

Run: `devenv shell -- uv run --frozen --extra agent prek run --all-files --show-diff-on-failure`
Expected: PASS for all hooks; keep any formatting-only change separate.

- [ ] **Step 5: Run portable/generic gates and commit**

```bash
mkdir -p /tmp/ds4-quality-empty-home
devenv shell -- env HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest -m "not build and not live" -q
grep -Rni "sat""yrn" src tests
```

Expected: portable PASS with counts recorded; grep finds nothing; no live parity claim.

```bash
git add src/ds4py/capabilities/pi_profile.py src/ds4py/agent.py tests/agent/test_pi_profile.py
git commit -m "feat(agent): resolve the auto tool profile per family"
```

## Delivery and review

- [ ] Review the diff against the spec and record every actual departure in a short PR body, including the provisional auto mapping and exact gate counts.
- [ ] Push `feat/tool-profiles` without force and keep PR A's reviewed head unchanged.
- [ ] In a scratch clone under `~/work/scratch/` whose `origin/HEAD` is main, run `gh stack init --base main feat/hermes-tool-calls feat/tool-profiles`, `gh stack view`, verify local branches equal remote tips, then `gh stack submit --auto`. If it fails, open PR B with base `feat/hermes-tool-calls` and report the fallback.
- [ ] Run `pr-review-loop` on PR B, alternating `deepseek/deepseek-v4-pro` and `kimi-coding/k3.1-pro`; ask the owner for every decline/defer ruling.
- [ ] Draft the downstream issue comment with both PR URLs and head commits. Show it to the owner and do not post without explicit approval.
- [ ] Report PR B URL, head commit, stack order, and gate counts.
