# Tool Error Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix #25, #26, and #22 in one branch without changing unrelated tools, prompts, policy, ADR 0002, or #27.

**Architecture:** Add one agent capability that converts exhausted tool `ModelRetry` failures into normal `Tool error: ...` results at the execution hook and, for validation failures, the public call-node error boundary. Decorate ds4-py's existing filesystem toolset with bounded missing-text diagnostics derived from current file bytes and visible read hashes. Replace hard-coded GLM/Qwen example identifiers with explicit placeholders.

**Tech Stack:** Python 3.14, pydantic-ai 2.59.0, pydantic-ai-harness 0.59.0, pytest, uv, devenv.

**Spec:** `docs/specs/2026-11-19-tool-error-recovery-design.md`

## Global Constraints

- Work in `.worktrees/tool-error-recovery` on `fix/tool-error-recovery`, based on `origin/main` at `95d52cbb9f189c9a940e777e9fab6baa15eac622`.
- Write and watch one failing behavioral test per issue before its production change; use scripted `FunctionModel` responses and no real model.
- Run project commands through `devenv shell`; use the locked `uv` environment and the isolated-home portable test command.
- Make exactly one commit per issue with `Fixes #25`, `Fixes #26`, and `Fixes #22` respectively, with no co-author or tool trailers.
- Do not change #27, ADR 0002, tool registration, line-range editing, or unrelated prompts/policy.

## Review Focus

- Retry recovery must not swallow cancellation, output-validation retries, ambiguous multi-call failures, or non-retry `UnexpectedModelBehavior`; Task 1 tests the specific retry-cause and node boundaries.
- The recovered result must let a scripted run finish and therefore follow the existing noninteractive success path; Task 1 asserts normal completion and a model-visible tool result.
- Edit diagnostics must not claim staleness without a matching visible read hash; Task 2 covers changed and unchanged reads plus the no-history case.
- Similarity output must be deterministic and bounded for long lines; Task 2 pins line selection and the 200-character excerpt cap.
- Placeholder examples must still parse after explicit substitution and must not name `read` when only `read_file` is registered; Task 3 exercises GLM and Qwen parsing.

---

### Task 1: Recover exhausted tool retries (#25)

**Files:**
- Create: `src/ds4py/capabilities/tool_errors.py`
- Modify: `src/ds4py/agent.py`
- Test: `tests/agent/test_file_system.py`
- Test: `tests/agent/test_noninteractive.py`
- Include the approved spec and this plan in this commit.

**Interfaces:**
- Produces: `ToolErrorRecovery`, an `AbstractCapability` that recovers execution exhaustion in `on_tool_execute_error` and validation exhaustion in `on_node_run_error`; every other error is re-raised.
- Consumes: existing pydantic-ai capability hooks, `RunContext.last_attempt`, public `CallToolsNode`/`ModelRequestNode`, and the interrupted request/call history; no retry-limit changes.

- [ ] **Step 1: Write the failing test**

Add `test_two_bad_tool_calls_finish_with_tool_error` to `tests/agent/test_file_system.py`. Use `_run_agent` with a real temporary `app.py` and an `edit_file` call whose `old_text` is absent. Assert that the real agent returns `done`, emits exactly two edit calls, and ends the edit exchange with a normal `ToolReturnPart` whose string content starts with `Tool error: old_text not found in app.py.`. Add a scripted two-unknown-tool-call test through the real `run_noninteractive`, asserting exit 0, final output `done`, empty stderr, and a normal `ToolReturnPart` after the second call. Cover invalid arguments, three consecutive execution failures, and verify an unrelated node error still propagates.

- [ ] **Step 2: Run the test and verify RED**

Run: `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_file_system.py::test_two_bad_tool_calls_finish_with_tool_error -q'`
Expected: the execution test FAILs with `UnexpectedModelBehavior: Tool 'edit_file' exceeded max retries count of 1`; the noninteractive validation test exits 1 for the same retry-exhaustion reason.

- [ ] **Step 3: Implement retry-exhaustion recovery**

Create `ToolErrorRecovery` in `src/ds4py/capabilities/tool_errors.py`. In the execution-error hook, recover only the direct `UnexpectedModelBehavior`/`ModelRetry` shape on `ctx.last_attempt`, returning `Tool error: {retry}`. In the node-error hook, accept only `UnexpectedModelBehavior` directly caused by `ModelRetry` or `pydantic.ValidationError` on `CallToolsNode` with a trailing interrupted request and exactly one unanswered tool call. Remove that interrupted request, use `dataclasses.replace` to make it complete with the added normal `ToolReturnPart`, and return `ModelRequestNode` for that request. Re-raise unrelated and ambiguous failures unchanged. Register one instance in the default component list in `src/ds4py/agent.py`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the new tests and the existing noninteractive tests:
`devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_file_system.py tests/agent/test_noninteractive.py -q'`
Expected: all pass, including direct exit-0 coverage after two bad validation-stage calls.

- [ ] **Step 5: Commit #25**

```bash
git add docs/specs/2026-11-19-tool-error-recovery-design.md docs/plans/2026-11-19-tool-error-recovery.md src/ds4py/capabilities/tool_errors.py src/ds4py/agent.py tests/agent/test_file_system.py tests/agent/test_noninteractive.py
git commit -m "fix(agent): recover exhausted tool retries" -m "Fixes #25"
```

### Task 2: Explain missing exact-edit text (#26)

**Files:**
- Modify: `src/ds4py/capabilities/file_system.py`
- Test: `tests/agent/test_file_system.py`

**Interfaces:**
- Consumes: `DS4FileSystem.get_toolset`, `RunContext.messages`, `ToolReturnPart` read results, workspace `read_bytes`, and stdlib `difflib`/`hashlib`.
- Produces: the same `ModelRetry` contract with an enriched message for only a single-`old_text` not-found error; no public tool schema changes.

- [ ] **Step 1: Write the failing tests**

Add a scripted read-then-edit test in which the file changes after `read_file`. Assert the first `RetryPromptPart` contains `Nearest match: line 2:`, the current `return response("hi")` line, and `The file has changed since the model last read it; re-read it before retrying.` Add coverage for an unchanged read hash, no visible read hash, and a spilled read result whose spill notice precedes the hash header. Make a long nearest line prove that the excerpt is capped at 200 characters.

- [ ] **Step 2: Run the tests and verify RED**

Run: `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_file_system.py -k "missing_old_text" -q'`
Expected: each new diagnostic test fails because the upstream message contains only `old_text not found ... No changes were written.`

- [ ] **Step 3: Implement bounded edit diagnostics**

Extend the prepared filesystem toolset's `call_tool` path. Catch `ModelRetry` only for `edit_file`'s single-`old_text` not-found message, leaving every other error untouched. Select the highest `difflib.SequenceMatcher` line ratio across non-empty old/file lines, comparing at most the first 200 characters of each line, iterating file lines outer and old-text lines inner, and stopping after 10,000 comparisons. Update only on a strictly better score so ties keep the earliest candidate. Append `Nearest match: line <N>:` and at most 200 characters of that file line. Hash current raw bytes with SHA-256's first 12 hex characters and compare with the newest visible `read_file` hash for the exact path spelling, using an unanchored header search so a `ToolOutputLimits` spill notice does not hide it; append the changed or unchanged sentence only when that hash exists. If current content cannot be read or no non-empty candidate exists, re-raise the original retry unchanged.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_file_system.py -q'`
Expected: all filesystem and #25 recovery tests pass.

- [ ] **Step 5: Commit #26**

```bash
git add src/ds4py/capabilities/file_system.py tests/agent/test_file_system.py
git commit -m "fix(agent): explain missed exact edits" -m "Fixes #26"
```

### Task 3: Use non-callable prompt examples (#22)

**Files:**
- Modify: `src/ds4py/engine/tool_prompt.py`
- Test: `tests/engine/test_tool_prompt.py`

**Interfaces:**
- Consumes: existing `ToolPromptGenerator._glm` and `._qwen` template methods.
- Produces: examples containing `$TOOL_NAME`, `$PARAMETER_NAME`, and `$PARAMETER_VALUE`; no serialized schema changes.

- [ ] **Step 1: Write the failing test**

Add a parameterized GLM/Qwen test using only a `read_file` definition. Extract the call example and assert it uses all three placeholders and does not contain a concrete `read` tool call or `README.md`. Update the exact-prompt expectations and example parser test so explicit placeholder substitution still parses to `read(path="README.md")`.

- [ ] **Step 2: Run the test and verify RED**

Run: `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/engine/test_tool_prompt.py -q'`
Expected: the new placeholder test fails on the hard-coded `read` example; no production code has changed yet.

- [ ] **Step 3: Replace the concrete examples**

Change only the GLM and Qwen call-example spans to `$TOOL_NAME`, `$PARAMETER_NAME`, and `$PARAMETER_VALUE`, matching the DeepSeek template's placeholder convention.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/engine/test_tool_prompt.py tests/engine/tool_syntax -q'`
Expected: all pass.

- [ ] **Step 5: Commit #22**

```bash
git add src/ds4py/engine/tool_prompt.py tests/engine/test_tool_prompt.py
git commit -m "fix(engine): use placeholders in tool examples" -m "Fixes #22"
```

### Final verification and publication

- [ ] Run `devenv shell -- bash -lc 'uv run --frozen --extra agent prek run --all-files --show-diff-on-failure'`; record every check and failure count.
- [ ] Run `devenv shell -- bash -lc 'HOME=$(mktemp -d) uv run --frozen --extra agent pytest -m "not build and not live" -q'`; record passed/skipped/deselected counts.
- [ ] Inspect `git diff origin/main...HEAD`, verify exactly three issue commits and a clean worktree, then push `fix/tool-error-recovery` and open one PR with causes, fixes, #25 rationale, upstream #26 note, and gate results.
- [ ] Run the `pr-review-loop` skill on the open PR, alternate reviewers between `deepseek/deepseek-v4-pro` and `kimi-coding/k3.1-pro`, and ask the user for every ruling. Re-run the gates after any material fix.
