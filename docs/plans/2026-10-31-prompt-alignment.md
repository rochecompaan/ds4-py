# Coding-Agent Prompt Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align ds4-py's default coding-agent and native DSML guidance with C while preserving Python's tools and adding a truthful 160-line default read contract with a 50,000-character cap.

**Architecture:** Keep family-specific DSML syntax in the engine and coding workflow in a new agent-prompt module. Render agent instructions before tool schemas. At the agent/tool boundary, use a cached `PreparedToolset` to update only `read_file.limit` schema metadata from the configured filesystem default; tool names, arguments, and execution remain unchanged.

**Tech Stack:** Python 3.14, Pydantic AI, pydantic-ai-harness, pytest, Pyright, Ruff.

**Spec:** `docs/specs/2026-10-31-prompt-alignment-design.md`

## Global Constraints

- Keep all existing Python tool names and parameter names.
- Do not add `whole`, `raw`, `start_line`, `max_lines`, or `more` interfaces.
- Use `max_read_lines=160` and `max_read_chars=50_000` for the default agent filesystem.
- Keep `offset` zero-based and keep `path` as the only required `read_file` parameter.
- Preserve the read schema's `"default": null`; the capability supplies the configured line default.
- Make the schema description follow the configured line default rather than hardcoding 160 in the adapter.
- Keep trusted DSML markup in engine-owned `ToolPromptSpan(..., rendered=True)` spans; agent instructions and tool schemas remain plain payloads.
- An explicit `system` argument continues to replace the default agent instructions exactly.
- Do not patch installed dependency files.
- Do not start a paid GPU instance or live bake-off automatically.

## Review Focus

- **Custom instructions:** an explicit `system` value must not acquire the default coding prompt; Task 3 retains and runs `test_factory_applies_persistent_instructions`.
- **Prompt injection boundaries:** new DSML teaching markup must be trusted while instructions, descriptions, and schemas remain plain; Task 1 runs the existing markup-security tests and executable-stanza test.
- **Read continuation:** explicit limits and default-limited reads must name the first unseen zero-based offset without skipping content; Task 2 adds default and explicit-limit tests.
- **Configured schema defaults:** a nonstandard configured line default must appear in the schema, proving the adapter is not another hardcoded constant; Task 2 adds a 77-line schema test.
- **Tool narrowing:** schema preparation must not reintroduce `list_directory` or writable tools in a read-only workspace; Task 2 extends the exposed-tool tests.
- **Other model families:** GLM and Qwen native prompt bodies must not acquire DeepSeek DSML wording; Task 1 retains their exact-prompt tests.

---

### Task 1: Align DeepSeek DSML Guidance and Instruction Ordering

**Files:**
- Modify: `src/ds4py/engine/tool_prompt.py:66-88`
- Modify: `src/ds4py/engine/prompt.py:31-64`
- Test: `tests/engine/test_tool_prompt.py`
- Test: `tests/engine/test_prompt.py:117-124`
- Test: `tests/agent/test_binding.py:77-94`

**Interfaces:**
- Consumes: existing `PromptBuilder.begin(tools, instructions)` and `ToolPromptGenerator.spans(definitions)` callers.
- Produces: unchanged public signatures; plain `instructions` render before native tool guidance and schemas, and DeepSeek prompts contain the C-style native DSML teaching block.

- [ ] **Step 1: Commit the approved design and this plan**

```bash
git add docs/specs/2026-10-31-prompt-alignment-design.md docs/plans/2026-10-31-prompt-alignment.md
git commit -m "docs(agent): design prompt alignment"
```

- [ ] **Step 2: Write the failing native-prompt and ordering tests**

Update `deepseek_expected` so the fixed DeepSeek prefix contains this exact guidance, with the existing `v41` tag-name substitutions:

```text
## Tools

You have access to native DSML tools. Invoke tools by writing exactly this shape:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>

Tool calls are not allowed inside <think></think>; finish thinking before emitting DSML.

String parameters use raw text and string="true". Numbers and booleans use JSON text and string="false".

Inside string values only, escape a literal closing parameter tag as &lt;/｜DSML｜parameter>. To write that escaped spelling literally, use &amp;lt;/｜DSML｜parameter>. Other HTML entities are unchanged.

### Available Tool Schemas
```

Keep the existing `<tools>...</tools>` schema envelope. Remove the redundant DeepSeek `Call example` suffix because the invocation shape now precedes the schemas; leave GLM and Qwen expected strings unchanged. Update the executable-example test to parse the DeepSeek teaching stanza from its new location.

Rename `test_instructions_are_a_separate_prefixed_system_message` to `test_instructions_precede_tool_schemas` and assert that the decoded instruction payload precedes `# Tools`/`## Tools`, remains plain, and is separated from the following tool prompt by a blank line. Update the binding order expectation without changing its message-content assertions.

- [ ] **Step 3: Run the focused tests and confirm the old behavior fails**

Run:

```bash
uv run --extra agent pytest tests/engine/test_tool_prompt.py tests/engine/test_prompt.py tests/agent/test_binding.py -v
```

Expected: DeepSeek exact-prompt tests fail on the old `# Tools` prefix and suffix, and the ordering tests fail because instructions currently follow tool tokens.

- [ ] **Step 4: Implement the DeepSeek template and instruction-order change**

In `ToolPromptGenerator._deepseek`, build the approved prefix as trusted spans and retain plain, compact schema serialization. Use the existing family-specific `calls`, `invoke`, `parameter`, and escaped closing-tag values; do not insert Python tool names or coding workflow rules.

In `PromptBuilder.begin`, render nonempty `instructions` before `_tool_tokens`, with a trailing blank line instead of the current leading blank line. Preserve existing family role wrappers and trusted/untrusted tokenization boundaries.

- [ ] **Step 5: Run native prompt, security, parser, and binding verification**

Run:

```bash
uv run --extra agent pytest tests/engine/test_tool_prompt.py tests/engine/test_prompt.py tests/engine/tool_syntax tests/agent/test_binding.py tests/agent/test_binding_rewrite.py -v
```

Expected: PASS, including exact GLM/Qwen prompts, plain caller payload tests, successful parsing of every generated invocation example, and unchanged message binding.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/ds4py/engine/tool_prompt.py src/ds4py/engine/prompt.py tests/engine/test_tool_prompt.py tests/engine/test_prompt.py tests/agent/test_binding.py
git commit -m "feat(engine): align native tool prompt guidance"
```

### Task 2: Align the Default Read Window and Schema

**Files:**
- Create: `src/ds4py/capabilities/file_system.py`
- Modify: `src/ds4py/agent.py:55-62`
- Create: `tests/agent/test_file_system.py`
- Test: `tests/agent/test_agent_factory.py`

**Interfaces:**
- Consumes: Pydantic AI `PreparedToolset`, `ToolDefinition`, and the harness `FileSystem` capability.
- Produces:

```python
def prepare_file_tool_definitions(
    tool_defs: list[ToolDefinition], *, default_read_lines: int
) -> list[ToolDefinition]: ...

class DS4FileSystem(FileSystem):
    def get_toolset(self) -> AbstractToolset[Any]: ...
```

`DS4FileSystem` retains the harness constructor and execution behavior while caching one prepared wrapper around its real toolset.

- [ ] **Step 1: Write failing schema, read-window, and filtering tests**

In `test_file_system.py`, add these focused tests:

1. `test_read_schema_uses_configured_default`: pass a `read_file` definition through `prepare_file_tool_definitions(..., default_read_lines=77)` and assert only the limit description changes to `Maximum number of lines to return (default: 77).`; properties remain `path`, `offset`, and `limit`, `required` remains `['path']`, and the JSON default remains `None`.
2. `test_default_read_returns_160_lines_and_next_offset`: run the real default agent against a 165-line text file, call `read_file` without `limit`, and assert line 160 is present, line 161's content is absent, and the continuation hint contains `Use offset=160`.
3. `test_explicit_read_limit_overrides_default`: call the real `read_file` with `limit=20` and assert the result stops at line 20 with `Use offset=20`.
4. `test_read_retains_50000_character_cap`: read a text file larger than 50,000 characters and assert the complete tool result, including header and hint, is at most 50,000 characters and contains a continuation hint.
5. `test_read_only_workspace_keeps_writable_tools_hidden`: build with `read_only=True` and assert `write_file`, `edit_file`, and `create_directory` are absent while `read_file` remains.

Extend the existing no-`list_directory` test to verify the prepared filesystem still exposes the expected tool set.

- [ ] **Step 2: Run the new tests and confirm current behavior fails**

Run:

```bash
uv run --extra agent pytest tests/agent/test_file_system.py tests/agent/test_agent_factory.py::test_factory_does_not_expose_list_directory -v
```

Expected: the schema helper is missing, the default read currently reaches beyond 160 lines, and the explicit-limit/read-only cases provide controls for unchanged harness behavior.

- [ ] **Step 3: Implement the prepared filesystem adapter and configuration**

Implement `prepare_file_tool_definitions` by deep-copying only the `read_file` parameter schema, updating `properties.limit.description` from `default_read_lines`, and using `dataclasses.replace` on that definition. Return every other definition unchanged.

Implement `DS4FileSystem.get_toolset` with one cached `PreparedToolset` around `super().get_toolset()`. Its prepare callback delegates to `prepare_file_tool_definitions` with `self.max_read_lines`; do not wrap or replace `call_tool`.

In `build_agent_components`, use `DS4FileSystem` with `max_read_lines=160` and `max_read_chars=50_000`. Keep the existing selected tool names, root, hash, and read-only behavior.

- [ ] **Step 4: Run filesystem and agent integration tests**

Run:

```bash
uv run --extra agent pytest tests/agent/test_file_system.py tests/agent/test_agent_factory.py -v
```

Expected: PASS, including actual shell/read execution, hidden-file discovery, unchanged tool names, the 160-line default, explicit limits, and the 50,000-character cap.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/ds4py/capabilities/file_system.py src/ds4py/agent.py tests/agent/test_file_system.py tests/agent/test_agent_factory.py
git commit -m "feat(agent): bound default file reads"
```

### Task 3: Add C-Style Default Coding-Agent Instructions

**Files:**
- Create: `src/ds4py/agent_prompt.py`
- Modify: `src/ds4py/agent.py:82-85`
- Test: `tests/agent/test_agent_factory.py:164-184`
- Create: `tests/agent/test_prompt_alignment.py`

**Interfaces:**
- Consumes: Task 1's instruction-before-tool ordering and Task 2's 160-line/50,000-character read contract.
- Produces: `DEFAULT_CODING_AGENT_INSTRUCTIONS: str`, used only when `build_agent_components(..., system=None)` receives no explicit override.

- [ ] **Step 1: Write failing default-prompt and override tests**

In `test_agent_factory.py`, add a test that a default run's first `ModelRequest.instructions` equals `DEFAULT_CODING_AGENT_INSTRUCTIONS`. Retain the existing explicit-override assertion that `"persistent rules"` remains the complete instructions value.

In `test_prompt_alignment.py`, render `DEFAULT_CODING_AGENT_INSTRUCTIONS` and one `read_file` definition through `PromptBuilder` with the fake DeepSeek engine. Decode the result and assert:

- C's exact opening line is present:

```text
You are a coding agent running in a local workspace. Use tools for local file and system work. Avoid printing large file contents or large code blocks as answers; create or edit files with tools, then summarize results briefly.
```

- The opening precedes `You have access to native DSML tools.`, which precedes the serialized `read_file` schema.
- The following exact read paragraph is present:

```text
read_file returns a bounded chunk, rather than unconditionally reading the whole file. For first looks at large files, prefer read_file with an explicit limit around 80-160. By default, read_file returns at most 160 lines and 50,000 characters. If the result says more lines are available, call read_file again with the offset shown in the result. A character-limited read ends on the last complete line that fits.
```

- Editing guidance names `edit_file`, `old_text`, and `new_text`; command guidance names `start_command`, `check_command`, and `stop_command`; general rules name `read_file`, `search_files`, and `edit_file`.
- The prompt does not advertise `whole=true`, `raw=true`, `more` continuation, `google_search`, or `visit_page`.

- [ ] **Step 2: Run the new tests and confirm the missing default fails**

Run:

```bash
uv run --extra agent pytest tests/agent/test_prompt_alignment.py tests/agent/test_agent_factory.py::test_factory_applies_persistent_instructions -v
```

Expected: FAIL because no default instructions module/value exists; the explicit-override test remains the control.

- [ ] **Step 3: Implement the agent-owned prompt and default selection**

Create `DEFAULT_CODING_AGENT_INSTRUCTIONS` with:

- C's opening verbatim.
- The approved read paragraph verbatim, now matching Task 2's actual behavior.
- C's editing workflow adapted to `edit_file(path, old_text, new_text)` without unsupported `raw` advice.
- C's long-running-command workflow adapted to `start_command`, `check_command`, and `stop_command`.
- C's reliable-code and system-integrity rules, with `read_file`, `search_files`, and `edit_file` substituted for C names.
- A final identity line naming `ds4-py`, not `ds4-agent`.

In `build_agent_components`, always set `options["instructions"]` to the explicit `system` when provided, otherwise to `DEFAULT_CODING_AGENT_INSTRUCTIONS`.

- [ ] **Step 4: Run agent prompt and composition tests**

Run:

```bash
uv run --extra agent pytest tests/agent/test_prompt_alignment.py tests/agent/test_agent_factory.py tests/agent/test_file_system.py tests/agent/test_binding.py tests/agent/test_binding_rewrite.py -v
```

Expected: PASS, including exact custom-instruction override, unchanged binding, and agreement between the read guidance and real read behavior.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/ds4py/agent_prompt.py src/ds4py/agent.py tests/agent/test_prompt_alignment.py tests/agent/test_agent_factory.py
git commit -m "feat(agent): add aligned coding instructions"
```

## Final Verification

- [ ] Run the focused prompt, parser, binding, and agent suites:

```bash
uv run --extra agent pytest tests/engine/test_tool_prompt.py tests/engine/test_prompt.py tests/engine/tool_syntax tests/agent/test_prompt_alignment.py tests/agent/test_file_system.py tests/agent/test_agent_factory.py tests/agent/test_binding.py tests/agent/test_binding_rewrite.py -v
```

- [ ] Run the complete non-live, non-build suite:

```bash
uv run --extra agent pytest -m "not live and not build"
```

- [ ] Run static checks and whitespace validation:

```bash
uv run ruff check src tests
uv run pyright
git diff --check HEAD~4..HEAD
```

Expected: all commands exit successfully. Record the results in the completion report. Do not infer live token-use improvements from unit tests; evaluate those separately in an explicitly approved bake-off.
