# Coding-agent prompt and read-guidance alignment

**Status:** Approved
**Milestone:** M3
**Related evidence:** `docs/reports/2026-10-31-bakeoff-8192-analysis.md`

## Intent

Reduce avoidable differences between the C and Python coding-agent system prompts before another bake-off. Port C's useful coding workflow and bounded-read guidance to ds4-py without renaming or expanding the Python toolset.

Read caps directly bound tool-result input, not generated output. Any reduction in generated reasoning tokens is an indirect hypothesis to test, not a promised outcome.

## Goals

- Give the default ds4-py coding agent C-style opening instructions: work in the local workspace, use tools for file and system work, avoid large answers, edit files with tools, and summarize briefly.
- Preserve the agreed ownership boundary:
  - the engine owns family-specific native tool syntax, parameter encoding, escaping, and schema rendering;
  - the agent layer owns coding workflow and tool-use guidance.
- For DeepSeek DSML families, explicitly tell the model that it has native DSML tools and align the invocation guidance with C while retaining trusted rendering of DSML markup.
- Keep the existing Python tool names and describe their actual behavior.
- Make `read_file` default to 160 lines while retaining the existing 50,000-character result cap.
- Explain bounded reads and continuation through `offset` in both the agent guidance and the tool schema where applicable.
- Ensure the `read_file` schema describes the configured default instead of the currently hardcoded 2,000-line default.

## Non-goals and anti-goals

- Do not rename `read_file`, `write_file`, `edit_file`, `run_command`, or other Python tools to C names.
- Do not add C's `whole`, `raw`, `start_line`, `max_lines`, or `more` interfaces.
- Do not add web tools or claim unsupported tool behavior.
- Do not replace the 50,000-character cap with C's larger 128 KiB byte cap.
- Do not hardcode Python coding rules or Python tool names in the low-level engine.
- Do not put trusted DSML markup in plain caller instructions.
- Do not claim that prompt alignment alone will prove generation parity or isolate the cause of token-use differences.
- Do not start a paid GPU instance or full bake-off automatically.

## Approaches considered

### Prompt-only alignment

Port the wording but leave the 2,000-line default unchanged. This is the smallest change, but it would not test the C-like bounded-read behavior requested here.

### Prompt alignment plus a smaller read default — selected

Port the coding guidance, set the read default to 160 lines, retain the 50,000-character cap, and make the schema truthful. This measures the combined prompt and read-default change. It does not attribute an improvement to either change independently.

### Staged configurable experiment

Add switches and test prompt-only and read-default changes separately. This gives cleaner attribution but adds configuration and requires more live runs. It is unnecessary for the requested first alignment pass.

## Agreed design

### Agent instructions

Use the C-style coding-agent instructions by default when `build_agent_components` receives no explicit `system` value. An explicit `system` argument keeps its current replacement semantics.

Include this literal read guidance:

```text
read_file returns a bounded chunk, rather than unconditionally reading the whole file. For first looks at large files, prefer read_file with an explicit limit around 80-160. By default, read_file returns at most 160 lines and 50,000 characters. If the result says more lines are available, call read_file again with the offset shown in the result. A character-limited read ends on the last complete line that fits.
```

Adapt C's editing and command guidance to Python's actual names and behavior. In particular, `run_command` guidance must describe shell execution only to the extent supported by its implementation. Do not mention unavailable C-only parameters or continuation tools.

### Native tool instructions

Keep family-specific rendering in `ToolPromptGenerator` and `PromptBuilder`. For DeepSeek DSML, port C's native-tool statement, invocation shape, thinking/tool-call boundary, parameter typing, and escaping rules. Keep DSML markup in trusted template spans.

The final system prompt must contain the agent guidance and native tool instructions before the tool schemas. Exact byte-for-byte equality with C is not required because tool names, schema formats, and supported behavior differ.

### Read behavior and schema

Configure the default filesystem capability with:

```python
max_read_lines=160
max_read_chars=50_000
```

Keep `path`, `offset`, and `limit` unchanged. `offset` remains zero-based, and `path` remains the only required parameter. Explicit limits continue to override the default.

Make the only intended literal read-schema text change:

```diff
-"description": "Maximum number of lines to return (default: 2000)."
+"description": "Maximum number of lines to return (default: 160)."
```

The JSON schema's `"default": null` remains unchanged: the tool function receives `None` when the argument is omitted, and the filesystem capability supplies the configured default. The 50,000-character cap remains in system guidance rather than in the parameter schema, matching C's placement of its cap outside the read schema.

The schema-description fix must follow the configured value rather than merely replace one hardcoded number with another. Keep this adaptation at ds4-py's agent/tool boundary unless the existing harness exposes a supported configuration hook; do not patch installed dependency files.

## Verification

Add focused automated tests that prove behavior and contracts:

1. The default rendered system prompt contains the C-style coding guidance, the native DSML statement for DeepSeek, and the exact read guidance; it uses Python tool names and does not advertise `whole`, `raw`, or `more`.
2. An explicit `system` argument retains its existing override behavior.
3. The exposed `read_file` schema keeps its existing parameter shape and reports the configured 160-line default.
4. An omitted `limit` returns at most 160 lines, an explicit limit still works, and the continuation hint identifies the correct next zero-based `offset`.
5. The existing 50,000-character result cap remains enforced, including its metadata header and continuation hint.

Run the relevant agent, prompt, binding, and replay tests plus the project's normal lint and type checks. Treat any live bake-off as a separate step after local verification and explicit confirmation of the remote instance.

## Builder choices

The builder may choose the smallest supported mechanism for composing agent instructions before tool schemas and for preparing the configured read-schema description. Those choices must preserve the ownership boundary, public tool schema shape, and existing engine parsing behavior.
