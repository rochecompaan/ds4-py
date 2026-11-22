# Hermes Tool Calls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add generic Mellum identity, strict Hermes streaming tool calls, bounded quote repair, Hermes prompt/history rendering, and portable regression coverage.

**Architecture:** Extend the existing native info and `ToolParser` contracts, then add Hermes branches to the established prompt and message renderers. Keep caller payloads plain and structural markers trusted; reuse generic compaction masking and the deterministic fake native vocabulary.

**Tech Stack:** Python 3.14, ctypes, pytest, JSON, devenv, uv, prek.

**Spec:** `docs/specs/2026-11-22-hermes-tool-calls-design.md`

## Global Constraints

- Base is `origin/main` at `a164141e70722590c8bf1aa6263ce041afe0502f`; follow current APIs rather than old line numbers.
- Mellum may set `is_qwen4=True`, but Hermes family and parser selection take precedence.
- Duplicate JSON keys, including duplicate `arguments`, keep the last value.
- No downstream product identifier enters `src/` or `tests/`; no tool-profile work enters this PR.
- Do not modify the reference checkout or run a live model.
- Run project commands inside devenv with `uv --frozen --extra agent`.
- Write each focused failing test before its runtime change and keep commits small.
- The `src/ds4py/message_rendering.py` overlap with RollingWindow is explicitly authorized; do not restructure that file.

## Review Focus

- **Legacy native library without `ds4_engine_is_mellum`:** opening must preserve today's family and never call the missing symbol. Task 1 pins the compiled absent-symbol case.
- **Mellum plus Qwen flags:** both native family selection and parser construction must choose Hermes. Tasks 1 and 2 pin both paths.
- **Malformed, truncated, deeply nested, NUL-containing, or marker-bearing JSON:** never publish a partial call; marker text inside strings stays data. Task 2 pins these inputs at chunk sizes 1, 7, and 4096.
- **Bare quotes in write/edit payloads:** repair only the evidenced shapes and reject path or member-folding ambiguity. Task 3 pins accepted and refused vectors.
- **Untrusted prompt/history payloads and grouped results:** markup cannot inject trusted tokens, imported reasoning is dropped, and results keep individual response boundaries. Tasks 4 and 5 pin exact text and fake-token spans.

---

### Task 1: Mellum identity and family precedence

**Files:**
- Modify: `src/ds4py/_native/protocol.py`
- Modify: `src/ds4py/_native/bindings.py`
- Modify: `src/ds4py/_native/api.py`
- Modify: `tests/native/api_stub.c`
- Modify: `tests/native/conftest.py`
- Modify: `tests/native/test_bindings.py`
- Modify: `tests/abi/test_probe.py`, `tests/abi/test_manifest.py`, `tests/abi/test_check.py`, `tests/abi/test_build.py`
- Modify: `tests/fixtures/ds4-headers/fork/ds4.h`
- Test: `tests/native/test_api.py`, `tests/engine/tool_syntax/test_qwen.py`

**Interfaces:**
- Consumes: `Library.has(name)`, `Library.fn(name)`, and `CORE_BINDINGS`.
- Produces: `EngineInfo(..., is_mellum: bool = False)` with the new field last; `family_of(*, deepseek41: bool, glm: bool, qwen: bool, mellum: bool = False) -> str` returning `hermes` first.

- [ ] **Step 1: Write the failing identity tests**

Add a second compiled-stub fixture using `-DSTUB_MELLUM`. Under that define, make `ds4_engine_is_qwen4` true and export `ds4_engine_is_mellum` returning true. Assert:

```python
assert legacy.info().is_mellum is False
assert legacy.info().family == "glm"
assert mellum.info().is_mellum is True
assert mellum.info().is_qwen4 is True
assert mellum.info().family == "hermes"
```

Keep the existing info-read-once assertion and add direct cases proving every old `family_of` result is unchanged when `mellum` is omitted or false. Update the exact binding census, optional-absent sets, ABI report lines, and build assertion to expect the new optional predicate; declare it in the fork fixture header, leaving the upstream fixture header without it.

- [ ] **Step 2: Run and verify RED**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/native/test_api.py -q`
Expected: FAIL because `EngineInfo` has no `is_mellum` and `family_of` has no Mellum precedence.

- [ ] **Step 3: Implement optional detection**

Add `Binding("ds4_engine_is_mellum", "bool ds4_engine_is_mellum(ds4_engine *e)", required=False)`. In `_read_info`, call it only when `self._lib.has(...)` is true. Pass `mellum=` to `family_of` and append the field to `EngineInfo` with default false.

- [ ] **Step 4: Run identity and unchanged-family regressions**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/native tests/abi tests/engine/tool_syntax -q -m "not build"`
Expected: PASS; all ABI census checks and existing DeepSeek, GLM, and Qwen cases remain consistent.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/_native/protocol.py src/ds4py/_native/bindings.py src/ds4py/_native/api.py tests/native/api_stub.c tests/native/conftest.py tests/native/test_api.py tests/native/test_bindings.py tests/abi/test_probe.py tests/abi/test_manifest.py tests/abi/test_check.py tests/abi/test_build.py tests/fixtures/ds4-headers/fork/ds4.h
git commit -m "feat(native): identify Mellum and select the hermes protocol family"
```

### Task 2: Strict Hermes tool-call parser

**Files:**
- Create: `src/ds4py/engine/tool_syntax/hermes.py`
- Create: `tests/engine/tool_syntax/test_hermes.py`
- Create: `tests/fixtures/tool_syntax/hermes.json`
- Modify: `src/ds4py/engine/tool_syntax/base.py`
- Modify: `src/ds4py/engine/tool_syntax/model.py`
- Modify: `tests/engine/tool_syntax/conftest.py`
- Test: `tests/engine/tool_syntax/test_stream_equivalence.py`, `tests/engine/tool_syntax/test_qwen.py`

**Interfaces:**
- Consumes: `ToolParser`, `ToolParserState`, `freeze_calls`, `ToolCall`, and `EngineInfo.is_mellum`.
- Produces: `Hermes(ToolParser)` with complete capture/restore/fork state; `make_parser(info)` returns `Hermes()` for Mellum before Qwen.

- [ ] **Step 1: Create C-derived fixtures and failing tests**

Use the current fixture schema (`source`, `input`, `state`, `calls`, optional `error`). Cover prose around calls, multiple calls, JSON escapes, markers inside strings, typed nested values, string-encoded arguments, unknown top-level keys, duplicate `arguments`, duplicate argument members, invalid numbers/constants, raw and escaped NUL, trailing commas, backticks, truncated input, and a 5,000-level value. Assert later duplicate values win. Add a call opener inside `<think>` and pin ds4-py's existing `ERROR`/no-calls rule rather than the C agent's ignore-and-warn behavior.

In `test_hermes.py`, drive every fixture at chunk sizes 1, 7, and 4096; assert typed args, exact `raw`, and no calls on errors. Add the 256/257 nesting boundary, greedy structural versus string-value sampling, mid-call capture/restore/fork equivalence, and Mellum-before-Qwen selection.

- [ ] **Step 2: Register fixtures and verify RED**

Add `hermes` to the shared parser fixture loader/factory so the existing single-split, bytewise, and 100 seeded random-partition tests also run.

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/engine/tool_syntax/test_hermes.py -q`
Expected: FAIL because `ds4py.engine.tool_syntax.hermes` does not exist.

- [ ] **Step 3: Implement the strict incremental parser**

Subclass `ToolParser` directly. Track the current stanza, JSON scanner phase, after-call prose, and all partial lexical state in an immutable Hermes family snapshot. Parse exactly one JSON object per call and publish only after `</tool_call>`.

Use `json.loads` with `parse_constant` rejection, require a nonempty string `name` and object `arguments`, decode string-encoded arguments as one object, recursively reject NUL and nesting above 256, preserve insertion order and typed values, and rely on default last-key-wins behavior. Catch JSON `RecursionError` and return an `ERROR` snapshot. Keep literal markers inside JSON strings inert. `finish()` rejects an unfinished active call without publishing it. Set `ToolCall.raw` to the complete `<tool_call>` through `</tool_call>` stanza, and update the model docstring to name Hermes alongside GLM/Qwen.

Register Hermes before `is_qwen4` in `make_parser`.

- [ ] **Step 4: Run parser and stream-equivalence regressions**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/engine/tool_syntax -q`
Expected: PASS for every chunking and all existing families.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/engine/tool_syntax/hermes.py src/ds4py/engine/tool_syntax/base.py src/ds4py/engine/tool_syntax/model.py tests/engine/tool_syntax/test_hermes.py tests/engine/tool_syntax/conftest.py tests/fixtures/tool_syntax/hermes.json
git commit -m "feat(tool-syntax): parse Hermes tool calls strictly"
```

### Task 3: Bounded quote repair for write/edit

**Files:**
- Modify: `src/ds4py/engine/tool_syntax/hermes.py`
- Create: `tests/engine/tool_syntax/test_hermes_repair.py`

**Interfaces:**
- Consumes: strict `Hermes` parsing from Task 2.
- Produces: `repair_hermes_quotes(text: str) -> str | None`; `text` is the object body beginning at `{`, without either tool-call tag. `None` means unrepairable. The parser applies it only to a complete eligible write/edit body after strict parsing fails.

- [ ] **Step 1: Write failing direct and streaming repair tests**

Port the reference vectors for write `content`, edit `edits[]` old/new text, and legacy single-edit old/new text. Assert the exact repaired JSON and resulting typed values at chunk sizes 1, 7, and 4096. Add refusals for a path value, ambiguous edit boundary, folded `path`/`content`/`edits` member, backticks, NUL, truncated input, and an already-valid call (no repair needed).

- [ ] **Step 2: Run and verify RED**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/engine/tool_syntax/test_hermes_repair.py -q`
Expected: FAIL because `repair_hermes_quotes` is absent and malformed bare-quote calls do not parse.

- [ ] **Step 3: Implement only the bounded reference forms**

Recognize the exact write/edit member order and valid path boundary from the reference, identify unambiguous old/new value boundaries, and escape only bare quotes inside those values. Do not invoke a generic JSON-repair library and do not repair an already-valid JSON string, a path, or ambiguous/member-folding content. Re-run strict parsing on the repaired body before publishing.

- [ ] **Step 4: Run all parser tests and commit**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/engine/tool_syntax -q`
Expected: PASS, including all strict malformed-input guards.

```bash
git add src/ds4py/engine/tool_syntax/hermes.py tests/engine/tool_syntax/test_hermes_repair.py
git commit -m "feat(tool-syntax): repair the bounded Hermes quote forms"
```

### Task 4: Hermes tool prompt and single initial system turn

**Files:**
- Modify: `src/ds4py/engine/tool_prompt.py`
- Modify: `src/ds4py/engine/prompt.py`
- Modify: `src/ds4py/_native/fake_chat.py`
- Create: `tests/fixtures/hermes_prompt.json`
- Test: `tests/engine/test_tool_prompt.py`, `tests/engine/test_prompt.py`

**Interfaces:**
- Consumes: `ToolPromptGenerator.spans`, `ToolPromptSpan(text, rendered)`, and Qwen-style trusted `<|im_start|>`/`<|im_end|>` helpers.
- Produces: `ToolPromptGenerator._hermes(schemas) -> list[ToolPromptSpan]`; one Hermes system segment containing instructions and tools; Mellum-specific fake assistant-prefix behavior for think on/off.

- [ ] **Step 1: Add exact reference fixtures and failing prompt tests**

Vendor the reference tool definitions and expected tools/system text. Assert standard-spaced `json.dumps`, Unicode preservation, insertion order, function wrappers, and `rendered=False` on schema/instruction payloads. Assert fixed tool markers use dedicated fake special tokens, `begin()` emits one system turn with no unwrapped tools fragment, think on/off prefixes equal the static reference text, and `ThinkMode.MAX` adds no DeepSeek reasoning-effort text.

- [ ] **Step 2: Run and verify RED**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/engine/test_tool_prompt.py tests/engine/test_prompt.py -q`
Expected: FAIL with `unknown tool prompt family: hermes` and missing Hermes begin/prefix behavior.

- [ ] **Step 3: Implement generation and one-turn begin**

Serialize Hermes definitions separately from the existing compact families so old output stays byte-identical. Add `_hermes` with the reference head/tail and standard-spaced wrappers. In `PromptBuilder.begin`, return one Hermes system turn, plain instructions followed by `\n\n` when present, then tool spans and one close; skip non-Hermes think-prefix injection.

Extend only the fake native Hermes prefix: think on opens an assistant turn without injecting `<think>`; think off emits `<think>\n\n</think>\n\n`.

- [ ] **Step 4: Run prompt regressions and commit**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/engine/test_tool_prompt.py tests/engine/test_prompt.py -q`
Expected: PASS, including exact reference text and trusted/plain fake-token assertions. No live token-id claim.

```bash
git add src/ds4py/engine/tool_prompt.py src/ds4py/engine/prompt.py src/ds4py/_native/fake_chat.py tests/engine/test_tool_prompt.py tests/engine/test_prompt.py tests/fixtures/hermes_prompt.json
git commit -m "feat(prompt): render the Hermes tools block in one system turn"
```

### Task 5: Hermes assistant and grouped tool history

**Files:**
- Modify: `src/ds4py/engine/prompt.py`
- Modify: `src/ds4py/engine/tool_syntax/render.py`
- Modify: `src/ds4py/message_rendering.py`
- Create: `tests/fixtures/hermes_history.json`
- Test: `tests/engine/test_prompt.py`, `tests/engine/tool_syntax/test_call_raw.py`, `tests/agent/test_tool_render.py`

**Interfaces:**
- Consumes: `PromptBuilder.message`, `tool_results`, `restored_assistant`, `MessageRenderer.request_segments`, and sampled `ds4py.raw` details.
- Produces: `render_tool_call("hermes", name, args) -> str`; one user turn containing one `<tool_response>` per consecutive result; restored Hermes calls accepted by `Hermes`; imported reasoning through the first `</think>` dropped.

- [ ] **Step 1: Write failing history and round-trip tests**

Vendor the reference assistant/tool history text and pin the exact close/EOS sequence. Assert exact restored call JSON with standard separators, sampled raw preservation, escaping of literal `</tool_response>` in plain payloads, one-user/two-response rendering without joining results by blank lines, first-`</think>` reasoning removal, and EOS closure. Round-trip restored non-sampled calls through `Hermes` whole and bytewise. Add two adjacent `ToolCallPart`s and assert exactly one newline between their stanzas.

For a restored argument containing `</tool_call>` or `<|im_end|>`, assert JSON uses `\u003c` and no injected dedicated close/start token occurs inside the string; decoding must still return the original value. This intentionally sacrifices verbatim restored-reference spelling to preserve ds4-py's trusted-markup boundary.

Add a `MessageRenderer`/`MessageBinder` test with two `ToolReturnPart`s to prove restored history retains two responses, not one pre-joined payload.

- [ ] **Step 2: Run and verify RED**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/engine/test_prompt.py tests/engine/tool_syntax/test_call_raw.py tests/agent/test_tool_render.py -q`
Expected: FAIL because Hermes is rejected by `render_tool_call` and history uses Qwen/native fallbacks.

- [ ] **Step 3: Implement Hermes history branches**

Render `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>` with `ensure_ascii=False`, standard separators, JSON validation, no non-finite values, and `<` escaped as `\u003c` in restored JSON. Reuse Qwen-style open/close token helpers but keep assistant/tool payloads plain. Give `message(..., "assistant")` and `restored_assistant` explicit Hermes branches: open with `_qwen_open("assistant")`, remove and later restore `_qwen_close()`, and never append a body after a closed turn. Add the reference newline between adjacent restored Hermes call spans.

For Hermes, make `tool_results` render each result separately in one user turn and make `_tool_message` delegate a single result. In `MessageRenderer.request_segments`, pass the result list directly to `tool_results` only for Hermes; preserve every other family's existing joined rendering. Remove imported reasoning through the first close and keep sampled raw call spans unchanged.

- [ ] **Step 4: Run history/parser regressions and commit**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/engine/test_prompt.py tests/engine/tool_syntax tests/agent/test_tool_render.py -q`
Expected: PASS for Hermes and every existing family.

```bash
git add src/ds4py/engine/prompt.py src/ds4py/engine/tool_syntax/render.py src/ds4py/message_rendering.py tests/engine/test_prompt.py tests/engine/tool_syntax/test_call_raw.py tests/agent/test_tool_render.py tests/fixtures/hermes_history.json
git commit -m "feat(prompt): render Hermes assistant and grouped tool history"
```

### Task 6: Generic Hermes regression gate

**Files:**
- Modify: `src/ds4py/engine/compaction.py` only if a characterization test proves a current gap
- Test: `tests/engine/test_compaction.py`, `tests/engine/tool_syntax/test_stream_equivalence.py`

**Interfaces:**
- Consumes: Tasks 1–5 and the current generic `_TOOL_MARKERS` mask.
- Produces: portable proof that Hermes summary generation masks dedicated `<tool_call>`/`</tool_call>` tokens and sanitizes their literal spellings.

- [ ] **Step 1: Add Hermes masking characterization cases**

Parametrize the dedicated-token and literal byte/mixed/split marker tests with `family="hermes"`, `is_qwen4=True`, `is_mellum=True` for both tool-call markers. Assert the summary and emitted token events stop before the marker and retain only `safe`.

- [ ] **Step 2: Run the focused characterization tests**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/engine/test_compaction.py -q`
Expected on current `main`: PASS without a production edit because `_TOOL_MARKERS` already contains both markers. If a case fails, reduce it and make the smallest Hermes-specific mask fix; do not add redundant constants without a demonstrated gap.

- [ ] **Step 3: Run the complete static gate**

Run: `devenv shell -- uv run --frozen --extra agent prek run --all-files --show-diff-on-failure`
Expected: PASS for all hooks. Keep formatting-only changes separate from behavioral commits.

- [ ] **Step 4: Run the complete portable gate and genericity check**

```bash
mkdir -p /tmp/ds4-quality-empty-home
devenv shell -- env HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest -m "not build and not live" -q
grep -Rni "sat""yrn" src tests
```

Expected: portable suite PASS with counts recorded; grep finds nothing. Do not call this live parity.

- [ ] **Step 5: Commit the regression coverage**

```bash
git add src/ds4py/engine/compaction.py tests/engine/test_compaction.py
git commit -m "test(compaction): cover Hermes tool marker masking"
```

## Delivery and review

- [ ] Review the diff against the spec and list every actual departure, including any discovered during implementation.
- [ ] Push `feat/hermes-tool-calls` and open the PR with a short body containing the feature summary, departures, exact static/portable gate counts, and the full downstream issue URL from the handoff. Add no tool or agent trailers.
- [ ] Run the repository `pr-review-loop` skill, alternate the two reviewer models specified by the handoff, ask the owner for every decline/defer ruling, and record all findings in the PR evidence.
- [ ] Report the PR URL, head commit, duplicate-key ruling, departures, and gate counts. After the review loop, read the separate tool-profile handoff requested by the owner.
