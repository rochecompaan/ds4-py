# PR #30 Qwen Review Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all seven PR #30 review threads with a single-parse Qwen value helper, fixture-driven edge coverage, renderer cleanup, verification, replies, and a pushed fix commit.

**Architecture:** A single helper owns Qwen value trimming, numeric recognition, and strict JSON conversion. Both the streaming parser and history renderer consume its `(value, is_string)` result; fixture cases provide the behavioral matrix and existing stream-equivalence tests provide chunk invariance.

**Tech Stack:** Python 3.14, pytest, JSON fixtures, devenv, uv, prek, GitHub CLI.

**Spec:** `docs/superpowers/specs/2026-11-11-pr30-qwen-review-design.md`

## Global Constraints

- After the pinned trim, full matches of `agent_qwen_value_is_json`'s ASCII numeric grammar follow its 64-byte cap; other values use `ds4_server.c:qwen_param_value_is_json`-style complete strict JSON parsing with trailing content, top-level strings, and non-finite constants kept as text.
- Schema-declared string support is out of scope and remains documented in the PR body.
- Use `TypeIs`, never `TypeGuard`; no broad suppressions or unsafe fixes.
- Run commands from the repository root through the existing devenv + uv environment.
- Do not touch the pre-existing untracked `.ds4py-devenv/`, `.ds4py/`, `c-agent.trace`, or `docs/reviews/` paths.

## Review Focus

- Numeric-looking values at 63 and 64 bytes: preserve the agent boundary; Task 2 keeps `test_numeric_classification_uses_the_c_64_byte_guard` and adds direct helper assertions.
- Valid non-string JSON and malformed/trailing JSON: convert only complete supported values; Tasks 1 and 2 cover these through fixtures and direct helper cases.
- Top-level JSON strings, `NaN`, and `Infinity`: keep text under the deliberate strict rule; Tasks 1 and 2 add explicit cases with accurate labels.
- BOM/NUL and whitespace variants: preserve text or use exactly the documented trimming rule; Tasks 1 and 2 exercise the edge vectors and asymmetric trimming.
- History rendering of typed values and unrepresentable strings: retain existing round trips and errors while consuming the new helper; Task 2 adds a 64-digit string case and runs `tests/agent/test_tool_render.py`.
- Extremely nested complete JSON: CPython 3.14.7 raises `RecursionError` at 200,000 array levels; Task 2 keeps the guard and pins degradation to text in the direct helper test rather than adding a very large stream-equivalence fixture.

---

### Task 1: Consolidate Qwen Fixture Coverage

**Files:**
- Modify: `tests/fixtures/tool_syntax/qwen.json`
- Modify: `tests/engine/tool_syntax/test_qwen.py`

**Interfaces:**
- Consumes: existing `test_qwen_fixtures` and `test_stream_equivalence` fixture loaders.
- Produces: fixture cases for malformed JSON, trailing content, top-level strings, non-finite constants, BOM, and NUL; no production interface changes.

- [ ] **Step 1: Move the eight `test_incomplete_or_string_json_values_stay_text` vectors into `qwen.json`**

Give each case a literal input, `DONE` state, and literal string argument expectation. Reuse and relabel the existing BOM and malformed-bracket fixtures rather than appending duplicates; add the remaining six vectors (`[1,]`, trailing content, top-level string, `NaN`, `Infinity`, and NUL). Use labels such as `strict JSON: NaN stays text` rather than attributing mixed-rule behavior to `ds4-server`. Keep the existing Jinja fixture and delete its byte-for-byte duplicate `test_issue_29_jinja_write_file_repro_stays_text` plus `test_incomplete_or_string_json_values_stay_text`. Also delete `test_complete_non_string_json_values_still_parse`, whose values are already covered by the JSON-kind and numeric fixtures; keep the distinct C numeric-forms test.

- [ ] **Step 2: Run fixture and stream-equivalence tests**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/engine/tool_syntax/test_qwen.py tests/engine/tool_syntax/test_stream_equivalence.py -q`
Expected: PASS; the moved cases run at chunk sizes 1, 7, and 4096 and through split/random-partition equivalence.

- [ ] **Step 3: Commit the test consolidation**

```bash
git add tests/fixtures/tool_syntax/qwen.json tests/engine/tool_syntax/test_qwen.py
git commit -m "test: consolidate Qwen edge fixtures"
```

### Task 2: Replace Classification and Conversion with One Parse

**Files:**
- Modify: `src/ds4py/engine/tool_syntax/qwen.py`
- Modify: `src/ds4py/engine/tool_syntax/render.py`
- Modify: `tests/engine/tool_syntax/test_qwen.py`
- Modify: `tests/agent/test_tool_render.py`

**Interfaces:**
- Consumes: fixture behavior from Task 1 and existing renderer contracts.
- Produces: `parse_qwen_value(text: str) -> tuple[object, bool]`, where the flag is `True` for text. Text results contain the original, still-escaped input; `Qwen._parse_parameter_value` performs delimiter unescaping.

- [ ] **Step 1: Write failing direct helper tests**

Import `parse_qwen_value` inside a new parameterized test so collection of existing tests remains usable. Assert literal `(value, is_string)` results for `42`, `+1`, `01`, `1.`, `.5`, a 64-digit number, `true`, `null`, an object, an array, a top-level string, malformed JSON, trailing JSON, `NaN`, `Infinity`, and empty text. Include an asymmetric-whitespace case proving only leading space/tab and trailing space/tab/newline are trimmed. Add a separate deep-nesting test using 200,000 complete array levels and expecting the original text; it must prove the `RecursionError` guard without entering the O(n²) every-split fixture path. Add `"9" * 64` to `test_qwen_text_arguments_round_trip_through_history` to pin the renderer's string-side cap behavior.

The production break each test catches is a wrong numeric boundary, acceptance of non-strict JSON, loss of original text, inconsistent trimming between parsing and rendering, or an escaping `RecursionError`.

- [ ] **Step 2: Run the helper test and verify RED**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/engine/tool_syntax/test_qwen.py::test_parse_qwen_value tests/engine/tool_syntax/test_qwen.py::test_parse_qwen_value_deep_nesting_stays_text -q`
Expected: FAIL because `parse_qwen_value` does not exist. Python 3.14 permits the existing unparenthesized multi-exception syntax; the baseline focused and portable suites have already imported this module successfully.

- [ ] **Step 3: Implement the single-parse helper**

In `qwen.py`, change `_NUMBER` to an ASCII-only `str` pattern retaining `[0-9]` and implement `parse_qwen_value(text: str) -> tuple[object, bool]`. Trim leading space/tab and trailing space/tab/newline once. Return `(text, True)` for empty text. If the numeric regex matches, return `_parse_number(trimmed), False` only below 64 characters; otherwise return `(text, True)` without falling through to JSON. For all other values, call `json.loads(trimmed, parse_constant=_reject_json_constant)` once, catch `(ValueError, RecursionError)`, and return `(parsed, False)` only for non-strings. Otherwise return `(text, True)`. Remove `value_is_json` but retain the evidenced recursion guard. The docstring must name `agent_qwen_value_is_json` for the full-match ASCII numeric/64-byte branch and `ds4_server.c:qwen_param_value_is_json` for the complete strict JSON branch, state that top-level strings/non-finite constants/trailing content stay text, and avoid claiming the cap applies to non-matching whitespace variants.

Update `_parse_parameter_value` to call the helper once, unescape only text results, and set `ToolArgument.is_string` from the returned flag. Remove its second numeric check and JSON parse.

- [ ] **Step 4: Update the renderer to consume the helper**

In `render.py`, import `parse_qwen_value`. For string arguments, reject when `is_string` is false, then assign escaped text once. For non-string arguments, assign `_json(value)` once, parse it with the helper, and reject when `is_string` is true. Remove the duplicate `isinstance` check and reassignment.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `devenv shell -- uv run --frozen --extra agent pytest tests/engine/tool_syntax/test_qwen.py tests/engine/tool_syntax/test_stream_equivalence.py tests/agent/test_tool_render.py -q`
Expected: PASS with no parser exceptions or renderer round-trip regressions.

- [ ] **Step 6: Commit the implementation**

```bash
git add src/ds4py/engine/tool_syntax/qwen.py src/ds4py/engine/tool_syntax/render.py tests/engine/tool_syntax/test_qwen.py
git commit -m "refactor: parse Qwen parameter values once"
```

### Task 3: Verify, Review, Push, and Reply

**Files:**
- Modify only if verification or final review produces a evidenced finding within the approved scope.

**Interfaces:**
- Consumes: Tasks 1 and 2 commits and all seven GitHub inline comment IDs.
- Produces: pushed commits on `fix/qwen-json-param-values` and one inline reply per review thread.

- [ ] **Step 1: Run the complete static gate**

Run: `devenv shell -- uv run --frozen --extra agent prek run --all-files --show-diff-on-failure`
Expected: all hooks PASS.

- [ ] **Step 2: Run the portable regression suite**

Run: `devenv shell -- env HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest -m "not build and not live" -q`
Expected: all selected tests PASS; report the exact totals. Setting HOME only for the test process prevents personal `~/.claude/skills` from changing agent instructions without hiding devenv/Nix's cache.

- [ ] **Step 3: Run the required final diff review**

Run `/review-loop` on the PR follow-up diff with one structural/maintainability reviewer using `thermo-nuclear-code-quality-review` and one correctness/tests reviewer. Fix evidenced P0 findings and worthwhile P1 findings within scope; record any design expansion instead of implementing it without approval.

- [ ] **Step 4: Push the approved branch**

Run: `git push origin fix/qwen-json-param-values`
Expected: remote PR head includes all follow-up commits and working tree has no new tracked modifications.

- [ ] **Step 5: Reply in every inline review thread**

Use `gh api repos/rochecompaan/ds4-py/pulls/30/comments/{id}/replies` for IDs `4535323767`, `4535332078`, `4535340444`, `4535346949`, `4535361192`, `4535364283`, and `4535371257`. Reference the concrete fix or, for schema-declared strings, the existing PR-body follow-up and intentional scope limit. Do not post a substitute top-level PR comment.
