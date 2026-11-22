# Hermes tool calls design

| | |
| --- | --- |
| Date | 2026-11-22 |
| Status | Approved |
| Branch | `feat/hermes-tool-calls` |
| Base | `origin/main` at `a164141e70722590c8bf1aa6263ce041afe0502f` |
| Reference | `ds4-mellum` at `d2d9338`, read-only |

## Intent

Add generic Hermes tool-call support so Mellum can use ds4-py's existing agent,
streaming parser, history, persistence, and compaction paths. Mellum keeps its
native `is_qwen4` graph identity, while its Python protocol family is `hermes`.

## Goals

1. Detect the optional native Mellum predicate without breaking older libraries.
2. Parse strict Hermes JSON calls incrementally, including immutable parser
   snapshots and the reference's bounded write/edit quote repair.
3. Render the Hermes tool prompt, one initial system turn, assistant calls, and
   grouped tool responses without turning caller payloads into trusted markup.
4. Pin compaction masking and existing-family regressions with portable tests.

## Non-goals

- No tool-profile option or Pi tool adapters; those belong to the next PR.
- No native engine, tokenizer, chat-template, sampling, or live-model changes.
- No second agent, runtime, or general-purpose JSON repair mechanism.
- No downstream product identifiers or policy in production code or tests.

## Constraints

- Keep the devenv + `uv --frozen --extra agent` workflow.
- Follow test-first development for every runtime behavior change.
- Preserve existing DeepSeek, GLM, and Qwen rendering byte for byte.
- Treat schemas, instructions, user text, and tool results as plain payloads.
- Use only static reference fixtures in the portable gate. Do not run a GGUF.
- Keep `src/ds4py/message_rendering.py` changes minimal and coordinate the known
  RollingWindow overlap; the owner explicitly authorized proceeding.

## Reference ruling: duplicate keys

The reference accepts duplicate keys with the last value winning. This includes
both repeated top-level `arguments` members and repeated members inside an
arguments object. The implementation will use that behavior, superseding the
conflicting rejection wording in the downstream tracking issue.

## Approaches

### Chosen: extend the existing Python protocol boundaries

Add a Hermes `ToolParser`, a tool-prompt branch, and Hermes branches in the
existing prompt/history renderer. Reuse `EngineInfo`, `ToolCall`, snapshot
freezing, trusted token spans, payload escaping, and compaction masking.

This keeps one agent and one history pipeline and makes existing-family
regressions executable. It also permits a strict JSON parser without importing
the C agent or server runtime.

### Rejected: port the C agent/server parser as a parallel runtime

That would duplicate execution, history, and rendering policy and would violate
the requirement that the C code remain reference material only.

### Rejected: treat Mellum as Qwen throughout

Mellum reports `is_qwen4=True`, but Qwen's function/parameter syntax and history
format are not Hermes. A Qwen fallback would silently misparse every tool call.

## Design

### Native identity

Declare `ds4_engine_is_mellum` as an optional binding. Append
`EngineInfo.is_mellum: bool = False` after all existing fields. `_read_info()`
returns false when the symbol is absent and otherwise calls it once with the
other info predicates. `family_of(..., mellum=True)` returns `hermes` before the
Qwen branch.

### Streaming parser and repair

Implement `Hermes` directly on `ToolParser`; `DelimiterParser`'s
function/parameter lifecycle does not fit one JSON object per call. Buffer and
scan one `<tool_call>` object at a time, publish a call only after its matching
close, and preserve the exact stanza as `ToolCall.raw`.

Use strict RFC 8259 decoding with non-finite constants rejected, typed nested
values, a JSON string containing an arguments object, unknown top-level members
skipped, a 256-level value-depth bound, and NUL rejection. JSON recursion errors
become parser errors, never exceptions escaping `feed()`. Prose and multiple
calls follow the reference. A call opener inside `<think>` retains ds4-py's
existing cross-family rejection rule rather than the C agent's ignore-and-warn
behavior. Snapshot capture/restore/fork must include every incremental scanner
field. `ToolCall.raw` includes both tool-call tags so sampled replay is valid.

On strict failure, defer only eligible write/edit bodies until the closing tag
and apply the reference's bounded bare-quote repair. Never repair `path`,
backticks, NUL, truncated input, ambiguous edit boundaries, or member-folding
values.

### Prompt and history

Serialize Hermes schemas with `json.dumps(..., ensure_ascii=False)` and its
standard `", "` / `": "` separators, wrapped as
`{"type": "function", "function": ...}`. Fixed template markers are trusted;
schema and instruction text are plain.

Render one initial `<|im_start|>system` turn containing optional instructions, a
blank line, and the tools block. Render restored calls as Hermes JSON stanzas,
with `\n` between adjacent calls. Escape `<` in restored JSON strings as
`\u003c` so payload text cannot become trusted special tokens; JSON decoding
preserves the value, although these restored bytes intentionally differ from
the reference's verbatim spelling. Keep sampled raw calls unchanged. Drop
imported reasoning through the first `</think>` and render consecutive tool
results as one user turn with one `<tool_response>` per result.
`MessageRenderer` must pass the individual results to `PromptBuilder` rather
than pre-joining them for Hermes.

### Compaction

Current `main` already includes `<tool_call>` and `</tool_call>` in the generic
summary mask for every family. Add Hermes characterization tests; no production
compaction edit is expected unless those tests expose a concrete gap.

## Current-code departures from the older plan

- Start at `a164141`, not `95d52cb` or `7d80c7c`.
- Use `tests/fixtures/tool_syntax/hermes.json`, matching this repository's
  fixture layout.
- The reference test file is `tests/test_mellum_frontends.c`; the older plan's
  `tests/test_mellum_frontends_agent` does not exist.
- Add `message_rendering.py` so restored consecutive tool results retain their
  individual boundaries and adjacent calls retain the reference newline. This
  is the authorized RollingWindow overlap.
- Update the exact ABI binding census and declare the new predicate in the fork
  fixture header; the older plan did not list those current-code guards.
- Extend the fake chat vocabulary/prefix behavior for portable Hermes tests.
- Keep the existing parser error for a tool call inside `<think>` instead of
  adopting the C agent's ignore-and-warn path.
- Verify exact reference text and fake-vocabulary token/span boundaries. A real
  Mellum token-id generation check is not portable and is not claimed.
- Compaction production code is expected to remain unchanged because generic
  markers are already masked on current `main`.

## Verification

Run focused red/green tests for each task, all native and tool-syntax
regressions, the complete static gate, and the complete portable suite with the
documented empty-`HOME` isolation. Record exact counts in the PR. Portable tests
are not evidence of live numerical or tool-use parity.
