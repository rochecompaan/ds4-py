# M2 — chat & decode

**Goal:** Add native chat rendering, generated tool prompts, streaming tool parsers, display decoding, decode-step and logits hooks, resumable context limits, and strict `--trace` replay reading.

**Spec:** [`docs/specs/2026-10-21-ds4-py-design.md`](../specs/2026-10-21-ds4-py-design.md), **M2 — chat & decode** and the **Done when** conditions. Behavioral references are sections 5–10 and 20. The replay capture target is section 21.

**Base:** `main` after the M1 squash merge. Work in this existing worktree; do not create a nested worktree.

M1 already provides token/segment `Transcript` operations, `PlainStep`, the basic worker `GenerateJob`, and `run_round`. Extend those interfaces; do not create parallel chat-only transcript or decode implementations.

**Design references:** The plan pins `ds4-agent` behavior at `antirez/ds4@5379c1c9ac315f1540371b06a3db88491d4059ff`, matching M1. Prompts intentionally differ where the spec says they differ: ds4-py generates its own tool prompt and renders GLM/Qwen trusted markup even where upstream C tokenizes it as plain text.

**Default verification:** `uv run pytest -m "not live and not build"` must pass. Live replay is a separate, explicitly marked check; its absence must not block the non-live suite.

**Review focus:** (1) parser leniencies and byte-split independence, (2) fake multi-token steps obeying stop and rewind rules, and (3) proving a hook bias changes the sampled token. Also review that untrusted text never enters rendered tokenization and that replay uses the same GGUF as C.

---

## Architecture and boundaries

### 1. Native rendering supplies primitives; `PromptBuilder` owns trust boundaries

Add small native wrappers around `ds4_chat_begin`, `ds4_chat_append_think_prefix`, `ds4_chat_append_message`, `ds4_chat_append_assistant_prefix`, and `ds4_qwen4_reasoning_effort_text`. They return appended token deltas and accept the existing transcript where C needs it.

`PromptBuilder` decides which bytes go through plain or rendered tokenization:

- fixed role wrappers and trusted tool-syntax markup may become special tokens;
- user prompts, instructions, prefix-file contents, tool names/descriptions/schemas, and tool-result payloads stay plain;
- Qwen tool markup is rendered explicitly, rather than relying on upstream `ds4_chat_append_message`, which treats Qwen tool markup as plain text;
- the DeepSeek V4.1 empty system message remains a separate segment;
- the assistant prefix has its own `assistant_prefix` segment; generated tokens extend one `assistant` segment; EOS is an `eos` segment except on GLM.

There will be no prompt byte/token fixtures against C. The spec removed those fixtures because the prompt is intentionally different. Tests inspect trust boundaries and segment structure with `FakeNative`; live replay supplies end-to-end token parity.

### 2. Raw and chat rounds share decode, not stop/finalization policy

Low-level `Session.append_tokens` keeps the session in raw mode. PromptBuilder-backed helpers (`start_chat`, `append_user`, `append_prefix`, `append_tool_results`, and `append_system_note`) put it in chat mode. Store this flag on `Transcript` so M4 can persist it.

Raw rounds keep M1 behavior: `is_stop`, no tool parser, no automatic assistant prefix/EOS, and `Stopped("context_reserve")`. Chat rounds use `is_stop_for_think_mode`, create a fresh parser per assistant round, add one assistant prefix, and close the assistant turn according to C. `ds4py raw` therefore needs no prefix-file option: C rejects `--prefix-file` with `--raw-prompt`. M2 exposes the prefix parser and session API; M3 wires it to the chat CLI.

### 3. Parser state is an immutable snapshot

Use a `Phase` enum (`SEARCH`, `STRUCTURAL`, `PARAM_VALUE`, `DONE`, `ERROR`) and:

```python
@dataclass(frozen=True)
class ParserState:
    phase: Phase
    greedy: bool
    in_think: bool
    error: str | None = None
```

`ToolParser` is the common streaming parser protocol. Each implementation exposes `feed(bytes) -> ParserState`, `finish() -> ParserState`, `state`, and `calls() -> list[ToolCall]`. `finish()` is the explicit equivalent of C's end-of-generation handling: GLM/Qwen trailing whitespace can complete a block; an unfinished active call becomes `ERROR`; plain prose stays `SEARCH`.

`ToolCall` contains `name: str`, `args: dict[str, object]`, and `raw: str`. String values use the exact `ds4_tool_text.h` unescaping. DSML `string="false"` values are parsed as JSON. Qwen uses C's JSON inference for numbers, booleans, null, arrays, and objects. Invalid JSON in a value that must be JSON is a parser error.

### 4. Decode commits before publishing parser/display events

Keep M1's invariant that an evaluated token enters the transcript before its `Token` event. Split judging from observation:

- `accept(token)` handles stop-token and budget decisions without mutating parser or transcript state;
- after eval, `commit(token)` appends the token, emits `Token`, then feeds the stream state and emits any `Thinking`, `TextDelta`, or `ToolSyntax` events;
- `commit` returns the possibly strengthened verdict, so a completed call or parser error can stop immediately and takes precedence over a simultaneous budget stop;
- uncommitted draft tokens never reach the parser, transcript, hooks' sampled-token history, or callers.

A completed call yields `ToolCalls` before the final `Stopped("tool_call_complete")`. A parser failure yields `Stopped("parser_error")` and no `ToolCalls`. `finish()` runs on EOS or the caller's token limit, but not while paused at the context reserve or after interruption.

### 5. Replay is a strict state machine, not a trace-text comparison

`read_replay` uses token IDs, not rendered text. It tracks the transcript C had synchronized, applies each `prefill_suffix` at its cached position, and records the sampled IDs until `generation finished`. `compacted_transcript` replaces the rebuilt base. A cache longer than the rebuilt base, wrong lengths or indexes, duplicate blocks, missing initial state, and raw traces are errors. A shorter cache is a valid rewind/rebuild: use `base[:cached] + suffix`.

For normal rounds, live replay generates up to one token beyond C's sampled count and requires ds4-py to stop naturally at the same boundary. For `edit old preflight failed` rounds, only the prefix through C's early stop is compared. Context-limited rounds reproduce the traced boundary explicitly. Capture instructions require a token limit high enough that C does not stop merely at `-n`; the trace format does not record that limit.

### 6. Keep native adapters and chat orchestration at focused seams

The M1 files are already large enough that adding all M2 behavior in place would obscure their responsibilities. Use these boundaries:

- `_native/chat.py` owns native chat-delta construction. `_native/fake_chat.py` owns fake special tokens and family rendering.
- `_native/logits.py` owns native copy/set buffer validation. The existing handle classes keep only small delegating methods.
- `engine/prompt.py` owns rendering decisions. `engine/chat.py` owns session-level chat helpers, assistant-round finalization, and reserve continuation. `Session` keeps the public API and raw-round ownership.
- `engine/stream.py` owns the per-round parser/decoder state. `engine/decode.py` owns sampling, acceptance, commit, and step dispatch.

This keeps each file near or below 400 meaningful lines and separates native adaptation, prompt policy, stream parsing, and session ownership.

---

## Task 1: Native chat-rendering primitives and family-aware fake tokenization

**Files:**
- Modify: `src/ds4py/_native/bindings.py`
- Modify: `src/ds4py/_native/api.py`
- Create: `src/ds4py/_native/chat.py`
- Modify: `src/ds4py/_native/protocol.py`
- Create: `src/ds4py/_native/fake_chat.py`
- Modify: `src/ds4py/_native/fake.py`
- Modify: `src/ds4py/engine/session.py`
- Test: `tests/native/test_bindings.py`
- Test: `tests/native/test_fake.py`
- Create: `tests/native/test_chat_render.py`

**Interfaces:**
- `NativeEngine.chat_begin() -> list[int]`.
- `NativeEngine.think_prefix(prefix, mode) -> list[int]`, returning only the appended delta.
- `NativeEngine.render_message(prefix, role, content) -> list[int]`, returning only the appended delta.
- `NativeEngine.assistant_prefix(prefix, mode) -> list[int]`.
- `NativeEngine.qwen_reasoning_effort(mode) -> str | None`.
- Private `Engine` counterparts that queue these calls on its worker for `PromptBuilder`.

- [ ] **Step 1: Add failing binding and native-wrapper tests**

Add these `Binding` records to `CORE_BINDINGS`, using the repository's prototype-string format:

```python
Binding("ds4_chat_begin", "void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens)"),
Binding("ds4_chat_append_think_prefix",
        "void ds4_chat_append_think_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode mode)"),
Binding("ds4_chat_append_message",
        "void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content)"),
Binding("ds4_chat_append_assistant_prefix",
        "void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode)"),
Binding("ds4_qwen4_reasoning_effort_text",
        "const char *ds4_qwen4_reasoning_effort_text(ds4_think_mode mode)"),
```

Test that an appended delta leaves the supplied prefix unchanged, returns only new tokens, rejects a closed handle, and decodes a null Qwen effort string as `None`. The effort string is borrowed static C storage: decode it without calling `free`. Test M1's positional-argument convention for the new functions. Do not add a live prompt-token golden.

- [ ] **Step 2: Extend `FakeNative` before writing PromptBuilder**

Add a deterministic special-token registry with IDs above the existing byte-token range. Include the family role markers, think markers, `<tool_call>`, `<arg_key>`, `<arg_value>`, `<tool_response>` and their closing forms, `<|im_start|>`, `<|im_end|>`, and `｜DSML｜`.

`FakeEngine.tokenize(..., rendered=True)` recognizes those literals using longest match; plain tokenization always uses the existing byte mapping. `token_text` reverses both mappings. Increase the fake vocabulary size to cover the special IDs and update M1's one vocabulary-size assertion. Fake chat rendering mirrors the pinned upstream C, including upstream Qwen's plain system/tool content. This is deliberate: later tests must fail if `PromptBuilder` accidentally delegates Qwen trusted markup to the upstream behavior.

Add fake tests proving that `"<tool_call>"` is one special token only in rendered mode and that arbitrary user text remains byte tokens in plain mode.

- [ ] **Step 3: Implement the native and public-Engine wrappers**

Use a helper in `_native/chat.py` that builds `structs.Tokens` from the prefix, calls the void C function, verifies that the old prefix is intact, and returns `tokens[len(prefix):]`. Free native token storage in `finally`. The `NativeEngine` methods in `api.py` only delegate to this adapter. Keep all native calls on the owning worker thread. Skip empty deltas at the higher layer rather than weakening `Transcript.append`'s non-empty rule.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/native/test_bindings.py tests/native/test_fake.py tests/native/test_chat_render.py -v`

```bash
git add src/ds4py/_native src/ds4py/engine/session.py tests/native
git commit -m "feat(native): add chat rendering primitives"
```

---

## Task 2: Port the `--prefix-file` parser

**Files:**
- Create: `src/ds4py/engine/prefix.py`
- Test: `tests/engine/test_prefix.py`

**Interfaces:**
- `PrefixTurn(role: Literal["user", "assistant"], content: str)`.
- `PrefixError(ValueError)`.
- `parse_prefix(data: bytes) -> tuple[PrefixTurn, ...]`.
- `load_prefix(path: str | Path) -> tuple[PrefixTurn, ...]`.

- [ ] **Step 1: Write failing parser tests**

Cover the C contract, not just a happy path:

1. One user/assistant pair, multiple pairs, and multiline contents.
2. A UTF-8 BOM is accepted only at the start.
3. Empty input, BOM-only input, NUL bytes, and no turns are errors.
4. Every turn must begin at column zero with `USER:` or `ASSISTANT:`; leading whitespace before a marker is an error.
5. Roles alternate starting with user; repeated or out-of-order roles report the line and expected role.
6. Exactly one optional space or tab after the marker is removed.
7. Marker-like text inside a line is content; a marker after a newline starts the next turn.
8. The newline before the next marker and one preceding CR are removed; other internal newlines are preserved.
9. Zero-length turns and a final user turn are errors. A whitespace-only turn whose remaining content is non-empty follows C and is accepted.
10. Invalid UTF-8 reports `PrefixError` with line context.
11. File open/read failures include the path and chained original error.

Use the C error text where practical (`line N: expected USER:`, `prefix file must end with an ASSISTANT turn`). Tests check behavior and line numbers, not documentation wording.

- [ ] **Step 2: Implement the byte parser and loader**

Port `ds4_prompt_prefix_parse` directly enough to preserve its edge cases. Decode accepted content as strict UTF-8 and report invalid bytes as `PrefixError` with line context. Parsing operates on bytes so non-ASCII and marker handling cannot depend on Python character indexes. Do not silently ignore malformed trailing data.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/engine/test_prefix.py -v`

```bash
git add src/ds4py/engine/prefix.py tests/engine/test_prefix.py
git commit -m "feat(engine): parse prefix files"
```

---

## Task 3: Generate family-native tool prompts and escape tool text

**Files:**
- Create: `src/ds4py/engine/tool_prompt.py`
- Create: `src/ds4py/engine/tool_syntax/text.py`
- Create: `src/ds4py/engine/tool_syntax/__init__.py`
- Test: `tests/engine/test_tool_prompt.py`
- Test: `tests/engine/test_tool_text.py`

**Interfaces:**
- `ToolDefinition(name: str, description: str, parameters: Mapping[str, object])`.
- `ToolPromptGenerator(family: str).generate(definitions) -> str`.
- Shared `escape_tool_text(text, closing_tag)` and `unescape_tool_text(text, closing_tag)`.

- [ ] **Step 1: Write exact generator tests**

Use two small definitions: `read(path: string)` and `edit(path: string, new_string: string, whole: boolean)`. Pin exact output for each family, including the V4.1 spaces in ` calls`, ` invoke`, and ` parameter` closing tags. Tests cover:

- DeepSeek's `<tools>{"name":...,"description":...,"parameters":...}</tools>` list and DSML syntax example.
- GLM's `<｜tool▁call▁begin｜>read<｜tool▁sep｜>...` example with `...` outside the quoted parameter value.
- Qwen's `<tool_call>\n<function=read>...` example and one `{"type":"function","function":...}` object per line.
- Compact deterministic JSON with `ensure_ascii=False`; no Python repr or non-deterministic key ordering.
- Empty definitions produce the same fixed structure with an empty schema list, rather than failing.
- No C agent tool contracts, C-specific tool names beyond the examples, or `edit-upto` text is copied into the generator.

The fixed instructions must say to escape a literal closing delimiter and its escaped spelling, while other entities stay unchanged. They need not reproduce C's longer tool-usage rules.

- [ ] **Step 2: Port `ds4_tool_text.h` behavior and test it**

For each closing delimiter (`</｜DSML｜parameter>`, the V4.1 spelling, `</arg_value>`, `</parameter>`, `</tool_result>`, and `</tool_response>`), test:

- the literal closing tag becomes `&lt;...`;
- an already escaped closing tag gets its leading `&` escaped so it round-trips literally;
- repeated `amp;` prefixes follow C's loop;
- ordinary `&amp;`, `&lt;`, HTML, and unrelated text are unchanged by unescaping;
- escape followed by unescape returns the original payload.

- [ ] **Step 3: Implement the generator and escaping helpers**

Keep trusted template strings separate from serialized definitions. The generator returns text only; Task 4 decides which spans are rendered or plain. Use `json.dumps(..., ensure_ascii=False, separators=(",", ":"))`. Construct top-level objects in `name`, `description`, `parameters` order and preserve caller schema insertion order; identical inputs then produce identical output without sorting nested schema keys.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/engine/test_tool_prompt.py tests/engine/test_tool_text.py -v`

```bash
git add src/ds4py/engine/tool_prompt.py src/ds4py/engine/tool_syntax/text.py src/ds4py/engine/tool_syntax/__init__.py tests/engine/test_tool_prompt.py tests/engine/test_tool_text.py
git commit -m "feat(engine): generate native tool prompts"
```

---

## Task 4: Add `PromptBuilder` and chat transcript helpers

**Files:**
- Create: `src/ds4py/engine/prompt.py`
- Create: `src/ds4py/engine/chat.py`
- Modify: `src/ds4py/engine/transcript.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/__init__.py` if public re-exports belong there
- Test: `tests/engine/test_prompt.py`
- Test: `tests/engine/test_session_chat.py`

**Interfaces:**
- `PromptBuilder(engine).begin(tools=(), instructions=None) -> list[RenderedSegment]`.
- `PromptBuilder.message(prefix, role, text) -> list[int]`.
- `PromptBuilder.assistant_prefix(prefix) -> list[int]`.
- `PromptBuilder.tool_results(prefix, results) -> list[int]`.
- `PromptBuilder.append_assistant_end() -> list[int]`, returning EOS except for an empty GLM delta.
- `Session.start_chat(tools=(), instructions=None)`, `append_user(text, images=())`, `append_prefix(turns)`, `append_tool_results(results)`, and `append_system_note(text)`.

`RenderedSegment` is a small internal value with `kind` and `tokens`; it is not a second transcript history.

- [ ] **Step 1: Write trust-boundary tests**

Using the family-aware fake, prove:

1. A user string containing `<tool_call>`, `<|im_start|>`, `<think>`, and `｜DSML｜` contains no special-token IDs.
2. Fixed `<tool_call>` and `<tool_response>` markup does produce special IDs on Qwen. GLM argument tags produce their special IDs when available. DSML uses its dedicated marker ID.
3. A tool description or JSON schema string containing those same literals remains plain even though surrounding generator markup is rendered.
4. GLM instructions, system notes, assistant prefix contents, and tool payloads containing role/argument markers remain plain. Qwen assistant prefix contents and tool payloads remain plain as well.
5. Instructions are a separate system message prefixed by `\n\n`. System notes use `\n` plus the note and kind `note`.
6. A Qwen reasoning-effort instruction is a system message. Non-Qwen families use the native think-prefix operation.
7. DeepSeek V4.1 emits the empty-system role delta before the tools text and records it separately.

These are structural fake tests, not C prompt-byte goldens.

- [ ] **Step 2: Write session-helper tests**

Cover:

- `start_chat` appends BOS/think/system deltas in order and rejects a second initialization. `append_user` or `append_prefix` on an empty session performs a tool-less chat start. `append_tool_results` and `append_system_note` require an initialized chat.
- The assistant prefix is returned as one non-empty delta for supported modes and is stored as `assistant_prefix`; no zero-length segment is created when a native optional prefix is empty.
- Prefix turns append user/assistant segments in order and EOS after every assistant turn except GLM.
- `append_tool_results(["first", "second"])` creates exactly one `tool` segment whose payload is `first\n\nsecond`; it never creates one message per result.
- Tool-result text containing the exact closing sentinel is escaped and cannot create a second closing special token. On Qwen the fixed `<tool_response>` wrapper itself is special.
- `append_system_note` and all chat helpers refuse to mutate during an active round, matching M1 ownership rules.
- Non-empty `images` raises `EngineError("image input requires M5 vision support")`; the empty default stays text-only.
- Low-level `append_tokens` on a fresh session leaves raw mode unchanged.

- [ ] **Step 3: Implement family-specific rendering**

For DeepSeek V4/V4.1, follow C's split between native role deltas and rendered tools text. For GLM, render an empty native system message to obtain its role prefix, then append trusted and plain spans separately. For Qwen, compose the fixed ChatML wrapper from rendered trusted literals and append serialized definitions through plain tokenization; do not call upstream Qwen `render_message` with the whole tools prompt.

Use native `render_message` only where C already treats the content as plain text. This includes user messages on every family and DeepSeek system/assistant/tool messages. Qwen system and user content is also plain in C.

Compose the other cases from trusted native deltas and plain payload tokens:

- For a GLM system message, use the native empty-system delta as the role prefix and append plain content.
- For GLM tool results, use the native empty-tool delta. Remove the separately rendered `</tool_response>` suffix from that delta, then append plain escaped payload and the trusted suffix.
- For Qwen assistant messages and tool results, compose the fixed ChatML wrapper from rendered trusted literals and append the content through plain tokenization.
- For a GLM assistant prefix turn, preserve C's conditional empty-think wrapper. Use the native empty-assistant delta when the content does not start with a think marker. If it starts with `<think>` or `</think>`, derive the role-only delta from native rendering of that marker and remove the separately computed rendered-marker suffix. Append the prefix-file content as plain tokens in both cases.

Never infer a payload span by retokenizing untrusted content as rendered chat. Escape only the matching closing delimiter in tool payloads before plain tokenization. Add `Transcript.chat` with the same lock discipline as tokens and segments. Chat helpers set it, while raw token append does not. Put the session-level helper operations in `engine/chat.py`. Keep `Session` methods as small public delegates that enforce the existing ownership rules. All builder methods return deltas, and the session-level helper alone mutates transcript history. `append_assistant_end` gets EOS through the engine worker and returns an empty delta on GLM. Chat finalization uses this single helper rather than duplicating family rules.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/engine/test_prompt.py tests/engine/test_session_chat.py tests/engine/test_transcript.py -v`

```bash
git add src/ds4py/engine/prompt.py src/ds4py/engine/chat.py src/ds4py/engine/transcript.py src/ds4py/engine/session.py src/ds4py/engine/__init__.py tests/engine/test_prompt.py tests/engine/test_session_chat.py
git commit -m "feat(engine): render chat prompts and transcript turns"
```

---

## Task 5: Add the parser model, streaming base, and DeepSeek DSML parser

**Files:**
- Create: `src/ds4py/engine/tool_syntax/model.py`
- Create: `src/ds4py/engine/tool_syntax/base.py`
- Create: `src/ds4py/engine/tool_syntax/dsml.py`
- Modify: `src/ds4py/engine/tool_syntax/__init__.py`
- Create: `tests/fixtures/tool_syntax/dsml.json`
- Test: `tests/engine/tool_syntax/test_dsml.py`

- [ ] **Step 1: Write failing DSML tests and fixtures**

Port relevant C vectors from `test_agent_dsml_stream_tool_call_chunked` and `test_agent_tool_argument_literal_markup`, then add spec-required cases:

- V4 and V4.1 single and multiple invocations, including multiple arguments.
- `string="true"` returns a string; `string="false"` parses objects, arrays, numbers, booleans, and null as JSON; malformed required JSON is `ERROR`.
- The missing-leading-bar forms `<DSML｜tool_calls>` and `<DSML｜ calls>` canonicalize to the normal parser.
- Bare `<｜DSML｜invoke` and `<｜DSML｜ invoke` start an implicit block, including the missing-bar variants.
- Closing tags allow ASCII whitespace and an optional `｜` before `>`; opening tags remain strict.
- Literal `</tool_call>`, `<think>`, and `</think>` inside a parameter value do not terminate the value.
- Closing-delimiter escapes round-trip exactly; ordinary entities remain unchanged.
- Unknown tags, missing invoke/parameter names, malformed DSML markers outside a valid block, and markup inside `<think>` produce `ERROR` and no executable calls.
- Structural regions request greedy sampling; parameter values do not; a partial closing delimiter does; DONE and ERROR never do.

Store the reusable input/expected-call cases in JSON with a `source` field naming the C test or spec rule. Keep state-transition-only assertions in Python.

- [ ] **Step 2: Implement the streaming base**

The base owns byte buffers, think-marker detection across chunks, opening-marker tails, canonicalization of accepted opening typos, and the immutable state snapshot. Feeding after DONE or ERROR is a no-op. Do not use Python string indexing for the byte stream.

The opening detector follows `agent_stream_dsml_start_match`: a direct invoke opener is accepted before its closing `>` and seeds the canonical invoke text. Ordinary text must not be re-tokenized or added to `raw`; `raw` contains the canonical tool stanza only.

- [ ] **Step 3: Implement DSML parsing and `make_parser` selection**

Implement `Dsml(*, v41: bool = False)` with the incremental `parse_pos` state machine rather than reparsing the whole buffer on every byte. Preserve C's leniency at closing tags and strictness after the opener. `finish()` converts an incomplete active stanza or partial opener into an explanatory `ERROR`.

`make_parser(info)` selects `Qwen()` for `is_qwen4`, `Glm()` for `is_glm_dsa`, `Dsml(v41=True)` for `is_deepseek41`, and `Dsml()` otherwise. Later tasks register the GLM/Qwen implementations; for this task their tests may remain absent, not skipped.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/engine/tool_syntax/test_dsml.py -v`

```bash
git add src/ds4py/engine/tool_syntax tests/engine/tool_syntax tests/fixtures/tool_syntax/dsml.json
git commit -m "feat(engine): parse streaming DSML tool calls"
```

---

## Task 6: Add the GLM parser

**Files:**
- Create: `src/ds4py/engine/tool_syntax/glm.py`
- Modify: `src/ds4py/engine/tool_syntax/model.py`
- Create: `tests/fixtures/tool_syntax/glm.json`
- Test: `tests/engine/tool_syntax/test_glm.py`

- [ ] **Step 1: Port the C GLM vectors and add boundary cases**

Include the behaviors covered by `test_agent_glm_tool_parser_single_arg`, `...chunked_multi_arg`, `...streams_param_state`, `...multiple_adjacent_calls`, `...rejects_missing_value`, and `test_agent_glm_stream_ignores_tool_inside_think`:

- prose before a call is ignored; function and argument names are trimmed with ASCII whitespace;
- all argument values are strings and use `</arg_value>` unescaping;
- adjacent calls separated by whitespace produce two calls; another `<tool_call>` after a completed call continues the same block;
- `finish()` accepts trailing whitespace after the last call and reaches DONE;
- a missing `<arg_value>`, empty key, missing function name, or unexpected structural tag is ERROR;
- literal tool/think markup in a value is data, while a tool opener in thinking is an error;
- `<tool` followed by the opener requests greedy decoding, a value does not, and `</arg` does.

- [ ] **Step 2: Implement GLM without weakening DSML**

Implement `Glm`. Reuse only the streaming base and text helpers. Keep GLM's name-first grammar and `after_call` state explicit; do not force it into the Qwen function-tag grammar. `finish()` must not invent a call for an opener with no completed `</tool_call>`.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/engine/tool_syntax/test_glm.py tests/engine/tool_syntax/test_dsml.py -v`

```bash
git add src/ds4py/engine/tool_syntax/glm.py src/ds4py/engine/tool_syntax/model.py tests/engine/tool_syntax tests/fixtures/tool_syntax/glm.json
git commit -m "feat(engine): parse streaming GLM tool calls"
```

---

## Task 7: Add the Qwen parser

**Files:**
- Create: `src/ds4py/engine/tool_syntax/qwen.py`
- Modify: `src/ds4py/engine/tool_syntax/model.py`
- Create: `tests/fixtures/tool_syntax/qwen.json`
- Test: `tests/engine/tool_syntax/test_qwen.py`

- [ ] **Step 1: Port the C Qwen vectors and JSON rules**

Include `test_agent_qwen_tool_parser_chunked_multi_arg`, `...two_calls_and_error`, `test_agent_qwen_argument_markers_bytewise`, and Qwen's literal-markup case. Assert:

- one leading newline and one trailing newline around a parameter value are removed, while other whitespace is preserved;
- numbers, `true`, `false`, `null`, objects, and arrays become JSON values; ordinary text stays a string;
- malformed bracketed JSON is ERROR rather than silently changing type;
- function and parameter names are ASCII-trimmed and cannot be empty;
- two adjacent calls work; trailing whitespace completes at `finish()`;
- missing `</function>`, a missing tool close, unexpected tags, and an unfinished value are ERROR;
- `</parameter>` escaping and literal tool/think tags inside values match C.

- [ ] **Step 2: Implement Qwen and register the factory**

Implement `Qwen` and register it with the factory. Port `agent_qwen_value_is_json`'s classification before calling `json.loads`; do not classify every string by attempting JSON indiscriminately. Keep numeric length behavior bounded as in C so an oversized numeric-looking value remains a string.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/engine/tool_syntax -v`

```bash
git add src/ds4py/engine/tool_syntax/qwen.py src/ds4py/engine/tool_syntax/model.py tests/engine/tool_syntax/test_qwen.py tests/fixtures/tool_syntax/qwen.json
git commit -m "feat(engine): parse streaming Qwen tool calls"
```

---

## Task 8: Prove parser chunking independence and fuzz stream boundaries

**Files:**
- Create: `tests/engine/tool_syntax/conftest.py`
- Create: `tests/engine/tool_syntax/test_stream_equivalence.py`
- Create: `tests/engine/tool_syntax/test_fuzz.py`

- [ ] **Step 1: Run every fixture through every single split**

For every fixture byte string of length `N`, compare one-shot parsing with:

1. `feed(data[:i]); feed(data[i:])` for every `i` from 0 through `N`;
2. one-byte feeds;
3. 100 deterministic pseudo-random chunk partitions seeded from `random.Random(0)` and the fixture name.

Compare final phase, error, greedy flag, ordered calls, names, parsed argument values, and raw canonical stanza. Include splits inside multibyte `｜`, inside think markers, and inside escape spellings. This is linear in the number of single split points, not an exponential set of partitions.

- [ ] **Step 2: Add focused stateful fuzz**

Without adding a new dependency, generate seeded valid calls with random Unicode, `<`, `>`, `&`, newlines, and escaped closing sentinels in parameter values. Randomize chunking and assert whole/chunked equivalence and escape round-trip. Also generate plain prose excluding tool/DSML markers and assert that no calls appear and no markup becomes executable.

Add regression tests that feed after DONE/ERROR, call `finish()` repeatedly, and alternate empty chunks with partial markers. The tests must terminate and leave state stable. These tests prove behavior and guard streaming regressions; they are not implementation restatements.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/engine/tool_syntax -v`

```bash
git add tests/engine/tool_syntax
git commit -m "test(engine): verify parser stream equivalence"
```

---

## Task 9: Add the display-only `Decoder`

**Files:**
- Create: `src/ds4py/engine/decoder.py`
- Modify: `src/ds4py/engine/events.py`
- Test: `tests/engine/test_decoder.py`

**Interfaces:**
- `Decoder(*, thinking: bool = False)` with `feed(bytes) -> list[TextDelta]` and `finish() -> list[TextDelta]`.
- Add `TextDelta(text: str, thinking: bool)` to `engine/events.py`.

- [ ] **Step 1: Write failing streaming-display tests**

Cover:

1. ASCII text passes through with the current thinking flag.
2. Multibyte UTF-8 split at every byte produces the same text as one-shot input; no replacement character appears for a valid split sequence.
3. `<think>` and `</think>` are recognized across arbitrary chunk splits, are not emitted as text, and flip subsequent deltas.
4. A 16-byte pending buffer retains only possible marker prefixes; ordinary `<`, `</`, `<thinking>`, and marker-like text with extra characters flush correctly.
5. Empty feeds do not emit empty deltas; `finish()` flushes safe pending text and incomplete UTF-8 deterministically.
6. Initial `thinking=True` labels pre-marker output correctly.
7. Feeding the decoder never changes a transcript or parser. Its output is not used as decode input.

- [ ] **Step 2: Implement bounded marker scanning and incremental UTF-8**

Use `codecs.getincrementaldecoder("utf-8")` and a byte pending buffer capped at 16 bytes for ambiguous marker suffixes. Decode only spans known not to contain a complete think marker. Keep display state local to one decoder/assistant round.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/engine/test_decoder.py -v`

```bash
git add src/ds4py/engine/decoder.py src/ds4py/engine/events.py tests/engine/test_decoder.py
git commit -m "feat(engine): decode assistant text for display"
```

---

## Task 10: Integrate tool-aware decode, chat rounds, and `FakeMultiTokenStep`

**Files:**
- Modify: `src/ds4py/_native/api.py`
- Modify: `src/ds4py/_native/protocol.py`
- Modify: `src/ds4py/_native/fake.py`
- Modify: `src/ds4py/engine/decode.py`
- Create: `src/ds4py/engine/stream.py`
- Modify: `src/ds4py/engine/events.py`
- Modify: `src/ds4py/engine/worker.py`
- Modify: `src/ds4py/engine/chat.py`
- Modify: `src/ds4py/engine/session.py`
- Create: `tests/engine/fake_steps.py`
- Test: `tests/engine/test_decode.py`
- Test: `tests/engine/test_decode_tools.py`
- Test: `tests/engine/test_multi_token_step.py`
- Test: `tests/engine/test_session_chat.py`

**Events:** Add `Thinking(on)`, `ToolSyntax(state)`, and `ToolCalls(calls)` with immutable event values.

- [ ] **Step 1: Add stop-policy and stream-state tests first**

M1 already binds `ds4_token_is_stop_for_think_mode`; add the `NativeEngine`/`FakeEngine` method and protocol entry without redeclaring the ABI. Extend `GenerateJob` with optional `think_mode`, stream state, logit hooks, and pause policy while preserving M1 defaults.

Tests must show:

- raw still calls `is_stop`; chat calls `is_stop_for_think_mode` with the current mode;
- parser state starts fresh per assistant round and initially reflects enabled thinking;
- structural markup switches the next sample to greedy (`temperature=0`, `top_k=0`, `top_p=1`, `min_p=0`), values restore configured sampling, and DONE/ERROR cannot leave greedy on;
- token commit order remains eval → transcript → `Token` → stream events;
- completed calls stop after the closing token, emit ordered `ToolCalls`, and discard later drafts;
- markup in thinking and malformed/incomplete calls stop as `parser_error` without `ToolCalls`;
- EOS at a GLM/Qwen block that needs `finish()` still produces its calls;
- no parser events or chat EOS are added to raw rounds.

- [ ] **Step 2: Refactor `DecodeContext` without weakening M1 tests**

`accept` retains stop/budget logic. `commit` performs stream observation and returns the final verdict. A parser terminal state overrides a simultaneous max/context limit, matching C's `!got_tool && !malformed_tool` guard around compaction handling.

Add a reusable per-round stream object in `engine/stream.py` holding parser, decoder, sampled IDs, thinking count, and the last emitted states. `sampling()` derives greedy policy from its snapshot. `before_sample()` remains a no-op until Task 11. On EOS or a caller token limit, finalize both parser and decoder. On a tool/error stop, finish only the decoder because the parser is already terminal. Do not finalize either one on a context pause or interruption. Emit `ToolCalls` once after generation and before `Stopped`.

- [ ] **Step 3: Implement chat `run_round` orchestration**

When `transcript.chat` is true, append one assistant prefix before the initial sync, pass the parser/think mode in `GenerateJob`, and close with `PromptBuilder.append_assistant_end`. Put this orchestration in `engine/chat.py`; `Session.run_round` selects it without duplicating raw `_round` logic. Extend an existing open assistant segment after any continuation rather than creating another one. Close on EOS, caller limit, parser error, interruption, or failed context reserve; do not close between a reserve pause and resume.

Keep M1 raw tests unchanged except where a test intentionally constructs a chat session. Verify `Synced.prefilled`, transcript tokens/segments, and event order together.

- [ ] **Step 4: Implement `FakeMultiTokenStep` against the same context contract**

The test-only step accepts a sampled token plus at most 16 draft tokens (17 total), evaluates a block, and tracks `block_start`. It must:

- stop before a first or later stop token without committing it;
- rewind to `block_start + kept_count` for an unkept tail, including a second stop token;
- commit kept tokens in order only after the block's eval and required rewind;
- honor `STOP_AFTER` at a budget or tool boundary and discard the remaining block;
- cap drafts at 16 and return to sampling for the next block;
- treat an empty engine block (`ntok == 0`) as an immediate step return with no eval, commit, or new events for that block. After the step returns, `generate` must detect a non-cancelled exit with no stop verdict and raise `EngineError("decode returned no tokens")`, matching C;
- leave the session position, transcript, and emitted token IDs in agreement after every test.

Use scripted block providers in `tests/engine/fake_steps.py`; do not add speculative native bindings until M5.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest tests/engine/test_decode.py tests/engine/test_decode_tools.py tests/engine/test_multi_token_step.py tests/engine/test_session.py tests/engine/test_session_chat.py tests/engine/test_session_transcript.py -v`

```bash
git add src/ds4py/_native src/ds4py/engine tests/engine/fake_steps.py tests/engine/test_decode.py tests/engine/test_decode_tools.py tests/engine/test_multi_token_step.py tests/engine/test_session.py tests/engine/test_session_chat.py tests/engine/test_session_transcript.py
git commit -m "feat(engine): integrate tool-aware chat decoding"
```

---

## Task 11: Add worker-thread logits hooks

**Files:**
- Create: `src/ds4py/engine/logits.py`
- Create: `src/ds4py/_native/logits.py`
- Modify: `src/ds4py/_native/api.py`
- Modify: `src/ds4py/_native/protocol.py`
- Modify: `src/ds4py/_native/fake.py`
- Modify: `src/ds4py/engine/decode.py`
- Modify: `src/ds4py/engine/session.py`
- Test: `tests/engine/test_logits.py`
- Test: `tests/native/test_fake.py`

**Interface:** `session.add_logits_hook(hook) -> None` appends a hook between rounds. Each hook receives one reusable `LogitsView` per generation invocation. Document that hooks run on the engine worker thread and must not call the engine, session, harness, blocking I/O, or cancellation APIs.

- [ ] **Step 1: Write the behavior tests, especially actual bias**

Test all of the following:

1. With no hooks, `copy_logits` is never called.
2. `logits()` copies into the same reusable `array('f')` buffer and returns a view of that buffer; `set_logits(values)` validates exact vocabulary length and copies through the same facade.
3. Position, `in_think`, `think_tokens`, parser state, and sampled IDs describe the current round and update after committed tokens only.
4. **A hook that biases token 8 over the fake's preferred token 7 makes the next emitted/sampled token 8.** Assert both the hook's mutation and the generated token/transcript; checking only `set_logits` is insufficient.
5. Hooks run in registration order on the worker thread.
6. A hook exception fails the round, does not commit an unevaluated token, and is re-raised by the caller without poisoning later session ownership.
7. Installing a hook during an active round is refused; raw rounds without hooks keep M1 overhead.

- [ ] **Step 2: Add native/fake copy/set wrappers**

`NativeSession.copy_logits(out)` uses an array-backed ctypes float buffer and checks `len(out) == info.vocab_size`; `set_logits(values)` uses the same length check. Put buffer conversion and status handling in `_native/logits.py`, with small `NativeSession` delegates. Native functions already exist in M1's ABI map, so only add a binding if a test shows it absent.

Make the fake's current distribution observable and mutable. A simple valid model is to cache a one-hot/weighted vector for the next scripted candidate, let `copy_logits` expose it, let `set_logits` replace it, and have `sample` select the resulting argmax while advancing the existing script only once. Invalidate that cache on eval, rewind, sync, or snapshot load. Default fake tests must retain their current scripted outputs.

- [ ] **Step 3: Implement `LogitsView` and hook dispatch**

`LogitsView` owns the reusable buffer and exposes read-only `position`, `in_think`, `think_tokens`, `parser_state`, and `sampled_tokens`. `sampled_tokens` is a tuple snapshot; the logits buffer is reused. `bias(token_id, delta)` validates the ID, reads, adds, and sets.

`DecodeContext.before_sample()` updates the view and runs hooks. If hooks exist and the selected step has `hooks_every_sample = False`, `generate` uses `PlainStep` instead. Add a test proving a multi-token step is not entered in that case; without hooks the multi-token step remains selected.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/engine/test_logits.py tests/engine/test_multi_token_step.py tests/engine/test_decode.py tests/native/test_fake.py -v`

```bash
git add src/ds4py/engine/logits.py src/ds4py/engine/decode.py src/ds4py/engine/session.py src/ds4py/_native tests/engine/test_logits.py tests/engine/test_multi_token_step.py tests/engine/test_decode.py tests/native/test_fake.py
git commit -m "feat(engine): add decoding-time logits hooks"
```

---

## Task 12: Resume or fail at the context reserve

**Files:**
- Modify: `src/ds4py/engine/events.py`
- Modify: `src/ds4py/engine/decode.py`
- Modify: `src/ds4py/engine/chat.py`
- Modify: `src/ds4py/engine/session.py`
- Test: `tests/engine/test_context_reserve.py`

**Interface:** On a chat `Stopped`-reason of `context_reserve`, `run_round` instead yields `ReserveReached`. The caller must choose before advancing the iterator:

```python
for event in session.run_round(max_tokens=100):
    if isinstance(event, ReserveReached):
        event.resume(new_reserve)  # continue the same assistant round
        # or event.fail()          # finish with Stopped("context_reserve")
```

- [ ] **Step 1: Write pause/resume/fail tests**

Cover:

- raw rounds still yield `Stopped("context_reserve")` directly;
- a chat pause emits no final `Stopped`, no duplicate assistant prefix, and no EOS before the decision;
- `resume(smaller_reserve)` continues from the existing transcript/parser/sampled history and preserves the remaining caller token budget;
- resumed output extends the same assistant segment and does not duplicate earlier `Token`, `Thinking`, or `ToolSyntax` events;
- repeated reserve pauses are allowed, each requiring a decision;
- `fail()` closes the chat turn once and then yields `Stopped("context_reserve")`;
- advancing without a decision raises a clear `EngineError`; a second decision raises instead of changing an in-flight continuation;
- bool and negative reserves are rejected; zero is allowed and may produce another immediate pause when no generation room remains;
- closing the outer generator still runs M1's interruption/drain behavior and never detaches native work.

- [ ] **Step 2: Implement a single-assistant continuation**

`ReserveReached` stores a one-shot decision and callbacks into the active `run_round` generator. On resume, submit a replacement `GenerateJob` with the same mutable transcript and stream object, remaining `max_tokens`, and new reserve. Its initial sync must be an extend/rebuild of the same prompt; do not append another prefix. `DecodeContext` detects the existing open assistant segment so its first resumed commit extends rather than appends a new segment.

The worker continues to emit `Stopped("context_reserve")` internally; the session translates that event into the pause. Other reasons pass through unchanged. On fail, perform the normal chat close and emit the public `Stopped`.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest tests/engine/test_context_reserve.py tests/engine/test_session_interrupt.py tests/engine/test_session.py tests/engine/test_decode_tools.py -v`

```bash
git add src/ds4py/engine/events.py src/ds4py/engine/decode.py src/ds4py/engine/chat.py src/ds4py/engine/session.py tests/engine/test_context_reserve.py
git commit -m "feat(engine): resume rounds at the context reserve"
```

---

## Task 13: Implement strict `--trace` replay reading

**Files:**
- Create: `src/ds4py/engine/replay.py`
- Create: `tests/fixtures/replay/basic.trace`
- Create: `tests/fixtures/replay/compacted.trace`
- Create: `tests/fixtures/replay/early_edit.trace`
- Test: `tests/engine/test_replay.py`

**Interfaces:**
- `ReplayTrace(ctx, model, rounds)`.
- `ReplayRound(number, prompt, sampled, cached, context_limited, prefix_only, think)`.
- `read_replay(path) -> ReplayTrace`.
- `ReplayError(ValueError)` with path/line context.

- [ ] **Step 1: Build synthetic traces from the pinned C grammar**

Use minimal token IDs and exact C lines, not prompt-output snapshots:

- `agent worker start ctx=... backend=... model=... trace=...`;
- `tokens label=initial_system_prompt start=0 len=N` and N `token index=... id=...` lines;
- `prefill tool_round=... transcript=... prompt=... cached=... suffix=... think=...`;
- `tokens label=prefill_suffix start=C len=T` and exactly T−C suffix token lines;
- sampled `token index=1 id=...` through `index=G` followed by `generation finished ... generated=G ...`;
- `tokens label=compacted_transcript start=0 len=N` replacing the base;
- `edit old preflight failed: ...` marking an early-stop round.

Include escaped `text=` and `hex=` fields to prove the reader keys on structure and IDs rather than trying to decode token text.

- [ ] **Step 2: Write rebuilding and rejection tests**

Assert exact prompts and sampled IDs for consecutive rounds, including the EOS and tool response that appear in the next suffix. The rebuilt live base after a round is `prompt + sampled`; the stop token is absent. A compaction block replaces that base before the next round. Build each prompt as `base[:cached] + suffix`; include a valid rewind/rebuild with `cached < len(base)` as well as the usual `cached == len(base)`.

Reject with useful line context:

- missing/duplicate initial prompt, suffix without prefill, or sampled tokens outside a round;
- `cached > len(rebuilt_base)` (the first uncached position cannot fit the rebuilt history), a negative cache, or a cache beyond the traced transcript;
- `start + suffix != transcript/prompt length`, wrong token count, non-contiguous indexes, or `generated` count mismatch;
- a second prefill before `generation finished`, unexpected token blocks, and truncated final rounds;
- `raw_prompt_tokens` or the raw-mode marker with `raw traces are not replayable`;
- missing/invalid context or unsupported think name in the live runner.

Also test a zero-length sampled round and a zero-length suffix when the trace grammar permits them; do not confuse `len=` (total transcript length) with the number of following token lines (`len-start`).

- [ ] **Step 3: Implement the reader**

A small line parser with explicit states is sufficient. Parse relevant lines with anchored regular expressions; ignore timestamps and unrelated status lines. Store only IDs needed for replay. Set `prefix_only` when the early-edit preflight line occurs during that round. Record `context_limited` and think mode for the live runner.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/engine/test_replay.py -v`

```bash
git add src/ds4py/engine/replay.py tests/engine/test_replay.py tests/fixtures/replay
git commit -m "feat(engine): read ds4-agent replay traces"
```

---

## Task 14: Add live replay parity and run the full verification

**Files:**
- Create: `tests/live/test_replay_parity.py`
- Create: `tests/live/REPLAY.md`
- Modify: `docs/specs/2026-10-21-ds4-py-design.md` only if implementation exposed a concrete spec mismatch; otherwise leave it unchanged

- [ ] **Step 1: Add the explicitly live replay test**

Use `DS4_TEST_MODEL`, `DS4_REPLAY_TRACE`, and the existing `live_lib` fixture. Skip with a clear reason when either variable is absent. Resolve and compare the trace's model path with `DS4_TEST_MODEL`; fail rather than silently using a different GGUF. Use the trace's context size and per-round think mode.

Reuse one engine/worker for all rounds. For each round:

1. Create a fresh `Transcript` containing exactly `round.prompt` and submit `SyncJob(round.prompt)`; this permits rebuilds without a public session-reset API.
2. Generate with the appropriate family parser and the traced sampling policy used by the capture (default capture: `--temp 0 --seed 1`). No ds4-py tools execute.
3. Compare emitted token IDs and transcript additions with `round.sampled`.
4. For a normal round, allow one token beyond C's count and require natural `stop_token`, `tool_call_complete`, or `parser_error` at C's boundary. A tool/error stop must agree with parsing C's sampled text.
5. For `prefix_only`, compare only through C's early edit stop and do not require the same final reason.
6. For a context-limited round, use `max_tokens=len(sampled)` and `reserve=trace.ctx-len(prompt)-len(sampled)` to reproduce the traced boundary; reject a negative derived reserve. Compare the partial prefix and do not run C's compaction. A parser unfinished at that boundary is not converted to a final parser error.
7. For every completed tool stanza in the trace, also parse C's sampled token bytes with the matching ds4-py parser and compare one-shot, every single split, and bytewise results. When a focused traced case is suitable for a non-live fixture, save it under `tests/fixtures/tool_syntax/` with its trace source/hash so the default suite retains the regression.
8. Never execute a parsed call. The next round always starts from C's prompt tokens.

Keep this test marked `live`; synthetic reader tests remain in the default suite. A live failure prints round number, prompt/sample lengths, first differing index, both token IDs, and their byte text when available.

- [ ] **Step 2: Document a reproducible C capture**

In `tests/live/REPLAY.md`, show a capture using the same absolute `MODEL` path, backend, context, sampling, and seed for C and replay. Use a high `-n` so C does not stop at its caller limit. Include one simple tool task and explain that traces from the section-21 bake-off can be replayed individually.

Record that the section-21 capture target is the frozen **1221 quantized DS4-Flash** model listed in the spec. Do not introduce a different model into the bake-off. The generic M2 reader/runner still accepts the other families for focused live checks.

Example shape (fill paths, do not hardcode a machine-specific checkout in the test):

```bash
MODEL=/absolute/path/deepseek-v4-flash-1106-Q4_K_M.gguf
$DS4_AGENT --non-interactive --model "$MODEL" --backend "$BACKEND" \
  --ctx 4096 --tokens 2048 --temp 0 --seed 1 \
  --prompt 'Read pyproject.toml and report the project name.' \
  --trace c-agent.trace
DS4_TEST_MODEL="$MODEL" DS4_REPLAY_TRACE=c-agent.trace \
  uv run pytest -m live tests/live/test_replay_parity.py -v
```

- [ ] **Step 3: Run non-live verification and inspect the complete diff**

```bash
uv run pytest -m "not live and not build"
uv run pytest tests/engine tests/native -m "not live and not build"
git diff --check main...HEAD
git status --short
```

All selected tests must pass; do not claim replay parity from the synthetic reader tests alone. If live assets are available, run the exact command in `tests/live/REPLAY.md` and record its result. If they are not, report live replay as not run.

- [ ] **Step 4: Commit the live harness**

```bash
git add tests/live/test_replay_parity.py tests/live/REPLAY.md
git commit -m "test(live): add ds4-agent replay parity"
```

---

## Completion checklist

- [ ] All new public interfaces have docstrings specifying worker-thread confinement where relevant.
- [ ] `ds4py raw` remains raw and has no `--prefix-file`; M3 owns the chat CLI flag.
- [ ] Every parser fixture passes one-shot, every single split, bytewise, and seeded random chunking.
- [ ] GLM/Qwen trusted markup and `<tool_response>` are rendered; untrusted spans remain plain.
- [ ] `FakeMultiTokenStep` tests prove rewind/stop behavior, not just event snapshots.
- [ ] The logits test proves a bias changes the next sampled token and hook presence falls back to `PlainStep`.
- [ ] Context-reserve resume keeps one assistant prefix/segment and the remaining budget.
- [ ] Synthetic replay tests fail loudly on rebuilt-prefix mismatches.
- [ ] Live replay is optional in the default suite, uses the same GGUF as C, and is reported separately as run or not run.
- [ ] `uv run pytest -m "not live and not build"` passes and `git diff --check main...HEAD` is clean.
