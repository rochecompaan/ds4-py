# ds4-py design

| | |
| --- | --- |
| Date | 2026-10-21 |
| Status | Draft for review |
| Replaces | [`archive/2026-10-21-ds4-py-first-draft.md`](archive/2026-10-21-ds4-py-first-draft.md), the first draft, kept for its framework comparison |
| Upstream ds4 | antirez/ds4 `5379c1c` (2026-10-19). Line numbers below refer to this commit |
| Harness | `pydantic-ai-harness==0.59.0`, which pins `pydantic-ai-slim==2.59.0`. Both were released 2026-10-21 |
| Downstream fork | The ds4 fork that the first downstream agent builds against, compared with upstream on 2026-10-21 (section 18) |

## 1. Intent

ds4-py is a Python port of `ds4_agent.c`, the coding agent in [antirez/ds4](https://github.com/antirez/ds4). It runs the ds4 engine inside the Python process through `ctypes` and a new `libds4` shared library. The agent is built on Pydantic AI and its Harness package.

ds4-py is a general package for anyone who wants to use the ds4 engine from Python. It contains nothing specific to any one downstream project. The first downstream agent will be a customisation of ds4-py, built against a fork of ds4, and building it is the test of ds4-py's extensibility. This spec calls them *the downstream agent* and *the downstream fork*.

The aim is to own as little agent machinery as possible. The harness supplies the run loop, tool dispatch, validation, guards, tracing and persistence. ds4-py adds only what is specific to the ds4 engine and to `ds4-agent`'s behaviour.

## 2. Goals, non-goals and anti-goals

### Goals

1. Reproduce `ds4-agent`'s behaviour in Python. This covers the token loop, the four tool-call syntaxes, its tools and system prompt, compaction, system notes and reminders, session files and the interactive commands.
2. Support every model and model family that ds4 supports at the pin: DeepSeek V4 (DSML), DeepSeek V4.1 (DSML with V4.1 tags), GLM and Qwen.
3. Use the harness for everything it already does. Write only the ds4-specific parts.
4. Keep session files compatible with `ds4-agent` in both directions.
5. Be extensible enough for the downstream agent (section 18). That means a hook in the token loop, libraries built from any ds4 tree including forks, and a way to add native bindings and decode steps without forking ds4-py.
6. Make context management a plug-in point.
7. Let people use the engine without the harness. The engine core installs on its own.

### Non-goals for this port

- Engine work that runs between turns, such as compaction while the user reads.
- Any change to ds4's inference code. ds4-py adds a library build next to it.
- DFlash itself. DFlash lives in the downstream fork; ds4-py provides the seam it needs.
- Subagents. `ds4-agent` has none, and a delegation tool would change the tool list that KV compatibility depends on (decisions 3 and 5). The engine also makes them costly (section 24).
- Saving sessions that contain images. The C agent refuses this too.
- Any HTTP server or socket between the agent and the engine.

### Anti-goals

Each of these counts as a failure even if the code works:

- A local HTTP shim between the agent and the engine.
- Re-rendering sampled tokens from their text. That breaks KV reuse and changes how `<think>` content and tool markup replay.
- Chat templates written in Python. The engine renders each family's role markers, think tags and tool-result framing in C (`ds4_chat_append_*`). A Python copy would drift from it without any error.
- Loading a library whose struct layout or function signatures differ from what ds4-py expects. That corrupts memory instead of failing.
- Code specific to the downstream agent inside ds4-py.

## 3. Constraints

- Python 3.11 or later, the harness's minimum.
- Pin the harness to an exact version. It releases often (0.17.0 on 2026-08-05, 0.59.0 on 2026-10-21) and pins its core to an exact version. Upgrade on purpose, one bump at a time, with the full test suite.
- Pin upstream ds4 by commit. `ds4.h` changes often.
- One engine per machine. `ds4_engine_open` takes an exclusive lock on `/tmp/ds4.lock` (or `DS4_LOCK_FILE`) and exits if another process holds it.
- Support development on a machine without a suitable GPU. Every non-live test runs against `FakeNative`, a fake of the native layer. Live tests and the bake-off run on a machine with 96 GB of GPU memory, which is not chosen yet.
- The Metal backend loads its kernels relative to the ds4 checkout, so a Metal library needs that path at run time.

## 4. Decisions

1. **The token transcript is the engine's truth, the message list is the harness's truth, and `DS4Model` binds them.** Sampled tokens are appended by id and never re-rendered from text, so `<think>` content and native tool markup replay byte for byte and the live KV is reused. When the harness rewrites history, only the part after the first change is re-rendered. What a rewrite costs in prefill is set by the engine's reuse rule (section 10).
2. **Context management is a plug-in point with exact costs.** A strategy is a capability. It rewrites either the request view (paging) or the stored history (compaction, bookmarks). The engine reuses its KV only when the new tokens extend the whole live checkpoint. Any other change either restores a saved checkpoint that is a prefix of the new tokens and prefills the rest, or rebuilds from token 0. `session.plan()` reports which of these three cases applies, and its exact cost, before a move. The default strategy is the C agent's compaction recipe.
3. **The C agent's tools and system prompt are the default.** `DS4AgentTools` provides the C agent's tools with their names, schemas and output bytes. `DS4Model` renders the C agent's prompt text word for word. The harness's own coding tools are a swap-in option. This follows from decision 5: a session file opened in `ds4-agent` only works if the model's tools exist there (section 12).
4. **Native tool format, rendered and parsed in ds4-py.** The tools prompt is rendered for the model's family. A streaming parser over sampled text produces `ToolCallPart`s, switches sampling to greedy inside tool syntax, and stops generation when a block closes.
5. **KV files stay compatible with `ds4-agent`.** Same location (`~/.ds4/kvcache/<sha40>.kv`), same 48-byte `KVC` header, rendered text, payload and title trailer. Sessions move between the two agents.
6. **One engine worker thread** makes every call into the engine. Progress and cancel callbacks re-enter Python with the GIL and only post to a queue or read an `Event`.
7. **The native layer is ready for forks and guarded against ABI drift.** One list of C prototypes drives the ctypes types, a compile-time signature check and a struct layout check. Libraries build from any ds4 tree and load from any path. Each binding is required or optional, and extensions can add their own.
8. **Generation is a loop over pluggable decode steps, with a hook before every sample.** ds4-py ships a plain step and, later, a speculative step. The downstream agent can add DFlash as a third step without changing the loop.

## 5. What ds4 exposes

`ds4.h` (675 lines) is already the boundary this design needs. `ds4_engine` is the loaded model and `ds4_session` is one mutable inference timeline that owns the live KV cache and logits. The caller passes a full token prefix. `ds4_session_sync()` extends the live state when that prefix starts with the whole live checkpoint, and otherwise rebuilds it from token 0 (section 10). `AGENT.md` makes this a rule: CLI and server code must not know tensor internals. So Python can sit where `ds4_agent.c` sits, with no engine changes.

| Group | Functions | Notes for the port |
| --- | --- | --- |
| Engine lifecycle | `ds4_engine_open(&e, &opts)`, `ds4_engine_close`, `ds4_engine_model_name`, `ds4_engine_model_id`, `ds4_engine_is_deepseek41`, `ds4_engine_is_glm_dsa`, `ds4_engine_is_qwen4`, `ds4_engine_has_vision`, `ds4_engine_routed_quant_bits`, `ds4_engine_mtp_draft_tokens`, `ds4_engine_vocab_size` | `ds4_engine_options` is a plain struct (`ds4.h:179`). The family flags choose the tool syntax (`ds4_agent.c:413`) |
| Prompt rendering | `ds4_chat_begin`, `ds4_chat_append_message(e, toks, role, content)`, `ds4_chat_append_think_prefix`, `ds4_chat_append_assistant_prefix`, `ds4_chat_append_multimodal_message` | The engine owns the chat templates. Roles: `system`, `user`, `assistant`, `tool` |
| Tokenizer | `ds4_tokenize_text`, `ds4_tokenize_rendered_chat`, `ds4_token_text`, `ds4_token_eos`, `ds4_token_is_stop`, `ds4_token_is_stop_for_think_mode`, `ds4_tokens_push / copy / free / starts_with` | `tokenize_rendered_chat` turns control-token spellings (`<｜User｜>`, `｜DSML｜`, `<think>`) into special tokens. Use it only for trusted text (the tools prompt, saved transcripts), never for user input |
| Session | `ds4_session_create(&s, e, ctx)`, `ds4_session_free`, `ds4_session_sync`, `ds4_session_sync_multimodal`, `ds4_session_eval`, `ds4_session_eval_speculative`, `ds4_session_sample(s, temp, top_k, top_p, min_p, &rng)`, `ds4_session_argmax`, `ds4_session_rewind(s, pos)`, `ds4_session_pos / ctx / tokens / common_prefix`, `ds4_session_invalidate` | `sync` returns `DS4_SESSION_SYNC_INTERRUPTED` (2) on cancel. `sample` picks one token from the current logits; `eval` feeds it and refreshes the logits |
| Logits | `ds4_session_copy_logits(s, out, cap)`, `ds4_session_set_logits(s, logits, n)` (`ds4.h:532–533`), `ds4_session_top_logprobs`, `ds4_session_token_logprob` | Used by the token-loop hook (section 10) |
| Callbacks | `ds4_session_set_progress(s, fn, ud)`, `ds4_session_set_display_progress`, `ds4_session_set_cancel(s, fn, ud)` | `progress_fn(ud, event, current, total)`. Events the agent sees: `prefill_chunk`, `prefill_display`, `prefill_work`. `cancel_fn(ud) -> bool` is polled during sync |
| Think and steering | `ds4_think_mode_for_context`, `ds4_think_mode_parse_level`, `DS4_THINK_LEVEL_BASE + N` (1000), `ds4_session_set_directional_steering_ffn`, `ds4_session_set_power` | Changing the think level rebuilds the prefix, so it costs a full prefill. Steering does not |
| Checkpoints | `ds4_session_save_snapshot`, `ds4_session_load_snapshot`, `ds4_session_snapshot_free` (`ds4.h:659–661`) | An in-memory copy of the whole session state. Loading one restores its token prefix with no prefill (section 10). The C agent does not use them |
| KV payload | `ds4_session_payload_bytes`, `ds4_session_stage_payload`, `ds4_session_write_staged_payload`, `ds4_session_save_payload`, `ds4_session_load_payload(s, fp, bytes, err, n)` (`ds4.h:651–658`) | The engine owns the serialized state. The file header and policy belong to the caller |
| KV store format | `ds4_kvstore.h`: `DS4_KVSTORE_FIXED_HEADER` (48), reason codes (`5` agent-system, `6` agent-session), `ext_flags` bits (bit 3 = session title), `ds4_kvstore_render_tokens_text` | A separate object, `ds4_kvstore.o`, that both `ds4-agent` and `ds4-server` link. `libds4` includes it |
| Vision | `ds4_engine_vision_encode_file`, `ds4_vision_embedding_free`, `ds4_vision_span` | Milestone 5 |

The rest of the header is serving and distributed plumbing (`ds4_sessions_eval_batch*`, tensor-parallel and RDMA calls, `ds4_engine_generate_argmax`) that the agent does not use.

Two things are missing on the C side, and neither needs a change to inference code:

- **There is no library target.** The Makefile links `CORE_OBJS` straight into each binary (`Makefile:29` for Metal, `:56` for CUDA, `CPU_CORE_OBJS` at `:30` for the CPU build). ds4-py adds a `libds4` build (section 19).
- **The agent logic lives in the binary.** `ds4_agent.c` (13,635 lines) holds the tool-call parsers, tools, compaction and persistence. ds4-py reimplements them in Python. `ds4_web.c` (1,454 lines, the browser tools) and `ds4_prompt_prefix.c` (249 lines, the `--prefix-file` parser) are ported too.

## 6. What ds4-agent does, and what ds4-py must reproduce

The C agent keeps **one `ds4_tokens transcript` as its only conversation state**. There is no message list. Text is derived by joining `ds4_token_text` over the transcript. Each behaviour below becomes a Python responsibility.

1. **User turn.** The agent appends system notes (item 12), then the `user` message, then `ds4_chat_append_assistant_prefix(think_mode)`. It passes the whole transcript to `ds4_session_sync`, which compares it with the live prefix and prefills only the new suffix.
2. **Generation.** Sample one token. Check `ds4_token_is_stop_for_think_mode` **before** eval, because stop tokens are never pushed or evaluated. Then eval, push the sampled id onto the transcript, and stream `ds4_token_text` to the renderer. With speculative decoding, tokens evaluated past a stop or a completed tool call are undone with `ds4_session_rewind(block_start + i)`, so the KV and the transcript stay aligned.
3. **End of turn.** Push `ds4_token_eos` onto the transcript for DeepSeek and Qwen, but not GLM. The next sync evaluates it.
4. **Tool calls are detected in the sampled text**, not by a grammar. A streaming parser watches for one of four syntaxes (`ds4_agent.c:295–298`):
   - DSML (V4): `<｜DSML｜tool_calls>`, `<｜DSML｜invoke name="…">`, `<｜DSML｜parameter name="…">`
   - DSML41 (V4.1): `<｜DSML｜ calls>`, `<｜DSML｜ invoke name="…">`, `<｜DSML｜ parameter name="…">`, with a space after the bar
   - GLM: `<tool_call>NAME<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`
   - Qwen: `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>`

   While the parser is inside tool syntax, the sampler switches to greedy. Generation stops the moment the closing tag completes; no stop token is needed.
5. **Tool results.** All calls in one block run in order. The results are joined, each starting `Tool result N (name):\n` (`ds4_agent.c:9521`), and appended as **one** `tool` message through `ds4_chat_append_multimodal_message`. The role is `user` when images are attached. Then a new assistant prefix is appended and the loop syncs again.
6. **Tools.** `read`, `more`, `write`, `edit`, `list`, `search`, `bash`, `bash_status`, `bash_stop`, `google_search` and `visit_page`, plus `view_image` when the model has vision. Each tool's output is capped at 128 KiB. `bash` runs through `posix_spawn` in its own process group with a monitor thread and returns head or tail excerpts plus the path of the full output file. The web tools drive a real Chrome through the DevTools protocol (section 12).
7. **Compaction is written by the model.** At 85% of the context, or when the reserve is hit, the agent syncs `transcript[:bottom] + user(compaction prompt) + assistant_prefix(no-think)`. It generates a summary greedily with `ds4_session_argmax`. Then it rebuilds the transcript as system tokens + `user(summary)` + a verbatim tail of `min(ctx/10, 50000)` tokens, cut at a user-token boundary.
8. **Prefill progress** comes from the progress callback, relative to the cached prefix (`done = current - cached_prefix`). Tokens per second are computed in the same callback.
9. **Cancel.** Ctrl+C sets a flag that the cancel callback returns. An interrupted sync leaves the transcript intact and appends EOS. Interrupted generation keeps the partial tokens. Nothing is truncated.
10. **Threading.** One worker thread makes every `ds4_session_*` call. The UI thread never touches the session.
11. **Persistence.** `~/.ds4/kvcache/<sha40>.kv` holds a 48-byte header, the rendered transcript text, the KV payload and an optional title trailer. Restore loads the payload directly, with no prefill, and takes the token history **from the session**, not from re-tokenizing the text. A stripped file has no payload; its text is re-tokenized with `ds4_tokenize_rendered_chat` and prefilled.
12. **System notes** (`ds4_agent.c:1478–1625`):
    - A one-time date note before the first user message: `Current local date and time at session start: <time>. Use this only when date or time matters.`
    - Programming hints, off by default and switched with `/hints on|off`. The full instructions are sent once per live context; after that, a short note is sent only when the setting changes. Compaction resets this state.
13. **System prompt.** For DSML, V4.1 first gets an empty `system` message; then the tools prompt goes through the rendered tokenizer, so its `｜DSML｜` markers become the special token. For GLM and Qwen, the tools prompt is a plain `system` message. Text from `--system` follows as a separate plain `system` message prefixed with `\n\n`. It never goes through the rendered tokenizer.
14. **System prompt reminder.** After the transcript has grown 50,000 tokens since the system prompt was last seen, the agent appends `[System prompt reminder follows.]`, the tools prompt and `[End system prompt reminder.]`, using the same tokenizer rule as item 13, and then the `--system` text again.
15. **Rounds.** A user turn may contain any number of assistant and tool rounds. There is deliberately no limit on tool calls. Before each assistant continuation, the agent checks for compaction, appends the reminder if due, appends the hints note if due, and then appends the assistant prefix (`ds4_agent.c:10340`).
16. **`--prefix-file`.** A file of `USER:` and `ASSISTANT:` turns is appended after the system prompt. Assistant turns get an EOS, except on GLM.
17. **Command line.** ds4-py accepts the agent-side flags: `--prompt`, `--prompt-file`, `--system`, `--prefix-file`, `--non-interactive`, `--raw`, `--temp`, `--top-p`, `--min-p`, `--seed`, `--think`, `--think-level`, `--think-max`, `--nothink`, `--edit-upto`, `--trace`, `--tokens` and `--ctx`. The engine flags pass through to the engine options: `--model`, the backend flags, `--mtp*`, `--dspark*`, `--ssd-streaming*`, `--gpu-*`, `--threads`, `--prefill-chunk`, `--power`, `--dir-steering-*`, `--vision`, `--warm-weights`, `--quality` and `--chdir`.

## 7. Framework choice

Build on **Pydantic AI with the Harness package**. Its model layer is an abstract class (`Model`, with `request()` and `request_stream()`) that has no HTTP in its contract. Its capability hooks (`before_model_request`, `before_tool_execute`, `after_tool_execute`, `get_toolset`, `get_instructions`, `for_run`, `on_event`, `prepare_tools`) cover every behaviour in section 6. It also ships the guards, tracing and persistence ds4-py would otherwise write.

LocalHarness was the other serious candidate, and it is not the base. Its inner loop would accept an in-process model, but its `start` command, token counting, window probes, `/model` and reconnect logic all need an OpenAI-compatible server URL. Its plugin API has no model-provider slot. A local HTTP shim is exactly what this design rules out. Two of its ideas are worth copying into capabilities later: the rule that untrusted-ingest tools and dangerous host tools may not be active together, and its JSONL event log.

If the harness's 0.x API churn becomes a cost, Pydantic AI core alone is the fallback. `Model`, capabilities, `FunctionToolset`, MCP, instrumentation and persistence are all in core. Since ds4-py ports the C agent's tools anyway, the harness supplies less here than the first draft assumed.

The first draft's full comparison (LocalHarness, mini-swe-agent, smolagents, Pi, OpenCode, llama-cpp-python) still stands and is not repeated here.

## 8. Architecture

The harness calls the model adapter as it would call any provider. The adapter talks to the engine facade in Python objects. The facade is the only client of the native layer, and the native layer is the only client of `libds4`. Tools, guards and records never see the engine.

### Process and threads

- **Main thread:** the asyncio loop that runs `Agent.run_stream_events()`, the TUI and the event log.
- **Engine worker thread:** created by `ds4py.open()`. It is the only thread that calls into `libds4`. `DS4Model.request_stream()` waits on it through futures and an `asyncio.Queue` fed by `loop.call_soon_threadsafe`.
- **Tool subprocesses:** `bash` jobs and Chrome. `bash` never blocks the engine thread.

There is no inference server, no socket and no second process holding model weights.

### Who owns what

| Concern | Owner | Where |
| --- | --- | --- |
| Conversation flow, turn loop, retries | Python | `pydantic_ai.Agent` run loop |
| Token loop (sample, stop, eval, push, rewind) | Python | `ds4py.engine.worker` and its decode steps |
| Prompt rendering to tokens | C templates, driven by Python | `PromptBuilder` calling `ds4_chat_append_*` |
| Tools prompt text | Python, copied word for word from C | `ds4py.prompts` |
| Tool-call parsing | Python | `ds4py.engine.toolsyntax` |
| Tool execution | Python, ported from C | `ds4py.tools`, exposed as the `DS4AgentTools` capability |
| Policy and approvals | Python | `DS4Approvals` capability |
| Context management | Python | `DS4Compaction` (default) or another strategy |
| Session-lifetime state of capabilities (paging cuts, bookmark tables, `bash` jobs) | Python | `session.state()` in `ds4py.engine.session` (section 13) |
| System notes and reminders | Python | `DS4SystemNotes` capability |
| Canonical history, save records, resume | Python | `StepPersistence` snapshots and `ds4py.store` save records (section 15) |
| Run records and tracing (telemetry) | Python | `DS4RunRecord`, `Instrumentation` |
| Model computation, live KV, logits, speculative decoding | C | `libds4` |
| KV payload bytes | C | `ds4_session_*_payload` |
| KV file header, naming, listing | Python | `ds4py.store` |

### Packages

- **`ds4py`** (the core) has no dependencies outside the standard library. It holds the native layer, the ABI tools, the engine facade, the tool-call parsers, the vendored prompts, the tool implementations, the KV store and a `ds4py raw` command. People who only want the engine install this.
- **`ds4py[agent]`** adds `pydantic-ai-harness`. It holds `DS4Model`, the capabilities and the TUI.

### Module map

| Module | Lines (estimate) |
| --- | --- |
| `_native/` — the binding list, loader, ctypes calls and `FakeNative` | 700 |
| `abi.py` — probe generator, layout and signature checks, `ds4py build` and `ds4py check` | 300 |
| `engine/transcript.py` | 150 |
| `engine/worker.py` — jobs, the decode loop, decode steps, the logits hook | 450 |
| `engine/prompt.py` | 200 |
| `engine/toolsyntax/{dsml,glm,qwen}.py` | 500 |
| `engine/session.py` — the public facade | 300 |
| `prompts/` — vendored C prompt text and the builders that assemble it | 150, plus the text |
| `tools/` — `read`, `more`, `write`, `edit`, `list`, `search`, `bash` jobs, web, `view_image`, result framing | 1,800 |
| `store.py` | 250 |
| `model.py` — `DS4Model` and the binding | 400 |
| `capabilities/` | 500 |
| `cli.py`, `tui.py` | 700 |

That is about 6,400 lines, up from the first draft's 3,400, because ds4-py now ports the C agent's tools (about 1,800 lines) and adds the ABI guard and the extension API. It still replaces 15,338 lines of C (`ds4_agent.c`, `ds4_web.c` and `ds4_prompt_prefix.c`).

## 9. The native layer: `ds4py._native`

One package is the only place that imports `ctypes`, declares `argtypes` and `restype`, or holds a raw pointer. Everything above it receives Python objects. Three rules keep this true:

- No other ds4-py module imports `_native` internals. The one documented way past this rule is the extension API below.
- `_native` exports functions and opaque handle classes, never `ctypes.Structure` instances.
- Every C error path becomes a Python exception at this layer.

### One list of bindings

Each binding is a record with a name, its C prototype as a string, and a `required` flag:

```python
Binding("ds4_session_sync",
        "int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen)",
        required=True)
```

The ctypes types are derived from the prototype through a fixed type map (`int`, `bool`, `float`, `double`, the fixed-width integers, `size_t`, `char *`, `const char *`, `void *`, `FILE *`, pointers to the opaque engine types, pointers to the structs `_native` defines, and the callback typedefs). A type outside the map is an import error. The prototype is written once, and the same string feeds the signature check.

What the list wraps:

| Python name | C functions | Marshalling |
| --- | --- | --- |
| `open_engine(opts) -> EngineHandle` | `ds4_engine_open`, `ds4_engine_close` | Fills a `ds4_engine_options` structure from a frozen dataclass. Strings are kept alive on the handle for the engine's lifetime |
| `engine_info(h) -> EngineInfo` | `ds4_engine_model_name`, `model_id`, the `is_*` flags, `has_vision`, `routed_quant_bits`, `mtp_draft_tokens`, `vocab_size` | Read once at open. The facade branches on `family` (`deepseek4`, `deepseek41`, `glm`, `qwen`) |
| `render(h, ops) -> list[int]` | `ds4_chat_begin`, `ds4_chat_append_message`, `ds4_chat_append_think_prefix`, `ds4_chat_append_assistant_prefix`, `ds4_chat_append_multimodal_message` | Takes a list of operations, renders them into one `ds4_tokens`, copies it out as a Python list and frees the C vector. Python never holds a `ds4_tokens` |
| `tokenize(h, text, rendered=False) -> list[int]` | `ds4_tokenize_text`, `ds4_tokenize_rendered_chat` | `rendered=True` only for trusted text |
| `token_text(h, id) -> bytes` | `ds4_token_text` | Bytes, not `str`: a token can split a UTF-8 sequence, and the facade's decoder reassembles it |
| `is_stop`, `eos`, `special_token(h, text) -> int \| None` | `ds4_token_is_stop_for_think_mode`, `ds4_token_eos`, `ds4_tokenize_rendered_chat` | `special_token` requires the text to tokenize to exactly one id, like the C agent's `agent_special_token_id` |
| `create_session(h, ctx) -> SessionHandle`, `free` | `ds4_session_create`, `ds4_session_free` | `SessionHandle.__del__` frees; the facade also calls `close()` explicitly |
| `sync(s, tokens, images=None, *, progress, cancel) -> SyncResult` | `ds4_session_sync`, `ds4_session_sync_multimodal`, `ds4_session_set_progress`, `ds4_session_set_display_progress`, `ds4_session_set_cancel` | Builds a `ds4_tokens`, installs the callbacks, calls sync, and clears the callbacks in a `finally`. Return 0 is success, 2 raises `Interrupted`, anything else raises `EngineError(err)` |
| `sample(s, temp, top_k, top_p, min_p, rng) -> int` | `ds4_session_sample` | `rng` is a `c_uint64` owned by the facade, so `--seed` runs are reproducible |
| `copy_logits(s, buf) -> int`, `set_logits(s, buf)` | `ds4_session_copy_logits`, `ds4_session_set_logits` | `buf` is a reusable float buffer of vocabulary size, owned by the facade |
| `eval(s, token)`, `eval_speculative(s, token, max, eos, sampling, rng) -> list[int]` | `ds4_session_eval`, `ds4_session_eval_speculative` | The speculative call returns the accepted ids; the first is the token that was fed. Capacity 17, as in the C agent |
| `rewind`, `pos`, `ctx`, `tokens`, `common_prefix`, `invalidate` | the matching `ds4_session_*` | `tokens` copies out of `ds4_session_tokens` |
| `payload_bytes`, `stage_payload`, `write_staged(p, file)`, `load_payload(s, file, n)` | `ds4_session_payload_bytes`, `ds4_session_stage_payload`, `ds4_session_write_staged_payload`, `ds4_session_payload_file_free`, `ds4_session_load_payload` | The C API wants a `FILE *`. The module opens one with libc `fdopen(os.dup(file.fileno()))`, so Python keeps ownership of its own file object |
| `save_snapshot(s) -> Snapshot`, `load_snapshot(s, snap)` | `ds4_session_save_snapshot`, `ds4_session_load_snapshot`, `ds4_session_snapshot_free` | `Snapshot` wraps a `ds4_session_snapshot`, a struct `_native` defines and the layout check covers. The bytes stay in C-allocated memory until `Snapshot.close()` calls `ds4_session_snapshot_free`. Python sees only the size |
| `set_steering`, `set_power`, `think_mode_for_context`, `parse_think_level` | `ds4_session_set_directional_steering_ffn`, `ds4_session_set_power`, `ds4_think_mode_*` | Think modes are an `IntEnum` mirroring `ds4_think_mode`, with `level(n) = 1000 + n` |
| `render_tokens_text` | `ds4_kvstore_render_tokens_text` | Used by the store to write the transcript text exactly as C does |
| `vision_encode(h, path)` | `ds4_engine_vision_encode_file`, `ds4_vision_embedding_free` | Milestone 5. The embedding's `float *` is never exposed |

Every function in this table exists in both upstream and the downstream fork, so every binding is required today.

### The ABI guard

Declaring `argtypes` does not protect against a changed library. ctypes cannot read function signatures from a shared library, so a changed signature or struct layout still loads and then corrupts memory. The symbol check in the first draft only proves the names exist. ds4-py runs three checks instead.

1. **Signature check (at build time).** For each binding, the probe compiles one tiny C file that includes `ds4.h` and `ds4_kvstore.h` and assigns the function to a pointer of the declared type, for example:

   ```c
   uint64_t (*p)(ds4_session *) = ds4_session_payload_bytes;
   ```

   It is compiled with `-Werror=incompatible-pointer-types -Werror=implicit-function-declaration`. One file per binding means a missing optional function does not hide the result of the others. A required binding that fails stops the build with the compiler's message. An optional binding that fails is recorded as absent.

   This catches the case the other checks miss: a function that keeps its name but changes its arguments. Upstream added a `uint32_t flags` argument to `ds4_session_tp_spec_cycle`, and the downstream fork still has the old version.

2. **Layout check (at build and load time).** The probe also builds a small program that prints `sizeof` and every field's `offsetof` for each struct `_native` fills or reads, starting with `ds4_engine_options` and `ds4_tokens`. At load time, `_native` compares these numbers with the layout of its own `ctypes.Structure` definitions. Any difference refuses the library.

3. **Header hash (informational).** The probe records the SHA-256 of `ds4.h` and `ds4_kvstore.h`. ds4-py lists the hashes of the pinned upstream commit. An unknown hash prints a warning, not a refusal; otherwise no fork could load. The signature and layout checks make the hash informational.

The probe writes its results to `libds4.abi.json` next to the library. For each binding, the file records the normalized prototype the probe compiled and the result. Normalizing collapses whitespace and drops parameter names, so `const ds4_tokens *prompt` and `const ds4_tokens*` compare equal. The file also records the library's own SHA-256 and the version of `_native`'s type map. At load time, `_native` refuses the library if:

- `libds4.abi.json` is missing;
- the library's hash does not match the one recorded, which means the library was rebuilt without a new probe;
- the type-map version differs from the installed one;
- an installed binding, including an extension's, has no recorded result, or its normalized prototype differs from the recorded one, which means the bindings changed after the probe ran;
- the layout differs; or
- any required binding failed its signature check.

The prototype comparison runs before any callable is created, so a stale probe result never reaches ctypes. Without it, an upgrade that changes a prototype, such as the `uint32_t flags` added to `ds4_session_tp_spec_cycle`, would pass with the old library and the old manifest, and ctypes would call the old function with the new `argtypes`. The error names the binding, shows both prototypes and says to run `ds4py build` again.

### Required and optional bindings

Each binding has a `required` flag. Optional bindings are bound only if they passed the probe; calling an absent one raises `MissingBinding`. Today nothing needs an optional binding, so this is one field and a branch in the loader. It matters once ds4-py wraps something newer than the downstream fork has.

`ds4py check --lib PATH` loads the library without opening a model and prints the library path, the ds4 tree's commit if it is a git checkout, whether the header hash is known, the layout result, and each optional binding as present or absent.

### Loading

The library comes from `ds4py.open(..., lib=PATH)`, then the `DS4_LIB` environment variable, then the default location that `ds4py build` writes to. It is loaded once with `ctypes.CDLL(path, mode=RTLD_LOCAL)`. A second load with a different path in the same process is refused.

### The extension API

The downstream agent needs to add its own bindings, such as the DFlash functions and `ds4_engine_sampling_defaults`, without forking ds4-py. Two rules block that: only `_native` touches ctypes, and only the worker thread calls the engine. The extension API is the documented way past both.

- **Declaring bindings.** An extension lists extra `Binding` records in a module registered under the `ds4py.bindings` entry point group. `ds4py build` includes them in the probe, so the ABI guard covers them, and `ds4py check` lists them.
- **Calling them.** `engine.native.bind(name)` returns a callable for a declared binding that passed the probe. A name that is not in the probe results is refused. Its prototype was already compared with the recorded one when the library loaded.
- **Running on the worker.** `session.on_worker(fn)` runs `fn(handles)` on the worker thread between jobs and returns a future. `handles` carries the raw engine and session pointers for use with bound functions.
- **Sampling defaults.** `engine.sampling_defaults` can be overridden per engine, so an extension can apply the values its fork's `ds4_engine_sampling_defaults` returns.

### Callbacks and the GIL

`ds4_session_sync` runs for seconds to minutes and calls `progress_fn` and `cancel_fn` from inside the engine, on the calling thread. ctypes releases the GIL around a foreign call and re-acquires it for each callback, so the TUI and the event log keep running while the engine prefills on the worker thread.

The callbacks do as little as possible. `progress_fn` puts `(event, current, total)` on a `queue.SimpleQueue` and returns. `cancel_fn` reads one `threading.Event` and returns a bool. The `CFUNCTYPE` objects are created once per session and stored on the `SessionHandle`, so they are not garbage-collected while the engine holds their addresses.

The cancel path works because the engine polls the callback inside its prefill loop and returns `DS4_SESSION_SYNC_INTERRUPTED` after draining in-flight GPU work. Python must not raise out of a callback, because ctypes prints the traceback and returns 0. Exceptions are stored and re-raised after the foreign call returns.

### Threading contract

All `SessionHandle` methods are called from one thread: the engine worker the facade owns. In debug builds, `_native` asserts this by recording the creating thread's id. The engine is not re-entrant per session, and neither the C agent nor ds4-py calls it from two threads.

### Errors and memory

Every call that takes `char *err, size_t errlen` gets a 512-byte buffer and raises `EngineError(msg)` on a non-zero return, except the documented `2` from sync. `ds4_tokens` vectors are created, filled and freed inside one `_native` function. A vision embedding's ownership passes to the session on `sync_multimodal`, as the C API specifies, and the Python wrapper marks it consumed so it is not freed twice.

### `FakeNative`

`FakeNative` implements the same functions over a toy tokenizer and scripted logits. It supports logits copy and set, rewind, progress and cancel callbacks, payload save and load to bytes, snapshots, and a fake multi-token decode step. Every non-live test runs on it.

`FakeNative` follows the engine's reuse rule, not a simpler one. A sync keeps the state only when the prompt extends the whole live checkpoint; any other sync prefills from token 0. A rewind past the last speculative step invalidates the checkpoint. A snapshot restores the whole state. So a `Plan` test on `FakeNative` checks the same three cases as the engine (section 10).

## 10. The engine facade: `ds4py.engine`

The facade is the part of `ds4_agent.c` that deals with tokens rather than files or terminals, rewritten in Python on top of `_native`. It owns the worker thread, the transcript, prompt rendering, tool-syntax detection, sampling policy, decode steps and the logits hook. It exposes ordinary Python types. The harness adapter (section 11) is its main client, and `ds4py raw` uses it directly.

### `Transcript`

A `Transcript` is a `list[int]` plus a parallel list of `Segment` records `(start, end, kind, meta)`. `kind` is one of `system`, `user`, `assistant_prefix`, `assistant`, `tool`, `eos`, `note`, `reminder` or `compaction`. The token list is the truth, as in C. The segments are an index over it, so the facade can find turn boundaries, cut a compaction tail at a `<｜User｜>` boundary, or render `/history` without scanning for special tokens. Segments never change the token list. Sampled assistant tokens are appended one at a time by id and never re-rendered from text.

### `EngineWorker`

One `threading.Thread` owns the `SessionHandle`. Callers submit jobs to a queue and get back an iterator of events. The worker runs one job at a time.

| Job | What it does | Events it yields |
| --- | --- | --- |
| `SyncJob(tokens, images)` | plans the sync (below), loads the checkpoint if the plan is `restore`, then calls `sync` with progress and cancel callbacks installed | `Prefill(done, total, cached_prefix, tps)` at most every 100 ms, then `Synced(pos, plan)` or `Interrupted(pos)` |
| `CheckpointJob(key, drop=False)` | saves or frees a snapshot of the live state (below) | `Checkpointed(key, tokens, bytes)` or `Dropped(key)` |
| `GenerateJob(sampling, stop_policy, max_tokens)` | the decode loop below | `Token(id, text_bytes)`, `Thinking(on/off)`, `ToolSyntax(state)`, `Stopped(reason)`, where the reason is `stop_token`, `tool_call_complete`, `max_tokens`, `context_reserve`, `interrupted` or `parser_error` |
| `CompactJob(reason)` | the compaction recipe (section 13), generated with `argmax` | `Prefill…`, `Token…`, `Compacted(summary, dropped, kept)` |
| `SaveJob(path)`, `LoadJob(path)` | section 17 | `Saved(bytes)`, `Loaded(tokens)` |
| `ControlJob(steer=, power=, think=)` | between turns only | `Applied` |
| `WorkerCallJob(fn)` | `session.on_worker(fn)` | `Done(result)` or `Failed(error)` |

The UI sets one shared cancel `threading.Event` on Ctrl+C. The worker clears it when it reports `Interrupted`. This keeps the C agent's contract: an interrupted sync keeps the transcript, and the caller appends EOS; interrupted generation keeps the partial tokens.

A `context_reserve` stop does not end the round. `run_round` yields `ReserveReached(needed)` and waits. Its consumer makes room, rebinds and calls `round.resume()`, which syncs the new transcript and starts a `GenerateJob` that continues the same assistant message. The partial tokens stay at the end of the new transcript, as in C. A consumer that cannot make room calls `round.fail()`, and the round ends with `Stopped(context_exhausted)`. `ds4py raw` has no context strategy, so it always fails here. Section 11 describes how `DS4Model` makes room inside an open stream.

### KV reuse and `Plan`

The engine keeps its KV for a sync only when the new tokens start with the whole live checkpoint. Every backend checks `prompt->len >= s->checkpoint.len && ds4_tokens_starts_with(prompt, &s->checkpoint)` (for DeepSeek V4, `ds4.c:77689`). Any other sync resets the graph and prefills from token 0. `ds4_session_common_prefix` (`ds4.c:77893`) only counts matching tokens; it does not say how much KV can be kept. `ds4_session_rewind` cannot close the gap, because the DeepSeek compressors cannot be rolled back without a saved frontier (`ds4.c:87429–87431`). The C agent relies on the same rule: `agent_worker_sync_tokens` counts any sync that does not extend the session as a full prefill (`ds4_agent.c:5291`).

ds4-py adds checkpoints, which the C agent does not use. A checkpoint is an in-memory snapshot of the whole session state (`ds4_session_save_snapshot`) at a known token prefix. Loading it (`ds4_session_load_snapshot`) restores that prefix with no prefill, and the sync after it is an extension. A checkpoint costs host memory equal to `ds4_session_payload_bytes()` at that point, which grows with the prefix length.

- The facade holds one checkpoint by default, `system`. It is saved when the system prefix is first synced or loaded from `sysprompt.kv` (section 17).
- A strategy can hold more through `session.checkpoint(key)`, up to `max_checkpoints` (default 4) besides `system`. A request beyond the limit is refused; the facade never evicts a checkpoint on its own.
- Distributed sessions cannot snapshot (`ds4.c:64881`), so they have no checkpoints.

`session.plan(tokens)` renders nothing and syncs nothing. It returns `Plan(kind, reused, to_prefill, fits, checkpoint)`:

| `kind` | When | `reused` | `to_prefill` |
| --- | --- | --- | --- |
| `extend` | the tokens start with the whole live checkpoint | the live position | the tokens after it |
| `restore` | not an extension, and a held checkpoint's tokens are a prefix of the new tokens; the longest such checkpoint is chosen | that checkpoint's length | the tokens after it |
| `rebuild` | neither | 0 | all the tokens |

`fits` is true when the tokens plus the generation reserve fit the context. `SyncJob` carries out the plan exactly, so `Synced` reports the same `kind`, and the prefill total equals `to_prefill`. A `Plan` describes the state when it was made; the worker plans again before it acts.

A restore followed by an extension may not give the same state as a rebuild, bit for bit. The two paths do not evaluate the tokens in the same batches, so their floating-point results can differ. A live check (section 20) compares them on each backend. On a backend that fails the check, `restore` is disabled, and `plan()` reports `rebuild` instead.

### Decode steps

The decode loop and the way tokens are produced are separate. The loop owns the stop checks, the tool-syntax parser, the greedy switch, the logits hook and pushing tokens onto the transcript. A decode step owns producing tokens and keeping the KV in line with them.

The contract:

1. Before each sample, the step calls `ctx.before_sample()`, which runs the logits hooks, and uses `ctx.sampling()` for the current settings. The settings change to greedy while the parser is inside tool syntax.
2. After each token, the step calls `ctx.accept(token)`. The loop answers `continue`, `stop_before` (a stop token: not kept, and not evaluated or rewound away) or `stop_after` (a tool block closed: the token is kept).
3. When the loop says stop, or Ctrl+C is set, the step leaves the KV equal to the kept tokens before it returns. It rewinds any extra evaluated tokens and closes any open engine state.
4. A step declares whether it calls `before_sample` before every sample. The loop only uses steps that do while a logits hook is installed.

ds4-py ships these steps:

| Step | Milestone | Calls the hook before every sample? |
| --- | --- | --- |
| `PlainStep`: sample, check, eval, one token at a time | M1 | yes |
| `FakeMultiTokenStep`, in `FakeNative` tests only, to prove the seam handles multi-token blocks | M2 | yes |
| `SpeculativeStep` over `ds4_session_eval_speculative` | M5 | no. The engine samples the draft tokens internally (`ds4.h:578`), so while a hook is installed the loop uses `PlainStep` |

The downstream agent's DFlash would be a third step. It drives `ds4_session_set_verify_mode`, `ds4_session_dflash_draft`, `ds4_session_verify`, `ds4_session_verify_select` with a sample for each row, and `ds4_session_verify_commit`. While a verify is open, the downstream fork's engine refuses eval, sync, rewind and payload save or load. Rule 3 covers this: the step closes its verify before it returns. Because `verify_select(row)` makes the logits calls act on that row, DFlash can call `before_sample` for each row and keep the hook active.

### The logits hook

`session.add_logits_hook(hook)` installs a function that runs before every sample. It receives a `LogitsView` with:

- `logits()`, which copies the current logits into the facade's reusable buffer through `ds4_session_copy_logits`. Nothing is copied unless the hook asks.
- `set_logits(values)`, through `ds4_session_set_logits`.
- `bias(token_id, delta)`, a helper that copies, changes one value and sets.
- Read-only state: the position, whether the model is inside `<think>`, the number of tokens in the current think block, the parser state, and the tokens sampled so far in this round.

Hooks run on the worker thread and must not call the engine. With no hooks installed, the loop does no extra work. If a hook makes a multi-token step produce a different token than it would have, the accepted run ends at that token, in the same way as a draft mismatch.

This is the hook the downstream agent asked for, for example to stop overthinking by raising the `</think>` logit after a number of think tokens.

### `PromptBuilder`

`PromptBuilder` builds the token prefix for a round from the transcript and the engine's family, using only `_native.render` and `_native.tokenize`. It knows:

- the role names the templates accept: `system`, `user`, `assistant` and `tool`;
- when to use the rendered tokenizer (the DSML tools prompt, the reminder, a restored transcript) and when never to (user text, tool results, `--system` text);
- each family's rules: DeepSeek and Qwen get an EOS at the end of every assistant turn and GLM does not; V4.1 uses its own DSML tag spellings and gets an empty `system` message before the tools prompt; Qwen takes its reasoning effort as a `system` message rather than a think prefix.

The tools prompt comes from `ds4py.prompts` when the C agent's tools are active (section 12). Otherwise it is generated from the harness's tool definitions.

### `ToolSyntax`

A streaming parser over sampled text, with one implementation per syntax family:

- `DsmlParser` handles both the V4 and the V4.1 tag spellings.
- `GlmParser` handles `<tool_call>NAME<arg_key>…<arg_value>…</tool_call>`.
- `QwenParser` handles `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>`.

Each exposes `feed(bytes) -> ParserState` and `calls() -> list[ToolCall(name, args, raw)]`, where each argument is a string or parsed JSON. The parsers copy the C parser's leniencies: a missing leading bar, a bare `<｜DSML｜invoke`, an optional `｜` before `>` in closing tags, and GLM's rule that another `<tool_call>` continues the block. They apply the `ds4_tool_text.h` un-escaping to string parameters. Markup inside `<think>` is suppressed and reported as a parser error, as in C.

### `SamplingPolicy`

`temperature`, `top_p`, `min_p`, `top_k=0` and `seed`, with the per-family defaults the C agent applies (GLM: `top_p` 0.95, `min_p` 0), the per-engine override from the extension API, and the greedy override from the parser. The `rng` state lives here, so a `--seed` run is reproducible in both Python and C.

### `Decoder`

The decoder reassembles UTF-8 from `token_text` bytes, tracks `<think>` and `</think>` with a 16-byte pending buffer, and emits `TextDelta(text, thinking)`. It is for display only; nothing it does feeds back into the transcript.

### Public surface

```python
engine = ds4py.open(model="gguf/ds4flash.gguf", ctx=100_000, backend="metal",
                    think=ThinkMode.HIGH, seed=None, lib=None)
session = engine.session()                  # one per process
session.transcript                           # read-only Transcript view
session.append_user(text, images=())
session.append_tool_results(results)         # one "tool" message for the whole block
session.append_system_note(text)             # date note, hints, reminder
for ev in session.run_round():               # sync + generate
    ...                                      # Prefill / TextDelta / ToolCalls / Stopped
session.plan(view_tokens)                    # exact Plan(kind, reused, to_prefill, fits, checkpoint) without syncing
session.checkpoint(key); session.drop_checkpoint(key)  # snapshots for later restores
session.state(key, factory)                  # session-owned state for capabilities (section 13)
session.ctx_used                             # ds4_session_pos, exact
session.compact(reason)
session.save(path); session.load(path)
session.fork()                               # new session id and created_at; transcript and live KV kept (section 17)
session.add_logits_hook(hook)
session.on_worker(fn)                        # extension API
session.cancel()                             # thread-safe
engine.native.bind(name)                     # extension API
```

## 11. The harness adapter

The harness sees one `Model` subclass and a list of capabilities. The hard part is that Pydantic AI treats the message list as the truth while the engine treats the token transcript as the truth. `DS4Model` holds the binding between them.

### `DS4Model(Model)`

| Member | Implementation |
| --- | --- |
| `system` | `'ds4'` |
| `model_name` | `engine_info.model_name` |
| `profile` | `context_window=ctx`, tools and thinking supported, `thinking_tags=('<think>', '</think>')`, inline system prompts supported (the ds4 templates accept a `system` message anywhere, which is how the C agent appends its notes), `default_structured_output_mode='prompted'`, no JSON-schema output, no tool-return schema |
| `request()` | `prepare_request`, then `request_stream` collected into a `ModelResponse` |
| `request_stream()` | `prepare_request`; bind the messages to the transcript (below); run `session.run_round()` on the worker; yield a `DS4StreamedResponse`. Its `_get_event_iterator` turns facade events into `handle_text_delta(vendor_part_id='content', content=…, thinking_tags=…)` and, when the parser completes a block, one `handle_tool_call_part(vendor_part_id=i, tool_name=…, args=…, tool_call_id=…)` per call. `finish_reason` is `'tool_call'`, `'stop'` or `'length'`, and `'length'` means only that `max_tokens` was reached. A `ReserveReached` event never ends the stream (below). Usage is `RequestUsage(input_tokens=prefilled, cache_read_tokens=cached_prefix, output_tokens=generated)` |
| `close_stream()` | `session.cancel()`. The partial tokens stay in the transcript, as in C |
| `count_tokens()` | renders the messages without syncing and counts the tokens |

### Running out of context inside a response

When generation reaches the reserve in the middle of a response, the response must not end. `before_model_request` has already run, and ending the response early does not lead to another request in the pinned harness. With `str` output, partial text becomes the final result, because a text response has no special `'length'` handling. A thinking-only or empty response with `finish_reason='length'` raises `UnexpectedModelBehavior` (`_agent_graph.py:2449–2485`).

So `DS4StreamedResponse` handles `ReserveReached` itself, inside the open stream:

1. It finds the context owner in session state (section 13) and awaits `owner.make_room(run_context, needed)`. The harness passes `run_context` to `request_stream`, and `run_context.messages` is the run's live history list.
2. The owner frees space in its own way and returns the new request view. `DS4Compaction` runs its recipe with the open assistant message kept and rewrites `run_context.messages[:]`. `RollingWindow` moves its cut and returns the new view.
3. `DS4Model` rebinds to that view, with the partial response's tokens after it, and calls `round.resume()`. Generation continues, and its text and thinking deltas extend the same parts of the same `ModelResponse`.
4. If there is no owner, or `make_room` returns `None`, the stream raises `ContextExhausted`. The run fails instead of returning partial text as an answer. The transcript keeps the partial tokens, as after an interrupt.

The TUI shows the compaction through the session event bus (`before_compact`, `compacted`), not as response text.

### Binding messages to the transcript

`DS4Model` keeps a binding: for each `ModelMessage` it has rendered, the transcript segment that message produced, keyed by the message's identity and a content hash. On every request it walks `request_context.messages` from the front:

1. While messages match bound segments in order, nothing is rendered. This is the common case: the new transcript extends the live one, so the sync is an `extend` plan and prefills only the new tokens.
2. A `ModelResponse` that `DS4Model` produced is bound to its sampled tokens plus EOS, so it is never re-rendered from text. A `ModelResponse` the harness made up (a compaction summary, a replayed history) is rendered as a normal assistant message.
3. New request parts render in order:
   - `SystemPromptPart` and `InstructionPart` become `system` messages, never through the rendered tokenizer.
   - `UserPromptPart` becomes `user`, through the multimodal call when it has images.
   - All `ToolReturnPart`s of one request become **one** `tool` message, each body starting `Tool result N (name):\n`.
   - `RetryPromptPart` goes into the same `tool` message as the C agent's error text. An unknown tool reads `Tool error: unknown tool: <name>` (`ds4_agent.c:9508`). Validation failures use the C agent's text for the same case.
   - Parts that a capability marks as ds4 notes (`metadata['ds4.note']` is `date`, `hints` or `reminder`) are rendered the way C renders them (section 6, items 12–14).
4. The first time a request has tool definitions (`params.function_tools`), the tools prompt is rendered once at the front of the transcript. If the tools carry the `DS4AgentTools` marker, the prompt is the C text from `ds4py.prompts`. Otherwise it is generated from the definitions. The harness's instructions go after it as a separate plain `system` message, in the slot where C puts `--system` text. A later change in the tool set re-renders from that point, which costs a full prefill. So capabilities that add or hide tools during a run (`ToolSearch`, deferred loading) are left out.
5. If the message list differs from the binding, because a capability rewrote the messages, the binding keeps the common prefix and re-renders the rest. That saves rendering work and keeps sampled tokens exact, but it does not limit the prefill. The engine cannot keep KV past a change, so the sync restores the longest held checkpoint before the change and prefills from there, or rebuilds from token 0 (section 10).
6. Finally the assistant prefix for the current think mode is appended, and `run_round` syncs the whole transcript.

With the default capability list and no `--system` text, the harness's instructions are empty, so the system prefix is byte-identical to `ds4-agent`'s and `sysprompt.kv` is shared between the two agents.

### Default capability list

```python
agent = Agent(
    DS4Model(session),
    capabilities=[
        DS4AgentTools(project_dir, edit_upto=False),  # the C agent's tools and prompt; view_image when the model has vision
        DS4SystemNotes(),                             # date note, hints, 50,000-token reminder
        DS4Approvals(),                               # section 14
        DS4Compaction(soft=0.85),                     # a context strategy; swap for RollingWindow(...) or Bookmarks(...)
        DS4RunRecord(path),                           # events.jsonl, telemetry only (section 15)
        StepPersistence(FileStepStore(session_dir / 'steps')),  # canonical history; each run gets conversation_id=session.id
        Instrumentation(),
    ],
)
```

Each item in this list either matches something the C agent does or adds nothing to the prompt. The following are left out of the default:

- `RepoContext`: the C agent does not read `AGENTS.md`, so adding it would change the system prefix.
- `AskUser`: it is a tool the model can call, and the C agent has no such tool.
- `ToolOutputLimits`: the C tools enforce their own 128 KiB cap with the C agent's truncation note.
- The guardrail classes (`InputGuardrail`, `OutputGuardrail`, `ToolGuardrail`): the C agent has none. They stay available to extensions.
- `SubAgents`: subagents are a non-goal (sections 2 and 24).
- `RepairToolArguments`: arguments arrive as strings from the native parser, not as JSON.
- `ToolCallJudge` and `ApprovalReviewer`: a second model call on the live session would destroy the cached prefix. Policy here is rules, not a judge model.

### The harness-tools option

`ds4py --tools harness` replaces `DS4AgentTools` with the harness's coding tools: `LocalWorkspace`, `FileSystem`, `Shell`, `RepoContext`, `WebSearch`, `WebFetch` and `ToolOutputLimits`. `DS4Model` then generates the tools prompt from their definitions. This option is the extensibility check for tool sets. With it, the parity checks and KV compatibility with `ds4-agent` do not apply, because `ds4-agent` cannot run `read_file`. The pass-rate gate still does.

## 12. The C agent's tools as harness capabilities

The C agent's tools have three parts, and each needs a home.

### a. The prompt is hand-written text

For DSML, `agent_build_dsml_tools_prompt` (`ds4_agent.c:1291`) joins static strings: an introduction with usage advice, one of two edit variants (`edit_exact` or `edit_upto`), the JSON schemas, and a `# Rules` section. The `view_image` schema goes just before `\n# Rules\n` when the model has vision. V4.1 then re-tags the DSML examples (`agent_dsml41_tools_prompt`, `ds4_agent.c:4652`). GLM and Qwen share one schema list with their own wrappers (`ds4_agent.c:1310–1435`).

A renderer working from tool definitions cannot reproduce that prose. So:

- `ds4py.prompts` holds vendored copies of the C strings and builders with the same switches (syntax, edit variant, vision).
- `DS4AgentTools.get_toolset()` returns a `FunctionToolset` with the tools whose names and schemas match the C ones. The harness then handles dispatch, approvals, events and records as for any tool.
- Each of those tools carries a marker in `ToolDefinition.metadata`, set through `SetMetadataToolset`. When `DS4Model` sees marked tools, it uses the vendored prompt.
- Extra tools, such as an extension's tools or MCP tools, have their schemas inserted where C inserts `view_image`.
- Tests need no GPU. A small C program that includes `ds4_agent.c`, as `tests/ds4_agent_test.c` already does, prints every prompt variant. The builders take only the edit variant and vision, not an engine. The Python builders must match its output byte for byte, and the fixtures catch upstream prompt changes whenever the pin moves.

### b. The implementations produce the same bytes

The model reads the tools' output, so ds4-py's tools must produce the C agent's bytes. The M3 plan lists each tool's argument handling and output format from `ds4_agent.c` (file tools at 6599–7049, edit and search at 7049–8554, web at 8554–8700, `bash` jobs at 8700–9360, dispatch at 9360–9675). Known examples:

- `list` prints `"%c %10lld %s%s\n"` for each entry and stops at 300 entries.
- Errors start with `Tool error: `.
- Long output ends with `[Output truncated at the tool byte limit. Narrow the request.]`.
- `bash` reports `bash job=N pid=P status=running elapsed_sec=E timeout_sec=T`, or `status=done … timed_out=0|1` followed by `exit_status=`. The first observation shows the output inline when it is short and complete; otherwise it shows `output_path=…` and a `<head -N path>` excerpt. Later observations show a `<tail -N path>` excerpt. A running job ends with the `bash_status` and `bash_stop` hint. `timeout_sec` defaults to 3600 and `refresh_sec` to 60.
- `bash` jobs keep running, and keep their timeouts, between `bash_status` calls, and those calls can come in later prompts. So the job table is session state (section 13), not part of the capability instance, and the session stops the jobs when it ends.

Argument checks must be no stricter than C's. C clamps values (for example, `refresh_sec` to 0–3600) where pydantic would raise a validation error. So the functions accept loose arguments and clamp the way C does.

The web tools are a port of `ds4_web.c`. They find Chrome through `DS4_CHROME` or the standard install paths, start it with its own profile directory and a DevTools port of 9333, and talk to it over a WebSocket with a 20-second timeout. Results are capped at 1 MiB. The first start asks `Start visible Chrome browser? (y/n)`. In non-interactive mode the tool fails with `visible Chrome browser startup requires interactive approval` (`ds4_agent.c:5171`). The port uses only the standard library, including a small WebSocket client.

Golden tests come from the same C test program: it runs each tool that needs no worker state on fixture directories and prints the output. Tools that need worker state (`bash` jobs, web) are checked against recorded output from live `ds4-agent` runs. Comparisons normalize process ids, elapsed times and temporary paths.

### c. Result framing lives in `DS4Model`

All results of a round go into one `tool` message, each starting `Tool result N (name):\n`. The harness's own errors arrive as `RetryPromptPart`s and are rendered as the C text (section 11).

## 13. Context strategies: the extension point

Context management is a plug-in point, not a fixed recipe. A strategy is a capability. The binding makes any history change correct, and `session.plan()` gives its exact cost before the move (section 10). The cost follows the engine's reuse rule:

- **Appending** to the view costs only the new tokens.
- **Any other change** restores the longest held checkpoint that is a prefix of the new view and prefills the rest. If no checkpoint fits, it rebuilds from token 0.

The facade always holds the `system` checkpoint, so on a backend that passed the restore check a rewrite costs at most the view minus the system prefix. A rewrite never costs only the tokens after the first change: the engine cannot keep KV past a change without a checkpoint at or before it. Strategies differ in *when* they move the front of the window, *what* the model can get back, and which checkpoints they hold.

The contract a strategy implements:

| Hook or API | Use |
| --- | --- |
| `before_model_request(ctx, request_context)` | Set `request_context.messages` for a **view-only** change, which leaves the stored history alone (paging, rolling windows). Set `ctx.messages[:]` for a **persistent** change (compaction, bookmarks). The view's prefix must stay the same between moves so the KV is reused on every other request |
| `get_toolset()` | Recovery tools that read the **stored** history and return exact text: search, browse, load and read-page, or `expand_bookmark`. They must be registered from the start of the run, because a tool-set change costs a full prefill |
| `session.state(key, factory)` | Session-owned state: the cut point, the bookmark table, token accounting. The harness calls `for_run` once per `Agent.run`, which is once per prompt, so a strategy keeps none of this on `self`. Its `for_run` returns an instance that refers to this state (see "Session state" below) |
| `on_event` | Turn-end bookkeeping. Stamp stable keys into `message.metadata['ds4.key']` so cut points are keyed by message, not by list index |
| `session.plan(view) -> Plan(kind, reused, to_prefill, fits, checkpoint)` | Renders a candidate view without syncing and says whether it extends the live KV, restores a held checkpoint or rebuilds, with the exact prefill cost (section 10) |
| `session.checkpoint(key)`, `session.drop_checkpoint(key)` | Hold or free a snapshot of the live state at its current end. A strategy holds one where it knows later moves will start, such as a turn boundary that its next batch of stubs will not cross. Each costs host memory, and the facade limits how many are held |
| `session.compact(...)` | Only `DS4Compaction` uses the engine-side summary job. Other strategies never call the model themselves |
| `make_room(ctx, needed_tokens) -> list[ModelMessage] \| None` | Called on the strategy that owns the context trigger when a tool result does not fit, or when generation reaches the reserve inside a response (section 11). The strategy frees space in its own way and returns the new request view, or `None` if it cannot. A persistent strategy also rewrites `ctx.messages[:]` |

Exactly one strategy owns the trigger in a session. A strategy declares `owns_context_trigger = True` and records itself as the owner in session state. `DS4Compaction` stands down when another owner is present, and `DS4Model` finds the owner there while a stream is open. Manual `/compact` still works.

| Strategy | Kind | What moves, and when | Cost per move | What the model can recover |
| --- | --- | --- | --- | --- |
| `DS4Compaction` (default; the C recipe) | persistent | At 85% of the context, the history becomes `system + user(summary) + verbatim tail`; an open assistant message is resumed | the compaction prompt (an extension), the summary (at most 4096 tokens, generated with `argmax`), then the summary message and the tail after restoring `system`. C rebuilds the system prefix here too | the summary and the tail |
| `RollingWindow` (a port of `pi-context-paging`) | view-only | When the rendered view exceeds `token_budget`, drop whole oldest turns until it is at most `trim_to` (default 5/8 of the budget). The cut stays until the budget is exceeded again. A paging notice sits right after the system prefix. The active request and unread tool results are never cut | the notice and the window after restoring `system`, once per cut; nothing between cuts | everything, exactly, through `search_history`, `browse_history`, `load_history` and `read_context_output`. Thinking stays private |
| `Bookmarks` | persistent | Replace chosen messages (large tool results, a finished sub-task) with a one-line stub such as `[bookmark b17: bash output, 12 KiB, 3 errors; expand_bookmark("b17")]`, in batches at turn boundaries | everything after the latest held checkpoint before the first replaced message; with only `system`, the whole view after the system prefix | the exact original, through `expand_bookmark` |
| Harness `ClearToolResults`, `SlidingWindowCompaction`, `SummarizingCompaction`, `TieredCompaction` | persistent | as the harness documents | the same rule: the whole view after the system prefix, each time an old message changes. `ClearToolResults` changes old messages most often, so it is the most expensive | per strategy |

`DS4Compaction` in detail: a `before_model_request` hook with the C agent's trigger, `ctx_used >= 0.85 * ctx` or `ctx - used <= max(min(8192, ctx/8), reserve + 128)`. `reserve` is the rendered length of the compaction prompt, plus an assistant prefix, plus `clamp(ctx/8, 256, 4096)` for the summary. When it fires, it calls `session.compact(reason)`, which runs the whole recipe on the worker in one pass:

1. Sync `transcript[:bottom] + user(prompt) + assistant_prefix(no-think)`.
2. Generate the summary with `argmax` until a stop token or the DSML special token.
3. Choose the tail and resync `system + user(summary) + tail`.

Then the hook rewrites `ctx.messages[:]` to the system parts, a `UserPromptPart(summary)` and the tail's messages, and rebinds. Step 3's resync restores the `system` checkpoint and prefills the summary message and the tail, where C rebuilds from token 0.

The hook covers compaction between requests only. When generation reaches the reserve in the middle of a response, the open stream handles it (section 11): `DS4Model` calls `make_room`, and `DS4Compaction` runs the same recipe with the open assistant message kept. The partial response's tokens follow the tail, and generation continues in the same `ModelResponse`, as the C agent does (`ds4_agent.c:9675–10229`).

The check before a tool result is added lives in `after_tool_execute` and is shared by every strategy. If the result does not fit, the check calls `make_room` on the strategy that owns the context trigger. The mid-response reserve uses the same call:

- `DS4Compaction` compacts, as C does.
- `RollingWindow` moves its cut forward by whole turns. It never summarises.
- `Bookmarks` replaces more messages with stubs.

If the result still does not fit, it is replaced with `Tool error: tool result still does not fit`. If the reserve still cannot be met inside a response, the stream raises `ContextExhausted`.

`DS4SystemNotes` owns the 50,000-token reminder. Its trigger is a predicate on exact transcript growth, which fits `Reminder(trigger=…)` in the harness's `SystemReminders`. `DS4Model` applies the C wrapping and the tokenizer rule when it sees the `reminder` note marker. If `SystemReminders` cannot carry that marker, `DS4SystemNotes` implements the trigger itself (builder's call, M4).

`RollingWindow` gets two things from the engine that the Pi version has to estimate: exact token counts for the budget, and an exact `ctx_used`. Its design choices carry over unchanged: whole-turn cuts, stable keys, protected content, the notice counted inside the budget, no summarising, and private assistant content excluded from recovery. `ds4_session_create(ctx)` can be given the budget directly so the engine reserves only the KV the window needs.

### Session state

The harness calls `for_run` once per `Agent.run`, and by default it shares one capability instance across runs (`capabilities/abstract.py:447–461`). The TUI starts one run per prompt, so one ds4 session spans many runs. State that must last for the session cannot live in a per-run copy, and a shared instance has no hook for `/new` or `/switch`. The same docstring says not to acquire resources such as processes in `for_run`, and the harness skips `after_run` when a run produces no result, so neither can own `bash` jobs.

ds4-py keeps this state in the session:

- `session.state(key, factory)` returns the state object for `key` in the live ds4 session, and creates it with `factory()` on first use.
- A capability keeps no session state on `self`. Its `for_run` returns an instance that refers to `session.state(...)`. So the paging cut, the bookmark table, the reminder's growth counter and the `bash` job table last from one prompt to the next.
- The session drives the state's lifecycle through the event bus (section 22). `/new`, `/switch` and exit all send `session_end`. On it, the session calls `close()` on every state object and drops them all, so `bash` jobs stop and nothing leaks into the next session. On `forked`, the state is kept, because a fork keeps the transcript and the message keys. On `compacted`, a strategy that does not own the trigger resets its own state.

## 14. Policy: `DS4Approvals`

`before_tool_execute` applies a small rule table:

| Tool | Rule |
| --- | --- |
| `bash` | Ask, unless `--yolo` |
| `write`, `edit` | Ask when the path is outside the project directory, unless `--yolo` |
| `google_search`, `visit_page` | The C agent's one-time Chrome consent. Interactive mode asks; non-interactive mode refuses with the C text |
| any write or `bash` with `--read-only` | Deny |

A denied call returns `Tool error:` text, so the model's next round reads as it would in C, and the tool result carries `outcome='denied'` for the records.

`ds4-agent` itself asks only for the Chrome consent. The other prompts come from the first draft and are ds4-py additions. An approved call leaves the transcript as C would have it. A denied call adds text C would never produce, so runs with a denial are left out of parity checks.

## 15. History, save records and run records

ds4-py keeps three records. Only the first two are used to recover a session.

| Record | Where | What it holds | Used to recover? |
| --- | --- | --- | --- |
| Message history | `StepPersistence(FileStepStore(<session dir>/steps))` | `ContinuableSnapshot`s: the full harness message list, with every part and `metadata['ds4.key']` | yes; it is the canonical history |
| Save record | `~/.ds4py/saves/<sha40>.json`, one per KV file, rewritten with each save | the snapshot the KV file was saved from, and the token provenance of the transcript (below) | yes |
| Run record | `<session dir>/events.jsonl`, written by `DS4RunRecord` | one line per `AgentStreamEvent`, plus engine events | no; telemetry only |

The session directory is `~/.ds4py/sessions/<session id>/`. The session id is kept apart from the KV file name (section 22).

### Message history

The TUI starts one `Agent.run` per prompt. It passes `conversation_id=session.id` and the previous messages as `message_history`. `StepPersistence` saves a snapshot at each settled tool boundary and at the end of the run, and it saves the history at the point of failure if the run fails (`step_persistence/_capability.py:50–62`). Each run has its own `run_id`, so `list_runs(conversation_id=session.id)` lists the session's runs in order.

A snapshot holds what the run event stream does not:

- the user prompt passed to `agent.run`, and the system and instruction parts;
- message metadata, including the stable keys;
- every rewrite of `ctx.messages[:]` by a persistent strategy, such as compaction or a bookmark batch.

Older runs keep their snapshots, so the history before a compaction stays readable. A `RollingWindow` session's KV file holds only the request view, so the messages outside the view exist only here.

### Save record

The save record ties a KV file to the history it was saved from. `ds4py.store` writes it with each `/save`, after the KV file (section 17). It holds:

- `session_id`, and the `run_id` and `step_index` of the save's snapshot. `/save` writes the live message list as a `complete` snapshot in a run of its own, `<session id>.save-<n>`, under the session's `conversation_id`. So every KV file has an exact snapshot, even when the last run ended in an interrupt or an error.
- `tokens` and `tokens_sha256`: the length and hash of the saved token list.
- `segments`: the token provenance. Each entry is `(start, end, kind, key, content_hash, source)`, where `key` is the message's `ds4.key`, or empty for the system prefix and notes, and `source` is `sampled` for tokens `DS4Model` generated or `rendered` for tokens `PromptBuilder` produced.

In memory, the binding keys a message by its identity and a content hash (section 11). A restored message is a new object, so on restore the binding keys it by `ds4.key` and the content hash instead.

### Run record

`DS4RunRecord` is an `on_event` capability. It appends one JSON line per `AgentStreamEvent` to `<session dir>/events.jsonl`. It also writes the engine events the harness does not see (`Prefill`, `Synced` with its plan, `Compacted`, `Saved`, `Interrupted`), forwarded from the facade's event queue. The record is for reading and measuring a session, such as bake-off gate 3. Nothing is rebuilt from it: it does not hold the user prompt, the system parts, history rewrites or token ids.

## 16. Interface

The TUI is ds4-py's own, written fresh with `prompt_toolkit`. [`clai`](https://pydantic.dev/docs/ai/integrations/cli/), the harness's terminal chat, is used for one day in M3 as a smoke test of the harness stack and then set aside. It has a fixed command set, no footer, no notification hook and no plugin API, so it cannot show the prefill bar or host `/save`, `/think` or an extension's commands.

Most of the work is not UI. `agent.run_stream_events()` supplies the token, tool-call and tool-result events, and the facade supplies prefill progress and engine events. The TUI (`tui.py`, about 600 lines) is an input loop, a renderer for those two streams, a footer, and two registries that exist from the first commit:

- `register_command(name, handler, help)`. Every built-in command (`/save`, `/list`, `/switch`, `/del`, `/strip`, `/new`, `/compact`, `/think`, `/steer`, `/power`, `/hints`, `/history`) registers through this call, so an extension's command is not a special case.
- `set_status(key, text)` and `notify(text, level)`. The footer shows `ctx used/total`, the prefill bar (`▶`/`·`, 32 cells, with a tokens-per-second overlay), generation speed, the ❄️ marker while sampling is greedy, and any key an extension sets.

Capabilities reach these through the session event bus (section 22), never by importing the TUI. The TUI thread never calls the engine; it submits jobs and reads the event queue.

Non-interactive mode (`--non-interactive` with `--prompt` or `--prompt-file`) runs one prompt to completion and exits. The bake-off script uses it.

## 17. Session persistence and KV checkpoints

Sessions are saved in the C agent's file format, so `ds4-agent` and ds4-py can open each other's files. The header and trailer are written from Python with `struct`, using the constants and the text renderer from `ds4_kvstore.h`. Only the KV payload bytes are produced and read by the engine.

### File layout: `~/.ds4/kvcache/<sha40>.kv`

| Offset | Field | Source |
| --- | --- | --- |
| 0–2 | magic `KVC` | Python |
| 3 | version `1` | Python |
| 4 | `quant_bits` (2, 4, 5, 6 or 8) | `engine_info.routed_quant_bits` |
| 5 | `reason` (`5` agent-system, `6` agent-session) | `ds4_kvstore.h` |
| 6 | `ext_flags` (bit 3 = session-title trailer present) | `ds4_kvstore.h` |
| 7 | `model_id` | `engine_info.model_id` |
| 8–11 | `u32 tokens` | `len(transcript)` |
| 12–15 | `u32 hits` | 0 |
| 16–19 | `u32 ctx_size` | the session's context size |
| 20 | payload ABI `2` | Python |
| 24–31 | `u64 created_at` | the first user turn |
| 32–39 | `u64 last_used` | the save time |
| 40–47 | `u64 payload_bytes` | `payload_bytes`, or 0 for a stripped file |
| 48– | `u32 text_bytes` + the rendered transcript text | `render_tokens_text` |
| … | KV payload | `write_staged(payload, f)` |
| … | `u32 title_len` + title | Python |

The file name is `sha1(title_bytes + created_at.to_bytes(8, 'little'))`, where the title is the normalized first user prompt. The M4 plan confirms this against the C code at `ds4_agent.c:4709`. Files are written to `path.tmp.XXXXXX` and renamed, as in C.

A session restored with `/switch` keeps its `created_at`, so saving it again overwrites its own file, as in C. A fork is different: the live session takes a new identity and continues from the same transcript, while the original stays in its saved file. `session.fork()` keeps the transcript and the live KV and changes only the identity, so a fork needs no prefill and no second KV cache. A fork keeps the title but takes the fork time as its `created_at`, so it gets its own file under the same naming formula. It also gets a new session id; existing message keys are kept. ds4-py has no `/fork` command in M1 to M4. Forks come from `session.fork()`, which extensions and the conformance test use.

### Save

`/save` is a `SaveJob` on the worker. Before staging, the worker syncs the transcript so the live KV equals it. This evaluates the trailing EOS, which the C agent also leaves unevaluated until the next sync. Then it calls `stage_payload`, writes the header and text, streams the staged payload into the file, and appends the title trailer. After the KV file is renamed into place, `/save` writes the session's snapshot through the step store, then the save record (section 15) through a rename. The record comes last, so it never names a snapshot that was not written. Sessions with images are refused with the C agent's message.

### Restore

`/switch <sha-prefix>` reads the header and checks `model_id` and `quant_bits` against the open engine. If `payload_bytes > 0`, the worker calls `load_payload` and **takes the token list from the session**, not from re-tokenizing the text. So the restored transcript is identical to what was saved, and nothing is prefilled. If the file is stripped and has no ds4-py save record, the text is re-tokenized with the rendered tokenizer and synced, which prefills from scratch, as in C.

The history and the binding come from the save record (section 15), not from `events.jsonl`:

1. **A ds4-py save.** The record exists, and its `tokens` and `tokens_sha256` match the loaded tokens. ds4-py loads the snapshot the record names and rebuilds the binding from `segments`, matching each segment to a snapshot message by `ds4.key` and content hash. Nothing is rendered or prefilled. If the snapshot is missing, or a segment matches no message, `/switch` fails with an error that names the snapshot or the key.
2. **A file saved again without its record.** The record exists, but the token hash does not match. Usually `ds4-agent` saved the file last; a crash between the two writes of a `/save` has the same result. ds4-py says so and treats the file as a C-agent file.
3. **A C-agent file.** There is no record. The restored tokens are scanned for the role special tokens. Each segment becomes a message bound to its tokens, under a new session id.
4. **A stripped ds4-py save.** The file has no payload, so the token hash cannot match. The history comes from the record's snapshot, and `DS4Model` renders it fresh. This costs the same full prefill as any stripped file.

Then the session sends `switched` with the save record (section 22). `RollingWindow` sets its cut at the first message bound in the transcript, so the first request after `/switch` extends the loaded KV.

Two cases deliberately behave as in C: a tensor-parallel session ignores the payload and rebuilds from text, and a mismatched `model_id` is an error, not a warning.

### `sysprompt.kv`

The rendered system prefix (think prefix, tools prompt, extra system text) is saved once as `sysprompt.kv` with reason `agent-system`, keyed by its rendered bytes. At start-up it is loaded if the bytes match, so a new session begins with the system prompt already in the KV. The facade then saves its `system` checkpoint, which later rewrites restore (section 10). The C agent refuses a cached file whose text does not match (`ds4_agent.c:4937`).

### Strip, delete and list

`/strip` rewrites the file with `payload_bytes = 0` and a new token count, and keeps the save record. `/del` deletes the file and its save record. It keeps the session's history in `~/.ds4py/sessions/`. `/list` reads only headers and trailers, sorts by `last_used`, and shows only files for the open engine's `model_id`.

## 18. The downstream agent: what ds4-py provides

The downstream agent will be a customisation of ds4-py, built against the downstream fork. ds4-py provides:

| Need | What ds4-py provides | Milestone |
| --- | --- | --- |
| A hook in the token loop | `session.add_logits_hook` (section 10) | M2 |
| Build `libds4` from any ds4 tree | `ds4py build --src <tree>`, reusing the tree's own object lists | M1 |
| Load a library from a given path | `ds4py.open(lib=…)`, `DS4_LIB`, `ds4py check --lib` | M1 |
| Required and optional bindings | a `required` flag per binding; `ds4py check` lists the optional ones | M1, flag only |
| The ABI guard for each build | signature and layout probes per build; an unknown header hash warns | M1 |
| Its own bindings | the `ds4py.bindings` entry point, `engine.native.bind`, `session.on_worker`, the sampling-defaults override | M1 |
| A pluggable decode step | the decode-step contract (section 10), proved with `FakeMultiTokenStep` | M2 |

Left out, by the user's decision: engine work between runs, a second front end, and records in the downstream agent's own schema.

What the comparison of the two trees found:

- The downstream fork has 16 functions that upstream lacks: the DFlash set (`ds4_engine_has_dflash`, `ds4_session_dflash_*`, `ds4_session_verify*`), `ds4_engine_sampling_defaults` and `ds4_session_supports_payload`.
- Upstream has 4 functions that the fork lacks, including `ds4_session_rewind_speculative` and `ds4_session_restore_speculative_prefix`. The fork lags upstream.
- `ds4_engine_options` has the same 44 fields in both. `ds4_kvstore.h` is identical in both.
- The fork's ROCm object list has no `ds4_engram.o`. Copying upstream's list would break the fork's ROCm build, which is why `Makefile.lib` reuses each tree's own `$(CORE_OBJS)`.
- The fork passes the DFlash drafter through the existing `mtp_path` and `mtp_draft_tokens` options, so opening an engine with it needs nothing new.
- The fork's header describes its DFlash as "dense qwen35 on Metal". A CUDA bake-off machine cannot run it, so without a Mac, a DFlash step can only be tested on `FakeNative`.

## 19. Build and packaging

- ds4-py is a `uv` project for Python 3.11 or later. The wheel has no compiled extension.
- `ds4py build --src <ds4 tree> [--backend metal|cuda|rocm|cpu]` builds `libds4` and its ABI probe:
  - `Makefile.lib`, kept in this repo, includes the tree's own Makefile and links the tree's `$(CORE_OBJS)` (or `$(CPU_CORE_OBJS)` for the CPU build) plus `ds4_kvstore.o` into `libds4.so` or `libds4.dylib` with `-shared`.
  - Objects are compiled with `-fPIC` (on CUDA, `-Xcompiler -fPIC` for `nvcc`), and the link uses `$(DS4_LINK)` and the same `METAL_LDLIBS` or `CUDA_LDLIBS` as the binaries. Whether objects are built inside the tree or in a separate directory is the builder's call; PIC objects also link into the tree's own binaries.
  - Then `ds4py build` generates the probe files from the binding list (including extension bindings), compiles them with the tree's compiler, runs the layout program, and writes `libds4.abi.json`.
- `ds4.c` already exports every function in `ds4.h` with default visibility, so nothing in the engine changes.
- For Metal, the loader passes the ds4 checkout path and changes to it, or sets the equivalent, before `ds4_engine_open`.
- The pins live in one module: the upstream commit and the known header hashes, and the exact harness version in `pyproject.toml`.
- Offering the library target upstream as a pull request to ds4 is worth trying. It does not touch inference code.

## 20. Testing

| Layer | How | Needs a model? |
| --- | --- | --- |
| ABI guard | CI builds the CPU `libds4` from the upstream pin and from the downstream fork's pin and runs `ds4py check` on both. A doctored header that changes one argument type of a required function must fail the build; one that moves a field of `ds4_engine_options` must make loading refuse; so must a manifest whose recorded prototype differs from an installed binding's; an unknown header hash must only warn | no |
| `_native` | load the library and check it; open a small CPU session if `DS4_TEST_MODEL` is set | optional |
| `Transcript`, `PromptBuilder`, `EngineWorker`, decode steps, logits hook | `FakeNative` | no |
| Prompts | the C prompt printer's output for every variant (4 syntaxes × 2 edit variants × vision) must equal the Python builders' output byte for byte. Regenerated on every pin move | no |
| `ToolSyntax` parsers | fixtures from the C prompts and from `ds4-agent --trace` captures, plus every fixture split at every byte boundary | no |
| Tools | golden output from the C test program, and recorded output for `bash` and web, after normalizing process ids, times and temporary paths | no |
| `DS4Model` | feed `ModelRequest` lists; check the rendered operations and the `ModelResponse` parts | no |
| Capabilities | `FunctionModel` stands in for `DS4Model` | no |
| Extension points | the conformance test (section 22) | no |
| Raw parity | `ds4py raw` and `ds4-agent --raw` on the same prompt, `--temp 0 --seed 1`, same model: identical tokens | yes |
| Replay parity | below | yes |
| Sync plans and checkpoints | on each backend, for an append, a checkpoint restore and a rebuild, the prefill count in `Synced` equals `plan().to_prefill`. Then the restore check: after a restore plus extension, 256 tokens at `--temp 0` must equal those after a rebuild of the same tokens. ds4-py disables `restore` on a backend that fails the check | yes |
| Integration | the bake-off (section 21), the KV round trip, and a 150k-token session | yes |

### Parity with `ds4-agent`: replay, not side by side

The first draft planned to run both agents on the same prompt and compare the transcripts token by token. That cannot work past the first tool call. `bash` results contain process ids, elapsed times and `mkstemp` paths, and every session starts with a date note, so the two transcripts differ even when both agents behave correctly.

Replay parity tests the same thing without that problem:

1. Run `ds4-agent` on a task and take its exact token transcript from its saved KV file. Loading the payload gives the exact token ids, not a re-tokenization.
2. Split the transcript into rounds at the assistant prefixes.
3. For each round, sync ds4-py on the C agent's prefix up to that round's assistant prefix and generate at `--temp 0`.
4. Compare ds4-py's tokens with the C agent's tokens for that round. They must be identical.
5. Compare the tool results separately with the C agent's, after normalizing process ids, times and temporary paths.

This proves that the token loop, the EOS rule, the greedy switch inside tool syntax and the result framing are identical to C. That is what makes KV files portable between the two agents.

The M3 plan must find out how the script gets `ds4-agent` to save its session in non-interactive mode. If it cannot, the script drives `ds4-agent` interactively through a pseudo-terminal and sends `/save`.

## 21. M3 exit test: the 12-task bake-off

M3 is done when ds4-py matches `ds4-agent` on the same 12 tasks, with the same model (DeepSeek V4 Flash Q2, the `ds4f-q2` download, about 81 GB; section 25 explains why not V4.1), on the same machine, with `--temp 0 --seed 1`. A script drives both agents in non-interactive mode, unattended, and grades with hidden tests. ds4-py runs with `--yolo`, except for the approval run in gate 4.

| Tasks | Source | Grading |
| --- | --- | --- |
| 10 Python exercises | the [Aider polyglot benchmark](https://aider.chat/2024/12/21/polyglot.html) (Exercism, 225 exercises in 6 languages). The 10 Python exercises are chosen once with a fixed seed and pinned in `bench/m3.yaml` | the exercise's hidden tests, run by the script after the agent says it is done. The agent gets the instructions file and up to 2 attempts, as the benchmark allows |
| 2 repo tasks on ds4-py itself | a planted bug (`tests/test_toolsyntax_glm.py` fails because of an off-by-one in the `<arg_value>` close handling) and a feature ("add a `/ctx` slash command that prints `ctx used/total`", which exercises the command registry) | the test suite passes; the feature's own test passes |

Gates, all required:

1. **Pass rate** at least `ds4-agent`'s on the same run, with one task of slack. Gate 6 proves the sampling paths match, so a larger gap is a bug.
2. **No parser failures.** Zero `parser_error` stops and zero `Tool error: invalid …` results across the 12 runs. Every tool call the model made was executed or refused by policy.
3. **Speed and KV reuse.** Wall time at most 1.5× `ds4-agent`'s per task. In every round, ds4-py prefills only the tokens appended since the last sync; the run record shows an `extend` plan for each round, except rounds after compaction or a strategy move. This replaces the first draft's "prefilled tokens per round equal within 1% of `ds4-agent`", which cannot hold once the two runs see different process ids and timings.
4. **Interrupt and approval.** One run is interrupted with Ctrl+C during prefill and resumed from the next prompt. One run triggers an approval prompt for `bash` and continues after `y`.
5. **Records and files.** Every run leaves a complete `events.jsonl`, the session's snapshots, and a saved KV file with its save record that `ds4-agent` can `/switch` into.
6. **Replay parity.** For every round of `ds4-agent`'s 12 runs, ds4-py produces the same tokens from the same prefix (section 20).

Not tested here, on purpose: long-context behaviour (M4 adds a 150k-token session and the compaction and `RollingWindow` cases), multi-file repository tasks (M5, a 10-task SWE-bench Verified slice), and GLM, Qwen and DeepSeek V4.1 (M5).

## 22. Extension points reserved now

Full plugin support is not an M1 feature, but some things are cheap to build in now and expensive to retrofit. The list comes from the API surface `pi-context-paging` uses, plus the downstream agent's needs. A conformance test keeps them honest.

| Reserved now | Why it is structural |
| --- | --- |
| The stored history differs from the request view. `DS4Model` binds to `request_context.messages` only and never writes `ctx.messages`. `StepPersistence` snapshots store the full harness message list with all parts and metadata, not just tokens; `events.jsonl` is telemetry only | A strategy that pages or bookmarks needs the complete history, with thinking kept apart from public text |
| Stable message keys in `message.metadata['ds4.key']`, set at creation. The session id is kept apart from the KV file name | Cut points, bookmarks and forks all key on messages, and two forks of one title must not share a file name |
| A session event bus in `ds4py` (`session_start`, `session_end`, `turn_end`, `before_compact`, `compacted`, `model_changed`, `switched`, `forked`), separate from the harness's per-run `on_event`. It drives the lifecycle of `session.state` (section 13) | Lifecycle events the harness does not model; adding them later means finding every call site |
| `session.plan()`, `session.ctx_used`, `session.system_tokens` and `session.active_tools` as public calls | Strategies need exact cost and usage, and the facade must be designed so these are cheap |
| A command registry and a status-key registry in the TUI, open to capabilities; project settings with a trust flag | Without them, every extension needs a fork of the UI |
| The extension API: declared bindings, `engine.native.bind`, `session.on_worker` | The downstream agent's bindings must pass the same ABI guard as ds4-py's |
| The decode-step contract and the logits hook | Retrofitting a step interface into a fixed loop is the expensive change; the downstream agent's DFlash depends on it |

The conformance test runs in CI on `FakeNative`. A stub extension must:

1. rewrite the request view and see the binding re-render only from the cut;
2. register tools that read paged-out history with thinking excluded;
3. own the context trigger and see `DS4Compaction` stand down;
4. get `plan()` kinds and costs that match what the next sync does, for an append, a checkpoint restore and a rebuild;
5. survive `/switch` and a fork with its keys intact, with the history rebuilt from the save's snapshot and the binding from its save record, and no prefill;
6. add a slash command and a status key;
7. declare a binding, see it in the probe and call it through `on_worker`;
8. install a logits hook that changes a sampled token, and run a multi-token decode step that stops mid-block with the KV equal to the kept tokens;
9. keep state in `session.state` across two prompts, and see it closed on `/new` and `/switch` and kept on a fork.

The first real strategy is the `RollingWindow` port in M4, and it is the acceptance test for the context half of this API.

## 23. Milestones

Every milestone is developed against `FakeNative`. Criteria marked *live* run on the 96 GB machine once it exists; they do not block the next milestone from starting. M1 to M4 come to about 10 to 11 weeks.

### M1: native layer and raw loop (2 to 2½ weeks)

Scope: `Makefile.lib` and `ds4py build`; the binding list and the type map; the ABI probes; the loader with required and optional bindings; `ds4py check`; the extension API; `EngineWorker` with sync, sample, eval, rewind, progress and cancel; checkpoints and sync plans; `PlainStep`; `ds4py raw`; `FakeNative`.

Done when:

- CI builds a CPU `libds4` and `libds4.abi.json` from the upstream pin and from the downstream fork's pin, and `ds4py check` passes for both, reporting the fork's header hash as unknown.
- The doctored-header tests in section 20 behave as described.
- An extension binding declared through the entry point is probed, listed by `ds4py check` and callable through `on_worker` on `FakeNative`.
- On `FakeNative`: prefill progress events arrive at most every 100 ms; Ctrl+C during sync returns `Interrupted` with the transcript intact; generation stops on a stop token without evaluating it.
- On `FakeNative`: `plan()` reports `extend`, `restore` and `rebuild` as section 10 defines them, and each `Synced` reports the plan and prefill count that `plan()` predicted.
- *Live:* `ds4py raw "<prompt>" --temp 0 --seed 1` produces the same tokens as `ds4-agent --raw`.
- *Live:* the sync-plan and restore checks in section 20 pass on the bake-off machine, or `restore` is disabled there.

### M2: chat, tool syntax, decode steps and the hook (2½ weeks)

Scope: `Transcript`; `PromptBuilder`; `ds4py.prompts` and the C prompt printer; the three parsers; `run_round` with the tool-call stop; the decode-step contract and `FakeMultiTokenStep`; the logits hook; `--prefix-file`; `ds4py.store` reading and writing the file format; the `read` and `bash` tools; a test CLI of about 200 lines that runs `read` and `bash` directly, before the harness exists.

Done when:

- The Python prompt builders match the C printer for every variant.
- The parsers pass every fixture and every byte split, and markup inside `<think>` is reported as a parser error.
- On `FakeNative`: a multi-token step that meets a stop token mid-block ends with the KV at the stop; a tool block that closes mid-block keeps the closing token and drops the rest; Ctrl+C mid-step leaves the KV equal to the kept tokens.
- On `FakeNative`: a hook that biases a token changes the sample, and while a hook is installed the loop does not use a step that skips the hook.
- `read` and `bash` match their golden output.
- *Live:* replay parity on recorded `ds4-agent` sessions with the bake-off model.

### M3: harness, the C agent's tools, TUI and bake-off (3 to 3½ weeks)

Scope: `DS4Model` and the binding; `DS4AgentTools` with every tool, including the web port; `DS4Approvals`; `DS4SystemNotes` for the date note and hints; `DS4RunRecord`; `StepPersistence`; `/save` with its snapshot and save record; the session event bus and `session.state`; the one-day `clai` smoke test; the TUI with its registries; non-interactive mode; the harness-tools option; the bake-off script.

Done when:

- Every tool matches its golden output.
- `DS4Model` passes its request-list tests, including retry and unknown-tool framing.
- Conformance items 1, 6, 7 and 9 pass.
- On `FakeNative`: a `bash` job started in one prompt is reported by `bash_status` in the next and stops on `/new`; a reserve stop inside a response with no context owner raises `ContextExhausted` and does not return partial text.
- *Live:* all six bake-off gates pass.

### M4: context and persistence (2 weeks)

Scope: `DS4Compaction`, including the reserve stop inside an open stream; the 50,000-token reminder; `sysprompt.kv`; `/list`, `/switch`, `/strip` and `/del`, with history recovery from save records; the `RollingWindow` port; the full conformance test.

Done when:

- On `FakeNative`: the compaction trigger, recipe and tail cut match C. A reserve stop inside a streamed response compacts and continues in the same `ModelResponse`, as C resumes an open assistant message. The reminder fires after 50,000 tokens of growth.
- On `FakeNative`: `/switch` handles the four cases in section 17, and a missing snapshot fails with an error that names it.
- The full conformance test passes.
- *Live:* a 150k-token session compacts and continues; a session saved in ds4-py opens in `ds4-agent` and back again with no prefill; `RollingWindow` runs a long session within its budget.

### M5: polish (ongoing)

`SpeculativeStep`, vision through `view_image`, GLM, Qwen and DeepSeek V4.1 replay parity and bake-off runs on hardware that can hold each model, `/think`, `/steer` and `/power`, ACP for editors, MCP servers, and a 10-task SWE-bench Verified slice.

## 24. Risks

- **`ds4.h` churn.** The probes turn a silent mismatch into a refused library, but every pin move still needs a review of the bindings. Keep the wrapped surface small.
- **Porting the tools exactly.** This is the largest single piece of new work. The golden tests are the defence; the web tools are the hardest to test because they need Chrome.
- **Subagents later.** ds4 allows one ds4 process per machine (an exclusive lock on `/tmp/ds4.lock`), so a subagent cannot be a second process. One engine can hold several sessions, as `ds4-server --batched-session N` does, but each reserves its own KV cache when the engine opens, and forward passes run one at a time. A subagent that shares the main session either sees the parent's context or needs the parent's KV saved and loaded back around it.
- **GIL and callbacks.** Progress callbacks re-acquire the GIL about 10 times a second during prefill, which is negligible. The real risk is a Python exception inside a callback, which ctypes swallows. The facade stores and re-raises it, and the callbacks stay trivial.
- **Checkpoint memory.** Each checkpoint holds a full session snapshot in host memory, and its size grows with the prefix. The limit of four plus `system` bounds the total, `Checkpointed` reports each one's bytes, and the facade never evicts one on its own. A strategy that holds no extra checkpoints needs no extra memory, but each of its rewrites prefills the whole view after the system prefix.
- **Compaction fidelity.** The C recipe cuts the tail at a `<｜User｜>` boundary and resumes an open assistant message after compaction. Both are reproduced. The resume case is the hardest to test; it gets a `FakeNative` scenario and one live run.
- **Harness churn.** `pydantic-ai-harness` is 0.x and releases often. The adapter touches only `Model`, `StreamedResponse`, capability hooks and a few named capability classes. Each bump is a deliberate change with the full suite.
- **Memory on the bake-off machine.** The README's "96 GB or more for DeepSeek" refers to Mac unified memory. Check that the bake-off model fits in the CUDA card's memory before booking the machine, and build `ds4-agent` and `libds4` from the same commit there.
- **DFlash testing.** The downstream fork's DFlash runs only on Metal, so without a Mac it gets `FakeNative` tests only.
- **Metal kernel path.** A Metal library needs the ds4 checkout at run time, and the installer records that path.
- **Images.** The engine's text-keyed cache cannot save sessions with images. `view_image` works but marks the session unsaveable, as in C.

## 25. Open questions

None. The two questions in the review draft were settled from the record:

1. **The downstream fork's pin.** The downstream agent's own records settle which fork revision it pins. `ds4.h` is the same at both candidate revisions, so the ABI probes give the same result for either.
2. **The bake-off model is DeepSeek V4 Flash Q2.** At `5379c1c`, `download_model.sh` lists `ds4f-q2` at about 81 GB and recommends it for 96 and 128 GB machines. V4.1 Flash Q2 (`ds41f-q2`) is about 341 GiB on disk with 152 GiB of main weights, and it runs only on Metal or a DGX Spark, with SSD streaming or across two machines. It cannot run on a 96 GB GPU. The commit message's "defaults to DS4.1 Flash" does not mean the script picks V4.1 by itself: the script needs a target name. The V4.1 tag spellings are covered by parser and prompt fixtures. Live V4.1 runs move to M5.

## 26. Calls I made

These are my decisions, not the user's. Each can be changed.

- The C agent's tools are the default and the harness's tools are an option (decision 3). This follows from KV compatibility, which the user requires.
- The parity test replays rounds instead of comparing whole runs, and bake-off gate 3 measures KV reuse within ds4-py instead of comparing prefill counts with `ds4-agent` (sections 20 and 21). The first draft's version of both cannot pass.
- A sixth bake-off gate, replay parity, takes over the reasoning gate 1 used to carry.
- The approval prompts for `bash` and for writes outside the project stay as the first draft had them, although `ds4-agent` never asks.
- `RepoContext`, `AskUser`, `ToolOutputLimits` and the guardrail classes are not in the default list, because each would change the prompt or the output relative to `ds4-agent`.
- The engine core has no dependencies outside the standard library, including the web tools' WebSocket client.
- The C prototype string is the single source for each binding's ctypes types and its signature check.
- The probe compiles one file per binding, so a missing optional function does not hide the others' results.
- Extension bindings are declared through a `ds4py.bindings` entry point, so the probe sees them at build time.
- The KV file reading and writing moves to M2, because replay parity needs to read `ds4-agent`'s files, and `/save` moves to M3, because bake-off gate 5 needs it. The first draft had all persistence in M4.
- The date note and hints move to M3, because they are part of every `ds4-agent` session. The reminder stays in M4.
- When a tool result does not fit, the strategy that owns the context trigger makes room in its own way through `make_room`. Only `DS4Compaction` compacts (section 13).
- A fork keeps its title and takes the fork time as its `created_at`, so it gets its own file under C's naming formula. Forks come from `session.fork()`; there is no `/fork` command in M1 to M4 (section 17).
- ds4-py uses engine checkpoints, which the C agent does not. The facade holds the `system` checkpoint, and strategies may hold up to four more. `restore` is disabled on any backend that fails the live restore check (sections 10 and 20).
- When generation reaches the reserve inside a response, the open stream asks the context owner to make room and continues. With no owner, the run fails with `ContextExhausted`, so partial text never becomes a final answer (section 11).
- Each `/save` writes the live history as its own snapshot and a save record in `~/.ds4py/saves/`, so every KV file maps to an exact history. `events.jsonl` is telemetry only (section 15).
- Capabilities keep session-lifetime state in `session.state`, not on `self`. The session event bus moves from M4 to M3, because `bash` jobs need it (sections 13 and 23).

## 27. Changes from the first draft

- The tools: the C agent's tools and prompt, as `DS4AgentTools` and `ds4py.prompts`, replace `FileSystem`, `Shell`, `WebSearch` and `WebFetch` in the default list. The web tools are a port of `ds4_web.c` (Chrome over DevTools), not a search API.
- Harness names, checked against the 0.59.0 wheel: `Guardrails(...)` becomes `InputGuardrail`, `OutputGuardrail` and `ToolGuardrail`. The `Compaction` strategies are `SlidingWindowCompaction`, `SummarizingCompaction` and `TieredCompaction`. `SystemReminders(every_tokens=…)` does not exist; `Reminder.trigger` is the predicate to use. `StepPersistence(FileStore(…))` becomes `StepPersistence(FileStepStore(…))`. `params.declared_function_tools` becomes `params.function_tools`.
- The harness pin moves from 0.17.0 to 0.59.0.
- The ABI claim: `argtypes` does not make a signature change fail at import. The signature and layout probes replace that claim.
- The downstream agent's needs: building from any tree, loading from a path, required and optional bindings, the extension API, the decode-step contract and the logits hook are new.
- `libds4` includes `ds4_kvstore.o`, and the store uses its constants and text renderer.
- The parity test and bake-off gates 1, 3 and 6, as listed in section 26.
- The module map grows from 3,400 to about 6,400 lines.
- The milestones have acceptance criteria and new estimates; the first draft's M1 to M4 were 6 to 8 weeks.
- The package splits into `ds4py` and `ds4py[agent]`.
- Behaviours the first draft did not list: the date note, hints, the reminder format, `--prefix-file`, the full command-line flag list, and the Chrome consent rules.

## 28. Sources

- antirez/ds4 at `5379c1c`: [`ds4.h`](https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4.h), [`ds4_agent.c`](https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_agent.c), [`ds4_kvstore.h`](https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_kvstore.h), [`ds4_web.c`](https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_web.c), [`ds4_prompt_prefix.c`](https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/ds4_prompt_prefix.c), [`Makefile`](https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/Makefile), [`AGENT.md`](https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/AGENT.md).
- antirez/ds4 at `5379c1c`: [`download_model.sh`](https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/download_model.sh) and [`README.md`](https://github.com/antirez/ds4/blob/5379c1c9ac315f1540371b06a3db88491d4059ff/README.md), for model sizes and supported hardware.
- The downstream fork's `ds4.h`, compared with upstream on 2026-10-21.
- [`pydantic-ai-harness` 0.59.0](https://pypi.org/project/pydantic-ai-harness/0.59.0/) and [`pydantic-ai-slim` 2.59.0](https://pypi.org/project/pydantic-ai-slim/2.59.0/): class names and signatures read from the wheels on 2026-10-21.
- [`pi-context-paging`](https://github.com/rochecompaan/pi-context-paging) v0.3.9, for the strategy contract and `RollingWindow`.
- The first draft, [`archive/2026-10-21-ds4-py-first-draft.md`](archive/2026-10-21-ds4-py-first-draft.md), for the framework comparison and the LocalHarness findings.
