# ds4-py: a Python-hosted ds4 coding agent

Oct 21, 2026 · @Roché Compaan

## Summary

Build `ds4-py` on **Pydantic AI with the Harness package**, with one `Model` subclass (`DS4Model`) talking to a Python facade that owns a single `ctypes` module over a new `libds4` shared library. Python owns the conversation, the token loop, tool parsing, tools, policy, context management and run records; C owns weights, the live KV cache, logits and the KV payload bytes. One process, no HTTP, no C host loop. About 3,400 lines of Python replace the 13,635-line `ds4_agent.c`, because the harness supplies the tools, guards, tracing and persistence, and `ds4.h` already exposes the engine as `ds4_engine` + `ds4_session` with full-prefix `sync`, per-token `eval`/`sample`, `rewind`, progress and cancel callbacks, and payload save/load.

LocalHarness is not the base: its loop would accept an in-process model object, but `start`, token counting, window probing, runtimes and `/model` all require an OpenAI-compatible server URL, and a local HTTP shim is what the brief rules out. Its co-residence rule for untrusted-ingest vs dangerous-host tools and its JSONL event record are worth copying as capabilities. Pydantic AI core alone is the fallback if the 0.x harness API churn becomes a cost; `mini-swe-agent`, `smolagents`, Pi and OpenCode are not better starting points for a Python host with an embedded C engine.

Five decisions carry the design:

1. **The token transcript is the engine-side truth, the message list is the harness-side truth, and `DS4Model` binds them.** Sampled tokens are appended by id and never re-rendered from text, so `<think>` content and native tool markup replay byte-for-byte and the live KV is reused. Harness history rewrites re-render only from the divergence point.
2. **Context management is a plug-in point with one cost rule.** Any strategy is a capability that rewrites the request view (paging) or the stored history (compaction, bookmarks); every change costs one prefill of the tokens after the first changed position, and `session.plan()` reports that cost exactly before a move. The default strategy is the C agent's compaction recipe; a port of `pi-context-paging` (stable-cut rolling window with exact recovery tools) and a bookmark strategy drop in beside it.
3. **Native tool format, rendered and parsed in `DS4Model`.** Tool definitions from the harness become the DSML (V4 / V4.1), GLM or Qwen tools prompt; a streaming parser over sampled text produces `ToolCallPart`s, switches sampling to greedy inside tool syntax, and stops generation when a block closes.
4. **One engine worker thread** owns every `ds4_session_*` call; progress and cancel callbacks re-enter Python with the GIL and do nothing but post to a queue or read an `Event`.
5. **KV files stay compatible with `ds4-agent`** (`~/.ds4/kvcache/<sha>.kv`, 48-byte `KVC` header, text, payload, title), so sessions move between the two agents and the parity test can diff transcripts token by token.

The C side needs only a `libds4` link target; `ds4.c` is untouched. The plan is five milestones, with a working raw loop in 1–2 weeks and a usable coding agent at M3 (\~6 weeks).

## What ds4 actually exposes

`ds4.h` is already the boundary this design needs: `ds4_engine` is the loaded model and `ds4_session` is one mutable inference timeline that owns the live KV cache and logits. The caller hands it a full token prefix and `ds4_session_sync()` reuses, extends or rebuilds the state. The header comment says exactly this, and `AGENT.md` makes it a rule: CLI/server code must not know tensor internals. So Python can sit where `ds4_agent.c` sits today, with no engine changes.

Two things are missing on the C side. There is no library target: the Makefile links `CORE_OBJS` (`ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o ds4_engram.o`, or `ds4_cuda.o` plus the vendored `cuda/mmq/*.o` on Linux) straight into each binary. And `ds4_agent.c` (13,635 lines) is a monolith: the tool-call parsers, tools, compaction and persistence live in the binary, not in the engine. Both are fixable without touching `ds4.c`: add a `libds4` shared-library target, and reimplement the agent logic in Python.

### The engine API the agent needs

| Group | Functions (from `ds4.h`) | Notes for the Python port |
| --- | --- | --- |
| Engine lifecycle | `ds4_engine_open(&e, &opts)`, `ds4_engine_close`, `ds4_engine_model_name`, `ds4_engine_model_id`, `ds4_engine_is_deepseek41 / is_glm_dsa / is_qwen4`, `ds4_engine_has_vision`, `ds4_engine_routed_quant_bits`, `ds4_engine_mtp_draft_tokens` | `ds4_engine_options` is a plain struct; model family flags choose the tool syntax |
| Prompt rendering | `ds4_chat_begin`, `ds4_chat_append_message(e, toks, role, content)`, `ds4_chat_append_think_prefix(e, toks, mode)`, `ds4_chat_append_assistant_prefix(e, toks, mode)`, `ds4_chat_append_multimodal_message(e, toks, role, parts, embeddings, n, spans, err)` | The engine owns the chat templates; Python never sees template strings. Roles used by the C agent: `system`, `user`, `assistant`, `tool` |
| Tokenizer | `ds4_tokenize_text`, `ds4_tokenize_rendered_chat`, `ds4_token_text(e, id, &len)`, `ds4_token_eos`, `ds4_token_is_stop`, `ds4_token_is_stop_for_think_mode`, `ds4_tokens_push / copy / free / starts_with` | `tokenize_rendered_chat` turns control-token spellings (`<｜User｜>`, `｜DSML｜`, `<think>`) into special tokens. Only for trusted text (tool prompt, saved transcripts), never for user input |
| Session | `ds4_session_create(&s, e, ctx)`, `ds4_session_free`, `ds4_session_sync(s, prompt, err, n)`, `ds4_session_sync_multimodal`, `ds4_session_eval(s, token, err, n)`, `ds4_session_eval_speculative(...)`, `ds4_session_sample(s, temp, top_k, top_p, min_p, &rng)`, `ds4_session_argmax`, `ds4_session_rewind(s, pos)`, `ds4_session_pos / ctx / tokens / common_prefix`, `ds4_session_invalidate` | `sync` returns `DS4_SESSION_SYNC_INTERRUPTED` (2) on cancel. `sample` is one token from the current logits; `eval` feeds it and refreshes logits |
| Callbacks | `ds4_session_set_progress(s, fn, ud)`, `ds4_session_set_display_progress`, `ds4_session_set_cancel(s, fn, ud)` | `progress_fn(ud, event, current, total)`; events seen by the agent: `prefill_chunk`, `prefill_display`, `prefill_work`. `cancel_fn(ud) -> bool` is polled during sync |
| Think and steering | `ds4_think_mode_for_context`, `ds4_think_mode_parse_level`, `DS4_THINK_LEVEL_BASE + N`, `ds4_session_set_directional_steering_ffn`, `ds4_session_set_power` | Changing the think level rebuilds the prefix (full re-prefill); steering does not |
| KV persistence | `ds4_session_stage_payload`, `ds4_session_write_staged_payload`, `ds4_session_save_payload`, `ds4_session_load_payload(s, fp, bytes, err, n)`, `ds4_session_save_snapshot / load_snapshot` | The engine owns the serialized graph state; the outer file header and policy are the caller's (`ds4_kvstore.h`, 48-byte `KVC` header) |
| Vision | `ds4_engine_vision_encode_file`, `ds4_vision_embedding_free`, `ds4_vision_span` | Optional; images make sessions unsaveable today |

The rest of the header is serving or distributed plumbing (`ds4_sessions_eval_batch*`, `ds4_session_eval_layer_slice`, TP/RDMA, `ds4_engine_generate_argmax`) that the agent does not use.

### How ds4-agent uses it, and what must be reproduced

The C agent keeps **one `ds4_tokens transcript` as the only conversation state**. There is no message list. Text is re-derived by concatenating `ds4_token_text` over the transcript. The behaviours below are what make it work, and each becomes a Python responsibility:

1. Every user turn appends system notes (one-time date, optional hints), then `user`, then `ds4_chat_append_assistant_prefix(think_mode)`; the whole transcript is passed to `ds4_session_sync`, which diffs it against the live prefix and prefills only the suffix.
2. Generation samples one token, checks `ds4_token_is_stop_for_think_mode` **before** eval (stop tokens are never pushed or evaluated), evals, pushes the sampled id onto the transcript, and streams `ds4_token_text` to the renderer. With speculative decoding, overshoot after a stop or a completed tool call is undone with `ds4_session_rewind(block_start + i)`, so KV and transcript stay exactly aligned.
3. The turn ends by pushing `ds4_token_eos` onto the transcript for DeepSeek and Qwen, not GLM; it is evaluated by the next sync.
4. Tool calls are detected **in the sampled text**, not by a grammar: a streaming parser watches for `<｜DSML｜tool_calls>` (V4), `<｜DSML｜ calls>` (V4.1), `<tool_call>…<arg_key>…<arg_value>` (GLM) or `<tool_call>\n<function=…>\n<parameter=…>` (Qwen). While the parser is inside tool syntax the sampler switches to greedy. Generation stops the moment the closing tag completes; no stop token is needed.
5. All calls in one block run sequentially; results are concatenated as `Tool result N (name):` and appended as **one** `tool`-role message via `ds4_chat_append_multimodal_message` (role `user` when images are attached). Then a new assistant prefix is appended and the loop syncs again.
6. Tools: `read`, `more`, `write`, `edit`, `list`, `search`, `bash`, `bash_status`, `bash_stop`, `google_search`, `visit_page`, `view_image`. Per-tool output cap 128 KiB. `bash` runs via `posix_spawn` in its own process group with a monitor thread, and returns head/tail excerpts plus a path to the full output file.
7. Context: compaction is model-written. At 85% of ctx (or when the reserve is hit) the agent syncs `transcript[:bottom] + user(compaction prompt) + assistant_prefix(no-think)`, generates a summary greedily with `ds4_session_argmax`, and rebuilds the transcript as system tokens + `user(summary)` + a verbatim tail (`min(ctx/10, 50000)` tokens, cut at a user-token boundary). The system prompt is re-injected as a reminder every 50,000 tokens.
8. Prefill progress is drawn from the progress callback relative to the cached prefix (`done = current - cached_prefix`), with t/s computed on the Python side of the same callback.
9. Cancel: Ctrl+C sets a flag the cancel callback returns; an interrupted sync leaves the transcript intact and appends EOS; interrupted generation keeps the partial tokens. Nothing is truncated.
10. Threading: a single worker thread owns every `ds4_session_*` call, synchronously. The UI thread never touches the session. Progress callbacks run on the worker thread.
11. Persistence: `~/.ds4/kvcache/<sha40>.kv` = 48-byte header + rendered transcript text + KV payload (+ title trailer). Restore loads the payload directly (no prefill) and takes the token history **from the session**, not from re-tokenizing text. A stripped file (no payload) is re-tokenized with `ds4_tokenize_rendered_chat` and prefilled.

Sources: [ds4.h](https://github.com/antirez/ds4/blob/main/ds4.h), [ds4\_agent.c](https://github.com/antirez/ds4/blob/main/ds4_agent.c), [AGENT.md](https://github.com/antirez/ds4/blob/main/AGENT.md), [Makefile](https://github.com/antirez/ds4/blob/main/Makefile), [docs/SERVER.md](https://github.com/antirez/ds4/blob/main/docs/SERVER.md) (read from a clone of `main` on 2026-10-21).

## Framework evaluation

Pydantic AI with the Harness package is the right base. Its model layer is an abstract class with an in-process reference implementation, its hook system covers every behaviour the C agent has, and it ships the tools, guards, compaction, tracing and persistence the brief asks Python to own. LocalHarness has the better local-first instincts but is wired to an OpenAI-compatible HTTP server at every layer except the inner loop. The other candidates are either not Python or not harnesses.

| Criterion | Pydantic AI + Harness | LocalHarness | Others |
| --- | --- | --- | --- |
| In-process model backend | `Model` ABC: implement `request()` / `request_stream()`; `FunctionModel` is a working in-process example; no HTTP anywhere in the contract | `LLMClient` wraps `openai.AsyncOpenAI`; the loop takes any object with `stream_complete() -> (message, usage)` (duck-typed), but `start`, `TokenCounter`, window probes, `/model`, runtimes and reconnect all require a `base_url`; no provider plugin slot | `mini-swe-agent`, `smolagents`: callable-style model, trivial to satisfy |
| Tool loop, tool definitions | `ToolDefinition(name, description, parameters_json_schema)` arrives in `ModelRequestParameters.declared_function_tools`; the model returns `ToolCallPart(tool_name, args, tool_call_id)`; validation, retries, `sequential`, approval are framework code | `ToolRegistry` + pydantic validation + pluggy pre/post hooks; native or XML (`FnCallConverter`: Hermes, Qwen `<function=>`) — no DSML | thin or absent |
| Policy and guards | `Guardrails` (input/tool-call/result/output, secret masking), human-in-the-loop tool approval, `ToolCallJudge`, `ApprovalReviewer`, `PromptInjectionDefender`, `SpendLimits`, `AskUser` | `PermissionGate` with `auto/guarded/trusted/read-only/unattended`, capability floor (`ingest=untrusted` and `host=dangerous` may not co-reside), `ContentStore` taint, stuck-detector, act-guard, self-check | mostly none |
| Context management | `Compaction` strategies (tool-result clearing, sliding window, LLM summary, tiered), `ToolOutputLimits`, `SystemReminders`, `ConversationSearch`, `ProcessHistory` hook for a custom recipe | 4-stage `CompactionPipeline` at 80%, tool-result eviction to a content store at 50%, `compact.md` carried across sessions | none |
| Run records | OpenTelemetry spans for every model and tool call, `StepPersistence` (file/SQLite/Mongo, `continue_run`, `fork_run`), `on_event` hook for a JSONL log | `EventBus`: 26 typed events, `bus-events.jsonl` + per-session JSONL, replay and resubmit | none |
| Coding tools | `Coder` = `FileSystem` (read/write/edit/list/grep), `Shell` (persistent, background jobs), `RepoContext` (AGENTS.md), `SubAgents`, `ClearToolResults`, `ToolOutputLimits`, `RepairToolArguments` | read/write/edit/glob/grep/bash, web, agent, cruncher | bash-only (`mini-swe-agent`) |
| Interfaces | `clai` CLI, web chat, AG-UI/Vercel adapters, ACP for Zed | Rich/prompt\_toolkit REPL, Discord, mobile | CLI |
| Maturity | core has a strict version policy; harness is 0.x (0.17.0, 2026-08-05), MIT, Python ≥ 3.11 | 0.20.2 (2026-10-20), alpha, MIT, Python ≥ 3.12, \~104k lines, 76 releases since June | varies |
| Thinking and local-model quirks | `profile['thinking_tags']` splits `<think>` into `ThinkingPart`; `ignore_streamed_leading_whitespace`; `default_structured_output_mode='prompted'` | `disable_thinking`, reasoning budget via `extra_body` keys per runtime | n/a |

### Why Pydantic AI + Harness

- The contract is exactly the seam the brief describes. `DS4Model.request_stream()` receives messages and tool definitions and yields parts; nothing above it knows whether a socket or a `ctypes` call produced them.
- Every ds4-agent behaviour maps to a hook: `before_model_request` for compaction and reminders, `before_tool_execute` for approvals and policy, `on_event` for records, `get_instructions` for the system prompt, `for_run` for per-session state.
- `Coder` is a `CombinedCapability` that comes apart into named pieces, so the agent is assembled from parts with the pieces that fight the KV cache left out (next section).
- The brief wants the model's native tool format. Pydantic AI does not render tools into text for you, which is a feature here: the rendering lives in `DS4Model`, next to the parser, with the C agent's prompts as fixtures.

### Why not LocalHarness as the base

At loop level an `InProcessLLM` with `config`, `stream_complete()` and the `Provider*Error` types would run inside `AgentLoop`, and the bench and tests already build the loop that way. But `localharness start` is 3.7k lines that assume a server: it aborts if `server_reached` is false, exits if `/tokenize` is missing for vLLM/llama.cpp, probes `/v1/models` for the window, and keys reconnect, speed stats, autostart and `/model` on `base_url`. The plugin API has no model-provider slot, and core builds `LLMClient` before plugins load. The alternative, a local HTTP shim in front of the engine, is exactly what the brief rules out. Keep LocalHarness as a design reference: its co-residence rule, taint-tracked content store and event vocabulary are worth copying into capabilities.

### Other starting points considered

| Project | Language | Verdict |
| --- | --- | --- |
| Pydantic AI core without the Harness package | Python | Fallback if the 0.x harness churn becomes a cost: `Model`, `Capability`, `FunctionToolset`, MCP, instrumentation and persistence are in core; file and shell tools would be written by hand (about 800 lines) |
| `mini-swe-agent` (Princeton) | Python | \~100-line loop, bash as the only tool; a good reference for how little an agent loop needs, but no guards, records or context management |
| `smolagents` (Hugging Face) | Python | Code-agent oriented; model is a callable; coding-tool and safety layers are thin |
| Pi (`pi-mono`), OpenCode, Codex CLI, Claude Code | TypeScript / Rust | ds4 already documents them as `ds4-server` clients; none embeds a C engine and none is Python |
| `llama-cpp-python` | Python | Not a harness, but the reference for wrapping a C engine with ctypes: a generated low-level module mirroring the header, one high-level class owning state. Section 5 follows that shape |

Sources: [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/), [pydantic-ai-harness 0.17.0 on PyPI](https://pypi.org/project/pydantic-ai-harness/0.17.0/), [pydantic-ai `main` (2026-10-21)](https://github.com/pydantic/pydantic-ai), [localharness.dev](https://localharness.dev/), [LocalHarness architecture](https://localharness.dev/architecture/), [ahwurm/localharness 0.20.2](https://github.com/ahwurm/localharness), [ds4 CLIENTS.md](https://github.com/antirez/ds4/blob/main/docs/CLIENTS.md). Pydantic AI and LocalHarness facts come from reading their source at the commits named; the "Others" row is from general knowledge and was not re-verified.

## Architecture

&#91;embedded content: process layout · 7 components, one native boundary\]

The harness calls the model adapter as it would call any provider; the adapter talks to the facade in Python objects; the facade is the only client of the ctypes module; the ctypes module is the only client of `libds4`. Tools, guards and records never see the engine.

### Ownership split

| Concern | Owner | Where |
| --- | --- | --- |
| Conversation flow, turn loop, retries | Python | `pydantic_ai.Agent` run loop |
| Token loop (sample, stop, eval, push, rewind) | Python | `ds4py.engine.EngineWorker` |
| Prompt rendering to tokens | C templates, driven by Python | `PromptBuilder` → `ds4_chat_append_*` |
| Tool-call parsing (DSML / GLM / Qwen) | Python | `ds4py.engine.ToolSyntax` |
| Tool execution, sandboxing, output caps | Python | harness capabilities (`FileSystem`, `Shell`, …) |
| Policy, guards, approvals, injection defence | Python | harness capabilities (`Guardrails`, tool approval, `ToolCallJudge`) |
| Context management (compaction, reminders, tool-result clearing) | Python | `DS4Compaction` capability + `SystemReminders` |
| Run records, tracing, resume | Python | `StepPersistence`, OpenTelemetry, `events.jsonl` |
| Model computation, live KV, logits, speculative decode | C | `libds4` |
| KV serialization bytes | C | `ds4_session_*_payload` |
| KV file header, naming, listing | Python | `ds4py.store` |

### Module map (`ds4py/`)

| Module | Lines (est.) | Depends on |
| --- | --- | --- |
| `_native.py` | 400 | ctypes, `libds4` |
| `engine/transcript.py` | 150 | — |
| `engine/worker.py` | 300 | `_native` |
| `engine/prompt.py` | 150 | `_native` |
| `engine/toolsyntax/{dsml,glm,qwen}.py` | 500 | — |
| `engine/session.py` (public facade) | 250 | all of `engine/` |
| `model.py` (`DS4Model`) | 350 | `pydantic_ai.models`, `engine` |
| `capabilities/{tools_prompt,compaction,reminders,records}.py` | 500 | `pydantic_ai`, `pydantic_ai_harness` |
| `store.py` (KV files) | 200 | `engine` |
| `cli.py` / `tui.py` | 600 | `pydantic_ai` CLI or `prompt_toolkit` |

About 3,400 lines of Python replace the 13,635-line C agent, because the harness already supplies the file and shell tools, the permission gate, compaction strategies, tracing and persistence, and the C engine keeps the templates and tokenizer.

### Process and threads

- **Main thread**: the asyncio loop that runs `Agent.run_stream_events()`, the TUI, and the event log.
- **Engine worker thread**: created by `ds4py.engine.open()`; the only thread that calls into `libds4`. `DS4Model.request_stream()` awaits results from it through `asyncio.to_thread`-style futures and an `asyncio.Queue` fed by `loop.call_soon_threadsafe`.
- **Tool subprocesses**: `Shell`'s persistent shell and background jobs, as the harness implements them; `bash` never blocks the engine thread.

There is no inference server, no socket, and no second process holding model weights.

## The ctypes boundary: `ds4py/_native.py`

One module, about 400 lines, is the only place that imports `ctypes`, declares `argtypes`/`restype`, or holds a raw pointer. Everything above it receives Python objects. Three rules keep this true: no other module imports `_native`; `_native` exports only functions and opaque handle classes, never `ctypes.Structure` instances; and every C error path becomes a Python exception at this layer.

### Building the library

Add one target to the ds4 Makefile (or a `make lib` wrapper in this project) that links `CORE_OBJS` into `libds4.dylib` / `libds4.so` with `-shared -fPIC` and the same `METAL_LDLIBS` / `CUDA_LDLIBS` as the binaries. `ds4.c` already exports every function in `ds4.h` with default visibility, so nothing in the engine changes. On Linux the CUDA objects are compiled by `nvcc`, so the link step uses `$(DS4_LINK)` as the existing targets do. The ROCm path already compiles with `-fPIC` (`ROCM_HOST_CFLAGS`).

Two runtime details carry over from the binaries. The Metal backend locates kernels relative to the ds4 checkout (`--chdir` exists for this reason), so the loader passes the engine's base directory and the Python side `os.chdir`s or sets the equivalent before `ds4_engine_open`. And `ds4_engine_open` takes an instance lock (AGENT.md: "Do not run multiple huge model processes concurrently"), so a second engine in the same process is refused; the facade enforces one engine per process.

### What the module wraps

| Python name | C function(s) | Marshalling |
| --- | --- | --- |
| `open_engine(opts: EngineOptions) -> EngineHandle` | `ds4_engine_open`, `ds4_engine_close` | Fills a `ds4_engine_options` Structure from a frozen dataclass; strings are kept alive as `c_char_p` attributes on the handle for the engine's lifetime |
| `engine_info(h) -> EngineInfo` | `ds4_engine_model_name/model_id/is_*/has_vision/routed_quant_bits/mtp_draft_tokens/vocab_size` | Read once at open; the facade branches on `family` (`deepseek4`, `deepseek41`, `glm`, `qwen`) |
| `render(h, ops) -> list[int]` | `ds4_chat_begin`, `ds4_chat_append_message`, `ds4_chat_append_think_prefix`, `ds4_chat_append_assistant_prefix`, `ds4_chat_append_multimodal_message` | Takes a list of `("begin")`, `("message", role, text)`, `("think_prefix", mode)`, `("assistant_prefix", mode)` tuples, renders into one `ds4_tokens`, copies out a Python `list[int]`, frees the C vector. Python never holds a `ds4_tokens` |
| `tokenize(h, text, rendered=False) -> list[int]` | `ds4_tokenize_text`, `ds4_tokenize_rendered_chat` | `rendered=True` only for trusted text |
| `token_text(h, id) -> bytes` | `ds4_token_text` | Returns bytes, not str: tokens split UTF-8 sequences, and the decoder in the facade reassembles them |
| `is_stop(h, id, think_mode)`, `eos(h)`, `special_token(h, text) -> int \| None` | `ds4_token_is_stop_for_think_mode`, `ds4_token_eos`, `ds4_tokenize_rendered_chat` | `special_token` requires the text to tokenize to exactly one id (the C agent's `agent_special_token_id`) |
| `create_session(h, ctx) -> SessionHandle`, `free` | `ds4_session_create`, `ds4_session_free` | `SessionHandle.__del__` frees; the facade also calls `close()` explicitly |
| `sync(s, tokens, images=None, *, progress, cancel) -> SyncResult` | `ds4_session_sync`, `ds4_session_sync_multimodal`, `ds4_session_set_progress`, `ds4_session_set_display_progress`, `ds4_session_set_cancel` | Builds a `ds4_tokens` from the list, installs the callbacks, calls sync, clears the callbacks (NULL) in a `finally`, maps return 0 → ok, 2 → `Interrupted`, else `EngineError(err_buf)` |
| `sample(s, temp, top_k, top_p, min_p, rng) -> int` | `ds4_session_sample` | `rng` is a `ctypes.c_uint64` owned by the facade so the sequence is reproducible with `--seed` |
| `eval(s, token)`, `eval_speculative(s, token, max, eos, sampling, rng) -> list[int]` | `ds4_session_eval`, `ds4_session_eval_speculative` | The speculative call returns the accepted ids (first is the fed token); capacity 17 as in the C agent |
| `rewind(s, pos)`, `pos(s)`, `ctx(s)`, `tokens(s) -> list[int]`, `common_prefix(s, tokens)`, `invalidate(s)` | the matching `ds4_session_*` | `tokens` copies out of `ds4_session_tokens` |
| `stage_payload(s) -> StagedPayload`, `write_staged(p, fileobj)`, `load_payload(s, fileobj, nbytes)` | `ds4_session_stage_payload`, `ds4_session_write_staged_payload`, `ds4_session_payload_file_free`, `ds4_session_load_payload` | The C API wants a `FILE*`; the module opens one with `fdopen(os.dup(fileobj.fileno()))` via libc so Python keeps ownership of the Python file |
| `set_steering(s, scale)`, `set_power(s, pct)`, `think_mode_for_context`, `parse_think_level` | `ds4_session_set_directional_steering_ffn`, `ds4_session_set_power`, `ds4_think_mode_*` | Think modes are an `IntEnum` mirroring `ds4_think_mode`, with `level(n) = DS4_THINK_LEVEL_BASE + n` |
| `vision_encode(h, path) -> VisionEmbedding` | `ds4_engine_vision_encode_file`, `ds4_vision_embedding_free`, `ds4_prompt_append_vision` | Phase 2; the embedding's `float*` is never exposed, only its fingerprint and token count |

### Callbacks and the GIL

`ds4_session_sync` runs for seconds to minutes and calls `progress_fn` and `cancel_fn` from inside the engine on the calling thread. ctypes releases the GIL around a foreign call and re-acquires it for each `CFUNCTYPE` callback, so the rest of the Python process (the TUI, the event log) keeps running while the engine prefills on the worker thread. The two callbacks do minimal work: `progress_fn` writes `(event, current, total)` into a `queue.SimpleQueue` and returns; `cancel_fn` reads one `threading.Event` and returns a bool. The `CFUNCTYPE` objects are created once per session and stored on the `SessionHandle` so they are not garbage-collected while the engine holds their address. `ds4_session_eval` and `ds4_session_sample` are short calls with no callbacks and are made the same way.

The cancel path works because the engine polls the callback inside its prefill loop and returns `DS4_SESSION_SYNC_INTERRUPTED` after draining in-flight GPU work. Python must not raise out of a callback (ctypes prints the traceback and returns 0); exceptions are stashed and re-raised after the foreign call returns.

### Threading contract

All `SessionHandle` methods are called from exactly one thread, the engine worker that the facade owns (section 6). `_native` asserts this in debug builds by recording the creating thread id. The engine itself is not re-entrant per session and the C agent never calls it from two threads; neither does this design.

### Error and memory ownership

Every call that takes `char *err, size_t errlen` gets a 512-byte buffer and raises `EngineError(msg)` on non-zero return (except the documented `2` from sync). `ds4_tokens` vectors are created, filled and freed inside one `_native` function; `ds4_vision_embedding` ownership transfers to the session on `sync_multimodal` as the C API specifies, and the Python wrapper marks the object consumed so it is not double-freed. `libds4` is loaded with `ctypes.CDLL(path, mode=RTLD_LOCAL)` once; the module caches the handle and refuses a second load with a different path.

## The Python engine facade: `ds4py.engine`

The facade is the part of `ds4_agent.c` that is about tokens rather than about files or terminals, rewritten in Python on top of `_native`. It owns the one engine worker thread, the transcript, prompt rendering, tool-syntax detection and sampling policy. It exposes ordinary Python types: `str`, `bytes`, dataclasses, iterators. The harness adapter (section 7) is its only client.

### `Transcript`

A `Transcript` is a `list[int]` plus a parallel list of `Segment` records: `(start, end, kind, meta)` where `kind` is `system`, `user`, `assistant_prefix`, `assistant`, `tool`, `eos`, `reminder`, `compaction`. The token list is the truth, exactly as in C; segments are an index over it so the facade can find turn boundaries, strip reminders, cut a compaction tail at a `<｜User｜>` boundary, or render the `/history` view without scanning for special tokens. Segments never change the token list. Sampled assistant tokens are appended one at a time by id; they are never re-rendered from text, so `<think>` content and tool-call markup replay byte-for-byte on the next sync. That is what makes the live KV reusable.

### `EngineWorker`

One `threading.Thread` owns the `SessionHandle`. Callers submit `Job` objects to a queue and get back an iterator of events; the worker runs one job at a time. Jobs:

| Job | What it does | Events it yields |
| --- | --- | --- |
| `SyncJob(tokens, images)` | `sync` with progress/cancel callbacks installed | `Prefill(done, total, cached_prefix, tps)` at most every 100 ms, then `Synced(pos)` or `Interrupted(pos)` |
| `GenerateJob(sampling, stop_policy, max_tokens)` | the sample→stop-check→eval→push loop from section 2, including speculative rewind | `Token(id, text_bytes)`, `Thinking(on/off)`, `ToolSyntax(state)`, `Stopped(reason)` where reason ∈ `stop_token`, `tool_call_complete`, `max_tokens`, `context_reserve`, `interrupted`, `parser_error` |
| `CompactJob(reason)` | the compaction recipe (section 7) run with `argmax` | `Prefill…`, `Token…`, `Compacted(summary, dropped, kept)` |
| `SaveJob(path)`, `LoadJob(path)` | section 8 | `Saved(bytes)`, `Loaded(tokens)` |
| `ControlJob(steer=, power=, think=)` | between turns only | `Applied` |

The cancel `threading.Event` is shared by all jobs and set by the UI on Ctrl+C; the worker clears it when it reports `Interrupted`. This reproduces the C agent's contract: an interrupted sync keeps the transcript and gets an EOS appended by the caller; interrupted generation keeps the partial tokens.

### `PromptBuilder`

Builds the token prefix for a round from the transcript and the engine's family, using only `_native.render`. It knows the role names the engine's templates accept (`system`, `user`, `assistant`, `tool`), when to use `tokenize(rendered=True)` (the tools prompt, the reminder, a restored transcript) and when never to (user text, tool results), and which family needs what: DeepSeek V4 and Qwen get an EOS pushed at the end of every assistant turn, GLM does not; V4.1 uses the `  calls ` / `  invoke ` / `  parameter ` tag spellings; Qwen takes its reasoning-effort as a system message rather than a think prefix. The system prompt text and the tool schemas come from the harness (section 7); the builder only renders.

### `ToolSyntax`

A streaming parser over sampled text with one implementation per family: `DsmlParser` (V4 and V4.1 tag spellings), `GlmParser` (`<tool_call>NAME<arg_key>…<arg_value>…</tool_call>`), `QwenParser` (`<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>`). Each exposes `feed(bytes) -> ParserState` and `calls() -> list[ToolCall(name, args: dict[str, str | json], raw: str)]`, mirrors the C parser's leniencies (missing leading bar, bare `<｜DSML｜invoke`, optional `｜` before `>` in closing tags, GLM's "another `<tool_call>` continues the block"), and applies the `ds4_tool_text.h` un-escaping (`&lt;` + close tag, `&amp;lt;…`) to string parameters. The parser also drives two sampling decisions the C agent makes: switch to greedy while inside tool syntax, and stop generation the instant a block closes. Markup inside `<think>` is suppressed and reported as a parser error, exactly as in C.

### `SamplingPolicy`

`temperature`, `top_p`, `min_p`, `top_k=0`, `seed`, with the per-family defaults the C agent applies (GLM: top\_p 0.95, min\_p 0) and the greedy override from the parser. The `rng` state lives here so a run with `--seed` is reproducible across Python and C.

### `Decoder`

Reassembles UTF-8 from `token_text` bytes, tracks `<think>`/`</think>` with a 16-byte pending buffer, and emits `TextDelta(text, thinking: bool)` to the harness. It is display-side only; nothing it does feeds back into the transcript.

### Public surface

```python
engine = ds4py.open(model="gguf/ds4flash.gguf", ctx=100_000, backend="metal", think=ThinkMode.HIGH, seed=None)
session = engine.session()                 # one per process today
session.transcript                          # Transcript, read-only view
session.append_user(text, images=())        # renders and appends
session.append_tool_results(results)        # one "tool" message for the whole block
session.append_system_note(text)            # date note, hints, reminders
for ev in session.run_round(tools_prompt_already_in_prefix=True):  # sync + generate
    ...                                      # Prefill / TextDelta / ToolCalls / Stopped
session.plan(view_tokens)                   # exact (reused, to_prefill, fits) without syncing
session.compact(reason)                     # CompactJob
session.save(path); session.load(path)
session.cancel()                            # thread-safe
```

Everything here is testable without a model: `_native` has a `FakeNative` implementing the same functions over a toy tokenizer, so `Transcript`, `PromptBuilder`, the three parsers and the generation loop run in CI. The parsers additionally get the C agent's own fixtures: the DSML/GLM/Qwen syntax examples from its system prompts and traces recorded with `--trace`.

## Plugging into the harness

The harness sees one `Model` subclass and a list of capabilities. The hard problem is that Pydantic AI treats the message list as the truth while the engine treats the token transcript as the truth. `DS4Model` holds the binding between them; the capability list is chosen so that nothing rewrites history behind the binding's back except the one compaction recipe.

### `DS4Model(Model)`

| Member | Implementation |
| --- | --- |
| `system` | `'ds4'` |
| `model_name` | `engine_info.model_name` |
| `profile` | `{'context_window': ctx, 'supports_tools': True, 'supports_thinking': True, 'thinking_tags': ('<think>', '</think>'), 'supports_inline_system_prompts': True, 'default_structured_output_mode': 'prompted', 'supports_json_schema_output': False, 'supports_tool_return_schema': False}`. Inline system prompts are declared supported because the ds4 templates accept a `system` message anywhere, which is how the C agent appends its date note, hints and reminders |
| `request()` | `prepare_request`, then `request_stream` collected into a `ModelResponse` |
| `request_stream()` | `prepare_request`; bind messages to the transcript (below); `session.run_round()` on the worker; yield a `DS4StreamedResponse` whose `_get_event_iterator` turns facade events into `handle_text_delta(vendor_part_id='content', content=…, thinking_tags=profile tags)` and, when the parser completes a block, one `handle_tool_call_part(vendor_part_id=i, tool_name=, args=dict, tool_call_id=generated)` per call. `finish_reason` is `'tool_call'`, `'stop'` or `'length'`; `_usage` is `RequestUsage(input_tokens=suffix_prefilled, cache_read_tokens=cached_prefix, output_tokens=generated)` |
| `close_stream()` | `session.cancel()`; the partial tokens stay in the transcript as in C |
| `count_tokens()` | `len(PromptBuilder.render(messages))` without syncing; used by `UsageLimits` and the compaction trigger |

### Binding messages to the transcript

`DS4Model` keeps a `Binding`: for each `ModelMessage` it has rendered, the transcript segment it produced, keyed by the message's identity and a content hash. On every request it walks `request_context.messages` from the front:

1. While messages match bound segments in order, nothing is rendered; this is the common case and the engine's `common_prefix` covers all of it.
2. For a `ModelResponse` that `DS4Model` itself produced, the bound segment is the sampled tokens plus EOS, so the response is never re-rendered from its text. A `ModelResponse` the harness synthesised (a compaction summary, a replayed history) is rendered canonically as an assistant message.
3. New request parts render in order: `SystemPromptPart` and `InstructionPart` → `system`; `UserPromptPart` → `user` (with images, the multimodal call); all `ToolReturnPart`s of one request → **one** `tool` message, each body prefixed `Tool result N (name):` as the C agent does; `RetryPromptPart` → the same `tool` message with the C agent's `Tool error: …` text plus the family's syntax reminder.
4. The first time a request has tool definitions, the tools prompt is rendered once at the front of the transcript, from `params.declared_function_tools`: DeepSeek gets the DSML prompt through `tokenize(rendered=True)` after an empty `system` message (V4.1 tag spellings when `is_deepseek41`), GLM and Qwen get it as a `system` message with their `<tools>` JSON lines. The user's own instructions go after it as a second `system` message, never through the rendered tokenizer. A later change in the tool set re-renders from that point, which costs a full prefill, so capabilities that add or hide tools mid-run (`ToolSearch`, deferred loading) are left out.
5. If the message list diverges from the binding (a capability rewrote `ctx.messages`), the binding keeps the common prefix and re-renders the rest canonically. The engine then prefills only from the divergence point. This is correct for any rewrite, and expensive for rewrites near the front, which is why `ClearToolResults` and sliding-window compaction are not used.
6. Finally the assistant prefix for the current think mode is appended and `run_round` syncs the whole transcript.

### Capability list

```python
agent = Agent(
    DS4Model(session),
    capabilities=[
        LocalWorkspace(project_dir),
        DS4Instructions(),                 # the C agent's role text, rules, edit guidance; get_instructions()
        FileSystem(tools=('read_file','write_file','edit_file','list_files','grep'), max_read_chars=50_000, content_hashes=False),
        Shell(allow_interactive=True, default_timeout=270, tools=['shell']),   # foreground + background jobs = bash / bash_status / bash_stop
        RepoContext(expose_inventory_tool=False),
        WebSearch(), WebFetch(),           # replaces google_search / visit_page; local DuckDuckGo path
        ToolOutputLimits(bands=[Band(over=128_000, action=Truncate(max_chars=128_000))]),
        Guardrails(...), AskUser(answerer=tui.ask),
        DS4Approvals(),                    # before_tool_execute: shell/write need a yes unless --yolo; web needs the C agent's one-time consent
        DS4Compaction(soft=0.85),          # a ContextStrategy; swap for RollingWindow(...) or Bookmarks(...)
        SystemReminders(every_tokens=50_000),
        DS4RunRecord(path),                # on_event → events.jsonl
        StepPersistence(FileStore(session_dir)),
        Instrumentation(),
    ],
)
```

Left out of the default list: `SubAgents` (the engine's instance lock allows one session; delegation returns later as a nested round that rewinds), `RepairToolArguments` (arguments arrive as strings from the native parser, not as JSON), `ToolCallJudge` and `ApprovalReviewer` (a second model call on the only session would bust the prefix; policy here is rules, not a judge model). The harness's own context capabilities (`ClearToolResults`, the `Compaction` strategies) are not forbidden; they are context strategies like any other and pay the cost rule below, so the default list carries the one that matches ds4-agent.

### Context strategies: the extension point

Context management is a plug-in point, not a fixed recipe. A strategy is a capability; the Binding (above) makes any history change correct, and the engine makes every change cost the same thing: **one prefill of the tokens after the first changed position**. The system prefix never changes (it is in `sysprompt.kv`), so the cost of a rewrite is bounded by the window, never by the full session. Strategies differ only in *when* they move the front and *what* the model can get back.

The contract a strategy implements:

| Hook or API | Use |
| --- | --- |
| `before_model_request(ctx, request_context)` | Set `request_context.messages` for a **view-only** change (stored history untouched: paging, rolling windows) or `ctx.messages[:]` for a **persistent** change (compaction, bookmarks). Return a view whose prefix is stable between moves so the KV is reused on every other request |
| `get_toolset()` | Recovery tools that read the **stored** history (`ctx.messages` plus the run record) and return exact text: search, browse, load, read-page; or `expand_bookmark` |
| `for_run(ctx)` | Per-session state: the cut point, bookmark table, token accounting. Reset on `/new`, `/switch` and after another strategy's rewrite |
| `on_event` | Turn-end bookkeeping, stamping stable keys into `message.metadata['ds4.key']` so cut points are keyed by message, not by list index |
| `session.plan(view) -> Plan(reused, to_prefill, fits)` | Facade call: renders a candidate view without syncing and compares it with the live `common_prefix`, so a strategy knows the exact prefill cost of a move before it makes it. `session.ctx_used` is `ds4_session_pos`, exact, not an estimate |
| `session.compact(...)` | Only `DS4Compaction` uses the engine-side summary job; other strategies never call the model themselves |

Exactly one strategy owns the trigger in a session. Like `pi-context-paging` cancelling Pi's automatic compaction, a strategy declares `owns_context_trigger = True` and `DS4Compaction` stands down when another owner is present; manual `/compact` still works.

**Strategies, with their KV cost**

| Strategy | Kind | What moves, and when | Cost per move | What the model can recover |
| --- | --- | --- | --- | --- |
| `DS4Compaction` (default; the C recipe) | persistent | At 85% of ctx: history becomes `system + user(summary) + verbatim tail` (`min(ctx/10, 50000)` tokens, cut at a `<｜User｜>` boundary); an open assistant message is resumed | summary (≤ 4096 tokens, generated with `argmax`) + the tail | the summary and the verbatim tail; nothing else |
| `RollingWindow` (port of `pi-context-paging`) | view-only | When the rendered view exceeds `token_budget`, drop whole oldest turns until ≤ `trim_to` (default 5/8 of the budget); the cut point is kept until the budget is exceeded again; a paging notice sits right after the system prefix; the active request and unread tool results are never cut | the window (`trim_to`) once per cut; zero between cuts because the cut is stable | everything, exactly: `search_history`, `browse_history`, `load_history`, `read_context_output` over the stored history; thinking stays private; the recovery tools' own calls are excluded |
| `Bookmarks` | persistent | Replace chosen messages (large tool results, a finished sub-task's exchanges) with a one-line stub such as `[bookmark b17: bash output, 12 KiB, 3 errors; expand_bookmark("b17")]`; applied in batches at turn boundaries | everything after the earliest replaced message; batching, and preferring recent items, keeps it small | the exact original via `expand_bookmark`, returned as a tool result |
| Harness `ClearToolResults`, sliding `Compaction`, LLM summary | persistent | As the harness documents | the same rule: tokens after the earliest change; `ClearToolResults` edits old messages, so it is the most expensive of these | per strategy |
| `ToolOutputLimits` | at the source | Truncates or spills before anything enters the transcript | free | the spilled file via `read_file` |

`RollingWindow` gets two things from the engine that the Pi version has to estimate: exact token counts for the budget (the facade renders the view) and exact `ctx_used`; its `/context-paging stats` report maps onto engine counters (`cache_read_tokens` = reused prefix, written = prefilled). Its design choices carry over unchanged: whole-turn cuts, stable keys, protected content, the notice counted inside the budget, no summarisation, private assistant content excluded from recovery. Setting `token_budget` below the model's "dumb zone" works the same way, and `ds4_session_create(ctx)` can be given the budget directly so the engine reserves only the KV the window needs.

`DS4Compaction` in detail: a `before_model_request` hook with the C agent's trigger, `ctx_used >= 0.85 * ctx` or `ctx - used <= max(min(8192, ctx/8), reserve + 128)`, where `reserve` is the rendered length of the compaction prompt plus an assistant prefix plus `clamp(ctx/8, 256, 4096)` for the summary. When it fires it calls `session.compact(reason)`, which runs the whole recipe on the worker in one pass: sync `transcript[:bottom] + user(prompt) + assistant_prefix(no-think)`, generate the summary with `argmax` until a stop token or the DSML special token, choose the tail, and resync `system + user(summary) + tail`. Then the hook rewrites `ctx.messages[:]` to `[system parts, UserPromptPart(summary), *tail messages]` and rebinds. If generation was mid-response when the reserve hit, the hook sets `resume_open_assistant` and `DS4Model` continues the same assistant message, as the C agent does. The pre-tool-result check ("does this observation fit, else compact, else replace it with the C agent's `Tool error: tool result still does not fit`") lives in `after_tool_execute` and is shared by every strategy.

Sources for the strategy table: [pi-context-paging README and architecture notes](https://github.com/rochecompaan/pi-context-paging) (v0.3.9, read 2026-10-21); the harness compaction and tool-output pages linked in section 3.

### `DS4Approvals` and policy

`before_tool_execute` applies a small rule table rather than a judge model: `shell` and `write_file`/`edit_file` outside the workspace ask through `AskUser`; `web_fetch`/`web_search` ask once per session (the C agent's web consent prompt, 30 s timeout defaulting to no); `--yolo` disables prompts; `--read-only` denies writes. A denied call returns the C agent's `Tool error:` text so the model's next round reads like it does in C. Tool results carry `outcome='denied'` so records show it.

### `DS4RunRecord`

An `on_event` capability appending one JSON line per `AgentStreamEvent` to `<session>/events.jsonl`, plus engine-side events the harness does not see (`Prefill`, `Compacted`, `Saved`, `Interrupted`) forwarded from the facade's event queue. The record and the KV file name are enough to reproduce a session: `StepPersistence` gives `continue_run`, and the transcript binding is rebuilt from the record on load.

### Interface

The TUI is our own, written fresh, not a fork of a stock CLI. [`clai`](https://pydantic.dev/docs/ai/integrations/cli/), the harness's terminal chat (`clai -a ds4py:agent` or `agent.to_cli_sync()`), is used for one day in M3 as a smoke test of the harness stack and then set aside: it has a fixed command set, no status or footer area, no notification hook and no plugin API, so it cannot show the prefill bar or host `/save`, `/think` or an extension's `/context-paging`. Forking it would mean maintaining a UI shaped around API providers; writing ours is smaller.

Most of the work is not UI. The harness's `agent.run_stream_events()` supplies the token, tool-call and tool-result events; the facade supplies prefill progress and engine events. The TUI (`tui.py`, \~600 lines, `prompt_toolkit`) is an input loop, a renderer for those two streams, a footer, and two registries that exist from the first commit:

- `register_command(name, handler, help)`: every built-in command (`/save`, `/list`, `/switch`, `/del`, `/strip`, `/new`, `/compact`, `/think`, `/steer`, `/power`, `/hints`, `/history`) is registered through this call, so an extension's command is not a special case.
- `set_status(key, text)` and `notify(text, level)`: the footer shows `ctx used/total`, the prefill bar (`▶`/`·`, 32 cells, t/s overlay), generation t/s and the ❄️ greedy marker from the engine's keys, and any key an extension sets (for example `context-paging: paging on`).

Capabilities reach these through the session event bus (section 9), never by importing the TUI. ACP for editors and the harness's web chat are adapters that sit beside this TUI later, not replacements for it. The TUI thread never calls the engine; it submits jobs and reads the event queue.

## Session persistence and KV checkpoints

Sessions are saved in the C agent's own file format so `ds4-agent` and `ds4-py` can open each other's files. The format is small enough to write from Python with `struct`; only the KV payload bytes are produced and consumed by the engine.

### File layout (`~/.ds4/kvcache/<sha40>.kv`)

| Offset | Field | Source |
| --- | --- | --- |
| 0–2 | magic `KVC` | Python |
| 3 | version `1` | Python |
| 4 | `quant_bits` (2, 4, 5, 6 or 8) | `engine_info.routed_quant_bits` |
| 5 | `reason` (`5` agent-system, `6` agent-session) | Python |
| 6 | `ext_flags` (bit 3 = session-title trailer present) | Python |
| 7 | `model_id` | `engine_info.model_id` |
| 8–11 | `u32 tokens` | `len(transcript)` |
| 12–15 | `u32 hits` | 0 |
| 16–19 | `u32 ctx_size` | session ctx |
| 20 | payload ABI `2` | Python |
| 24–31 | `u64 created_at` | first user turn |
| 32–39 | `u64 last_used` | save time |
| 40–47 | `u64 payload_bytes` | `stage_payload().bytes`, or 0 for a stripped file |
| 48– | `u32 text_bytes` + rendered transcript text | `b"".join(token_text(id) for id in transcript)` |
| … | KV payload | `write_staged(payload, f)` |
| … | `u32 title_len` + title | Python |

The file name is `sha1(title_bytes + created_at.to_bytes(8, 'little'))` where the title is the normalized first user prompt. Files are written to `path.tmp.XXXXXX` and renamed, as in C.

### Save

`/save` is a `SaveJob` on the worker. Before staging, the worker syncs the transcript so the live KV equals it (this evaluates the trailing EOS that the C agent also leaves unevaluated), then calls `stage_payload`, writes the header and text, streams the staged payload into the file, and appends the title trailer. Sessions with images are refused with the C agent's message; this lifts when the engine's text-keyed cache learns image fingerprints.

The run record (section 7) stores the file name and the save event, so a session can be resumed from the record alone.

### Restore

`/switch <sha-prefix>` reads the header and validates `model_id` and `quant_bits` against the open engine. If `payload_bytes > 0`, the worker calls `load_payload` and then **takes the token list from `tokens(session)`**, not from re-tokenizing the text, so the restored transcript is bit-identical to what was saved and no prefill happens. If the file is stripped, the text is re-tokenized with `tokenize(rendered=True)` and synced, which prefills from scratch. Segment metadata is rebuilt from the run record when it exists, and otherwise by scanning the restored tokens for the engine's role special tokens (`special_token("<｜User｜>")` and friends).

Two cases are deliberately the same as the C agent: a tensor-parallel session ignores the payload and rebuilds from text, and a mismatched `model_id` is an error, not a warning.

### `sysprompt.kv`

The rendered system prefix (think prefix + tools prompt + extra system text) is saved once as `sysprompt.kv` with reason `agent-system`, keyed by its rendered bytes. Startup loads it if the bytes match, so a new session begins with the system prompt already in KV. This is worth keeping: on a 96 GB Mac the tools prompt is several thousand tokens, and the C agent made this the first thing a session does.

### Strip and delete

`/strip` rewrites the file with `payload_bytes = 0` and a recomputed token count; `/del` unlinks. `/list` reads only headers and trailers, sorts by `last_used`, and filters on the open engine's `model_id`.

## Build, packaging, testing, milestones, risks

### Build and packaging

- `ds4py` is a normal `uv`/`pip` package (Python ≥ 3.11, the harness's floor) with **no compiled extension**. It loads `libds4` at runtime from `DS4_LIB` or `<ds4 checkout>/libds4.{dylib,so}`.
- The `libds4` target lives in a 20-line `Makefile.lib` kept in this repo and run against a ds4 checkout: `make -f Makefile.lib lib DS4=../ds4` builds `CORE_OBJS` with the platform's existing rules and links them shared. Upstreaming the target as a PR to ds4 is worth trying; the project's README asks users to shape it with coding agents, and a library target does not touch inference code.
- Pin ds4 by commit. `ds4.h` changes often (it is beta, 827 commits), and `_native` declares `argtypes` for every call, so a signature change fails at import with a clear message rather than corrupting memory. A tiny `ds4py check` command loads the library, opens no model, and verifies every symbol it needs is present.

### Testing

| Layer | How | Needs a model? |
| --- | --- | --- |
| `_native` | load the library, `check` symbols; open a tiny CPU session if `DS4_TEST_MODEL` is set | optional |
| `Transcript`, `PromptBuilder`, `EngineWorker` | `FakeNative` with a toy tokenizer and scripted logits | no |
| `ToolSyntax` parsers | fixtures: the C agent's own prompt examples, `--trace` captures from `ds4-agent`, fuzzed splits of each fixture at every byte boundary | no |
| `DS4Model` | pydantic-ai `TestModel`-style tests: feed `ModelRequest` lists, assert the rendered transcript ops and the `ModelResponse` parts | no |
| Capabilities | pydantic-ai's testing guide with `FunctionModel` standing in for `DS4Model` | no |
| Parity with `ds4-agent` | run both agents on the same prompt with `--temp 0 --seed`, same model, and diff the transcripts token by token after each round | yes (Metal or CUDA box) |
| Integration | `ds4-eval`-style capability checks (`core` suite) through the Python agent, and the KV round-trip: save in `ds4-py`, `/switch` in `ds4-agent`, and back | yes |

The parity test is the one that matters: it proves the token loop, the EOS rule, the greedy-in-tool-syntax rule and the tool-result rendering are byte-identical to C, which is what makes KV files portable between the two agents.

### M3 exit test: the 12-task bake-off

M3 is done when `ds4-py` matches `ds4-agent` on the same 12 tasks, same model (DeepSeek V4 Flash Q2), same machine, `--temp 0 --seed 1`, run unattended by a script that drives both agents in non-interactive mode and grades with hidden tests.

| Tasks | Source | Grading |
| --- | --- | --- |
| 10 Python exercises | the [Aider polyglot benchmark](https://aider.chat/2024/12/21/polyglot.html) (Exercism, 225 exercises across 6 languages); the 10 Python exercises are chosen once by a fixed seed and pinned in `bench/m3.yaml` | the exercise's hidden tests, run by the script after the agent says it is done; the agent gets the instructions file and up to 2 attempts as the benchmark allows |
| 2 repo tasks on `ds4py` itself | a planted bug (`tests/test_toolsyntax_glm.py` fails: an off-by-one in the `<arg_value>` close handling) and a feature ("add a `/ctx` slash command that prints `ctx used/total`", which exercises the command registry) | the test suite passes; the feature's own test passes |

Gates, all required:

1. Pass rate ≥ ds4-agent's on the same run, allowing one task of slack (the sampling paths are byte-identical at `--temp 0`, so a larger gap is a bug).
2. Zero `parser_error` stops and zero `Tool error: invalid …` observations across the 12 runs; every tool call the model made was executed or refused by policy.
3. Wall time ≤ 1.5× ds4-agent per task, and prefilled tokens per round equal within 1% (the KV prefix is being reused; a mismatch means a binding bug).
4. One run is interrupted with Ctrl+C mid-prefill and resumed from the next prompt; one run triggers an approval prompt for `shell` and continues after a `y`.
5. Every run leaves a complete `events.jsonl` and a saved KV file that `ds4-agent` can `/switch` into.

What this does not test, on purpose: long-context behaviour (M4 adds a 150k-token session and the compaction and `RollingWindow` cases), SWE-bench-style multi-file repository tasks (M5, a 10-task SWE-bench Verified slice once the harness is stable), and GLM / Qwen parity (M5).

### Extension points reserved in the MVP

Pi-level extensibility is not an MVP feature, but five things are structural and cheap now and expensive later, so the MVP builds them in and a conformance test keeps them honest. The list comes from the API surface `pi-context-paging` actually uses (`pi.on(context | session_start | session_shutdown | turn_end | session_tree | model_select | session_compact | session_before_compact)`, `pi.registerTool`, `pi.registerCommand`, `pi.getActiveTools`, `ctx.sessionManager.getEntries / getBranch / buildSessionProjection`, `ctx.getContextUsage`, `ctx.getSystemPrompt`, `ctx.ui.setStatus / notify`, `ctx.isProjectTrusted`).

| Reserved now | Why it is structural | Pi equivalent |
| --- | --- | --- |
| Stored history ≠ request view. `DS4Model` binds to `request_context.messages` only and never writes `ctx.messages`; the run record stores the full harness message list with parts (text, thinking, tool calls, results), not just tokens | A strategy that pages or bookmarks needs the complete history with thinking separable from public text; if the MVP keeps only the token transcript, recovery tools cannot be built later | `context` hook, `sessionManager.getEntries`, private-block filtering |
| Stable message keys stamped in `message.metadata['ds4.key']` at creation, and session identity separate from the KV file name (session id in the run record; KV files are a content-keyed cache) | Cut points, bookmarks, branches and forks all key on messages, and two forks of one title must not collide on `sha1(title, created_at)` | stable history IDs, `session_tree`, `getBranch` |
| A session event bus in `ds4py` (`session_start`, `session_end`, `turn_end`, `before_compact`, `compacted`, `model_changed`, `switched`) that capabilities subscribe to, separate from the harness's per-run `on_event` | Lifecycle events that the harness does not model; retrofitting them means finding every call site later | `pi.on(session_*)`, `turn_end`, `model_select` |
| `session.plan()`, `session.ctx_used`, `session.system_tokens` and `session.active_tools` as public facade calls | Strategies need exact cost and usage; exposing them later is easy, but designing the facade so they are cheap (render without sync) is not | `getContextUsage`, `getSystemPrompt`, `getActiveTools` |
| A command registry and a status-key registry in the TUI, with capabilities able to register both; project-level settings with a trust flag | If M3 adopts a stock CLI with no registry, every extension needs a fork of the UI | `registerCommand`, `ui.setStatus`, `ui.notify`, `isProjectTrusted` |

The proof is a conformance test, run in CI without a model: a stub strategy must (1) rewrite the request view and see the Binding re-render from the cut only, (2) register tools that read paged-out history with thinking excluded, (3) own the trigger and see `DS4Compaction` stand down, (4) get `plan()` costs that match the engine's `common_prefix` on `FakeNative`, (5) survive `/switch` and a fork with its keys intact, and (6) add a slash command and a status key. The first real implementation is the `RollingWindow` port in M4; it exercises every row above and is the acceptance test for the extension API.

### Milestones

1. **M1 – library and raw loop (1–2 weeks).** `libds4` target, `_native`, `EngineWorker` with sync/sample/eval/rewind, `ds4py raw "prompt"` matching `ds4-agent --raw`. Prefill progress and Ctrl+C working.
2. **M2 – chat and tools in the facade (2 weeks).** `Transcript`, `PromptBuilder`, three `ToolSyntax` parsers, `run_round` with tool-call stop. A 200-line test CLI that executes `read` and `bash` directly, to validate against C traces before the harness exists.
3. **M3 – harness (2 weeks).** `DS4Model`, `DS4ToolsPrompt`, `Coder` from parts, approvals, `events.jsonl`, `a one-day agent.to_cli_sync() smoke test, then our own TUI with the command and status registries`. First end-to-end coding task.
4. **M4 – context and persistence (1–2 weeks).** `DS4Compaction`, reminders, `ds4py.store`, `/save /list /switch /strip /del`, `sysprompt.kv`. KV round-trip test with `ds4-agent`.
5. **M5 – polish (ongoing).** Speculative decoding (`eval_speculative`), vision via `view_image`, GLM/Qwen parity runs, `/think`, `/steer`, ACP for editors, MCP servers.

### Risks and open questions

- **`ds4.h` churn.** Mitigated by pinning and `argtypes`, but every ds4 release needs a `_native` review. Keep the wrapped surface small (the table in section 5) so reviews are short.
- **Instance lock.** The engine refuses a second instance per machine. Subagents must share the one session (the C agent delegates nothing); `delegate_task` therefore means "run a nested conversation in the same KV, then rewind", which is a later milestone, not M3. Until then `SubAgents` is left out of the Coder composition.
- **GIL and callbacks.** Progress callbacks re-acquire the GIL \~10 times a second during prefill; negligible. The risk is a Python exception inside a callback, which ctypes swallows; the facade stashes and re-raises, and the callbacks are kept trivial.
- **Compaction fidelity.** The C recipe cuts the tail at a `<｜User｜>` token boundary and resumes an open assistant message after compaction. Both are reproduced; the resume case is the hardest to test and is covered by a `FakeNative` scenario plus one live parity run.
- **Harness API stability.** `pydantic-ai-harness` is 0.x and moves fast (0.17.0 on 2026-08-05). The adapter touches only `Model`, `StreamedResponse`, `Capability` hooks and the documented capability classes; a quarterly bump is expected.
- **Metal kernel path and `--chdir`.** The library needs the ds4 checkout at runtime for `metal/*.metal`; the installer records that path. A future ds4 change that embeds kernels would remove this.
- **Images.** Saving sessions with images is not supported by the engine's text-keyed cache; `view_image` works but marks the session unsaveable, as in C.
