# M3 — harness agent

**Goal:** Add the harness adapter, stable message/token binding, capabilities, session persistence, non-interactive runner, interactive TUI, and the 12-task live bake-off with all six M3 gates.

**Spec:** [`docs/specs/2026-10-21-ds4-py-design.md`](../specs/2026-10-21-ds4-py-design.md), especially **M3 — harness agent**, sections 6, 9, 11–18, 21, and 23. M3 implements conformance items **1, 6, 7, and 9**. M2 already covers item 8; items 2–5 are completed by M4 except where a focused M3 test protects a prerequisite.

**Base:** `main` and this worktree at the verified M2 squash merge (`f3d21d2`). Work in this existing worktree: `/home/roche/projects/satyrn/ds4-py/.worktrees/m3-harness-agent`. Do not create a nested worktree.

**Current interfaces:** Extend `ds4py.engine.Session`, `PromptBuilder`, `Transcript`, `Plan`, `SamplingPolicy`, `FakeNative`, and `ds4py.engine.replay`. Do not add a second conversation history, token loop, tool parser, workspace implementation, approval mechanism inside the TUI, or C-tool port.

**Harness pin:** `pydantic-ai-harness[cli]==0.59.0`, whose package metadata selects `pydantic-ai-slim[cli]==2.59.0`. Direct resolver and installed-API inspection found that the spec's `pydantic-ai 1.84.0` statement and several capability names are stale; Task 1 amends those facts without changing the approved behavior. The inspected adapter surface is `Model.request`, `Model.request_stream`, `Model.prepare_request`, `StreamedResponse._get_event_iterator`, `StreamedResponse.get`, `_parts_manager`, `RequestUsage`, `Agent`, `StepPersistence`, `FileStepStore`, `LocalWorkspace`, `FileSystem`, `Shell`, `RepoContext`, `Skills`, `Instrumentation`, and `AskUser`. `LocalWorkspace` and `Instrumentation` come from `pydantic_ai.capabilities`; the other capabilities come from the harness modules.

**Default verification:** `uv run --extra agent pytest -m "not live and not build"` must pass. Also verify that core imports and the existing raw/check commands work without the `agent` extra. Live bake-off and replay checks are separate, explicitly marked, and reported separately from synthetic tests.

**Review focus:** (1) cancellation, context-reserve handling, and speculative rewind against live KV; (2) save ordering and token/hash/segment agreement after restore; (3) binding by stable key and content hash, never object identity; (4) real denied tool outcomes without tool execution; (5) footer truncation and stripped/corrupt restore behavior; (6) meaningful gate metrics and honest live evidence. The self-review at the end is completed once, after the tasks are written.

---

## Architecture and boundaries

### 1. One token history, with a separate binding index

`Transcript` remains the only token history. `MessageBinder` maps harness messages to token ranges using each message's persistent `metadata['ds4.key']` and a deterministic content hash. It does not use Python object identity.

A request first retains the longest bound prefix whose keys, hashes, tool definition hash, instructions, and think mode still match. It truncates and re-renders only from the first changed segment. A DS4Model-generated response binds the tokens sampled by the existing decode loop; it is never serialized back into text during ordinary operation. Re-rendering is allowed only for a genuinely rebuilt history, including a stripped save.

Use focused modules:

- `src/ds4py/binding.py`: harness messages, stable keys/hashes, rendering decisions, provenance, and count-without-sync.
- `src/ds4py/model.py`: `DS4Model` and `DS4StreamedResponse`; async harness adaptation only.
- `src/ds4py/engine/tool_syntax/render.py`: canonical family syntax when a restored assistant tool call has no preserved raw form.
- `src/ds4py/engine/session.py` and `chat.py`: small public primitives needed by the binder; no harness imports.

Every message receives a key when first seen. A response key is assigned before its sampled segments are bound. System notes and generated tool reminders have an empty message key and are not inserted into harness message history.

### 2. The worker remains the only native-call path

All payload, checkpoint, reset, and generation operations go through `EngineWorker`. `Session.save_payload` uses a `SavePayloadJob`; `Session.load_payload` uses a `LoadPayloadJob`. The file object remains owned by Python. Native calls receive a libc `FILE *` from `fdopen(os.dup(file.fileno()))`, and the duplicated descriptor is closed exactly once.

The M3 native layer adds the required serialized-state bindings omitted by M1: `ds4_session_stage_payload`, `ds4_session_write_staged_payload`, `ds4_session_payload_file_free`, and `ds4_session_load_payload`. `payload_bytes` already exists. These bindings are required, not optional.

### 3. `AgentRuntime` is the composition root

`AgentRuntime` owns the current harness history, binder, agent components, step store, save store, and lifecycle operations. The TUI talks to `AgentRuntime` and session events, not native handles. `DS4Model` talks to the facade, not the TUI.

The public runtime surface is:

```python
class AgentRuntime:
    async def run(self, prompt: str) -> AgentRunResult: ...
    async def save(self) -> SaveRecord: ...
    async def switch(self, session_prefix: str) -> SaveRecord: ...
    async def new(self) -> str: ...
    async def fork(self) -> str: ...
    def cancel(self) -> None: ...
    async def close(self) -> None: ...
```

`new` closes session-scoped state and resets the live transcript/KV. `switch` closes old state, restores the selected identity and history, and loads KV when valid. `fork` changes only the live identity: tokens, KV, binding, messages, and state objects survive; the old on-disk directory remains. Runtime methods are safe to call from the asyncio UI because blocking facade waits happen off the event-loop thread; native work still stays on the single engine worker.

### 4. Harness capabilities own harness concerns

`src/ds4py/agent.py` builds the default `Agent` with:

`LocalWorkspace`, `FileSystem`, `Shell`, `DS4ShellJobs`, `RepoContext`, `Skills`, `ToolOutputLimits`, `DS4RunRecord`, `StepPersistence`, `Instrumentation`, `DS4SystemNotes`, `DS4Approvals`, then `AskUser` only in interactive mode.

The verified `0.59.0` API has no `WorkspaceToolset`, `SkillDiscovery`, or `FileChangeTracker`; `FileSystem`/`Shell` provide the workspace tools, `Skills` provides skill discovery, and file/tool change evidence goes through `FileSystem` events, `StepPersistence`, and `DS4RunRecord`. Do not write local replacements for those filesystem, shell, or skill implementations. `DS4ShellJobs` records background-command IDs in session state and stops live jobs on `session_end`; `ToolOutputLimits` handles large results without another model call. `StepPersistence` receives `store=FileStepStore(...)` directly. Interactive `AskUser` receives `answerer=...`; it has no `approvals=` argument in this API, so `DS4Approvals` remains a separate capability.

M3 deliberately does **not** install `DS4Compaction`. `DS4SystemNotes` exposes the compaction/reset hook and is tested with a stub compaction subject; real compaction and the 50,000-token system-prompt reminder are M4. Interactive `/think` is M5, although the engine retains its think-mode primitive for command-line configuration and binding invalidation tests. Denials use `SkipToolExecution(ToolDenied(...))` so history contains `ToolReturnPart(outcome='denied')` and the tool body never runs. `LocalWorkspace(read_only=True)` wraps the backend used by `FileSystem` and `Shell`, so workspace writes and commands remain forbidden even when approvals are bypassed.

### 5. Persistence has one canonical history and one atomic record order

`FileStepStore` is canonical for messages and run status. `ds4py.store` remains stdlib-only and owns the KV header, save record, atomic file replacement, prefix lookup, and model/quant validation.

The fixed KV header is 20 bytes: `struct.Struct('>8sIQ')`; magic is `b'DS4PYKV\0'`, version is `1`, and the last field is the unsigned payload byte count. Token hashes use SHA-256 over big-endian unsigned 32-bit token IDs (`>{n}I`). Save order is:

1. sync the transcript and atomically replace `session.kv`;
2. write a `complete` snapshot in run `<session id>.save-<n>` through `FileStepStore`;
3. atomically replace `save.json` last.

A missing/mismatched payload follows the specified fresh-render path. A missing snapshot or unmatched segment is an error naming the run, snapshot, or key. `/list`, `/strip`, `/del`, `/compact`, and `/fork` as user commands remain out of M3; `AgentRuntime.fork()` exists for lifecycle conformance and M5.

### 6. The bake-off is sequential, pinned, and evidence-producing

The 10 exercises come from `Aider-AI/polyglot-benchmark` commit `7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f`. Selection uses `random.Random(20261021).sample(sorted(exercise_paths), 10)` over the 34 Python practice exercises. The resulting pinned order is:

1. `scale-generator`
2. `affine-cipher`
3. `zipper`
4. `grep`
5. `go-counting`
6. `proverb`
7. `tree-building`
8. `dot-dsl`
9. `paasio`
10. `variable-length-quantity`

The two repository tasks are `repo-glm-close` and `repo-ctx-command`. Agents run one process at a time on one machine, with the same model, context, sampling, task files, and at most two attempts. Tests stay outside the agent-visible workspace except for the intentionally visible GLM regression test. The runner emits per-task metrics and a six-gate report; it does not infer live success from unit tests.

---

## Task 1: Add the agent extra and bind initial requests, messages, and tool results

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `docs/specs/2026-10-21-ds4-py-design.md` for the verified harness API corrections below
- Modify: `.github/workflows/ci.yml`
- Create: `src/ds4py/binding.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/chat.py`
- Modify: `src/ds4py/engine/prompt.py` only if a small public rendering primitive is needed
- Create: `tests/agent/__init__.py`
- Create: `tests/agent/conftest.py`
- Create: `tests/agent/test_binding.py`

**Interfaces:**
- `Session.engine` is a read-only property exposing the existing facade-owned `Engine`; it does not transfer ownership.
- `ToolDefinition` from the harness converts to `ds4py.engine.tool_prompt.ToolDefinition(name, description, parameters)` through `parameters_json_schema`; ordering is preserved.
- `Session.append_message(role, text, *, images=(), meta=None) -> Segment` renders one ordinary message through `PromptBuilder` and keeps payload text plain. M3 rejects images with the documented M5 error.
- `MessageBinder.prepare(messages, *, tools=(), instructions=None) -> BindingChange` binds a request without syncing.
- `BindingChange(kind, start, continuation)` reports `unchanged`, `append`, or `rewrite`, the first changed token offset, and whether the request follows tool results.

- [x] **Step 1: Add the dependency pin without adding a version-only test**

First correct the inspected harness facts in the spec: the pin selects `pydantic-ai-slim[cli]==2.59.0`, not the `1.84.0` distributions; `LocalWorkspace` and `Instrumentation` live in `pydantic_ai.capabilities`; `FileSystem`, `Shell`, and `Skills` replace the stale `WorkspaceToolset`, `FileChangeTracker`, and `SkillDiscovery` names; `StepPersistence(store=...)` receives its store directly; and `AskUser(answerer=...)` has no approvals parameter. Also correct the profile's two `supports_json_*` fields and the computed `RequestUsage.requests`/`total_tokens` properties described in Task 3. Keep the behavioral requirements, harness pin, and no-local-tool-port decision unchanged.

Add:

```toml
[project.optional-dependencies]
agent = [
  "pydantic-ai-harness[cli]==0.59.0",
  "prompt-toolkit>=3.0.52,<4",
]
```

Update `uv.lock` with `uv lock`. Update CI to install/run with `--extra agent`. The prompt-toolkit direct dependency is declared because the TUI imports it directly; the lock file selects the exact build. Do not write tests that merely assert dependency or lock versions.

Run the import smoke checks:

```bash
uv sync --extra agent
uv run --extra agent python -c "from pydantic_ai import Agent; from pydantic_ai.capabilities import LocalWorkspace, Instrumentation; from pydantic_ai_harness import RepoContext, AskUser, StepPersistence; from pydantic_ai_harness.filesystem import FileSystem; from pydantic_ai_harness.shell import Shell; from pydantic_ai_harness.skills import Skills; from pydantic_ai_harness.step_persistence import FileStepStore; import prompt_toolkit"
uv run --isolated --no-project --with . python -c "import ds4py; import ds4py.cli"
```

The isolated import must succeed without the harness extra. Keep `ds4py/__init__.py` and raw/check CLI paths free of harness imports.

- [x] **Step 2: Write failing binding tests**

Create `tests/agent/test_binding.py` with:

- `test_first_request_renders_tools_and_binds_each_message`
- `test_system_and_instruction_parts_are_separate_system_segments`
- `test_user_text_is_plain_and_tool_markup_is_rendered`
- `test_tool_results_join_in_history_order_as_one_tool_message`
- `test_scalar_and_structured_tool_results_have_stable_text`
- `test_unknown_tool_result_uses_tool_error_text`
- `test_images_fail_before_mutating_the_transcript`

Use `ModelRequest`, `ModelResponse`, `UserPromptPart`, `SystemPromptPart`, `InstructionPart`, `TextPart`, `ToolCallPart`, `ToolReturnPart`, and `RetryPromptPart` from the pinned harness. Construct an unknown-tool retry with `RetryPromptPart('Tool error: unknown tool missing', tool_name='missing', tool_call_id='call-1')`.

The expected joined result shape is:

```text
Tool result 1 (read_file):
<result 1>

Tool result 2 (edit_file):
<result 2>
```

Render that as **one** `tool` segment for one contiguous result group. String results pass through unchanged; bytes decode as UTF-8 with replacement; mappings/sequences use compact, sorted JSON; other scalar values use `str`. Non-text image content raises before any new segment is appended.

Run:

```bash
uv run --extra agent pytest tests/agent/test_binding.py -q
```

Expect failure because `MessageBinder` and `append_message` do not exist.

- [x] **Step 3: Implement the initial binder**

`MessageBinder` owns bound message records, not another copy of message content. Assign `uuid4().hex` to missing `metadata['ds4.key']` values. Hash a canonical JSON projection of exactly the fields used for rendering—message kind, request instructions, and part content/tool identity/raw trusted syntax—using `ModelMessagesTypeAdapter` plus SHA-256. Exclude the key itself and volatile bookkeeping such as run/conversation IDs, timestamps, state, usage, workspace refs, history digests, and failed-attempt counters; normal harness bookkeeping must not invalidate a bound prefix. Store `key`, `content_hash`, `start`, and `end` for each message; multiple token segments may share one message range/key.

Use `PromptBuilder` for all tokenization. `Session.start_chat` creates the family system/tool prefix once. Ordinary system/instruction content is separate and plain. `ModelResponse` supplied by the harness is rendered as an assistant message and terminated with the family EOS. A contiguous group of `ToolReturnPart`/`RetryPromptPart` becomes one tool message, not one message per part.

`prepare` never calls `sync`, `run_round`, or a native session method directly. Add focused tests proving native sync/sample counts stay zero.

- [x] **Step 4: Run the tests and commit**

```bash
uv run --extra agent pytest tests/agent/test_binding.py tests/engine/test_prompt.py -q
uv run --extra agent pytest tests/engine tests/agent -m "not live and not build" -q
git add pyproject.toml uv.lock docs/specs/2026-10-21-ds4-py-design.md .github/workflows/ci.yml src/ds4py/binding.py src/ds4py/engine tests/agent
git commit -m "feat(agent): bind harness messages to chat tokens"
```

---

## Task 2: Add rewrite, count, provenance, and restored-assistant rendering

**Files:**
- Modify: `src/ds4py/binding.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/transcript.py`
- Modify: `src/ds4py/engine/prompt.py`
- Create: `src/ds4py/engine/tool_syntax/render.py`
- Test: `tests/agent/test_binding.py`
- Create: `tests/agent/test_binding_rewrite.py`
- Create: `tests/agent/test_tool_render.py`

**Interfaces:**
- `MessageBinder.count_tokens(messages, *, tools=(), instructions=None) -> int` renders against scratch transcript state, includes the next assistant prefix, and changes neither live transcript nor native KV.
- `MessageBinder.bind_response(response) -> None` tags trailing sampled assistant/EOS segments using the response key/hash and `source='sampled'`.
- `MessageBinder.export_segments() -> tuple[SegmentProvenance, ...]` returns `(start, end, kind, key, content_hash, source)` records covering the transcript without gaps.
- `Session.truncate(token_count) -> None` truncates only between rounds and drops checkpoints whose saved prefix is now beyond the transcript.
- `render_tool_call(family, name, args) -> str` produces canonical DSML/DSML41/GLM/Qwen syntax for a reconstructed assistant call.

- [x] **Step 1: Write failing rewrite and counting tests**

Add:

- `test_identical_reserialized_messages_keep_bound_prefix`
- `test_changed_message_renders_only_from_its_segment_start`
- `test_same_key_with_changed_hash_does_not_reuse_old_segment`
- `test_runtime_ids_and_timestamps_do_not_invalidate_unchanged_rendered_content`
- `test_tool_definition_change_rebuilds_from_tool_prefix`
- `test_think_change_rebuilds_system_prefix`
- `test_count_tokens_does_not_mutate_or_sync`
- `test_provenance_covers_every_token_and_marks_sampled_response`

Build the history once, deep-copy/serialize it into new Python objects, and prepare again. Assert reuse by token ranges and plan behavior, never by `id(message)`. Change an early message and assert that later equal messages are re-rendered while the unchanged prefix stays byte/token identical. Assert checkpoints beyond the cut are dropped and no later native call can restore them.

Run the focused tests and confirm they fail before implementation:

```bash
uv run --extra agent pytest tests/agent/test_binding_rewrite.py -q
```

- [x] **Step 2: Implement longest-valid-prefix rewriting**

Compare the incoming sequence to bound records in order. The first key/hash mismatch, removed message, tool signature change, instruction change, or think change selects the cut. Retain earlier segments and their binding; truncate the rest; render the replacement suffix. Do not compare object identities and do not automatically re-render all history because a new request arrived.

`Session.truncate` refuses while a round/job is active. It coordinates checkpoint invalidation through the worker and emits the existing `Dropped` event for each invalidated key. The next `plan()` must honestly choose a valid checkpoint restore or rebuild; it must not call an arbitrary rewind a valid extension.

Implement counting with a scratch `Transcript` and the same rendering code on the worker. It returns the would-be prompt token count plus one assistant prefix, with no live mutation.

- [x] **Step 3: Write failing restored-tool rendering tests**

For each family, reconstruct an assistant `ModelResponse` containing text and a `ToolCallPart`, then:

1. render it;
2. parse the resulting assistant content with the existing family parser one-shot and bytewise;
3. require the same tool name/arguments;
4. require tool markers to be special/rendered tokens and ordinary user/tool-result payloads to remain plain.

Also test that a DS4Model-sampled response can preserve raw tool syntax in `ToolCallPart.provider_details['ds4py.raw']`, and that raw text is trusted only when message metadata marks the response as `source='sampled'`.

- [x] **Step 4: Implement canonical restored-assistant rendering**

Use preserved raw syntax for a sampled response restored from the step store. Otherwise call `render_tool_call` with the part's name/args. Reuse M2 escaping and parser constants; do not create a fourth tool syntax. Add a narrow `PromptBuilder` path for trusted generated assistant markup so Qwen/GLM tool delimiters are rendered rather than plain-tokenized. Compaction summaries and ordinary external assistant text remain ordinary plain content.

`bind_response` binds the existing sampled assistant and EOS ranges. It must not call the renderer on the normal generation path.

- [x] **Step 5: Verify and commit**

```bash
uv run --extra agent pytest tests/agent tests/engine/tool_syntax tests/engine/test_prompt.py -q
uv run --extra agent pytest -m "not live and not build" -q
git add src/ds4py/binding.py src/ds4py/engine tests/agent
git commit -m "feat(agent): preserve and rewrite bound history"
```

---

## Task 3: Implement `DS4Model` and streamed responses

**Files:**
- Create: `src/ds4py/model.py`
- Modify: `src/ds4py/binding.py`
- Create: `tests/agent/test_model.py`
- Create: `tests/agent/test_model_stream.py`

**Interfaces:**
- `DS4Model(session, *, binder=None, sampling, max_tokens=50000, stop_sequences=(), note_provider=None)`.
- `model_name = session.engine.info.name`; `system = 'ds4'`; `profile` uses the actual `ModelProfile` TypedDict with `supports_thinking=True`, `supports_tools=True`, `supports_json_schema_output=False`, `supports_json_object_output=False`, `default_structured_output_mode='prompted'`, `json_schema_transformer=OpenAIJsonSchemaTransformer`, and `thinking_tags=('<think>', '</think>')`, plus the documented harness defaults. Do not use the stale `supports_json_output` field.
- `supported_native_tools = frozenset()`; `supported_tool_addition_modes` and `supported_tool_deferral_modes` remain empty.
- `DS4StreamedResponse` implements `model_name`, `provider_name`, `provider_url`, `timestamp`, and `_get_event_iterator`; `get()` remains the harness implementation.

- [x] **Step 1: Write failing model-contract tests**

Cover:

- identity/profile fields and absence of native tool support;
- `request` calling `prepare_request` and rejecting missing/non-finite concrete sampling values rather than leaving `temperature=None` in `ModelSettings`;
- a text response converted into `TextPart` and `ThinkingPart`;
- a scripted `ToolCalls` event converted into `ToolCallPart` with stable `tool_call_id` and preserved raw syntax;
- stop reasons mapped to `stop`, `length`, `tool_call`, and `error` as appropriate;
- a real `Agent` using `DS4Model` executing a harness function tool and sending the result back for a second model round.

Use `FakeNative` and the existing scripted decode helpers. Do not create a second fake token loop inside the tests.

Run:

```bash
uv run --extra agent pytest tests/agent/test_model.py tests/agent/test_model_stream.py -q
```

Expect import/abstract-method failures before the adapter exists.

- [x] **Step 2: Implement request preparation and the response stream**

`request` calls `prepare_request`, enters `request_stream`, consumes the stream, and returns `stream.get()`. `request_stream` converts the concrete policy, calls `binder.prepare`, and yields one `DS4StreamedResponse`.

The stream consumes `session.run_round(max_tokens=..., sampling=..., stop_sequences=...)`. Bridge each blocking `next()` through `asyncio.to_thread` with a sentinel helper so the asyncio loop remains responsive and `StopIteration` never crosses a Future. The facade still owns all sampling and native work.

Map events as follows:

- `TextDelta` to the parts manager's text or thinking delta handler;
- `ToolCalls` to tool-call parts, with raw syntax in provider details;
- `Token` to output-token usage and the trace/session stream;
- `Synced` to prefill usage;
- `Stopped` to the harness finish reason;
- `Interrupted` to an interrupted response, not a fabricated successful finish.

Set `response.metadata['ds4.key']` and `metadata['ds4.source']='sampled'`, then bind the sampled segments before the final response is returned. Perform this binding in the stream's terminal/finalization path so both `request` and direct `request_stream`/`Agent.run_stream_events` consumption persist the same provenance; `request` must not bind a second time. Add a direct-stream test whose next request extends rather than re-renders the sampled response.

- [x] **Step 3: Implement usage accounting and partial responses**

At round completion use `RequestUsage(input_tokens=synced.prefilled, output_tokens=<Token event count>, details={'prefill_tokens': synced.prefilled})`. In this API `requests` and `total_tokens` are computed properties, not constructor fields; assert `requests == 1` and `total_tokens == prefill + output` in tests. Do not count stop tokens or unevaluated drafts. With no prefill, input/prefill are zero.

If the stream closes early, keep already committed parts, bind their sampled ranges to the partial response, and mark the response interrupted/incomplete through the harness API. Do not append an EOS that the engine did not commit. Test empty output, thinking-only output, tool-only output, and a response interrupted after text.

- [x] **Step 4: Verify and commit**

```bash
uv run --extra agent pytest tests/agent/test_model.py tests/agent/test_model_stream.py tests/agent/test_binding.py -q
uv run --extra agent pytest tests/engine tests/agent -m "not live and not build" -q
git add src/ds4py/model.py src/ds4py/binding.py tests/agent
git commit -m "feat(agent): adapt ds4 rounds to harness streams"
```

---

## Task 4: Prove cancellation, reserve failure, and speculative alignment through the adapter

**Files:**
- Modify: `src/ds4py/model.py`
- Modify: `src/ds4py/engine/session.py` only if a cancellation/reset race is exposed
- Test: `tests/agent/test_model_stream.py`
- Create: `tests/agent/test_model_cancel.py`

- [x] **Step 1: Add failing cancellation and reserve tests**

Add:

- `test_asyncio_cancellation_cancels_facade_and_preserves_partial_response`
- `test_close_stream_cancels_without_waiting_for_another_token`
- `test_cancel_during_prefill_leaves_transcript_intact`
- `test_cancel_during_speculative_block_rewinds_uncommitted_drafts`
- `test_context_reserve_default_fails_clearly_without_compaction`
- `test_paused_round_can_resume_without_second_assistant_prefix`

Drive cancellation both while `next()` is waiting and after a committed text delta. Assert the native token list equals the transcript, uncommitted drafts never appear in parts/events, and `response.state == 'interrupted'`. Use `FakeMultiTokenStep` to force a draft block beyond the cancellation point.

For M3, the model's default reserve policy calls `ReserveReached.fail()` and surfaces a clear context-reserve exception; it must not hang waiting for M4 compaction. The existing facade pause/resume behavior remains available and is tested by explicitly resuming with the documented remaining budget.

```bash
uv run --extra agent pytest tests/agent/test_model_cancel.py -q
```

- [x] **Step 2: Implement the narrow adapter changes**

On `asyncio.CancelledError` or stream close, call `session.cancel()` immediately, drain only the events needed to reach the facade's terminal interruption, and re-apply the harness cancellation state. Never call native rewind from the asyncio thread. Do not swallow cancellation and return a complete response.

Keep the M2 stop-before-eval and rewind rules unchanged. If a race requires an engine change, add a regression at the facade level first and make the smallest worker/decode change.

- [x] **Step 3: Run focused and regression verification**

```bash
uv run --extra agent pytest tests/agent/test_model_cancel.py tests/engine/test_decode.py tests/engine/test_context_reserve.py -q
uv run --extra agent pytest -m "not live and not build" -q
git add src/ds4py/model.py src/ds4py/engine tests/agent
git commit -m "fix(agent): align cancellation with committed tokens"
```

---

## Task 5: Add session identity, lifecycle events, scoped state, and reset/fork primitives

**Files:**
- Create: `src/ds4py/engine/lifecycle.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/worker.py`
- Modify: `src/ds4py/engine/transcript.py`
- Create: `tests/engine/test_lifecycle.py`
- Create: `tests/engine/test_scoped_state.py`

**Interfaces:**
- `Session.id: str`, initialized with `uuid4().hex`.
- `Session.on_event(listener, *, kinds=None) -> Callable[[], None]`; the listener receives `SessionEvent(kind, session_id, payload)`, and the returned function unsubscribes.
- `Session.emit(kind, **payload)` for extension/status events.
- `Session.state(key, factory)` returns the same state object until its session ends.
- `Session.new() -> str`, `Session.fork() -> str`, and `Session.set_think(mode)`.
- Engine events are also published on the bus under their event class names; this is how run records, traces, and footers observe events the harness does not consume.

- [x] **Step 1: Write failing lifecycle/state tests**

Test the exact event order:

- `new`: `session_end(old)`, close states, reset, `session_start(new)`;
- `switching`: `switching(old)`, `session_end(old)`, close states, restore, `session_start(new)`, `switched(new)`;
- `fork`: `forked(old,new)` only, with no state close and no transcript/KV change.

Add a state class with a `close()` counter. Verify identity across two prompts, close on new/switch/exit, and retention on fork. Verify resetting drops transcript segments/checkpoints and synchronizes an empty native prefix, while fork causes zero native sync/prefill calls. Verify `set_think` invalidates the bound system prefix on the next request rather than mutating a live round.

Run:

```bash
uv run --extra agent pytest tests/engine/test_lifecycle.py tests/engine/test_scoped_state.py -q
```

- [x] **Step 2: Implement the lifecycle bus and reset job**

Keep lifecycle code out of the already large session module. A small `ResetJob` clears checkpoints, syncs the empty token list on the worker, resets the transcript, and leaves sampling settings unchanged. It does not close/recreate the `EngineWorker`.

Publish engine events in the facade's single event-delivery path, immediately before yielding them to the consumer, so no event is duplicated or hidden from observers. Listener errors must not corrupt native state; route them to the run record/stderr while allowing the engine event through.

Document that listeners run on the thread consuming facade events and must not call back into a blocking session method. State factories/close are main-orchestration concerns, not worker callbacks.

- [x] **Step 3: Verify and commit**

```bash
uv run --extra agent pytest tests/engine/test_lifecycle.py tests/engine/test_scoped_state.py tests/agent -q
uv run --extra agent pytest -m "not live and not build" -q
git add src/ds4py/engine tests/engine tests/agent
git commit -m "feat(engine): add session lifecycle and scoped state"
```

---

## Task 6: Add native KV payload bindings and worker save/load jobs

**Files:**
- Modify: `src/ds4py/_native/bindings.py`
- Modify: `src/ds4py/_native/api.py`
- Modify: `src/ds4py/_native/protocol.py`
- Create: `src/ds4py/_native/payload.py`
- Modify: `src/ds4py/_native/fake.py`
- Modify: `src/ds4py/engine/worker.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/events.py`
- Create: `tests/native/test_payload.py`
- Create: `tests/engine/test_payload_jobs.py`
- Update existing ABI binding fixtures only where the new required signatures require it

**Interfaces:**
- `SessionHandle.stage_payload() -> StagedPayload`; `StagedPayload.write(file)`, `StagedPayload.close()`.
- `SessionHandle.load_payload(file, payload_bytes) -> None`.
- `Session.save_payload(file, *, header: bytes) -> Saved` syncs first, writes the header and staged payload on the worker, and emits `Saved(tokens, payload_bytes)`.
- `Session.load_payload(file, *, payload_bytes) -> list[int]` loads on the worker, adopts the native token list, and emits `Loaded(tokens, payload_bytes)`.

- [x] **Step 1: Write failing fake round-trip and ownership tests**

Test:

- staged save/load restores token IDs and position without tokenizing text;
- save synchronizes a trailing EOS before staging;
- load replaces the transcript token list and permits a following extension;
- zero-length payload is distinguishable from missing payload;
- truncated/corrupt fake payloads raise and never silently produce an empty session;
- staged resources close on success and exceptions;
- Python file objects remain usable after native-style write/load helpers release their duplicated descriptors.

Run:

```bash
uv run --extra agent pytest tests/native/test_payload.py tests/engine/test_payload_jobs.py -q
```

- [x] **Step 2: Add required ABI bindings and native wrappers**

Take exact prototypes from the pinned `ds4.h`; one signature probe per binding remains the build-time gate. Do not mark these optional. Treat staged payload pointers as opaque unless the header requires a public struct layout.

Implement `fdopen(os.dup(...))` with mode `wb`/`rb`, flush the Python file before native I/O, and `fclose` only the duplicated stream. On any non-success native return, include the native error buffer in `EngineError`. Free staged payloads in `finally`.

The fake payload can be a versioned binary token encoding; its purpose is behavioral round-trip/corruption testing, not matching the C byte format.

- [x] **Step 3: Add `SavePayloadJob` and `LoadPayloadJob`**

`SavePayloadJob` performs the normal sync/progress path before staging, so live KV equals the transcript including trailing EOS. It writes the caller-supplied 20-byte header followed by exactly the declared payload bytes. `LoadPayloadJob` expects the file positioned after the header and reads exactly `payload_bytes`.

After load, take tokens from `SessionHandle.tokens()`, not from file text or re-tokenization. Restore transcript chat mode and replace segments initially with one loaded range; Task 9 applies the save record's detailed provenance.

- [x] **Step 4: Verify ABI and behavior, then commit**

```bash
uv run --extra agent pytest tests/native/test_payload.py tests/engine/test_payload_jobs.py -q
uv run --extra agent pytest tests/native tests/engine -m "not live and not build" -q
uv run ds4py check --lib "${DS4_LIB:-$HOME/.ds4/lib/libds4.so}"  # when a built library is available
git add src/ds4py/_native src/ds4py/engine tests/native tests/engine
git commit -m "feat(engine): stage and restore native KV payloads"
```

Do not claim the real-library check ran if no library is available.

---

## Task 7: Implement the stdlib KV store, save records, and failure validation

**Files:**
- Create: `src/ds4py/store.py`
- Modify: `src/ds4py/errors.py`
- Create: `tests/agent/test_store.py`
- Create: `tests/agent/test_store_failures.py`

**Interfaces:**
- `KV_HEADER = struct.Struct('>8sIQ')`, `KV_MAGIC = b'DS4PYKV\0'`, `KV_VERSION = 1`.
- `SaveRecord` and `SegmentProvenance` have explicit JSON round trips. The record contains format version, `session_id`, `conversation_id`, `run_id`, `step_index`, `model_id`, `quant_bits`, `ctx`, `title`, `created_at`, `last_used`, `payload_bytes`, `tokens`, `tokens_sha256`, and `segments`.
- `SessionStore(root)` resolves the default root from `DS4PY_HOME` or `~/.ds4py`, and exposes `session_dir`, `write_kv_atomic`, `read_kv`, `write_record_atomic`, `read_record`, `find`, and `validate_for_engine`.
- `token_digest(tokens) -> str` uses the fixed big-endian uint32 encoding.

- [x] **Step 1: Write failing format and validation tests**

Use behavior round trips, not static JSON/YAML assertions:

- header/payload and save record round trips preserve all fields;
- prefix lookup resolves a unique ID and rejects unknown/ambiguous prefixes;
- model ID and quant mismatch are errors, not warnings;
- bad magic/version, truncated header, truncated payload, extra undeclared payload data, malformed record JSON, invalid token counts, and non-covering segments produce path-specific `StoreError`s;
- an interrupted write leaves the prior KV/record intact;
- if snapshot writing fails after KV replacement, no new save record appears and the old record remains available for the specified mismatch fallback.

Run:

```bash
uv run --extra agent pytest tests/agent/test_store.py tests/agent/test_store_failures.py -q
```

- [x] **Step 2: Implement atomic files and strict validation**

Write temporary files in the destination directory, flush and `fsync`, then `os.replace`. Read exactly the declared payload length and reject truncation. `save.json` is written last by the runtime; the store provides the primitive and fault-safe behavior, not cross-module orchestration.

`find` scans only `sessions/*/save.json` and returns a record plus path. `validate_for_engine` compares `model_id` and `routed_quant_bits` against `EngineInfo`. Keep harness imports out of this module so core `ds4py.store` works without the agent extra.

- [x] **Step 3: Verify and commit**

```bash
uv run --extra agent pytest tests/agent/test_store.py tests/agent/test_store_failures.py -q
uv run --isolated --no-project --with . python -c "import ds4py.store"
git add src/ds4py/store.py src/ds4py/errors.py tests/agent
git commit -m "feat(store): add atomic ds4-py save records"
```

---

## Task 8: Add system notes, reminders, approvals, and agent assembly

**Files:**
- Create: `src/ds4py/capabilities/__init__.py`
- Create: `src/ds4py/capabilities/system_notes.py`
- Create: `src/ds4py/capabilities/approvals.py`
- Create: `src/ds4py/capabilities/run_record.py`
- Create: `src/ds4py/capabilities/shell_jobs.py`
- Create: `src/ds4py/agent.py`
- Modify: `src/ds4py/model.py`
- Modify: `src/ds4py/engine/prompt.py`
- Modify: `src/ds4py/engine/chat.py`
- Create: `tests/agent/test_system_notes.py`
- Create: `tests/agent/test_approvals.py`
- Create: `tests/agent/test_shell_jobs.py`
- Create: `tests/agent/test_agent_factory.py`

**Interfaces:**
- `DS4SystemNotes(session, *, clock=None, hints=False)` implements the note-provider protocol used by `MessageBinder` plus `before_model_request` for compaction-subject discovery. `restore_from_segments(segments)` restores date/hints/reminder state after a valid KV load.
- `Session.append_system_note` accepts an optional note subtype. Save provenance uses kinds such as `system_note:date`, `system_note:hints_full`, `system_note:hints_on`, `system_note:hints_off`, and `system_note:reminder`; the default remains `system_note` for existing callers.
- `DS4Approvals(answerer, *, yolo=False, project_dir)` is a capability, not a TUI policy class. `answerer` may be sync or async.
- `ApprovalRequest(tool_name, args, command=None, path=None)` and `ApprovalDecision` (`ONCE`, `DENY`, `TOOL`, `PATTERN`, `STOP`) define the callback boundary.
- `DS4RunRecord(session, path)` subscribes to engine events and implements harness run/tool hooks.
- `DS4ShellJobs(session, workspace)` records `start_command` job IDs in scoped state and stops running jobs when that state closes.
- `build_agent_components(...) -> AgentComponents` constructs the default capabilities/toolset and returns the agent plus references needed by runtime/TUI.

- [x] **Step 1: Write failing system-note tests**

Port the exact date/hints text from pinned `ds4_agent.c:1478–1625`; use an injected clock for deterministic tests. Cover:

- one date note before the first user message only;
- hints off by default;
- enabling hints sends the full instructions once, then only a short note when the setting changes;
- disabling sends the short change note;
- a compaction notification resets hint/full-note state;
- note segments have empty message keys and do not become extra harness messages.

Do not implement the M4 compaction capability. Use a stub compaction subject to prove `before_model_request` discovers it and subscribes once.

- [x] **Step 2: Implement date and hints notes**

The binder asks the note provider before a new user request and before an assistant continuation after tool results. Notes append through `append_system_note` and never through rendered user content. Do not add the M4 reminder renderer or its 50,000-token trigger in this task.

On case-1 restore, scan the saved note subtypes so the next request does not emit a second date note or lose the hints setting. On case-2 fresh rendering, emit the normal first-request notes once. A compaction reset starts the hint lifecycle again. Actual `DS4Compaction` and the system-prompt reminder remain M4.

- [x] **Step 3: Write failing approval tests with real harness normalization**

Use a `FunctionModel` or equivalent scripted harness model plus a real `Agent` and toolset. Test:

- `run_command` and `start_command` ask unless `yolo`;
- `write_file`, `edit_file`, and `create_directory` ask only for resolved paths outside `project_dir`, unless `yolo`;
- `yolo` bypasses approval but `LocalWorkspace(read_only=True)` still refuses writes and commands;
- `DENY` raises `SkipToolExecution(ToolDenied(...))`, the tool body is never entered, and history contains `ToolReturnPart(outcome='denied')`;
- `ONCE`, per-`TOOL`, and per-`PATTERN` decisions have distinct scope;
- `STOP` denies and cancels the active session;
- a scripted answer sequence `y, n, t, a, s` exercises all five menu inputs.

Map menu keys as `y=ONCE`, `n=DENY`, `t=TOOL`, `a=PATTERN`, `s=STOP`. A command pattern is its first `shlex.split` word; a file pattern is the resolved parent directory. Noninteractive EOF denies safely with an actionable message.

- [x] **Step 4: Write failing shell-job lifecycle tests, then implement cleanup**

In `tests/agent/test_shell_jobs.py`, use the real harness shell/workspace seam and short-lived test commands. Prove a command started with `start_command` in one prompt is visible to `check_command` in the next, and that closing the scoped state on `/new`, `/switch`, and exit stops every recorded command that is still running. Already-exited jobs need no action.

Implement `DS4ShellJobs` as a small capability that records job IDs from tool hooks in `session.state`. Stop jobs through the pinned harness's supported `stop_command`/workspace interface, or through the process group recorded in its handle file as section 12 permits; do not depend on its private `Job` objects or touch the engine. Run the focused tests red before implementation and green afterward.

- [x] **Step 5: Implement approvals and agent assembly**

`before_tool_execute` returns validated args unchanged on approval. On denial it raises `SkipToolExecution` with `ToolDenied`; do not convert denial into `ModelRetry`, a successful string result, or a TUI-only retry part. Resolve paths with `Path.resolve()` and test `is_relative_to(project_dir)`; do not approve by string prefix.

Construct `LocalWorkspace(project_dir, read_only=read_only)` from `pydantic_ai.capabilities`, `FileSystem(root_dir=project_dir, cwd=project_dir, content_hashes=True)`, `Shell(cwd=project_dir)`, `DS4ShellJobs(...)`, `RepoContext(workspace_dir=project_dir)`, `Skills(directories=<standard project/user skill directories>, missing_directories='skip')`, `ToolOutputLimits(...)`, `StepPersistence(store=step_store)`, `Instrumentation()`, notes, and approvals. Keep `FileSystem.read_only` false so write tools remain visible; the `LocalWorkspace` read-only backend is the enforcement boundary for both file writes and shell commands. `ToolOutputLimits` must not call the model. Add interactive-only `AskUser(answerer=tui_answerer)` with no approvals argument. Record filesystem/tool changes through `StepPersistence` and `DS4RunRecord`; there is no separate `FileChangeTracker` in this API. Keep `DS4Compaction` out. Verify a tiny agent run can read the workspace and writes events/snapshots; do not write a test that merely compares the static capability list.

- [x] **Step 6: Verify and commit**

```bash
uv run --extra agent pytest tests/agent/test_system_notes.py tests/agent/test_approvals.py tests/agent/test_shell_jobs.py tests/agent/test_agent_factory.py -q
uv run --extra agent pytest tests/agent tests/engine -m "not live and not build" -q
uv run --extra agent clai --help  # CLI entry-point smoke only; Task 3 covers real DS4Model behavior
git add src/ds4py/capabilities src/ds4py/agent.py src/ds4py/model.py src/ds4py/engine tests/agent
git commit -m "feat(agent): add notes approvals and default capabilities"
```

---

## Task 9: Add `AgentRuntime`, step persistence, and save/switch/new/fork

**Files:**
- Create: `src/ds4py/runtime.py`
- Modify: `src/ds4py/binding.py`
- Modify: `src/ds4py/store.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/transcript.py`
- Create: `tests/agent/test_runtime.py`
- Create: `tests/agent/test_restore.py`
- Create: `tests/agent/test_run_record.py`

**Interfaces:**
- `MessageBinder.restore(messages, segments) -> None` rebinds matching snapshot messages to loaded token ranges and verifies every keyed segment.
- `AgentRuntime.messages` is the current canonical harness history; `session_id`, `title`, and `status` are read-only views.
- `AgentRuntime.load_prefix(turns)` converts prefix-file turns into initial request/response messages so stripped restores can rebuild them.
- `AgentRuntime.run`, `save`, `switch`, `new`, `fork`, and `close` follow the signatures in the architecture section.

- [x] **Step 1: Write failing run/persistence tests**

Test two sequential prompts and a tool round. After each step, inspect `FileStepStore` snapshots and require an exact `messages` round trip, run/conversation/step IDs, usage, workspace ref, and tool effects where supplied by the harness. Assert `events.jsonl` contains valid JSON lines for run start/end, sync/prefill/token/tool events, and terminal statuses `success`, `interrupted`, and `error`.

Make one scripted run fail and require the error status and latest interrupted/error snapshot to remain loadable. Do not replace `StepPersistence` with custom JSON message files.

Run:

```bash
uv run --extra agent pytest tests/agent/test_runtime.py tests/agent/test_run_record.py -q
```

- [x] **Step 2: Implement runtime runs and run records**

Create a small session-scoped `SessionStepStore` holder whose current delegate is `FileStepStore(sessions/<session id>/steps)`. Keep the holder in `session.state`; this preserves the state object's identity on fork while its delegate moves to the new session directory and old files stay untouched. For each run, install `StepPersistence(store=holder.current)` through `Agent.run(capabilities=[...])`, and pass `conversation_id` and a fresh `run_id` directly to `Agent.run`; the current API has no `Agent.run(store=...)` argument. Start one `Agent.run` per user prompt and update runtime history from the result or from the latest loadable snapshot on failure. The holder is only a locator, never a second message store.

`DS4RunRecord` appends and flushes one JSON object per event with timestamp, session/run IDs, event kind, and JSON-safe payload. Encode token bytes explicitly (base64 or hex) rather than using `repr`. Listener/record failures surface as run errors; they must not leave a false `success` record.

- [x] **Step 3: Write failing save/switch tests**

Cover:

1. **Case 1, valid KV:** save, create a fresh runtime/engine against the same store, `/switch` by unique prefix, and require zero prefill during switch. Take token IDs from native load, verify count/hash against the record, load the exact named snapshot/step, and match every keyed segment by key and content hash. The next user request plans as an extension.
2. **Case 2, no/mismatched payload:** copy a save with a zero-payload header, then switch. Load the named snapshot and render fresh on the next request; require a full prefill and deterministic output equal to case 1 for the fake scripted model. Use the same injected clock in both runtimes and assert that the rebuilt pre-follow-up token list equals the saved list for this round-trip fixture, so the fake script cannot hide a rendering mismatch. A crash-like newer KV with old record takes the same fallback and reports the mismatch.
3. Missing named snapshot, missing segment key, wrong content hash, model mismatch, and corrupt KV fail with the required named path/key/run.
4. `/save` writes KV, then a `complete` snapshot in `<session id>.save-<n>`, then the record. The title is whitespace-normalized first user prompt, truncated to 80 characters.
5. `/new` preserves old files but clears live history/KV/state. `fork` keeps live history/KV/binding/state and old files, while subsequent snapshots/saves use the new session directory.
6. Exit closes step stores and other scoped state.

Run:

```bash
uv run --extra agent pytest tests/agent/test_restore.py -q
```

- [x] **Step 4: Implement save and both restore paths**

Before saving, cancel/wait for active work, sync through `save_payload`, and hash the actual saved token list. Determine the next save number from existing `<session id>.save-<n>` runs in the conversation. Save a complete snapshot with the live message list even when the previous run was interrupted or failed; write `save.json` last.

On valid load, apply `SegmentProvenance` to the loaded transcript and call `binder.restore`. Generated system/tool/note ranges may have empty keys; every keyed range must match a snapshot message by stable key/hash. Do not render or prefill case 1.

On missing/mismatched payload, reset to an empty chat transcript, retain the restored messages, and let the binder render them fresh. No model text is generated during switch itself. Record `switched` with the save record and which restore path was used.

- [x] **Step 5: Verify and commit**

```bash
uv run --extra agent pytest tests/agent/test_runtime.py tests/agent/test_restore.py tests/agent/test_run_record.py -q
uv run --extra agent pytest -m "not live and not build" -q
git add src/ds4py/runtime.py src/ds4py/binding.py src/ds4py/store.py src/ds4py/engine tests/agent
git commit -m "feat(agent): persist and restore harness sessions"
```

---

## Task 10: Add non-interactive execution and ds4-py prefill traces

**Files:**
- Modify: `src/ds4py/cli.py`
- Create: `src/ds4py/noninteractive.py`
- Create: `src/ds4py/prefill_trace.py`
- Modify: `src/ds4py/engine/replay.py`
- Create: `tests/agent/test_noninteractive.py`
- Create: `tests/agent/test_prefill_trace.py`

**Interfaces:**
- `run_noninteractive(runtime, prompt) -> int` prints the final result to stdout, sends diagnostics to stderr, saves in `finally`, and closes scoped state/engine through the caller.
- `PrefillTraceRecord(session_id, run_id, step_index, prompt, round_index, think, cached, prefill_suffix, sampled)` has an append-only JSONL writer.
- The replay reader accepts ds4-py trace records and produces the same replay-round representation used for C traces.

- [x] **Step 1: Write failing CLI and trace tests**

Use a fake runtime/engine; do not open a model. Verify:

- `--non-interactive` requires `--prompt` or `--prompt-file` and runs one prompt;
- final output goes to stdout and status/progress does not contaminate it;
- save is attempted on success, interruption, and error;
- `--temp 0 --seed 1 --ctx 100000 --tokens ...` maps to `SamplingPolicy` and runtime settings;
- `--system` and `--prefix-file` become persistent instructions/history;
- `--read-only` and `--yolo` reach the workspace/approval components;
- raw/build/check commands still route to their existing implementations without importing the harness.

Accept the section-6 agent flags: `--prompt`, `--prompt-file`, `--system`, `--prefix-file`, `--non-interactive`, `--raw` as the raw-mode alias, `--temp`, `--top-p`, `--min-p`, `--seed`, `--think`, `--think-level`, `--think-max`, `--nothink`, `--trace`, `--tokens`, and `--ctx`, plus existing M1 engine options and M3 `--yolo`/`--read-only`. `--edit-upto` stays absent. Vision/steering/distributed engine options remain with their later milestones rather than gaining no-op flags.

For traces, run one text round and one tool round. Read the JSONL back through the replay reader and compare prompt tokens, cached count, prefill suffix, sampled IDs, round index, and think mode. Verify two runs append rather than overwrite and a tool trace never executes a tool.

- [x] **Step 2: Implement the runner and trace writer**

`run_noninteractive` awaits one runtime run, prints `result.output`, and calls `runtime.save()` in `finally`. If both run and save fail, report both errors and return nonzero. Do not print intermediate text a second time.

The trace writer subscribes before the run. At each `Synced`, capture the complete prompt, `plan.reused` as `cached`, and the actual synchronized suffix. Accumulate committed `Token.id` values until the round terminal event, then append one record with the current run/step/round identity. `prefill_suffix` is tokens, not text. No unevaluated drafts or stop tokens enter `sampled`.

Extend `ds4py.engine.replay` with a strict ds4-py record parser; malformed lengths (`cached + len(prefill_suffix) != len(prompt)` for an extension) and missing fields fail loudly. Replay configures parsers but has no executable toolset.

- [x] **Step 3: Verify and commit**

```bash
uv run --extra agent pytest tests/agent/test_noninteractive.py tests/agent/test_prefill_trace.py tests/engine/test_replay.py -q
uv run --extra agent pytest -m "not live and not build" -q
uv run --isolated --no-project --with . python -c "import ds4py; import ds4py.cli; import ds4py.store"
git add src/ds4py/cli.py src/ds4py/noninteractive.py src/ds4py/prefill_trace.py src/ds4py/engine/replay.py tests/agent
git commit -m "feat(agent): add noninteractive runs and prefill traces"
```

---

## Task 11: Add the prompt-toolkit TUI, command registry, renderer, and footer

**Files:**
- Create: `src/ds4py/tui.py`
- Create: `src/ds4py/tui_commands.py`
- Create: `src/ds4py/tui_render.py`
- Modify: `src/ds4py/cli.py`
- Modify: `src/ds4py/runtime.py`
- Modify: `src/ds4py/capabilities/run_record.py`
- Create: `tests/agent/test_tui_commands.py`
- Create: `tests/agent/test_tui.py`
- Create: `tests/agent/test_footer.py`

**Interfaces:**
- `SlashCommandRegistry.register(name, handler, *, description, usage=None)`, `execute(line)`, and `completions(prefix)` form the extension seam.
- `TUI(runtime, input=None, output=None)` owns the async prompt loop; it does not call `Engine`, `SessionHandle`, or native functions.
- `StreamRenderer` displays text/thinking/tool previews; `FooterModel` subscribes to session events and renders plan/context/prefill/service/extension status.

- [x] **Step 1: Write failing command and lifecycle tests**

Cover `/exit`, `/new`, `/save`, `/switch <prefix>`, and `/hints on|off`. Test unknown commands, missing arguments, errors that leave the loop usable, `/` completion, and extension registration of a new command. `/list`, `/strip`, `/del`, `/compact`, `/fork`, `/think`, `/role`, and `/ctx` are not pre-registered in M3; `/ctx` is intentionally the repository feature task in the bake-off, and interactive `/think` remains M5.

Use a fake runtime to prove `/new` and `/switch` close old state and `/switch` renders the returned save record. Do not test only parser configuration; execute the handlers.

- [x] **Step 2: Implement the registry and prompt loop**

Use `PromptSession` with prompt-toolkit key bindings and completion. Enter submits; Alt+Enter inserts a newline; pasted multiline text is preserved. Ctrl+C calls `runtime.cancel()` and does not exit or discard partial text; Ctrl+D exits cleanly. At idle, Ctrl+C is a no-op rather than a process interrupt.

Run one `Agent.run` per prompt through `AgentRuntime`. Save in the noninteractive runner as Task 10 specifies; interactive `/save` is explicit. `/new` and `/switch` wait for active work cancellation to settle before changing identity.

- [x] **Step 3: Write failing rendering/footer tests**

Test:

- text and thinking deltas are visually distinct, with thinking dim;
- tool calls/results have one-line previews and long values are ellipsized;
- Markdown headings, emphasis, and code spans become styled prompt-toolkit fragments;
- status lines are not appended to conversation output;
- a several-hundred-character cwd and narrow terminal leave a one-line footer with no exception or wrap;
- `Prefill` updates done/total/TPS, `Synced` updates plan/ctx, service events update PID/status, and extension `status` events update a custom key;
- plan/ctx/status is live during generation and returns to idle after terminal events.

Use prompt-toolkit pipe input and dummy output, or renderer/fake-runtime seams; no test requires a real terminal.

- [x] **Step 4: Implement rendering and the footer**

Keep Markdown/fragment construction in `tui_render.py` and layout/status state in `FooterModel`; keep `tui.py` focused on event orchestration. Use formatted-text fragments and terminal-width-aware truncation. `DS4RunRecord` republishes harness service-started/stopped/status events onto the session bus so the footer can show the shell service PID without knowing harness internals.

The main thread renders model events. The worker never calls prompt-toolkit. No footer update mutates the transcript or calls a native method.

- [x] **Step 5: Verify and commit**

```bash
uv run --extra agent pytest tests/agent/test_tui_commands.py tests/agent/test_tui.py tests/agent/test_footer.py -q
uv run --extra agent pytest -m "not live and not build" -q
git add src/ds4py/tui.py src/ds4py/tui_commands.py src/ds4py/tui_render.py src/ds4py/cli.py src/ds4py/runtime.py src/ds4py/capabilities tests/agent
git commit -m "feat(tui): add interactive harness agent shell"
```

---

## Task 12: Add the M3 conformance slice and audit the public seams

**Files:**
- Create: `tests/agent/test_conformance.py`
- Modify: `tests/engine/test_session.py` only if the existing extension fixture needs a facade-level call
- Modify: public M3 docstrings where worker/thread confinement is unclear
- Modify: implementation only when a failing conformance test exposes a defect

- [x] **Step 1: Add conformance item 1 — binding and rewriting**

Rewrite the harness request view with reserialized message objects and then change an early message. Assert the common token/bound prefix is retained and only the suffix from the first changed segment is rendered. Include a checkpoint that remains valid before the cut and one invalidated after it. Do not use object identity as evidence.

- [x] **Step 2: Add conformance item 6 — extension command and status**

Register a stub slash command through `SlashCommandRegistry`, emit a custom status key through the session bus, execute the command, and render the footer. Assert both extension effects are visible without modifying built-in command/renderer source.

- [x] **Step 3: Add conformance item 7 — declared native extension and worker call**

Reuse the M1 `_ds4py_declare` fixture. Declare the probe binding, load the extension through the existing manifest/loader path, and call it through `session.on_worker`. Assert the call executes on the engine worker and the facade result is returned. Do not bypass the ABI manifest or call the raw handle from the test thread.

- [x] **Step 4: Add conformance item 9 — scoped state lifecycle**

Use a close-tracking state and a real `FileStepStore`. Prove `session.state(key, factory)` returns one object across prompts; `/new`, `/switch`, and exit close states; `AgentRuntime.fork()` keeps the same state objects and live KV while changing IDs and future storage paths.

Do not mark conformance items 2–5 complete here. M4 owns the full deferral, resume, and compaction/restored-binding conformance tests.

- [x] **Step 5: Run conformance and full regressions, then commit**

```bash
uv run --extra agent pytest tests/agent/test_conformance.py tests/engine/test_session.py -q
uv run --extra agent pytest -m "not live and not build" -q
git add tests/agent/test_conformance.py tests/native src/ds4py
git commit -m "test(agent): cover M3 conformance seams"
```

---

## Task 13: Pin and stage the 12 bake-off tasks with hidden grading

**Files:**
- Create: `bench/__init__.py`
- Create: `bench/m3.yaml`
- Create: `bench/m3.py`
- Create: `bench/prepare_m3.py`
- Create: `bench/patches/glm-off-by-one.patch`
- Create: `bench/graders/test_ctx_command.py`
- Modify: `.gitignore` for `bench/results/`
- Create: `tests/bench/__init__.py`
- Create: `tests/bench/test_m3_workspace.py`

- [x] **Step 1: Write the manifest with fixed source values**

Pin the repository/commit/seed and the ten exercise paths under `python/exercises/practice/` in the order listed in the architecture. Add:

- `repo-glm-close`: copy the current repository, apply `glm-off-by-one.patch`, keep the focused GLM regression test visible, and ask the agent to fix the straddled-close boundary defect;
- `repo-ctx-command`: copy the current repository with no feature patch and ask for `/ctx` that reports used and remaining context without generating or mutating the session.

The `/ctx` grader stays outside the agent workspace. Its behavioral contract is: a fake runtime with `ctx_used=123` and `ctx=1000` executes `/ctx`, reports `123` used and `877` remaining, appears in `/` completion, and causes no model/native generation call.

Use model/context/sampling values `ds4f-q2`, `ctx=100000`, `temperature=0`, `seed=1`, and `max_attempts=2`. The manifest is data; validate it by running preparation, not with a static-content test.

- [x] **Step 2: Write failing workspace/grading tests**

Use a tiny synthetic exercise and repository copy to prove:

- instructions and starter source are visible;
- exercise test files and `/ctx` grader are absent during the agent phase;
- grading mounts tests temporarily, runs them, captures output, and removes them before the next agent attempt;
- a first failure's output is passed to attempt two, with at most two attempts;
- each agent gets a separate initial workspace copy while C and ds4-py receive equivalent files;
- preparing at the wrong benchmark commit fails clearly.

Run:

```bash
uv run --extra agent pytest tests/bench/test_m3_workspace.py -q
```

- [x] **Step 3: Implement staging and hidden grading**

`prepare_m3.py` resolves `POLYGLOT_BENCHMARK`, verifies the pinned commit, and materializes the ten exercise workspaces. Copy `.docs/instructions.md` to a visible `TASK.md`, retain starter source, and hold `*_test.py`/test files in a separate grader directory. Do not fetch or select random tasks on each run.

For repository tasks, use `git archive HEAD` so the task tests the code from the current M3 branch rather than an installed release. Keep the visible GLM test at its current path, `tests/engine/tool_syntax/test_glm.py`; the older test name in the spec referred to the pre-M2 layout.

Create the off-by-one patch against the actual GLM close matcher. Prove the patch behavior directly in a temporary copy: the focused GLM test fails with the patch and passes without it. Do not weaken the test to accommodate the planted defect.

- [x] **Step 4: Verify and commit**

```bash
uv run --extra agent pytest tests/bench/test_m3_workspace.py -q
uv run --extra agent python bench/prepare_m3.py --manifest bench/m3.yaml --check-only  # requires POLYGLOT_BENCHMARK
uv run --extra agent pytest tests/engine/tool_syntax/test_glm.py -q
git add bench tests/bench .gitignore
git commit -m "test(bench): pin and stage M3 bakeoff tasks"
```

Report `--check-only` as not run if the benchmark checkout is absent.

---

## Task 14: Run agents sequentially and compute per-task metrics and gates 1–5

**Files:**
- Create: `bench/run_m3.py`
- Create: `bench/metrics.py`
- Create: `bench/report.py`
- Create: `bench/restore_check.py`
- Create: `tests/bench/test_m3_metrics.py`
- Create: `tests/bench/test_m3_runner.py`
- Create: `tests/bench/test_m3_restore.py`

**Interfaces:**
- `TaskRunResult(task_id, agent, passed, attempts, wall_seconds, progress_events, tokens, tool_calls, avg_prefill_reuse, non_extend_rounds, save_dir, trace_path)`.
- `GradeResult(passed, output)` and `RunReport` include both attempts and grader output without dumping full logs into the report summary.
- `build_report(task_results, restore_results, replay_results=None) -> BakeoffReport` emits JSON and a compact table with all required fields and six gate statuses.

- [x] **Step 1: Write failing metric tests**

From synthetic event/trace fixtures, compute:

- pass and attempts from hidden grading;
- agent wall seconds summed across attempts, excluding setup/grading;
- progress event count;
- committed output token count;
- tool-call count;
- average prefill reuse, `plan.reused / (plan.reused + plan.to_prefill)`, excluding only a zero denominator;
- non-extension rounds, with an explicit exception reason for compaction/strategy rather than silently dropping events.

Require internally consistent values: nonnegative counts, reuse in `[0,1]`, attempts in `{1,2}`, and complete metrics for every finished run. Test malformed/missing events fail the relevant gate instead of producing zeros that look valid.

- [x] **Step 2: Write failing runner and gate tests with fake executables**

Use fake agent scripts and temporary tasks; no GPU/model. Prove:

- C and ds4-py never run concurrently;
- both receive the same model/backend/ctx/temp/seed/task parameters;
- ds4-py uses `--yolo` except for the approval run;
- a failed first attempt is followed by exactly one second attempt containing the grader output;
- every ds4-py attempt saves in `finally` and leaves events/snapshots/KV/record paths in the result;
- gate 1 requires ds4-py's pass count to meet C's with one task of slack, and reports pass/attempt evidence separately from timing;
- gate 2 requires zero `parser_error` stops and accounts for every parsed tool call as executed, refused by policy, or sent back as a retry; bad-argument retries are counted but do not fail the gate;
- gate 3 enforces both the spec's per-task `1.5x` C wall-time limit and extension-only KV reuse, allowing only a recorded compaction/strategy exception for a non-extension round;
- gate 4 interrupts one run during prefill and resumes it with the next prompt, and drives one approval run that continues after `y`;
- gate 5 requires complete records/files and passing case-1/case-2 restore checks for every ds4-py run.

The report also prints absolute and delta values for all six comparison metrics on every common task. These are diagnostics, not gate 2: the spec defines no numeric tolerance for different harness/C tool histories, so do not invent an equality threshold. Missing or inconsistent metric data still fails the relevant evidence check rather than becoming a plausible zero.

- [x] **Step 3: Implement the sequential runner, interrupt/resume driver, and approval PTY**

Run tasks sequentially and close each agent process before starting the next. Set `DS4PY_HOME` per attempt so records cannot overwrite one another. Use the same absolute model and backend for both agents. C receives a fresh `--trace` path per attempt. ds4-py receives `--trace`, `--temp 0`, `--seed 1`, `--ctx 100000`, and the same token limit.

For gate 4's interruption case, drive Ctrl+C through a stdlib `pty`/`select` helper while one ds4-py run is prefilling, then submit the next prompt and require the run to resume successfully. Record the interruption and resumed terminal status as evidence.

For gate 4's approval case, use the `repo-glm-close` task: run ds4-py without `--yolo`, drive the noninteractive approval prompt through the PTY helper, answer `y` to the first command approval, and require the tool/run to continue and the hidden grader to pass. EOF before approval denies and fails the gate; never auto-approve merely because the driver is unattended.

- [x] **Step 4: Implement restore checks**

After every ds4-py run, verify the save record, complete snapshot, events, and KV file. Start a **new ds4-py process** and drive `/switch <id>` through the public interactive command path, then submit the fixed follow-up:

```text
Continue the previous task: state the next concrete action, then stop.
```

At `--temp 0`, compare up to the first 256 committed token IDs against a stripped copy rebuilt from the same record/snapshot. If both stop before 256, compare the complete shorter sequence and terminal reason. Case 1 must emit no prefill during switch and extend on the follow-up; case 2 must prefill. The bench helper creates the stripped copy by copying the save directory and writing a zero-payload header; it does not add the M4 `/strip` command.

- [x] **Step 5: Verify and commit**

```bash
uv run --extra agent pytest tests/bench/test_m3_metrics.py tests/bench/test_m3_runner.py tests/bench/test_m3_restore.py -q
uv run --extra agent pytest tests/bench tests/agent -m "not live and not build" -q
git add bench tests/bench
git commit -m "test(bench): measure M3 outcomes and restore gates"
```

---

## Task 15: Add gate 6 replay and the explicit live bake-off entry point

**Files:**
- Create: `bench/replay_m3.py`
- Create: `bench/README.md`
- Create: `tests/live/test_m3_bakeoff.py`
- Modify: `bench/run_m3.py`
- Modify: `bench/report.py`
- Test: `tests/bench/test_m3_runner.py`

- [x] **Step 1: Write failing replay-selection tests with synthetic traces**

Require gate 6 to select every round from every C `--trace` file, generate at temperature 0, and reproduce the complete recorded committed-token sequence and terminal boundary. Do not cap C replay at 16 tokens or pad a round that ended naturally. A tool-call round must parse/stop without executing any tool. Fail if the full 12-run C set contains no tool-call round to replay.

Reuse the existing C replay reader/runner directly; do not translate C traces into another fixture format or substitute ds4-py self-replay for C parity. Test malformed C records and a token mismatch produce a failing gate with task/round/index details.

- [x] **Step 2: Implement replay without tools**

Load the model once for the sequential C replay phase and reuse the M2 replay machinery. For each C record, rebuild the exact prompt tokens, sync a fresh transcript/session, and generate the full recorded round with its think mode and temperature 0. Compare committed IDs and the terminal boundary only. Use an empty executable toolset and a guard that fails if any tool execution is requested.

Use the same absolute GGUF and the M2 parity rules for every C trace. Close the engine before any later replay phase starts. Write per-round C replay results into the same six-gate report.

- [x] **Step 3: Document and guard the live run**

`bench/README.md` records:

- the approximately 81 GB `ds4f-q2` model and 96 GB RAM assumption;
- the pinned benchmark checkout/commit;
- the same-model/backend requirement;
- no concurrent agent/replay processes;
- commands to prepare, run, restore-check, and replay;
- the report location and the distinction between non-live tests and live evidence.

Add `tests/live/test_m3_bakeoff.py`, marked `live`, which skips unless `DS4_RUN_M3_BAKEOFF=1`, `DS4_TEST_MODEL`, `DS4_AGENT`, and `POLYGLOT_BENCHMARK` are all set. It invokes the complete runner and asserts all six gates pass in the written report. The normal live marker alone must not accidentally launch this long suite.

Example:

```bash
export DS4_TEST_MODEL=/absolute/path/ds4f-q2.gguf
export DS4_AGENT=/absolute/path/ds4-agent
export POLYGLOT_BENCHMARK=/absolute/path/polyglot-benchmark
export DS4_RUN_M3_BAKEOFF=1
uv run --extra agent python bench/prepare_m3.py --manifest bench/m3.yaml --check-only
uv run --extra agent pytest tests/live/test_m3_bakeoff.py -m live -vv -s
```

The runner uses `--yolo` for all normal ds4-py runs and the PTY approval path for gate 4. It writes `bench/results/m3.json` and per-task logs/traces/saves under `bench/results/`.

- [x] **Step 4: Verify non-live logic, then run live evidence when assets are available**

Non-live verification passed on 2026-10-28: 51 bench tests, two guarded live skips, and the full 1,420-test regression. Live evidence was not run: all four opt-in variables were unset and the machine reported 62.7 GiB RAM, below the 96 GB requirement. M3 remains not live-verified.

```bash
uv run --extra agent pytest tests/bench -m "not live and not build" -q
env -u DS4_RUN_M3_BAKEOFF -u DS4_TEST_MODEL -u DS4_AGENT -u POLYGLOT_BENCHMARK uv run --extra agent pytest tests/live/test_m3_bakeoff.py -q  # confirms guarded skip, not marker deselection
# Full live command from Step 3, only with the required model and machine.
git add bench tests/bench tests/live/test_m3_bakeoff.py
git commit -m "test(bench): add live six-gate M3 acceptance run"
```

Do not mark the live completion item done from a guarded skip. If assets are unavailable, report the live gate as not run and leave M3 completion unresolved.

---

## Task 16: Document usage and run the complete verification

**Files:**
- Modify: `README.md`
- Create: `docs/agent.md`
- Modify: `docs/specs/2026-10-21-ds4-py-design.md` only for a further concrete, evidenced mismatch found during implementation; the verified harness corrections are already in Task 1
- Verify all M3 files

- [x] **Step 1: Document installation and behavior**

Document `uv sync --extra agent`, interactive and noninteractive examples, `DS4PY_HOME`, approvals/read-only behavior, save/switch restore cases, trace files, and the live bake-off command. State clearly that core/raw/build/check do not require the harness extra and that M4 owns compaction and the remaining persistence commands.

- [x] **Step 2: Run non-live verification**

```bash
uv sync --extra agent
uv run --extra agent pytest -m "not live and not build"
uv run --extra agent pytest tests/agent tests/bench -m "not live and not build"
uv run --isolated --no-project --with . python -c "import ds4py; import ds4py.cli; import ds4py.store"
uv run --extra agent python -c "from ds4py.model import DS4Model; from ds4py.runtime import AgentRuntime; import prompt_toolkit"
git diff --check main...HEAD
git status --short
```

All selected tests must pass. The core isolated import must not import `pydantic_ai`, `pydantic_ai_harness`, or `prompt_toolkit`. Run `ds4py raw --help`, `ds4py check --help`, and `ds4py build --help` in that isolated environment as an extension-loading/import smoke check; help is sufficient here only for CLI import verification, not for behavior coverage.

- [x] **Step 3: Inspect module boundaries and the complete diff**

Review changed modules against the module-size guide: `model.py`, `binding.py`, `runtime.py`, `tui.py`, and `store.py` should each have one clear responsibility and stay near or below 400 meaningful lines. Split async streaming from request preparation, restore matching from file I/O, or footer state from prompt orchestration if a file grows beyond that. Do not create generic `utils`/`helpers` modules.

Inspect `git diff main...HEAD` for the review focus: cancellation/rewind, save ordering, key/hash binding, denied outcomes, truncation/corrupt restore, and honest gate calculations. Check every new public facade/runtime API documents thread and active-round restrictions.

The final audit extracted `message_rendering.py` and `runtime_restore.py`. Meaningful line counts are 411 for binding, 405 for runtime, 342 for model, 349 for store, and 159 for the TUI. Existing binding, restore, runtime, and noninteractive tests verified the behavior-preserving moves.

- [x] **Step 4: Record live evidence separately**

live bake-off: not run (missing assets). All four opt-in variables were unset, and the machine reported 62.7 GiB RAM. M3 is not yet live-verified.

If the required assets exist, run the exact live command in `bench/README.md` and record the report path/result. All six gates must pass before claiming M3 is complete. If not, state `live bake-off: not run (missing assets)` and do not substitute unit/integration tests for it.

- [x] **Step 5: Commit documentation and final verification updates**

```bash
git add README.md docs/agent.md docs/plans/2026-10-21-m3-harness-agent.md
git commit -m "docs(agent): document M3 usage and acceptance"
```

---

## Completion checklist

- [x] The agent extra pins `pydantic-ai-harness[cli]==0.59.0` and resolves `pydantic-ai-slim[cli]==2.59.0`; the spec's stale API names are corrected and core imports remain dependency-free.
- [x] `DS4Model` supports tools/thinking, no native tools, structured-output prompting, exact usage, finish reasons, and interrupted partial responses.
- [x] Stable key/hash binding reuses identical reserialized prefixes and re-renders only from the first change; sampled tokens are not re-rendered on the normal path.
- [x] Tool results are one ordered tool message; all payloads stay plain; canonical restored calls round-trip through all family parsers.
- [x] Cancellation and speculative rewind leave native tokens and transcript aligned; M3 reserve failure is explicit and facade resume still works.
- [x] Required payload bindings pass ABI probing; save syncs before staging; load takes token IDs from the native session.
- [x] Save order is KV → complete named snapshot → save record; case 1 does no prefill and case 2 renders fresh; corrupt/missing records fail clearly.
- [x] Approvals produce real denied tool results without executing denied tools; read-only remains effective under `--yolo`.
- [x] Date/hints notes match C behavior; the 50,000-token reminder and real compaction remain M4, and interactive `/think` remains M5.
- [x] Background commands persist across prompts and stop on new/switch/exit; `ToolOutputLimits` handles large results without another model call.
- [x] StepPersistence and events.jsonl retain success/interrupted/error evidence; new/switch close state, fork preserves it, and old files remain.
- [x] Noninteractive runs save in `finally`; ds4-py and C traces feed strict replay readers without executing tools.
- [x] TUI supports cancellation, editing, slash completion, Markdown/status rendering, and bounded one-line footers.
- [x] Conformance items 1, 6, 7, and 9 pass; items 2–5 are not prematurely marked complete.
- [x] The 12-task manifest is pinned and hidden grading works; per-task reports contain all required metrics.
- [x] `uv run --extra agent pytest -m "not live and not build"` passes and `git diff --check main...HEAD` is clean.
- [x] The explicit live bake-off has run on the required model/machine and all six gates pass, or M3 is explicitly reported as not yet live-verified.

Non-live implementation and acceptance completed on 2026-10-28: 1,420 tests passed and 17 live tests were deselected. Isolated core imports, agent imports, dependency pins, and diff checks passed. M3 remains not yet live-verified because the live bake-off did not run.

---

## Review focus and plan self-review

This section is the single plan-level review pass; it is not a substitute for code review during execution.

1. **Worker/cancellation/rewind:** Tasks 3–6 keep all native calls on the worker. Tests cancel during prefill and a speculative block, require uncommitted drafts to stay invisible, and compare native tokens to transcript after interruption. Context reserve cannot silently hang while compaction is deferred.
2. **Save/restore integrity:** Tasks 6–9 test trailing-EOS sync, exact payload lengths, resource cleanup, atomic replacement, record-last ordering, native token hashing, exact snapshot/step selection, and keyed segment matching. Case 1 and stripped case 2 are distinct tests and distinct live gate-5 paths.
3. **Binding correctness:** Tasks 1–2 and 12 use reserialized objects, same-key content changes, tool/think changes, checkpoint cuts, and provenance coverage. Ordinary sampled responses bind existing token ranges rather than re-rendering text.
4. **Approval semantics:** Task 8 uses a real harness `Agent` to prove `SkipToolExecution(ToolDenied(...))` becomes `ToolReturnPart(outcome='denied')`, with no tool-body execution. Yolo and read-only are tested independently.
5. **TUI and corrupted input:** Task 11 uses long-cwd/narrow-width tests. Tasks 7 and 9 cover malformed JSON, bad headers, truncated/extra payloads, mismatched hashes, and missing snapshots/keys. No path or stream is silently treated as a successful empty session.
6. **Bake-off meaning:** Tasks 13–15 pin all tasks and source values, hide graders, test second attempts and sequential execution with fake agents, define every metric, and write six explicit gate results. Gate 2 reports comparisons without inventing an unspecified tolerance. Only the guarded live run can satisfy the final acceptance item.

Plan coverage checked against the spec's M3 Interface, Done-when gates, conformance items, and section-18 exclusions. Resolver metadata and installed signatures establish one required factual spec amendment: harness `0.59.0` uses slim `2.59.0` and the current capability/profile/usage APIs; Task 1 records that amendment before code while preserving the approved behavior. Intentional builder decisions fixed in this plan are the 20-byte KV header, big-endian uint32 token hashing, approval key meanings/pattern scopes, 80-character titles, the gate-4 repository task, and gate-2 comparability semantics. These are implementation details left open by the spec, not deferred requirements or new open questions.
