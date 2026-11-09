# M4 standard compaction implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic and manual LLM compaction, which replaces old history with a model-written summary, and continue the current response after compaction.

**Architecture:** `LLMCompaction` replaces the stored history with the system parts, a user summary, and a retained tail. The engine prepares the summary and replacement on its worker. `MessageBinder` uses one provenance installer for restore and compaction, then commits transcript, binding, and messages as one atomic change. Composition selects one automatic context owner and exposes a separate manual compactor.

**Diagram:** See [standard compaction architecture](../specs/2026-10-21-ds4-py-design.md#standard-compaction-architecture) for the full class and module view.

**Tech Stack:** Python, Pydantic AI, the existing ds4 engine worker, `FakeNative`, pytest, devenv, and uv.

**Spec:** [`docs/specs/2026-10-21-ds4-py-design.md`](../specs/2026-10-21-ds4-py-design.md). Read sections 11, 13, 15, 17, and 23. This plan implements only the compaction requirements.

**Base:** Start at `5c98c6a` on the `m4-context-persistence` branch. Use `/home/roche/projects/satyrn/ds4-py/.worktrees/m4-context-persistence`. Do not edit the main checkout.

## Global Constraints

A reserve is token space kept for the next response. KV is the model's cached token state. Token provenance links token ranges to their source messages.

- Install `LLMCompaction(soft=0.85)` as the default context strategy.
- Keep `DS4Compaction = LLMCompaction` as the public alias used by the spec.
- Do not implement `RollingWindow`, paging notices, recovery tools, or a port of `pi-context-paging` in this branch.
- Do not implement the 50,000-token reminder, the system-prefix cache, `Bookmarks`, `/list`, `/strip`, or `/del` in this branch.
- Use the existing `MessageBinder.count_tokens()` and `Session.ctx_used` for exact token counts. Do not add text-length estimates.
- Use the existing engine worker, tokenizer, prompt builder, and decoder. Do not create another model session or tool loop for summaries.
- Use one `MessageBinder.install_transcript()` for compaction and `/switch`. Do not add a second provenance matcher.
- Select one automatic context owner in `build_agent_components`. Expose manual compaction through `AgentComponents.compactor`.
- Pass a typed `ContextRequest` to `make_room()`. Do not store the current response key on `ContextManager`.
- Keep the save-record format at version 1. Do not add fields for paged views.
- Keep old step snapshots unchanged. They preserve history that compaction removes from the live message list.
- Run project commands from the worktree inside `devenv shell`.
- Write a failing test before each behavior change. Make one focused commit after each task.
- Keep formatting changes in separate commits.

## Review Focus

- A failed or cancelled summary must leave the live transcript and message history unchanged. Task 3 and Task 4 cover this case.
- A tail cut must keep whole messages and tool exchanges. Task 3 covers user boundaries and open assistant output.
- A reserve pause must keep the same `ModelResponse`, parser state, response key, and output parts. Task 5 covers this case.
- A compacted save must restore the same summary, tail, keys, and token provenance. Task 6 covers exact KV restore, fresh rendering, and fork.
- Usage must include real prefill, cache reuse, compaction output, and resumed output once. Task 5 covers zero-prefill and reserve-resume cases.

## File Structure

- `engine/compaction.py` runs the summary job and prepares the retained tail and replacement.
- `engine/rebind.py` replaces transcript tokens and segments through the worker.
- `engine/chat.py` creates the `system` checkpoint after the first system-only sync.
- `binding.py` owns the shared provenance installer and the atomic compaction commit.
- `runtime_restore.py` uses the same installer for `/switch`.
- `capabilities/context.py` defines typed overflow requests and the shared fit check.
- `capabilities/compaction.py` implements the persistent history rewrite and the manual compactor protocol.
- `model_context.py` connects an open model stream to the context owner without holding the pull lock.
- Existing public classes remain thin facades for these focused modules.

---

## Task 1: Share transcript installation and add one atomic commit

**Files:**
- Modify: `src/ds4py/engine/transcript.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/chat.py`
- Modify: `src/ds4py/binding.py`
- Modify: `src/ds4py/runtime_restore.py`
- Create: `src/ds4py/engine/rebind.py`
- Create: `tests/engine/test_transcript_replace.py`
- Modify: `tests/agent/test_binding.py`
- Modify: `tests/agent/test_conformance_persistence.py`

**Interfaces:**
- Produces: `TranscriptReplacement(tokens: tuple[int, ...], segments: tuple[Segment, ...])`.
- Produces: `Session.replace_transcript(replacement: TranscriptReplacement, *, reason: str) -> Plan`.
- Produces: `MessageBinder.install_transcript(messages: Sequence[ModelMessage], segments: Sequence[_Provenance]) -> None`.
- Produces: `MessageBinder.commit_compaction(messages: list[ModelMessage], compacted_messages: Sequence[ModelMessage], replacement: TranscriptReplacement) -> None`.
- Consumes: Existing restore validation, transcript segments, message keys, content hashes, checkpoints, and sync plans.

- [ ] **Step 1: Write the failing tests**

Add tests for a replaced transcript and its binding. Use a system prefix, two user turns, tool output, and a summary segment.

```python
def test_replace_transcript_uses_longest_valid_checkpoint() -> None:
    result = session.replace_transcript(compacted_replacement, reason="compaction")
    assert result.kind == "restore"
    assert result.checkpoint == "system"
    assert session.transcript.tokens == compacted_replacement.tokens
    assert session.transcript.segments == compacted_replacement.segments


def test_shared_installer_preserves_retained_message_keys() -> None:
    session.replace_transcript(compacted_replacement, reason="compaction")
    binder.install_transcript(compacted_messages, compacted_replacement.segments)
    assert retained_keys(binder) == original_tail_keys
    assert summary_segment(binder).meta["source"] == "compaction"
```

Test the first system-only sync in `chat_ops.start_chat`. It must hold the in-memory `system` checkpoint before any user tokens are synced. This checkpoint is not the deferred system-prefix disk cache.

Add cases for gaps, overlaps, stale keys, stale hashes, and a missing checkpoint. Invalid input must not change the transcript or binding. Run the same validation cases through compaction and `/switch` restore.

Add commit failure cases at native replacement, binding installation, and message rewrite. Assert that tokens, segments, binder records, bound messages, and `ctx.messages` all return to their original state.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/engine/test_transcript_replace.py tests/agent/test_binding.py tests/agent/test_conformance_persistence.py -q
```

Expected: FAIL because transcript replacement and the shared commit do not exist.

- [ ] **Step 3: Implement the shared installer and atomic commit**

Make `engine/rebind.py` validate contiguous segments from token zero. Run replacement through the existing worker. Restore the longest valid checkpoint or rebuild from token zero. Keep checkpoint creation out of this module.

In `chat_ops.start_chat`, sync only the initial system segments through the existing `_job_events(SyncJob(...))` path. Call `session.checkpoint("system")` after that sync succeeds. Later rewrites only consume this checkpoint.

Refactor `MessageBinder._restore_state` into one pure preparation helper and use `install_transcript()` for both compaction and `/switch`. Replace the `runtime_restore.py` call to `restore()` with `install_transcript()`. Keep `validate_restore()` as the non-mutating check before a native payload load. Do not add a compaction-specific binding module or a second provenance matcher.

Make `commit_compaction()` capture the old tokens, segments, binder state, and message list before any change. Prevalidate the new binding against the replacement token count. Call `Session.replace_transcript()`, call `install_transcript()`, and rewrite `messages[:]`. On any error, restore the old native replacement and the exact old binder and message state through one rollback path. Return only after every change succeeds.

- [ ] **Step 4: Run the focused tests and make sure that they pass**

```bash
uv run --frozen --extra agent pytest tests/engine/test_transcript_replace.py tests/agent/test_binding.py tests/agent/test_conformance_persistence.py -q
uv run --frozen --extra agent pytest tests/engine/test_plan.py tests/engine/test_worker.py tests/agent/test_store.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/engine/transcript.py src/ds4py/engine/session.py src/ds4py/engine/chat.py src/ds4py/engine/rebind.py src/ds4py/binding.py src/ds4py/runtime_restore.py tests/engine/test_transcript_replace.py tests/agent/test_binding.py tests/agent/test_conformance_persistence.py
git commit -m "feat: add atomic transcript compaction"
```

---

## Task 2: Add typed overflow requests and async reserve recovery

**Files:**
- Create: `src/ds4py/capabilities/context.py`
- Create: `src/ds4py/model_context.py`
- Modify: `src/ds4py/model.py`
- Modify: `src/ds4py/errors.py`
- Create: `tests/agent/test_context_manager.py`
- Modify: `tests/agent/test_model_cancel.py`
- Modify: `tests/agent/test_model_stream.py`

**Interfaces:**
- Produces: `ContextRequest(needed_tokens: int, reason: Literal["tool_result", "reserve"], response_key: str | None = None)`.
- Produces: `ContextStrategy.make_room(ctx: RunContext[object], request: ContextRequest) -> list[ModelMessage] | None`.
- Produces: `ContextManager(owner: ContextStrategy | None)` in `session.state()` under the key `ds4.context`.
- Produces: `make_room_for_response(ctx: RunContext[object], request: ContextRequest) -> int` in `model_context.py`.
- Produces: `ContextExhausted`. Keep `ContextReserveError` as an alias.
- Consumes: `MessageBinder.count_tokens()`, `Session.ctx_used`, and Task 1 rebinding.

- [ ] **Step 1: Write the failing tests**

Use a test-only owner to prove request contents and overflow behavior.

```python
async def test_tool_overflow_asks_owner_once() -> None:
    result = await invoke_after_tool_execute(oversized_result)
    assert owner.requests == [ContextRequest(needed_tokens, "tool_result")]
    assert result.content == fitting_result


async def test_reserve_recovery_passes_response_key_without_lock() -> None:
    response = await consume_reserve_paused_response()
    assert owner.requests == [ContextRequest(1, "reserve", original_response_key)]
    assert owner.pull_lock_was_locked is False
    assert response_keys == [original_response_key]
```

Add a missing-owner case that raises `ContextExhausted` and preserves partial tokens without returning partial text as an answer. Add a cleanup-race case that fails a reserve pause without calling async owner code while `_pull_lock` is held. A failed tool-result retry must return exactly `Tool error: tool result still does not fit`.

Replace the synchronous `reserve_handler` test in `test_model_cancel.py`. Assert that `DS4Model` no longer accepts that argument and that the async owner path resumes the same response.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/agent/test_context_manager.py tests/agent/test_model_cancel.py tests/agent/test_model_stream.py -q
```

Expected: FAIL because typed owner requests and async reserve recovery do not exist.

- [ ] **Step 3: Implement typed requests and the stream state transition**

Define `ContextRequest` as a frozen dataclass. Require a response key when `reason == "reserve"`. Reject a response key when `reason == "tool_result"`.

Define `ContextStrategy` with only `make_room(ctx, request)`. Define `ContextManager` with an owner fixed at construction and one shared `after_tool_execute` fit check. Do not add registration methods, ownership mode flags, stand-down behavior, or current-request fields. Task 4 selects the owner once during composition.

Make `make_room_for_response()` pass the same request object to the owner. Return reserve budget `1` after a successful rewrite. Raise `ContextExhausted` when no owner can free space. Do not store the response key in session state.

Delete the `reserve_handler` constructor argument, model attribute, synchronous `_dispatch` branch, and old test. In `_get_event_iterator`, handle `ReserveReached` after `_pull()` returns and after `_pull_lock` is released. Await the owner, then call `event.resume(1)` or `event.fail()`. Keep `_dispatch()` as pure synchronous event translation. If cleanup encounters a reserve pause while holding `_pull_lock`, fail the pause instead of awaiting owner code.

- [ ] **Step 4: Run the focused tests and make sure that they pass**

```bash
uv run --frozen --extra agent pytest tests/agent/test_context_manager.py tests/agent/test_model_cancel.py tests/agent/test_model_stream.py -q
uv run --frozen --extra agent pytest tests/agent/test_agent_factory.py tests/agent/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/capabilities/context.py src/ds4py/model_context.py src/ds4py/model.py src/ds4py/errors.py tests/agent/test_context_manager.py tests/agent/test_model_cancel.py tests/agent/test_model_stream.py
git commit -m "feat: add typed context overflow handling"
```

---

## Task 3: Prepare the engine summary and replacement

**Files:**
- Create: `src/ds4py/engine/compaction.py`
- Modify: `src/ds4py/engine/events.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/decode.py`
- Modify: `src/ds4py/_native/fake.py`
- Create: `tests/engine/test_compaction.py`

**Interfaces:**
- Produces: `Session.prepare_compaction(reason: str, *, response_key: str | None = None) -> CompactionResult`.
- Produces: `CompactionResult(summary: str, dropped: int, kept: int, tail_keys: tuple[str, ...], replacement: TranscriptReplacement)`.
- Produces: `CompactionStarted` before summary generation.
- Consumes: Task 1 `TranscriptReplacement` and the existing decode primitives. Task 4 emits `Compacted` only after the atomic commit.

- [ ] **Step 1: Write the failing tests**

Use the exact compaction prompt from `ds4_agent.c:9675–10229`. Assert its bytes in a fixture.

```python
@pytest.mark.parametrize(("ctx", "limit"), [(1024, 256), (16384, 2048), (65536, 4096)])
def test_summary_limit_matches_c(ctx: int, limit: int) -> None:
    assert summary_limit(ctx) == limit


def test_compaction_prepares_system_summary_and_verbatim_tail() -> None:
    before = (session.transcript.tokens, session.transcript.segments)
    result = session.prepare_compaction("threshold")
    assert result.summary == expected_summary
    assert result.replacement.tokens == system_tokens + summary_tokens + original_tail_tokens
    assert result.tail_keys == expected_tail_keys
    assert result.dropped + result.kept == original_history_tokens
    assert (session.transcript.tokens, session.transcript.segments) == before
```

Add stop-token, maximum-summary, masked tool-syntax, and leaked DSML cases. A leaked DSML start must stop the summary and remove its incomplete text.

Add tail cases at user boundaries. Limit the tail to `min(ctx / 10, 50000)` tokens. Keep whole tool exchanges and any open assistant output.

Add failure and cancellation cases. Assert the original native state, tokens, and segments remain unchanged. Assert no `Compacted` event occurs.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/engine/test_compaction.py -q
```

Expected: FAIL because `Session.prepare_compaction()` does not exist.

- [ ] **Step 3: Implement `CompactJob` and `Session.prepare_compaction()`**

Build the summary prompt as `transcript[:bottom] + user(prompt) + assistant_prefix(no-think)`. Generate plain text with `argmax`.

Set the summary limit to `clamp(ctx / 8, 256, 4096)`. Stop at a normal stop token or a DSML special token. Never parse summary text as a tool call.

Select the tail at a user-token boundary. After a valid summary exists, build a `TranscriptReplacement` from `system + user(summary) + verbatim tail`. Append any open assistant prefix and sampled tokens. Do not change the live transcript, binder, or messages in this task.

Emit `CompactionStarted` before summary tokens. Restore the native state that existed before summary generation before returning or raising. Treat `dropped` and `kept` as token counts. Do not emit `Compacted` in this task.

- [ ] **Step 4: Run the focused tests and make sure that they pass**

```bash
uv run --frozen --extra agent pytest tests/engine/test_compaction.py tests/engine/test_context_reserve.py tests/engine/test_decode.py -q
uv run --frozen --extra agent pytest tests/engine/test_transcript_replace.py tests/engine/test_session.py tests/engine/test_worker.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/engine/compaction.py src/ds4py/engine/events.py src/ds4py/engine/session.py src/ds4py/engine/decode.py src/ds4py/_native/fake.py tests/engine/test_compaction.py
git commit -m "feat: prepare engine LLM compaction"
```

---

## Task 4: Add automatic compaction and explicit manual composition

**Files:**
- Create: `src/ds4py/capabilities/compaction.py`
- Modify: `src/ds4py/capabilities/context.py`
- Modify: `src/ds4py/capabilities/__init__.py`
- Modify: `src/ds4py/agent.py`
- Modify: `src/ds4py/runtime.py`
- Create: `tests/agent/test_compaction_capability.py`
- Modify: `tests/agent/test_agent_factory.py`
- Modify: `tests/agent/test_runtime.py`
- Modify: `tests/agent/test_noninteractive.py`

**Interfaces:**
- Produces: `LLMCompaction(binder: MessageBinder, *, soft: float = 0.85)`.
- Produces: `DS4Compaction = LLMCompaction`.
- Produces: `ManualCompactor` with `compact_messages(messages: list[ModelMessage], *, reason: str, response_key: str | None = None) -> tuple[list[ModelMessage], CompactionResult]`.
- Produces: `AgentComponents.compactor: ManualCompactor`.
- Produces: `build_agent_components(..., context_strategy: ContextStrategy | None = None)`.
- Produces: `AgentRuntime(..., context_strategy: ContextStrategy | None = None)`.
- Default: One `LLMCompaction(soft=0.85)` owns automatic context management and also serves as the explicit manual compactor.
- Consumes: Task 1 atomic commit, Task 2 `ContextManager`, and Task 3 `Session.prepare_compaction()`.

- [ ] **Step 1: Write the failing tests**

Test both trigger rules and the persistent rewrite.

```python
def test_soft_limit_trigger() -> None:
    assert should_compact(ctx_used=8499, ctx=10000, reserve=100, soft=0.85) is False
    assert should_compact(ctx_used=8500, ctx=10000, reserve=100, soft=0.85) is True


async def test_compaction_rewrites_live_history() -> None:
    new_messages, result = await compaction.compact_messages(old_messages, reason="threshold")
    assert summary_request(new_messages).parts[0].content == result.summary
    assert retained_messages(new_messages) == expected_tail
    assert retained_keys(new_messages) == original_tail_keys
```

Cover the reserve predicate `ctx - used <= max(min(8192, ctx / 8), reserve + 128)`. Include the rendered prompt, assistant prefix, and summary limit in `reserve`.

Add composition tests for the default and a custom strategy:

```python
def test_default_agent_installs_one_llm_compaction_owner() -> None:
    compactors = [c for c in components.components if isinstance(c, LLMCompaction)]
    assert len(compactors) == 1
    assert compactors[0].soft == 0.85
    assert context_owner(session) is compactors[0]
    assert components.compactor is compactors[0]


def test_custom_strategy_replaces_only_automatic_owner() -> None:
    components = build_with_context_strategy(custom_strategy)
    assert context_owner(session) is custom_strategy
    assert not any(isinstance(c, LLMCompaction) for c in components.components)
    assert isinstance(components.compactor, LLMCompaction)
    assert components.compactor is not custom_strategy
```

Add a noninteractive FakeNative run that crosses the soft limit and continues with the compacted history.

Add cases for no safe tail, summary failure, binder failure, message rewrite failure, and a custom context owner. Every case must leave `ctx.messages`, the native transcript, and the binding unchanged. Assert no `compacted` event after a failed commit.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_capability.py tests/agent/test_agent_factory.py tests/agent/test_runtime.py tests/agent/test_noninteractive.py -q
```

Expected: FAIL because the capability and explicit composition do not exist.

- [ ] **Step 3: Implement the capability and static owner selection**

Add a pure `should_compact()` helper for the two C trigger rules. Build the new history from system and instruction parts, one `ModelRequest` with `UserPromptPart(summary)`, and the retained tail. Give the summary a new `ds4.key`. Keep retained messages and their keys unchanged.

In `compact_messages()`, emit `before_compact`, call `Session.prepare_compaction()`, and build the new message list. Pass the live list, new list, and `result.replacement` to Task 1 `MessageBinder.commit_compaction()`. Do not implement a second rollback path. During automatic compaction, set `request_context.messages` to the committed list. Emit `compacted` only after the commit succeeds. Manual compaction returns the committed list without requiring a request context.

In `build_agent_components`, create one `LLMCompaction` with `model.binder`. Select `context_strategy if context_strategy is not None else compactor` as the only automatic owner. Store that owner in `session.state("ds4.context", ...)` once. Add only the selected owner to the capability list. When a custom strategy is supplied, do not install `LLMCompaction` as an automatic capability.

Return the manual compactor through `AgentComponents.compactor`. In `AgentRuntime`, store this explicit dependency and pass through `context_strategy`. Do not scan capabilities to locate the compactor. Keep configuration on the capability and mutable state in `session.state()`.

- [ ] **Step 4: Run the focused tests and make sure that they pass**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_capability.py tests/engine/test_compaction.py tests/agent/test_context_manager.py -q
uv run --frozen --extra agent pytest tests/agent/test_binding.py tests/agent/test_system_notes.py -q
uv run --frozen --extra agent pytest tests/agent/test_agent_factory.py tests/agent/test_runtime.py tests/agent/test_noninteractive.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/capabilities/compaction.py src/ds4py/capabilities/context.py src/ds4py/capabilities/__init__.py src/ds4py/agent.py src/ds4py/runtime.py tests/agent/test_compaction_capability.py tests/agent/test_agent_factory.py tests/agent/test_runtime.py tests/agent/test_noninteractive.py
git commit -m "feat: add automatic LLM history compaction"
```

---

## Task 5: Continue the same response after reserve compaction

**Files:**
- Modify: `src/ds4py/model.py`
- Modify: `src/ds4py/model_context.py`
- Modify: `src/ds4py/binding.py`
- Create: `tests/agent/test_compaction_stream.py`
- Modify: `tests/agent/test_model_stream.py`

**Interfaces:**
- Consumes: `make_room_for_response(ctx, request)` with `ContextRequest(reason="reserve", response_key=...)`.
- Consumes: `Session.prepare_compaction("reserve", response_key=...)` and Task 1 `commit_compaction()` so `LLMCompaction` can retain the open assistant segments.
- Produces: Reserve recovery that resumes the existing round and `DS4StreamedResponse`.
- Produces: Usage based on each real `Synced.plan`, including `cache_read_tokens`.

- [ ] **Step 1: Write the failing tests**

Create a `FakeNative` script with text before reserve, compaction, more text, and a tool call.

```python
async def test_reserve_compaction_continues_same_response() -> None:
    response = await consume_response()
    assert response_count == 1
    assert response.parts == [TextPart("before after"), expected_tool_call]
    assert response_keys == [original_response_key]
    assert parser_finish_calls == 1
    assert response.usage.output_tokens == before_tokens + summary_tokens + after_tokens
```

Add thinking and partial-tool cases. Assert no duplicate assistant prefix, text delta, response part, or tool call. Keep compaction events out of response text.

Add a zero-prefill sync case and a reserve-resume case:

```python
assert response.usage.input_tokens == sum(sync.prefilled for sync in syncs)
assert response.usage.cache_read_tokens == sum(sync.plan.reused for sync in syncs)
assert response.usage.details["prefill_tokens"] == response.usage.input_tokens
```

Add a failed recovery case. It must raise `ContextExhausted` and keep partial tokens without returning partial text as an answer.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_stream.py tests/agent/test_model_stream.py -q
```

Expected: FAIL because reserve handling ends the stream and cache usage is incomplete.

- [ ] **Step 3: Implement reserve compaction and usage accounting**

Pass the saved `RunContext` and typed `ContextRequest` to `make_room_for_response()`. Pass `request.response_key` directly to `LLMCompaction.compact_messages()` and `Session.prepare_compaction()`. The engine finds the open assistant prefix and sampled tokens at the transcript tail. Do not read response state from `ContextManager`.

Keep the trailing assistant prefix, sampled tokens, response key, parser, and parts manager alive. Complete the Task 1 atomic commit before calling `ReserveReached.resume(1)`. Do not call parser `finish()` at the reserve boundary. Keep reserve recovery outside `_pull_lock` and keep `_dispatch()` free of async work, as Task 2 defines.

Use `CompactionStarted` and the post-commit `Compacted` event to suppress summary text in the response. Count all generated tokens once in `output_tokens`. Count real prefill and cache reuse from every `Synced` event.

Call `ReserveReached.fail(...)` and raise `ContextExhausted` when recovery fails.

- [ ] **Step 4: Run the focused tests and make sure that they pass**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_stream.py tests/agent/test_model_stream.py tests/agent/test_model_cancel.py -q
uv run --frozen --extra agent pytest tests/agent/test_model.py tests/agent/test_compaction_capability.py tests/engine/test_context_reserve.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/model.py src/ds4py/model_context.py src/ds4py/binding.py tests/agent/test_compaction_stream.py tests/agent/test_model_stream.py
git commit -m "feat: resume responses after reserve compaction"
```

---

## Task 6: Add compacted-session persistence regression coverage

**Files:**
- Create: `tests/agent/test_compaction_persistence.py`
- Modify: `tests/agent/test_conformance_persistence.py`

**Interfaces:**
- Consumes: Existing version 1 save records, `AgentRuntime.save()`, `switch()`, and `fork()`.
- Consumes: Task 1 `MessageBinder.install_transcript()` and compaction provenance from Task 4.
- Produces: Regression tests only. No new persistence format, production restore path, or public record fields.

- [ ] **Step 1: Write the regression tests**

Compact a session, save it, and switch back to it.

```python
async def test_exact_restore_after_compaction_has_no_prefill() -> None:
    restored = await save_compact_and_switch()
    assert restored.messages == compacted_messages
    assert restored.binding_keys == compacted_keys
    assert restored.segments == compacted_segments
    assert restored.prefill_events == []
```

Add a missing-payload case that renders the compacted snapshot fresh. Add a fork case that keeps keys, messages, binding, and live KV. Assert old runs still contain the history from before compaction.

Do not require messages from old step runs to have segments in the current KV. Do not add paged-view rules. A fork changes only the session identity and creation time.

- [ ] **Step 2: Run the regression tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_persistence.py -q
```

Expected: PASS because compaction and `/switch` already share the Task 1 installer. If a case fails, return to Task 1 and fix the shared installer or atomic commit. Do not add compaction-specific restore production code in this task.

- [ ] **Step 3: Run the related persistence tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_persistence.py tests/agent/test_conformance_persistence.py -q
uv run --frozen --extra agent pytest tests/agent/test_store.py tests/agent/test_store_failures.py tests/agent/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the regression tests**

```bash
git add tests/agent/test_compaction_persistence.py tests/agent/test_conformance_persistence.py
git commit -m "test: cover compacted session persistence"
```

---

## Task 7: Add manual `/compact`

**Files:**
- Modify: `src/ds4py/runtime.py`
- Modify: `src/ds4py/tui_commands.py`
- Modify: `src/ds4py/tui_render.py`
- Modify: `src/ds4py/tui.py`
- Create: `tests/agent/test_context_commands.py`
- Modify: `tests/agent/test_tui_commands.py`

**Interfaces:**
- Produces: `AgentRuntime.compact() -> CompactionResult`.
- Produces: `/compact` with no arguments.
- Consumes: `AgentComponents.compactor: ManualCompactor` and `compact_messages(messages, reason="manual")`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_compact_command_rewrites_current_history_once() -> None:
    output = await registry.execute("/compact")
    assert compact_calls == [("manual",)]
    assert runtime.messages == compacted_messages
    assert "Compacted" in output
```

Add no-argument validation, completion, and failure cases. A failed command must leave the current session usable and unchanged.

Add an active-run case. Use the same stop-and-await rule as `save()` before manual compaction changes history. Assert `before_compact` and `compacted` update the TUI without adding response text.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/agent/test_context_commands.py tests/agent/test_tui_commands.py -q
```

Expected: FAIL because the runtime method and command do not exist.

- [ ] **Step 3: Implement runtime compaction and the command**

Keep `AgentRuntime.compact()` as a thin delegate to `self.components.compactor`. Do not scan capabilities or use `isinstance` to find it. Stop and await an active run before changing messages. Assign `self._messages` only after the Task 1 atomic commit succeeds.

Register `/compact` through `SlashCommandRegistry`. Reuse the existing no-argument validation and command error format. Render progress and completion from session events. Do not import TUI code from a capability.

Manual compaction must work even when a custom strategy owns automatic context management.

- [ ] **Step 4: Run the focused tests and make sure that they pass**

```bash
uv run --frozen --extra agent pytest tests/agent/test_context_commands.py tests/agent/test_tui_commands.py tests/agent/test_runtime.py -q
uv run --frozen --extra agent pytest tests/agent/test_tui.py tests/agent/test_compaction_capability.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/runtime.py src/ds4py/tui_commands.py src/ds4py/tui_render.py src/ds4py/tui.py tests/agent/test_context_commands.py tests/agent/test_tui_commands.py
git commit -m "feat: add manual compact command"
```

---

## Task 8: Document compaction and run the full gate

**Files:**
- Modify: `docs/development.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Default automatic compaction from Task 4 and `/compact` from Task 7.
- Produces: User and developer documentation. No API changes.

- [ ] **Step 1: Update the documentation**

Document the 85 percent trigger, reserve recovery, persistent summary history, and `/compact`. State that old step snapshots retain removed history. Explain that an exact KV restore needs no prefill. Do not describe deferred M4 features as complete.

Do not add tests for documentation text. Use the existing gates for these edits.

- [ ] **Step 2: Run the focused and conformance suites**

```bash
uv run --frozen --extra agent pytest tests/agent tests/engine -m "not build and not live" -q
uv run --frozen --extra agent pytest tests/agent/test_conformance.py tests/agent/test_conformance_persistence.py -q
```

Expected: PASS. Do not treat FakeNative results as live-inference evidence.

- [ ] **Step 3: Run the complete static and portable gates**

```bash
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
uv run --frozen --extra agent pytest -m "not build and not live" -q
```

Expected: PASS.

If personal skills affect agent tests, repeat the portable suite with an empty home:

```bash
mkdir -p /tmp/ds4-quality-empty-home
HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest -m "not build and not live" -q
```

- [ ] **Step 4: Commit the documentation**

```bash
git add docs/development.md README.md
git commit -m "docs: explain LLM compaction"
```

---

## Completion Checklist

- [ ] `LLMCompaction` is the default and `DS4Compaction` remains an alias.
- [ ] The 85 percent and reserve triggers match the C rules.
- [ ] Summary generation uses the C prompt, no-think prefix, `argmax`, summary clamp, stop rules, and tail limit.
- [ ] Automatic compaction rewrites live history without changing old step snapshots.
- [ ] Reserve compaction continues the same `ModelResponse` without duplicate output.
- [ ] Tool-result overflow either fits after compaction or returns the exact tool error.
- [ ] `/compact` compacts once and leaves the session unchanged on failure.
- [ ] Save, switch, and fork preserve compacted messages, keys, and token provenance.
- [ ] No `RollingWindow`, paging tool, token reminder, prefix cache, or unrelated session command is included.
- [ ] Existing conformance tests and the complete portable gate pass.

## Plan Self-Review

This plan covers the compaction requirements in sections 11, 13, 15, 17, and 23. It leaves the other M4 features for later plans. Each behavior has a failing-test-first task and focused commands. The tasks use one engine, one binder, one context owner, and one persistence format. They do not add paging or a second summary model.
