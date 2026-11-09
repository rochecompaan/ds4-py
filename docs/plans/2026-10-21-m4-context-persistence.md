# M4 standard compaction implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic and manual LLM compaction, which replaces old history with a model-written summary, and continue the current response after compaction.

**Architecture:** `LLMCompaction` replaces the stored history with the system parts, a user summary, and a retained tail. The engine creates the summary on its worker and replaces the token transcript. `MessageBinder` keeps message keys and token provenance aligned with that transcript. One context owner handles automatic compaction, tool-result overflow, and reserve pauses.

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

- `engine/compaction.py` runs the summary job and selects the retained tail.
- `engine/rebind.py` replaces transcript tokens and segments through the worker.
- `binding_compaction.py` aligns compacted messages with the replaced transcript.
- `capabilities/context.py` defines the context-owner contract and shared fit check.
- `capabilities/compaction.py` implements the persistent history rewrite.
- `model_context.py` connects an open model stream to the context owner.
- Existing public classes remain thin facades for these focused modules.

---

## Task 1: Add transcript replacement and compaction binding

**Files:**
- Modify: `src/ds4py/engine/transcript.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/chat.py`
- Modify: `src/ds4py/binding.py`
- Create: `src/ds4py/engine/rebind.py`
- Create: `src/ds4py/binding_compaction.py`
- Create: `tests/engine/test_transcript_replace.py`
- Create: `tests/agent/test_binding_compaction.py`

**Interfaces:**
- Produces: `Session.replace_transcript(segments: Sequence[Segment], *, reason: str) -> Plan`.
- Produces: `MessageBinder.adopt_compaction(messages: Sequence[ModelMessage]) -> None`.
- Consumes: Existing transcript segments, message keys, content hashes, checkpoints, and sync plans.

- [ ] **Step 1: Write the failing tests**

Add tests for a replaced transcript and its binding. Use a system prefix, two user turns, tool output, and a summary segment.

```python
def test_replace_transcript_uses_longest_valid_checkpoint() -> None:
    result = session.replace_transcript(compacted_segments, reason="compaction")
    assert result.kind == "restore"
    assert result.checkpoint == "system"
    assert session.transcript.tokens == expected_tokens
    assert session.transcript.segments == compacted_segments


def test_adopt_compaction_preserves_retained_message_keys() -> None:
    binder.adopt_compaction(compacted_messages)
    assert retained_keys(binder) == original_tail_keys
    assert summary_segment(binder).meta["source"] == "compaction"
```

Add a test that the first system sync holds the `system` checkpoint. This checkpoint is in memory. It is not the deferred system-prefix disk cache.

Add cases for gaps, overlaps, stale keys, stale hashes, and a missing checkpoint. Invalid input must not change the transcript or binding.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/engine/test_transcript_replace.py tests/agent/test_binding_compaction.py -q
```

Expected: FAIL because transcript replacement and compaction adoption do not exist.

- [ ] **Step 3: Implement the replacement and binding operations**

Make `engine/rebind.py` validate contiguous segments from token zero. Run replacement through the existing worker. Hold the `system` checkpoint after the first system sync. Restore the longest valid checkpoint or rebuild from token zero.

Make `binding_compaction.py` match retained messages by key and content hash. Require one new summary message with `source="compaction"`. Keep token positions and sampled tail tokens unchanged.

Keep `Session` and `MessageBinder` as thin delegates. Do not add request-only views or paging state.

- [ ] **Step 4: Run the focused tests and make sure that they pass**

```bash
uv run --frozen --extra agent pytest tests/engine/test_transcript_replace.py tests/agent/test_binding_compaction.py tests/agent/test_binding.py -q
uv run --frozen --extra agent pytest tests/engine/test_plan.py tests/engine/test_worker.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/engine/transcript.py src/ds4py/engine/session.py src/ds4py/engine/chat.py src/ds4py/engine/rebind.py src/ds4py/binding.py src/ds4py/binding_compaction.py tests/engine/test_transcript_replace.py tests/agent/test_binding_compaction.py
git commit -m "feat: add compaction transcript rebinding"
```

---

## Task 2: Add one context owner and shared overflow handling

**Files:**
- Create: `src/ds4py/capabilities/context.py`
- Create: `src/ds4py/model_context.py`
- Modify: `src/ds4py/model.py`
- Modify: `src/ds4py/errors.py`
- Create: `tests/agent/test_context_manager.py`
- Modify: `tests/agent/test_model_cancel.py`

**Interfaces:**
- Produces: `ContextStrategy` with `owns_context_trigger: bool` and `make_room(ctx, needed_tokens)`.
- Produces: `ContextManager` in `session.state()` under the key `ds4.context`.
- Produces: `ContextManager.register(owner, *, compaction: bool) -> bool`.
- Produces: `ContextManager.open_response_key: str | None` for reserve compaction.
- Produces: `make_room_for_response(ctx, needed_tokens, *, response_key: str)` in `model_context.py`.
- Produces: `ContextExhausted`. Keep `ContextReserveError` as an alias.
- Consumes: `MessageBinder.count_tokens()`, `Session.ctx_used`, and Task 1 rebinding.

- [ ] **Step 1: Write the failing tests**

Use a test-only owner to prove registration and overflow behavior.

```python
async def test_tool_overflow_asks_owner_once() -> None:
    result = await invoke_after_tool_execute(oversized_result)
    assert owner.make_room_calls == [(run_context, needed_tokens)]
    assert result.content == fitting_result


async def test_missing_owner_raises_context_exhausted() -> None:
    with pytest.raises(ContextExhausted):
        await consume_reserve_paused_response()
    assert partial_text_was_returned is False
    assert partial_tokens_remain_in_transcript is True
```

Add tests for per-run capability copies and duplicate custom owners. A failed retry must return exactly `Tool error: tool result still does not fit`.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/agent/test_context_manager.py tests/agent/test_model_cancel.py -q
```

Expected: FAIL because owner registration and reserve recovery do not exist.

- [ ] **Step 3: Implement owner registration and overflow handling**

Define this protocol in `capabilities/context.py`:

```python
class ContextStrategy(AbstractCapability, ABC):
    owns_context_trigger: bool

    @abstractmethod
    async def make_room(
        self,
        ctx: RunContext[object],
        needed_tokens: int,
    ) -> list[ModelMessage] | None: ...
```

Store the active owner and current request data in session state. Register the compactor with `compaction=True` and a custom owner with `compaction=False`.

A custom owner replaces a registered compactor. A compactor stands down when `register()` returns `False`. Raise `ValueError` for two custom owners.

Make `make_room_for_response()` set `open_response_key` while the owner runs. Clear it in `finally`. Tool-result overflow leaves it as `None`.

Run one shared `after_tool_execute` fit check. Put async reserve coordination in `model_context.py`. Preserve the partial transcript when no owner can free space.

- [ ] **Step 4: Run the focused tests and make sure that they pass**

```bash
uv run --frozen --extra agent pytest tests/agent/test_context_manager.py tests/agent/test_model_cancel.py -q
uv run --frozen --extra agent pytest tests/agent/test_agent_factory.py tests/agent/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/capabilities/context.py src/ds4py/model_context.py src/ds4py/model.py src/ds4py/errors.py tests/agent/test_context_manager.py tests/agent/test_model_cancel.py
git commit -m "feat: add context overflow coordination"
```

---

## Task 3: Implement the engine summary job

**Files:**
- Create: `src/ds4py/engine/compaction.py`
- Modify: `src/ds4py/engine/events.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/decode.py`
- Modify: `src/ds4py/_native/fake.py`
- Create: `tests/engine/test_compaction.py`

**Interfaces:**
- Produces: `Session.compact(reason: str) -> CompactionResult`.
- Produces: `CompactionResult(summary: str, dropped: int, kept: int, tail_keys: tuple[str, ...])`.
- Produces: `CompactionStarted` and `Compacted` engine events.
- Consumes: Task 1 transcript replacement and the existing decode primitives.

- [ ] **Step 1: Write the failing tests**

Use the exact compaction prompt from `ds4_agent.c:9675–10229`. Assert its bytes in a fixture.

```python
@pytest.mark.parametrize(("ctx", "limit"), [(1024, 256), (16384, 2048), (65536, 4096)])
def test_summary_limit_matches_c(ctx: int, limit: int) -> None:
    assert summary_limit(ctx) == limit


def test_compaction_rebuilds_system_summary_and_verbatim_tail() -> None:
    result = session.compact("threshold")
    assert result.summary == expected_summary
    assert session.transcript.tokens == system_tokens + summary_tokens + original_tail_tokens
    assert result.tail_keys == expected_tail_keys
    assert result.dropped + result.kept == original_history_tokens
```

Add stop-token, maximum-summary, masked tool-syntax, and leaked DSML cases. A leaked DSML start must stop the summary and remove its incomplete text.

Add tail cases at user boundaries. Limit the tail to `min(ctx / 10, 50000)` tokens. Keep whole tool exchanges and any open assistant output.

Add failure and cancellation cases. Assert the original tokens and segments remain unchanged. Assert no `Compacted` event occurs.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/engine/test_compaction.py -q
```

Expected: FAIL because `Session.compact()` does not exist.

- [ ] **Step 3: Implement `CompactJob` and `Session.compact()`**

Build the summary prompt as `transcript[:bottom] + user(prompt) + assistant_prefix(no-think)`. Generate plain text with `argmax`.

Set the summary limit to `clamp(ctx / 8, 256, 4096)`. Stop at a normal stop token or a DSML special token. Never parse summary text as a tool call.

Select the tail at a user-token boundary. Replace the live transcript only after a valid summary exists. Use `system + user(summary) + verbatim tail`, then append any open assistant prefix and sampled tokens.

Emit `CompactionStarted` before summary tokens. Emit `Compacted(summary, dropped, kept)` only after transcript replacement succeeds. Treat `dropped` and `kept` as token counts.

- [ ] **Step 4: Run the focused tests and make sure that they pass**

```bash
uv run --frozen --extra agent pytest tests/engine/test_compaction.py tests/engine/test_context_reserve.py tests/engine/test_decode.py -q
uv run --frozen --extra agent pytest tests/engine/test_transcript_replace.py tests/engine/test_session.py tests/engine/test_worker.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/engine/compaction.py src/ds4py/engine/events.py src/ds4py/engine/session.py src/ds4py/engine/decode.py src/ds4py/_native/fake.py tests/engine/test_compaction.py
git commit -m "feat: add engine LLM compaction"
```

---

## Task 4: Add automatic `LLMCompaction` between requests

**Files:**
- Create: `src/ds4py/capabilities/compaction.py`
- Modify: `src/ds4py/capabilities/context.py`
- Modify: `src/ds4py/capabilities/__init__.py`
- Modify: `src/ds4py/binding.py`
- Modify: `src/ds4py/agent.py`
- Modify: `src/ds4py/runtime.py`
- Create: `tests/agent/test_compaction_capability.py`
- Modify: `tests/agent/test_agent_factory.py`
- Modify: `tests/agent/test_runtime.py`
- Modify: `tests/agent/test_noninteractive.py`

**Interfaces:**
- Produces: `LLMCompaction(session: Session, *, soft: float = 0.85)`.
- Produces: `DS4Compaction = LLMCompaction`.
- Produces: `compact_messages(messages, *, reason) -> tuple[list[ModelMessage], CompactionResult]`.
- Produces: `build_agent_components(..., context_strategy: ContextStrategy | None = None)`.
- Produces: `AgentRuntime(..., context_strategy: ContextStrategy | None = None)`.
- Default: One `LLMCompaction(soft=0.85)` owns automatic context management.
- Consumes: Task 1 binding, Task 2 owner registration, and Task 3 `Session.compact()`.

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
    compactors = [c for c in components if isinstance(c, LLMCompaction)]
    assert len(compactors) == 1
    assert compactors[0].soft == 0.85
    assert context_owner(session) is compactors[0]


def test_custom_strategy_replaces_only_automatic_owner() -> None:
    components = build_with_context_strategy(custom_strategy)
    assert context_owner(session) is custom_strategy
    assert manual_compactor(components) is not custom_strategy
```

Add a noninteractive FakeNative run that crosses the soft limit and continues with the compacted history.

Add cases for no safe tail, summary failure, binder failure, and a custom context owner. Every case must leave `ctx.messages` and the native transcript unchanged.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_capability.py tests/agent/test_agent_factory.py tests/agent/test_runtime.py tests/agent/test_noninteractive.py -q
```

Expected: FAIL because the capability and its default installation do not exist.

- [ ] **Step 3: Implement the capability**

Add a pure `should_compact()` helper for the two C trigger rules. Register with `compaction=True`. Run automatic compaction only when `register()` returns `True`.

Build the new history from system and instruction parts, one `ModelRequest` with `UserPromptPart(summary)`, and the retained tail. Give the summary a new `ds4.key`. Keep retained messages and their keys unchanged.

Emit `before_compact` before the engine job. Adopt the new binding and rewrite `ctx.messages[:]` only after a successful job. Restore the original native segments if binding adoption fails. Emit `compacted` after both changes succeed.

Set `request_context.messages` to the same compacted history. Keep configuration on the capability and mutable state in `session.state()`.

Create one `LLMCompaction` in `build_agent_components`. Let an optional custom strategy own automatic management. Keep the compactor available for manual compaction in Task 7.

- [ ] **Step 4: Run the focused tests and make sure that they pass**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_capability.py tests/engine/test_compaction.py tests/agent/test_context_manager.py -q
uv run --frozen --extra agent pytest tests/agent/test_binding_compaction.py tests/agent/test_system_notes.py -q
uv run --frozen --extra agent pytest tests/agent/test_agent_factory.py tests/agent/test_runtime.py tests/agent/test_noninteractive.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/capabilities/compaction.py src/ds4py/capabilities/context.py src/ds4py/capabilities/__init__.py src/ds4py/binding.py src/ds4py/agent.py src/ds4py/runtime.py tests/agent/test_compaction_capability.py tests/agent/test_agent_factory.py tests/agent/test_runtime.py tests/agent/test_noninteractive.py
git commit -m "feat: add automatic LLM history compaction"
```

---

## Task 5: Continue the same response after reserve compaction

**Files:**
- Modify: `src/ds4py/model.py`
- Modify: `src/ds4py/model_context.py`
- Modify: `src/ds4py/binding.py`
- Modify: `src/ds4py/binding_compaction.py`
- Create: `tests/agent/test_compaction_stream.py`
- Modify: `tests/agent/test_model_stream.py`

**Interfaces:**
- Consumes: `make_room_for_response(ctx, needed_tokens, response_key=...)` and `Session.compact("reserve")`.
- Consumes: `ContextManager.open_response_key` so `LLMCompaction` can retain the open assistant segments.
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

Pass the saved `RunContext` and response key to `make_room_for_response()`. Let the active owner compact with the partial assistant output present. The engine finds the open assistant prefix and sampled tokens at the transcript tail.

Keep the trailing assistant prefix, sampled tokens, response key, parser, and parts manager alive. Update binder offsets before calling `ReserveReached.resume(...)`. Do not call parser `finish()` at the reserve boundary.

Use `CompactionStarted` and `Compacted` to suppress summary text in the response. Count all generated tokens once in `output_tokens`. Count real prefill and cache reuse from every `Synced` event.

Call `ReserveReached.fail(...)` and raise `ContextExhausted` when recovery fails.

- [ ] **Step 4: Run the focused tests and make sure that they pass**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_stream.py tests/agent/test_model_stream.py tests/agent/test_model_cancel.py -q
uv run --frozen --extra agent pytest tests/agent/test_model.py tests/agent/test_compaction_capability.py tests/engine/test_context_reserve.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/model.py src/ds4py/model_context.py src/ds4py/binding.py src/ds4py/binding_compaction.py tests/agent/test_compaction_stream.py tests/agent/test_model_stream.py
git commit -m "feat: resume responses after reserve compaction"
```

---

## Task 6: Preserve compacted sessions across save, switch, and fork

**Files:**
- Modify: `src/ds4py/runtime.py`
- Modify: `src/ds4py/runtime_restore.py`
- Modify: `src/ds4py/binding.py`
- Create: `tests/agent/test_compaction_persistence.py`
- Modify: `tests/agent/test_conformance_persistence.py`

**Interfaces:**
- Consumes: Existing version 1 save records, `AgentRuntime.save()`, `switch()`, and `fork()`.
- Consumes: Compaction keys and provenance from Task 4.
- Produces: No new persistence format or public record fields.

- [ ] **Step 1: Write the failing tests**

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

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_persistence.py -q
```

Expected: FAIL on any compaction provenance that restore does not rebuild. If all cases pass, keep them as regression coverage and record that result.

- [ ] **Step 3: Fix the provenance gaps found by the tests**

Reuse the existing snapshot, save record, and segment restore paths. Match the summary and retained tail by key and content hash.

Do not require messages from old step runs to have segments in the current KV. Do not add paged-view rules. A fork changes only the session identity and creation time.

- [ ] **Step 4: Run the focused tests and make sure that they pass**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_persistence.py tests/agent/test_conformance_persistence.py -q
uv run --frozen --extra agent pytest tests/agent/test_store.py tests/agent/test_store_failures.py tests/agent/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/runtime.py src/ds4py/runtime_restore.py src/ds4py/binding.py tests/agent/test_compaction_persistence.py tests/agent/test_conformance_persistence.py
git commit -m "test: cover compacted session persistence"
```

If production fixes are needed, use `fix: restore compacted session provenance` as the commit subject.

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
- Consumes: `LLMCompaction.compact_messages(messages, reason="manual")`.

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

Keep `AgentRuntime.compact()` as a thin delegate to the compactor. Stop and await an active run before changing messages. Assign `self._messages` only after compaction succeeds.

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
