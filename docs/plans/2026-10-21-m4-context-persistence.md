# M4 standard compaction implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic and manual LLM compaction, which replaces old history with a model-written summary, and continue the current response after compaction.

**Architecture:** The engine writes a summary but does not choose messages. `LLMCompaction` chooses the message tail and builds the replacement. `Session` is the only writer for transcript replacement. A pure provenance step runs before the native change, so the commit needs no rollback protocol. Composition stores the automatic owner on `DS4Model` and registers that same object for capability hooks.

**Diagram:** See [standard compaction architecture](../specs/2026-10-21-ds4-py-design.md#standard-compaction-architecture) for the full class and module view.

**Tech Stack:** Python, Pydantic AI, the existing ds4 engine worker, `FakeNative`, pytest, devenv, and uv.

**Spec:** [`docs/specs/2026-10-21-ds4-py-design.md`](../specs/2026-10-21-ds4-py-design.md). Read sections 11, 13, 15, 17, and 23. This plan implements only the compaction requirements.

**Base:** Start at `5c98c6a` on the `m4-context-persistence` branch. Use `/home/roche/projects/satyrn/ds4-py/.worktrees/m4-context-persistence`. Do not edit the main checkout.

## Global Constraints

A reserve is token space kept for the next response. KV is the model's cached token state. Token provenance links token ranges to their source messages.

- Install `LLMCompaction(soft=0.85)` as the default context strategy. Do not add a `DS4Compaction` alias or a manual-compactor protocol.
- Do not implement `RollingWindow`, paging notices, recovery tools, or a port of `pi-context-paging` in this branch.
- Do not implement the 50,000-token reminder, the system-prefix cache, `Bookmarks`, `/list`, `/strip`, or `/del` in this branch.
- Use `MessageBinder.count_tokens()` and `Session.ctx_used` for exact token counts. Do not add text-length estimates.
- Use the existing engine worker, tokenizer, prompt builder, and decoder. Do not create another model session or tool loop for summaries.
- Keep message keys, message grouping, and tail selection out of the engine. The engine receives only a token offset for the summary prompt.
- Make `Session` the only writer for token and segment replacement. Keep existing bound-message snapshots for adoption and persistence.
- Store the automatic owner as `model.context_owner`. Do not store composition dependencies in `session.state()`.
- Register the selected owner itself for `after_tool_execute`. Do not add `ContextManager` or `model_context.py`.
- Keep the save-record format at version 1. Keep old step snapshots unchanged.
- Run project commands from the worktree inside `devenv shell`.
- Write a failing test before each behavior change. Make one focused commit after each task.
- Keep formatting changes in separate commits.

## Review Focus

- `/new` and `/switch` clear session state. Task 6 proves that automatic and manual compaction still use the live owner afterward.
- An unregistered hook cannot run. Task 4 and Task 6 prove that the selected capability receives real tool-result overflow calls.
- Native replacement can fail. Task 3 proves that provenance and messages remain unchanged, with no rollback writer.
- A tail cut must keep whole messages and tool exchanges. Task 6 covers message boundaries. Task 7 covers an unkeyed open assistant tail.
- Summary text must never become response text. Task 5 defines a separate `SummaryToken`. Task 7 tests usage without a timed event window.
- A compacted save must restore exact keys and provenance. Task 8 covers exact KV restore, fresh rendering, and fork.

## File Structure

- `engine/compaction.py` generates a summary up to a caller-supplied token offset.
- `engine/rebind.py` replaces transcript tokens and segments through the worker.
- `engine/chat.py` creates the `system` checkpoint after the first system-only sync.
- `binding_provenance.py` validates provenance and prepares pure record swaps.
- `binding.py` keeps the public binder facade and applies prepared records.
- `compaction_commit.py` orders provenance preparation, native replacement, record swap, and message assignment.
- `capabilities/context.py` defines the selected-owner contract and registered tool-result hook.
- `capabilities/compaction_tail.py` chooses whole messages and their token tail.
- `capabilities/compaction.py` implements automatic, overflow, reserve, and manual compaction.
- `model.py` owns the composition dependency and async reserve decision.
- Existing public classes remain thin facades for these focused modules.

---

## Task 1: Add native transcript replacement

**Files:**
- Modify: `src/ds4py/engine/transcript.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/chat.py`
- Create: `src/ds4py/engine/rebind.py`
- Create: `tests/engine/test_transcript_replace.py`
- Modify: `tests/engine/test_session.py`

**Interfaces:**
- Produces: `TranscriptReplacement(tokens: tuple[int, ...], segments: tuple[Segment, ...])`.
- Produces: `Session.replace_transcript(replacement: TranscriptReplacement, *, reason: str) -> Plan`.
- Consumes: Existing checkpoints, transcript segments, and sync plans.

- [ ] **Step 1: Write the failing tests**

```python
def test_replace_transcript_uses_longest_valid_checkpoint() -> None:
    result = session.replace_transcript(compacted_replacement, reason="compaction")
    assert result.kind == "restore"
    assert result.checkpoint == "system"
    assert session.transcript.tokens == compacted_replacement.tokens
    assert session.transcript.segments == compacted_replacement.segments
```

Add gaps, overlaps, stale token counts, and a missing checkpoint. Invalid input must not change native or transcript state. Test that `chat_ops.start_chat` syncs only the system segments and then holds the in-memory `system` checkpoint before user tokens.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/engine/test_transcript_replace.py tests/engine/test_session.py -q
```

Expected: FAIL because replacement and the first system checkpoint do not exist.

- [ ] **Step 3: Implement replacement and checkpoint creation**

Make `engine/rebind.py` validate contiguous segments from token zero. Run replacement through the existing worker. Restore the longest valid checkpoint or rebuild from token zero. Keep checkpoint creation out of this module.

In `chat_ops.start_chat`, sync only the initial system segments through `_job_events(SyncJob(...))`. Call `session.checkpoint("system")` after that sync succeeds. `Session` owns every token and segment write in the replacement path.

- [ ] **Step 4: Run the focused tests**

```bash
uv run --frozen --extra agent pytest tests/engine/test_transcript_replace.py tests/engine/test_session.py -q
uv run --frozen --extra agent pytest tests/engine/test_plan.py tests/engine/test_worker.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/engine/transcript.py src/ds4py/engine/session.py src/ds4py/engine/chat.py src/ds4py/engine/rebind.py tests/engine/test_transcript_replace.py tests/engine/test_session.py
git commit -m "feat: add native transcript replacement"
```

---

## Task 2: Extract shared provenance preparation

**Files:**
- Create: `src/ds4py/binding_provenance.py`
- Modify: `src/ds4py/binding.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/runtime_restore.py`
- Modify: `tests/agent/test_binding.py`
- Modify: `tests/agent/test_restore.py`

**Interfaces:**
- Produces: `BoundMessage` and `PreparedProvenance` with installed segments, records, and deep-copied bound messages.
- Produces: `prepare_provenance(messages, segments, *, loaded: int | None) -> PreparedProvenance`.
- Produces: `MessageBinder.records() -> tuple[BoundMessage, ...]` for tail planning and assertions.
- Produces: `MessageBinder.apply_restore(prepared: PreparedProvenance) -> None` and `apply_compaction(prepared) -> None`.
- Produces: `Session.install_segments(segments: Sequence[Segment]) -> None`.
- Keeps: `MessageBinder.validate_restore()` and `restore()` behavior.

- [ ] **Step 1: Write the failing characterization test**

Assert that restore prepares records without changing state, installs segments through `Session`, and then applies records. Keep the existing save/switch assertions for keys, hashes, order, ranges, and notes.

```python
def test_restore_uses_session_segment_writer() -> None:
    binder.restore(snapshot_messages, saved_segments)
    assert session.installed_segment_calls == [expected_segments]
    assert binder.export_segments() == saved_segments
```

- [ ] **Step 2: Run the test and make sure that it fails**

```bash
uv run --frozen --extra agent pytest tests/agent/test_binding.py tests/agent/test_restore.py -q
```

Expected: FAIL because `Session.install_segments()` and the extracted preparation API do not exist.

- [ ] **Step 3: Move validation without changing restore behavior**

Move `_restore_state` and its private record types into `binding_provenance.py`. Make preparation pure, including bound-message deep copies. Keep all current validation errors unchanged.

Make `Session.install_segments()` the only segment installer for a loaded payload. Make `binder.restore()` call preparation, then `Session.install_segments()`, then `apply_restore()`. The apply methods only assign prepared fields. Restore sets restored-session state; compaction preserves the current request signature. Neither method parses or validates.

Keep `validate_restore()` as the non-mutating check before a payload load. Do not remove `_bound_messages`; adoption and persistence still use it.

- [ ] **Step 4: Run the restore and binding tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_binding.py tests/agent/test_restore.py -q
uv run --frozen --extra agent pytest tests/agent/test_conformance_persistence.py tests/agent/test_store.py -q
```

Expected: PASS with no behavior change outside the new writer boundary.

- [ ] **Step 5: Commit the refactor**

```bash
git add src/ds4py/binding_provenance.py src/ds4py/binding.py src/ds4py/engine/session.py src/ds4py/runtime_restore.py tests/agent/test_binding.py tests/agent/test_restore.py
git commit -m "refactor: extract transcript provenance preparation"
```

---

## Task 3: Add an ordered compaction commit

**Files:**
- Create: `src/ds4py/compaction_commit.py`
- Create: `tests/agent/test_compaction_commit.py`

**Interfaces:**
- Produces: `commit_compaction(binder, messages: list[ModelMessage], compacted_messages: Sequence[ModelMessage], replacement: TranscriptReplacement) -> Plan`.
- Consumes: Task 1 replacement and Task 2 pure provenance preparation.

- [ ] **Step 1: Write the failing tests**

Record the call order and inject a native failure.

```python
def test_commit_prepares_before_native_replacement() -> None:
    plan = commit_compaction(binder, messages, compacted_messages, replacement)
    assert calls == ["prepare", "replace", "apply_compaction", "messages"]
    assert messages == compacted_messages


def test_native_failure_needs_no_rollback() -> None:
    with pytest.raises(EngineError):
        commit_compaction(binder, messages, compacted_messages, bad_replacement)
    assert messages == original_messages
    assert binder.export_segments() == original_segments
    assert binder.records() == original_records
```

Assert that preparation receives `len(replacement.tokens)`. The apply and message steps are pure assignments on concrete types. Do not inject failures into those steps or add a rollback API.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_commit.py -q
```

Expected: FAIL because the ordered commit does not exist.

- [ ] **Step 3: Implement the four-step pipeline**

Prepare and deep-copy provenance before any mutation. Call `Session.replace_transcript()` as the only fallible mutation. Apply the prepared binder records with `apply_compaction()`, then assign `messages[:] = compacted_messages`.

Do not capture old native state. Do not restore binder fields. Do not let the binder write `session._transcript`. A native failure occurs before the record and message writes, so no compensation is needed.

- [ ] **Step 4: Run the commit and replacement tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_commit.py tests/engine/test_transcript_replace.py -q
uv run --frozen --extra agent pytest tests/agent/test_binding.py tests/agent/test_restore.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/compaction_commit.py tests/agent/test_compaction_commit.py
git commit -m "feat: add ordered compaction commit"
```

---

## Task 4: Add one registered context owner

**Files:**
- Create: `src/ds4py/capabilities/context.py`
- Modify: `src/ds4py/model.py`
- Modify: `src/ds4py/errors.py`
- Create: `tests/agent/test_context_strategy.py`
- Modify: `tests/agent/test_model_cancel.py`
- Modify: `tests/agent/test_model_stream.py`

**Interfaces:**
- Produces: `ContextRequest(needed_tokens: int, reason: Literal["tool_result", "reserve"])`.
- Produces: `RoomResult(made_room: bool, generated_tokens: int = 0)`.
- Produces: `ContextStrategy.make_room(messages: list[ModelMessage], request: ContextRequest) -> RoomResult`.
- Produces: `ContextStrategy.after_tool_execute()` with the shared fit check.
- Produces: `DS4Model.context_owner: ContextStrategy | None`, initialized to `None`.
- Produces: `ContextExhausted`. Delete `ContextReserveError` instead of aliasing it.
- Consumes: Exact token counts and the existing async stream.

- [ ] **Step 1: Write the failing tests**

Register a test owner directly with a harness `Agent`. Send an oversized tool result through the real hook path.

```python
async def test_registered_owner_receives_tool_overflow() -> None:
    result = await run_tool(oversized_result)
    assert owner.requests == [ContextRequest(needed_tokens, "tool_result")]
    assert owner.message_lists == [run_messages]
    assert result.content == fitting_result


async def test_reserve_uses_same_owner_without_pull_lock() -> None:
    response = await consume_reserve_paused_response()
    assert owner.requests == [ContextRequest(1, "reserve")]
    assert owner.message_lists == [request_messages]
    assert owner.pull_lock_was_locked is False
    assert response.usage.output_tokens == response_tokens + summary_tokens
```

Add no-owner and failed-room cases. They must raise `ContextExhausted` or return exactly `Tool error: tool result still does not fit`. Add a cleanup-race case that never awaits the owner while `_pull_lock` is held. Replace the old `reserve_handler` and `ContextReserveError` tests.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/agent/test_context_strategy.py tests/agent/test_model_cancel.py tests/agent/test_model_stream.py -q
```

Expected: FAIL because the owner contract and async reserve path do not exist.

- [ ] **Step 3: Implement the owner contract and reserve decision**

Make `ContextStrategy` the registered capability. Its concrete `after_tool_execute` calls `make_room(ctx.messages, request)`. Its abstract `make_room` mutates that list in place and returns only `RoomResult`. Do not return a second message view.

Set `model.context_owner` through composition in Task 6. Pass the request message list from `_open_stream` into `DS4StreamedResponse`. The reserve path reads the same model field as the tool hook; it does not use session state.

Delete `reserve_handler`, its synchronous dispatch branch, and `ContextReserveError`. In `_get_event_iterator`, handle `ReserveReached` after `_pull()` releases `_pull_lock`. Await `owner.make_room(self._messages, request)`, add `generated_tokens` once, and resume with budget `1` when `made_room` is true. Otherwise fail the pause and raise `ContextExhausted`. Keep `_dispatch` synchronous. Cleanup fails a pause without awaiting owner code under the lock.

- [ ] **Step 4: Run the context and model tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_context_strategy.py tests/agent/test_model_cancel.py tests/agent/test_model_stream.py -q
uv run --frozen --extra agent pytest tests/agent/test_agent_factory.py tests/agent/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/capabilities/context.py src/ds4py/model.py src/ds4py/errors.py tests/agent/test_context_strategy.py tests/agent/test_model_cancel.py tests/agent/test_model_stream.py
git commit -m "feat: add registered context owner"
```

---

## Task 5: Add a summary-only engine job

**Files:**
- Modify: `src/ds4py/engine/compaction.py`
- Modify: `src/ds4py/engine/events.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/decode.py`
- Modify: `src/ds4py/_native/fake.py`
- Create: `tests/engine/test_compaction.py`

**Interfaces:**
- Produces: `Session.prepare_summary(bottom: int) -> SummaryResult`.
- Produces: `SummaryResult(summary: str, tokens: tuple[int, ...])`.
- Produces: `SummaryToken(id: int, text_bytes: bytes)`, separate from response `Token` and `TextDelta`.
- Consumes: The existing tokenizer, prompt builder, decoder, and worker.

- [ ] **Step 1: Write the failing tests**

Use the exact C prompt fixture and the C summary limits.

```python
def test_summary_uses_only_caller_token_prefix() -> None:
    before = (session.transcript.tokens, session.transcript.segments)
    result = session.prepare_summary(tail_start)
    assert result.summary == expected_summary
    assert summary_prompt_tokens == original_tokens[:tail_start] + prompt_suffix
    assert (session.transcript.tokens, session.transcript.segments) == before
```

Add stop-token, maximum-summary, masked tool-syntax, and leaked DSML cases. Add failure and cancellation cases. The native and transcript state must return to their original form.

Assert that `SummaryToken` never calls `handle_text_delta`. Do not add compaction-window state or a `Compacted` engine event.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/engine/test_compaction.py -q
```

Expected: FAIL because `Session.prepare_summary()` does not exist.

- [ ] **Step 3: Implement summary generation only**

Build `transcript[:bottom] + user(prompt) + assistant_prefix(no-think)`. Generate plain text with `argmax`. Use `clamp(ctx / 8, 256, 4096)` for the summary limit. Stop at a normal stop token or a DSML special token.

Return the summary and its generated tokens. Do not accept a reason, response key, messages, or tail keys. Do not build or install a replacement. Restore the native state that existed before generation before returning or raising.

- [ ] **Step 4: Run the engine tests**

```bash
uv run --frozen --extra agent pytest tests/engine/test_compaction.py tests/engine/test_context_reserve.py tests/engine/test_decode.py -q
uv run --frozen --extra agent pytest tests/engine/test_session.py tests/engine/test_worker.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/engine/compaction.py src/ds4py/engine/events.py src/ds4py/engine/session.py src/ds4py/engine/decode.py src/ds4py/_native/fake.py tests/engine/test_compaction.py
git commit -m "feat: add engine summary generation"
```

---

## Task 6: Add automatic compaction and stable composition

**Files:**
- Create: `src/ds4py/capabilities/compaction_tail.py`
- Create: `src/ds4py/capabilities/compaction.py`
- Modify: `src/ds4py/capabilities/context.py`
- Modify: `src/ds4py/capabilities/system_notes.py`
- Modify: `src/ds4py/capabilities/__init__.py`
- Modify: `src/ds4py/agent.py`
- Modify: `src/ds4py/runtime.py`
- Create: `tests/agent/test_compaction_capability.py`
- Modify: `tests/agent/test_agent_factory.py`
- Modify: `tests/agent/test_runtime.py`
- Modify: `tests/agent/test_noninteractive.py`
- Modify: `tests/agent/test_system_notes.py`

**Interfaces:**
- Produces: `CompactionReason = Literal["threshold", "tool_result", "reserve", "manual"]`.
- Produces: `CompactionResult(summary: str, dropped: int, kept: int, generated_tokens: int)`.
- Produces: `LLMCompaction(binder: MessageBinder, *, soft: float = 0.85)`.
- Produces: `compact_messages(messages: list[ModelMessage], *, reason: CompactionReason) -> CompactionResult`.
- Produces: `ContextComposition(owner: ContextStrategy | None = None)`.
- Produces: `AgentComponents.compactor: LLMCompaction`.
- Consumes: Task 3 ordered commit, Task 4 owner contract, and Task 5 summary generation.

- [ ] **Step 1: Write the failing tests**

Test both C trigger rules and the in-place rewrite.

```python
async def test_compaction_rewrites_live_history() -> None:
    result = await compaction.compact_messages(messages, reason="threshold")
    assert summary_request(messages).parts[0].content == result.summary
    assert retained_messages(messages) == expected_tail
    assert retained_keys(messages) == original_tail_keys
```

Test whole tool exchanges, user boundaries, and `min(ctx / 10, 50000)` tokens. Assert that the engine receives only `bottom`. It never receives messages, `ds4.key`, a reason, or a response key.

Add composition tests. The default registers one `LLMCompaction` and stores it on `model.context_owner`. A custom owner replaces it in the capability list and model field. `components.compactor` remains a concrete `LLMCompaction` for manual use. Use `ContextComposition` at both public composition boundaries.

Add real `/new`-then-compact and `/switch`-then-compact runs. Add a tool overflow run after each identity change. The same owner must receive the hook and reserve calls.

Add native summary and replacement failures. Messages and provenance must stay unchanged. Assert no `compacted` session event after failure.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_capability.py tests/agent/test_agent_factory.py tests/agent/test_runtime.py tests/agent/test_noninteractive.py -q
```

Expected: FAIL because the capability and stable composition do not exist.

- [ ] **Step 3: Implement message-tail planning and compaction**

In `compaction_tail.py`, choose whole messages through binder records and `ds4.key`. Keep whole tool exchanges. Return the tail messages, token start, verbatim token tail, and token counts. Detect a trailing unkeyed assistant prefix and sampled segments without passing message identity to the engine.

In `compact_messages()`, emit `before_compact` on the session event bus. Call `Session.prepare_summary(bottom)`. Build system and instruction parts, one summary `ModelRequest`, and the retained tail. Build a `TranscriptReplacement` and call Task 3 `commit_compaction()`. Emit `compacted` through the same capability and session bus only after success.

`make_room()` calls `compact_messages(messages, reason=request.reason)`. It returns `RoomResult` with the post-compaction fit result and generated token count. It never returns another message list.

`before_model_request()` compacts `ctx.messages` when either C trigger matches. Set `request_context.messages = ctx.messages` after the in-place commit. Manual compaction needs no request context.

- [ ] **Step 4: Implement composition and remove dead probing**

In `build_agent_components`, create the concrete compactor with `model.binder`. Select `context.owner` when supplied, else the compactor. Assign that object to `model.context_owner` and add only it to the capability list. Return the manual compactor through `AgentComponents.compactor`.

Pass one `ContextComposition` value through `AgentRuntime` and `build_agent_components`. Do not store the owner in session state. Do not scan capabilities to find it. The same `Session`, model, and binder survive identity changes, so no re-registration is needed.

Delete `DS4SystemNotes._compaction_subject` and both `getattr(..., "compaction_subject")` probes. Keep its session subscription to `compacted`. `LLMCompaction` is the only emitter of `before_compact` and `compacted`. Do not add a class alias or manual-compactor protocol.

- [ ] **Step 5: Run the compaction and lifecycle tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_capability.py tests/agent/test_context_strategy.py tests/engine/test_compaction.py -q
uv run --frozen --extra agent pytest tests/agent/test_agent_factory.py tests/agent/test_runtime.py tests/agent/test_noninteractive.py -q
uv run --frozen --extra agent pytest tests/agent/test_system_notes.py tests/agent/test_compaction_commit.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the task**

```bash
git add src/ds4py/capabilities/compaction_tail.py src/ds4py/capabilities/compaction.py src/ds4py/capabilities/context.py src/ds4py/capabilities/system_notes.py src/ds4py/capabilities/__init__.py src/ds4py/agent.py src/ds4py/runtime.py tests/agent/test_compaction_capability.py tests/agent/test_agent_factory.py tests/agent/test_runtime.py tests/agent/test_noninteractive.py tests/agent/test_system_notes.py
git commit -m "feat: add stable automatic LLM compaction"
```

---

## Task 7: Continue the same response after reserve compaction

**Files:**
- Modify: `src/ds4py/model.py`
- Modify: `src/ds4py/capabilities/compaction_tail.py`
- Create: `tests/agent/test_compaction_stream.py`
- Modify: `tests/agent/test_model_stream.py`

**Interfaces:**
- Consumes: The registered owner, in-place `make_room()`, and summary-only engine job.
- Produces: Reserve recovery with an unkeyed open assistant token tail.
- Produces: Usage that counts summary and resumed output once.

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

Assert that the summary prompt ends at the selected message tail. The open assistant prefix and sampled tokens remain unkeyed until normal response finalization. Assert exact `Synced` prefill and cache usage across compaction.

Assert that `SummaryToken` produces no text, thinking, or tool events. Do not use `CompactionStarted`, `Compacted`, or timing flags in `DS4StreamedResponse`.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_stream.py tests/agent/test_model_stream.py -q
```

Expected: FAIL because open-assistant compaction is not complete.

- [ ] **Step 3: Complete reserve compaction and usage accounting**

Keep the assistant prefix, sampled tokens, response key, parser, and parts manager alive. Preserve the trailing unkeyed native tail in `compaction_tail.py`. Final `bind_response()` assigns its key after generation, as it does today.

Complete the ordered commit before `ReserveReached.resume(1)`. Add `RoomResult.generated_tokens` once. Count real prefill and cache reuse from every `Synced` event. Keep recovery outside `_pull_lock` and keep `_dispatch` free of async work.

- [ ] **Step 4: Run the stream tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_stream.py tests/agent/test_model_stream.py tests/agent/test_model_cancel.py -q
uv run --frozen --extra agent pytest tests/agent/test_model.py tests/agent/test_binding.py tests/engine/test_context_reserve.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```bash
git add src/ds4py/model.py src/ds4py/capabilities/compaction_tail.py tests/agent/test_compaction_stream.py tests/agent/test_model_stream.py
git commit -m "feat: resume response after compaction"
```

---

## Task 8: Add compacted-session persistence regression coverage

**Files:**
- Create: `tests/agent/test_compaction_persistence.py`
- Modify: `tests/agent/test_conformance_persistence.py`

**Interfaces:**
- Consumes: Existing version 1 save records, `AgentRuntime.save()`, `switch()`, and `fork()`.
- Consumes: Shared provenance preparation and ordered compaction commit.
- Produces: Regression tests only. No production restore path or record fields.

- [ ] **Step 1: Write the regression tests**

Compact, save, and switch back. Assert the same messages, keys, segments, and zero prefill for an exact KV restore. Add missing-payload fresh rendering and a fork that keeps keys, messages, binding, and live KV. Assert old runs keep their pre-compaction history.

Do not require old step messages to have ranges in the current KV. Do not add paged-view rules.

- [ ] **Step 2: Run the regression tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_compaction_persistence.py tests/agent/test_conformance_persistence.py -q
```

Expected: PASS because compaction and restore share Task 2 provenance preparation. If a case fails, fix that shared code. Do not add a second restore path.

- [ ] **Step 3: Run related persistence tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_store.py tests/agent/test_store_failures.py tests/agent/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the regression tests**

```bash
git add tests/agent/test_compaction_persistence.py tests/agent/test_conformance_persistence.py
git commit -m "test: cover compacted session persistence"
```

---

## Task 9: Add manual `/compact`

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
- Consumes: `AgentComponents.compactor: LLMCompaction` and `compact_messages(messages, reason="manual")`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_compact_command_rewrites_current_history_once() -> None:
    output = await registry.execute("/compact")
    assert compact_reasons == ["manual"]
    assert runtime.messages == compacted_messages
    assert "Compacted" in output
```

Add no-argument validation, completion, and native failure cases. A failed command leaves history unchanged. Add an active-run case that uses the same stop-and-await rule as `save()`. Assert session `before_compact` and `compacted` events update the TUI without response text. Repeat manual compaction after `/new` and with a custom automatic owner.

- [ ] **Step 2: Run the tests and make sure that they fail**

```bash
uv run --frozen --extra agent pytest tests/agent/test_context_commands.py tests/agent/test_tui_commands.py -q
```

Expected: FAIL because the runtime method and command do not exist.

- [ ] **Step 3: Implement runtime compaction and the command**

Stop and await an active run. Delegate to `self.components.compactor.compact_messages(self._messages, reason="manual")`. The method mutates the owned list in place. Do not scan capabilities and do not assign a second message view.

Register `/compact` through `SlashCommandRegistry`. Reuse existing validation and error formats. Render progress and completion from the session event bus. Do not import TUI code from a capability.

- [ ] **Step 4: Run the command tests**

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

## Task 10: Document compaction and run the full gate

**Files:**
- Modify: `docs/development.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Automatic compaction from Task 6 and `/compact` from Task 9.
- Produces: User and developer documentation. No API changes.

- [ ] **Step 1: Update the documentation**

Document the 85 percent trigger, reserve recovery, persistent summary history, and `/compact`. State that old step snapshots retain removed history. Explain that exact KV restore needs no prefill. Do not describe deferred M4 features as complete. Do not add tests for documentation text.

- [ ] **Step 2: Run the focused and conformance suites**

```bash
uv run --frozen --extra agent pytest tests/agent tests/engine -m "not build and not live" -q
uv run --frozen --extra agent pytest tests/agent/test_conformance.py tests/agent/test_conformance_persistence.py -q
```

Expected: PASS. FakeNative results are not live-inference evidence.

- [ ] **Step 3: Run the complete static and portable gates**

```bash
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
uv run --frozen --extra agent pytest -m "not build and not live" -q
```

Expected: PASS. If personal skills affect agent tests, repeat the portable suite with an empty home:

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

- [ ] `LLMCompaction` is the only public class name, and no alias or identity-only protocol exists.
- [ ] The selected owner survives `/new` and `/switch`, and the registered capability receives tool overflow hooks.
- [ ] The 85 percent and reserve triggers match the C rules.
- [ ] The engine knows only the summary token prefix. Message-tail selection stays in the capability layer.
- [ ] Native replacement is the only fallible mutation in the ordered commit. No rollback protocol exists.
- [ ] Summary tokens never become response text, and usage counts them once.
- [ ] Reserve compaction continues the same `ModelResponse` without duplicate output.
- [ ] `/compact` works with a custom automatic owner and after identity changes.
- [ ] Save, switch, and fork preserve compacted messages, keys, and token provenance.
- [ ] No deferred M4 feature is included.
- [ ] Existing conformance tests and the complete portable gate pass.

## Plan Self-Review

This plan covers the compaction requirements in sections 11, 13, 15, 17, and 23. The tasks separate native replacement, provenance refactoring, ordered commit, owner composition, summary generation, and message-tail planning. Each behavior has focused verification. The design uses one owner dependency, one transcript writer, one provenance implementation, and one persistence format.
