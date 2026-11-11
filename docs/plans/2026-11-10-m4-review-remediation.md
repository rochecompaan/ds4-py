# M4 Code-Quality Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make request projection and binding share one transition algorithm, and make automatic/manual compaction responsive, authorized, cancellation-safe, and observable without changing persisted formats or custom-owner semantics.

**Architecture:** The binder plans a request once against exact live token spans and an isolated note-provider cursor; scratch counting and real preparation apply that same plan. Session owns structural validation, token-tail utilities, reserve authorization, and blocking native operations. The async compactor offloads only those operations, drains cancellation before publication/teardown, and reports a balanced attempt lifecycle; the registry, not TUI command-name checks, controls immediate busy routing.

**Tech Stack:** Python 3.14, pinned Pydantic AI/harness, existing EngineWorker, asyncio/AnyIO, FakeNative, pytest, devenv, uv.

**Spec:** [`docs/specs/2026-10-21-ds4-py-design.md`](../specs/2026-10-21-ds4-py-design.md), especially §§3–4, 10–11, 13, 15, 17, 20. Also read the [original M4 plan](2026-10-21-m4-context-persistence.md), the user review in `.superpowers/sdd/2026-10-21-m4-context-persistence/code-quality-review.md`, and verified triage at `/home/roche/.pi/agent/sessions/--home-roche-projects-satyrn-ds4-py--/subagent-artifacts/991fa378-7096-48e6-a1e6-0e1868c00837_reviewer_output.md`.

**Base/worktree:** `ea0a94acf2ca11021a77dcfa7c1b700bbf57396b`, branch `m4-context-persistence`, `/home/roche/projects/satyrn/ds4-py/.worktrees/m4-context-persistence`. Do not reset HEAD or edit the main checkout. Baseline attested portable result: 1816 passed, 17 deselected, 278 pre-existing warnings.

## Global Constraints

- “Python 3.14 or later. Use 3.14 as the development default.”
- “Pin the harness to an exact version.” No dependency or native ABI upgrade in this remediation.
- “One engine worker thread makes every call into the engine.” Executor threads drive the facade; they do not call native handles directly.
- “The token transcript is the engine's truth, the message list is the harness's truth, and `DS4Model` binds them.” Matched sampled spans remain verbatim token ids.
- “Make `Session` the only writer for token and segment replacement.” Helpers return immutable values, never mutate live transcripts outside Session.
- “Keep the save-record format at version 1. Keep old step snapshots unchanged.”
- “Do not create another model session or tool loop for summaries.” SummaryToken remains separate from response Token/TextDelta.
- Exactly one selected automatic owner is registered and stored on the model. Manual compaction always uses concrete `AgentComponents.compactor: LLMCompaction`, including with a custom automatic owner.
- Keep canonical harness history distinct from the model request view. No second authoritative history, blanket canonical-view substitution, owner signature changes, or compaction-phase flags on DS4StreamedResponse.
- Preserve C trigger policy: `soft=0.85`, reason-bearing trigger reserve, `+64`, and `+128` threshold margin. Execution uses the fixed reason-free summary prompt, one spare native slot, and its existing room guard. Sharing primitives does not make these policies identical.
- `RoomResult` reports the owner's attempted result and generated usage; exact post-owner fit verification remains mandatory. Preserve `Tool error: tool result still does not fit` and ContextExhausted behavior.
- `before_compact` precedes planning; `compacted` follows only successful commit. New `compaction_finished` terminates every started attempt, with outcome `compacted`, `noop`, `failed`, or `cancelled`. Busy rejection emits no attempt events.
- Run all project commands from the worktree inside `devenv shell`; never install another interpreter, environment, or hook runner. Use precise types, TypeIs not TypeGuard, safe fixes only, and no blanket suppressions.
- Each behavioral task starts with focused RED evidence, then minimal implementation and GREEN; one focused commit per task. Preserve behavioral assertions when replacing synthetic fixture setup. Formatting-only changes belong in separate commits.
- No new `/strip`, rolling windows, reminders, bookmarks, payload fields, second restored-context cutoff, or native/live parity claims. Existing documented deferrals remain.

## Review Focus

1. Pending/restored notes across several request boundaries must count exactly once and never mutate provider state while counting (A).
2. A custom owner may commit and then raise; reconciliation must find the moved open assistant and never include summary tokens (B, F).
3. Cancellation while a thread still runs must not leave late history mutation, deadlock event delivery, leak authorization, or poison the next run (B, F).
4. No-op, failure, and pending-question busy commands must restore footer state without consuming an answer or firing success-only note hooks (D, E).
5. An owner may report made_room=True without making enough room; grouped wrappers, sampled spans, notes, and exact-capacity boundaries must still be checked (A).

## File Structure and Dependencies

- `binding.py`: binder facade, named counting APIs, real plan application, summary-message rendering helper.
- Create `binding_plan.py`: shared pure binding-transition planner/data, replacing `binding_request.py`; no second count-specific transition walk.
- `capabilities/system_notes.py`: provider-owned preview cursor/checkpoints; live state application stays explicit.
- `capabilities/context.py`: single exact-fit helper, clarified history selection contract; custom owners retain request-view selection.
- `engine/transcript.py`: structural coverage validation and immutable slice/concat/open-tail observations.
- `engine/session.py`, `engine/rebind.py`, `engine/lifecycle.py`: pre-native validation, reserve authorization, safe event bridge, async-operation-compatible native lifecycle.
- `capabilities/compaction_tail.py`: whole-message planning with explicit units and one relative token tail value.
- `compaction_commit.py`: pure preparation, native replacement, then binder/messages publication; retain this boundary.
- `engine/compaction.py`: shared summary sizing/rendering primitives with separately named trigger and execution policies.
- `capabilities/compaction.py`: compaction attempt lifecycle and async orchestration, not scratch rendering internals.
- `tui_commands.py`, `tui.py`, `tui_render.py`: registry-owned busy metadata, event-only footer lifecycle, validated event payloads.
- Create `async_session.py`: one retained/shielded/drained async facade helper used for blocking compaction calls; no model/session clone.
- `runtime.py`, `model.py`: operation ownership and teardown coordination; no copied canonical state.
- Test additions stay in focused modules named below; existing coverage remains enabled.

Execute **A → B → C → D → E → F** for reviewable commits. B/C are logically independent of A; F requires B's authorization/tail interface, D's terminal protocol, and integration against A. Do not run writers concurrently in this worktree.

---

## Task A: One Exact Binding Projection (review 1–3)

**Files:** Modify `src/ds4py/binding.py`, `src/ds4py/capabilities/system_notes.py`, `src/ds4py/capabilities/context.py`, `src/ds4py/model.py`; create `src/ds4py/binding_plan.py`; delete `src/ds4py/binding_request.py`. Test `tests/agent/test_binding_projection.py` (new), existing `test_binding.py`, `test_binding_rewrite.py`, `test_context_notes.py`, `test_context_paused.py`, `test_context_request.py`, `test_context_strategy.py`, `test_restore.py`, `test_system_notes.py`.

**Interfaces:**
- `DS4SystemNotes.preview() -> NotePreview`: provider-owned isolated cursor supporting `pending(messages)`, consuming `notes(messages)`, `resync_from_segments(segments)`, and `checkpoint() -> NoteState`. The cursor includes settings, pending-compaction reset, and date/hints state, not subscriptions or Session ownership. `DS4SystemNotes.apply_preview(state: NoteState) -> None` assigns a prepared checkpoint without executing callbacks or reading the clock.
- `MessageBinder.plan_request(messages: Sequence[ModelMessage], *, tools: Iterable[object] = (), instructions: str | None = None, note_provider: DS4SystemNotes | None = None) -> BindingPlan` plans the same configuration transition as prepare. `BindingPlan` is a frozen value carrying BindingChange, keyed message copies, retained token prefix/truncation, rendered appends and resulting records/signature/restored-note state, plus note checkpoints at application boundaries. It holds no live mutable provider cursor.
- `MessageBinder.count_rendered_request(messages, *, tools=(), instructions=None) -> int` projects configured preparation through that planner, including assistant-prefix length. This is not text-length estimation or a lossily rerendered matched prefix.
- `MessageBinder.count_bound_request(messages: Sequence[ModelMessage]) -> int` projects against the actual current preamble and sampled spans. Raises EngineError if no usable initialized chat/binding exists; initialized zero-record chat and restored chat are valid. It accepts no ignored tools/instructions arguments.
- `MessageBinder.count_tokens(messages, *, tools=(), instructions=None) -> int` remains a compatibility facade for configured projection, with no `preserve_bound` parameter. Migrate every preserve-bound caller/test to the named bound API.
- `ContextStrategy._needed_tokens(messages: Sequence[ModelMessage], *, request_context: ModelRequestContext | None = None) -> int`: one exact projected-count + 1 reserve − configured capacity calculation. Before-request passes its context and explicitly selects configured projection with effective tool definitions/instructions if no chat exists. After-tool requires prepared chat, as real tool calls do; out-of-protocol invocation raises a clear EngineError. No guessed fallback configuration.

- [ ] **A1 — Add failing projection tests.** Fixed-clock FakeNative tests `test_pending_note_rewrites_unchanged_bound_request`, `test_fresh_configured_notes_match_prepare`, and `test_preview_consumes_hints_once_across_suffix_requests` reproduce triage's counted/actual pairs **1435/1528**, **676/1440**, **2113/1443**. Pin expected actual counts 1528, 1440, 1443 with the corresponding family/config fixture; record historical wrong values in comments, not accepted output. Each asserts count equals prepare's resulting transcript plus PromptBuilder assistant prefix. Snapshot messages/metadata, records, restored notes, live provider flags, transcript, native tokens, and sync count before repeated counts; all remain unchanged.
- [ ] **A2 — Add boundary/failure tests.** Cover restored saved notes, compaction-invalidated hints, two or more suffix request boundaries, changed tools/instructions, same-key edit, key recovery, canonical merge, sampled/public text differences, and paused Qwen **8191 vs 8192**. Cover fresh, initialized-zero-record, native restore, rendered restore, and `/new`. Add provider preview/clock failure and injected apply/native failure: consumption must match exactly the successfully applied prefix, never a speculative suffix. Planning failure changes nothing. Add dishonest custom owner made_room=True, made_room=False tool error, grouped wrappers, and capacity −1/0/+1.
- [ ] **A3 — Run RED.** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_binding_projection.py tests/agent/test_context_notes.py tests/agent/test_context_paused.py -q'`. Expect the reproduced count mismatches/new interfaces to fail; save output.
- [ ] **A4 — Implement preview and shared plan.** Move common-prefix, restored preamble, unchanged/pending-note rewrite, paused retirement, note resync/injection, and rendering decisions into one traversal. Generate rendered note content once on the preview cursor; real application must not re-call live `notes()` or clock. Use exact matched native spans. `prepare()` applies planned structural operations/appends, assigns records/signature at the matching boundaries, and applies the checkpoint for the prefix actually installed. A partial append must be truncated to the last completed planned boundary before propagating its failure; synchronize binder/provider to that boundary. Do not replace prepare with a full native sync or move round-owned assistant-prefix insertion into prepare. Scratch applies the same appends to _ScratchSession and projects the prefix without committing anything.
- [ ] **A5 — Migrate fit callers.** Replace `_needed_tool_tokens`'s rendered-delta algorithm with `_needed_tokens([*messages, candidate])`; use the same helper for grouped pending results and exact verification after make_room. Add generated summary usage once. Honor failed-owner tool-error semantics but never trust made_room=True without measuring. Re-select live grouped parts after a persistent commit. Synthetic after-tool tests establish a genuine prepared chat, not weakened expectations.
- [ ] **A6 — Run GREEN and gate.** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_binding*.py tests/agent/test_context*.py tests/agent/test_restore.py tests/agent/test_system_notes.py -q'`, then the full static gate below. Expected all pass; no count-time sync or state mutation.
- [ ] **A7 — Commit intended files:** `fix(binding): share exact request planning and fit accounting`.

## Task B: Token-Tail Ownership, Scoped Reserve Permission, Exact Reconciliation (review 5, 7–8, 10–12)

**Files:** Modify `src/ds4py/engine/transcript.py`, `engine/rebind.py`, `engine/session.py`, `engine/lifecycle.py`, `binding.py`, `binding_provenance.py`, `capabilities/compaction_tail.py`, `capabilities/compaction.py`, `capabilities/context.py`, `model.py`. Tests: new `tests/engine/test_reserve_window.py`; existing `tests/engine/test_transcript_replace.py`, `test_context_reserve.py`, `tests/agent/test_compaction_capability.py`, `test_compaction_stream.py`, `test_compaction_commit.py`, `test_compaction_integration.py`, `test_context_request.py`.

**Interfaces:**
- `validate_segment_coverage(segments: Sequence[Segment], token_count: int) -> None` in engine/transcript.py is the single contiguous-positive-range/coverage check, raising ValueError.
- `Segment.shifted(delta: int) -> Segment`; `TranscriptReplacement.slice(start_tokens: int, end_tokens: int | None = None) -> TranscriptReplacement`; `TranscriptReplacement.concat(*parts: TranscriptReplacement) -> TranscriptReplacement`. Relative ranges and metadata are preserved; invalid slice/concat inputs reject. No live write.
- `MessageBinder.render_request(message: ModelRequest, *, prefix: TranscriptReplacement) -> TranscriptReplacement` owns scratch/message rendering and returns the prefix plus request, no assistant prefix. Used for the summary user message; capability no longer imports _ScratchSession.
- `CompactionTail(snapshot: tuple[ModelMessage, ...], cut_messages: int, bottom_tokens: int, system_end_tokens: int, retained: TranscriptReplacement)`, with derived retained messages `snapshot[cut_messages:]`. Remove returned budget/open_start diagnostics; keep local open-assistant detection and tail_boundary unit tests.
- `Session.reserve_window(event: ReserveReached) -> AbstractContextManager[None]` authorizes only current selected-owner work for this Session/live pause. No custom-owner method signature changes.
- `Session.open_assistant_start() -> int | None` observes and validates the trailing **unkeyed** assistant-prefix/assistant/eos structure; returns first sampled-token offset (prefix end when zero sampled), or None if no open assistant exists. No message/key knowledge in engine.

- [ ] **B1 — Add failing structural tests.** Direct Transcript.replace/install and Session replacement reject gaps, overlap, reversed/zero-length ranges and incomplete coverage before native/Python mutation; empty valid replacement succeeds. Assert slice/concat exact ids, metadata and offsets, including notes immediately before retained message, empty tail, and long unkeyed assistant without retained bound messages. Test provenance translates structural ValueError to existing EngineError while retaining key/hash/order validation.
- [ ] **B2 — Add failing authorization/reconciliation tests.** Selected reserve owner can summarize/replace synchronously and via `asyncio.to_thread`; unrelated task, child task that inherits context, stale captured thread context after window exit, listener callback, nested summary, ordinary active round and expired/closed pause all reject. Permission is revoked before resume/fail/close. Open-tail tests cover zero tokens, repeated reserves, unchanged/custom-view owner, moved tail, commit-then-raise, cancellation and early close; exact response key, transcript/provenance, usage and summary exclusion remain correct.
- [ ] **B3 — Run RED.** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/engine/test_reserve_window.py tests/engine/test_transcript_replace.py tests/agent/test_compaction_stream.py -q'`. Expected new direct validation/authorization/boundary tests fail before implementation.
- [ ] **B4 — Centralize values/validation.** Transcript.replace validates before assignment; install and Session pre-native validation call the same coverage function. Delete rebind.validate's duplicate loop, retain rebind's worker responsibility. Provenance translates shared structural failures at its boundary, not a gratuitous public error-type migration. Replace hand-shifts with slice/concat and binder.render_request. Update tail fields/callers and test budget through retained boundaries, not diagnostic return fields.
- [ ] **B5 — Scope permission and report exact open-tail observations.** Replace `_reserve_boundary`'s blanket permission with authorization tied to Session, live event identity, owning asyncio task and a revocable token. Context propagation allows the owner's to_thread call, not another task or callback. Event delivery runs without this token, including cross-thread call_soon delivery. `_make_room_for_reserve` enters the window only around owner work; exit/revocation precedes every decision outcome. In finally, obtain `_sampled_start` from validated live open-tail structure even if the owner committed then raised. An invalid/missing open tail is an explicit EngineError, not silent subtraction; preserve the original exception during teardown where appropriate. Clarify reserve_messages selects an existing authority while make_room returns no second view. Keep the selector and custom default; do not add an exclusive RoomResult.tail_start dependency.
- [ ] **B6 — Run GREEN/gate.** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/engine/test_reserve_window.py tests/engine/test_transcript_replace.py tests/engine/test_context_reserve.py tests/agent/test_compaction*.py tests/agent/test_context_request.py -q'`, then full static gate. Expected pass with ordered commit and original custom-view tests intact.
- [ ] **B7 — Commit:** `refactor(context): centralize token tails and scope reserve authorization`.

## Task C: Shared Summary Sizing Primitives, Distinct Policies (review 9)

**Files:** Modify `src/ds4py/engine/compaction.py`, `engine/session.py`, `capabilities/compaction.py`; tests `tests/engine/test_compaction.py`, `tests/agent/test_compaction_capability.py`; documentation `docs/development.md` sizing clarification.

**Interfaces:** In engine/compaction.py expose `summary_budget(ctx: int) -> int` returning `max(256, min(ctx // 8, 4096))`; `summary_suffix(engine: Engine, prefix: Sequence[int], *, prompt: str = SUMMARY_PROMPT) -> tuple[int, ...]` renders user prompt plus no-think assistant prefix with the same prefix context. Define named `TRIGGER_SUMMARY_MARGIN = 64`, `SUMMARY_SPARE_TOKENS = 1`, and `TRIGGER_REASON` text at the owning policy. No reason field added to SummaryJob or Session.prepare_summary.

- [ ] **C1 — Add focused contract tests.** Assert budgets for ctx 1024→256, 8192→1024, 65536→4096. Preserve test_exact_c_triggers values, reason-bearing reserve text, +64/+128 margins and 0.85. Assert execution suffix never contains reason text, preserves fixed prompt, rejects room 255, allows 256, and caps output at min(room,budget). Compare shared suffix rendering against current exact family fixtures with nonempty prefix.
- [ ] **C2 — Run contracts before refactoring.** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/engine/test_compaction.py tests/agent/test_compaction_capability.py -q'`. New primitive tests fail; current exact-trigger tests stay green. No artificial production mutation needed for this behavior-preserving extraction.
- [ ] **C3 — Implement named primitives.** Session execution and capability trigger use shared budget/rendering. Trigger passes reason-bearing text and adds 64; execution uses fixed default text and one spare slot. Document why conservative trigger policy is intentionally not the private generation request.
- [ ] **C4 — Run same suites GREEN and static gate.** Exact trigger/output fixtures must remain unchanged.
- [ ] **C5 — Commit:** `refactor(compaction): share sizing primitives without changing policy`.

## Task D: Balanced Attempt Lifecycle and Validated Footer (review 6, 17)

**Files:** Modify `src/ds4py/capabilities/compaction.py`, `tui_render.py`, `tui.py`; tests `tests/agent/test_compaction_capability.py`, `test_context_commands.py`, `test_tui_render.py`; documentation `docs/development.md` event contract.

**Interfaces:** `CompactionOutcome = Literal["compacted", "noop", "failed", "cancelled"]`. `session.emit("compaction_finished", reason=reason, outcome=outcome)` occurs exactly once after every corresponding before_compact, in finally, after successful compacted if any. Existing compacted payload remains reason/old_tokens/new_tokens/generated_tokens.

- [ ] **D1 — Add RED lifecycle tests.** Success sequence before_compact→compacted→compaction_finished(compacted); no-op before_compact→finished(noop); planning, generation and replacement exceptions before_compact→finished(failed); task cancellation before_compact→finished(cancelled). Busy rejects without either attempt event. Notes subscribed to compacted reset only on success. Assert footer leaves compacting for every terminal outcome and preserves original idle/running status.
- [ ] **D2 — Add malformed payload tests.** All displayed compacted count fields reject missing, negative, bool, str and float values before any footer mutation; valid integers update ctx_used and compact text. compaction_finished validates outcome before restoring status. Keep run_end precedence: a late finish must not resurrect running after run_end.
- [ ] **D3 — Run RED:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_compaction_capability.py tests/agent/test_context_commands.py tests/agent/test_tui_render.py -q'`.
- [ ] **D4 — Implement the attempt try/finally.** before_compact stays before plan_tail. Set outcome only at defined terminal boundaries; cancellation is not a false compacted. Footer clears attempt state through compaction_finished, not a manual command reset. Remove FooterModel.finish_compaction public workaround and the dispatch finally `/compact` comparison; keep the separate pending-question comparison until E. Update event documentation in the same commit.
- [ ] **D5 — Run same tests GREEN plus `tests/agent/test_system_notes.py` and full static gate.** Assert one terminal event even when an internal error follows native commit; no fabricated success events.
- [ ] **D6 — Commit:** `fix(compaction): terminate every attempt through session events`.

## Task E: Registry-Driven Immediate Busy Routing (review 6)

**Files:** Modify `src/ds4py/tui_commands.py`, `tui.py`; tests `tests/agent/test_tui_commands.py`, `test_tui.py`, `test_context_commands.py`.

**Interfaces:** `_Command.immediate_while_running: bool = False`; `SlashCommandRegistry.register(..., immediate_while_running: bool = False) -> None`; `SlashCommandRegistry.is_immediate(line: str) -> bool` shares registry command-name parsing, handles whitespace/arguments, and returns False for ordinary text, unknown commands, malformed quoting and non-immediate commands. Command execution still owns detailed parse/usage errors. Register compact with True; all existing commands retain False unless separately approved.

- [ ] **E1 — Add RED generic routing tests.** Register a non-compact test command marked immediate and prove it routes while a question awaits. Keep `/compact` pending-question busy regression: no answer enqueued, no second operation, no attempt/progress events, no cancellation, subsequent normal answer works. `/cancel` and unregistered slash text keep existing answer handling; unmarked commands keep existing queue semantics. Include leading spaces, `/compact extra` usage rejection and malformed input.
- [ ] **E2 — Run RED:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_tui_commands.py tests/agent/test_tui.py tests/agent/test_context_commands.py -q'`.
- [ ] **E3 — Implement registry metadata/query and replace the remaining command-name comparison.** Consult is_immediate before `_answering`; do not route all slash-prefixed answers. No command identifier appears in generic TUI control flow or footer cleanup.
- [ ] **E4 — Run same suites GREEN plus static gate.** Verify ordinary answer/command behavior is unchanged.
- [ ] **E5 — Commit:** `refactor(tui): route immediate commands through registry metadata`.

## Task F: Cancellation-Safe Async Compaction (review 4, required parts of 13/17)

**Files:** Create `src/ds4py/async_session.py`; modify `src/ds4py/capabilities/compaction.py`, `compaction_commit.py`, `engine/session.py`, `engine/lifecycle.py`, `model.py`, `runtime.py`; test new `tests/agent/test_compaction_async.py`, plus existing `test_model_cancel.py`, `test_runtime.py`, `test_context_commands.py`, `test_compaction_integration.py`, `test_compaction_commit.py`, `test_compaction_stream.py`, `tests/engine/test_lifecycle.py`. Clarify async/busy behavior and pinned accessor in `docs/development.md` and beside the import in model.py.

**Interfaces:**
- `async def run_session_operation[T](session: Session, operation: Callable[[], T]) -> T` in async_session.py retains the executor task, awaits it shielded, signals session.cancel on caller cancellation, and drains under an AnyIO cancellation shield before teardown/re-raising. It never abandons an executor future and never acquires `_pull_lock`. It propagates B's scoped authorization into the thread, but not callbacks. Keep primitive operations separate from non-cancellable ordered publication.
- `async def commit_compaction(...) -> Plan` replaces the synchronous same-module interface with the same arguments. It prepares provenance on the event loop, offloads only native replacement, then applies prepared binder/messages on the event loop after replacement success; all callers/tests migrate. If cancellation races successful replacement, shield/drain and publish the matching prepared state before re-raising. If replacement failed, publish nothing. No await between successful native result observation and Python publication.
- Session event-loop delivery becomes reentrant only for the same loop/owner, with nesting counted; another loop still rejects. Events emitted during manual/threshold compaction have the same thread identity as streamed events.
- LLMCompaction holds one active-attempt task/guard for the entire attempt and its drain; concurrent automatic/manual attempts reject before before_compact. Runtime tracks a manual-compaction operation separately from `_running`; run/compact/save/new/switch reject while that operation exists. `cancel()` signals active manual compaction as well as a run. Close cancels then awaits/drains the operation before touching session state.

- [ ] **F1 — Add deterministic responsiveness RED tests.** Gate FakeNative summary prefill/argmax and replacement sync using threading events, not time-based guesses. While each gate is held, prove an asyncio heartbeat and queued footer consumer execute, compacting paints before completion, and all listener callbacks run on the owning loop. A listener observing Synced/Prefill must not deadlock on a Session lock held by the offloaded producer. Cover manual, threshold, tool-overflow and reserve entry points.
- [ ] **F2 — Add cancellation/overlap RED matrix.** Cancel during summary, before replacement, during replacement with both interrupted and already-successful native outcomes, and just before publication. Repeat task.cancel and AnyIO level deadline cancellation; race close_stream and runtime close. Assert threads/jobs drained, exact native/transcript/binder/messages agreement, one finalization/terminal event, no late writes after return, correct no-publication versus committed-publication rule, and cancellation flags cleared for a subsequent successful run. While manual summary is gated, run/compact/save/new/switch must each reject immediately without changing state; runtime.cancel must release the blocked native work. Use `asyncio.wait_for` bounds only as deadlock guards and release gates in finally.
- [ ] **F3 — Add regression coupling tests.** Re-run reserve→continue→save→switch for success/cancel/error; include owner commit-then-raise, custom request view, selected-owner task identity through executor, and unauthorized event callback. Verify same response key/parser/output parts, no summary leak, and exact generated/prefilled/cache usage once. Assert Synced replacement accounting is not counted a second time after resume.
- [ ] **F4 — Run RED:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_compaction_async.py -q'`. Expect heartbeat/ownership/lifecycle failures, not a hanging uncontrolled worker. Save reproduction output.
- [ ] **F5 — Implement retained/drained operations and ownership.** Extend the established _finish shielding pattern, rather than naked to_thread. Bind the event bridge around all compaction entry points without creating competing loop owners. Do not hold Session's mutation RLock across worker emissions that synchronously wait for loop listeners; reserve operation ownership under lock, release while waiting, and recheck/commit under lock. Ordinary idle/mutation guards reject overlap, cancel remains nonblocking, and cleanup awaits active compaction before reconciliation. Offload prepare_summary and native replacement only; note/provider, provenance preparation and binder/message publication stay on the loop. Runtime restores manual operation state in finally after full drain, not on cancellation arrival.
- [ ] **F6 — Integrate ordered async commit.** A cancellation observed during replacement cannot skip Python publication if Session replacement already committed. Distinguish worker result from caller cancellation, drain first, then synchronously publish or leave state unchanged according to that result. Emit D's terminal outcome after drain/publication; cancellation surface remains CancelledError, not an injected success or masked native failure. Keep authorization alive during required drain and revoke before deciding/resuming/closing the paused round. Add the pinned private harness-accessor rationale beside its import; no new accessor/dependency.
- [ ] **F7 — Run GREEN focused subsystem checks.** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_compaction*.py tests/agent/test_context*.py tests/agent/test_model_cancel.py tests/agent/test_runtime.py tests/engine/test_lifecycle.py tests/engine/test_context_reserve.py tests/engine/test_reserve_window.py -q'`. Expected pass, no leaked jobs, no new warnings.
- [ ] **F8 — Run both complete gates below on the final state; self-review then commit:** `fix(compaction): offload native work with cancellation-safe ownership`.

## Optional Task G: Remove the Test-Only Private Re-export (review 16)

**Files:** `src/ds4py/binding.py`, `tests/agent/test_tool_render.py` only. Independent of A–F; skip as a separate commit if A already necessarily removes the import.

- [ ] **G1 — Point the test's `_arguments` import at `ds4py.binding_provenance`; remove production re-export and F401 suppression.** No runtime behavior changes; no artificial failing test for an import-only cleanup.
- [ ] **G2 — Verify:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_tool_render.py -q'`, then full static gate.
- [ ] **G3 — Commit:** `test(binding): import argument helper from its owner`.

## Final Acceptance and Handoff

- [ ] Run from final HEAD after all intended changes (and G if used):

```bash
devenv shell -- bash -lc 'uv run --frozen --extra agent prek run --all-files --show-diff-on-failure'
devenv shell -- bash -lc 'uv run --frozen --extra agent pytest -m "not build and not live" -q'
```

Expected: every static hook and portable test passes; report actual counts and warning categories rather than reusing the 1816 baseline. If personal skill catalogs contaminate tests, use the AGENTS.md empty-HOME workaround and disclose both outputs. Do not run packaging gates unless packaging/CI files actually changed.

- [ ] Check `git diff --check`, intended file list, no staged/untracked product artifacts, and one focused commit per task. No generated native libraries/build output, venv, traces or devenv state committed.
- [ ] Whole-branch integration review focuses on five Review Focus cases and shared-plan count==bind equivalence, not file-size aesthetics. Record each addressed finding, RED/GREEN commands/results, controller adaptations, residual risk, and changed interfaces in `.superpowers/sdd/2026-10-21-m4-context-persistence/` reports.
- [ ] Return final commits, gate evidence and explicit residual risks. Preserve branch/worktree for user full review. Do not claim native inference/live replay parity or merge automatically.

## Confirmed Adaptations, Rejected Suggestions, and Deferred Work

- **1–3 → A:** remove duplicate transition logic and preserve_bound; retain a thin count_tokens compatibility facade and exact post-owner verification. Do not replace sampled ids with a public-text rerender or trust made_room=True alone. Fresh/unbound calls use explicit request config; out-of-protocol after-tool calls raise rather than guess.
- **4 → F:** reject bare to_thread; async cancellation needs retained futures, Session ownership, drain, publication and loop event delivery.
- **5 → B:** clarify make_room's no-second-view contract while keeping reserve_messages as selection of an existing authoritative list. Reject a second model-owned canonical history or blanket custom-owner canonicalization.
- **6 → D/E:** additive compaction_finished preserves before-planning and success-only events. Do not emit fake compacted on no-op; it would reset note state. Busy routing is registry metadata, not all slash answers.
- **7/8/10 → B:** values and coverage move to their owners; retain provenance key/hash/order checks and intentional error translation. CompactionTail currently has nine fields, not ten; remove returned diagnostics, not production open-tail detection.
- **9 → C:** share budget/rendering but retain distinct trigger/execution policies. Reject deleting reason reserve/+64 or adding reason to SummaryJob.
- **11/12 → B/F:** scoped permission, validated open-tail observation in finally; reject broad active-round permission or relying solely on RoomResult, which is absent after commit-then-raise.
- **13 deferred:** do not merge cleanup rejection and normal reserve recovery into a stateful controller. If F needs a common decision-failure helper, extract only that helper and preserve no-owner/owner-error/drain behavior. No separate cleanup project.
- **14 partially done/rejected:** delete binding_request in A because its algorithm is duplicated. Keep compaction_commit and engine/rebind as ordered cross-layer responsibilities; small file size alone is not grounds to fold them.
- **15 deferred:** retain ContextComposition optional surface; clarify manual LLM service in F docs. Reject adding manual compaction to custom strategy contract; it changes approved behavior.
- **16 optional G:** remove test-shaped production import, no behavioral scope growth.
- **17:** validate footer payloads in D and comment the pinned accessor in F. Defer optional partial-response helper extraction. A comment on FakeNative rewind may describe fake script bookkeeping, but must not assert native equivalence: real rewind delegates to backend and remains unverified. No native parity claim or unrelated fake rewrite.

## Self-Review (completed during planning)

- **Spec coverage:** §§11/13 exact binding, single owner, open-response continuation and events map to A/B/D/E/F; §10 structural/native invariants map to B/C/F; §§15/17 persistence/keys stay covered by B/F integrations with no format changes. All 17 review items are mapped above. Broader milestone features remain explicit non-goals.
- **Step scan:** each task has named files/interfaces, failing cases or extraction contracts, exact focused command, implementation boundary, GREEN gate and commit subject. F's cancellation publication and A's provider prefix-commit timing are explicit, not left as “handle cancellation/state.”
- **Type consistency:** tasks B/C/D define the exact interfaces consumed by F. Task B keeps RoomResult source-compatible; A's bound/fresh APIs take no ambiguous mode flag; only D adds a bus event. No new owner signature required.
- **Review Focus:** notes→A; dishonest fit→A; moved/failing owner→B/F; authorization/cancellation→B/F; footer/question/no-op→D/E. Every line has a named regression matrix.
- **Proportion:** six independently reviewable behavioral/refactor commits, one optional two-file import cleanup; no implementation bodies or generic scaffolding. Source navigation confirmed existing RenderTarget, provider, event bridge and worker boundaries.

**Execution handoff:** Subagent-driven execution was already selected in the parent session. Parent should present this plan for user approval and wait before production edits, then use that preserved execution method. No implementation, tests, commits, or new reviewer dispatch were performed by the planning agent.
