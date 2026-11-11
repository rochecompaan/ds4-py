# M4 Second Code-Quality Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix interrupted native restoration and binding interleaving, then consolidate reserve authority, usage, guards, fit recovery, and retained finalization without breaking existing owners or persistence.

**Architecture:** Worker-confined restoration has one fail-closed implementation. Strict reserve owners use an explicit operation handle and one-shot Session tickets; a deliberately limited compatibility adapter preserves legacy custom owners. Session owns binding transactions and atomic append groups, while shared attempt accounting and retained-wait utilities remove duplicated mechanics without absorbing caller policy.

**Tech Stack:** Python 3.14, exactly pinned Pydantic AI/harness, existing EngineWorker, asyncio/AnyIO, FakeNative, pytest, devenv and uv.

**Spec:** [Original design](../specs/2026-10-21-ds4-py-design.md), especially §§3, 10–11, 13, 15, 17 and 20; [prior remediation plan](2026-11-10-m4-review-remediation.md). Review input: `.superpowers/sdd/2026-10-21-m4-context-persistence/code-quality-review-2.md`. Verified triage: `/home/roche/.pi/agent/sessions/--home-roche-projects-satyrn-ds4-py--/subagent-artifacts/e5cd93de-7169-4b66-b8da-b2d84aa4a43a_reviewer_output.md`.

**Base/worktree:** `0e2b5cefc269f3c42c38ff42900c37d5e649ad77`, branch `m4-context-persistence`, this worktree. Baseline attested: **2070 passed, 17 deselected, 278 existing warnings**. Do not reset or use the main checkout.

## Global Constraints

- “Python 3.14 or later. Use 3.14 as the development default.”
- “Pin the harness to an exact version.” No dependency/ABI upgrade or second Python environment.
- “The token transcript is the engine's truth, the message list is the harness's truth, and `DS4Model` binds them.” Preserve exact sampled token spans and keys.
- “One engine worker thread makes every call into the engine.” Executor threads invoke the facade, not native handles.
- Session remains the sole live token/segment writer. Pure helpers return values; binder never restores Session private paused fields.
- Save-record format stays version 1; old step snapshots, canonical/request-view distinction, single model/session/tool loop and custom `make_room` signatures remain unchanged.
- Exactly one automatic owner; custom composition replaces automatic ownership only. Manual compaction remains concrete LLMCompaction.
- Keep exact trigger policies (`0.85`, reason-bearing reserve, `+64`, `+128`) separate from reason-free summary execution and one spare native slot.
- Preserve `before_compact` before planning, success-only `compacted`, exactly one terminal `compaction_finished`, and no attempt events for busy rejection.
- Preserve drain → committed Python publication → authority revocation → reserve decision → round reconciliation → exactly-once finalization. Never wait for owner drain while holding `_pull_lock`.
- Exact post-owner full-request measurement remains mandatory; failed owners force minimal trailing rejection and the specified `Tool error: tool result still does not fit` string.
- Use precise types/TypeIs and safe fixes. Run every project command from the worktree inside `devenv shell`. No generated native/build output or local state in commits.
- Behavioral changes require focused RED then GREEN; pure extractions use existing behavioral contracts first and afterward. Do not manufacture failures or add tests of private call structure merely to satisfy TDD. Mechanical formatting is a separate commit.
- Native/live parity, `/strip`, a second restored-context cutoff and previously documented feature deferrals remain outside scope.

## Review Focus

1. A restore can return INTERRUPTED without raising: no successful summary or reusable mismatched worker may escape (A).
2. A strict owner's child task, listener or stale executor must not borrow its reserve permission; legacy naked-to_thread compatibility must remain honestly documented (B).
3. Interleaving between binding boundaries or failure during retirement/provider publication must leave a valid committed prefix, never stale absolute record offsets (C, G).
4. Owner failure/cancellation before RoomResult must still charge actual summary work once; no-event custom usage and dishonest fit results remain compatible (D, E).
5. Repeated cancellation/concurrent close must drain the original work and publish committed history before finalization, preserving the primary error and subsequent usability (B, F).

---

## Approval-Sensitive Authority Trade-off

**This plan does not remove all ContextVars or isolate every existing custom owner.** Explicit authority cannot accompany an unchanged call such as `await asyncio.to_thread(session.replace_transcript, ...)`. Python copies the same ambient context into both owner and child executors, so the old heuristic cannot distinguish them.

The supervisor approved this concrete adaptation for user approval:

- Add `ContextRequest.operations: ReserveOperations | None = None` and additive `ContextStrategy.reserve_operations: bool = False`.
- Built-in LLMCompaction sets that property to True. It and opted-in custom owners receive a strict handle; raw Session mid-round operations reject in their window.
- Validate handle ownership **on the owning loop before executor submission**, then pass explicit, operation-specific one-shot authority to Session. Strict child→executor calls therefore reject before submission.
- Legacy custom owners default to their existing direct/naked-to_thread behavior through a localized compatibility shim. Its child-executor ambiguity remains; it is lifecycle compatibility, **not security isolation**.
- Move reserve-specific ContextVars/permission heuristics out of the generic lifecycle module into the reserve compatibility implementation. Keep one generic listener-delivery/reentrancy marker so inline listeners deliberately handed a handle also reject. Generic event delivery no longer copies/clears a reserve-specific variable.
- Keep nested same-loop event registration (`_loop_depth`): streaming and compaction legitimately share the bridge. Removing a valid nesting mechanism is not a goal.

A later removal of legacy mode is a separate compatibility decision, not an implementer shortcut in this plan.

## File Ownership and Sequence

- Create `engine/native_restore.py`: worker-only token restoration and fail-closed status handling.
- Create `engine/reserve.py`: strict ReserveOperations, opaque execution tickets, revocation and localized legacy adapter; no message knowledge.
- Create `engine/session_operations.py`: Session-owned operation claims/guard mechanics and binding-transaction handle. Session retains admission and live publication; do not grow `session.py` beyond project module-size limits.
- Modify `engine/session.py`, `engine/lifecycle.py`, `engine/transcript.py`: explicit authority, generic listener marker, staged append/transaction primitives and policy-preserving guards.
- Modify `binding.py`, `binding_plan.py`, `model.py`: transaction use, one canonicalization owner, incremental key publication and named partial-response publication.
- Create `summary_usage.py`: one shared event collector plus one named legacy report adapter; no response parsing or compaction policy.
- Modify `capabilities/context.py`, `capabilities/compaction.py`, `compaction_commit.py`: consume strict handles, shared accounting and explicit fit/rejection helpers.
- Modify `async_session.py`, `runtime.py`, `tui.py`: retained waiting mechanics only; keep caller-specific error/close/input policy.
- Modify `capabilities/compaction_tail.py`: optional behavior-preserving named planning steps.
- Modify `engine/events.py` and relevant development/agent documentation: accurate diagnostics and compatibility contracts.

Execute **A → B → C → D → E → F → G → H → I** sequentially with one writer. A is independently mergeable and addresses the critical native-state defect first. C uses B's operation admission; D precedes E; F uses B's retained strict operations; G follows C's publication boundary. H may be skipped as explicitly optional; I is a tiny independent documentation commit.

## Task A: One Fail-Closed Native Restoration Primitive (#3)

**Files:** Create `src/ds4py/engine/native_restore.py`; modify `engine/rebind.py`, `engine/compaction.py`. Tests: new `tests/engine/test_native_restore.py`; existing `tests/engine/test_compaction.py`, `test_replacement_restore.py`, `test_fatal_state.py`, `test_transcript_replace.py`.

**Interfaces:** `restore_native_tokens(native: SessionHandle, tokens: Sequence[int], *, valid: bool, poison: Callable[[BaseException], None]) -> None`. Worker-confined; it must not submit another worker job, observe cancellation or emit usage events. Rebind's facade wrapper submits it once; summarize calls it directly on its worker. No Session.poison method may recursively wait on that same worker.

- [ ] **A1 — Write the failing interrupted-restore regression.** `test_invalid_summary_restore_interrupted_fails_closed` seeds actual invalid native tokens `(31, 32)`, uses FakeNative summary `"ok"`, and makes only restoration sync return `SyncStatus.INTERRUPTED`. Assert prepare_summary raises EngineError, the worker is poisoned and rejects another job, Python history is unchanged and no public response tokens appear. Record the old probe's **926 private-summary tokens/native-side successful return** in the reproduction evidence, not as acceptable state.
- [ ] **A2 — Write native restoration matrix tests.** `test_restore_tokens_preserves_prefix_and_validity` covers empty/nonempty/full-capacity and valid/invalid inputs; capture native calls and assert empty rewind(0), nonempty sync DONE, full sync(tokens[:-1]) DONE then eval(last). `test_restore_tokens_poison_on_incomplete_or_failed_restore` injects INTERRUPTED/sync/eval/rewind failure and asserts poison/error context. Both summary fallback and replacement recovery run the matrix; snapshot-valid summary continues to prefer load_snapshot, and snapshot disposal/load failure remains fail-closed. Preserve actual native prefix when Python has unevaluated EOS.
- [ ] **A3 — Run RED:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/engine/test_native_restore.py tests/engine/test_compaction.py tests/engine/test_replacement_restore.py -q'`. Expected interrupted fallback exposes false success/new primitive tests fail; retain output.
- [ ] **A4 — Extract and migrate.** Invalid rebuild invalidates before rebuilding and reinstates invalid semantics afterward; valid restoration remains valid. Require DONE before any last-token eval. Any restoration exception/non-DONE poisons and raises. Keep the failed original operation in the chained exception/context, including when restoration also fails; successful restoration re-raises the original operation unchanged. Retain preferred summary snapshot restoration and guaranteed disposal; disposal/load failures must not leave a usable unknown worker.
- [ ] **A5 — Run GREEN:** same command plus `tests/engine/test_fatal_state.py tests/engine/test_transcript_replace.py`; complete static gate. Expect all pass and no recovery events charged as usage.
- [ ] **A6 — Commit:** `fix(engine): share fail-closed native token restoration`.

## Task B: Strict Reserve Operations with a Legacy Compatibility Adapter (#1)

**Files:** Create `src/ds4py/engine/reserve.py`; modify `engine/session.py`, `engine/lifecycle.py`, `async_session.py`, `capabilities/context.py`, `capabilities/compaction.py`, `compaction_commit.py`, `model.py`. Tests: `tests/engine/test_reserve_window.py`, new `tests/agent/test_reserve_operations.py`, existing `test_compaction_async.py`, `test_compaction_stream.py`, `test_context_strategy.py`, `test_compaction_runtime_cancel.py`. Document migration in `docs/development.md`.

**Interfaces:**
- `ContextRequest.operations: ReserveOperations | None = None`; `ContextStrategy.reserve_operations: bool = False`; LLMCompaction overrides True. Existing positional ContextRequest and owner signatures remain valid.
- `Session.reserve_window(event: ReserveReached, *, strict: bool = False) -> AbstractContextManager[ReserveOperations]`. Strict mode is selected explicitly by model from the owner property. Leaving the window, deciding or closing revokes unconsumed tickets.
- `ReserveOperations.prepare_summary(bottom: int) -> SummaryResult`, `replace_transcript(replacement: TranscriptReplacement, *, reason: str) -> Plan`: synchronous selected-owner methods; they validate current task/listener before issuing and consuming a ticket.
- `ReserveOperations.prepare_summary_async(bottom: int) -> SummaryResult` and `replace_transcript_async(replacement: TranscriptReplacement, *, reason: str) -> Plan`: retained/drained methods validating task **before** submission. Strict owners use these rather than naked-to_thread(handle.method).
- Session `prepare_summary(..., *, authority: ReserveAuthority | None = None)` and `replace_transcript(..., reason: str, authority: ReserveAuthority | None = None)` accept opaque session/pause/operation-bound one-shot tickets. Idle callers need no ticket. Tickets are issued by the handle, consumed at native-operation claim under the Session lock and cannot authorize ordinary mutation.
- Add optional `operations: ReserveOperations | None = None` to internal `LLMCompaction.compact_messages` and `commit_compaction`; make_room forwards request.operations. Threshold/manual idle calls retain None. Cancellation-safe native-result publication remains in compaction_commit, not in the handle.

- [ ] **B1 — Add RED strict boundary tests.** `test_strict_owner_child_executor_is_denied` uses a child that calls the async handle API and a child using raw Session via to_thread; both reject without native mutation. Test raw inline listeners, a listener deliberately handed the handle, raw owner Session calls in strict mode, child-direct calls, wrong Session/operation/reused ticket, wrong pause and stale executor/ticket after fail/resume/window exit/close. No denial may revoke a newer pause.
- [ ] **B2 — Add success/compatibility/drain tests.** `test_opted_in_owner_uses_explicit_operations` covers sync owner and retained async calls. Preserve unchanged legacy direct/naked-to_thread tests; explicitly characterize legacy child-executor ambiguity without advertising denial. Cancellation while strict replacement is committed still drains/publishes then revokes; summary and replacement remain mutually exclusive. Include commit-then-raise, concurrent close and reserve→save/switch exact provenance.
- [ ] **B3 — Run RED:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/engine/test_reserve_window.py tests/agent/test_reserve_operations.py -q'`. New strict ticket tests fail before implementation; unchanged legacy coverage stays green.
- [ ] **B4 — Implement handle/tickets and migrate built-in.** Validate task on-loop, then transfer only the issued ticket as an executor argument; no task/thread identity inference in strict execution. Remove reserve-specific imports/context manipulation from lifecycle.emit; preserve generic listener reentrancy protection and loop-depth ordering. Localize old ambient permission to legacy adapter only. Strict callbacks/children never gain a ticket. Keep permission alive through required drain/publication, then revoke before deciding. Document exact remaining ContextVars and legacy limitation.
- [ ] **B5 — Run GREEN:** prior command plus `tests/agent/test_context_strategy.py tests/agent/test_compaction_async.py tests/agent/test_compaction_stream.py tests/agent/test_compaction_runtime_cancel.py tests/engine/test_lifecycle.py`; static gate.
- [ ] **B6 — Commit:** `refactor(context): pass explicit reserve operation authority`.

## Task C: Session-Owned Binding Transaction and Guard Mechanics (#4, #5, #9)

**Files:** Create `src/ds4py/engine/session_operations.py`; modify `engine/session.py`, `engine/transcript.py`, `binding.py`, `binding_plan.py`, `async_session.py`, `capabilities/compaction.py`; adjust engine recovery callers only where the new poison seam is needed. Tests: new `tests/engine/test_binding_transaction.py`; existing `tests/agent/test_binding_projection.py`, `test_binding_rewrite.py`, `test_context_paused.py`, `tests/engine/test_session_chat.py`, `test_reserve_window.py`, `test_lifecycle.py`, `test_replacement_restore.py`.

**Interfaces:**
- `Session.apply_appends(replacement: TranscriptReplacement, *, retire_paused: bool = False) -> None`: staged Python-only append under one lock/idle check, preserving chat flag and Session-owned pause state. Validate kinds, coverage, copied metadata and resulting offsets before one publication. Appends do not synchronize native state.
- `Session.binding_transaction() -> AbstractContextManager[BindingTransaction]`: claim before real preparation planning and release in finally. Handle owns `truncate(token_count)`, `start_chat(tools=(), instructions=None)`, and `apply_appends(replacement, *, retire_paused=False, publish: Callable[[], None])`. The last method installs the staged group and calls synchronous binder/provider/key publication within the lock. Handle checks its expected prefix/revision before every operation. It never yields authority to arbitrary public mutation.
- Session operation state uses one explicit exclusive operation (`type OperationKind = Literal["binding", "summary", "replacement"]`) rather than adding a fourth boolean. `_blocked_reason(policy: Literal["open", "idle", "native", "close"], *, authority: ReserveAuthority | None = None, binding: BindingTransaction | None = None) -> str | None` and `_exclusive(kind: OperationKind, *, authority: ReserveAuthority | None = None) -> AbstractContextManager[None]` centralize policy-specific admission/claim/release; preserve the table below. Native worker/event waits release the mutation lock while retaining the operation claim.
- `Session.clear_cancellation() -> None`: callable only after retained work drains/no native operation owns the signal. Replace capability/async helper access to `_worker.cancel_event` with this method.
- Worker poison remains worker-confined: reuse A's injected `poison` callback for worker recovery. If facade cleanup needs `Session.poison(error: BaseException) -> None`, dispatch once there only; never call it synchronously from the worker. No capability imports EngineWorker internals.

**Guard policy to preserve:** open-only/on_worker reject closed; idle mutation/start/logits-hook reject closed/active round/any exclusive operation; summary/replacement reject closed/any exclusive operation or active round without valid B authority; decode-step additionally rejects paused stream/control; close is idempotent and rejects native/binding operations but permits an active round under existing orchestration; reserve-window rejects closed/wrong event/listener/existing claim. `plan`/`ctx_used` gain no idle restriction. Binding handle's own operations are the only allowed mutations while its claim exists; reserve authority cannot authorize append/truncate/checkpoint.

- [ ] **C1 — Add RED atomicity/interleaving tests.** `test_binding_boundary_cannot_interleave_external_append` deterministically gates between appends while a second thread attempts mutation; the old **record end 670/transcript length 671** divergence cannot occur—external write rejects or waits until a wholly published transaction. `test_binding_plan_cannot_apply_against_changed_prefix` proves stale subsequent absolute offsets reject before group installation. Include reentrant listener attempts during preamble sync and no event-delivery deadlock.
- [ ] **C2 — Add failure-prefix tests.** Invalid later segment/metadata staging and pause-retirement failure leave the entire group and all three paused fields unchanged; later-boundary failure retains earlier completed records/provider/key checkpoints. Inject provider.apply_preview failure before/after assignment and restore its checkpoint plus records/keys/group under the same publication boundary. Initial start_chat write-then-fail preserves the existing reusable native system prefix, with Python binding/provider uncommitted; retry notes exactly once. Verify no native call occurs for successful/failing pure append groups.
- [ ] **C3 — Run RED:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/engine/test_binding_transaction.py tests/agent/test_binding_projection.py tests/agent/test_context_paused.py -q'`. Expect interleaving/ownership regressions and new interfaces to fail; record deterministic gate evidence.
- [ ] **C4 — Implement staged Session publication and shared ownership.** Claim transaction before plan_request; execute initial truncate/start_chat and each append via handle. Stage Transcript state and complete pause state in Session, install group and publish binder/provider/keys synchronously; restore the last completed boundary on publication failure. Other round/mutation/close calls reject throughout the plan, including between groups. Release mutation lock for native preamble work and event delivery, but not the operation claim; stage prefix observations coherently. Counting remains unclaimed, pure and non-consuming. Remove binder reads/writes of private paused state and segment-by-segment live appends. Preserve caller-specific guard errors and paused sampling semantics.
- [ ] **C5 — Run GREEN:** prior command plus `tests/agent/test_binding_rewrite.py tests/engine/test_session_chat.py tests/engine/test_reserve_window.py tests/engine/test_lifecycle.py tests/engine/test_replacement_restore.py`, then static and portable gates. Existing native restoration failures stay poisoned; do not add native rollback to Python-only groups.
- [ ] **C6 — Commit:** `fix(binding): apply boundaries in session-owned transactions`.

## Task D: One Attempt Summary-Usage Collector (#2)

**Files:** Create `src/ds4py/summary_usage.py`; modify `capabilities/context.py`, `capabilities/compaction.py`, `model.py`. Tests: existing `tests/agent/test_compaction_review.py`, `test_context_strategy.py`, `test_compaction_async.py`, `test_compaction_runtime_cancel.py`; add `tests/agent/test_summary_usage.py` only for uncovered collector/adapter outcomes.

**Interfaces:** `summary_usage(session: Session, charge: Callable[[int], None]) -> AbstractContextManager[SummaryUsageAttempt]`; `SummaryUsageAttempt.report_legacy(generated_tokens: int) -> None` is the single report-shortfall adapter for custom owners. Events immediately charge observed tokens; the adapter charges only a positive unobserved remainder once. Repeated reporting cannot double-charge. Keep compatibility arithmetic here, not in two callers. `RoomResult`/CompactionResult remain unchanged.

- [ ] **D1 — Run current usage contracts:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_compaction_review.py tests/agent/test_context_strategy.py tests/agent/test_compaction_runtime_cancel.py -q'`. These should already pass; duplicated accounting is not a claim of ordinary double charging.
- [ ] **D2 — Add only missing behavioral RED cases:** `test_summary_usage_keeps_observed_output_without_result`, `test_legacy_report_shortfall_is_charged_once` cover 13 emitted events plus report 13→13 total, 13 events plus report 17→17, no events/report 7→7, no RoomResult after error/cancel→observed total, repeated report→unchanged. Verify unsubscribe after drain and no subsequent attempt charged to an old sink. Reuse existing threshold/tool/reserve success/fail/cancel tests rather than duplicate a full matrix.
- [ ] **D3 — Implement shared collector.** Replace ContextStrategy._summary_usage and the reserve SummaryToken counter/shortfall in model with the same service, selecting ctx.usage versus response `_summary_tokens` sinks. Keep the reserve Synced subscription separate for replacement accounting. Install collector before owner work, remove only after awaited work/drain, and call report_legacy only on a returned result. Manual compaction gains no unrelated response sink; never subscribe a second collector inside LLMCompaction.
- [ ] **D4 — Run GREEN:** D1 command plus `tests/agent/test_summary_usage.py tests/agent/test_compaction_async.py`; static gate. Assert exact-once public output+summary usage and no response/parser summary text.
- [ ] **D5 — Commit:** `refactor(context): centralize summary attempt accounting`.

## Task E: Explicit Fit Entry Points and Shared Trailing Rejection (#6, #7)

**Files:** Modify `src/ds4py/capabilities/context.py`; tests `tests/agent/test_context_failed_owner.py`, `test_context_projection.py`, `test_context_notes.py`, `test_context_paused.py`, `test_compaction_capability.py`.

**Interfaces:** `_needed_for_request(messages: Sequence[ModelMessage], request_context: ModelRequestContext) -> int` always uses effective configured projection, including existing-chat changes. `_needed_for_chat(messages: Sequence[ModelMessage]) -> int` always uses strict bound projection with the existing uninitialized-after-tool EngineError. `_reject_until_fit(results: Sequence[ToolReturnPart], room: RoomResult, needed: Callable[[], int]) -> None` mutates the supplied live result parts, forcing one trailing rejection on explicit failed-owner result before checking size, then rejects only until complete projected fit; error-only overflow raises ContextExhausted. A small `_recover_room(ctx, needed_tokens) -> RoomResult` owns one make_room invocation and D's usage scope; no stateful recovery controller.

- [ ] **E1 — Run exact-fit contracts:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_context_failed_owner.py tests/agent/test_context_projection.py tests/agent/test_context_notes.py tests/agent/test_context_paused.py tests/agent/test_compaction_capability.py -q'`. Existing tests should pass; do not add method-name tests for extraction.
- [ ] **E2 — Extract the named entry points/recovery/rejection.** Pending grouped history is reselected after owner commit. Both call sites use the same rejection policy; after-tool wraps its single candidate part and returns original result or the specified tool error accordingly. Preserve exactly one owner invocation, configured preamble changes, sampled native spans, report-only custom usage, minimal trailing failure rejection and dishonest-True measurement. Do not replace necessary repeated exact projections with estimated subtraction or cached counts after mutation.
- [ ] **E3 — Verify uncovered boundary behavior only if absent:** capacity −1/0/+1 with notes/wrappers/assistant prefix, already-fitting owner-False and insufficient error replacements. Add RED before any behavior correction; retain unchanged GLM recovery expectations and five-case failed-owner tests. Run E1 GREEN and static gate.
- [ ] **E4 — Commit:** `refactor(context): share explicit fit and rejection mechanics`.

## Task F: Retained Waiting and Named Partial Publication (#10, #11)

**Files:** Modify `src/ds4py/async_session.py`, `runtime.py`, `tui.py`, `model.py`; tests `tests/agent/test_model_cancel.py`, `test_runtime_close_cancel.py`, `test_compaction_async.py`, `test_compaction_runtime_cancel.py`, `test_compaction_tui_async.py`, `test_context_request.py`; new `tests/agent/test_retained_wait.py` only for uncovered shared primitive outcomes.

**Interfaces:** `async def await_retained[T](future: asyncio.Future[T], *, on_cancel: Callable[[], object] | None = None) -> T`: await the same future under shield; on caller cancellation signal once, drain despite repeated cancellation and re-raise the caller cancellation after completion, chaining completion failure. AnyIO shielding prevents level-cancel spin. It does not choose whether caller suppresses an error or perform publication. `drain_operation` remains the no-rethrow-until-result primitive. `async def DS4StreamedResponse.publish_partial(self, run_context: RunContext[object] | None) -> None` awaits existing finish, gets finalized response and appends to canonical history only with sampled start and absent stable key.

- [ ] **F1 — Run existing cancellation/publication matrix first:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_model_cancel.py tests/agent/test_runtime_close_cancel.py tests/agent/test_compaction_async.py tests/agent/test_compaction_runtime_cancel.py tests/agent/test_compaction_tui_async.py tests/agent/test_context_request.py -q'`.
- [ ] **F2 — Add missing primitive/partial behavioral cases.** Repeated edge cancellation, AnyIO deadline, completion error and already-completed task preserve result/error/cancellation ordering; on_cancel fires once. `test_publish_partial_is_exactly_once` invokes publication twice with an existing/missing key, None context, pre-sampling failure and commit-then-raise/cancel; no duplicate history/keys. New primitives fail RED; do not weaken integration timing assertions.
- [ ] **F3 — Consolidate mechanics, preserve caller policy.** Runtime _stop_active retains a run-completion waiter rather than re-invoking the run; close retains one teardown owner and does not release admission/completion until actual teardown. Model _finish uses the existing cleanup task and keeps a primary exception from being masked. TUI retains a control-consumer task while preserving input/answer/cancel routing, signals the original operation on outer cancellation and drains it before returning; never strand controls/prompt work. Use await_retained for common shield/drain mechanics, not a universal exception-swallowing policy. Keep compaction commit's successful-result-before-publication handling distinct. `DS4Model.request` catches BaseException and delegates one-line partial publication; no inline stream-private checks remain.
- [ ] **F4 — Run F1 GREEN plus new primitive tests and all context commands/TUI tests; static and portable gates.** Prove drain→publication→revocation→decision→reconciliation→finalize ordering; concurrent close, saved response keys, original exception surfaces and real footer paint remain unchanged. No new cancellation-intent signal (#13) in this task.
- [ ] **F5 — Commit:** `refactor(async): share retained waits and partial publication`.

## Task G: Incremental Key Publication and One Canonicalization Owner (#12)

**Files:** Modify `src/ds4py/binding.py`, `binding_plan.py`, `model.py`; tests `tests/agent/test_binding_projection.py`, `test_binding_rewrite.py`, `test_context_request.py`, `test_compaction_integration.py`.

**Interfaces:** Introduce binder-owned `PreparedMessages` with `targets: tuple[ModelMessage, ...]` (canonical publication targets) and `candidates: tuple[ModelMessage, ...]` (independent planned copies). `MessageBinder.prepare_messages(messages: Sequence[ModelMessage]) -> PreparedMessages` canonicalizes exactly once and isolates plan metadata. `MessageBinder.plan_prepared(prepared: PreparedMessages, *, tools: Iterable[object] = (), instructions: str | None = None, note_provider: DS4SystemNotes | None = None) -> BindingPlan` is the sole prepared-plan entry. Existing public prepare/plan/count inputs remain Sequence-compatible through this preprocessing owner, not an `already_canonical` flag. Add `MessageBinder.prepare_model_request(messages: Sequence[ModelMessage], params: ModelRequestParameters, *, note_provider: DS4SystemNotes | None = None) -> list[ModelMessage]`: claim C transaction, preprocess once, resolve instructions from canonical targets, apply through the same internal prepared-plan path, then return those targets for the response view. DS4Model uses this entry instead of separately canonicalizing and calling prepare. `_publish_keys(..., start_messages: int, end_messages: int)` publishes only the newly committed range at C's locked boundary.

- [ ] **G1 — Write deterministic work-count RED tests.** `test_binding_key_publication_visits_each_committed_message_once` instruments publication metadata copies only: historical 10→55 and 20→210 visits become **10 and 20**, not zero/copy aliasing. `test_model_prepare_canonicalizes_once` proves one preprocessing canonicalize call for direct prepare and complete model request, including a unique merged-request recovery; speculative count/plan remains pure. Do not assert timing or claim the whole planner is linear.
- [ ] **G2 — Pin failure/isolation behavior.** Later-boundary failure publishes keys only for completed prefix; unmatched suffix stays untouched, original metadata is not aliased to plan or _bound_messages. Initial unchanged prefix missing keys recovers them once; same-key rewrites and partial sampled responses remain exact. Preserve count/prepare equivalence, unchanged provider state and custom request-only view tests.
- [ ] **G3 — Run RED:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_binding_projection.py tests/agent/test_binding_rewrite.py tests/agent/test_context_request.py -q'`.
- [ ] **G4 — Implement range publication and shared preprocessing.** Counting prepares its own independent value without writing caller keys; live prepare uses the same canonical targets/copies as model response construction. Perform canonicalization inside C's claimed real preparation snapshot and avoid a second _transition pass over raw messages. Keep prefix publication incremental on failure, not an end-only batch. No change to stable-key/hash recovery rules.
- [ ] **G5 — Run G3 GREEN plus `tests/agent/test_compaction_integration.py`; static and portable gates.
- [ ] **G6 — Commit:** `perf(binding): publish new keys and canonicalize requests once`.

## Task H (Optional): Named Tail-Planning Steps (#8)

**Files:** `src/ds4py/capabilities/compaction_tail.py`; existing `tests/agent/test_compaction_capability.py`, `test_compaction_stream.py`, `test_compaction_commit.py`.

**Interfaces:** Preserve `plan_tail`/CompactionTail. Extract `_system_end(segments: Sequence[Segment]) -> int`, `_cut_index(records: Sequence[BoundMessage], positions: Mapping[str, int], target_tokens: int, message_count: int) -> int`, `_bottom_tokens(messages: Sequence[ModelMessage], cut_messages: int, starts: Mapping[str, int], open_start_tokens: int | None, total_tokens: int) -> int`, and `_note_start(bottom_tokens: int, segments: Sequence[Segment]) -> int`; keep existing _exchange_start and adopted-key validation. No enum, controller or blanket coercion.

- [ ] **H1 — Run focused suites before extraction:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_compaction_capability.py tests/agent/test_compaction_stream.py tests/agent/test_compaction_commit.py -q'`.
- [ ] **H2 — Extract only named steps and explicit units.** Preserve budget/user boundary policy, pending-request clamp, whole tool exchanges, notes before retained message, zero sampled/open assistant without retained bound messages and verbatim slicing. Existing integration tests are the verification contract; add a helper test only for an uncovered observable boundary, not for each new private name.
- [ ] **H3 — Run same suites and static gate GREEN.** Record if skipped; no production behavior changes hidden in cleanup.
- [ ] **H4 — Commit if executed:** `refactor(compaction): name tail-planning boundaries`.

## Task I: Accurate Summary Diagnostic Contract (#17)

**Files:** `src/ds4py/engine/events.py`; `docs/development.md` only if its event text repeats the ambiguity. Existing test `tests/agent/test_run_record.py`.

- [ ] **I1 — Change SummaryToken wording to:** “Internal summary-generation events; excluded from response/parser output, but observable on the session bus and in diagnostic run records.” Do not filter DS4RunRecord or change serialized bytes/hex/payload shape.
- [ ] **I2 — Verify:** `devenv shell -- bash -lc 'uv run --frozen --extra agent pytest tests/agent/test_run_record.py -q'` and static gate. Documentation-only change needs no manufactured RED or new behavior test.
- [ ] **I3 — Commit:** `docs(events): clarify summary diagnostic visibility`.

## Final Verification and Execution Handoff

- [ ] Run both complete gates after the final intended commit, not just subsystem tests:

```bash
devenv shell -- bash -lc 'uv run --frozen --extra agent prek run --all-files --show-diff-on-failure'
devenv shell -- bash -lc 'uv run --frozen --extra agent pytest -m "not build and not live" -q'
```

Expected: every static hook and portable test passes; record actual totals/warning categories against the 2070/17/278 baseline. Use the AGENTS.md isolated-HOME workaround only if personal catalogs contaminate results and disclose both attempts. Packaging/CI checks are unnecessary unless those files change.

- [ ] Run `devenv shell -- bash -lc 'git diff --check; git status --short'`; verify intended files/commits, no staged artifacts, no generated output. Keep mechanical formatting separate.
- [ ] Each implementation report in `.superpowers/sdd/2026-10-21-m4-context-persistence/` records exact commands, RED/GREEN evidence, atomicity/authority adaptations and residual risks. Whole-remediation review covers cross-task cancellation, provider-prefix publication, strict/legacy owners, restoration poisoning and diagnostic compatibility.
- [ ] Preserve branch/worktree for user review; no automatic merge/push. Do not claim native inference/live parity or complete strict isolation for legacy owners.

## All Review Findings: Adaptations and Deferrals

| Item | Disposition/task |
|---|---|
| 1 | B: strict explicit operations for built-in/opted-in owners; compatibility shim remains for legacy owners. Reject claiming token-only source compatibility or total ContextVar removal. |
| 2 | D: one collector, separate Synced observation and one legacy report adapter. Existing paths do not ordinarily double-charge; do not remove report-only custom usage. |
| 3 | A: critical interrupted fallback restore is fail-closed and shared. |
| 4 | C: atomic group plus whole binding claim, Session-owned pause state and provider-prefix publication. |
| 5 | C: one mechanics model; preserve distinct guard policies rather than one universal idle predicate. |
| 6–7 | E: explicit configured/bound entry points and named exact minimal rejection; remeasure full request after each necessary mutation. |
| 8 | H optional: decomposition only; existing adoption/keying already validates relevant metadata. |
| 9 | C/A: Session cancellation facade and worker-confined poison callback; reject worker→blocking facade recursion. |
| 10–11 | F: retained-wait mechanics and named partial publication, preserving caller errors and canonical exactly-once history. |
| 12 | G: deterministic incremental-copy/canonicalization reduction, not a claim all planning is linear. |
| 13 | Deferred P3: cancellation-count ambiguity is real after previously suppressed cancellation, ordinary-flow defect unproven. No new cancellation-intent API without separate approval/tests. |
| 14 | Deferred P3 naming/rationale cleanup; trigger values/attribution remain correct. Do not merge sizing policies. |
| 15 | Deferred: blanket planning offload rejected. A later pure snapshot planner requires its own owned-snapshot and deterministic planning/redraw contract; keep current paint yield and real PromptSession coverage. |
| 16 | Existing approved composition deferral retained; no concrete new defect. No trigger-policy redesign. |
| 17 | I: clarify diagnostic visibility, retain telemetry persistence. |

## Self-Review (Completed)

1. **Spec coverage:** native single-worker/private-summary restoration→A; custom extension compatibility→B/D/E; transcript/message/key ownership→C/G; async failure/persistence order→B/F; diagnostics→I. No format/dependency/new strategy feature added. All 17 review items map above.
2. **Step scan:** behavioral tasks name deterministic failing outcomes and focused commands; refactor tasks begin with existing contracts rather than artificial RED. Publication failure, initial native cache, repeated cancellation and stale-ticket decisions are explicit. Commit subjects and task files are assigned.
3. **Type/interface consistency:** B defines operations/tickets consumed by Session/compaction_commit; C defines the binding claim used by G; D report_legacy is the sole compatibility remainder adapter; E keeps named configured versus bound projections; F does not absorb successful-native-result publication. No task relies on RoomResult on error.
4. **Review Focus:** interrupted restore→A1/A2; child/listener/stale permission→B1/B2; stale/failed binding prefix→C1/C2/G2; no-result usage/dishonest fit→D2/E3; repeated close/cancel/order→B2/F2/F4. Each risk has behavioral evidence, not only an extraction assertion.
5. **Proportion/YAGNI:** nine reviewable commits, H optional; small mechanics files only where ownership crosses callers. No complete algorithm bodies, task factory interception, broad enum/controller, new telemetry filter, or blanket offload. Compatibility cost is stated prominently rather than hidden.

**Handoff:** The existing execution method is subagent-driven. Parent should present this plan for approval, explicitly including strict-versus-legacy authority behavior, and wait before product edits. After approval, use that preserved method with one writer and task review gates. Planning made no production/test edits and ran no suites.
