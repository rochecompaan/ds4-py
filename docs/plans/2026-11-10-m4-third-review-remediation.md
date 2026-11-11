# M4 Third-Review Ownership Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the M4 branch's fake binder seam, duplicate provenance model, split admission state, implicit history-lock protocol, and response-owned reserve recovery with real ownership boundaries.

**Architecture:** Persisted provenance has one value type. Binding plans consume an explicit state snapshot rather than a binder. `OperationState` owns admission state, `SessionHistory` owns history and its claimed mutation pipeline, and `ReserveSettlement` owns reserve recovery and accounting. Public facades stay thin while existing behavior remains pinned by regression tests.

**Tech Stack:** Python 3.14+, pydantic-ai, pytest, Ruff, ty, devenv + uv.

**Spec:** `docs/specs/2026-11-10-m4-ownership-remediation-design.md`

## Global Constraints

- Preserve save-record version 1 and old snapshot compatibility.
- Keep exact token projection and the one-token response reserve; never estimate from text length.
- Keep `Session` as the only public writer for token/segment replacement and the native-job boundary.
- Do not change compaction behavior, cancellation surfaces, partial-response publication, message keys, or tail selection.
- `binding_plan.py` must not import `MessageBinder`, including under `TYPE_CHECKING`.
- `blocked_reason` must not accept or inspect a `Session`; admission state and policy live together.
- Native sync/checkpoint delivery must run without the history mutation lock while retaining the exclusive claim.
- Write a failing test before each new API or behavior contract. Pure moves use existing regression tests as their contract and run them before and after the move.
- Run project commands from the worktree inside `devenv shell`; make one focused commit per task.

## Review Focus

- A restored version-1 snapshot uses the same validated provenance value at export, runtime persistence, and restore.
- Bound and configured projections still count the exact request that preparation applies, including notes and paused terminators.
- Close, active rounds, live reserve windows, and invalid/stale authority retain their current admission decisions.
- `Synced`, `Checkpointed`, and `Dropped` delivery cannot deadlock a listener attempting a rejected mutation.
- Reserve recovery still charges summary and replacement usage once, observes moved sampled tails, and refuses an overflowing recovered request.

---

### Task 1: Use the canonical persisted provenance value

**Files:**
- Modify: `src/ds4py/binding.py`
- Modify: `src/ds4py/binding_provenance.py`
- Modify: `src/ds4py/runtime.py`
- Test: `tests/agent/test_conformance_persistence.py`
- Test: `tests/agent/test_binding_projection.py`

**Interfaces:**
- Produces: `MessageBinder.export_segments() -> tuple[ds4py.store.SegmentProvenance, ...]`.
- Produces: `prepare_provenance(messages, segments: Sequence[Segment | SegmentProvenance], *, loaded: int | None)`.
- Removes: `ds4py.binding.SegmentProvenance`, `binding_provenance._Provenance`, and runtime's field-by-field conversion.

- [ ] **Step 1: Write the failing canonical-type test**

Add a conformance assertion that every exported value is a `store.SegmentProvenance` and that the exact tuple returned by `export_segments()` is the tuple placed in the runtime save record. Run it and confirm the binder's local dataclass fails the type assertion.

- [ ] **Step 2: Replace the duplicate model**

Import `store.SegmentProvenance` in binder and provenance restoration. Delete the local dataclass and Protocol, use `provenance.text` directly, and assign the exported tuple directly in runtime. Keep `Segment` support for transcript replacements.

- [ ] **Step 3: Verify provenance and persistence**

Run `uv run --frozen --extra agent pytest tests/agent/test_conformance_persistence.py tests/agent/test_binding_projection.py tests/agent/test_binding_rewrite.py -q`. Confirm the new test passes and existing export/restore comparisons remain green.

- [ ] **Step 4: Commit**

`refactor(binding): use canonical persisted provenance`

### Task 2: Make binding planning depend on explicit state

**Files:**
- Modify: `src/ds4py/binding_plan.py`
- Modify: `src/ds4py/binding.py`
- Modify: `src/ds4py/binding_provenance.py`
- Modify: `src/ds4py/capabilities/context.py`
- Modify: `src/ds4py/model.py`
- Test: `tests/agent/test_binding_projection.py`
- Test: callers of the removed count/plan methods

**Interfaces:**
- Produces: `BindingState`, a snapshot of records, bound messages, signature, restored flag/notes, session, and renderer.
- Produces: `plan_transition(state: BindingState, prepared: PreparedMessages, config: BindingConfig, provider: DS4SystemNotes | None, *, system_changed: bool) -> BindingPlan`.
- Produces: `MessageBinder.project_request(messages, *, tools=(), instructions=None, bound: bool = False) -> int`.
- Keeps: `prepare(...)` for plan-and-apply and `prepare_model_request(...)` as the harness adapter.
- Removes: `plan_request`, `plan_prepared`, `count_rendered_request`, `count_bound_request`, `count_tokens`, and every binder private access from `binding_plan.py`.

- [ ] **Step 1: Write failing state-planner and projection tests**

Construct a `BindingState` from a real session without constructing a second binder and prove `plan_transition` projects the same request as binder preparation. Add exact-count tests for `project_request(..., bound=False)` and `project_request(..., bound=True)`, including the existing notes, paused-response, and uninitialized-bound-chat cases. Confirm they fail because the state API and projection verb do not exist.

- [ ] **Step 2: Introduce the explicit state boundary**

Move key recovery helpers that need only messages/records into `binding_provenance.py`. Pass `BindingState` to the planner and remove `MessageBinder` imports and binder method calls. Keep state snapshots immutable from the planner's perspective; application still publishes each `BindingBoundary` through the binder.

- [ ] **Step 3: Decompose the transition**

Extract prefix selection, boundary construction, and candidate append/note rendering helpers. Replace the `nonlocal published` closure with an explicit boundary builder value, remove `PLR0912/PLR0915` suppressions, and implement replacement slicing with `TranscriptReplacement.slice()` and `Segment.shifted()` rather than hand-shifted offsets.

- [ ] **Step 4: Collapse projection entry points**

Implement `project_request` with an explicit `bound` option that rejects bound projection before chat initialization. Update production and test callers; delete the seven-method plan/count family rather than retaining aliases.

- [ ] **Step 5: Verify binding behavior**

Run `uv run --frozen --extra agent pytest tests/agent/test_binding_projection.py tests/agent/test_binding_rewrite.py tests/agent/test_context_notes.py tests/agent/test_context_projection.py tests/agent/test_context_request.py tests/agent/test_reserve_operations.py -q`.

- [ ] **Step 6: Commit**

`refactor(binding): plan transitions from explicit state`

### Task 3: Move admission state into OperationState

**Files:**
- Modify: `src/ds4py/engine/session_operations.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/reserve.py`
- Test: `tests/engine/test_reserve_window.py`
- Test: `tests/engine/test_context_reserve.py`
- Test: `tests/engine/test_session.py`

**Interfaces:**
- Produces: `OperationState` ownership of closed, active-round, reserve-event, and reserve-operations state, with explicit transition methods.
- Produces: `blocked_reason(state: OperationState, policy: GuardPolicy, *, authority=None, binding=None) -> str | None`.
- Produces: `_POLICIES: dict[OperationKind, tuple[GuardPolicy, ReserveOperation | None]]` used by `exclusive`; no `policy_for` cast.
- Removes: Session-owned duplicates of those four admission fields.

- [ ] **Step 1: Write failing direct-state admission tests**

Exercise closed, open-round, claimed-native, live-reserve, and no-live-window states through `blocked_reason(OperationState(), ...)`. Cover explicit authority consumption and legacy admission through the existing reserve-window tests. Confirm the new direct-state calls fail against the Session-based signature.

- [ ] **Step 2: Move state and policy**

Add explicit closed/round/reserve transitions to `OperationState`; update Session round, close, reserve decision, and reserve-window paths to call them. Make `exclusive` evaluate `OperationState` and the `_POLICIES` mapping. Remove `Session._reserve_authorized` and the cast rather than forwarding the same policy back to Session.

- [ ] **Step 3: Verify admission and reserve compatibility**

Run `uv run --frozen --extra agent pytest tests/engine/test_reserve_window.py tests/engine/test_context_reserve.py tests/engine/test_session.py tests/agent/test_reserve_operations.py -q`.

- [ ] **Step 4: Commit**

`refactor(session): colocate admission state and policy`

### Task 4: Give SessionHistory ownership of the mutation pipeline

**Files:**
- Create: `src/ds4py/engine/history.py`
- Modify: `src/ds4py/engine/session_operations.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `src/ds4py/engine/chat.py`
- Modify: `src/ds4py/engine/rebind.py`
- Modify: affected native/checkpoint callers and tests
- Test: `tests/engine/test_binding_transaction.py`
- Test: `tests/engine/test_session_chat.py`
- Test: `tests/engine/test_transcript_replace.py`

**Interfaces:**
- Produces: `SessionHistory`, owning `Transcript`, pause state, checkpoint tokens, mutation lock, `OperationState`, and `BindingTransaction` operations.
- Produces: claimed truncate/append-group/install/replace/start-chat methods used by `Session` and binding.
- Keeps: `Session` as the public facade and the owner of native job submission/event publication.
- Removes: `_truncate_python`, `_commit_appends`, `_checkpoint_native`, `_drop_checkpoint`, and duplicate public start-chat lock policy from `Session`.

- [ ] **Step 1: Write the failing public start-chat lock test**

Adapt the bounded `Synced`-listener mutation probe so plain `Session.start_chat()` retains its claim, rejects the listener's mutation, and never leaves that listener blocked on the producer's mutation lock. Confirm the old public path blocks or violates the claimed-path contract.

- [ ] **Step 2: Move history state and transaction ownership**

Construct `SessionHistory` in `Session`; move transcript/view, pause/checkpoint state, mutation lock, and operation state behind it. Move `BindingTransaction` and the claimed truncate/append/install/replace pipeline into `history.py`. Update chat/rebind callers to use the history owner rather than Session duplicate fields. Do not leave policy helpers that reach back into Session.

- [ ] **Step 3: Unify preamble and checkpoint paths**

Route public `start_chat` through the transaction's stage/sync/set-chat sequence. Move checkpoint/drop operations into the history owner, keeping native calls outside the mutation lock and retaining the exclusive claim across gaps. Keep rollback and revision checks unchanged.

- [ ] **Step 4: Verify history and native-state contracts**

Run `uv run --frozen --extra agent pytest tests/engine/test_binding_transaction.py tests/engine/test_session_chat.py tests/engine/test_transcript_replace.py tests/engine/test_compaction.py tests/engine/test_context_reserve.py tests/agent/test_model_cancel.py -q`.

- [ ] **Step 5: Commit**

`refactor(session): centralize owned history mutations`

### Task 5: Extract reserve settlement from response translation

**Files:**
- Create: `src/ds4py/reserve_settlement.py`
- Modify: `src/ds4py/model.py`
- Test: `tests/agent/test_reserve_operations.py`
- Test: `tests/agent/test_context_failed_owner.py`
- Test: `tests/agent/test_model_stream.py`
- Test: `tests/agent/test_model_cancel.py`

**Interfaces:**
- Produces: `ReserveSettlement.recover(event: ReserveReached) -> None`, owning owner selection, reserve window, history selection, recovery, exact fit verification, and resume/fail.
- Produces: settlement-owned accumulated summary/prefill/cache usage read during response finalization.
- Keeps: response callbacks only for reconciled-state observation and moved-tail observation where response ownership requires them.
- Removes: `_recover_reserve`, `_make_room_for_reserve`, `_charge_summary_token`, `_reserve_needed`, and the response's owner-only `Synced` subscription.

- [ ] **Step 1: Establish the behavioral contract**

Run the focused reserve tests before the move. Where a test reaches into the removed response helpers, rewrite it first against `ReserveSettlement.recover` and confirm it fails because the controller does not exist. Do not change externally visible expected usage, messages, exceptions, or resume values.

- [ ] **Step 2: Move recovery and accounting**

Implement the controller with explicit collaborator arguments. Keep summary and sync subscriptions scoped to the awaited owner attempt. Accumulate replacement prefill/cache and summary tokens in the controller; response finalization adds those totals instead of listening to another operation's events.

- [ ] **Step 3: Reduce response reserve handling**

The event iterator delegates recovery to the controller. Drain/close paths use one controller settlement method for an undecided pause. Delete duplicate reserve fields and helper layers from the response while preserving cancellation and moved-tail reconciliation.

- [ ] **Step 4: Verify recovery, accounting, and teardown**

Run `uv run --frozen --extra agent pytest tests/agent/test_reserve_operations.py tests/agent/test_context_failed_owner.py tests/agent/test_context_strategy.py tests/agent/test_model_stream.py tests/agent/test_model_cancel.py -q`.

- [ ] **Step 5: Commit**

`refactor(model): extract reserve settlement controller`

### Task 6: Finish the scoped small cleanups

**Files:**
- Modify: `src/ds4py/compaction_commit.py`
- Modify: `src/ds4py/capabilities/context.py`
- Modify: `src/ds4py/binding.py`
- Modify: `src/ds4py/engine/reserve.py`
- Modify: `src/ds4py/model.py`
- Modify: `src/ds4py/runtime.py`
- Test: existing exact-overflow, compaction, reserve, runtime, and conformance tests

**Interfaces:**
- Produces: one binder `reserve_overflow(projected_tokens: int) -> int` using the one-token response reserve.
- Removes: list-as-boolean compaction commit state, triplicated overflow arithmetic, nested legacy conditional, and split runtime admission messages.
- Adds: a clear import-time failure if the exactly pinned pydantic-ai private run-context accessor disappears.

- [ ] **Step 1: Write failing shared-overflow boundary tests**

Test `reserve_overflow(ctx)` below, at, and above the exact boundary and update context/model callers to use it. Confirm the helper is absent before implementation. Keep the existing recovered-request exact-fit regression unchanged in intent.

- [ ] **Step 2: Apply the mechanical cleanups**

Use `plan: Plan | None` in compaction commit and always pass `authority=authority`. Make `legacy_allows` a linear sequence of explicit guards. Convert the private harness import to a guarded import with an actionable pinned-version error. Add a per-kind runtime rejection-message table so `_require_open` and `_claim_activity` share one admission decision.

- [ ] **Step 3: Verify affected behavior and import contract**

Run focused compaction, context projection, reserve operations, runtime close/cancel, and conformance tests. Verify a normal base-environment `import ds4py` still succeeds.

- [ ] **Step 4: Commit**

`refactor(agent): consolidate reserve and admission helpers`

### Task 7: Run the complete gates and independent review

**Files:**
- Review only unless verification exposes a regression.

- [ ] **Step 1: Run the static gate**

`uv run --frozen --extra agent prek run --all-files --show-diff-on-failure`

- [ ] **Step 2: Run the portable suite**

`uv run --frozen --extra agent pytest -m "not build and not live" -q`

- [ ] **Step 3: Run the independent whole-branch review**

Review base `4a0a289` against the remediation head. Include the third-review findings and the design/plan. Require explicit verdicts for the two blockers, admission/history ownership, reserve settlement, and regressions. Do not rerun native/live tests or claim their coverage locally.
