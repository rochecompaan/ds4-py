# M4 ownership remediation design

## Intent

Rework the third M4 review's structural findings without changing request, persistence, compaction, cancellation, or native-restore behavior. The goal is real ownership boundaries, not smaller files that still reach across each other's private state.

## Decisions

- Use `ds4py.store.SegmentProvenance` as the only persisted provenance value. Binder export returns it directly, restore accepts it directly, and runtime stores the exported values without conversion.
- Keep pure transition planning in `binding_plan.py`, but replace its `MessageBinder` back-reference with an explicit `BindingState` snapshot. The state contains the binder's persisted fields plus the public session/rendering collaborators needed to plan. `binding_plan.py` must not import `MessageBinder`, even under `TYPE_CHECKING`.
- Split prefix selection, boundary construction, and candidate rendering out of the transition traversal. Remove the transition's complexity suppressions and `nonlocal` publication counter.
- Expose exact projection through `MessageBinder.project_request(..., bound=False)`. Preparation remains plan-and-apply; `prepare_model_request` is only the harness adapter. Remove the parallel plan/count method family.
- Move closed, round, claim, and live-reserve state into `OperationState`. Admission evaluates that state directly and maps each operation kind to its guard and reserve operation without a cast.
- Add `SessionHistory` as the owner of transcript, pause/checkpoint state, the mutation lock, operation state, and the claimed mutation pipeline. `Session` remains the public facade and native-job boundary. No second module may define admission over `Session` private fields.
- Route public `Session.start_chat` through the same claimed staging/sync/publication path as binding. Native sync and checkpoint delivery run without the mutation lock.
- Add `ReserveSettlement` to own owner selection, reserve authorization, owner recovery, post-recovery fit, summary/sync accounting, and resume/fail decisions. `DS4StreamedResponse` retains event translation, round draining, reconciliation, and response finalization.
- Apply the review's small cleanups where they touch these components. Keep the existing worker-confined poison callback: facade cleanup already routes through `Session.poison()`, and calling that facade from the worker would deadlock.

## Alternatives considered

Moving the entire transition back into `MessageBinder` would remove the circular dependency, but it would grow an already large class and lose the opportunity to test planning from an explicit state value. Moving only methods into `SessionHistory` while leaving its state in `Session` would repeat the fake-seam problem, so the state moves with the pipeline.

## Constraints

- No public behavior changes except removal of the redundant binder planning/count API identified by the review.
- Save format remains version 1 and existing snapshots remain readable.
- Exact token projections and the one-token response reserve remain unchanged.
- `Session` remains the only public writer for token and segment replacement.
- Existing cancellation, partial publication, provenance, and native fail-closed tests must remain intact.
- No new native/live behavior is claimed without native/live verification.
