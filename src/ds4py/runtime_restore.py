"""Match saved records to snapshots and restore native or rendered state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from ds4py.binding_provenance import prepare_request_view_provenance, validate_canonical_snapshot
from ds4py.context_paging.history import derive_frontier
from ds4py.errors import EngineError, StoreError
from ds4py.store import KV_HEADER, SaveRecord, SessionStore, token_digest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pydantic_ai_harness.step_persistence import ContinuableSnapshot

    from ds4py.binding import MessageBinder
    from ds4py.capabilities.system_notes import DS4SystemNotes
    from ds4py.context_paging.selection import ContextCutFrontier
    from ds4py.engine.session import Session


class RestoreOutcome(TypedDict, total=False):
    rejected: str


class RestoreResult(RestoreOutcome):
    restore_path: str
    mismatch: str | None


@dataclass(frozen=True)
class RestorePlan:
    path: str
    payload_bytes: int | None
    mismatch: str | None


@dataclass(frozen=True)
class RestoreCasePlan:
    plan: RestorePlan
    use_request_view: bool = False
    ignore_view_segments: bool = False


def plan_restore_case(
    *,
    record: SaveRecord,
    plan: RestorePlan,
    rolling_enabled: bool,
    payloads_supported: bool,
) -> RestoreCasePlan:
    """Choose exact view binding, legacy binding, or full canonical fresh rendering."""
    if record.request_view is None:
        return RestoreCasePlan(plan)
    if rolling_enabled and payloads_supported and plan.path == "kv":
        return RestoreCasePlan(plan, use_request_view=True)
    mismatch = plan.mismatch or "the saved request view is not eligible for exact binding"
    return RestoreCasePlan(RestorePlan("render", None, mismatch), ignore_view_segments=True)


def prepare_restore_case(  # noqa: PLR0913 - Restore planning needs the validated record, snapshot, and native capability.
    *,
    binder: MessageBinder,
    snapshot: ContinuableSnapshot,
    record: SaveRecord,
    session_dir: Path,
    plan: RestorePlan,
    rolling_enabled: bool,
    payload_support: Callable[[], bool],
) -> tuple[RestoreCasePlan, ContextCutFrontier | None]:
    """Validate in legacy order, then choose and validate the request-view case."""
    frontier = None
    if record.request_view is None:
        validate_restore_snapshot(binder=binder, snapshot=snapshot, record=record, session_dir=session_dir)
        payloads_supported = plan.path == "kv" and payload_support()
        frontier = None
    else:
        payloads_supported = rolling_enabled and plan.path == "kv" and payload_support()
    if plan.path == "kv" and not payloads_supported and (record.request_view is None or rolling_enabled):
        plan = RestorePlan("render", None, "the live engine does not support file payloads")
    case = plan_restore_case(
        record=record,
        plan=plan,
        rolling_enabled=rolling_enabled,
        payloads_supported=payloads_supported,
    )
    if record.request_view is not None:
        frontier = validate_restore_snapshot(
            binder=binder,
            snapshot=snapshot,
            record=record,
            session_dir=session_dir,
            case=case,
        )
    return case, frontier


def plan_restore(
    record: SaveRecord,
    session_dir: Path,
    store: SessionStore,
) -> RestorePlan:
    """Select KV loading only when the declared payload is present and complete."""
    kv_path = session_dir / "session.kv"
    if record.payload_bytes == 0:
        return RestorePlan("render", None, "the save record declares no KV payload")
    if not kv_path.is_file():
        return RestorePlan("render", None, f"{kv_path}: the KV file is missing")
    declared = store.read_kv_header(kv_path)
    if declared != record.payload_bytes:
        return RestorePlan(
            "render",
            None,
            f"{kv_path}: the KV payload is {declared} bytes, the save record declares {record.payload_bytes}",
        )
    return RestorePlan("kv", declared, None)


def validate_restore_snapshot(
    *,
    binder: MessageBinder,
    snapshot: ContinuableSnapshot,
    record: SaveRecord,
    session_dir: Path,
    case: RestoreCasePlan | None = None,
) -> ContextCutFrontier | None:
    """Validate snapshot/provenance agreement without mutating live state."""
    if case is not None and case.ignore_view_segments:
        try:
            validate_canonical_snapshot(snapshot.messages)
        except EngineError as error:
            raise StoreError(f"{session_dir / 'save.json'}: cannot restore canonical history: {error}") from error
        return None
    if case is not None and case.use_request_view:
        if record.request_view is None:
            raise StoreError(f"{session_dir / 'save.json'}: the exact restore case requires a saved request view")
        try:
            prepared = prepare_request_view_provenance(
                canonical_messages=snapshot.messages,
                request_view=record.request_view,
                segments=record.segments,
                loaded=None,
            )
            return derive_frontier(snapshot.messages, prepared)
        except EngineError as error:
            raise StoreError(
                f"{session_dir / 'save.json'}: cannot restore session {record.session_id} "
                f"run {record.run_id} step {record.step_index}: {error}"
            ) from error
    try:
        binder.validate_restore(snapshot.messages, record.segments)
    except EngineError as error:
        raise StoreError(f"{session_dir / 'save.json'}: cannot restore: {error}") from error
    return None


def restore_snapshot(  # noqa: PLR0913 - Restoration needs the complete saved execution context.
    *,
    session: Session,
    binder: MessageBinder,
    notes: DS4SystemNotes | None,
    record: SaveRecord,
    snapshot: ContinuableSnapshot,
    plan: RestorePlan,
    session_dir: Path,
    use_request_view: bool = False,
    ignore_view_segments: bool = False,
) -> RestoreResult:
    """Install one snapshot validated by :func:`validate_restore_snapshot`.

    KV loading and provenance installation run through the session writer; the
    binder assigns its prepared records only after both succeed.
    """
    outcome: RestoreOutcome = {}
    restore_messages = (
        record.request_view if use_request_view and record.request_view is not None else snapshot.messages
    )
    note_segments = () if ignore_view_segments else record.segments
    loaded: list[int] | None = None
    if plan.payload_bytes is not None:
        with (session_dir / "session.kv").open("rb") as file:
            file.seek(KV_HEADER.size)
            loaded = session.load_payload(file, payload_bytes=plan.payload_bytes)
        if len(loaded) != record.tokens:
            outcome["rejected"] = (
                f"the loaded payload has {len(loaded)} tokens, the save record declares {record.tokens}"
            )
        elif token_digest(loaded) != record.tokens_sha256:
            outcome["rejected"] = "the loaded payload token hash does not match the save record"
        if "rejected" in outcome:
            loaded = None
    if loaded is not None:
        try:
            binder.restore(restore_messages, record.segments)
        except EngineError as error:
            raise StoreError(f"{session_dir / 'save.json'}: cannot restore: {error}") from error
        if notes is not None:
            notes.restore_from_segments(session.transcript.segments)
    else:
        if use_request_view:
            note_segments = ()
        if any(
            (segment.kind == "system_note" or segment.kind.startswith("system_note:")) and segment.text is None
            for segment in note_segments
        ):
            raise ValueError("saved note text is required for rendered restore")
        session.truncate(0)
        binder.forget()
        if notes is not None:
            notes.restore_from_segments(note_segments)
        binder.restore_notes(note_segments)
    return {
        **outcome,
        "restore_path": "kv" if loaded is not None else "render",
        "mismatch": (None if loaded is not None else outcome.get("rejected", plan.mismatch)),
    }
