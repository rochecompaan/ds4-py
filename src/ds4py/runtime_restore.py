"""Match saved records to snapshots and restore native or rendered state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from ds4py.errors import EngineError, StoreError
from ds4py.store import KV_HEADER, SaveRecord, SessionStore, token_digest

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic_ai_harness.step_persistence import ContinuableSnapshot

    from ds4py.binding import MessageBinder
    from ds4py.capabilities.system_notes import DS4SystemNotes
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
) -> None:
    """Validate snapshot/provenance agreement without mutating live state."""
    try:
        binder.validate_restore(snapshot.messages, record.segments)
    except EngineError as error:
        raise StoreError(f"{session_dir / 'save.json'}: cannot restore: {error}") from error


def restore_snapshot(  # noqa: PLR0913 - Restoration needs the complete saved execution context.
    *,
    session: Session,
    binder: MessageBinder,
    notes: DS4SystemNotes | None,
    record: SaveRecord,
    snapshot: ContinuableSnapshot,
    plan: RestorePlan,
    session_dir: Path,
) -> RestoreResult:
    """Install one snapshot validated by :func:`validate_restore_snapshot`."""
    outcome: RestoreOutcome = {}
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
            binder.restore(snapshot.messages, record.segments)
        except EngineError as error:
            raise StoreError(f"{session_dir / 'save.json'}: cannot restore: {error}") from error
        if notes is not None:
            notes.restore_from_segments(session.transcript.segments)
    else:
        session.truncate(0)
        binder.forget()
        if notes is not None:
            notes.restore_from_segments(record.segments)
            notes.sync_from_segments(())
    return {
        **outcome,
        "restore_path": "kv" if loaded is not None else "render",
        "mismatch": (None if loaded is not None else outcome.get("rejected", plan.mismatch)),
    }
