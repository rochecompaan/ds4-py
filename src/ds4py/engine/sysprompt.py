"""Persistent native KV cache for the exact rendered chat system prefix."""

from __future__ import annotations

import hashlib
from contextlib import suppress
from typing import TYPE_CHECKING

from ds4py.errors import EngineError, StoreError
from ds4py.store import KV_HEADER, KV_MAGIC, KV_VERSION, SessionStore, token_digest

from .events import Interrupted, Loaded, Saved
from .worker import LoadPayloadJob, SavePayloadJob, SyncJob

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import BinaryIO

    from .session import Engine, Session


def system_prefix_path(engine: Engine, tokens: Sequence[int]) -> Path:
    """Return the one cache filename for this engine identity and exact tokens."""
    info = engine.info
    identity = f"{info.model_id}\n{info.routed_quant_bits}\n{engine.config.ctx}\n{token_digest(tokens)}"
    digest = hashlib.sha256(identity.encode("ascii")).hexdigest()
    return SessionStore().root / "sysprompt" / f"{digest}.kv"


def _load(session: Session, store: SessionStore, path: Path, tokens: tuple[int, ...]) -> None:
    payload_bytes = store.read_kv_header(path)
    with path.open("rb") as file:
        file.seek(KV_HEADER.size)
        loaded = False
        for event in session._job_events(
            LoadPayloadJob(file, payload_bytes, None, expected_tokens=tokens),
        ):
            if isinstance(event, Loaded):
                loaded = True
    if not loaded:
        raise EngineError("system-prefix cache load did not complete")


def _prefill(session: Session, tokens: tuple[int, ...]) -> None:
    if any(isinstance(event, Interrupted) for event in session._job_events(SyncJob(tokens))):
        raise EngineError("chat start was interrupted")


def _save(session: Session, store: SessionStore, path: Path, tokens: tuple[int, ...]) -> None:
    def write_payload(file: BinaryIO) -> int:
        saved = None
        for event in session._job_events(
            SavePayloadJob(tokens, KV_HEADER.pack(KV_MAGIC, KV_VERSION, 0), file, already_synced=True),
        ):
            if isinstance(event, Saved):
                saved = event
        if saved is None:
            raise EngineError("system-prefix cache save did not complete")
        return saved.payload_bytes

    store.write_kv_stream_atomic(path, write_payload)


def sync_system_prefix(session: Session, tokens: tuple[int, ...]) -> None:
    """Load an exact cached prefix or prefill it, retaining chat-start semantics."""
    if not session.file_payloads_supported():
        _prefill(session, tokens)
        return

    path = system_prefix_path(session.engine, tokens)
    store = SessionStore()
    try:
        _load(session, store, path, tokens)
    except EngineError, StoreError, OSError:
        pass
    else:
        session.history.clear_checkpoints()
        session.history.clear_pause()
        return

    _prefill(session, tokens)
    # The live KV and its in-memory checkpoint remain authoritative.
    with suppress(StoreError):
        _save(session, store, path, tokens)
