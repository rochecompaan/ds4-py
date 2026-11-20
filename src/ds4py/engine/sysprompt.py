"""Persistent native KV cache for the exact rendered chat system prefix."""

from __future__ import annotations

import hashlib
from contextlib import suppress
from typing import TYPE_CHECKING

from ds4py.errors import EngineError, PayloadLoadError, PayloadMismatch, StoreError
from ds4py.store import KV_HEADER, KV_MAGIC, KV_VERSION, SessionStore, token_digest

from .events import Interrupted
from .worker import SyncJob

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import BinaryIO

    from .session import Engine, Session


def system_prefix_path(store: SessionStore, engine: Engine, tokens: Sequence[int]) -> Path:
    """Return the one cache filename for this engine identity and exact tokens."""
    info = engine.info
    library = engine.native.library_identity
    identity = (
        f"{info.model_id}\n{info.routed_quant_bits}\n{engine.config.ctx}\n"
        f"{library.source_commit or ''}\n{library.library_sha256}\n{token_digest(tokens)}"
    )
    digest = hashlib.sha256(identity.encode("ascii")).hexdigest()
    return store.sysprompt_path(digest)


def _load(session: Session, store: SessionStore, path: Path, tokens: tuple[int, ...]) -> None:
    payload_bytes = store.read_kv_header(path)
    with path.open("rb") as file:
        file.seek(KV_HEADER.size)
        session._load_payload_events(
            file,
            payload_bytes=payload_bytes,
            transcript=None,
            expected_tokens=tokens,
        )


def _prefill(session: Session, tokens: tuple[int, ...]) -> None:
    if any(isinstance(event, Interrupted) for event in session._job_events(SyncJob(tokens))):
        raise EngineError("chat start was interrupted")


def _save(session: Session, store: SessionStore, path: Path, tokens: tuple[int, ...]) -> None:
    def write_payload(file: BinaryIO) -> int:
        saved = session._save_payload_events(
            file,
            tokens=tokens,
            header=KV_HEADER.pack(KV_MAGIC, KV_VERSION, 0),
            already_synced=True,
        )
        if saved.tokens != len(tokens):
            raise EngineError(f"system-prefix cache save token count {saved.tokens} does not match {len(tokens)}")
        return saved.payload_bytes

    store.write_kv_stream_atomic(path, write_payload)


def sync_system_prefix(session: Session, tokens: tuple[int, ...]) -> None:
    """Load an exact cached prefix or prefill it, retaining chat-start semantics."""
    if session._cancellation_pending():
        _prefill(session, tokens)
        return
    if not session.file_payloads_supported() or not session.engine.native.restore_enabled:
        _prefill(session, tokens)
        return

    store = session._system_prefix_store
    path = system_prefix_path(store, session.engine, tokens)
    try:
        _load(session, store, path, tokens)
    except PayloadLoadError, PayloadMismatch, StoreError, OSError:
        pass
    else:
        return

    _prefill(session, tokens)
    # The live KV and its in-memory checkpoint remain authoritative.
    with suppress(StoreError):
        _save(session, store, path, tokens)
