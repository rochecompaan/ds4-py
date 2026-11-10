"""Explicit reserve operations and the localized legacy ambient adapter.

Legacy naked-to_thread owners cannot be distinguished from child executors:
that compatibility mode is a lifecycle guard, not security isolation.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import get_ident
from typing import TYPE_CHECKING, Literal

from ds4py.async_session import run_session_operation
from ds4py.errors import EngineError

from .lifecycle import in_listener

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from .compaction import SummaryResult
    from .plan import Plan
    from .session import Session
    from .transcript import TranscriptReplacement


type ReserveOperation = Literal["summary", "replacement"]


def current_task() -> asyncio.Task[object] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


@dataclass(eq=False)
class ReserveAuthority:
    """Opaque, single-use ticket; only its issuing live window can consume it."""

    _owner: ReserveOperations
    _operation: ReserveOperation


_legacy_authority: ContextVar[ReserveOperations | None] = ContextVar("legacy_reserve_authority", default=None)


class ReserveOperations:
    """Selected-owner operations, validated before any executor submission."""

    def __init__(self, session: Session, *, strict: bool) -> None:
        self._session = session
        self._strict = strict
        self._task = current_task()
        self._thread = get_ident()
        self._active = True
        self._tickets: set[ReserveAuthority] = set()

    def revoke(self) -> None:
        self._active = False
        self._tickets.clear()

    def _validate_owner(self) -> None:
        if not self._active or in_listener.get() or current_task() is not self._task or get_ident() != self._thread:
            raise EngineError("reserve operations require the selected live owner")

    def _authority(self, operation: ReserveOperation) -> ReserveAuthority:
        self._validate_owner()
        ticket = ReserveAuthority(self, operation)
        self._tickets.add(ticket)
        return ticket

    def consume(self, authority: ReserveAuthority, operation: ReserveOperation) -> bool:
        if not self._active or authority not in self._tickets or authority._operation != operation:
            return False
        self._tickets.remove(authority)
        return True

    def legacy_allows(self) -> bool:
        task = current_task()
        return (
            not self._strict
            and self._active
            and _legacy_authority.get() is self
            and not in_listener.get()
            and (task is self._task if task is not None else self._task is None or get_ident() != self._thread)
        )

    def prepare_summary(self, bottom: int) -> SummaryResult:
        return self._session.prepare_summary(bottom, authority=self._authority("summary"))

    def _replacement_operation(self, replacement: TranscriptReplacement, *, reason: str) -> Callable[[], Plan]:
        authority = self._authority("replacement")
        return lambda: self._session.replace_transcript(replacement, reason=reason, authority=authority)

    def replace_transcript(self, replacement: TranscriptReplacement, *, reason: str) -> Plan:
        return self._replacement_operation(replacement, reason=reason)()

    async def prepare_summary_async(self, bottom: int) -> SummaryResult:
        authority = self._authority("summary")
        return await run_session_operation(
            self._session, lambda: self._session.prepare_summary(bottom, authority=authority)
        )

    async def replace_transcript_async(self, replacement: TranscriptReplacement, *, reason: str) -> Plan:
        operation = self._replacement_operation(replacement, reason=reason)
        return await run_session_operation(self._session, operation)


@contextmanager
def legacy_context(operations: ReserveOperations, *, enabled: bool) -> Generator[None]:
    """Install ambient permission only for unchanged legacy owners."""
    token = _legacy_authority.set(operations if enabled else None)
    try:
        yield
    finally:
        _legacy_authority.reset(token)
