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
    from collections.abc import Generator

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
        """Revoke this window; Session holds its mutation lock."""
        self._active = False
        self._tickets.clear()

    def _validate_owner(self) -> None:
        if not self._active or in_listener.get() or current_task() is not self._task or get_ident() != self._thread:
            raise EngineError("reserve operations require the selected live owner")

    def issue_authority(self, operation: ReserveOperation) -> ReserveAuthority:
        """Issue one operation ticket on the selected owner's loop/thread.

        Custom retained executors must call this factory before submission and
        pass the ticket explicitly to the matching Session ``authority=``
        keyword. It is session/window/operation-bound, consumed once at claim,
        and revoked on window exit or decision. Prefer the handle's sync/async
        methods unless publication requires a separate executor callable.
        """
        self._validate_owner()
        ticket = ReserveAuthority(self, operation)
        self._tickets.add(ticket)
        return ticket

    def consume(self, authority: ReserveAuthority, operation: ReserveOperation) -> bool:
        """Claim a matching ticket; Session holds its mutation lock."""
        if (
            not self._active
            or in_listener.get()
            or authority._owner is not self
            or authority not in self._tickets
            or authority._operation != operation
        ):
            return False
        self._tickets.discard(authority)
        return True

    def legacy_allows(self) -> bool:
        """Return True only for the unchanged legacy owner in its own task/thread."""
        if self._strict:
            return False
        if not self._active:
            return False
        if _legacy_authority.get() is not self:
            return False
        if in_listener.get():
            return False
        task = current_task()
        if task is not None:
            return task is self._task
        return self._task is None or get_ident() != self._thread

    def prepare_summary(self, bottom: int) -> SummaryResult:
        return self._session.prepare_summary(bottom, authority=self.issue_authority("summary"))

    def replace_transcript(self, replacement: TranscriptReplacement, *, reason: str) -> Plan:
        return self._session.replace_transcript(
            replacement, reason=reason, authority=self.issue_authority("replacement")
        )

    async def prepare_summary_async(self, bottom: int) -> SummaryResult:
        authority = self.issue_authority("summary")
        return await run_session_operation(
            self._session, lambda: self._session.prepare_summary(bottom, authority=authority)
        )

    async def replace_transcript_async(self, replacement: TranscriptReplacement, *, reason: str) -> Plan:
        authority = self.issue_authority("replacement")
        return await run_session_operation(
            self._session, lambda: self._session.replace_transcript(replacement, reason=reason, authority=authority)
        )


@contextmanager
def legacy_context(operations: ReserveOperations, *, enabled: bool) -> Generator[None]:
    """Install ambient permission only for unchanged legacy owners."""
    token = _legacy_authority.set(operations if enabled else None)
    try:
        yield
    finally:
        _legacy_authority.reset(token)
