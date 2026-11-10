"""Session-owned exclusive operations and atomic binding transactions.

Claim/release mechanics live here so every caller shares one admission model.
The policies stay distinct: open-only observation, idle mutation, native
summary/replacement work and close each reject a different set of states, and
counting or planning carries no claim at all.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from ds4py.errors import EngineError

from . import chat

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from .reserve import ReserveAuthority, ReserveOperation
    from .session import Session
    from .tool_prompt import ToolDefinition
    from .transcript import TranscriptReplacement

type OperationKind = Literal["binding", "summary", "replacement"]
type GuardPolicy = Literal["open", "idle", "native", "close"]

_MESSAGES = {
    "closed": "session is closed",
    "busy": "a round is open",
    "native": "a native operation is active; drain before closing",
}


@dataclass(frozen=True)
class Claim:
    """Identity of one admitted exclusive operation."""

    kind: OperationKind


class OperationState:
    """At most one exclusive Session operation at a time."""

    def __init__(self) -> None:
        self._claim: Claim | None = None

    @property
    def kind(self) -> OperationKind | None:
        return None if self._claim is None else self._claim.kind

    def claim(self, kind: OperationKind) -> Claim:
        if self._claim is not None:
            raise EngineError(_MESSAGES["busy"])
        self._claim = Claim(kind)
        return self._claim

    def owns(self, claim: Claim) -> bool:
        """Return True only while ``claim`` is the live admitted operation."""
        return self._claim is claim

    def release(self, claim: Claim) -> None:
        if self._claim is claim:
            self._claim = None


def policy_for(kind: OperationKind) -> GuardPolicy:
    return "idle" if kind == "binding" else "native"


def blocked_reason(
    session: Session,
    policy: GuardPolicy,
    *,
    authority: ReserveAuthority | None = None,
    binding: BindingTransaction | None = None,
) -> str | None:
    """Return the caller-specific rejection for ``policy``, or admit it."""
    if session._closed:
        return _MESSAGES["closed"]
    if policy == "open":
        return None
    if policy == "close":
        return None if session._operations.kind is None else _MESSAGES["native"]
    if policy == "idle":
        admitted = (binding is not None and binding._live(session)) or not (
            session._active_round or session._operations.kind is not None
        )
        return None if admitted else _MESSAGES["busy"]
    window = session._reserve_event is not None and session._reserve_operations is not None
    busy = session._operations.kind is not None or ((session._active_round or authority is not None) and not window)
    # Only a live reserve window can admit paused native work; the window itself
    # verifies the supplied ticket for the caller's operation.
    return _MESSAGES["busy"] if busy else None


@contextmanager
def exclusive(
    session: Session,
    kind: OperationKind,
    *,
    authority: ReserveAuthority | None = None,
) -> Generator[Claim]:
    """Admit, claim and release one exclusive operation; the lock excludes writers."""
    policy = policy_for(kind)
    with session._lock:
        reason = blocked_reason(session, policy, authority=authority)
        if reason is not None:
            raise EngineError(reason)
        native_paused = policy == "native" and (session._active_round or authority is not None)
        if native_paused and not session._reserve_authorized(authority, cast("ReserveOperation", kind)):
            raise EngineError(_MESSAGES["busy"])
        claim = session._operations.claim(kind)
    try:
        yield claim
    finally:
        with session._lock:
            session._operations.release(claim)


class BindingTransaction:
    """One claimed binding boundary sequence over Session-owned history.

    The handle admits only the planned truncate/start/append sequence. Native
    preamble work and native checkpoint disposal release the mutation lock but
    never the claim, and every operation re-checks the prefix revision its plan
    observed.
    """

    def __init__(self, session: Session, claim: Claim) -> None:
        self._session = session
        self._claim = claim
        self._active = True
        self._revision, self._length = session._transcript.snapshot()

    def close(self) -> None:
        self._active = False

    def _observed(self) -> tuple[int, int]:
        return self._session._transcript.snapshot()

    def _live(self, session: Session) -> bool:
        return self._active and session._operations.owns(self._claim)

    def _check_current(self) -> None:
        revision, length = self._observed()
        if not self._live(self._session):
            raise EngineError("the binding transaction is no longer active")
        if (revision, length) != (self._revision, self._length):
            raise EngineError("the transcript changed outside its binding transaction")

    def _committed(self) -> None:
        self._revision, self._length = self._observed()

    def truncate(self, token_count: int) -> None:
        """Truncate history and invalidate later checkpoints inside the claim.

        The Python truncation commits under the mutation lock, then each stale
        native checkpoint is disposed with only the claim held: a dispatched
        drop can wait on cross-thread event delivery, and the claim already
        excludes every writer and every idle checkpoint reader from the gap.
        """
        session = self._session
        with session._lock:
            self._check_current()
            position = session._truncate_position(token_count)
            stale = session._stale_checkpoints(position)
            session._truncate_python(position)
            self._committed()
        for key in stale:
            session._drop_checkpoint(key)
        with session._lock:
            self._check_current()

    def start_chat(
        self,
        tools: Iterable[ToolDefinition] = (),
        instructions: str | None = None,
    ) -> None:
        """Render, sync and hold the system preamble; roll Python state back on failure."""
        session = self._session
        with session._lock:
            self._check_current()
            start = len(session._transcript)
            try:
                chat.stage_chat(session, tools, instructions)
            except BaseException:
                session._transcript.rollback(start)
                self._committed()
                raise
        try:
            chat.sync_chat(session)
        except BaseException:
            with session._lock:
                session._transcript.rollback(start)
                self._committed()
            raise
        with session._lock:
            session._transcript.set_chat()
            self._committed()

    def apply_appends(
        self,
        replacement: TranscriptReplacement,
        *,
        retire_paused: bool = False,
        publish: Callable[[], None] | None = None,
    ) -> None:
        """Install one staged group and publish its state under the same lock."""
        with self._session._lock:
            self._check_current()
            try:
                self._session._commit_appends(
                    replacement,
                    retire_paused=retire_paused,
                    publish=publish,
                    binding=self,
                )
            except BaseException:
                # A rejected publication restored the whole group; the claim is
                # still coherent, so keep the observed prefix as its expectation.
                self._committed()
                raise
            self._committed()
