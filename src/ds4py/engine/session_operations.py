"""Session-owned exclusive operations and atomic binding transactions.

Admission state and admission policy live together here so every caller shares
one model. ``OperationState`` owns the closed flag, the open round, the live
reserve pause and its window, plus the one exclusive claim. The policies stay
distinct: open-only observation, idle mutation, paused native summary/replacement
work and close each reject a different set of states, and counting or planning
carries no claim at all.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ds4py.errors import EngineError

from . import chat
from .lifecycle import in_listener

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from .events import ReserveReached
    from .reserve import ReserveAuthority, ReserveOperation, ReserveOperations
    from .session import Session
    from .tool_prompt import ToolDefinition
    from .transcript import TranscriptReplacement

type OperationKind = Literal["binding", "summary", "replacement"]
type GuardPolicy = Literal["open", "idle", "native", "close"]

_MESSAGES = {
    "closed": "session is closed",
    "busy": "a round is open",
    "native": "a native operation is active; drain before closing",
    "expired": "reserve pause has expired",
    "window": "reserve window requires an unclaimed live pause",
}

# Only paused native work is guarded by a reserve operation; binding is idle.
_POLICIES: dict[OperationKind, tuple[GuardPolicy, ReserveOperation | None]] = {
    "binding": ("idle", None),
    "summary": ("native", "summary"),
    "replacement": ("native", "replacement"),
}


@dataclass(frozen=True)
class Claim:
    """Identity of one admitted exclusive operation."""

    kind: OperationKind


class OperationState:
    """Every admission input of one Session, changed only by its transitions.

    At most one exclusive operation is claimed at a time, at most one round is
    open, and at most one reserve pause admits one live window. Admission reads
    this value directly, so it never reaches back into the facade.
    """

    def __init__(self) -> None:
        self._claim: Claim | None = None
        self._closed = False
        self._active_round = False
        self._reserve_event: ReserveReached | None = None
        self._reserve_operations: ReserveOperations | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def active_round(self) -> bool:
        return self._active_round

    @property
    def reserve_event(self) -> ReserveReached | None:
        """The live pause awaiting a decision, or None once it expired."""
        return self._reserve_event

    @property
    def reserve_operations(self) -> ReserveOperations | None:
        """The installed window, including a revoked one retained until its exit."""
        return self._reserve_operations

    @property
    def kind(self) -> OperationKind | None:
        return None if self._claim is None else self._claim.kind

    def close(self) -> None:
        """Terminally close admission and expire any reserve pause or window."""
        self._closed = True
        self.end_reserve()

    def begin_round(self) -> None:
        self._active_round = True

    def end_round(self) -> None:
        self._active_round = False

    def begin_reserve(self, event: ReserveReached) -> None:
        """Record the one pause a decision may address."""
        self._reserve_event = event

    def require_reserve(self, event: ReserveReached) -> None:
        """Reject a decision delivered for an expired pause."""
        if self._reserve_event is not event:
            raise EngineError(_MESSAGES["expired"])

    def end_reserve(self) -> None:
        """Expire the live pause and revoke whatever window it admitted."""
        self._reserve_event = None
        if self._reserve_operations is not None:
            self._reserve_operations.revoke()

    def install_reserve(self, operations: ReserveOperations) -> None:
        """Retain the one live window; only its own exits release it."""
        self._reserve_operations = operations

    def release_reserve(self, operations: ReserveOperations) -> None:
        """Revoke one window and clear it only while it is still installed."""
        operations.revoke()
        if self._reserve_operations is operations:
            self._reserve_operations = None

    def consume_reserve_authority(self, authority: ReserveAuthority | None, operation: ReserveOperation) -> bool:
        """Return True when a live window authorizes ``operation`` for this caller."""
        operations = self._reserve_operations
        if self._reserve_event is None or operations is None:
            return False
        if authority is not None:
            return operations.consume(authority, operation)
        return operations.legacy_allows()

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


def blocked_reason(
    state: OperationState,
    policy: GuardPolicy,
    *,
    authority: ReserveAuthority | None = None,
    binding: BindingTransaction | None = None,
) -> str | None:
    """Return the caller-specific rejection for ``policy``, or admit it."""
    if state.closed:
        return _MESSAGES["closed"]
    if policy == "open":
        return None
    if policy == "close":
        return None if state.kind is None else _MESSAGES["native"]
    if policy == "idle":
        admitted = (binding is not None and binding._live(state)) or not (state.active_round or state.kind is not None)
        return None if admitted else _MESSAGES["busy"]
    window = state.reserve_event is not None and state.reserve_operations is not None
    busy = state.kind is not None or ((state.active_round or authority is not None) and not window)
    # Only a live reserve window can admit paused native work; the window itself
    # verifies the supplied ticket for the caller's operation.
    return _MESSAGES["busy"] if busy else None


def open_reserve_window(
    state: OperationState,
    event: ReserveReached,
    operations: ReserveOperations,
) -> None:
    """Install the one window for an unclaimed live pause; listeners get none."""
    unclaimed = event is state.reserve_event and state.reserve_operations is None and state.kind is None
    if in_listener.get() or not unclaimed:
        raise EngineError(_MESSAGES["window"])
    state.install_reserve(operations)


@contextmanager
def exclusive(
    session: Session,
    kind: OperationKind,
    *,
    authority: ReserveAuthority | None = None,
) -> Generator[Claim]:
    """Admit, claim and release one exclusive operation; the lock excludes writers."""
    policy, reserve_operation = _POLICIES[kind]
    state = session._operations
    with session._lock:
        reason = blocked_reason(state, policy, authority=authority)
        if reason is not None:
            raise EngineError(reason)
        paused = state.active_round or authority is not None
        if (
            reserve_operation is not None
            and paused
            and not state.consume_reserve_authority(authority, reserve_operation)
        ):
            raise EngineError(_MESSAGES["busy"])
        claim = state.claim(kind)
    try:
        yield claim
    finally:
        with session._lock:
            state.release(claim)


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

    def _live(self, state: OperationState) -> bool:
        return self._active and state.owns(self._claim)

    def _check_current(self) -> None:
        revision, length = self._observed()
        if not self._live(self._session._operations):
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
