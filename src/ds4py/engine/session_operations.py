"""Session-owned admission state and exclusive operation policy.

Admission state and admission policy live together here so every caller shares
one model. ``OperationState`` owns the closed flag, the open round, the live
reserve pause and its window, plus the one exclusive claim. The policies stay
distinct: open-only observation, idle mutation, paused native summary/replacement
work and close each reject a different set of states, and counting or planning
carries no claim at all. History mutations own the lock and the claimed states;
this module never reaches back into the facade.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from ds4py.errors import EngineError

from .lifecycle import in_listener

if TYPE_CHECKING:
    from collections.abc import Generator
    from threading import RLock

    from .events import ReserveReached
    from .reserve import ReserveAuthority, ReserveOperation, ReserveOperations

type OperationKind = Literal["binding", "checkpoint", "summary", "replacement"]
type GuardPolicy = Literal["open", "idle", "native", "close"]

_MESSAGES = {
    "closed": "session is closed",
    "busy": "a round is open",
    "native": "a native operation is active; drain before closing",
    "expired": "reserve pause has expired",
    "window": "reserve window requires an unclaimed live pause",
}

# Only paused native work is guarded by a reserve operation; idle work is not.
_POLICIES: dict[OperationKind, tuple[GuardPolicy, ReserveOperation | None]] = {
    "binding": ("idle", None),
    "checkpoint": ("idle", None),
    "summary": ("native", "summary"),
    "replacement": ("native", "replacement"),
}


class BindingLiveness(Protocol):
    """The one live binding transaction an idle mutation may admit."""

    def live(self, state: OperationState) -> bool: ...


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

    @property
    def live_window(self) -> bool:
        """Whether a live pause and its installed window are both present."""
        return self._reserve_event is not None and self._reserve_operations is not None

    def paused(self, authority: ReserveAuthority | None = None) -> bool:
        """Whether a caller is a paused round or a ticket-bearing continuation."""
        return self._active_round or authority is not None

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
    binding: BindingLiveness | None = None,
) -> str | None:
    """Return the caller-specific rejection for ``policy``, or admit it."""
    if state.closed:
        return _MESSAGES["closed"]
    if policy == "open":
        return None
    if policy == "close":
        return None if state.kind is None else _MESSAGES["native"]
    if policy == "idle":
        admitted = (binding is not None and binding.live(state)) or not (state.active_round or state.kind is not None)
        return None if admitted else _MESSAGES["busy"]
    busy = state.kind is not None or (state.paused(authority) and not state.live_window)
    # Only a live reserve window can admit paused native work; the window itself
    # verifies the supplied ticket for the caller's operation.
    return _MESSAGES["busy"] if busy else None


def require_reserve_window(state: OperationState, event: ReserveReached) -> None:
    """Reject a window request that is not the unclaimed live pause."""
    if (
        in_listener.get()
        or event is not state.reserve_event
        or state.reserve_operations is not None
        or state.kind is not None
    ):
        raise EngineError(_MESSAGES["window"])


def open_reserve_window(
    state: OperationState,
    event: ReserveReached,
    operations: ReserveOperations,
) -> None:
    """Install the one window for an unclaimed live pause; listeners get none."""
    require_reserve_window(state, event)
    state.install_reserve(operations)


@contextmanager
def exclusive(
    state: OperationState,
    lock: RLock,
    kind: OperationKind,
    *,
    authority: ReserveAuthority | None = None,
) -> Generator[Claim]:
    """Admit, claim and release one exclusive operation; the lock excludes writers."""
    policy, reserve_operation = _POLICIES[kind]
    with lock:
        reason = blocked_reason(state, policy, authority=authority)
        if reason is not None:
            raise EngineError(reason)
        paused = state.paused(authority)
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
        with lock:
            state.release(claim)
