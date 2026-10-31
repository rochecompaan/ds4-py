"""Session identity, the session event bus, and session-scoped state.

Without an async bridge, listeners run on the facade's consumer thread and must
not call blocking session methods (rounds, syncs, or native jobs). During an
async model round, the bridge marshals every event to its owning event loop.
State factories and ``close()`` are orchestration concerns, not worker callbacks:
they may acquire ordinary Python resources, but the facade decides when they run.
"""
from asyncio import AbstractEventLoop
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
import sys
from threading import RLock, get_ident
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

Listener = Callable[["SessionEvent"], None]


@dataclass(frozen=True)
class SessionEvent:
    """One bus event; ``payload`` is the ``emit`` keyword arguments."""

    kind: str
    session_id: str
    payload: Mapping[str, Any]


class SessionLifecycle:
    """Identity, subscriptions, and state objects owned by one live session."""

    def __init__(self, *, error_handler: Callable[[Exception], None] | None = None) -> None:
        self._id = uuid4().hex
        self._listeners: list[tuple[Listener, frozenset[str] | None]] = []
        self._states: dict[str, object] = {}
        self._lock = RLock()
        self._error_handler = error_handler
        self._loop: AbstractEventLoop | None = None
        self._loop_thread: int | None = None

    @property
    def id(self) -> str:
        return self._id

    def set_id(self, new_id: str | None = None) -> str:
        self._id = uuid4().hex if new_id is None else new_id
        return self._id

    def on_event(self, listener: Listener, *, kinds: Iterable[str] | None = None
                 ) -> Callable[[], None]:
        allowed = None if kinds is None else frozenset((kinds,) if isinstance(kinds, str) else kinds)
        entry = (listener, allowed)
        with self._lock:
            self._listeners.append(entry)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._listeners.remove(entry)
                except ValueError:
                    pass

        return unsubscribe

    @contextmanager
    def deliver_on(self, loop: AbstractEventLoop) -> Iterator[None]:
        """Deliver emissions from other threads on ``loop``'s owning thread."""
        owner = get_ident()
        with self._lock:
            if self._loop is not None:
                raise RuntimeError("session event delivery already has an owning loop")
            self._loop = loop
            self._loop_thread = owner
        try:
            yield
        finally:
            with self._lock:
                if self._loop is loop and self._loop_thread == owner:
                    self._loop = None
                    self._loop_thread = None

    def emit(self, kind: str, **payload: Any) -> None:
        event = SessionEvent(kind, self._id, dict(payload))
        with self._lock:
            listeners = tuple(self._listeners)
            loop = self._loop
            owner = self._loop_thread
        if loop is not None and owner != get_ident():
            delivered: Future[None] = Future()

            def deliver() -> None:
                try:
                    self._emit_listeners(event, listeners)
                except BaseException as error:
                    delivered.set_exception(error)
                else:
                    delivered.set_result(None)

            loop.call_soon_threadsafe(deliver)
            # Keep event order and finish listeners before the producer proceeds.
            delivered.result()
            return
        self._emit_listeners(event, listeners)

    def _emit_listeners(
        self, event: SessionEvent,
        listeners: Iterable[tuple[Listener, frozenset[str] | None]],
    ) -> None:
        for listener, allowed in listeners:
            if allowed is not None and event.kind not in allowed:
                continue
            try:
                listener(event)
            except Exception as error:  # A listener must not hide the engine event.
                self._report(error)

    def _report(self, error: Exception) -> None:
        if self._error_handler is not None:
            try:
                self._error_handler(error)
                return
            except Exception as nested:  # pragma: no cover - defensive routing
                error = nested
        print(f"session event listener failed: {error}", file=sys.stderr)

    def state(self, key: str, factory: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._states:
                self._states[key] = factory()
            return self._states[key]

    def close_states(self) -> None:
        with self._lock:
            states = list(self._states.values())
            self._states.clear()
        error = None
        for state in states:
            close = getattr(state, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as current:
                if error is None:
                    error = current
        if error is not None:
            raise error
