"""Retain and drain blocking facade work before its async owner can unwind."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import dataclass
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import anyio

from ds4py.errors import EngineError

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from ds4py.engine.session import Session


@dataclass(frozen=True)
class _OperationOwner:
    task: asyncio.Task[object] | None
    done: asyncio.Future[None]


# Loop-confined ownership, not a second session/history or a native worker lock.
_owners: WeakKeyDictionary[Session, _OperationOwner] = WeakKeyDictionary()


@contextmanager
def session_operation_owner(session: Session) -> Generator[None]:
    """Own an attempt through drain and synchronous loop-side publication."""
    task = asyncio.current_task()
    existing = _owners.get(session)
    if existing is not None:
        if existing.task is not task:
            raise EngineError("a session operation is already active")
        yield
        return
    owner = _OperationOwner(task, asyncio.get_running_loop().create_future())
    _owners[session] = owner
    try:
        yield
    finally:
        del _owners[session]
        owner.done.set_result(None)


def cancel_session_operation(session: Session) -> asyncio.Future[None] | None:
    """Signal only retained work, not an unrelated custom owner's async wait."""
    owner = _owners.get(session)
    if owner is None:
        return None
    session.cancel()
    if owner.task is not None and owner.task is not asyncio.current_task():
        owner.task.cancel()
    return owner.done


async def drain_operation[T](future: asyncio.Future[T]) -> T:
    """Finish retained work despite repeated edge or level cancellation."""
    with anyio.CancelScope(shield=True):
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
    return future.result()


async def run_session_operation[T](session: Session, operation: Callable[[], T]) -> T:
    """Offload one Session call without abandoning it.

    Strict reserve callers validate on-loop and capture an explicit ticket in
    ``operation``. Context copying below serves legacy owners and other ambient
    application state; it is not the source of strict reserve authority.

    This boundary owns no response pull lock. A cancelled caller does not regain
    control until the facade (including native restoration and event delivery)
    has finished. Ordered loop-side publication belongs to the caller.
    """
    with session_operation_owner(session):
        return await _run_operation(session, operation)


@dataclass(frozen=True)
class _Completed[T]:
    value: T


@dataclass(frozen=True)
class _Failed:
    error: BaseException


def _capture[T](operation: Callable[[], T]) -> _Completed[T] | _Failed:
    # Python 3.14 shield logs inner exceptions even when its cancelled waiter
    # subsequently drains them. Transfer failures as values until the owner
    # observes the result, rather than creating unhandled-future diagnostics.
    try:
        return _Completed(operation())
    except BaseException as error:  # noqa: BLE001 - Transfer worker failures to the owning loop.
        return _Failed(error)


async def _run_operation[T](session: Session, operation: Callable[[], T]) -> T:
    context = copy_context()

    def execute() -> _Completed[T] | _Failed:
        return _capture(operation)

    future: asyncio.Future[_Completed[T] | _Failed] = asyncio.get_running_loop().run_in_executor(
        None, context.run, execute
    )
    try:
        result = await asyncio.shield(future)
    except asyncio.CancelledError as cancelled:
        session.cancel()
        try:
            result = await drain_operation(future)
            if isinstance(result, _Failed):
                raise cancelled from result.error
            raise
        finally:
            session.clear_cancellation()
    if isinstance(result, _Failed):
        session.clear_cancellation()
        raise result.error
    return result.value
