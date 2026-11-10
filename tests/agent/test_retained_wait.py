"""Shared retained-wait outcomes: edge/level cancellation and completion order."""

import asyncio
from typing import cast

import anyio
import pytest

from ds4py.async_session import await_retained
from ds4py.errors import EngineError


def _future() -> asyncio.Future[int]:
    return cast("asyncio.Future[int]", asyncio.get_running_loop().create_future())


def test_retained_wait_returns_already_completed_result() -> None:
    async def scenario() -> None:
        future = _future()
        future.set_result(7)
        assert await await_retained(future) == 7

    asyncio.run(scenario())


def test_retained_wait_preserves_already_completed_error() -> None:
    async def scenario() -> None:
        future = _future()
        future.set_exception(EngineError("boom"))
        with pytest.raises(EngineError, match="boom"):
            await await_retained(future)

    asyncio.run(scenario())


def test_retained_wait_preserves_already_completed_cancellation() -> None:
    async def scenario() -> None:
        future = _future()
        future.cancel()
        with pytest.raises(asyncio.CancelledError):
            await await_retained(future)

    asyncio.run(scenario())


def test_retained_wait_drains_repeated_edge_cancellation_and_signals_once() -> None:
    async def scenario() -> None:
        future = _future()
        signals: list[object] = []

        async def run() -> int:
            return await await_retained(future, on_cancel=lambda: signals.append("cancel"))

        task = asyncio.create_task(run())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "the retained future was abandoned"
        assert signals == ["cancel"], "the cancellation signal did not fire exactly once"
        future.set_result(7)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert future.result() == 7

    asyncio.run(scenario())


def test_retained_wait_keeps_anyio_deadline_result_without_level_spin() -> None:
    async def scenario() -> None:
        future = _future()
        scope = anyio.CancelScope()

        async def run() -> None:
            with scope:
                await await_retained(future)
            # The level cancellation was delivered, drained, and re-raised; the
            # enclosing scope catches it without abandoning retained work.
            assert scope.cancelled_caught

        task = asyncio.create_task(run())
        await asyncio.sleep(0)
        scope.deadline = anyio.current_time()
        await asyncio.sleep(0)
        assert not task.done(), "level cancellation abandoned the retained future"
        future.set_result(7)
        await asyncio.wait_for(task, 5)
        assert future.result() == 7

    asyncio.run(scenario())


def test_retained_wait_level_cancel_signals_on_cancel_once() -> None:
    async def scenario() -> None:
        future = _future()
        signals: list[object] = []
        scope = anyio.CancelScope()

        async def run() -> None:
            with scope:
                await await_retained(future, on_cancel=lambda: signals.append("cancel"))
            assert scope.cancelled_caught

        task = asyncio.create_task(run())
        await asyncio.sleep(0)
        scope.deadline = anyio.current_time()
        await asyncio.sleep(0)
        assert not task.done(), "level cancellation abandoned the retained future"
        assert signals == ["cancel"], "the level cancellation signal did not fire exactly once"
        future.set_result(7)
        await asyncio.wait_for(task, 5)
        assert future.result() == 7

    asyncio.run(scenario())


def test_retained_wait_chains_completion_error_in_caller_cancel_race() -> None:
    async def scenario() -> None:
        future = _future()

        async def run() -> int:
            return await await_retained(future)

        task = asyncio.create_task(run())
        await asyncio.sleep(0)
        # Cancel the caller first so the cancellation is already pending when
        # the retained future completes with its failure in the same tick.
        task.cancel()
        future.set_exception(EngineError("boom"))
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(task, 5)
        assert isinstance(cancelled.value.__cause__, EngineError)
        assert str(cancelled.value.__cause__) == "boom"

    asyncio.run(scenario())


def test_retained_wait_chains_completion_error_onto_caller_cancellation() -> None:
    async def scenario() -> None:
        future = _future()

        async def run() -> int:
            return await await_retained(future)

        task = asyncio.create_task(run())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "the retained future was abandoned"
        future.set_exception(EngineError("boom"))
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(task, 5)
        assert isinstance(cancelled.value.__cause__, EngineError)
        assert str(cancelled.value.__cause__) == "boom"

    asyncio.run(scenario())
