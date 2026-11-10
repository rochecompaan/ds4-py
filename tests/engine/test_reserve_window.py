"""A live reserve authorizes only the scoped owner, never inherited tasks/listeners."""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import copy_context

import pytest

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative
from ds4py.engine.events import Event, ReserveReached, Stopped
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.session import Session
from ds4py.engine.transcript import TranscriptReplacement
from ds4py.errors import EngineError


@contextmanager
def paused() -> Generator[tuple[Session, ReserveReached, Generator[Event]]]:
    with ds4py.open("f.gguf", ctx=673, think=ThinkMode.NONE, native=FakeNative()) as engine:
        session = engine.session()
        session.append_user("hi")
        round_ = session.run_round()
        try:
            pause = next(event for event in round_ if isinstance(event, ReserveReached))
            yield session, pause, round_
        finally:
            round_.close()


def replace(session: Session) -> None:
    session.replace_transcript(
        TranscriptReplacement(session.transcript.tokens, session.transcript.segments), reason="owner"
    )


def test_pause_alone_does_not_authorize_replacement_or_summary() -> None:
    with paused() as (session, _, _):
        with pytest.raises(EngineError, match="round is open"):
            replace(session)
        with pytest.raises(EngineError, match="round is open"):
            session.prepare_summary(0)


def test_owner_permission_propagates_only_to_executor_not_tasks_or_listeners() -> None:
    async def scenario() -> None:
        with paused() as (session, pause, _):
            denied = []

            def listener(event: object) -> None:
                with pytest.raises(EngineError, match="round is open"):
                    replace(session)
                denied.append(True)

            session.on_event(listener, kinds={"probe"})
            with session.events_on_loop(asyncio.get_running_loop()):
                with session.reserve_window(pause):
                    replace(session)
                    await asyncio.to_thread(replace, session)
                    session.emit("probe")
                    await asyncio.to_thread(session.emit, "probe")

                    async def child() -> None:
                        with pytest.raises(EngineError, match="round is open"):
                            replace(session)

                    await asyncio.create_task(child())
                    stale = copy_context()
                await asyncio.to_thread(stale.run, listener, None)
            assert denied == [True, True, True]

    asyncio.run(scenario())


@pytest.mark.parametrize("decision", ["resume", "fail", "close"])
def test_decisions_and_close_revoke_even_captured_permission(decision: str) -> None:
    with paused() as (session, pause, round_):
        with session.reserve_window(pause):
            stale = copy_context()
            if decision == "resume":
                pause.resume(1)
            elif decision == "fail":
                pause.fail()
            else:
                session.close()
            with pytest.raises(EngineError):
                stale.run(replace, session)
        with pytest.raises(EngineError), session.reserve_window(pause):
            pass
        round_.close()


def test_expired_pause_and_nested_windows_reject() -> None:
    with paused() as (session, pause, round_):
        with session.reserve_window(pause), pytest.raises(EngineError), session.reserve_window(pause):
            pass
        round_.close()
        with pytest.raises(EngineError), session.reserve_window(pause):
            pass


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("stale_action", ["fail", "resume", "close"])
def test_expired_round_cannot_revoke_another_rounds_reserve_window(stale_action: str, strict: bool) -> None:
    with paused() as (session, expired_pause, expired_round):
        expired_round.close()
        session.append_user("next")
        prefix = PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens)
        capacity = len(session.transcript) + len(prefix) + 1
        session._worker.call(lambda e, n: setattr(n, "_ctx", capacity)).result()
        current_round = session.run_round()
        try:
            current_pause = next(event for event in current_round if isinstance(event, ReserveReached))
            with session.reserve_window(current_pause, strict=strict) as operations:
                replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
                operations.replace_transcript(replacement, reason="current owner")
                if stale_action == "close":
                    expired_round.close()
                elif stale_action == "fail":
                    with pytest.raises(EngineError, match="expired"):
                        expired_pause.fail()
                else:
                    with pytest.raises(EngineError, match="expired"):
                        expired_pause.resume(1)
                # A stale handle must neither revoke permission nor consume the
                # live decision: native replacement and that decision still work.
                operations.replace_transcript(replacement, reason="still current owner")
                current_pause.fail()
            assert list(current_round) == [Stopped("context_exhausted")]
        finally:
            current_round.close()


@pytest.mark.parametrize("sampled", [0, 3])
def test_open_assistant_observation_is_exact_and_unkeyed(sampled: int) -> None:
    with ds4py.open("f.gguf", native=FakeNative()) as engine:
        session = engine.session()
        session.append_tokens([10, 11], "user", {"key": "u"})
        assert session.open_assistant_start() is None
        session.append_tokens([20, 21], "assistant_prefix")
        if sampled:
            session.append_tokens([30] * sampled, "assistant")
        assert session.open_assistant_start() == 4
        session.append_tokens([40], "eos")
        assert session.open_assistant_start() == 4
        replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments).slice(2)
        session.replace_transcript(replacement, reason="move")
        assert session.open_assistant_start() == 2


def test_unkeyed_assistant_without_prefix_is_invalid() -> None:
    with ds4py.open("f.gguf", native=FakeNative()) as engine:
        session = engine.session()
        session.append_tokens([10], "assistant")
        with pytest.raises(EngineError, match="assistant"):
            session.open_assistant_start()


def test_scoped_summary_supports_executor_and_denies_nested_summary() -> None:
    async def scenario() -> None:
        with paused() as (session, pause, _):
            session._worker.call(lambda e, n: setattr(n, "_ctx", 4096)).result()
            denied = []

            def nested(event: object) -> None:
                with pytest.raises(EngineError, match="round is open"):
                    session.prepare_summary(0)
                with pytest.raises(EngineError), session.reserve_window(pause):
                    pass
                denied.append(True)

            session.on_event(nested, kinds={"SummaryToken"})
            with session.reserve_window(pause):
                first = session.prepare_summary(0)
                second = await asyncio.to_thread(session.prepare_summary, 0)
                assert first.summary == second.summary
                assert denied

    asyncio.run(scenario())


def test_event_loop_callback_cannot_borrow_owner_permission() -> None:
    async def scenario() -> None:
        with paused() as (session, pause, _):
            completed = asyncio.get_running_loop().create_future()

            def callback() -> None:
                try:
                    replace(session)
                except EngineError:
                    completed.set_result(True)
                else:
                    completed.set_result(False)

            with session.reserve_window(pause):
                asyncio.get_running_loop().call_soon(callback)
                assert await completed

    asyncio.run(scenario())
