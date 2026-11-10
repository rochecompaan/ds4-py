"""Strict handles authorize selected owners, not inherited execution contexts."""

import asyncio
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import override

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from test_context_strategy import RecordingOwner, fake_engine, make_model, response_text, sequential_text, user

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative
from ds4py.capabilities.compaction import LLMCompaction
from ds4py.capabilities.context import ContextRequest, RoomResult
from ds4py.engine.events import Event, ReserveReached
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


def test_strict_owner_child_executor_is_denied() -> None:
    async def scenario() -> None:
        with paused() as (session, pause, _):
            before = session.transcript.tokens
            with session.reserve_window(pause, strict=True) as operations:
                replacement = TranscriptReplacement(before, session.transcript.segments)
                with pytest.raises(EngineError):
                    session.replace_transcript(replacement, reason="raw owner")

                async def child() -> None:
                    with pytest.raises(EngineError):
                        await operations.replace_transcript_async(replacement, reason="child")
                    with pytest.raises(EngineError):
                        operations.replace_transcript(replacement, reason="child direct")
                    with pytest.raises(EngineError):
                        await asyncio.to_thread(session.replace_transcript, replacement, reason="raw child")
                    with pytest.raises(EngineError):
                        await asyncio.to_thread(operations.replace_transcript, replacement, reason="inherited handle")
                    with pytest.raises(EngineError):
                        await operations.prepare_summary_async(0)

                await asyncio.create_task(child())
                assert session.transcript.tokens == before
                operations.replace_transcript(replacement, reason="selected owner")
                await operations.replace_transcript_async(replacement, reason="selected async owner")
            with pytest.raises(EngineError):
                await operations.replace_transcript_async(replacement, reason="expired")

    asyncio.run(scenario())


def test_strict_listener_cannot_use_deliberately_supplied_handle() -> None:
    with paused() as (session, pause, _):
        denied = []
        with session.reserve_window(pause, strict=True) as operations:
            replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)

            def listener(event: object) -> None:
                with pytest.raises(EngineError):
                    operations.replace_transcript(replacement, reason="listener")
                with pytest.raises(EngineError):
                    session.replace_transcript(replacement, reason="listener raw")
                denied.append(True)

            session.on_event(listener, kinds={"probe"})
            session.emit("probe")
            assert denied == [True]


def test_strict_listener_cannot_consume_deliberately_supplied_ticket() -> None:
    with paused() as (session, pause, _), session.reserve_window(pause, strict=True) as operations:
        replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
        ticket = operations.issue_authority("replacement")
        denied = []

        def listener(event: object) -> None:
            with pytest.raises(EngineError):
                session.replace_transcript(replacement, reason="listener ticket", authority=ticket)
            denied.append(True)

        session.on_event(listener, kinds={"probe"})
        session.emit("probe")
        assert denied == [True]
        # Listener denial must not consume the selected owner's ticket.
        session.replace_transcript(replacement, reason="owner ticket", authority=ticket)


def test_tickets_are_session_operation_and_one_shot_bound() -> None:
    with (
        paused() as (session, pause, _),
        paused() as (other, _, _),
        session.reserve_window(pause, strict=True) as operations,
    ):
        replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
        ticket = operations.issue_authority("replacement")
        with pytest.raises(EngineError):
            other.replace_transcript(replacement, reason="wrong session", authority=ticket)
        with pytest.raises(EngineError):
            session.prepare_summary(0, authority=ticket)
        session.replace_transcript(replacement, reason="correct", authority=ticket)
        with pytest.raises(EngineError):
            session.replace_transcript(replacement, reason="reused", authority=ticket)


@pytest.mark.parametrize("decision", ["resume", "fail", "exit", "close"])
def test_strict_ticket_expires_with_pause(decision: str) -> None:
    with paused() as (session, pause, _):
        with session.reserve_window(pause, strict=True) as operations:
            replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
            ticket = operations.issue_authority("replacement")
            if decision == "resume":
                pause.resume(1)
            elif decision == "fail":
                pause.fail()
            elif decision == "close":
                session.close()
        with pytest.raises(EngineError):
            session.replace_transcript(replacement, reason="stale", authority=ticket)


def test_legacy_child_executor_ambiguity_is_preserved() -> None:
    async def scenario() -> None:
        with paused() as (session, pause, _):
            replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
            with session.reserve_window(pause):

                async def child() -> None:
                    await asyncio.to_thread(session.replace_transcript, replacement, reason="legacy")

                await asyncio.create_task(child())

    asyncio.run(scenario())


def test_strict_summary_supports_sync_and_retained_async_owner() -> None:
    async def scenario() -> None:
        with paused() as (session, pause, _):
            session._worker.call(lambda e, n: setattr(n, "_ctx", 4096)).result()
            before = session.transcript.tokens
            with session.reserve_window(pause, strict=True) as operations:
                with pytest.raises(EngineError):
                    session.prepare_summary(0)
                first = operations.prepare_summary(0)
                second = await operations.prepare_summary_async(0)
                assert first.summary == second.summary == "ok"
                assert session.transcript.tokens == before

    asyncio.run(scenario())


def test_previous_window_ticket_cannot_authorize_or_revoke_current_window() -> None:
    async def scenario() -> None:
        with paused() as (session, pause, _):
            replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
            with session.reserve_window(pause, strict=True) as expired:
                stale = expired.issue_authority("replacement")
            with session.reserve_window(pause, strict=True) as current:
                with pytest.raises(EngineError):
                    await asyncio.to_thread(session.replace_transcript, replacement, reason="stale", authority=stale)
                await current.replace_transcript_async(replacement, reason="live")

    asyncio.run(scenario())


@pytest.mark.parametrize("retained", [False, True])
def test_opted_in_owner_uses_explicit_operations(retained: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    selections = []

    def select_strict(owner: RecordingOwner) -> bool:
        selections.append(True)
        return len(selections) == 1

    monkeypatch.setattr(RecordingOwner, "reserve_operations", property(select_strict))

    async def scenario() -> None:
        with fake_engine(script=sequential_text("abc")) as engine:
            session = engine.session()
            model = make_model(session)

            async def on_room(messages: list[ModelMessage], request: ContextRequest) -> None:
                assert request.operations is not None
                replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
                with pytest.raises(EngineError):
                    session.replace_transcript(replacement, reason="raw strict owner")
                session._worker.call(lambda e, n: setattr(n, "_ctx", 4096)).result()
                if retained:
                    await request.operations.replace_transcript_async(replacement, reason="custom")
                else:
                    request.operations.replace_transcript(replacement, reason="custom")

            owner = RecordingOwner(model.binder, on_room=on_room)
            model.context_owner = owner
            async with model.request_stream([user()], None, ModelRequestParameters()) as stream:
                prefix = len(PromptBuilder(engine).assistant_prefix(session.transcript.tokens))
                capacity = len(session.transcript) + prefix + 2
                session._worker.call(lambda e, n: setattr(n, "_ctx", capacity)).result()
                async for _ in stream:
                    pass
                assert response_text(stream.get()) == "abc"
            assert len(owner.requests) == 1
            operations = owner.requests[0].operations
            assert operations is not None
            with pytest.raises(EngineError):
                operations.prepare_summary(0)

    asyncio.run(scenario())


@pytest.mark.parametrize("legacy", [False, True])
def test_raw_call_compaction_subclass_requires_explicit_legacy_opt_out(legacy: bool) -> None:
    class RawCompactor(LLMCompaction):
        @override
        async def make_room(self, messages: list[ModelMessage], request: ContextRequest) -> RoomResult:
            session = self.binder.session
            replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
            session.replace_transcript(replacement, reason="raw subclass")
            session._worker.call(lambda e, n: setattr(n, "_ctx", 4096)).result()
            return RoomResult(True)

    async def scenario() -> None:
        with fake_engine(script=sequential_text("abc")) as engine:
            session = engine.session()
            model = make_model(session)
            owner = RawCompactor(model.binder)
            if legacy:
                owner.reserve_operations = False
            model.context_owner = owner
            async with model.request_stream([user()], None, ModelRequestParameters()) as stream:
                prefix = len(PromptBuilder(engine).assistant_prefix(session.transcript.tokens))
                capacity = len(session.transcript) + prefix + 2
                session._worker.call(lambda e, n: setattr(n, "_ctx", capacity)).result()

                async def consume() -> None:
                    async for _ in stream:
                        pass

                if legacy:
                    await consume()
                    assert response_text(stream.get()) == "abc"
                else:
                    with pytest.raises(EngineError, match="round is open"):
                        await consume()

    asyncio.run(scenario())


def test_round_close_revokes_retained_executor_ticket_before_claim() -> None:
    async def scenario() -> None:
        with paused() as (session, pause, round_), session.reserve_window(pause, strict=True) as operations:
            replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
            ticket = operations.issue_authority("replacement")
            entered = asyncio.Event()
            release = threading.Event()
            loop = asyncio.get_running_loop()

            def execute() -> None:
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5)
                with pytest.raises(EngineError):
                    session.replace_transcript(replacement, reason="closed round", authority=ticket)

            pending = asyncio.create_task(asyncio.to_thread(execute))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                await asyncio.to_thread(round_.close)
                closed_tokens = session.transcript.tokens
            finally:
                release.set()
                await pending
            assert session.transcript.tokens == closed_tokens

    asyncio.run(scenario())
