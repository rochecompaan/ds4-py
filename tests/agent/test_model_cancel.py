"""Adapter cancellation, reserve failure, and speculative alignment regressions."""
from typing import Any, cast
import asyncio
from dataclasses import replace
from pathlib import Path
import sys
import threading

import pytest

import ds4py
from ds4py.engine.session import Engine
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative, FakeSession
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.binding import MessageBinder
from ds4py.engine import decode as decode_module
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.sampling import SamplingPolicy
from ds4py.model import ContextReserveError, DS4Model
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import ModelRequestTimeout
from pydantic_ai.messages import ModelRequest, PartStartEvent, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from fake_steps import FakeMultiTokenStep  # type: ignore[import-not-found]  # noqa: E402


def fake_engine(*, script: Any=None, think: Any=ThinkMode.NONE, cancel: Any=None) -> Engine:
    native = FakeNative(script=script) if script is not None else FakeNative()
    native._info = replace(native._info, name="fake-ds4")
    return ds4py.open("f.gguf", ctx=8192, think=think, native=native, cancel=cancel)


def user(text: Any="hello") -> ModelRequest:
    return ModelRequest([UserPromptPart(text)])


def make_model(session: Any, **kwargs: Any) -> DS4Model:
    return DS4Model(session, sampling=SamplingPolicy(seed=1), **kwargs)


def native_tokens(session: Any) -> tuple[int, ...]:
    return tuple(session._worker.call(lambda engine, native: native.tokens()).result())


def native_calls(session: Any) -> list[Any]:
    return session._worker.call(lambda engine, native: list(native.calls)).result()


def scripted_text(text: Any) -> Any:
    tokens = tuple(tokenize_rendered(text))
    return lambda prompt, generated: (
        tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS
    )


def response_text(response: Any) -> str:
    return "".join(part.content for part in response.parts if isinstance(part, TextPart))


def test_asyncio_cancellation_cancels_facade_and_preserves_partial_response() -> None:
    # Cancel while next() is waiting on an in-flight sample. The third token is
    # already being sampled, so it is committed before the facade stops; no
    # further token may be sampled.
    reached, gate = threading.Event(), threading.Event()
    tokens = tuple(tokenize_rendered("abc"))
    calls = []

    def script(prompt: Any, generated: Any) -> Any:
        calls.append(len(generated))
        if len(generated) == 2:
            reached.set()
            assert gate.wait(5), "cancellation gate timed out"
        return tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS

    async def scenario() -> None:
        cancel = threading.Event()
        with fake_engine(script=script, cancel=cancel) as engine:
            session = engine.session()
            model = make_model(session)
            async with model.request_stream(
                [user()], None, ModelRequestParameters(),
            ) as stream:
                pull = asyncio.create_task(cast(Any, anext(stream.__aiter__())))
                await asyncio.wait_for(asyncio.to_thread(reached.wait, 5), 10)
                pull.cancel()
                await asyncio.wait_for(asyncio.to_thread(cancel.wait, 5), 10)
                gate.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(pull, 10)
                response = cast(Any, stream).get()

            assert response.state == "interrupted"
            assert response_text(response) == "abc"
            assert session.transcript.tokens == native_tokens(session)
            assert session.transcript.tokens[-3:] == tokens
            assert calls == [0, 1, 2]
            assert not cancel.is_set()

    asyncio.run(scenario())


def test_close_stream_cancels_without_waiting_for_another_token() -> None:
    # The consumer has a committed delta, then closes the stream while the next
    # sample is in flight. Closing cancels the facade and drains only that
    # in-flight sample: it must not ask the script for a further token.
    reached = threading.Event()
    cancel = threading.Event()
    calls = []
    tokens = tuple(tokenize_rendered("ab"))

    def script(prompt: Any, generated: Any) -> Any:
        calls.append(len(generated))
        if len(generated) == 1:
            reached.set()
            assert cancel.wait(5), "facade cancellation was not applied"
            return FAKE_EOS
        return tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS

    async def scenario() -> None:
        with fake_engine(script=script, cancel=cancel) as engine:
            session = engine.session()
            model = make_model(session)
            async with model.request_stream(
                [user()], None, ModelRequestParameters(),
            ) as stream:
                iterator = stream.__aiter__()
                first = await asyncio.wait_for(anext(iterator), 10)
                assert cast(Any, first).part == TextPart("a")
                await asyncio.wait_for(asyncio.to_thread(reached.wait, 5), 10)
                await asyncio.wait_for(stream.cancel(), 10)
                response = cast(Any, stream).get()

                assert response.state == "interrupted"
                assert response_text(response) == "a"
                assert session.transcript.tokens == native_tokens(session)
                assert calls == [0, 1]
                # Closing releases the facade round so the session is reusable.
                session.append_user("next")

    asyncio.run(scenario())


def test_cancel_during_prefill_leaves_transcript_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    waiting, release = threading.Event(), threading.Event()
    original_sync = FakeSession.sync

    def blocking_sync(self: Any, tokens: Any, *, progress: Any=None, cancel: Any=None) -> Any:
        waiting.set()
        assert release.wait(5), "prefill release timed out"
        return original_sync(self, tokens, progress=progress, cancel=cancel)

    monkeypatch.setattr(FakeSession, "sync", blocking_sync)

    async def scenario() -> None:
        with fake_engine(script=scripted_text("abc")) as engine:
            session = engine.session()
            model = make_model(session)
            async with model.request_stream(
                [user()], None, ModelRequestParameters(),
            ) as stream:
                prompt = session.transcript.tokens
                cancel = session._worker.cancel_event
                pull = asyncio.create_task(cast(Any, anext(stream.__aiter__())))
                await asyncio.wait_for(asyncio.to_thread(waiting.wait, 5), 10)
                pull.cancel()
                await asyncio.wait_for(asyncio.to_thread(cancel.wait, 5), 10)
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(pull, 10)
                response = cast(Any, stream).get()

            assert response.state == "interrupted"
            assert response.parts == []
            assert response.usage.output_tokens == 0
            assert session.transcript.tokens == prompt
            # M2 keeps an interrupted prefill out of live KV; the next round
            # restores it from the intact transcript.
            assert session.ctx_used == 0
            assert not [call for call in native_calls(session) if call[0] == "eval"]

    asyncio.run(scenario())


def test_cancel_during_speculative_block_rewinds_uncommitted_drafts(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens = tuple(tokenize_rendered("abcdefgh"))
    target = tokens[4]
    reached, gate = threading.Event(), threading.Event()
    original_eval = FakeSession.eval

    def blocking_eval(self: Any, token: Any) -> Any:
        if token == target:
            reached.set()
            assert gate.wait(5), "draft gate timed out"
        return original_eval(self, token)

    monkeypatch.setattr(FakeSession, "eval", blocking_eval)
    # First block commits "ab". The second block drafts "def"; cancellation is
    # requested while its drafts are evaluated, so none of them may commit.
    blocks = [tokens[1:2], tokens[3:6]]
    monkeypatch.setattr(
        decode_module, "PlainStep", lambda: FakeMultiTokenStep(blocks),
    )

    async def scenario() -> None:
        with fake_engine(script=scripted_text("abcdefgh")) as engine:
            session = engine.session()
            model = make_model(session)
            async with model.request_stream(
                [user()], None, ModelRequestParameters(),
            ) as stream:
                async def consume() -> None:
                    async for _ in stream:
                        pass

                cancel = session._worker.cancel_event
                consumer = asyncio.create_task(consume())
                await asyncio.wait_for(asyncio.to_thread(reached.wait, 5), 10)
                consumer.cancel()
                await asyncio.wait_for(asyncio.to_thread(cancel.wait, 5), 10)
                gate.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(consumer, 10)
                response = cast(Any, stream).get()

            assert response.state == "interrupted"
            assert response_text(response) == "ab"
            assert session.transcript.tokens == native_tokens(session)
            assert session.transcript.tokens[-2:] == tokens[:2]
            calls = native_calls(session)
            evaluated = {token for kind, token in calls if kind == "eval"}
            assert tokens[2] in evaluated and tokens[3] in evaluated
            prefix = max(
                segment.end for segment in session.transcript.segments
                if segment.kind == "assistant_prefix"
            )
            assert session.transcript.tokens[prefix:] == tokens[:2]
            assert any(kind == "rewind" for kind, *_ in calls)

    asyncio.run(scenario())


def test_context_reserve_default_fails_clearly_without_compaction() -> None:
    async def scenario() -> None:
        with fake_engine(script=scripted_text("abc")) as engine:
            session = engine.session()
            model = make_model(session)
            async with model.request_stream(
                [user()], None, ModelRequestParameters(),
            ) as stream:
                prompt_length = len(session.transcript)
                prefix = len(PromptBuilder(session.engine).assistant_prefix(
                    session.transcript.tokens
                ))
                session._worker.call(
                    lambda engine, native: setattr(
                        native, "_ctx", prompt_length + prefix + 2
                    )
                ).result()
                with pytest.raises(ContextReserveError) as raised:
                    async for _ in stream:
                        pass
                response = cast(Any, stream).get()

            message = str(raised.value)
            assert "context reserve" in message
            assert "compaction" in message and "M4" in message
            assert response.state == "incomplete"
            assert response_text(response) == "a"
            assert session.transcript.tokens == native_tokens(session)

    asyncio.run(scenario())


def sequential_text(text: Any) -> Any:
    tokens = tuple(tokenize_rendered(text))
    index = 0

    def script(prompt: Any, generated: Any) -> Any:
        nonlocal index
        token = tokens[index] if index < len(tokens) else FAKE_EOS
        index += 1
        return token

    return script


def test_paused_round_can_resume_without_second_assistant_prefix() -> None:
    async def baseline() -> Any:
        with fake_engine(script=sequential_text("abc")) as engine:
            session = engine.session()
            model = make_model(session)
            async with model.request_stream(
                [user()], None, ModelRequestParameters(),
            ) as stream:
                async for _ in stream:
                    pass
                return session.transcript.tokens, [
                    segment.kind for segment in session.transcript.segments
                ]

    expected_tokens, expected_kinds = asyncio.run(baseline())

    async def scenario() -> None:
        with fake_engine(script=sequential_text("abc")) as engine:
            session = engine.session()
            decisions = []

            def make_room(needed: Any) -> int:
                decisions.append(needed)
                session._worker.call(
                    lambda engine, native: setattr(native, "_ctx", 4096)
                ).result()
                return 0

            model = make_model(session, reserve_handler=make_room)
            async with model.request_stream(
                [user()], None, ModelRequestParameters(),
            ) as stream:
                prompt_length = len(session.transcript)
                prefix = len(PromptBuilder(session.engine).assistant_prefix(
                    session.transcript.tokens
                ))
                session._worker.call(
                    lambda engine, native: setattr(
                        native, "_ctx", prompt_length + prefix + 2
                    )
                ).result()
                async for _ in stream:
                    pass
                response = cast(Any, stream).get()

            assert decisions == [1]
            assert response_text(response) == "abc"
            assert session.transcript.tokens == expected_tokens
            kinds = [segment.kind for segment in session.transcript.segments]
            assert kinds == expected_kinds
            assert kinds.count("assistant_prefix") == 1
            assert kinds.count("assistant") == 1
            expected_prefill = prompt_length + prefix
            assert response.usage.input_tokens == expected_prefill
            assert cast(Any, response.usage.details) == {"prefill_tokens": expected_prefill}
            assert response.usage.output_tokens == 3
            assert response.usage.requests == 1
            assert response.usage.total_tokens == expected_prefill + 3

    asyncio.run(scenario())


def test_public_request_timeout_cleans_up_round_and_preserves_partial_response() -> None:
    cancel = threading.Event()
    reached = threading.Event()
    tokens = tuple(tokenize_rendered("ab"))

    def script(prompt: Any, generated: Any) -> Any:
        if len(generated) == 1:
            reached.set()
            assert cancel.wait(5), "request timeout did not cancel the sample"
            return FAKE_EOS
        return tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS

    async def scenario() -> None:
        with fake_engine(script=script, cancel=cancel) as engine:
            session = engine.session()
            binder = MessageBinder(session)
            model = make_model(session, binder=binder)
            bindings = 0
            original_bind = binder.bind_response

            def counted_bind(response: Any, **kwargs: Any) -> None:
                nonlocal bindings
                bindings += 1
                original_bind(response, **kwargs)

            binder.bind_response = counted_bind
            events = []
            stream = None
            with pytest.raises(ModelRequestTimeout):
                async with model_request_stream(
                    model, [user()], model_settings={"request_timeout": 0.2},
                ) as stream:
                    async for event in stream:
                        events.append(event)

            assert reached.is_set()
            assert [event.part for event in events if isinstance(event, PartStartEvent)] == [
                TextPart("a")
            ]
            response = cast(Any, stream).get()
            assert response.state == "interrupted"
            assert response_text(response) == "a"
            assert response.usage.input_tokens > 0
            assert cast(Any, response.usage.details) == {
                "prefill_tokens": response.usage.input_tokens,
            }
            assert response.usage.output_tokens == 1
            assert response.usage.requests == 1
            assert bindings == 1
            assert cast(Any, stream)._bound and cast(Any, stream)._round_exhausted
            assert session._active_round is False
            assert session.transcript.tokens == native_tokens(session)
            sampled = [p for p in binder.export_segments() if p.source == "sampled"]
            assert [p.kind for p in sampled] == ["assistant"]
            assert sampled[0].key == cast(Any, response.metadata)["ds4.key"]
            assert not cancel.is_set()

    asyncio.run(scenario())


def test_second_round_prefill_cancel_preserves_prior_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    waiting, release = threading.Event(), threading.Event()
    original_sync = FakeSession.sync
    syncs = []

    def blocking_sync(self: Any, tokens: Any, *, progress: Any=None, cancel: Any=None) -> Any:
        syncs.append(len(tokens))
        if len(syncs) == 2:
            waiting.set()
            assert release.wait(5), "prefill release timed out"
        return original_sync(self, tokens, progress=progress, cancel=cancel)

    monkeypatch.setattr(FakeSession, "sync", blocking_sync)
    outputs = [tuple(tokenize_rendered("abc")), tuple(tokenize_rendered("xy"))]

    def script(prompt: Any, generated: Any) -> Any:
        if not generated:
            cast(Any, script).current = outputs.pop(0)
        tokens = cast(Any, script).current
        return tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS

    async def scenario() -> None:
        with fake_engine(script=script) as engine:
            session = engine.session()
            binder = MessageBinder(session)
            model = make_model(session, binder=binder)
            request = user("hello")
            async with model.request_stream(
                [request], None, ModelRequestParameters(),
            ) as stream:
                async for _ in stream:
                    pass
                first = stream.get()
            assert response_text(first) == "abc"
            before_tokens = session.transcript.tokens
            before_native = native_tokens(session)
            before_provenance = binder.export_segments()
            before_records = tuple(binder._records)
            before_kinds = [segment.kind for segment in session.transcript.segments]

            second_user = user("second")
            messages = [request, first, second_user]
            async with model.request_stream(
                messages, None, ModelRequestParameters(),
            ) as stream:
                pull = asyncio.create_task(cast(Any, anext(stream.__aiter__())))
                await asyncio.wait_for(asyncio.to_thread(waiting.wait, 5), 10)
                cancel = session._worker.cancel_event
                pull.cancel()
                await asyncio.wait_for(asyncio.to_thread(cancel.wait, 5), 10)
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(pull, 10)
                cancelled = stream.get()

            assert cancelled.parts == []
            assert cancelled.state == "interrupted"
            assert session.transcript.tokens[:len(before_tokens)] == before_tokens
            assert [segment.kind for segment in session.transcript.segments[
                :len(before_kinds)
            ]] == before_kinds
            assert binder.export_segments()[:len(before_provenance)] == before_provenance
            assert tuple(binder._records[:len(before_records)]) == before_records
            records = tuple(binder._records)
            assert len(records) == len(before_records) + 1
            assert all(left.end <= right.start for left, right in zip(records, records[1:]))
            assert records[-1].key == cast(Any, second_user.metadata)["ds4.key"]
            assert records[-1].end == len(session.transcript)
            # The interrupted prefill kept the previous committed KV and added no
            # assistant prefix or draft: the new user message is transcript-only
            # until the next round syncs it.
            assert native_tokens(session) == before_native
            assert len(session.transcript) > len(before_tokens)
            assert session.transcript.segments[-1].kind == "user"
            assert not cancel.is_set()

            async with model.request_stream(
                messages, None, ModelRequestParameters(),
            ) as stream:
                async for _ in stream:
                    pass
                second = stream.get()

            assert response_text(second) == "xy"
            assert session.transcript.tokens[:len(before_tokens)] == before_tokens
            provenance = binder.export_segments()
            assert provenance[:len(before_provenance)] == before_provenance
            assert all(left.end == right.start for left, right in zip(provenance, provenance[1:]))
            records = tuple(binder._records)
            assert all(left.end <= right.start for left, right in zip(records, records[1:]))
            native = native_tokens(session)
            assert session.transcript.tokens[:len(native)] == native

            # A replacement turn rewrites only from the second user message;
            # the cancelled round left no overlapping prefix behind.
            change = binder.prepare([request, first, user("replacement")])
            assert change.kind == "rewrite"
            assert change.start == len(before_tokens)
            assert binder.export_segments()[:len(before_provenance)] == before_provenance

    asyncio.run(scenario())
