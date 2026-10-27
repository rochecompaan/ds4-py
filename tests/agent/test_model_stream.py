import asyncio
from dataclasses import replace
import threading

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeEngine, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.binding import MessageBinder
from ds4py.engine.sampling import SamplingPolicy
from ds4py.engine.worker import SyncJob
from ds4py.model import DS4Model
from pydantic_ai.messages import (
    ModelRequest,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters


def fake_engine(*, script=None, think=ThinkMode.NONE, cancel=None):
    native = FakeNative(script=script) if script is not None else FakeNative()
    info = replace(native._info, name="fake-ds4")
    native._info = info
    return ds4py.open("f.gguf", ctx=8192, think=think, native=native, cancel=cancel)


def scripted_text(text):
    tokens = tuple(tokenize_rendered(text))
    return lambda prompt, generated: (
        tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS
    )


def user(text="hello"):
    return ModelRequest([UserPromptPart(text)])


def make_model(session, **kwargs):
    return DS4Model(session, sampling=SamplingPolicy(seed=1), **kwargs)


async def collect(model, messages, settings=None):
    async with model.request_stream(
        list(messages), settings, ModelRequestParameters(),
    ) as stream:
        events = [event async for event in stream]
        return stream.get(), events


def test_empty_output_and_zero_prefill_have_zero_usage(monkeypatch):
    monkeypatch.setattr(FakeEngine, "assistant_prefix", lambda self, prefix, mode: [])

    async def scenario():
        with fake_engine(script=scripted_text("")) as engine:
            session = engine.session()
            model = make_model(session)
            async with model.request_stream(
                [user()], None, ModelRequestParameters(),
            ) as stream:
                # The binder has rendered the request, but the lazy round has not synced it.
                list(session._worker.submit(SyncJob(session.transcript.tokens)))
                assert [event async for event in stream] == []
                response = stream.get()

            assert response.parts == []
            assert response.finish_reason == "stop"
            assert response.usage.input_tokens == 0
            assert response.usage.output_tokens == 0
            assert response.usage.details == {"prefill_tokens": 0}
            assert response.usage.requests == 1
            assert response.usage.total_tokens == 0

    asyncio.run(scenario())


def test_thinking_only_output_is_retained():
    async def scenario():
        with fake_engine(script=scripted_text("thought"), think=ThinkMode.HIGH) as engine:
            response, events = await collect(make_model(engine.session()), [user()])
            assert response.parts == [ThinkingPart("thought")]
            assert response.finish_reason == "stop"
            assert any(isinstance(event, PartStartEvent) for event in events)

    asyncio.run(scenario())


def test_interrupted_round_returns_partial_text_as_interrupted():
    cancel = threading.Event()

    def script(prompt, generated):
        if len(generated) == 2:
            cancel.set()
        return ord("a") + 1

    async def scenario():
        with fake_engine(script=script, cancel=cancel) as engine:
            response, _ = await collect(make_model(engine.session()), [user()])
            assert response.parts == [TextPart("aaa")]
            assert response.state == "interrupted"
            assert response.finish_reason is None
            assert response.usage.output_tokens == 3
            assert response.metadata["ds4.source"] == "sampled"

    asyncio.run(scenario())


def test_closing_event_iteration_binds_only_the_published_partial_response():
    outputs = [tuple(tokenize_rendered("partial")), tuple(tokenize_rendered("next"))]
    prompts = []

    def script(prompt, generated):
        if not generated:
            prompts.append(tuple(prompt))
            script.current = outputs.pop(0)
        tokens = script.current
        return tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS

    async def scenario():
        with fake_engine(script=script) as engine:
            session = engine.session()
            binder = MessageBinder(session)
            model = make_model(session, binder=binder)
            bindings = 0
            original_bind = binder.bind_response

            def counted_bind(response):
                nonlocal bindings
                bindings += 1
                original_bind(response)

            binder.bind_response = counted_bind
            request = user()
            async with model.request_stream(
                [request], None, ModelRequestParameters(),
            ) as stream:
                iterator = stream.__aiter__()
                event = await anext(iterator)
                assert isinstance(event, PartStartEvent)
                assert event.part == TextPart("p")
                await iterator.aclose()
                response = stream.get()

            assert response.parts == [TextPart("p")]
            assert response.state == "incomplete"
            assert response.usage.output_tokens == 1
            assert bindings == 1
            provenance = binder.export_segments()
            sampled = [p for p in provenance if p.source == "sampled"]
            assert len(sampled) == 1
            assert sampled[0].kind == "assistant"
            assert sampled[0].key == response.metadata["ds4.key"]
            assert session.transcript.tokens[sampled[0].start:sampled[0].end] == tuple(
                tokenize_rendered("p")
            )
            assert sampled[0].end == len(session.transcript)
            assert not [p for p in provenance if p.kind == "eos"]
            assert binder._records[-1].end == len(session.transcript)
            native_tokens = session._worker.call(
                lambda engine, native: native.tokens()
            ).result()
            assert tuple(native_tokens) == session.transcript.tokens

            before = session.transcript.tokens
            second, _ = await collect(model, [request, response, user("next request")])

            assert second.parts == [TextPart("next")]
            assert second.finish_reason == "stop"
            assert session.transcript.tokens[:len(before)] == before
            assert binder.export_segments()[:len(provenance)] == provenance
            assert prompts[1][:len(before)] == before
            assert not outputs

    asyncio.run(scenario())


def test_request_relies_on_stream_terminal_binding_exactly_once():
    async def scenario():
        with fake_engine(script=scripted_text("done")) as engine:
            session = engine.session()
            binder = MessageBinder(session)
            model = make_model(session, binder=binder)
            bindings = 0
            original_bind = binder.bind_response

            def counted_bind(response):
                nonlocal bindings
                bindings += 1
                original_bind(response)

            binder.bind_response = counted_bind
            response = await model.request([user()], None, ModelRequestParameters())
            assert response.parts == [TextPart("done")]
            assert bindings == 1
            assert binder._records[-1].key == response.metadata["ds4.key"]

    asyncio.run(scenario())


def test_direct_stream_response_is_reused_by_the_next_request_without_rerendering():
    outputs = [tuple(tokenize_rendered("first")), tuple(tokenize_rendered("second"))]

    def script(prompt, generated):
        if not generated:
            tokens = outputs.pop(0)
            script.current = tokens
        tokens = script.current
        return tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS

    async def scenario():
        with fake_engine(script=script) as engine:
            session = engine.session()
            binder = MessageBinder(session)
            model = make_model(session, binder=binder)
            request = user("first request")
            first, _ = await collect(model, [request])
            first_key = first.metadata["ds4.key"]
            first_hash = next(
                p.content_hash for p in binder.export_segments()
                if p.key == first_key and p.source == "sampled"
            )
            before = session.transcript.tokens
            before_provenance = binder.export_segments()

            second, _ = await collect(model, [request, first, user("next")])

            assert second.parts == [TextPart("second")]
            assert session.transcript.tokens[:len(before)] == before
            assert len(binder._records) == 4
            assert [record.key for record in binder._records[:2]] == [
                request.metadata["ds4.key"], first_key,
            ]
            assert binder._records[1].end == len(before)
            later = binder.export_segments()
            assert later[:len(before_provenance)] == before_provenance
            assert next(
                p.content_hash for p in later if p.key == first_key and p.source == "sampled"
            ) == first_hash
            assert len([p for p in later if p.kind == "assistant"]) == 2
            assert not outputs

    asyncio.run(scenario())
