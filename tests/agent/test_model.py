from typing import Any, cast
from collections.abc import Callable, Sequence
import asyncio
import copy
from dataclasses import replace
import math

import pytest

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.binding import MessageBinder
from ds4py.engine.session import Engine
from ds4py.engine.sampling import SamplingPolicy
from ds4py.errors import EngineError
from ds4py.model import DS4Model
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer


def fake_engine(*, family: str="deepseek4", script: Any=None, think: ThinkMode=ThinkMode.NONE) -> Engine:
    info = replace(
        FakeNative()._info,
        name="fake-ds4",
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )
    native = FakeNative(info=info) if script is None else FakeNative(info=info, script=script)
    return ds4py.open("f.gguf", ctx=8192, think=think, native=native)


def scripted_text(text: str) -> Callable[[Sequence[int], Sequence[int]], int]:
    tokens = tuple(tokenize_rendered(text))
    return lambda prompt, generated: (
        tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS
    )


def user(text: str="hello") -> ModelRequest:
    return ModelRequest([UserPromptPart(text)])


def make_model(session: Any, **kwargs: Any) -> DS4Model:
    return DS4Model(session, sampling=SamplingPolicy(seed=1), **kwargs)


def request(model: Any, messages: Any=None, settings: Any=None, params: Any=None) -> ModelResponse:
    return asyncio.run(model.request(
        list(messages or [user()]), settings, params or ModelRequestParameters(),
    ))


def native_calls(session: Any) -> list[Any]:
    return session._worker.call(lambda engine, native: list(native.calls)).result()


def test_identity_profile_and_native_tool_capabilities() -> None:
    with fake_engine() as engine:
        model = make_model(engine.session())

        assert model.model_name == "fake-ds4"
        assert model.system == "ds4"
        profile = cast(Any, model.profile)
        assert profile["context_window"] == 8192
        assert profile["supports_thinking"] is True
        assert profile["supports_tools"] is True
        assert profile["supports_inline_system_prompts"] is True
        assert profile["supports_json_schema_output"] is False
        assert profile["supports_json_object_output"] is False
        assert profile["default_structured_output_mode"] == "prompted"
        assert profile["json_schema_transformer"] is OpenAIJsonSchemaTransformer
        assert profile["thinking_tags"] == ("<think>", "</think>")
        assert profile["supported_native_tools"] == frozenset()
        assert model.supported_native_tools() == frozenset()
        assert model.supported_tool_addition_modes == frozenset()
        assert model.supported_tool_deferral_modes == frozenset()


def test_request_prepares_settings_and_uses_concrete_sampling_policy() -> None:
    with fake_engine(script=scripted_text("sampled")) as engine:
        session = engine.session()
        model = make_model(session)
        original_prepare = model.prepare_request
        prepared = []

        def recording_prepare(settings: Any, params: Any) -> Any:
            prepared.append((settings, params))
            return original_prepare(settings, params)

        cast(Any, model).prepare_request = recording_prepare
        settings = {"temperature": 0.25, "top_p": 0.8, "top_k": 7}
        response = request(model, settings=settings)

        assert len(prepared) == 1
        assert prepared[0][0] is settings
        assert response.parts == [TextPart("sampled")]
        samples = session._worker.call(lambda engine, native: list(cast(Any, native).samples)).result()
        assert samples[0] == (0.25, 7, 0.8, 0.05)


@pytest.mark.parametrize("field,value", [
    ("temperature", None),
    ("temperature", math.nan),
    ("top_p", math.inf),
    ("min_p", -math.inf),
    ("top_k", None),
])
def test_request_rejects_missing_or_nonfinite_sampling_before_binding(field: Any, value: Any) -> None:
    with fake_engine() as engine:
        session = engine.session()
        model = DS4Model(session, sampling=SamplingPolicy(**{field: value}))

        with pytest.raises(ValueError, match=field):
            request(model)

        assert session.transcript.tokens == ()
        assert native_calls(session) == []


def test_request_rejects_none_override_and_noninteger_token_budget() -> None:
    with fake_engine() as engine:
        session = engine.session()
        model = make_model(session)
        with pytest.raises(ValueError, match="temperature"):
            request(model, settings={"temperature": None})
        with pytest.raises((TypeError, ValueError), match="max_tokens"):
            request(model, settings={"max_tokens": None})
        assert session.transcript.tokens == ()


def test_text_and_thinking_deltas_become_distinct_parts_with_token_usage() -> None:
    text = "consider</think>answer"
    with fake_engine(script=scripted_text(text), think=ThinkMode.HIGH) as engine:
        model = make_model(engine.session())
        response = request(model)

        assert response.parts == [ThinkingPart("consider"), TextPart("answer")]
        assert response.finish_reason == "stop"
        assert response.state == "complete"
        assert response.model_name == "fake-ds4"
        assert response.provider_name == "ds4"
        assert response.provider_url is None
        assert cast(Any, response.metadata)["ds4.source"] == "sampled"
        assert cast(Any, response.metadata)["ds4.key"]
        output = len(tokenize_rendered(text))
        assert response.usage.output_tokens == output
        assert response.usage.input_tokens > 0
        assert response.usage.details["prefill_tokens"] == response.usage.input_tokens
        assert response.usage.requests == 1
        assert response.usage.total_tokens == response.usage.input_tokens + output


def test_tool_calls_have_stable_ids_and_preserve_sampled_raw_syntax() -> None:
    raw = (
        "<tool_call>\n<function=read>\n<parameter=path>\n/tmp/x\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    with fake_engine(family="qwen", script=scripted_text(raw)) as engine:
        response = request(make_model(engine.session()))

        assert len(response.parts) == 1
        part = response.parts[0]
        assert isinstance(part, ToolCallPart)
        assert part.tool_name == "read"
        assert part.args == {"path": "/tmp/x"}
        assert part.tool_call_id
        assert part.tool_call_id.startswith(cast(Any, response.metadata)["ds4.key"])
        assert part.provider_details == {"ds4py.raw": raw}
        assert response.finish_reason == "tool_call"


def test_max_tokens_rejects_partial_response_text() -> None:
    with fake_engine(script=scripted_text("ab")) as engine:
        with pytest.raises(EngineError, match="max_tokens") as caught:
            request(make_model(engine.session()), settings={"max_tokens": 1})
        assert caught.value.reason == "max_tokens"


def test_parser_failure_rejects_malformed_response() -> None:
    malformed = "<tool_call><function=read><parameter=path>unfinished"
    with fake_engine(family="qwen", script=scripted_text(malformed)) as engine:
        with pytest.raises(EngineError, match="parser_error") as caught:
            request(make_model(engine.session()))
        assert caught.value.reason == "parser_error"


def test_real_agent_renders_instructions_once_and_invalidates_changes() -> None:
    async def scenario() -> None:
        with fake_engine(script=scripted_text("done")) as engine:
            session = engine.session()
            model = make_model(session)
            agent = Agent(model, instructions="unique instruction marker")

            result = await agent.run("hello")

            assert result.output == "done"
            marker = tuple(engine.tokenize("unique instruction marker"))

            def occurrences(tokens: Any, needle: Any) -> Any:
                return [
                    index for index in range(len(tokens) - len(needle) + 1)
                    if tokens[index:index + len(needle)] == needle
                ]

            locations = occurrences(session.transcript.tokens, marker)
            assert len(locations) == 1
            assert len([
                segment for segment in session.transcript.segments
                if any(
                    segment.start <= location and location + len(marker) <= segment.end
                    for location in locations
                )
            ]) == 1

            replacement = "replacement instruction marker"
            change = model.binder.prepare(result.all_messages(), instructions=replacement)
            assert change.kind == "rewrite"
            assert occurrences(session.transcript.tokens, marker) == []
            replacement_marker = tuple(engine.tokenize(replacement))
            assert len(occurrences(session.transcript.tokens, replacement_marker)) == 1

    asyncio.run(scenario())


def test_real_agent_executes_function_tool_and_samples_its_result_next_round() -> None:
    raw = (
        "<tool_call>\n<function=add>\n<parameter=a>\n2\n</parameter>\n"
        "<parameter=b>\n3\n</parameter>\n</function>\n</tool_call>"
    )
    tool_tokens = tuple(tokenize_rendered(raw))
    answer_tokens = tuple(tokenize_rendered("sum=5"))
    result_marker = tuple(byte + 1 for byte in b"Tool result 1 (add):")

    def contains(prompt: Any, marker: Any) -> bool:
        return any(
            tuple(prompt[index:index + len(marker)]) == marker
            for index in range(len(prompt) - len(marker) + 1)
        )

    def script(prompt: Any, generated: Any) -> Any:
        tokens = answer_tokens if contains(prompt, result_marker) else tool_tokens
        return tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS

    async def scenario() -> None:
        with fake_engine(family="qwen", script=script) as engine:
            model = make_model(engine.session())
            agent = Agent(model)
            calls = []

            @agent.tool_plain
            def add(a: int, b: int) -> int:
                calls.append((a, b))
                return a + b

            result = await agent.run("add two and three")
            assert result.output == "sum=5"
            assert calls == [(2, 3)]
            responses = [m for m in result.all_messages() if isinstance(m, ModelResponse)]
            assert len(responses) == 2
            assert isinstance(responses[0].parts[0], ToolCallPart)
            returns = [
                part for message in result.all_messages() if isinstance(message, ModelRequest)
                for part in message.parts if isinstance(part, ToolReturnPart)
            ]
            assert len(returns) == 1
            assert returns[0].tool_name == "add"
            assert returns[0].content == 5
            assert result.usage.requests == 2

    asyncio.run(scenario())


def test_adopt_uses_stable_keys_without_collapsing_harness_history() -> None:
    with fake_engine() as engine:
        binder = MessageBinder(engine.session())
        history = [
            user("question-one"),
            ModelResponse([TextPart("answer-one")]),
            user("question-two"),
            ModelResponse([TextPart("answer-two")]),
        ]
        binder.prepare(history)
        request_view = [copy.deepcopy(message) for message in history[2:]]
        binder.prepare(request_view)

        adopted = binder.adopt(history)

        assert [cast(Any, message.parts[0]).content for message in adopted] == [
            "question-one", "answer-one", "question-two", "answer-two",
        ]
        assert [cast(Any, message.metadata)["ds4.key"] for message in adopted] == [
            cast(Any, message.metadata)["ds4.key"] for message in history
        ]
