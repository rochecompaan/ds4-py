"""One registered context owner serves tool overflow and reserve recovery."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast, override

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.tools import ToolDefinition

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.binding import MessageBinder
from ds4py.capabilities.context import ContextRequest, ContextStrategy, RoomResult
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.sampling import SamplingPolicy
from ds4py.engine.session import Engine
from ds4py.errors import ContextExhausted
from ds4py.model import DS4Model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic_ai.messages import ModelMessage

TOOL_ERROR = "Tool error: tool result still does not fit"


class _TranscriptView:
    def __init__(self, length: int) -> None:
        self.length = length
        self.chat = True

    def __len__(self) -> int:
        return self.length


class StubBinder:
    """Stand-in for ``MessageBinder`` with literal exact token counts."""

    def __init__(self, counts: list[int], *, transcript_tokens: int, ctx: int) -> None:
        self._counts = iter(counts)
        self.session = SimpleNamespace(
            transcript=_TranscriptView(transcript_tokens),
            engine=SimpleNamespace(config=SimpleNamespace(ctx=ctx)),
        )

    def count_bound_request(self, messages: list[ModelMessage]) -> int:
        return next(self._counts)

    def count_rendered_request(self, messages: list[ModelMessage], **kwargs: object) -> int:
        return next(self._counts)


class RecordingOwner(ContextStrategy):
    def __init__(
        self,
        binder: object,
        *,
        room: RoomResult | None = None,
        on_room: Callable[[list[ModelMessage], ContextRequest], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(cast("Any", binder))
        self.room = room or RoomResult(made_room=True)
        self.on_room = on_room
        self.requests: list[ContextRequest] = []
        self.message_lists: list[list[ModelMessage]] = []
        self.live_message_lists: list[list[ModelMessage]] = []

    @override
    async def make_room(self, messages: list[ModelMessage], request: ContextRequest) -> RoomResult:
        self.requests.append(request)
        self.message_lists.append(list(messages))
        self.live_message_lists.append(messages)
        if self.on_room is not None:
            await self.on_room(messages, request)
        return self.room


def fake_engine(*, family: str = "deepseek4", script: Any = None) -> Engine:
    info = replace(
        FakeNative()._info,
        name="fake-ds4",
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )
    native = FakeNative(info=info, script=script) if script is not None else FakeNative(info=info)
    return ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=native)


def user(text: str = "hello") -> ModelRequest:
    return ModelRequest([UserPromptPart(text)])


def make_model(session: Any) -> DS4Model:
    return DS4Model(session, sampling=SamplingPolicy(seed=1))


def response_text(response: Any) -> str:
    return "".join(str(part.content) for part in response.parts if isinstance(part, TextPart))


def sequential_text(text: str) -> Any:
    tokens = tuple(tokenize_rendered(text))
    index = 0

    def script(prompt: Any, generated: Any) -> Any:
        nonlocal index
        token = tokens[index] if index < len(tokens) else FAKE_EOS
        index += 1
        return token

    return script


BIG_TOOL_RESULT = "z" * 20_000
TOOL_CALL_RAW = "<tool_call>\n<function=read_big>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>"


def tool_then_answer(tool_call: str, answer: str) -> Any:
    rounds = (tuple(tokenize_rendered(tool_call)), tuple(tokenize_rendered(answer)))
    state = {"round": 0}

    def script(prompt: Any, generated: Any) -> Any:
        tokens = rounds[min(state["round"], len(rounds) - 1)]
        if len(generated) < len(tokens):
            return tokens[len(generated)]
        state["round"] += 1
        return FAKE_EOS

    return script


def _has_tool_return(messages: Any) -> bool:
    return any(isinstance(part, ToolReturnPart) for message in messages for part in getattr(message, "parts", []))


def _tool_agent(owner: ContextStrategy, result: str) -> Agent[Any, str]:
    def scripted(messages: Any, info: Any) -> ModelResponse:
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("read_big", {}, tool_call_id="call-1")])

    agent: Agent[Any, str] = Agent(FunctionModel(scripted), capabilities=[owner])

    @agent.tool_plain
    def read_big() -> str:
        return result

    return agent


def _tool_return(result: Any) -> ToolReturnPart:
    return next(
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )


def prepared_tool_binder(session: Any) -> MessageBinder:
    binder = MessageBinder(session)
    # Match the no-argument tool declared by _tool_agent, including its schema.
    binder.prepare(
        [user("go"), ModelResponse([ToolCallPart("read_big", {}, tool_call_id="call-1")])],
        tools=[
            ToolDefinition(
                name="read_big",
                parameters_json_schema={"additionalProperties": False, "properties": {}, "type": "object"},
            )
        ],
    )
    return binder


def test_registered_owner_receives_tool_overflow() -> None:
    with fake_engine() as engine:
        binder = prepared_tool_binder(engine.session())

        async def make_room(messages: list[ModelMessage], request: ContextRequest) -> None:
            first = cast("ModelRequest", messages[0])
            cast("UserPromptPart", first.parts[0]).content = "g"

        owner = RecordingOwner(binder, on_room=make_room)
        result = asyncio.run(_tool_agent(owner, "x" * 7287).run("go"))

        assert owner.requests == [ContextRequest(1, "tool_result")]
        assert owner.message_lists == [list(result.all_messages()[:2])]
        assert _tool_return(result).content == "x" * 7287


def test_fitting_tool_result_does_not_ask_the_owner() -> None:
    with fake_engine() as engine:
        owner = RecordingOwner(prepared_tool_binder(engine.session()))
        result = asyncio.run(_tool_agent(owner, "small").run("go"))
        assert owner.requests == []
        assert _tool_return(result).content == "small"


def test_failed_tool_room_returns_the_still_does_not_fit_error() -> None:
    with fake_engine() as engine:
        owner = RecordingOwner(prepared_tool_binder(engine.session()), room=RoomResult(made_room=False))
        result = asyncio.run(_tool_agent(owner, "x" * 7287).run("go"))
        assert owner.requests == [ContextRequest(1, "tool_result")]
        assert _tool_return(result).content == TOOL_ERROR


def test_tool_fit_check_preserves_sampled_spans_and_counts_assistant_prefix() -> None:
    """The fit base is the live transcript, not a scratch render of the request list."""

    async def scenario() -> tuple[RecordingOwner, int, int, list[ModelMessage], Any]:
        with fake_engine(family="qwen", script=tool_then_answer(TOOL_CALL_RAW, "done")) as engine:
            session = engine.session()
            model = make_model(session)
            prompts: list[int] = []
            snapshots: list[list[ModelMessage]] = []

            async def on_room(messages: list[ModelMessage], _request: ContextRequest) -> None:
                prompts.append(len(session.transcript))
                snapshots.append(list(messages))

            owner = RecordingOwner(model.binder, room=RoomResult(made_room=False), on_room=on_room)
            model.context_owner = owner
            agent: Agent[Any, str] = Agent(model, capabilities=[owner])

            @agent.tool_plain
            def read_big(path: str) -> str:
                return BIG_TOOL_RESULT

            result = await agent.run("go")

            assert result.output == "done"
            assert len(prompts) == 1
            return owner, prompts[0], engine.config.ctx, snapshots[0], result

    owner, base, ctx, messages, result = asyncio.run(scenario())
    call = next(
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    )
    candidate = ModelRequest([ToolReturnPart(call.tool_name, BIG_TOOL_RESULT, tool_call_id=call.tool_call_id)])
    with fake_engine(family="qwen") as oracle:
        oracle_session = oracle.session()
        scratch = MessageBinder(oracle_session)
        scratch.prepare(messages)
        before = len(oracle_session.transcript)
        scratch.prepare([*messages, candidate])
        rendered = len(oracle_session.transcript) - before

    assert _tool_return(result).content == TOOL_ERROR
    # Hand-checked counts: live prompt 545, wrong scratch base 358, rendered tool
    # message 20038, assistant prefix 17, and an 8192 context. The old delta
    # calculation omitted that prefix (12392 instead of the exact 12409).
    assert (base, before, rendered, ctx) == (545, 358, 20038, 8192)
    assert owner.requests == [ContextRequest(base + rendered + 17 + 1 - ctx, "tool_result")]
    assert owner.requests[0].needed_tokens == 12409


def test_reserve_without_owner_raises_context_exhausted() -> None:
    async def scenario() -> Any:
        with fake_engine(script=sequential_text("abc")) as engine:
            session = engine.session()
            model = make_model(session)
            async with model.request_stream([user()], None, ModelRequestParameters()) as stream:
                prompt_length = len(session.transcript)
                prefix = len(PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens))
                session._worker.call(
                    lambda engine, native: setattr(native, "_ctx", prompt_length + prefix + 2)
                ).result()
                with pytest.raises(ContextExhausted):
                    async for _ in stream:
                        pass
                return stream.get()

    response = asyncio.run(scenario())
    assert response.state == "incomplete"
    assert response_text(response) == "a"


def test_reserve_uses_same_owner_without_pull_lock() -> None:
    summary_tokens = 2

    async def baseline() -> tuple[tuple[int, ...], list[str]]:
        with fake_engine(script=sequential_text("abc")) as engine:
            session = engine.session()
            model = make_model(session)
            async with model.request_stream([user()], None, ModelRequestParameters()) as stream:
                async for _ in stream:
                    pass
            return session.transcript.tokens, [segment.kind for segment in session.transcript.segments]

    expected_tokens, expected_kinds = asyncio.run(baseline())

    async def scenario() -> tuple[RecordingOwner, list[ModelMessage], list[bool], Any, tuple[int, ...], list[str]]:
        with fake_engine(script=sequential_text("abc")) as engine:
            session = engine.session()
            observations: list[bool] = []
            streams: list[Any] = []

            async def on_room(messages: list[ModelMessage], request: ContextRequest) -> None:
                observations.append(streams[0]._pull_lock.locked())
                session._worker.call(lambda engine, native: setattr(native, "_ctx", 4096)).result()

            owner = RecordingOwner(
                StubBinder([], transcript_tokens=0, ctx=8192),
                room=RoomResult(made_room=True, generated_tokens=summary_tokens),
                on_room=on_room,
            )
            model = make_model(session)
            model.context_owner = owner
            messages = [user()]
            async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
                streams.append(stream)
                prompt_length = len(session.transcript)
                system_used = session.ctx_used
                prefix = len(PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens))
                session._worker.call(
                    lambda engine, native: setattr(native, "_ctx", prompt_length + prefix + 2)
                ).result()
                async for _ in stream:
                    pass
                response = stream.get()
                transcript_tokens = session.transcript.tokens
                kinds = [segment.kind for segment in session.transcript.segments]

            assert response.usage.input_tokens == prompt_length + prefix - system_used
            return owner, messages, observations, response, transcript_tokens, kinds

    owner, messages, observations, response, transcript_tokens, kinds = asyncio.run(scenario())
    assert owner.requests == [ContextRequest(1, "reserve")]
    assert owner.live_message_lists[0] is messages
    assert owner.message_lists == [messages]
    assert observations == [False]
    assert response_text(response) == "abc"
    assert response.usage.requests == 1
    assert response.usage.details == {"prefill_tokens": response.usage.input_tokens}
    assert response.usage.output_tokens == 3 + summary_tokens
    assert response.usage.total_tokens == response.usage.input_tokens + 3 + summary_tokens
    assert transcript_tokens == expected_tokens
    assert kinds == expected_kinds
    assert kinds.count("assistant_prefix") == 1
    assert kinds.count("assistant") == 1


def test_failed_reserve_room_raises_context_exhausted_and_keeps_summary_usage() -> None:
    async def scenario() -> tuple[RecordingOwner, Any]:
        with fake_engine(script=sequential_text("abc")) as engine:
            session = engine.session()
            owner = RecordingOwner(
                StubBinder([], transcript_tokens=0, ctx=8192),
                room=RoomResult(made_room=False, generated_tokens=2),
            )
            model = make_model(session)
            model.context_owner = owner
            async with model.request_stream([user()], None, ModelRequestParameters()) as stream:
                prompt_length = len(session.transcript)
                prefix = len(PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens))
                session._worker.call(
                    lambda engine, native: setattr(native, "_ctx", prompt_length + prefix + 2)
                ).result()
                with pytest.raises(ContextExhausted):
                    async for _ in stream:
                        pass
                return owner, stream.get()

    owner, response = asyncio.run(scenario())
    assert owner.requests == [ContextRequest(1, "reserve")]
    assert response.state == "incomplete"
    assert response_text(response) == "a"
    assert response.usage.output_tokens == 1 + 2


def test_close_never_awaits_the_owner_while_the_pull_lock_is_held() -> None:
    async def scenario() -> tuple[list[bool], Any]:
        with fake_engine(script=sequential_text("abc")) as engine:
            session = engine.session()
            entered = asyncio.Event()
            release = asyncio.Event()
            observations: list[bool] = []
            streams: list[Any] = []

            async def on_room(messages: list[ModelMessage], request: ContextRequest) -> None:
                observations.append(streams[0]._pull_lock.locked())
                entered.set()
                await release.wait()

            owner = RecordingOwner(
                StubBinder([], transcript_tokens=0, ctx=8192),
                on_room=on_room,
            )
            model = make_model(session)
            model.context_owner = owner
            async with model.request_stream([user()], None, ModelRequestParameters()) as stream:
                streams.append(stream)
                prompt_length = len(session.transcript)
                prefix = len(PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens))
                session._worker.call(
                    lambda engine, native: setattr(native, "_ctx", prompt_length + prefix + 2)
                ).result()

                async def consume() -> None:
                    async for _ in stream:
                        pass

                consumer = asyncio.create_task(consume())
                await asyncio.wait_for(entered.wait(), 5)
                assert observations == [False]
                await asyncio.wait_for(stream.close_stream(), 5)
                release.set()
                await asyncio.wait_for(consumer, 5)
                return observations, stream.get()

    observations, response = asyncio.run(scenario())
    assert observations == [False]
    assert response.state == "interrupted"


def test_reserve_owner_gets_the_real_request_message_list_once() -> None:
    async def scenario() -> tuple[RecordingOwner, list[ModelMessage]]:
        with fake_engine(script=sequential_text("ab")) as engine:
            session = engine.session()
            owner = RecordingOwner(
                StubBinder([], transcript_tokens=0, ctx=8192),
                room=RoomResult(made_room=False),
            )
            model = make_model(session)
            model.context_owner = owner
            messages = [user("specific request")]
            async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
                prompt_length = len(session.transcript)
                prefix = len(PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens))
                session._worker.call(
                    lambda engine, native: setattr(native, "_ctx", prompt_length + prefix + 2)
                ).result()
                with pytest.raises(ContextExhausted):
                    async for _ in stream:
                        pass
            return owner, messages

    owner, messages = asyncio.run(scenario())
    assert owner.live_message_lists == [messages]
    assert owner.message_lists == [messages]
