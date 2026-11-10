"""Exact prospective and post-owner fit, including genuinely fresh requests."""

import asyncio

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
from test_binding import fake_engine, tool, user
from test_context_strategy import RecordingOwner

from ds4py.capabilities.context import TOOL_RESULT_DOES_NOT_FIT, ContextRequest, RoomResult
from ds4py.capabilities.system_notes import DS4SystemNotes
from ds4py.errors import EngineError
from ds4py.model import DS4Model


@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("made_room", [False, True])
def test_after_tool_verifies_exact_fit_even_when_owner_claims_success(delta: int, made_room: bool) -> None:
    async def scenario() -> None:
        with fake_engine() as engine:
            model = DS4Model(engine.session(), sampling=engine.sampling_defaults)
            messages: list[ModelMessage] = [user("go")]
            model.binder.prepare(messages)
            # Deepseek: preamble 666, user 3, tool framing 50, assistant prefix 2.
            content = "x" * (7470 + delta)
            owner = RecordingOwner(model.binder, room=RoomResult(made_room, 7))
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=messages)
            result = await owner.after_tool_execute(
                ctx,
                call=ToolCallPart("read", {}, tool_call_id="one"),
                tool_def=ToolDefinition(name="read"),
                args={},
                result=content,
            )
            assert result == (TOOL_RESULT_DOES_NOT_FIT if delta > 0 else content)
            assert owner.requests == ([ContextRequest(1, "tool_result")] if delta > 0 else [])
            assert ctx.usage.output_tokens == (7 if delta > 0 else 0)

    asyncio.run(scenario())


def test_after_tool_without_prepared_chat_raises_instead_of_guessing_config() -> None:
    async def scenario() -> None:
        with fake_engine() as engine:
            model = DS4Model(engine.session(), sampling=engine.sampling_defaults)
            owner = RecordingOwner(model.binder)
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=[user("go")])
            with pytest.raises(EngineError, match="initialized chat"):
                await owner.after_tool_execute(
                    ctx,
                    call=ToolCallPart("read", {}),
                    tool_def=ToolDefinition(name="read"),
                    args={},
                    result="x",
                )

    asyncio.run(scenario())


def test_fresh_before_request_uses_effective_tools_instructions_and_notes() -> None:
    async def scenario() -> None:
        with fake_engine() as engine:
            session = engine.session()
            model = DS4Model(session, sampling=engine.sampling_defaults)
            notes = DS4SystemNotes(session, hints=True, clock=lambda: "fixed")
            model.note_provider = notes
            messages: list[ModelMessage] = [user("go"), ModelRequest([ToolReturnPart("read", "x" * 7000)])]
            owner = RecordingOwner(model.binder, room=RoomResult(False))
            params = ModelRequestParameters(function_tools=[tool()])
            request = ModelRequestContext(
                model=model,
                messages=messages,
                model_settings=None,
                model_request_parameters=params,
            )
            first = messages[0]
            assert isinstance(first, ModelRequest)
            first.instructions = "rules"
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=messages)
            await owner.before_model_request(ctx, request)
            # 769 configured system + 764 notes + 3 user + 7050 tool + 2 prefix.
            assert owner.requests == [ContextRequest(397, "tool_result")]
            part = messages[-1].parts[0]
            assert isinstance(part, ToolReturnPart)
            assert part.content == TOOL_RESULT_DOES_NOT_FIT
            assert not session.transcript.chat
            assert not notes._date_emitted

    asyncio.run(scenario())


@pytest.mark.parametrize(("configuration", "overflow"), [("instructions", 1532), ("tools", 1614)])
@pytest.mark.parametrize("owner_reduces_result", [False, True])
def test_initialized_chat_fit_uses_changed_config_before_and_after_owner(
    configuration: str, overflow: int, owner_reduces_result: bool
) -> None:
    async def scenario() -> None:
        with fake_engine() as engine:
            session = engine.session()
            model = DS4Model(session, sampling=engine.sampling_defaults)
            first = user("go")
            messages: list[ModelMessage] = [first]
            model.binder.prepare(messages)
            old_tokens = session.transcript.tokens
            old_records = model.binder.records()
            instructions = None
            definitions = []
            if configuration == "instructions":
                first.instructions = instructions = "x" * 2000
            else:
                definition = tool()
                definition.description = "x" * 2000
                definitions = [definition]
            pending = ToolReturnPart("read", "x" * 7000)
            messages.append(ModelRequest([pending]))
            params = ModelRequestParameters(function_tools=definitions)
            request = ModelRequestContext(
                model=model, messages=messages, model_settings=None, model_request_parameters=params
            )

            async def make_room(live: list[ModelMessage], request: ContextRequest) -> None:
                if owner_reduces_result:
                    pending_request = live[-1]
                    assert isinstance(pending_request, ModelRequest)
                    pending_request.parts = [ToolReturnPart("read", "fits")]

            owner = RecordingOwner(model.binder, room=RoomResult(True, 7), on_room=make_room)
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=messages)
            # The old 7721-token preamble projection fits; either new configuration
            # overflows. Real configured preparation is 9721/9803 + 2 prefix.
            assert model.binder.count_bound_request(messages) == 7721
            assert model.binder.count_rendered_request(messages, tools=definitions, instructions=instructions) == (
                9723 if configuration == "instructions" else 9805
            )
            await owner.before_model_request(ctx, request)
            assert owner.requests == [ContextRequest(overflow, "tool_result")]
            assert ctx.usage.output_tokens == 7
            result_part = messages[-1].parts[0]
            assert isinstance(result_part, ToolReturnPart)
            assert result_part.content == ("fits" if owner_reduces_result else TOOL_RESULT_DOES_NOT_FIT)
            assert session.transcript.tokens == old_tokens
            assert model.binder.records() == old_records
            model.binder.prepare(messages, tools=definitions, instructions=instructions)
            assert len(session.transcript) + 2 + 1 <= 8192

    asyncio.run(scenario())
