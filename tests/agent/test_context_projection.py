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
