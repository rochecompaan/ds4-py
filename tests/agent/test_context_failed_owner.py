"""Grouped and individual tool returns must honor the owner's fit decision."""

import asyncio

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
from test_context_strategy import RecordingOwner, fake_engine, make_model, prepared_tool_binder

from ds4py.capabilities.context import TOOL_RESULT_DOES_NOT_FIT, ContextRequest, RoomResult
from ds4py.errors import ContextExhausted


@pytest.mark.parametrize("entry", ["pending", "after_tool"])
@pytest.mark.parametrize("decision", ["failed_reduced", "success_reduced", "dishonest_success"])
def test_reduced_fitting_return_still_honors_failed_owner(entry: str, decision: str) -> None:
    # A failed owner can mutate history enough to fit. Measured fit alone must
    # not turn its explicit failure into a successful tool result.
    async def scenario() -> None:
        with fake_engine() as engine:
            model = make_model(engine.session())
            binder = prepared_tool_binder(model.session)
            messages = list(binder._bound_messages)
            call = ToolCallPart("read_big", {}, tool_call_id="call-1")
            definition = ToolDefinition(
                name="read_big",
                parameters_json_schema={"additionalProperties": False, "properties": {}, "type": "object"},
            )
            value = "x" * 7287

            async def reduce(history: list[ModelMessage], request: ContextRequest) -> None:
                if decision != "dishonest_success":
                    first = history[0]
                    assert isinstance(first, ModelRequest) and isinstance(first.parts[0], UserPromptPart)
                    first.parts[0].content = "g"

            owner = RecordingOwner(binder, room=RoomResult(decision != "failed_reduced"), on_room=reduce)
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=messages)
            if entry == "pending":
                pending = ModelRequest([ToolReturnPart(call.tool_name, value, tool_call_id=call.tool_call_id)])
                messages.append(pending)
                request = ModelRequestContext(
                    model=model,
                    messages=messages,
                    model_settings=None,
                    model_request_parameters=ModelRequestParameters(function_tools=[definition]),
                )
                await owner._fit_pending_results(ctx, request)
                result = pending.parts[0]
                assert isinstance(result, ToolReturnPart)
                output = result.content
                assert owner._needed_for_request(messages, request) <= 0
            else:
                output = await owner.after_tool_execute(ctx, call=call, tool_def=definition, args={}, result=value)
                candidate = ModelRequest([owner._tool_return_part(call, output)])
                assert owner._needed_for_chat([*messages, candidate]) <= 0
            assert owner.requests == [ContextRequest(1, "tool_result")]
            assert output == (value if decision == "success_reduced" else TOOL_RESULT_DOES_NOT_FIT)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("decision", "size", "expected"),
    [
        ("failed_reduced", 6000, ("small", TOOL_RESULT_DOES_NOT_FIT)),
        ("failed_oversize", 6000, ("z" * 6000, TOOL_RESULT_DOES_NOT_FIT)),
        ("failed_too_large", 9000, (TOOL_RESULT_DOES_NOT_FIT, TOOL_RESULT_DOES_NOT_FIT)),
        ("dishonest_success", 6000, ("z" * 6000, TOOL_RESULT_DOES_NOT_FIT)),
        ("success_reduced", 6000, ("small", "tiny")),
    ],
)
def test_grouped_owner_failure_rejects_only_necessary_trailing_results(
    decision: str,
    size: int,
    expected: tuple[str, str],
) -> None:
    # Force at least one trailing error for an explicit failure, not all errors
    # if one suffices. A dishonest success still needs the exact-size backstop.
    async def scenario() -> None:
        with fake_engine(family="glm") as engine:
            model = make_model(engine.session())
            definition = ToolDefinition(name="read_big")
            messages: list[ModelMessage] = [
                ModelRequest([UserPromptPart("go")]),
                ModelResponse(
                    [
                        ToolCallPart("read_big", {}, tool_call_id="one"),
                        ToolCallPart("read_big", {}, tool_call_id="two"),
                    ]
                ),
            ]
            model.binder.prepare(messages, tools=[definition])
            parts = [
                ToolReturnPart("read_big", "z" * size, tool_call_id="one"),
                ToolReturnPart("read_big", "z" * size, tool_call_id="two"),
            ]
            messages.append(ModelRequest(parts))

            async def reduce(history: list[ModelMessage], request: ContextRequest) -> None:
                if decision.endswith("reduced"):
                    parts[0].content, parts[1].content = "small", "tiny"

            owner = RecordingOwner(
                model.binder,
                room=RoomResult(decision in {"dishonest_success", "success_reduced"}),
                on_room=reduce,
            )
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=messages)
            request = ModelRequestContext(
                model=model,
                messages=messages,
                model_settings=None,
                model_request_parameters=ModelRequestParameters(function_tools=[definition]),
            )
            await owner._fit_pending_results(ctx, request)
            assert tuple(part.content for part in parts) == expected
            assert len(owner.requests) == 1 and owner.requests[0].needed_tokens > 0
            assert owner._needed_for_request(messages, request) <= 0

    asyncio.run(scenario())


@pytest.mark.parametrize("entry", ["pending", "after_tool"])
def test_error_only_replacement_that_still_overflows_raises_context_exhausted(entry: str) -> None:
    # The 8119-token base fits; the 9000-token tool return overflows by 8982.
    # Even the complete error-text projection leaves 24 tokens of overflow.
    async def scenario() -> None:
        with fake_engine() as engine:
            model = make_model(engine.session())
            owner = RecordingOwner(model.binder, room=RoomResult(False))
            call = ToolCallPart("read_big", {}, tool_call_id="one")
            history: list[ModelMessage] = [ModelRequest([UserPromptPart("x" * 7450)])]
            if entry == "pending":
                pending = ModelRequest([ToolReturnPart(call.tool_name, "x" * 9000, tool_call_id=call.tool_call_id)])
                messages = [*history, pending]
                request = ModelRequestContext(
                    model=model,
                    messages=messages,
                    model_settings=None,
                    model_request_parameters=ModelRequestParameters(),
                )
                ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=messages)
                with pytest.raises(ContextExhausted, match="pending tool errors do not fit"):
                    await owner.before_model_request(ctx, request)
                rejected = pending.parts[0]
                assert isinstance(rejected, ToolReturnPart)
                assert rejected.content == TOOL_RESULT_DOES_NOT_FIT
                error_result = ModelRequest([owner._tool_return_part(call, TOOL_RESULT_DOES_NOT_FIT)])
                assert model.binder.count_rendered_request(history) == 8119
                assert owner._needed_for_request([*history, error_result], request) == 24
            else:
                model.binder.prepare(history)
                ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=history)
                with pytest.raises(ContextExhausted, match="pending tool errors do not fit"):
                    await owner.after_tool_execute(
                        ctx,
                        call=call,
                        tool_def=ToolDefinition(name="read_big"),
                        args={},
                        result="x" * 9000,
                    )
                error_result = ModelRequest([owner._tool_return_part(call, TOOL_RESULT_DOES_NOT_FIT)])
                assert model.binder.count_bound_request(history) == 8119
                assert owner._needed_for_chat([*history, error_result]) == 24
            assert len(owner.requests) == 1 and owner.requests[0].needed_tokens == 8982

    asyncio.run(scenario())
