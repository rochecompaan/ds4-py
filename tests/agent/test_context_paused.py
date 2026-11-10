"""Scratch counts include exactly the EOS that preparation adds to a paused response."""

import asyncio
from copy import deepcopy

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
from test_context_strategy import RecordingOwner, fake_engine, sequential_text
from test_runtime import make_model

from ds4py.capabilities.context import TOOL_RESULT_DOES_NOT_FIT, ContextRequest, RoomResult


@pytest.mark.parametrize(("family", "paused"), [("qwen", True), ("qwen", False), ("glm", True)])
def test_paused_retirement_guards_the_one_token_request_boundary(family: str, paused: bool) -> None:
    # Without a Qwen retirement EOS the prompt appears to leave the required
    # one-token reserve; actual preparation consumes that final token instead.
    async def scenario() -> None:
        with fake_engine(family=family, script=sequential_text("stop")) as engine:
            model = make_model(engine)
            session = model.session
            messages: list[ModelMessage] = [ModelRequest([UserPromptPart("go")])]
            response = await model.request(
                messages, {"stop_sequences": ["stop"]} if paused else None, ModelRequestParameters()
            )
            paused_stream = session.history.paused_stream
            assert (paused_stream is not None) is paused
            # One read_big header is 26 tokens; grouped result framing is 12
            # Qwen / 3 GLM, and the next assistant prefix is 17 Qwen / 3 GLM.
            overhead = 55 if family == "qwen" else 32
            body = "x" * (8191 - len(session.transcript) - overhead)
            pending = ModelRequest([ToolReturnPart("read_big", body, tool_call_id="one")])
            messages.extend([response, pending])
            needs_eos = family == "qwen" and paused
            before = (session.transcript.tokens, model.binder.records())
            count = model.binder.project_request(messages, bound=True)
            assert count == (8192 if needs_eos else 8191)
            assert model.binder.project_request(messages, bound=True) == count
            assert (session.transcript.tokens, model.binder.records()) == before
            assert pending.metadata is None
            assert session.history.paused_stream is paused_stream

            owner = RecordingOwner(model.binder, room=RoomResult(False))
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=messages)
            await owner.before_model_request(
                ctx,
                ModelRequestContext(
                    model=model,
                    messages=messages,
                    model_settings=None,
                    model_request_parameters=ModelRequestParameters(),
                ),
            )
            assert owner.requests == ([ContextRequest(1, "tool_result")] if needs_eos else [])
            part = pending.parts[0]
            assert isinstance(part, ToolReturnPart)
            assert part.content == (TOOL_RESULT_DOES_NOT_FIT if needs_eos else body)
            fitted = model.binder.project_request(messages, bound=True)
            async with model.request_stream(messages, None, ModelRequestParameters()):
                prefix = 17 if family == "qwen" else 3
                assert len(session.transcript) + prefix == fitted
                assert fitted + 1 <= 8192
                assert session.history.paused_stream is None
                assert sum(s.kind == "eos" for s in session.transcript.segments) == (1 if family == "qwen" else 0)
                # Retirement is already included in binding now: never add it twice.
                assert model.binder.project_request(messages, bound=True) == fitted

    asyncio.run(scenario())


@pytest.mark.parametrize("rewrite", ["unchanged", "unkeyed", "new-key", "same-key", "same-key-with-suffix"])
def test_unchanged_or_rewritten_paused_history_does_not_get_retirement_eos(rewrite: str) -> None:
    async def scenario() -> None:
        with fake_engine(family="qwen", script=sequential_text("stop")) as engine:
            model = make_model(engine)
            session = model.session
            messages: list[ModelMessage] = [ModelRequest([UserPromptPart("go")])]
            response = await model.request(messages, {"stop_sequences": ["stop"]}, ModelRequestParameters())
            messages.append(response)
            paused_stream = session.history.paused_stream
            assert paused_stream is not None
            changed = rewrite in {"new-key", "same-key", "same-key-with-suffix"}
            if changed:
                metadata = messages[0].metadata if rewrite.startswith("same-key") else None
                messages = [ModelRequest([UserPromptPart("replacement")], metadata=metadata)]
                if rewrite == "same-key-with-suffix":
                    messages.extend([response, ModelRequest([UserPromptPart("next")])])
            elif rewrite == "unkeyed":
                messages = deepcopy(messages)
                for message in messages:
                    message.metadata = {k: v for k, v in (message.metadata or {}).items() if k != "ds4.key"}
            before = (session.transcript.tokens, model.binder.records())
            count = model.binder.project_request(messages, bound=True)
            assert (session.transcript.tokens, model.binder.records()) == before
            assert session.history.paused_stream is paused_stream
            if rewrite == "unkeyed":
                assert all("ds4.key" not in (message.metadata or {}) for message in messages)
            model.binder.prepare(messages)
            assert len(session.transcript) + 17 == count
            # A rewritten response closes via the renderer, not paused retirement.
            assert [s.meta.get("source") for s in session.transcript.segments if s.kind == "eos"] == (
                ["rendered"] if rewrite == "same-key-with-suffix" else []
            )
            assert (session.history.paused_stream is None) is changed

    asyncio.run(scenario())


@pytest.mark.parametrize("paused", [False, True])
def test_after_tool_counts_paused_qwen_retirement_and_next_prefix(paused: bool) -> None:
    async def scenario() -> None:
        with fake_engine(family="qwen", script=sequential_text("stop")) as engine:
            model = make_model(engine)
            session = model.session
            messages: list[ModelMessage] = [ModelRequest([UserPromptPart("go")])]
            response = await model.request(
                messages, {"stop_sequences": ["stop"]} if paused else None, ModelRequestParameters()
            )
            messages.append(response)
            body = "x" * (8191 - len(session.transcript) - 55)
            before = (session.transcript.tokens, model.binder.records(), session.history.paused_stream)
            owner = RecordingOwner(model.binder, room=RoomResult(True, 5))
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=messages)
            result = await owner.after_tool_execute(
                ctx,
                call=ToolCallPart("read_big", {}, tool_call_id="one"),
                tool_def=ToolDefinition(name="read_big"),
                args={},
                result=body,
            )
            assert result == (TOOL_RESULT_DOES_NOT_FIT if paused else body)
            assert owner.requests == ([ContextRequest(1, "tool_result")] if paused else [])
            assert ctx.usage.output_tokens == (5 if paused else 0)
            assert (session.transcript.tokens, model.binder.records(), session.history.paused_stream) == before

    asyncio.run(scenario())
