"""Mid-response reserve recovery stays inside the RollingWindow request view."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from test_context_strategy import fake_engine, make_model, sequential_text

from ds4py.capabilities.compaction import LLMCompaction
from ds4py.capabilities.rolling_window import RollingWindow
from ds4py.engine.prompt import PromptBuilder
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY


def user(text: str, key: str) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)], metadata={DS4_KEY: key})


def test_reserve_rebinds_a_smaller_view_and_continues_the_same_response() -> None:
    async def scenario() -> None:
        with fake_engine(script=sequential_text("ab")) as engine:
            session = engine.session()
            model = make_model(session)
            owner = RollingWindow(model.binder, disabled_owner=LLMCompaction(model.binder))
            model.context_owner = owner
            canonical = [
                user("old " * 500, "old-user"),
                ModelResponse([TextPart("old answer " * 500)], metadata={DS4_KEY: "old-response"}),
                user("active request", "active-user"),
            ]
            before = deepcopy(canonical)
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=canonical)
            params = ModelRequestParameters()
            await owner.before_model_request(
                ctx,
                ModelRequestContext(
                    model=model,
                    messages=canonical,
                    model_settings=None,
                    model_request_parameters=params,
                ),
            )
            assert owner.request_messages(canonical, ctx) == canonical

            async with model.request_stream(canonical, None, params, run_context=ctx) as stream:
                prompt_length = len(session.transcript)
                prefix = len(PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens))
                engine.config = replace(engine.config, ctx=prompt_length + prefix)
                session._worker.call(
                    lambda _engine, native: setattr(native, "_ctx", prompt_length + prefix + 3)
                ).result()
                async for _ in stream:
                    pass
                response = stream.get()

            assert "".join(part.content for part in response.parts if isinstance(part, TextPart)) == "ab"
            assert len(session.transcript) < prompt_length
            assert canonical == before
            view = owner.request_messages([], ctx)
            assert view[-1] is canonical[-1]
            assert canonical[0] not in view and canonical[1] not in view
            assert view[0].metadata is not None
            assert view[0].metadata[DS4_CONTEXT_PAGING] == "notice"

    asyncio.run(scenario())
