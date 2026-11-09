"""Request-view owners and exact native-prefix token counts stay independent."""

import asyncio
from typing import override

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, TextPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import RunContext
from test_context_strategy import RecordingOwner, fake_engine, sequential_text
from test_runtime import make_model

from ds4py._native.fake import FAKE_EOS
from ds4py.capabilities.context import ContextRequest
from ds4py.capabilities.system_notes import DS4SystemNotes
from ds4py.engine.events import Synced
from ds4py.engine.lifecycle import SessionEvent


def test_custom_reserve_owner_keeps_request_view_not_stored_history() -> None:
    # Selecting canonical messages for every owner would leak the stored prompt
    # into a deliberately independent request-only view.
    async def scenario() -> None:
        with fake_engine(script=sequential_text("abc")) as engine:
            model = make_model(engine)
            session = model.session

            class ViewOwner(RecordingOwner):
                @override
                async def before_model_request(
                    self, ctx: RunContext[object], request_context: ModelRequestContext
                ) -> ModelRequestContext:
                    request_context.messages = [ModelRequest([UserPromptPart("request view")])]
                    return request_context

            async def room(messages: list[ModelMessage], request: ContextRequest) -> None:
                session._worker.call(lambda e, n: setattr(n, "_ctx", 8192)).result()

            owner = ViewOwner(model.binder, on_room=room)
            model.context_owner = owner
            armed = False

            def force_reserve(notification: SessionEvent) -> None:
                nonlocal armed
                event = notification.payload.get("event")
                if isinstance(event, Synced) and not armed:
                    armed = True
                    session._worker.call(lambda e, n: setattr(n, "_ctx", event.pos + 2)).result()

            model.binder.prepare([ModelRequest([UserPromptPart("request view")])])
            session.on_event(force_reserve, kinds={"Synced"})
            result = await Agent(model, capabilities=[owner]).run("stored history")
            assert result.output == "abc"
            assert owner.requests == [ContextRequest(1, "reserve")]
            assert [p.content for m in owner.message_lists[0] for p in m.parts if isinstance(p, UserPromptPart)] == [
                "request view"
            ]
            assert [p.content for m in result.all_messages() for p in m.parts if isinstance(p, UserPromptPart)] == [
                "stored history"
            ]

    asyncio.run(scenario())


@pytest.mark.parametrize("pending_note", [False, True])
def test_pending_count_keeps_native_sampled_prefix_and_grouped_wrappers(pending_note: bool) -> None:
    # A scratch re-render of a sampled response is not a valid token base.
    # GLM's grouped tool message adds 3 wrapper ids, two 26-byte headers,
    # a 2-byte separator, then the 3-id assistant prefix.
    def script(prompt: object, generated: list[int]) -> int:
        text = b"<think></think>sampled"
        return text[len(generated)] + 1 if len(generated) < len(text) else FAKE_EOS

    async def scenario() -> None:
        with fake_engine(family="glm", script=script) as engine:
            model = make_model(engine)
            messages: list[ModelMessage] = [ModelRequest([UserPromptPart("go")])]
            response = await model.request(messages, None, ModelRequestParameters())
            assert response.parts == [TextPart("sampled")]
            pending = ModelRequest(
                [
                    ToolReturnPart("read_big", "a" * 10, tool_call_id="one"),
                    ToolReturnPart("read_big", "b" * 20, tool_call_id="two"),
                ]
            )
            # Native prefix: 2 BOS + 190 default tools/system + 1 user
            # marker + 2 body + 3 assistant prefix + 22 sampled; no GLM EOS.
            # The sampled empty think block is absent from the public response.
            assert len(model.session.transcript) == 220
            if pending_note:
                notes = DS4SystemNotes(model.session, hints=True)
                notes.notes(messages)
                notes.hints_enabled = False
                model.binder.note_provider = notes
            before = (model.session.transcript.tokens, model.binder.records())
            # Pending hints-off: 92 body bytes + leading newline + system marker.
            expected = 404 if pending_note else 310
            assert model.binder.count_tokens([*messages, response, pending], preserve_bound=True) == expected
            assert (model.session.transcript.tokens, model.binder.records()) == before
            model.binder.prepare([*messages, response, pending])
            assert len(model.session.transcript) == expected - 3  # Assistant prefix belongs to run_round.

    asyncio.run(scenario())
