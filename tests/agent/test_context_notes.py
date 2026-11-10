"""The complete request uses the effective model note provider without consuming it."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import HarnessDeprecationWarning
from test_context_strategy import RecordingOwner, fake_engine, sequential_text

from ds4py.agent import build_agent_components
from ds4py.binding import MessageBinder
from ds4py.capabilities.context import TOOL_RESULT_DOES_NOT_FIT, ContextRequest, RoomResult
from ds4py.capabilities.system_notes import DS4SystemNotes
from ds4py.model import DS4Model


@pytest.mark.parametrize("configuration", ["constructor", "assigned", "binder", "composition", "reassigned", "cleared"])
def test_effective_pending_notes_guard_the_complete_request_boundary(
    configuration: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ignoring model-only notes admits 8191 tokens + reserve, but preparation
    # adds a 94-token hints-off note and overflows the 8192-token context.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    async def scenario() -> None:
        with fake_engine(family="glm", script=sequential_text("sampled")) as engine:
            session = engine.session()
            notes = DS4SystemNotes(session, hints=True, clock=lambda: "fixed")
            if configuration == "binder":
                binder = MessageBinder(session)
                binder.note_provider = notes
                model = DS4Model(session, binder=binder, sampling=engine.sampling_defaults)
            elif configuration == "constructor":
                model = DS4Model(session, sampling=engine.sampling_defaults, note_provider=notes)
            else:
                model = DS4Model(session, sampling=engine.sampling_defaults)
                if configuration in {"composition", "reassigned", "cleared"}:
                    with pytest.warns(HarnessDeprecationWarning, match="deprecated and ignored: commands/paths start"):
                        build_agent_components(model, tmp_path)
                    assert model.note_provider is not None
                    notes = model.note_provider
                    notes.hints_enabled = True
                else:
                    model.note_provider = notes
            # Establish a provider that already sent the initial date/hints.
            messages: list[ModelMessage] = [ModelRequest([UserPromptPart("go")])]
            notes.notes(messages)
            response = await model.request(messages, None, ModelRequestParameters())
            assert len(session.transcript) == 205
            if configuration == "reassigned":
                notes = DS4SystemNotes(session, hints=True, clock=lambda: "replacement")
                notes.notes(messages)
                model.note_provider = notes
            notes.hints_enabled = False
            if configuration == "cleared":
                model.note_provider = None
            pending = ModelRequest([ToolReturnPart("read_big", "x" * 7954, tool_call_id="one")])
            messages.extend([response, pending])
            pending_notes = notes.pending(messages)
            before = (session.transcript.tokens, model.binder.records())
            count = model.binder.count_tokens(messages, preserve_bound=True)
            assert count == (8191 if configuration == "cleared" else 8285)
            assert notes.pending(messages) == pending_notes
            assert (session.transcript.tokens, model.binder.records()) == before

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
            assert owner.requests == ([] if configuration == "cleared" else [ContextRequest(94, "tool_result")])
            part = pending.parts[0]
            assert isinstance(part, ToolReturnPart)
            assert part.content == ("x" * 7954 if configuration == "cleared" else TOOL_RESULT_DOES_NOT_FIT)
            assert notes.pending(messages) == pending_notes
            fitted = model.binder.count_tokens(messages, preserve_bound=True)
            async with model.request_stream(messages, None, ModelRequestParameters()):
                assert len(session.transcript) + 3 == fitted  # GLM assistant prefix.
                assert fitted + 1 <= 8192
            assert notes.pending(messages) == (pending_notes if configuration == "cleared" else [])

    asyncio.run(scenario())
