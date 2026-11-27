"""Mid-response reserve recovery stays inside the RollingWindow request view."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from test_context_strategy import fake_engine, make_model, sequential_text

import ds4py.capabilities.rolling_window as rolling_window_module
from ds4py.binding import RESPONSE_RESERVE
from ds4py.binding_provenance import prepare_request_view_provenance
from ds4py.capabilities.compaction import LLMCompaction
from ds4py.capabilities.rolling_window import RollingWindow
from ds4py.capabilities.system_notes import DS4SystemNotes
from ds4py.context_paging.stats import InputTokenReading, read_request_input
from ds4py.engine.prompt import PromptBuilder
from ds4py.message_metadata import (
    DS4_CONTEXT_PAGING,
    DS4_KEY,
    DS4_PAGING_INPUT_TOKENS,
    DS4_REQUEST_SYNC_COUNT,
    DS4_SOURCE,
)


def user(text: str, key: str) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)], metadata={DS4_KEY: key})


def test_recovery_render_keeps_the_open_tail_after_a_historical_response() -> None:
    async def scenario() -> None:
        with fake_engine(script=sequential_text("abcdef")) as engine:
            session = engine.session()
            model = make_model(session)
            canonical = [
                user("evicted request", "evicted-user"),
                ModelResponse([TextPart("evicted answer")], metadata={DS4_KEY: "evicted-response"}),
                user("old request", "old-user"),
                ModelResponse(
                    [TextPart("old answer")],
                    metadata={DS4_KEY: "old-response", DS4_SOURCE: "sampled"},
                ),
                user("active request", "active-user"),
            ]
            params = ModelRequestParameters()
            checked = False
            async with model.request_stream(canonical, None, params) as stream:
                async for _ in stream:
                    sampled_start = session.open_assistant_start()
                    if sampled_start is None or checked:
                        continue
                    prefix_start = next(
                        segment.start
                        for segment in session.transcript.segments
                        if segment.kind == "assistant_prefix"
                        and segment.end == sampled_start
                        and not segment.meta.get("key")
                    )
                    expected = session.transcript.tokens[prefix_start:]
                    rendered = model.binder.render_request_view(canonical[2:], params)
                    assert rendered.replacement.tokens[-len(expected) :] == expected
                    checked = True
                stream.get()
            assert checked

    asyncio.run(scenario())


@pytest.mark.parametrize("note_mode", ["hint", "restored"])
def test_reserve_rebinds_a_smaller_view_and_continues_the_same_response(note_mode: str) -> None:
    async def scenario() -> None:
        with fake_engine(script=sequential_text("ab")) as engine:
            session = engine.session()
            engine.config = replace(engine.config, ctx=32768)
            model = make_model(session)
            notes = DS4SystemNotes(session, hints=True, clock=lambda: "fixed")
            model.note_provider = notes
            owner = RollingWindow(
                model.binder,
                disabled_owner=LLMCompaction(model.binder),
                token_budget=16000,
                trim_to=15800,
            )
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
            initial_input = owner.statistics_state().input_tokens
            assert initial_input is not None

            async with model.request_stream(canonical, None, params, run_context=ctx) as stream:
                prompt_length = len(session.transcript)
                if note_mode == "hint":
                    notes.hints_enabled = False
                else:
                    model.binder.restore_notes(model.binder.export_segments())
                prefix = len(PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens))
                engine.config = replace(engine.config, ctx=prompt_length + prefix)
                session._worker.call(
                    lambda _engine, native: setattr(native, "_ctx", prompt_length + prefix + 3)
                ).result()
                async for _ in stream:
                    pass
                response = stream.get()

            assert "".join(part.content for part in response.parts if isinstance(part, TextPart)) == "ab"
            view = owner.request_messages([], ctx)
            reserve_projection = model.binder.project_request_view(
                canonical_messages=canonical,
                candidate_messages=view,
                params=params,
            )
            exact = owner.statistics_state().input_tokens
            assert exact is not None
            assert exact == reserve_projection.total_tokens
            assert exact < initial_input
            assert response.metadata is not None
            assert response.metadata[DS4_PAGING_INPUT_TOKENS] == exact
            assert response.metadata[DS4_REQUEST_SYNC_COUNT] > 1
            assert read_request_input(response) == InputTokenReading("exact", exact)
            assert len(session.transcript) < prompt_length
            assert canonical == before
            assert view[-1] is canonical[-1]
            assert canonical[0] not in view and canonical[1] not in view
            assert view[0].metadata is not None
            assert view[0].metadata[DS4_CONTEXT_PAGING] == "notice"
            note_kinds = [
                segment.kind
                for segment in model.binder.export_segments()
                if segment.kind == "system_note" or segment.kind.startswith("system_note:")
            ]
            if note_mode == "hint":
                assert note_kinds.count("system_note:hints_off") == 1
            assert notes.pending(view) == []
            model.binder.prepare(view)
            followup_note_kinds = [
                segment.kind
                for segment in model.binder.export_segments()
                if segment.kind == "system_note" or segment.kind.startswith("system_note:")
            ]
            assert followup_note_kinds == note_kinds

    asyncio.run(scenario())


def test_reserve_selection_accounts_for_sampled_response_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        with fake_engine(family="qwen", script=sequential_text("ab")) as engine:
            session = engine.session()
            engine.config = replace(engine.config, ctx=32768)
            model = make_model(session)
            owner = RollingWindow(
                model.binder,
                disabled_owner=LLMCompaction(model.binder),
                token_budget=16000,
                trim_to=16000,
            )
            model.context_owner = owner
            canonical = [
                user("first old request " * 500, "first-old-user"),
                ModelResponse([TextPart("first old answer " * 500)], metadata={DS4_KEY: "first-old-response"}),
                user("second old request " * 400, "second-old-user"),
                ModelResponse([TextPart("second old answer " * 400)], metadata={DS4_KEY: "second-old-response"}),
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
            selection_contexts: list[int] = []
            select_view = rolling_window_module.select_view

            def capture_selection_context(*args: Any, **kwargs: Any) -> Any:
                selection_contexts.append(kwargs["model_context"])
                return select_view(*args, **kwargs)

            monkeypatch.setattr(rolling_window_module, "select_view", capture_selection_context)
            make_room = owner.make_room

            async def check_recovery_context(messages: list[Any], request: Any) -> Any:
                sampled_start = session.open_assistant_start()
                assert sampled_start is not None
                sampled = len(session.transcript.tokens) - sampled_start
                result = await make_room(messages, request)
                assert selection_contexts[-1] == engine.config.ctx - sampled - RESPONSE_RESERVE
                return result

            cast("Any", owner).make_room = check_recovery_context
            async with model.request_stream(canonical, None, params, run_context=ctx) as stream:
                prompt_length = len(session.transcript)
                prefix = len(PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens))
                session._worker.call(
                    lambda _engine, native: setattr(native, "_ctx", prompt_length + prefix + 3)
                ).result()
                async for _ in stream:
                    pass
                response = stream.get()

            assert "".join(part.content for part in response.parts if isinstance(part, TextPart)) == "ab"
            assert canonical == before

    asyncio.run(scenario())


def test_strict_reserve_fit_measures_the_installed_open_transcript() -> None:
    async def scenario() -> None:
        with fake_engine(family="qwen", script=sequential_text("ab")) as engine:
            session = engine.session()
            engine.config = replace(engine.config, ctx=32768)
            model = make_model(session)
            owner = RollingWindow(
                model.binder,
                disabled_owner=LLMCompaction(model.binder),
                token_budget=16000,
                trim_to=15800,
            )
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
            make_room = owner.make_room
            recovered_lengths: list[int] = []

            async def recover_at_boundary(messages: list[Any], request: Any) -> Any:
                result = await make_room(messages, request)
                recovered_lengths.append(len(session.transcript))
                engine.config = replace(engine.config, ctx=len(session.transcript) + 1)
                return result

            cast("Any", owner).make_room = recover_at_boundary
            async with model.request_stream(canonical, None, params, run_context=ctx) as stream:
                prompt_length = len(session.transcript)
                prefix = len(PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens))
                session._worker.call(
                    lambda _engine, native: setattr(native, "_ctx", prompt_length + prefix + 3)
                ).result()
                async for _ in stream:
                    pass
                response = stream.get()

            assert "".join(part.content for part in response.parts if isinstance(part, TextPart)) == "ab"
            assert recovered_lengths
            assert engine.config.ctx == recovered_lengths[-1] + 1
            assert canonical == before

    asyncio.run(scenario())


def test_reserve_shrinks_a_remembered_view_after_output_fills_the_context() -> None:
    async def scenario() -> None:
        with fake_engine(script=sequential_text("ab" * 1000)) as engine:
            session = engine.session()
            model = make_model(session)
            owner = RollingWindow(
                model.binder,
                disabled_owner=LLMCompaction(model.binder),
                token_budget=7000,
                trim_to=6800,
            )
            model.context_owner = owner
            canonical = [
                user("evicted " * 500, "evicted-user"),
                ModelResponse([TextPart("evicted answer")], metadata={DS4_KEY: "evicted-response"}),
                user("old " * 1300, "old-user"),
                ModelResponse([TextPart("old answer")], metadata={DS4_KEY: "old-response"}),
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
            assert owner._state.selection is not None
            assert owner._state.request_input_tokens is not None
            assert 5000 < owner._state.request_input_tokens < 6800

            async with model.request_stream(canonical, None, params, run_context=ctx) as stream:
                prompt_length = len(session.transcript)
                session._worker.call(lambda _engine, native: setattr(native, "_ctx", 8192)).result()
                async for _ in stream:
                    pass
                response = stream.get()

            assert "".join(part.content for part in response.parts if isinstance(part, TextPart)) == "ab" * 1000
            assert canonical == before
            view = owner.request_messages([], ctx)
            assert all(message not in view for message in canonical[:-1])
            assert view[-1] is canonical[-1]
            assert len(session.transcript) < prompt_length + 2000
            bound_view = model.binder.request_view()
            assert len(bound_view) == len(view) + 1
            prepare_request_view_provenance(
                canonical_messages=[*canonical, response],
                request_view=bound_view,
                segments=model.binder.export_segments(),
                loaded=None,
            )

    asyncio.run(scenario())
