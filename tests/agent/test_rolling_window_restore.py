"""Derive and replay RollingWindow frontiers from validated saved views."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from test_binding import fake_engine
from test_restore import fake_engine as runtime_engine
from test_restore import fixed_clock, make_model, project_dir, scripted_rounds

from ds4py.binding_provenance import prepare_request_view_provenance
from ds4py.capabilities.context import ContextComposition
from ds4py.capabilities.rolling_window import RollingWindow
from ds4py.context_paging.history import HistoryProjectionError, derive_frontier, project_history
from ds4py.context_paging.public_history import PublicModelTurn, project_public_history
from ds4py.errors import StoreError
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY, DS4_VIEW_OF
from ds4py.runtime import AgentRuntime
from ds4py.store import SessionStore


def user(text: str, key: str) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)], metadata={DS4_KEY: key})


def response(text: str, key: str, *calls: str) -> ModelResponse:
    parts: list[Any] = [TextPart(text)]
    parts.extend(ToolCallPart("read", {}, tool_call_id=call) for call in calls)
    return ModelResponse(parts, metadata={DS4_KEY: key})


def results(key: str, *calls: str) -> ModelRequest:
    return ModelRequest(
        [ToolReturnPart("read", "result", tool_call_id=call) for call in calls],
        metadata={DS4_KEY: key},
    )


def prepared(canonical: list[ModelMessage], view: list[ModelMessage]) -> Any:
    with fake_engine() as engine:
        from ds4py.binding import MessageBinder

        binder = MessageBinder(engine.session())
        binder.prepare_model_request(view, ModelRequestParameters())
        return prepare_request_view_provenance(
            canonical_messages=canonical,
            request_view=view,
            segments=binder.export_segments(),
            loaded=None,
        )


def test_full_equal_view_has_an_empty_frontier() -> None:
    canonical: list[ModelMessage] = [user("first", "u1"), response("answer", "r1")]

    assert derive_frontier(canonical, prepared(canonical, canonical)) is None


def test_whole_turn_omissions_derive_the_last_completed_turn() -> None:
    canonical: list[ModelMessage] = [
        user("first", "u1"),
        response("one", "r1"),
        user("second", "u2"),
        response("two", "r2"),
        user("active", "u3"),
    ]
    view = [canonical[-1]]

    frontier = derive_frontier(canonical, prepared(canonical, view))

    assert frontier is not None
    assert frontier.kind == "completedTurn"
    assert frontier.last_evicted.history_id == "r2"


@pytest.mark.parametrize("continuation", ["switch", "fork"])
def test_saved_incomplete_exchange_continues_as_interrupted(tmp_path: Path, continuation: str) -> None:
    async def scenario() -> None:
        with runtime_engine() as engine:
            store = SessionStore(tmp_path / "home")
            runtime = AgentRuntime(
                make_model(engine),
                project_dir(tmp_path),
                store=store,
                clock=fixed_clock,
                context=ContextComposition.rolling(token_budget=16000),
            )
            try:
                runtime._messages = [response("", "interrupted-turn", "unfinished-call")]
                runtime.model.binder.prepare_model_request(runtime._messages, ModelRequestParameters())
                record = await runtime.save()
                await runtime.new()
                await runtime.switch(record.session_id)
                if continuation == "fork":
                    await runtime.fork()
                runtime._messages.append(user("next request", "next-user"))
                owner = runtime.model.context_owner
                assert isinstance(owner, RollingWindow)
                ctx = RunContext(deps=None, model=runtime.model, usage=RunUsage(), messages=runtime._messages)
                params = ModelRequestParameters()
                await owner.before_model_request(
                    ctx,
                    ModelRequestContext(
                        model=runtime.model,
                        messages=runtime._messages,
                        model_settings=None,
                        model_request_parameters=params,
                    ),
                )
                projected = project_history(runtime.messages, restore_ids=owner._state.restore_ids)
                turns = [item for item in project_public_history(projected) if isinstance(item, PublicModelTurn)]
                assert [turn.history_id for turn in turns] == ["interrupted-turn"]
                assert turns[0].interrupted_by_restore
                assert owner._state.restore_ids == {"interrupted-turn"}

                runtime._messages.extend(
                    [response("", "new-incomplete-turn", "new-call"), user("later request", "later-user")]
                )
                with pytest.raises(HistoryProjectionError, match="INCOMPLETE_TOOL_RESULTS"):
                    await owner.before_model_request(
                        ctx,
                        ModelRequestContext(
                            model=runtime.model,
                            messages=runtime._messages,
                            model_settings=None,
                            model_request_parameters=params,
                        ),
                    )
            finally:
                await runtime.close()

    asyncio.run(scenario())


def test_active_partial_turn_derives_user_anchor_and_last_evicted_exchange() -> None:
    canonical: list[ModelMessage] = [
        user("active", "u1"),
        response("first", "r1", "call-1"),
        results("tool-1", "call-1"),
        response("second", "r2", "call-2"),
        results("tool-2", "call-2"),
    ]
    view = [canonical[0], *canonical[3:]]

    frontier = derive_frontier(canonical, prepared(canonical, view))

    assert frontier is not None
    assert frontier.kind == "partialTurn"
    assert frontier.last_evicted.history_id == "r1"
    assert frontier.last_evicted.tool_call_id == "call-1"
    assert frontier.user_history_id == "u1"


async def save_paged(tmp_path: Path) -> tuple[Any, SessionStore, list[ModelMessage], tuple[int, ...]]:
    with runtime_engine(script=scripted_rounds("done")) as engine:
        store = SessionStore(tmp_path / "home")
        runtime = AgentRuntime(
            make_model(engine),
            project_dir(tmp_path),
            store=store,
            clock=fixed_clock,
            context=ContextComposition.rolling(token_budget=16000, trim_to=15800),
        )
        runtime._messages = [
            user("old " * 20000, "old-user"),
            response("old answer " * 20000, "old-response"),
        ]
        try:
            await runtime.run("current request")
            record = await runtime.save()
            return record, store, list(runtime.messages), runtime.model.session.transcript.tokens
        finally:
            await runtime.close()


async def save_active_partial(tmp_path: Path) -> tuple[Any, SessionStore, list[ModelMessage], tuple[int, ...]]:
    with runtime_engine(script=scripted_rounds("later answer")) as engine:
        store = SessionStore(tmp_path / "home")
        runtime = AgentRuntime(
            make_model(engine),
            project_dir(tmp_path),
            store=store,
            clock=fixed_clock,
            context=ContextComposition.rolling(token_budget=16000, trim_to=15800),
        )
        runtime._messages = [
            user("old " * 20000, "old-user"),
            response("old answer " * 20000, "old-response", "old-call"),
            results("old-results", "old-call"),
            *[
                message
                for index in range(3)
                for message in (
                    user(f"notice request {index}", f"notice-user-{index}"),
                    response(f"notice reply {index}", f"notice-response-{index}"),
                )
            ],
            user("active request", "active-user"),
            response("", "active-response", "active-call"),
        ]
        owner = runtime.model.context_owner
        assert isinstance(owner, RollingWindow)
        params = ModelRequestParameters()
        ctx = RunContext(deps=None, model=runtime.model, usage=RunUsage(), messages=runtime._messages)
        try:
            await owner.before_model_request(
                ctx,
                ModelRequestContext(
                    model=runtime.model,
                    messages=runtime._messages,
                    model_settings=None,
                    model_request_parameters=params,
                ),
            )
            async with runtime.model._open_stream(runtime._messages, None, params, run_context=ctx):
                pass
            record = await runtime.save()
            return record, store, list(runtime.messages), runtime.model.session.transcript.tokens
        finally:
            await runtime.close()


@pytest.mark.parametrize("continuation", ["switch", "fork"])
def test_active_partial_restore_completes_later_and_extends_loaded_view(tmp_path: Path, continuation: str) -> None:
    async def scenario() -> None:
        record, store, canonical, tokens = await save_active_partial(tmp_path)
        assert record.request_view is not None
        assert len(record.request_view) < len(canonical)
        with runtime_engine(script=scripted_rounds("later answer")) as engine:
            runtime = AgentRuntime(
                make_model(engine),
                tmp_path / "project",
                store=store,
                clock=fixed_clock,
                context=ContextComposition.rolling(token_budget=16000, trim_to=15800),
            )
            try:
                await runtime.switch(record.session_id)
                if continuation == "fork":
                    assert await runtime.fork() != record.session_id
                assert list(runtime.messages) == canonical
                assert runtime.model.session.transcript.tokens == tokens
                assert runtime.model.binder.request_view() == record.request_view
                immediate = await runtime.save()
                assert immediate.request_view == record.request_view
                assert immediate.tokens_sha256 == record.tokens_sha256

                retained = list(runtime.messages)
                runtime._messages.extend(
                    [results("active-results", "active-call"), user("later request", "later-user")]
                )
                synced = []
                runtime.model.session.on_event(
                    lambda event: synced.append(event.payload["event"]),
                    kinds={"Synced"},
                )
                owner = runtime.model.context_owner
                assert isinstance(owner, RollingWindow)
                params = ModelRequestParameters()
                ctx = RunContext(deps=None, model=runtime.model, usage=RunUsage(), messages=runtime._messages)
                await owner.before_model_request(
                    ctx,
                    ModelRequestContext(
                        model=runtime.model,
                        messages=runtime._messages,
                        model_settings=None,
                        model_request_parameters=params,
                    ),
                )
                selected = owner.request_messages(runtime._messages, ctx)
                notice = selected[0]
                assert isinstance(notice, ModelRequest)
                assert (notice.metadata or {}).get(DS4_CONTEXT_PAGING) == "notice"
                notice_part = notice.parts[0]
                assert isinstance(notice_part, UserPromptPart)
                notice_text = notice_part.content
                assert "old-response" in notice_text
                assert "old-call" in notice_text
                async with runtime.model._open_stream(runtime._messages, None, params, run_context=ctx) as stream:
                    async for _ in stream:
                        pass
                    runtime._messages.append(stream.get())

                assert list(runtime.messages)[: len(retained)] == retained
                assert synced[-1].plan.kind == "extend"
                reused = synced[-1].plan.reused
                assert reused > 0
                current_tokens = runtime.model.session.transcript.tokens
                assert current_tokens[:reused] == tokens[:reused]
            finally:
                await runtime.close()

    asyncio.run(scenario())


def test_exact_view_switch_and_second_save_preserve_tokens_and_canonical_history(tmp_path: Path) -> None:
    async def restore_exact(
        record: Any, store: SessionStore, canonical: list[ModelMessage], tokens: tuple[int, ...]
    ) -> None:
        with runtime_engine() as engine:
            runtime = AgentRuntime(
                make_model(engine),
                tmp_path / "project",
                store=store,
                clock=fixed_clock,
                context=ContextComposition.rolling(token_budget=16000, trim_to=15800),
            )
            try:
                await runtime.switch(record.session_id)
                assert runtime.model.session.transcript.tokens == tokens
                assert list(runtime.messages) == canonical
                assert runtime.model.binder.request_view() == record.request_view
                second = await runtime.save()
                assert second.tokens_sha256 == record.tokens_sha256
                assert second.request_view == record.request_view
            finally:
                await runtime.close()

    record, store, canonical, tokens = asyncio.run(save_paged(tmp_path))
    assert record.version == 2
    assert record.request_view is not None
    assert len(record.request_view) < len(canonical)
    assert record.request_view[0].metadata is not None
    assert record.request_view[0].metadata[DS4_CONTEXT_PAGING] == "notice"
    asyncio.run(restore_exact(record, store, canonical, tokens))


@pytest.mark.parametrize("mode", ["disabled-rolling", "compaction-owner", "missing-payload"], ids=lambda mode: mode)
def test_ineligible_view_restores_render_full_canonical_history(tmp_path: Path, mode: str) -> None:
    async def scenario() -> None:
        record, store, canonical, _tokens = await save_paged(tmp_path)
        context = {
            "disabled-rolling": ContextComposition.rolling(enabled=False),
            "compaction-owner": None,
            "missing-payload": ContextComposition.rolling(token_budget=16000, trim_to=15800),
        }[mode]
        if mode == "missing-payload":
            (store.session_dir(record.session_id) / "session.kv").unlink()
        with runtime_engine() as engine:
            runtime = AgentRuntime(
                make_model(engine),
                tmp_path / "project",
                store=store,
                clock=fixed_clock,
                context=context,
            )
            try:
                await runtime.switch(record.session_id)
                assert list(runtime.messages) == canonical
                assert runtime.model.session.transcript.tokens == ()
                assert runtime.model.binder.request_view() == ()
                assert all((message.metadata or {}).get(DS4_CONTEXT_PAGING) is None for message in runtime.messages)
            finally:
                await runtime.close()

    asyncio.run(scenario())


def test_rejected_exact_view_falls_back_without_frontier_or_view_notes(tmp_path: Path) -> None:
    async def scenario() -> None:
        record, store, canonical, _tokens = await save_paged(tmp_path)
        encoded = record.to_dict()
        encoded["tokens_sha256"] = "0" * 64
        (store.session_dir(record.session_id) / "save.json").write_text(json.dumps(encoded))

        with runtime_engine() as engine:
            runtime = AgentRuntime(
                make_model(engine),
                tmp_path / "project",
                store=store,
                clock=fixed_clock,
                context=ContextComposition.rolling(token_budget=16000),
            )
            try:
                await runtime.switch(record.session_id)
                owner = runtime.model.context_owner
                assert isinstance(owner, RollingWindow)
                assert list(runtime.messages) == canonical
                assert runtime.model.session.transcript.tokens == ()
                assert runtime.model.binder.request_view() == ()
                assert owner._state.selection is None
                assert not any(
                    segment.kind == "system_note" or segment.kind.startswith("system_note:")
                    for segment in runtime.model.binder.export_segments()
                )
            finally:
                await runtime.close()

    asyncio.run(scenario())


def test_malformed_view_is_atomic_for_exact_restore_and_ignored_for_render(tmp_path: Path) -> None:
    async def scenario() -> None:
        record, store, canonical, _tokens = await save_paged(tmp_path)
        encoded = record.to_dict()
        encoded["request_view"][0]["metadata"][DS4_VIEW_OF] = "missing-canonical-target"
        path = store.session_dir(record.session_id) / "save.json"
        path.write_text(json.dumps(encoded))

        with runtime_engine(script=scripted_rounds("target")) as engine:
            runtime = AgentRuntime(
                make_model(engine),
                tmp_path / "project",
                store=store,
                clock=fixed_clock,
                context=ContextComposition.rolling(token_budget=16000),
            )
            try:
                await runtime.run("keep target")
                before_messages = list(runtime.messages)
                before_tokens = runtime.model.session.transcript.tokens
                before_session = runtime.session_id
                with pytest.raises(StoreError, match=record.run_id):
                    await runtime.switch(record.session_id)
                assert list(runtime.messages) == before_messages
                assert runtime.model.session.transcript.tokens == before_tokens
                assert runtime.session_id == before_session
            finally:
                await runtime.close()

        with runtime_engine() as engine:
            runtime = AgentRuntime(
                make_model(engine),
                tmp_path / "project",
                store=store,
                clock=fixed_clock,
                context=ContextComposition.rolling(enabled=False),
            )
            try:
                await runtime.switch(record.session_id)
                assert list(runtime.messages) == canonical
                assert runtime.model.session.transcript.tokens == ()
            finally:
                await runtime.close()

    asyncio.run(scenario())
