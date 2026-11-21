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
from pydantic_ai.models import ModelRequestParameters
from test_binding import fake_engine
from test_restore import fake_engine as runtime_engine
from test_restore import fixed_clock, make_model, project_dir, scripted_rounds

from ds4py.binding_provenance import prepare_request_view_provenance
from ds4py.capabilities.context import ContextComposition
from ds4py.context_paging.history import derive_frontier
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


@pytest.mark.parametrize("disabled", [True, False], ids=["disabled-rolling", "compaction-owner"])
def test_ineligible_view_restores_render_full_canonical_history(tmp_path: Path, disabled: bool) -> None:
    async def scenario() -> None:
        record, store, canonical, _tokens = await save_paged(tmp_path)
        context = ContextComposition.rolling(enabled=False) if disabled else None
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
