"""The view-only RollingWindow owner and its composition boundary."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, override

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from test_agent_factory import MockSession
from test_context_strategy import RecordingOwner, StubBinder, fake_engine, make_model, sequential_text

from ds4py.agent import build_agent_components
from ds4py.capabilities.compaction import LLMCompaction
from ds4py.capabilities.context import ContextComposition
from ds4py.capabilities.rolling_window import RollingWindow
from ds4py.context_paging.notice import NOTICE_VIEW_KEY
from ds4py.context_paging.selection import SelectionResult, SelectionState
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY


def user(text: str, key: str) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)], metadata={DS4_KEY: key})


def request_context(model: Any, messages: list[Any]) -> ModelRequestContext:
    return ModelRequestContext(
        model=model,
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


def test_rolling_composition_constructs_one_owner_with_the_manual_compactor(tmp_path: Path) -> None:
    model: Any = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("done")]))
    model.session = MockSession()
    model.binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)

    components = build_agent_components(model, tmp_path, context=ContextComposition.rolling(enabled=False))
    owners = [component for component in components.components if isinstance(component, RollingWindow)]

    assert len(owners) == 1
    assert model.context_owner is owners[0]
    assert owners[0].disabled_owner is components.compactor
    assert set(owners[0].get_toolset().tools) == {
        "search_history",
        "browse_history",
        "load_history",
        "read_context_output",
    }


def test_enabled_and_disabled_owners_always_expose_all_recovery_tools() -> None:
    for enabled in (True, False):
        binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
        owner = RollingWindow(cast("Any", binder), disabled_owner=LLMCompaction(cast("Any", binder)), enabled=enabled)
        assert set(owner.get_toolset().tools) == {
            "search_history",
            "browse_history",
            "load_history",
            "read_context_output",
        }


def test_selected_view_is_stored_without_changing_canonical_history(monkeypatch: Any) -> None:
    canonical: list[ModelMessage] = [
        user("old", "old-key"),
        ModelResponse([TextPart("answer")], metadata={DS4_KEY: "answer-key"}),
    ]
    normalized: list[ModelMessage] = [user("normalized", "normalized-key")]
    selected = (
        ModelRequest(
            [UserPromptPart("notice")],
            metadata={DS4_KEY: "notice-key", DS4_CONTEXT_PAGING: "notice"},
        ),
        canonical[0],
    )
    remembered = SelectionState(
        frontier=cast("Any", SimpleNamespace()),
        notice=None,
    )
    result = SelectionResult(
        candidate_messages=selected,
        measurement=cast("Any", SimpleNamespace(total_tokens=1)),
        effective_budget=10,
        effective_target=5,
        mode="paged",
        state=remembered,
        notice=None,
    )
    monkeypatch.setattr("ds4py.capabilities.rolling_window.select_view", lambda *args, **kwargs: result)
    binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
    session: Any = MockSession()
    session.transcript = binder.session.transcript
    session.engine = binder.session.engine
    binder.session = session
    owner = RollingWindow(cast("Any", binder), disabled_owner=LLMCompaction(cast("Any", binder)), token_budget=10)
    model = SimpleNamespace(binder=binder)
    ctx = RunContext(deps=None, model=cast("Any", model), usage=RunUsage(), messages=canonical)
    before = deepcopy(canonical)

    returned = asyncio.run(owner.before_model_request(ctx, request_context(model, list(canonical))))

    assert returned.messages == canonical == before
    assert owner.request_messages(normalized, ctx) == list(selected)
    assert owner.reserve_messages(normalized, ctx) == list(selected)
    assert canonical[0].metadata == {DS4_KEY: "old-key"}
    assert selected[0].metadata == {DS4_KEY: "notice-key", DS4_CONTEXT_PAGING: "notice"}


def test_failed_selection_does_not_publish_a_new_request_view(monkeypatch: Any) -> None:
    from ds4py.errors import ContextSelectionError

    canonical: list[ModelMessage] = [user("old", "old-key")]
    selected = (canonical[0],)
    result = SelectionResult(
        candidate_messages=selected,
        measurement=cast("Any", SimpleNamespace(total_tokens=1)),
        effective_budget=10,
        effective_target=5,
        mode="within-budget",
        state=None,
        notice=None,
    )
    monkeypatch.setattr("ds4py.capabilities.rolling_window.select_view", lambda *args, **kwargs: result)
    binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
    session: Any = MockSession()
    session.transcript = binder.session.transcript
    session.engine = binder.session.engine
    binder.session = session
    owner = RollingWindow(cast("Any", binder), disabled_owner=LLMCompaction(cast("Any", binder)), token_budget=10)
    model = SimpleNamespace(binder=binder)
    ctx = RunContext(deps=None, model=cast("Any", model), usage=RunUsage(), messages=canonical)
    asyncio.run(owner.before_model_request(ctx, request_context(model, list(canonical))))
    monkeypatch.setattr(
        "ds4py.capabilities.rolling_window.select_view",
        lambda *args, **kwargs: (_ for _ in ()).throw(ContextSelectionError("FAIL", "failed")),
    )

    import pytest

    with pytest.raises(ContextSelectionError, match="failed"):
        asyncio.run(owner.before_model_request(ctx, request_context(model, list(canonical))))

    assert owner.request_messages([], ctx) == list(selected)


def test_real_selection_retains_canonical_history_and_returns_the_measured_view() -> None:
    async def scenario() -> None:
        with fake_engine() as engine:
            model = make_model(engine.session())
            owner = RollingWindow(
                model.binder,
                disabled_owner=LLMCompaction(model.binder),
                token_budget=1200,
                trim_to=600,
            )
            canonical: list[ModelMessage] = [
                user("old " * 1000, "old-user"),
                ModelResponse([TextPart("answer " * 1000)], metadata={DS4_KEY: "old-answer"}),
                user("current request", "current-user"),
            ]
            before = deepcopy(canonical)
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=canonical)

            await owner.before_model_request(ctx, request_context(model, list(canonical)))
            normalized: list[ModelMessage] = [user("harness normalized", "normalized")]
            actual = owner.request_messages(normalized, ctx)

            assert canonical == before
            assert actual[-1] is canonical[-1]
            assert canonical[0] not in actual
            assert canonical[1] not in actual
            assert isinstance(actual[0], ModelRequest)
            assert isinstance(actual[0].parts[0], SystemPromptPart)
            assert actual[0].metadata == {
                DS4_KEY: "context-paging-notice",
                DS4_CONTEXT_PAGING: "notice",
            }
            assert all((message.metadata or {}).get(DS4_CONTEXT_PAGING) is None for message in canonical)

            tools = await owner.get_toolset().get_tools(ctx)
            reply = await owner.get_toolset().call_tool(
                "search_history", {"query": "old old"}, ctx, tools["search_history"]
            )
            payload = json.loads(reply.return_value["content"][0]["text"])
            assert payload["references"][0]["historyId"] == "old-user"

    asyncio.run(scenario())


def test_agent_adopts_canonical_history_and_response_not_view_only_notices() -> None:
    async def scenario() -> None:
        from pydantic_ai import Agent

        with fake_engine(script=sequential_text("done")) as engine:
            model = make_model(engine.session())
            owner = RollingWindow(
                model.binder,
                disabled_owner=LLMCompaction(model.binder),
                token_budget=6000,
                trim_to=5800,
            )
            model.context_owner = owner
            agent: Agent[Any, str] = Agent(model, capabilities=[owner])
            history = [
                ModelRequest([UserPromptPart("old " * 1000)]),
                ModelResponse([TextPart("old answer " * 1000)]),
            ]

            result = await agent.run("current request", message_history=history)
            messages = result.all_messages()

            assert result.output == "done"
            assert len(messages) == 4
            assert cast("UserPromptPart", messages[0].parts[0]).content == "old " * 1000
            assert cast("UserPromptPart", messages[2].parts[0]).content == "current request"
            assert all((message.metadata or {}).get(DS4_CONTEXT_PAGING) is None for message in messages)
            assert any((message.metadata or {}).get(DS4_KEY) for message in messages)

    asyncio.run(scenario())


def test_open_stream_binds_the_ordinary_view_not_the_recovery_view() -> None:
    async def scenario() -> None:
        with fake_engine(script=sequential_text("ok")) as engine:
            model = make_model(engine.session())
            selected: list[ModelMessage] = [user("selected view", "selected-key")]
            normalized: list[ModelMessage] = [user("normalized view", "normalized-key")]

            class ViewOwner(RecordingOwner):
                @override
                def request_messages(self, request_messages: list[Any], run_context: Any) -> list[Any]:
                    assert request_messages == normalized
                    return selected

                @override
                def reserve_messages(self, request_messages: list[Any], run_context: Any) -> list[Any]:
                    raise AssertionError("ordinary requests must not use the recovery hook")

            owner = ViewOwner(model.binder)
            model.context_owner = owner
            bound: list[list[Any]] = []
            prepare = model.binder.prepare_model_request

            def capture(messages: list[Any], *args: Any, **kwargs: Any) -> list[Any]:
                bound.append(messages)
                return prepare(messages, *args, **kwargs)

            cast("Any", model.binder).prepare_model_request = capture
            async with model.request_stream(normalized, None, ModelRequestParameters()) as stream:
                async for _ in stream:
                    pass

            assert bound == [selected]
            assert bound[0][0].metadata == {DS4_KEY: "selected-key"}
            assert cast("UserPromptPart", normalized[0].parts[0]).content == "normalized view"

    asyncio.run(scenario())


@pytest.mark.parametrize("resident_first", [False, True])
def test_real_rolling_binds_selected_view_with_distinct_normalized_input(resident_first: bool) -> None:
    async def scenario() -> None:
        with fake_engine(script=sequential_text("ok")) as engine:
            model = make_model(engine.session())
            owner = RollingWindow(
                model.binder,
                disabled_owner=LLMCompaction(model.binder),
                token_budget=16000,
                trim_to=15800,
            )
            model.context_owner = owner
            resident = ModelRequest([SystemPromptPart("resident")], metadata={DS4_KEY: "resident"})
            canonical: list[ModelMessage] = [
                *([resident] if resident_first else []),
                user("old " * 20000, "old-user"),
                ModelResponse([TextPart("old answer " * 20000)], metadata={DS4_KEY: "old-response"}),
                user("canonical current", "current"),
            ]
            before = deepcopy(canonical)
            normalized = deepcopy(canonical)
            normalized_request = normalized[-1]
            assert isinstance(normalized_request, ModelRequest)
            normalized_part = normalized_request.parts[0]
            assert isinstance(normalized_part, UserPromptPart)
            normalized_part.content = "normalized current"
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=canonical)
            await owner.before_model_request(ctx, request_context(model, canonical))
            expected = owner.request_messages([], ctx)
            assert expected != canonical
            assert expected != normalized
            assert normalized != canonical
            notice = next(
                message for message in expected if (message.metadata or {}).get(DS4_CONTEXT_PAGING) == "notice"
            )
            assert notice.metadata is not None and notice.metadata[DS4_KEY] == NOTICE_VIEW_KEY

            bound: list[list[ModelMessage]] = []
            prepare = model.binder.prepare_model_request

            def capture(messages: list[ModelMessage], *args: Any, **kwargs: Any) -> list[ModelMessage]:
                bound.append(deepcopy(messages))
                return prepare(messages, *args, **kwargs)

            cast("Any", model.binder).prepare_model_request = capture
            async with model.request_stream(normalized, None, ModelRequestParameters(), run_context=ctx) as stream:
                async for _ in stream:
                    pass

            assert bound == [expected]
            assert [message.metadata for message in bound[0]] == [message.metadata for message in expected]
            assert [message.parts for message in bound[0]] == [message.parts for message in expected]
            assert canonical == before
            assert all((message.metadata or {}).get(DS4_CONTEXT_PAGING) is None for message in canonical)
            assert normalized_part.content == "normalized current"

    asyncio.run(scenario())


def test_oversized_pending_results_are_rejected_only_on_view_clones(monkeypatch: Any) -> None:
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart

    from ds4py.capabilities.context import TOOL_RESULT_DOES_NOT_FIT
    from ds4py.errors import ContextSelectionError
    from ds4py.message_metadata import DS4_VIEW_OF

    canonical: list[ModelMessage] = [
        user("current", "current"),
        ModelResponse(
            [ToolCallPart("read", {}, tool_call_id="one"), ToolCallPart("read", {}, tool_call_id="two")],
            metadata={DS4_KEY: "calls"},
        ),
        ModelRequest(
            [
                ToolReturnPart("read", "first oversized", tool_call_id="one"),
                ToolReturnPart("read", "second oversized", tool_call_id="two"),
            ],
            metadata={DS4_KEY: "results"},
        ),
    ]
    before = deepcopy(canonical)
    calls = 0

    def select(_history: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ContextSelectionError("ACTIVE_REQUEST_TOO_LARGE", "too large")
        candidate = _history.canonical_messages
        return SelectionResult(
            candidate_messages=candidate,
            measurement=cast("Any", SimpleNamespace(total_tokens=1)),
            effective_budget=10,
            effective_target=5,
            mode="recovery",
            state=None,
            notice=None,
        )

    monkeypatch.setattr("ds4py.capabilities.rolling_window.select_view", select)
    binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
    session: Any = MockSession()
    session.transcript = binder.session.transcript
    session.engine = binder.session.engine
    binder.session = session
    owner = RollingWindow(cast("Any", binder), disabled_owner=LLMCompaction(cast("Any", binder)), token_budget=10)
    model = SimpleNamespace(binder=binder)
    ctx = RunContext(deps=None, model=cast("Any", model), usage=RunUsage(), messages=canonical)

    asyncio.run(owner.before_model_request(ctx, request_context(model, list(canonical))))
    view = owner.request_messages([], ctx)

    assert canonical == before
    clone = view[-1]
    assert clone is not canonical[-1]
    assert clone.metadata == {
        DS4_KEY: "results:rejected:1",
        DS4_CONTEXT_PAGING: "recovery",
        DS4_VIEW_OF: "results",
    }
    assert cast("ToolReturnPart", clone.parts[0]).content == "first oversized"
    assert cast("ToolReturnPart", clone.parts[1]).content == TOOL_RESULT_DOES_NOT_FIT


def test_disabled_owner_delegates_policy_and_disables_only_recovery_tools() -> None:
    async def scenario() -> None:
        binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
        session: Any = MockSession()
        session.transcript = binder.session.transcript
        session.engine = binder.session.engine
        binder.session = session
        compactor = LLMCompaction(cast("Any", binder))
        owner = RollingWindow(cast("Any", binder), disabled_owner=compactor, enabled=False)
        canonical: list[ModelMessage] = [user("canonical", "canonical")]
        normalized: list[ModelMessage] = [user("normalized", "normalized")]
        model = SimpleNamespace(binder=binder)
        ctx = RunContext(deps=None, model=cast("Any", model), usage=RunUsage(), messages=canonical)
        calls: list[str] = []

        async def before(context: Any, request: Any) -> Any:
            calls.append("before")
            return request

        async def make_room(messages: list[Any], request: Any) -> Any:
            calls.append("room")
            from ds4py.capabilities.context import RoomResult

            return RoomResult(made_room=True)

        cast("Any", compactor).before_model_request = before
        cast("Any", compactor).make_room = make_room
        request = request_context(model, list(canonical))
        assert await owner.before_model_request(ctx, request) is request
        from ds4py.capabilities.context import ContextRequest

        await owner.make_room(canonical, ContextRequest(1, "tool_result"))

        assert calls == ["before", "room"]
        assert owner.request_messages(normalized, ctx) is normalized
        assert owner.reserve_messages(normalized, ctx) is canonical
        tools = await owner.get_toolset().get_tools(ctx)
        reply = await owner.get_toolset().call_tool("search_history", {"query": "any"}, ctx, tools["search_history"])
        assert reply.return_value == "Context paging recovery tools are disabled."

    asyncio.run(scenario())


def test_compaction_keeps_identity_ordinary_requests_and_canonical_reserve_recovery() -> None:
    binder = StubBinder([0], transcript_tokens=0, ctx=65536)
    owner = LLMCompaction(cast("Any", binder))
    normalized: list[ModelMessage] = [user("normalized", "normalized")]
    canonical: list[ModelMessage] = [user("canonical", "canonical")]
    ctx = RunContext(deps=None, model=cast("Any", SimpleNamespace()), usage=RunUsage(), messages=canonical)

    assert owner.request_messages(normalized, ctx) is normalized
    assert owner.reserve_messages(normalized, ctx) is canonical
