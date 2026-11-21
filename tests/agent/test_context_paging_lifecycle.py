"""RollingWindow settings and session-event lifecycle."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from test_restore import fake_engine, make_model

from ds4py.capabilities.context import ContextComposition
from ds4py.capabilities.rolling_window import RollingWindow
from ds4py.context_paging.selection import ContextCutFrontier, ContextCutKey, SelectionState
from ds4py.runtime import AgentRuntime
from ds4py.store import SessionStore


def marked(owner: RollingWindow) -> None:
    owner._state.selection = SelectionState(ContextCutFrontier("partialTurn", ContextCutKey("old"), "user"), None)
    owner._state.accounting["requests"] = 4
    owner._state.navigation["page"] = 2
    owner._state.restore_ids.add("interrupted")


def test_new_clears_session_state_and_status_reinitializes(tmp_path: Path) -> None:
    with fake_engine() as engine:
        store = SessionStore(tmp_path / "store")
        runtime = AgentRuntime(
            make_model(engine), tmp_path, store=store, context=ContextComposition.rolling(token_budget=16000)
        )
        owner = runtime.model.context_owner
        assert isinstance(owner, RollingWindow)
        events: list[Any] = []
        runtime.on_event(events.append, kinds={"status", "context_paging_warning"})

        async def scenario() -> None:
            assert runtime.context_paging("off") == "paging off"
            marked(owner)
            events.clear()
            await runtime.new()
            assert owner.enabled
            assert owner._state.selection is None
            assert owner._state.accounting == {}
            assert owner._state.navigation == {}
            assert owner._state.restore_ids == set()
            statuses = [
                event for event in events if event.kind == "status" and event.payload["key"] == "context-paging"
            ]
            assert [event.payload["text"] for event in statuses] == [None, "paging on"]

        try:
            asyncio.run(scenario())
        finally:
            asyncio.run(runtime.close())


def test_tui_renders_paging_warning_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ds4py.tui import TUI

    with fake_engine() as engine:
        runtime = AgentRuntime(make_model(engine), tmp_path, context=ContextComposition.rolling(token_budget=16000))
        with create_pipe_input() as input_:
            tui = TUI(runtime, input=input_, output=DummyOutput())
            messages: list[str] = []
            monkeypatch.setattr(tui, "show_message", messages.append)

            async def scenario() -> None:
                tui._loop = asyncio.get_running_loop()
                renderer = asyncio.create_task(tui._render_events())
                runtime.model.session.emit("context_paging_warning", message="trim warning")
                for _ in range(20):
                    await asyncio.sleep(0)
                    if messages:
                        break
                renderer.cancel()
                await asyncio.gather(renderer, return_exceptions=True)
                assert messages == ["warning: trim warning"]

            try:
                asyncio.run(scenario())
            finally:
                tui.close()
                asyncio.run(runtime.close())


def test_tui_replays_initial_paging_warning_after_subscribing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ds4py.tui import TUI

    with fake_engine() as engine:
        runtime = AgentRuntime(
            make_model(engine),
            tmp_path,
            context=ContextComposition.rolling(token_budget=16000, trim_to=16000),
        )
        with create_pipe_input() as input_:
            tui = TUI(runtime, input=input_, output=DummyOutput())
            messages: list[str] = []
            monkeypatch.setattr(tui, "show_message", messages.append)
            tui.exiting = True
            try:
                asyncio.run(tui.run())
                assert messages == [
                    (
                        "warning: The explicit trim target (16000) is at or above the token budget "
                        "(16000); stable cuts will trim to the effective budget."
                    )
                ]
            finally:
                tui.close()
                asyncio.run(runtime.close())


def test_compaction_and_model_changed_reset_only_their_documented_state(tmp_path: Path) -> None:
    with fake_engine() as engine:
        runtime = AgentRuntime(make_model(engine), tmp_path, context=ContextComposition.rolling(token_budget=16000))
        owner = runtime.model.context_owner
        assert isinstance(owner, RollingWindow)
        try:
            runtime.context_paging("off")
            marked(owner)
            selection = owner._state.selection
            runtime.model.session.emit("before_compact", reason="manual", old_tokens=10)
            assert owner._state.accounting == {}
            assert owner._state.selection is selection
            runtime.model.session.emit("compaction_finished", reason="manual", outcome="cancelled")
            assert owner._state.selection is selection

            marked(owner)
            runtime.model.session.emit("before_compact", reason="manual", old_tokens=10)
            runtime.model.session.emit("compacted", reason="manual", old_tokens=10, new_tokens=5, generated_tokens=1)
            assert owner._state.selection is None

            runtime.context_paging("off")
            marked(owner)
            runtime.model.session.emit("model_changed", model_id=9)
            assert not owner.enabled
            assert owner._state.selection is None
            assert owner._state.accounting == {}
        finally:
            asyncio.run(runtime.close())


def test_settings_parse_failure_publishes_off_and_aborts(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "store")
    store.root.mkdir(parents=True)
    (store.root / "settings.json").write_text("{broken")
    with fake_engine() as engine:
        model = make_model(engine)
        events: list[Any] = []
        model.session.on_event(events.append)
        with pytest.raises(json.JSONDecodeError):
            AgentRuntime(model, tmp_path, store=store, context=ContextComposition.rolling())
        statuses = [event for event in events if event.kind == "status" and event.payload["key"] == "context-paging"]
        assert [event.payload["text"] for event in statuses] == ["paging off"]
        errors = [event for event in events if event.kind == "context_paging_error"]
        assert len(errors) == 1
        assert "Expecting property name" in errors[0].payload["message"]


def test_project_trust_controls_runtime_settings_and_explicit_budget_wins(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "store")
    store.root.mkdir(parents=True)
    (store.root / "settings.json").write_text('{"contextPaging":{"enabled":false,"tokenBudget":9000}}')
    project = tmp_path / "project"
    (project / ".ds4py").mkdir(parents=True)
    (project / ".ds4py" / "settings.json").write_text('{"contextPaging":{"enabled":true,"tokenBudget":7000}}')

    with fake_engine() as engine:
        trusted = AgentRuntime(
            make_model(engine),
            project,
            store=store,
            context=ContextComposition.rolling(token_budget=8000),
            project_trusted=True,
        )
        owner = trusted.model.context_owner
        assert isinstance(owner, RollingWindow)
        assert owner.enabled
        assert owner._state.settings.token_budget == 8000
        asyncio.run(trusted.close())

    with fake_engine() as engine:
        untrusted = AgentRuntime(
            make_model(engine), project, store=store, context=ContextComposition.rolling(), project_trusted=False
        )
        owner = untrusted.model.context_owner
        assert isinstance(owner, RollingWindow)
        assert not owner.enabled
        assert owner._state.settings.token_budget == 9000
        asyncio.run(untrusted.close())


def test_trim_warning_is_emitted_once_per_pair_across_session_changes(tmp_path: Path) -> None:
    with fake_engine() as engine:
        model = make_model(engine)
        events: list[Any] = []
        model.session.on_event(events.append, kinds={"context_paging_warning"})
        runtime = AgentRuntime(
            model,
            tmp_path,
            context=ContextComposition.rolling(token_budget=16000, trim_to=16000),
        )

        async def scenario() -> None:
            assert len(events) == 1
            await runtime.new()
            runtime.model.session.emit("model_changed", model_id=9)
            assert len(events) == 1

        try:
            asyncio.run(scenario())
        finally:
            asyncio.run(runtime.close())


def test_selection_failure_records_run_end_and_reraises_the_same_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ds4py.errors import ContextSelectionError

    with fake_engine() as engine:
        runtime = AgentRuntime(make_model(engine), tmp_path, context=ContextComposition.rolling(token_budget=16000))
        owner = runtime.model.context_owner
        assert isinstance(owner, RollingWindow)
        failure = ContextSelectionError("RESIDENT_INPUT_TOO_LARGE", "resident input exceeds budget", 20, 20, 10)

        def reject(*args: Any, **kwargs: Any) -> Any:
            raise failure

        monkeypatch.setattr(owner, "_select", reject)
        events: list[Any] = []
        runtime.on_event(events.append, kinds={"run_start", "run_end"})
        try:
            with pytest.raises(ContextSelectionError) as caught:
                asyncio.run(runtime.run("hello"))
            assert caught.value is failure
            assert caught.value.code == "RESIDENT_INPUT_TOO_LARGE"
            assert [event.kind for event in events] == ["run_start", "run_end"]
            assert events[-1].payload["status"] == "error"
            assert events[-1].payload["error"] == str(failure)
        finally:
            asyncio.run(runtime.close())
