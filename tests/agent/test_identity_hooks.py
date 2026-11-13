"""Identity-change hook orchestration and telemetry tests."""

import asyncio
from pathlib import Path
from typing import Any, Never, cast

import pytest
from test_restore import dump, fake_engine, fixed_clock, make_model, read_events, saved_session, scripted_rounds

from ds4py.engine.decode import PlainStep
from ds4py.errors import EngineError
from ds4py.runtime import AgentRuntime


def test_switch_hooks_wrap_callback_relocation_and_bookkeeping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ds4py.runtime as runtime_module

    project, store, record, saved = saved_session(tmp_path)
    with fake_engine(script=scripted_rounds("old answer")) as engine:
        model = make_model(engine)
        order = []
        runtime: Any = None
        old_id = ""
        old_path = Path()

        def before() -> None:
            order.append("before")
            assert runtime.session_id == old_id
            assert cast("Any", runtime._record).path == old_path

        def after() -> None:
            order.append("after")
            assert runtime.session_id == record.session_id
            assert dump(runtime.messages) == dump(saved["messages"])
            assert runtime.title == record.title
            assert runtime.status == "success"
            assert cast("Any", runtime._record).path == store.session_dir(record.session_id) / "events.jsonl"

        runtime = AgentRuntime(
            model,
            project,
            store=store,
            clock=fixed_clock,
            before_identity_change=before,
            after_identity_change=after,
        )
        original_restore = runtime_module.restore_snapshot

        def checked_restore(**kwargs: Any) -> Any:
            order.append("restore")
            assert cast("Any", runtime._record).path == store.session_dir(record.session_id) / "events.jsonl"
            return original_restore(**kwargs)

        monkeypatch.setattr(runtime_module, "restore_snapshot", checked_restore)

        async def scenario() -> None:
            nonlocal old_id, old_path
            await runtime.run("old conversation")
            old_id = runtime.session_id
            old_path = cast("Any", runtime._record).path
            await runtime.switch(record.session_id)
            assert runtime.status == "idle"
            assert order == ["before", "restore", "after"]

        try:
            asyncio.run(scenario())
            old_kinds = [event["kind"] for event in read_events(old_path)]
            target_kinds = [
                event["kind"] for event in read_events(store.session_dir(record.session_id) / "events.jsonl")
            ]
            assert "switching" in old_kinds
            assert "session_end" in old_kinds
            assert "switching" not in target_kinds
            assert "session_start" in target_kinds
        finally:
            asyncio.run(runtime.close())


def test_default_switch_keeps_target_attribution_for_transition_events(tmp_path: Path) -> None:
    project, store, record, _saved = saved_session(tmp_path)
    with fake_engine(script=scripted_rounds("old answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)

        async def scenario() -> Path:
            await runtime.run("old conversation")
            old_path = cast("Any", runtime._record).path
            await runtime.switch(record.session_id)
            return cast("Path", old_path)

        try:
            old_path = asyncio.run(scenario())
            old_kinds = [event["kind"] for event in read_events(old_path)]
            target_kinds = [
                event["kind"] for event in read_events(store.session_dir(record.session_id) / "events.jsonl")
            ]
            assert "switching" not in old_kinds
            assert "session_end" not in old_kinds
            assert "switching" in target_kinds
            assert "session_end" in target_kinds
            assert runtime.status == "idle"
        finally:
            asyncio.run(runtime.close())


def test_switch_before_hook_failure_never_enters_session_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, store, record, _saved = saved_session(tmp_path)
    with fake_engine(script=scripted_rounds("old answer")) as engine:
        model = make_model(engine)
        failure = RuntimeError("before failed")

        def before() -> None:
            raise failure

        def forbidden_switch(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("session switch must not run")

        monkeypatch.setattr(model.session, "switch", forbidden_switch)
        runtime = AgentRuntime(
            model,
            project,
            store=store,
            clock=fixed_clock,
            before_identity_change=before,
            after_identity_change=lambda: None,
        )

        async def scenario() -> None:
            await runtime.run("old conversation")
            old_id = runtime.session_id
            old_messages = dump(runtime.messages)
            old_path = cast("Any", runtime._record).path
            with pytest.raises(RuntimeError) as caught:
                await runtime.switch(record.session_id)
            assert caught.value is failure
            assert runtime.session_id == old_id
            assert dump(runtime.messages) == old_messages
            assert runtime.status == "success"
            assert cast("Any", runtime._record).path == old_path

        try:
            asyncio.run(scenario())
        finally:
            asyncio.run(runtime.close())


def test_switch_after_hook_failure_keeps_target_and_plain_step(tmp_path: Path) -> None:
    project, store, record, saved = saved_session(tmp_path)
    with fake_engine(script=scripted_rounds("old answer")) as engine:
        model = make_model(engine)
        failure = RuntimeError("after failed")

        def before() -> None:
            model.session.set_decode_step(PlainStep())

        def after() -> None:
            raise failure

        runtime = AgentRuntime(
            model,
            project,
            store=store,
            clock=fixed_clock,
            before_identity_change=before,
            after_identity_change=after,
        )

        async def scenario() -> None:
            await runtime.run("old conversation")
            with pytest.raises(RuntimeError) as caught:
                await runtime.switch(record.session_id)
            assert caught.value is failure
            assert runtime.session_id == record.session_id
            assert dump(runtime.messages) == dump(saved["messages"])
            assert runtime.status == "error"
            assert cast("Any", runtime._record).path == store.session_dir(record.session_id) / "events.jsonl"
            assert isinstance(model.session._decode_step, PlainStep)

        try:
            asyncio.run(scenario())
        finally:
            asyncio.run(runtime.close())


def test_switch_restore_failure_skips_after_and_records_target_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, store, record, saved = saved_session(tmp_path)
    with fake_engine(script=scripted_rounds("old answer")) as engine:
        model = make_model(engine)
        calls = []

        def reject_load(file: Any, **kwargs: Any) -> Never:
            raise EngineError("native restore rejected")

        monkeypatch.setattr(model.session, "load_payload", reject_load)
        runtime = AgentRuntime(
            model,
            project,
            store=store,
            clock=fixed_clock,
            before_identity_change=lambda: calls.append("before"),
            after_identity_change=lambda: calls.append("after"),
        )

        async def scenario() -> None:
            await runtime.run("old conversation")
            with pytest.raises(EngineError, match="native restore rejected"):
                await runtime.switch(record.session_id)
            assert calls == ["before"]
            assert runtime.session_id == record.session_id
            assert dump(runtime.messages) == dump(saved["messages"])
            assert runtime.status == "error"
            target_path = store.session_dir(record.session_id) / "events.jsonl"
            assert cast("Any", runtime._record).path == target_path
            starts = [event for event in read_events(target_path) if event["kind"] == "session_start"]
            assert starts[-1]["payload"]["status"] == "error"

        try:
            asyncio.run(scenario())
        finally:
            asyncio.run(runtime.close())
