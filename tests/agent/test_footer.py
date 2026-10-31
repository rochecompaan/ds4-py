import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from prompt_toolkit.utils import get_cwidth
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandStartedEvent
from test_tui_commands import FakeRuntime

from ds4py.capabilities.run_record import DS4RunRecord
from ds4py.engine.events import Prefill, Synced
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.plan import Plan
from ds4py.tui_render import FooterModel


class EventRuntime(FakeRuntime):
    def emit(self, kind: Any, **payload: Any) -> None:
        event = SessionEvent(kind, "session", payload)
        for listener, kinds in tuple(self.listeners):
            if kinds is None or kind in kinds:
                listener(event)


def text(fragments: Any) -> str:
    return "".join(value for _style, value in fragments)


def test_footer_tracks_live_plan_prefill_service_and_extension_status() -> None:
    runtime: Any = EventRuntime()
    footer = FooterModel(runtime)
    runtime.emit("run_start", run_id="run")
    runtime.emit("Prefill", event=Prefill(done=7, total=9, cached_prefix=2, tps=12.5))
    runtime.emit(
        "Synced",
        event=Synced(
            pos=42,
            plan=Plan("extend", 2, 40, True, None),
            prefilled=40,
        ),
    )
    runtime.emit("service_started", pid=123, status="running")
    runtime.emit("status", key="custom", text="ready")

    rendered = text(footer.render(240))
    assert "running" in rendered
    assert "extend" in rendered
    assert "42/1000" in rendered
    assert "7/9" in rendered
    assert "12.5" in rendered
    assert "123" in rendered
    assert "custom" in rendered and "ready" in rendered
    assert runtime.calls == []

    runtime.emit("run_end", run_id="run", status="success")
    idle = text(footer.render(240))
    assert "idle" in idle
    assert "7/9" not in idle


def test_long_cwd_and_narrow_terminal_keep_one_line() -> None:
    runtime: Any = EventRuntime()
    runtime.project_dir = Path("/" + "very-long-directory/" * 40)
    footer = FooterModel(runtime)
    runtime.emit("status", key="custom", text="x" * 400)

    fragments = footer.render(24)
    rendered = text(fragments)
    assert len(fragments) == 1
    assert "\n" not in rendered
    assert get_cwidth(rendered) <= 24
    assert rendered.endswith("…")


class RecordSession:
    id = "session"

    def __init__(self) -> None:
        self.listeners = []

    def on_event(self, listener: Any, *, kinds: Any = None) -> Any:
        self.listeners.append((listener, kinds))
        return lambda: self.listeners.remove((listener, kinds))

    def emit(self, kind: Any, **payload: Any) -> None:
        event = SessionEvent(kind, self.id, payload)
        for listener, allowed in tuple(self.listeners):
            if allowed is None or kind in allowed:
                listener(event)


def test_run_record_republishes_actual_shell_service_events(tmp_path: Path) -> None:
    session: Any = RecordSession()
    record = DS4RunRecord(session, tmp_path / "events.jsonl")
    seen = []
    session.on_event(lambda event: seen.append(event))
    ctx = SimpleNamespace(conversation_id="session", run_id="run")

    async def scenario() -> None:
        await record.on_event(ctx, event=CommandStartedEvent(command="sleep 1", pid=321))
        await record.on_event(
            ctx,
            event=CommandFinishedEvent(
                pid=321,
                output_path="out",
                status_path="status",
                exit_code=None,
                truncated=False,
            ),
        )
        await record.on_event(
            ctx,
            event=CommandFinishedEvent(
                pid=321,
                output_path="out",
                status_path="status",
                exit_code=0,
                truncated=False,
            ),
        )

    asyncio.run(scenario())
    services = [event for event in seen if event.kind.startswith("service_")]
    assert [event.kind for event in services] == [
        "service_started",
        "service_status",
        "service_stopped",
    ]
    assert services[0].payload["pid"] == 321
    assert services[0].payload["status"] == "running"
    assert services[1].payload["status"] == "running"
    assert services[2].payload["status"] == "stopped"
