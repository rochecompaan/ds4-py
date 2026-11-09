import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from ds4py.engine.lifecycle import SessionEvent
from ds4py.errors import EngineError
from ds4py.store import SaveRecord
from ds4py.tui import TUI
from ds4py.tui_commands import CommandError, SlashCommandRegistry


def save_record(session_id: Any = "saved-session", title: Any = "Saved title") -> SaveRecord:
    return SaveRecord(
        session_id=session_id,
        conversation_id=session_id,
        run_id=f"{session_id}.save-1",
        step_index=0,
        model_id=1,
        quant_bits=4,
        ctx=1000,
        title=title,
        created_at="2026-01-01T00:00:00Z",
        last_used="2026-01-01T00:00:00Z",
        payload_bytes=0,
        tokens=0,
        tokens_sha256="0" * 64,
        segments=(),
    )


class FakeRuntime:
    project_dir = Path("/project")
    context_size = 1000

    def __init__(self) -> None:
        self.calls = []
        self.current = "old-session"
        self.closed = []
        self.hints = False
        self.listeners = []

    def on_event(self, listener: Any, *, kinds: Any = None) -> Any:
        self.listeners.append((listener, kinds))
        return lambda: self.listeners.remove((listener, kinds))

    async def new(self) -> Any:
        self.calls.append(("new",))
        self.closed.append(self.current)
        self.current = "new-session"
        return self.current

    async def save(self) -> Any:
        self.calls.append(("save",))
        return save_record(self.current)

    async def switch(self, prefix: Any) -> Any:
        self.calls.append(("switch", prefix))
        if prefix == "bad":
            raise EngineError("unknown save")
        record = save_record(f"{prefix}-session", "Restored title")
        self.closed.append(self.current)
        self.current = record.session_id
        return record

    def set_hints(self, enabled: Any) -> None:
        self.hints = enabled
        self.calls.append(("hints", enabled))

    def cancel(self) -> None:
        self.calls.append(("cancel",))

    async def run(self, prompt: Any) -> None:
        self.calls.append(("run", prompt))


def make_tui(runtime: Any = None) -> TUI:
    return TUI(cast("Any", runtime or FakeRuntime()), output=DummyOutput())


def test_builtin_commands_execute_runtime_lifecycle() -> None:
    runtime = FakeRuntime()
    tui = make_tui(runtime)

    async def scenario() -> None:
        assert await tui.commands.execute("/new") == "new session new-session"
        assert runtime.closed == ["old-session"]

        switched = cast("Any", await tui.commands.execute("/switch abc"))
        assert "abc-session" in switched
        assert "Restored title" in switched
        assert runtime.closed[-1] == "new-session"

        saved = cast("Any", await tui.commands.execute("/save"))
        assert "abc-session" in saved
        assert "Saved title" in saved

        assert await tui.commands.execute("/hints on") == "hints on"
        assert await tui.commands.execute("/hints off") == "hints off"
        assert runtime.hints is False
        await tui.commands.execute("/exit")
        assert tui.exiting is True

    asyncio.run(scenario())


def test_command_errors_name_the_problem_and_usage() -> None:
    tui = make_tui()

    async def scenario() -> None:
        with pytest.raises(CommandError, match="unknown command"):
            await tui.commands.execute("/list")
        with pytest.raises(CommandError, match="/switch <prefix>"):
            await tui.commands.execute("/switch")
        with pytest.raises(CommandError, match=r"/hints on\|off"):
            await tui.commands.execute("/hints perhaps")
        with pytest.raises(CommandError, match="usage: /compact"):
            await tui.commands.execute("/compact extra")

    asyncio.run(scenario())


def test_slash_completion_covers_only_builtin_commands_and_extensions() -> None:
    tui = make_tui()
    names = {item.text.strip() for item in tui.commands.completions("/")}
    assert names == {"/exit", "/new", "/save", "/switch", "/hints", "/compact"}
    assert {item.text.strip() for item in tui.commands.completions("/co")} == {"/compact"}
    assert {item.text.strip() for item in tui.commands.completions("/sw")} == {"/switch"}

    tui.commands.register("custom", lambda: "ok", description="Custom command")
    assert "/custom" in {item.text.strip() for item in tui.commands.completions("/")}


def test_extension_command_uses_the_same_execution_path() -> None:
    registry = SlashCommandRegistry()
    registry.register(
        "hello",
        lambda name: f"hello {name}",
        description="Greet",
        usage="/hello <name>",
    )
    assert asyncio.run(registry.execute("/hello world")) == "hello world"


def test_compaction_lifecycle_updates_footer_without_response_text() -> None:
    runtime = FakeRuntime()
    with create_pipe_input() as input_:
        tui = TUI(cast("Any", runtime), input=input_, output=DummyOutput())
        try:
            tui.footer.handle_event(SessionEvent("before_compact", "session", {"reason": "manual", "old_tokens": 900}))
            assert tui.footer.status == "compacting"
            assert "compacting" in "".join(fragment[1] for fragment in tui.footer.render(200))
            tui.footer.handle_event(
                SessionEvent(
                    "compacted",
                    "session",
                    {
                        "reason": "manual",
                        "old_tokens": 900,
                        "new_tokens": 200,
                        "generated_tokens": 13,
                    },
                )
            )
            assert tui.footer.status == "idle"
            assert tui.footer.ctx_used == 200
            assert "900 → 200 tokens, 13 generated" in "".join(fragment[1] for fragment in tui.footer.render(200))
            assert tui.renderer.handle_event(SessionEvent("compacted", "session", {})) == []
        finally:
            tui.close()


def test_compact_is_reported_busy_immediately_during_prompt_loop() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        busy_reported = asyncio.Event()
        finish = asyncio.Event()
        calls: list[str] = []
        runtime = FakeRuntime()

        async def run(prompt: str) -> None:
            started.set()
            await finish.wait()

        async def compact() -> None:
            calls.append("compact")
            raise EngineError("a run is already active")

        cast("Any", runtime).run = run
        cast("Any", runtime).compact = compact
        with create_pipe_input() as input_:
            tui = TUI(cast("Any", runtime), input=input_, output=DummyOutput())
            messages: list[str] = []

            def show(text: object) -> None:
                messages.append(str(text))
                busy_reported.set()

            cast("Any", tui).show_message = show
            task = asyncio.create_task(tui.run("hello"))
            try:
                await started.wait()
                input_.send_text("/compact\n")
                await asyncio.wait_for(busy_reported.wait(), 2)
                assert messages == ["error: a run is already active"]
                assert calls == ["compact"]
                assert not finish.is_set()
                assert runtime.calls == []
            finally:
                finish.set()
                input_.send_text("/exit\n")
                await asyncio.wait_for(task, 2)

    asyncio.run(scenario())
