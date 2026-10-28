import asyncio
from pathlib import Path

import pytest
from prompt_toolkit.output import DummyOutput

from ds4py.errors import EngineError
from ds4py.store import SaveRecord
from ds4py.tui import TUI
from ds4py.tui_commands import CommandError, SlashCommandRegistry


def save_record(session_id="saved-session", title="Saved title"):
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

    def __init__(self):
        self.calls = []
        self.current = "old-session"
        self.closed = []
        self.hints = False
        self.listeners = []

    def on_event(self, listener, *, kinds=None):
        self.listeners.append((listener, kinds))
        return lambda: self.listeners.remove((listener, kinds))

    async def new(self):
        self.calls.append(("new",))
        self.closed.append(self.current)
        self.current = "new-session"
        return self.current

    async def save(self):
        self.calls.append(("save",))
        return save_record(self.current)

    async def switch(self, prefix):
        self.calls.append(("switch", prefix))
        if prefix == "bad":
            raise EngineError("unknown save")
        record = save_record(f"{prefix}-session", "Restored title")
        self.closed.append(self.current)
        self.current = record.session_id
        return record

    def set_hints(self, enabled):
        self.hints = enabled
        self.calls.append(("hints", enabled))

    def cancel(self):
        self.calls.append(("cancel",))

    async def run(self, prompt):
        self.calls.append(("run", prompt))


def make_tui(runtime=None):
    return TUI(runtime or FakeRuntime(), output=DummyOutput())


def test_builtin_commands_execute_runtime_lifecycle():
    runtime = FakeRuntime()
    tui = make_tui(runtime)

    async def scenario():
        assert await tui.commands.execute("/new") == "new session new-session"
        assert runtime.closed == ["old-session"]

        switched = await tui.commands.execute("/switch abc")
        assert "abc-session" in switched
        assert "Restored title" in switched
        assert runtime.closed[-1] == "new-session"

        saved = await tui.commands.execute("/save")
        assert "abc-session" in saved
        assert "Saved title" in saved

        assert await tui.commands.execute("/hints on") == "hints on"
        assert await tui.commands.execute("/hints off") == "hints off"
        assert runtime.hints is False
        await tui.commands.execute("/exit")
        assert tui.exiting is True

    asyncio.run(scenario())


def test_command_errors_name_the_problem_and_usage():
    tui = make_tui()

    async def scenario():
        with pytest.raises(CommandError, match="unknown command"):
            await tui.commands.execute("/list")
        with pytest.raises(CommandError, match="/switch <prefix>"):
            await tui.commands.execute("/switch")
        with pytest.raises(CommandError, match=r"/hints on\|off"):
            await tui.commands.execute("/hints perhaps")

    asyncio.run(scenario())


def test_slash_completion_covers_only_m3_builtins_and_extensions():
    tui = make_tui()
    names = {item.text.strip() for item in tui.commands.completions("/")}
    assert names == {"/exit", "/new", "/save", "/switch", "/hints"}
    assert {item.text.strip() for item in tui.commands.completions("/sw")} == {"/switch"}

    tui.commands.register("custom", lambda: "ok", description="Custom command")
    assert "/custom" in {item.text.strip() for item in tui.commands.completions("/")}


def test_extension_command_uses_the_same_execution_path():
    registry = SlashCommandRegistry()
    registry.register(
        "hello", lambda name: f"hello {name}",
        description="Greet", usage="/hello <name>",
    )
    assert asyncio.run(registry.execute("/hello world")) == "hello world"
