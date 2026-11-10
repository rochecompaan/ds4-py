"""Manual compaction uses the concrete owner and the runtime's canonical history."""

import asyncio
from copy import deepcopy
from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness import HarnessDeprecationWarning
from test_compaction_capability import script
from test_context_strategy import RecordingOwner

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative
from ds4py.capabilities.compaction import CompactionResult
from ds4py.capabilities.context import ContextComposition
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.prefix import PrefixTurn
from ds4py.engine.sampling import SamplingPolicy
from ds4py.errors import EngineError
from ds4py.model import DS4Model
from ds4py.runtime import AgentRuntime
from ds4py.store import SessionStore
from ds4py.tui import TUI
from ds4py.tui_commands import SlashCommandRegistry, install_builtin_commands


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


def make_runtime(model: DS4Model, path: Path, *, context: ContextComposition | None = None) -> AgentRuntime:
    # Capture the existing harness cwd deprecations, not warnings from compaction.
    with pytest.warns(HarnessDeprecationWarning, match="deprecated and ignored: commands/paths start"):
        return AgentRuntime(model, path, store=SessionStore(path / "store"), context=context)


def seed(runtime: AgentRuntime) -> None:
    runtime.load_prefix(
        [
            PrefixTurn("user", "old " * 500),
            PrefixTurn("assistant", "old answer"),
            PrefixTurn("user", "recent"),
            PrefixTurn("assistant", "answer"),
        ]
    )
    runtime.model.binder.prepare(runtime.messages)


@pytest.mark.parametrize("custom_owner", [False, True])
def test_compact_command_rewrites_current_history_once_after_new(tmp_path: Path, custom_owner: bool) -> None:
    with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
        model = DS4Model(engine.session(), sampling=SamplingPolicy(seed=1))
        owner = RecordingOwner(model.binder) if custom_owner else None
        runtime = make_runtime(model, tmp_path, context=ContextComposition(owner=owner))
        registry = SlashCommandRegistry()
        install_builtin_commands(registry, runtime, request_exit=lambda: None)
        events: list[SessionEvent] = []
        runtime.on_event(events.append)
        compactor = runtime.components.compactor

        async def scenario() -> None:
            for _ in range(2):
                seed(runtime)
                tail = deepcopy(runtime.messages[2:])
                old_tokens = len(model.session.transcript)
                events.clear()
                output = await registry.execute("/compact")
                assert isinstance(output, str) and "Compacted" in output
                messages = runtime.messages
                assert len(messages) == 3
                assert isinstance(messages[0], ModelRequest)
                assert isinstance(messages[0].parts[0], UserPromptPart)
                assert messages[0].parts[0].content == "durable state"
                assert messages[1:] == tail
                assert model.binder.prepare(messages).kind == "unchanged"
                lifecycle = [
                    event for event in events if event.kind in {"before_compact", "compacted", "compaction_finished"}
                ]
                assert [event.kind for event in lifecycle] == ["before_compact", "compacted", "compaction_finished"]
                assert lifecycle[2].payload == {"reason": "manual", "outcome": "compacted"}
                assert lifecycle[0].payload == {"reason": "manual", "old_tokens": old_tokens}
                assert lifecycle[1].payload == {
                    "reason": "manual",
                    "old_tokens": old_tokens,
                    "new_tokens": len(model.session.transcript),
                    "generated_tokens": 13,
                }
                assert sum(event.kind == "SummaryToken" for event in events) == 13
                assert not {"Token", "TextDelta", "agent_event", "run_start"} & {event.kind for event in events}
                assert runtime.components.compactor is compactor
                assert model.context_owner is (owner if custom_owner else compactor)
                await registry.execute("/new")
            await runtime.close()

        asyncio.run(scenario())


def test_manual_noop_leaves_history_and_native_tokens_unchanged(tmp_path: Path) -> None:
    with ds4py.open("f.gguf", ctx=8192, native=FakeNative()) as engine:
        runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
        runtime.load_prefix([PrefixTurn("user", "recent")])
        runtime.model.binder.prepare(runtime.messages)
        before = deepcopy(runtime.messages), runtime.model.session.transcript.tokens
        result = asyncio.run(runtime.compact())
        assert (result.summary, result.dropped, result.kept, result.generated_tokens) == ("", 0, 1, 0)
        assert (runtime.messages, runtime.model.session.transcript.tokens) == before
        asyncio.run(runtime.close())


def test_failed_command_preserves_history(tmp_path: Path) -> None:
    with ds4py.open("f.gguf", ctx=8192, native=FakeNative(script=script)) as engine:
        runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
        seed(runtime)
        before = deepcopy(runtime.messages), runtime.model.binder.records(), runtime.model.session.transcript.tokens
        registry = SlashCommandRegistry()
        install_builtin_commands(registry, runtime, request_exit=lambda: None)
        runtime.model.session.cancel()
        with pytest.raises(EngineError):
            asyncio.run(registry.execute("/compact"))
        assert (runtime.messages, runtime.model.binder.records(), runtime.model.session.transcript.tokens) == before
        asyncio.run(runtime.close())


def test_manual_compact_rejects_active_run_without_progress_or_cancellation(tmp_path: Path) -> None:
    with ds4py.open("f.gguf", ctx=65536, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
        runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
        events: list[SessionEvent] = []
        runtime.on_event(events.append)

        def compact_during_run(event: SessionEvent) -> None:
            if event.kind == "run_start":
                attempts.append(asyncio.create_task(runtime.compact()))

        attempts: list[asyncio.Task[CompactionResult]] = []
        runtime.on_event(compact_during_run)

        async def scenario() -> None:
            result = await runtime.run("hello")
            assert result.output == "durable state"
            assert len(attempts) == 1
            with pytest.raises(EngineError, match="active"):
                await attempts[0]
            assert not {"before_compact", "compacted", "compaction_finished", "Interrupted"} & {
                event.kind for event in events
            }
            await runtime.close()
            with pytest.raises(EngineError, match="closed"):
                await runtime.compact()

        asyncio.run(scenario())


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("command", ["/compact", "/shrink"])
def test_tui_compact_clears_progress_on_noop_or_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: bool,
    command: str,
) -> None:
    with ds4py.open("f.gguf", ctx=8192, native=FakeNative(script=script)) as engine:
        runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
        if failure:
            seed(runtime)
            runtime.model.session.cancel()
        with create_pipe_input() as input_:
            tui = TUI(runtime, input=input_, output=DummyOutput())
            messages: list[object] = []
            monkeypatch.setattr(tui, "show_message", messages.append)
            tui.commands.register("shrink", runtime.compact, description="Compact through an alias", usage="/shrink")
            try:
                asyncio.run(tui.dispatch(command))
                assert tui.footer.status == "idle"
                assert len(messages) == 1
                if failure:
                    assert str(messages[0]).startswith("error:")
                    assert "interrupted" in str(messages[0])
                elif command == "/compact":
                    assert "nothing" in str(messages[0]).lower()
                else:
                    assert isinstance(messages[0], CompactionResult)
                    assert messages[0].dropped == 0
            finally:
                tui.close()
                asyncio.run(runtime.close())
