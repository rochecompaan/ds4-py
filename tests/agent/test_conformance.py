"""Agent extension conformance on the deterministic native test double."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai_harness.step_persistence import FileStepStore

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.binding import MessageBinder
from ds4py.engine.events import Synced
from ds4py.engine.sampling import SamplingPolicy
from ds4py.engine.session import Engine
from ds4py.engine.worker import SyncJob
from ds4py.errors import CheckpointError
from ds4py.model import DS4Model
from ds4py.runtime import AgentRuntime
from ds4py.store import SessionStore
from ds4py.tui_commands import SlashCommandRegistry
from ds4py.tui_render import FooterModel


def fake_engine(*, script: Any = None, ctx: Any = 8192) -> Engine:
    native = FakeNative() if script is None else FakeNative(script=script)
    return ds4py.open("f.gguf", ctx=ctx, think=ThinkMode.NONE, native=native)


def reserialize(messages: Any) -> list[Any]:
    return ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))


def sync(session: Any) -> Any:
    events = list(session._worker.submit(SyncJob(session.transcript.tokens)))
    return next(event for event in events if isinstance(event, Synced))


def user(text: Any) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)])


def scripted_rounds(*texts: Any) -> Any:
    rounds = [tuple(tokenize_rendered(text)) for text in texts]

    def script(prompt: Any, generated: Any) -> Any:
        round_index = sum(token == FAKE_EOS for token in generated)
        tokens = rounds[round_index]
        return tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS

    return script


class CloseState:
    def __init__(self) -> None:
        self.close_count = 0

    @property
    def closed(self) -> Any:
        return self.close_count > 0

    def close(self) -> None:
        self.close_count += 1


def test_binding_rewrite_keeps_prefix_and_checkpoint_boundary() -> None:
    """Conformance 1: reserialized views share content, not object identity."""
    system = ModelRequest([SystemPromptPart("extension rules")])
    original = reserialize(
        [
            system,
            user("first"),
            ModelResponse([TextPart("unchanged answer")]),
            user("unchanged later request"),
        ]
    )
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        binder.prepare(original[:1])
        sync(session)
        before_checkpoint = session.checkpoint("before-cut")
        binder.prepare(original)
        sync(session)
        session.checkpoint("after-cut")
        old_tokens = session.transcript.tokens
        old_segments = binder.export_segments()
        changed_key = original[1].metadata["ds4.key"]
        cut = next(segment.start for segment in old_segments if segment.key == changed_key)
        prefix_segments = tuple(segment for segment in old_segments if segment.end <= cut)

        rewritten = reserialize(original)
        rewritten[1].parts[0].content = "a longer replacement first request"
        change = binder.prepare(rewritten)

        assert change.kind == "rewrite"
        assert change.start == cut == before_checkpoint.tokens
        assert session.transcript.tokens[:cut] == old_tokens[:cut]
        assert binder.export_segments()[: len(prefix_segments)] == prefix_segments
        with pytest.raises(CheckpointError, match="after-cut"):
            session.drop_checkpoint("after-cut")

        synced = sync(session)
        assert synced.plan.kind == "restore"
        assert synced.plan.checkpoint == "before-cut"
        assert synced.plan.reused == cut
        assert b"longer replacement" in b"".join(engine.token_text(token) for token in session.transcript.tokens[cut:])


def test_extension_command_and_status_use_existing_registries(tmp_path: Path) -> None:
    """Conformance 6: an extension adds UI behavior without built-in edits."""
    with fake_engine() as engine:
        session = engine.session()
        commands = SlashCommandRegistry()
        calls = []

        def extension_command(value: Any) -> str:
            calls.append(value)
            return f"extension:{value}"

        commands.register(
            "stub",
            extension_command,
            description="Extension command",
            usage="/stub VALUE",
        )
        runtime = SimpleNamespace(
            context_size=1000,
            project_dir=tmp_path,
            on_event=session.on_event,
        )
        footer = FooterModel(cast("Any", runtime))
        try:
            session.emit("status", key="extension", text="ready")
            result = asyncio.run(commands.execute("/stub value"))

            assert result == "extension:value"
            assert calls == ["value"]
            assert any(item.text == "/stub " for item in commands.completions("/"))
            text = "\n".join(str(line) for line in footer.render(500))
            assert "extension" in text and "ready" in text
        finally:
            footer.close()


def test_declared_native_binding_is_called_on_worker(ext_site: Any) -> None:
    """Conformance 7: declarations pass discovery and handles stay on-worker."""
    import importlib

    ds4py_test_ext = importlib.import_module("ds4py_test_ext")
    from ds4py._native.bindings import all_bindings

    assert "ds4_engine_layer_count" in {binding.name for binding in ds4py_test_ext.BINDINGS}
    assert "ds4_engine_layer_count" in {binding.name for binding in all_bindings()}
    native = FakeNative(bindings={"ds4_engine_layer_count": lambda engine_ptr: 43})
    with ds4py.open("f.gguf", ctx=64, native=native) as engine:
        layer_count = engine.native.bind("ds4_engine_layer_count")
        count, session_ptr, worker_thread = (
            engine.session()
            .on_worker(
                lambda handles: (
                    layer_count(handles.engine),
                    handles.session,
                    threading.get_ident(),
                )
            )
            .result()
        )

        assert count == 43
        assert isinstance(session_ptr, int) and session_ptr
        assert worker_thread != threading.get_ident()


def test_scoped_state_survives_prompts_and_fork_but_not_replacement(tmp_path: Path) -> None:
    """Conformance 9: state follows the live session, not its saved identity."""
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "home")
    script = scripted_rounds("first answer", "second answer", "third answer")
    with fake_engine(script=script, ctx=100_000) as engine:
        model = DS4Model(engine.session(), sampling=SamplingPolicy(seed=1))
        runtime = AgentRuntime(model, project, store=store)
        session = model.session
        final_state = None
        try:
            asyncio.run(runtime.run("one"))
            first_state = CloseState()
            assert session.state("extension", lambda: first_state) is first_state
            asyncio.run(runtime.run("two"))
            assert session.state("extension", lambda: (_ for _ in ()).throw(AssertionError("recreated"))) is first_state
            saved = asyncio.run(runtime.save())
            saved_dir = store.session_dir(saved.session_id)

            asyncio.run(runtime.new())
            assert first_state.close_count == 1
            new_state = CloseState()
            session.state("extension", lambda: new_state)
            asyncio.run(runtime.switch(saved.session_id))
            assert new_state.close_count == 1

            final_state = CloseState()
            session.state("extension", lambda: final_state)
            asyncio.run(runtime.save())
            holder = cast("Any", session.state("step_store", lambda: None))
            assert isinstance(holder.current, FileStepStore)
            old_id = runtime.session_id
            old_tokens = session.transcript.tokens
            synced_events = []
            session.on_event(
                lambda event: synced_events.append(event.payload["event"]),
                kinds={"Synced"},
            )

            forked = asyncio.run(runtime.fork())

            assert forked != old_id == saved.session_id
            assert session.state("extension", lambda: CloseState()) is final_state
            assert not final_state.closed
            assert session.state("step_store", lambda: None) is holder
            assert session.transcript.tokens == old_tokens
            asyncio.run(runtime.run("three"))
            assert synced_events[-1].plan.kind == "extend"
            forked_record = asyncio.run(runtime.save())
            forked_dir = store.session_dir(forked)
            assert forked_record.session_id == forked
            assert (forked_dir / "save.json").is_file()
            assert (forked_dir / "session.kv").is_file()
            assert (forked_dir / "steps" / forked_record.run_id).is_dir()
            assert (saved_dir / "save.json").is_file()
            assert forked_dir != saved_dir
        finally:
            asyncio.run(runtime.close())

        assert final_state is not None and final_state.close_count == 1
        assert holder.closed
