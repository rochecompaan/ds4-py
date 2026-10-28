"""AgentRuntime runs, step snapshots, and run-record execution (Task 9)."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.engine.prefix import PrefixTurn
from ds4py.engine.sampling import SamplingPolicy
from ds4py.errors import EngineError
from ds4py.model import DS4Model
from ds4py.runtime import AgentRuntime
from ds4py.store import SessionStore
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

READ_CALL = (
    "<tool_call><function=read_file><parameter=path>hello.txt</parameter>"
    "</function></tool_call>"
)


def fixed_clock():
    return datetime(2026, 10, 21, 9, 30, 0, tzinfo=timezone.utc)


def fake_engine(*, script=None, family="qwen", model_id=7):
    info = replace(
        FakeNative()._info,
        name="fake-ds4",
        model_id=model_id,
        family=family,
        is_deepseek41=False,
        is_glm_dsa=False,
        is_qwen4=family == "qwen",
    )
    native = FakeNative(info=info) if script is None else FakeNative(info=info, script=script)
    return ds4py.open("f.gguf", ctx=65536, think=ThinkMode.NONE, native=native)


def scripted_rounds(*texts):
    """One chat round per scripted text, then the last text repeats."""
    rounds = [tuple(tokenize_rendered(text)) for text in texts]
    state = {"round": -1}

    def next_token(prompt, generated):
        if not generated:
            state["round"] += 1
        sequence = rounds[min(state["round"], len(rounds) - 1)]
        return sequence[len(generated)] if len(generated) < len(sequence) else FAKE_EOS

    return next_token


def project_dir(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "hello.txt").write_text("world")
    return project


def make_model(engine, **kwargs):
    return DS4Model(engine.session(), sampling=SamplingPolicy(seed=1), **kwargs)


def read_events(path):
    assert path.is_file(), f"missing run record {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump(messages):
    return ModelMessagesTypeAdapter.dump_json(list(messages)).decode()


def test_two_prompts_and_a_tool_round_persist_exact_snapshots(tmp_path):
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds(READ_CALL, "first answer", "second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)

        async def scenario():
            first = await runtime.run("read hello")
            first_messages = runtime.messages
            second = await runtime.run("and again")
            return first, first_messages, second

        try:
            first, first_messages, second = asyncio.run(scenario())
            session_id = runtime.session_id
            final_messages = list(runtime.messages)
        finally:
            asyncio.run(runtime.close())

    # The runtime's history is exactly the harness history of the last run.
    assert dump(final_messages) == dump(second.all_messages())
    assert dump(first_messages) == dump(first.all_messages())

    # A tool round ran for real: a keyed call and its returned content.
    all_parts = [part for message in final_messages for part in message.parts]
    assert any(
        isinstance(part, ToolCallPart) and part.tool_name == "read_file" for part in all_parts
    )
    assert any(
        isinstance(part, ToolReturnPart) and "world" in str(part.content)
        for part in all_parts
    )

    steps = store.session_dir(session_id) / "steps"

    async def inspect():
        store_ = runtime_steps(steps)
        return {
            run_id: await store_.latest_snapshot(run_id=run_id)
            for run_id in (first.run_id, second.run_id)
        }, {
            run_id: await store_.list_unresolved_tool_effects(run_id=run_id)
            for run_id in (first.run_id, second.run_id)
        }

    snapshots, unresolved = asyncio.run(inspect())

    first_snapshot = snapshots[first.run_id]
    assert first_snapshot is not None
    assert dump(first_snapshot.messages) == dump(first.all_messages())
    assert first_snapshot.conversation_id == session_id
    assert first_snapshot.state == "complete"
    assert isinstance(first_snapshot.step_index, int)

    second_snapshot = snapshots[second.run_id]
    assert second_snapshot is not None
    assert dump(second_snapshot.messages) == dump(second.all_messages())

    # Responses are persistent harness objects, so their binding key is stored.
    responses = [m for m in second_snapshot.messages if isinstance(m, ModelResponse)]
    assert responses
    assert all(
        m.metadata["ds4.key"] and m.metadata["ds4.source"] == "sampled" for m in responses
    )

    # Usage comes from the real engine round; the tool effect ledger is settled.
    assert first.usage.input_tokens > 0
    assert first.usage.output_tokens > 0
    assert first.usage.requests >= 2
    assert unresolved[first.run_id] == []
    call_id = next(
        part.tool_call_id
        for message in final_messages for part in message.parts
        if isinstance(part, ToolCallPart)
    )

    async def effect():
        return await runtime_steps(steps).get_tool_effect(
            run_id=first.run_id, tool_call_id=call_id,
        )

    record = asyncio.run(effect())
    assert record is not None and record.status == "completed"
    assert any(
        isinstance(part, ToolReturnPart) and part.tool_call_id == call_id
        for message in final_messages for part in message.parts
    )


def runtime_steps(directory):
    from pydantic_ai_harness.step_persistence import FileStepStore

    return FileStepStore(Path(directory))


def test_run_record_is_valid_jsonl_with_ids_and_hex_token_bytes(tmp_path):
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds(READ_CALL, "first answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.run("read hello"))
            session_id = runtime.session_id
        finally:
            asyncio.run(runtime.close())

    events = read_events(store.session_dir(session_id) / "events.jsonl")
    assert events
    for line in events:
        assert isinstance(line, dict)
        assert set(line) >= {"ts", "kind", "session_id", "run_id", "payload"}
        assert isinstance(line["kind"], str) and line["kind"]
        assert line["session_id"] == session_id
        assert line["run_id"]

    kinds = [line["kind"] for line in events]
    # The session's own close is recorded too; this test covers the run's line span.
    run_lines = [line for line in events if line["kind"] != "session_end"]
    assert run_lines[0]["kind"] == "run_start"
    assert run_lines[-1]["kind"] == "run_end"
    for required in ("run_start", "run_end", "Synced", "Prefill", "Token", "ToolCalls"):
        assert required in kinds, kinds

    terminal = [line for line in events if line["kind"] == "run_end"]
    assert len(terminal) == 1
    assert terminal[0]["payload"]["status"] == "success"

    token_lines = [line for line in events if line["kind"] == "Token"]
    encoded = {
        value
        for line in token_lines
        for value in _find_token_bytes(line["payload"])
    }
    assert encoded
    assert all(isinstance(value, str) for value in encoded)
    assert all(all(char in "0123456789abcdef" for char in value) for value in encoded)
    assert all(len(value) % 2 == 0 for value in encoded)
    assert "b'" not in (store.session_dir(session_id) / "events.jsonl").read_text()


def _find_token_bytes(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "text_bytes" and isinstance(item, str):
                yield item
            else:
                yield from _find_token_bytes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _find_token_bytes(item)


class FailingModel(DS4Model):
    """Fail the second model request, after a tool call was already committed."""

    def __init__(self, *args, fail_from_request=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_from_request = fail_from_request
        self.requests_seen = 0

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters,
                             run_context=None):
        self.requests_seen += 1
        if self.requests_seen >= self.fail_from_request:
            raise RuntimeError("scripted model failure")
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context,
        ) as stream:
            yield stream


def test_failed_run_reports_error_and_keeps_a_loadable_snapshot(tmp_path):
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds(READ_CALL, "never reached")) as engine:
        model = FailingModel(engine.session(), sampling=SamplingPolicy(seed=1))
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            with pytest.raises(RuntimeError, match="scripted model failure"):
                asyncio.run(runtime.run("read hello"))
            assert runtime.status == "error"
            failed_messages = list(runtime.messages)
            session_id = runtime.session_id
        finally:
            asyncio.run(runtime.close())

    # The at-failure history is retained and remains loadable from the step store.
    events = read_events(store.session_dir(session_id) / "events.jsonl")
    run_end = [line for line in events if line["kind"] == "run_end"]
    assert run_end and run_end[-1]["payload"]["status"] == "error"
    failed_run_id = run_end[-1]["run_id"]
    assert run_end[-1]["payload"]["error"]

    snapshot = asyncio.run(
        runtime_steps(store.session_dir(session_id) / "steps").latest_snapshot(
            run_id=failed_run_id, include_interrupted=True,
        )
    )
    assert snapshot is not None
    # The failed stream left an empty interrupted response; the live history
    # keeps only messages that carry content, so the rest must match exactly.
    stored = list(snapshot.messages)
    while stored and not stored[-1].parts:
        stored.pop()
    assert dump(stored) == dump(failed_messages)
    # The failure point carries the answered tool round and the pending result.
    assert any(
        isinstance(part, (UserPromptPart, ToolCallPart, ToolReturnPart))
        for message in failed_messages for part in message.parts
    )
    last = failed_messages[-1]
    assert isinstance(last, ModelRequest)
    assert any(isinstance(part, ToolReturnPart) for part in last.parts)


def test_canonical_snapshot_write_failure_ends_a_successful_run_as_error(
    tmp_path, monkeypatch,
):
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("answer")) as engine:
        runtime = AgentRuntime(make_model(engine), project, store=store, clock=fixed_clock)
        original_save = runtime._holder().current.save_snapshot

        async def fail_canonical_save(snapshot):
            if snapshot.idempotency_key is None:
                raise OSError("canonical snapshot write failed")
            return await original_save(snapshot)

        monkeypatch.setattr(runtime._holder().current, "save_snapshot", fail_canonical_save)
        try:
            with pytest.raises(OSError, match="canonical snapshot write failed"):
                asyncio.run(runtime.run("hello"))
            assert runtime.status == "error"
            session_id = runtime.session_id
        finally:
            asyncio.run(runtime.close())

    terminal = [
        line for line in read_events(store.session_dir(session_id) / "events.jsonl")
        if line["kind"] == "run_end"
    ]
    assert terminal[-1]["payload"]["status"] == "error"
    assert "canonical snapshot write failed" in terminal[-1]["payload"]["error"]


def test_canonical_snapshot_write_failure_preserves_the_model_error(
    tmp_path, monkeypatch,
):
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds(READ_CALL, "never reached")) as engine:
        model = FailingModel(engine.session(), sampling=SamplingPolicy(seed=1), fail_from_request=2)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        original_save = runtime._holder().current.save_snapshot

        async def fail_canonical_save(snapshot):
            if snapshot.idempotency_key is None:
                raise OSError("canonical snapshot write failed")
            return await original_save(snapshot)

        monkeypatch.setattr(runtime._holder().current, "save_snapshot", fail_canonical_save)
        try:
            with pytest.raises(RuntimeError, match="scripted model failure"):
                asyncio.run(runtime.run("read hello"))
            assert runtime.status == "error"
            session_id = runtime.session_id
        finally:
            asyncio.run(runtime.close())

    terminal = [
        line for line in read_events(store.session_dir(session_id) / "events.jsonl")
        if line["kind"] == "run_end"
    ]
    assert terminal[-1]["payload"]["status"] == "error"
    assert "scripted model failure" in terminal[-1]["payload"]["error"]


class BlockingModel(DS4Model):
    """Hold the first model request until the test cancels the run."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered = None
        self.release = None

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters,
                             run_context=None):
        self.entered.set()
        await self.release.wait()
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context,
        ) as stream:
            yield stream


def test_cancelled_run_reports_interrupted_status(tmp_path):
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("answer")) as engine:
        model = BlockingModel(engine.session(), sampling=SamplingPolicy(seed=1))
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)

        async def scenario():
            model.entered = asyncio.Event()
            model.release = asyncio.Event()
            task = asyncio.create_task(runtime.run("hello"))
            await asyncio.wait_for(model.entered.wait(), timeout=10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return runtime.session_id

        try:
            session_id = asyncio.run(scenario())
            assert runtime.status == "interrupted"
        finally:
            asyncio.run(runtime.close())

    events = read_events(store.session_dir(session_id) / "events.jsonl")
    terminal = [line for line in events if line["kind"] == "run_end"]
    assert terminal and terminal[-1]["payload"]["status"] == "interrupted"


def test_load_prefix_seeds_history_the_next_run_renders(tmp_path):
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("continued")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            prefix = (
                PrefixTurn("user", "start the story"),
                PrefixTurn("assistant", "The door opened"),
            )
            messages = runtime.load_prefix(prefix)

            assert [type(message) for message in messages] == [ModelRequest, ModelResponse]
            assert [part.content for m in messages for part in m.parts] == [
                "start the story", "The door opened",
            ]
            assert runtime.messages == messages
            assert runtime.title == "start the story"
            # A prefix only seeds the history; nothing is rendered before a run.
            assert len(model.session.transcript) == 0

            result = asyncio.run(runtime.run("go on"))
            assert result.output == "continued"
            rendered = [segment.kind for segment in model.session.transcript.segments]
            assert rendered.index("user") < rendered.index("assistant")
            assert model.session.transcript.chat
            assert [m.parts[0].content for m in runtime.messages[:2]] == [
                "start the story", "The door opened",
            ]
        finally:
            asyncio.run(runtime.close())


def test_terminal_success_record_failure_is_an_error(tmp_path, monkeypatch):
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        terminal_events = []
        runtime.on_event(terminal_events.append, kinds={"run_end"})
        original_write = runtime._record._write
        blocked = tmp_path / "terminal-events"
        blocked.mkdir()

        def fail_terminal_success(record):
            if record["kind"] == "run_end" and record["payload"].get("status") == "success":
                path = runtime._record.path
                runtime._record.path = blocked
                try:
                    return original_write(record)
                finally:
                    runtime._record.path = path
            return original_write(record)

        monkeypatch.setattr(runtime._record, "_write", fail_terminal_success)
        try:
            with pytest.raises(EngineError, match="run record failed"):
                asyncio.run(runtime.run("hello"))
            assert runtime.status == "error"
            events = read_events(store.session_dir(runtime.session_id) / "events.jsonl")
            terminal = [line for line in events if line["kind"] == "run_end"]
            assert terminal[-1]["payload"]["status"] == "error"
            assert [event.payload["status"] for event in terminal_events] == ["error"]
        finally:
            asyncio.run(runtime.close())


def test_run_record_failure_fails_the_run_instead_of_reporting_success(tmp_path):
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        blocked = store.session_dir(runtime.session_id) / "events.jsonl"
        blocked.mkdir(parents=True)
        try:
            with pytest.raises(Exception) as failed:
                asyncio.run(runtime.run("hello"))
            assert runtime.status == "error"
            assert "events.jsonl" in str(failed.value)
        finally:
            asyncio.run(runtime.close())
