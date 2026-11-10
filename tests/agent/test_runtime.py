"""AgentRuntime runs, step snapshots, and run-record execution (Task 9)."""

import asyncio
import json
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any, cast, override

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from test_context_strategy import RecordingOwner

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.capabilities.compaction import LLMCompaction
from ds4py.capabilities.context import ContextComposition, ContextRequest, RoomResult
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.prefix import PrefixTurn
from ds4py.engine.replay import read_replay
from ds4py.engine.sampling import SamplingPolicy
from ds4py.engine.session import Engine
from ds4py.errors import EngineError
from ds4py.model import DS4Model, DS4StreamedResponse
from ds4py.noninteractive import run_noninteractive
from ds4py.prefill_trace import PrefillTraceWriter
from ds4py.runtime import AgentRuntime
from ds4py.store import SessionStore

READ_CALL = "<tool_call><function=read_file><parameter=path>hello.txt</parameter></function></tool_call>"


def fixed_clock() -> datetime:
    return datetime(2026, 10, 21, 9, 30, 0, tzinfo=UTC)


def fake_engine(*, script: Any = None, family: str = "qwen", model_id: int = 7) -> Engine:
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


def scripted_rounds(*texts: Any) -> Any:
    """One chat round per scripted text, then the last text repeats."""
    rounds = [tuple(tokenize_rendered(text)) for text in texts]
    state = {"round": -1}

    def next_token(prompt: Any, generated: Any) -> Any:
        if not generated:
            state["round"] += 1
        sequence = rounds[min(state["round"], len(rounds) - 1)]
        return sequence[len(generated)] if len(generated) < len(sequence) else FAKE_EOS

    return next_token


def project_dir(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "hello.txt").write_text("world")
    return project


def make_model(engine: Any, **kwargs: Any) -> DS4Model:
    return DS4Model(engine.session(), sampling=SamplingPolicy(seed=1), **kwargs)


def read_events(path: Any) -> Any:
    assert path.is_file(), f"missing run record {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump(messages: Any) -> str:
    return ModelMessagesTypeAdapter.dump_json(list(messages)).decode()


def test_runtime_exposes_context_and_updates_hint_notes(tmp_path: Path) -> None:
    with fake_engine() as engine:
        runtime = AgentRuntime(make_model(engine), project_dir(tmp_path))
        try:
            assert runtime.context_size == 65536
            runtime.set_hints(True)
            assert cast("Any", runtime._notes).hints_enabled is True
            runtime.set_hints(False)
            assert cast("Any", runtime._notes).hints_enabled is False
        finally:
            asyncio.run(runtime.close())


def test_two_prompts_and_a_tool_round_persist_exact_snapshots(tmp_path: Path) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds(READ_CALL, "first answer", "second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)

        async def scenario() -> Any:
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
    assert any(isinstance(part, ToolCallPart) and part.tool_name == "read_file" for part in all_parts)
    assert any(isinstance(part, ToolReturnPart) and "world" in str(part.content) for part in all_parts)

    steps = store.session_dir(session_id) / "steps"

    async def inspect() -> Any:
        store_ = runtime_steps(steps)
        return {run_id: await store_.latest_snapshot(run_id=run_id) for run_id in (first.run_id, second.run_id)}, {
            run_id: await store_.list_unresolved_tool_effects(run_id=run_id) for run_id in (first.run_id, second.run_id)
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
        cast("Any", m.metadata)["ds4.key"] and cast("Any", m.metadata)["ds4.source"] == "sampled" for m in responses
    )

    # Usage comes from the real engine round; the tool effect ledger is settled.
    assert first.usage.input_tokens > 0
    assert first.usage.output_tokens > 0
    assert first.usage.requests >= 2
    assert unresolved[first.run_id] == []
    call_id = next(
        part.tool_call_id for message in final_messages for part in message.parts if isinstance(part, ToolCallPart)
    )

    async def effect() -> Any:
        return await runtime_steps(steps).get_tool_effect(
            run_id=first.run_id,
            tool_call_id=call_id,
        )

    record = asyncio.run(effect())
    assert record is not None and record.status == "completed"
    assert any(
        isinstance(part, ToolReturnPart) and part.tool_call_id == call_id
        for message in final_messages
        for part in message.parts
    )


def runtime_steps(directory: Any) -> Any:
    from pydantic_ai_harness.step_persistence import FileStepStore

    return FileStepStore(Path(directory))


def test_prefill_trace_uses_persistence_step_indices(tmp_path: Path) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    trace_path = tmp_path / "prefill.jsonl"
    with fake_engine(script=scripted_rounds(READ_CALL, "first answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)

        async def scenario() -> Any:
            with PrefillTraceWriter(model.session, trace_path, think=engine.think):
                result = await runtime.run("read hello")
            return await runtime_steps(
                store.session_dir(runtime.session_id) / "steps",
            ).list_snapshots(run_id=result.run_id)

        try:
            snapshots = asyncio.run(scenario())
        finally:
            asyncio.run(runtime.close())

    read_replay(trace_path)  # Validate the records before inspecting harness-only metadata.
    trace_steps = [json.loads(line)["step_index"] for line in trace_path.read_text().splitlines()]
    snapshot_steps = sorted({snapshot.step_index for snapshot in snapshots})
    assert snapshot_steps == [1, 2]
    assert trace_steps == snapshot_steps


def test_run_record_is_valid_jsonl_with_ids_and_hex_token_bytes(tmp_path: Path) -> None:
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
    encoded = {value for line in token_lines for value in _find_token_bytes(line["payload"])}
    assert encoded
    assert all(isinstance(value, str) for value in encoded)
    assert all(all(char in "0123456789abcdef" for char in value) for value in encoded)
    assert all(len(value) % 2 == 0 for value in encoded)
    assert "b'" not in (store.session_dir(session_id) / "events.jsonl").read_text()


def _find_token_bytes(value: Any) -> Iterator[Any]:
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

    def __init__(self, *args: Any, fail_from_request: Any = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fail_from_request = fail_from_request
        self.requests_seen = 0

    @asynccontextmanager
    @override
    async def request_stream(
        self,
        messages: Any,
        model_settings: Any,
        model_request_parameters: Any,
        run_context: Any = None,
    ) -> AsyncGenerator[Any]:
        self.requests_seen += 1
        async with super().request_stream(
            messages,
            model_settings,
            model_request_parameters,
            run_context,
        ) as stream:
            if self.requests_seen >= self.fail_from_request:
                raise RuntimeError("scripted model failure")
            yield stream


def test_failed_run_reports_error_and_keeps_a_loadable_snapshot(tmp_path: Path) -> None:
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
            run_id=failed_run_id,
            include_interrupted=True,
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
        for message in failed_messages
        for part in message.parts
    )
    last = failed_messages[-1]
    assert isinstance(last, ModelRequest)
    assert any(isinstance(part, ToolReturnPart) for part in last.parts)


def test_request_limit_failure_saves_only_executed_bound_history(
    tmp_path: Path,
) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds(READ_CALL, "never requested")) as engine:
        runtime = AgentRuntime(
            make_model(engine),
            project,
            store=store,
            clock=fixed_clock,
            request_limit=1,
        )
        session_id = runtime.session_id

        async def scenario() -> Any:
            with pytest.raises(UsageLimitExceeded, match="request_limit"):
                await runtime.run("read hello")
            failed_messages = list(runtime.messages)
            record = await runtime.save()
            snapshot = await runtime_steps(
                store.session_dir(session_id) / "steps",
            ).latest_snapshot(run_id=record.run_id, include_interrupted=True)
            assert snapshot is not None
            runtime.model.binder.validate_restore(snapshot.messages, record.segments)
            return failed_messages, snapshot

        try:
            failed_messages, saved_snapshot = asyncio.run(scenario())
        finally:
            asyncio.run(runtime.close())

    run_ends = [
        line for line in read_events(store.session_dir(session_id) / "events.jsonl") if line["kind"] == "run_end"
    ]
    assert run_ends[-1]["payload"]["status"] == "error"
    assert "request_limit" in run_ends[-1]["payload"]["error"]
    assert dump(failed_messages) == dump(saved_snapshot.messages)
    assert isinstance(failed_messages[-1], ModelResponse)
    assert any(isinstance(part, ToolCallPart) for part in failed_messages[-1].parts)


@pytest.mark.parametrize(
    ("reason", "text", "max_tokens"),
    [
        ("max_tokens", "ab", 1),
        (
            "parser_error",
            "I'll write code\n<tool_call><function=read><parameter=path>unfinished",
            50_000,
        ),
    ],
)
def test_response_terminal_failures_have_error_records_and_nonzero_exit(
    tmp_path: Path,
    reason: str,
    text: str,
    max_tokens: int,
) -> None:
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds(text)) as engine:
        runtime = AgentRuntime(
            make_model(engine, max_tokens=max_tokens),
            project_dir(tmp_path),
            store=store,
        )
        session_id = runtime.session_id
        output, errors = StringIO(), StringIO()
        try:
            code = asyncio.run(
                run_noninteractive(
                    runtime,
                    "hello",
                    out=output,
                    err=errors,
                )
            )
            assert code == 1
            assert reason in errors.getvalue()
            assert output.getvalue() == ""
            assert runtime.status == "error"
        finally:
            asyncio.run(runtime.close())

    run_ends = [
        line for line in read_events(store.session_dir(session_id) / "events.jsonl") if line["kind"] == "run_end"
    ]
    assert run_ends[-1]["payload"]["status"] == "error"
    assert reason in run_ends[-1]["payload"]["error"]


def test_canonical_snapshot_write_failure_ends_a_successful_run_as_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("answer")) as engine:
        runtime = AgentRuntime(make_model(engine), project, store=store, clock=fixed_clock)
        original_save = runtime._holder().current.save_snapshot

        async def fail_canonical_save(snapshot: Any) -> Any:
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
        line for line in read_events(store.session_dir(session_id) / "events.jsonl") if line["kind"] == "run_end"
    ]
    assert terminal[-1]["payload"]["status"] == "error"
    assert "canonical snapshot write failed" in terminal[-1]["payload"]["error"]


def test_canonical_snapshot_write_failure_preserves_the_model_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds(READ_CALL, "never reached")) as engine:
        model = FailingModel(engine.session(), sampling=SamplingPolicy(seed=1), fail_from_request=2)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        original_save = runtime._holder().current.save_snapshot

        async def fail_canonical_save(snapshot: Any) -> Any:
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
        line for line in read_events(store.session_dir(session_id) / "events.jsonl") if line["kind"] == "run_end"
    ]
    assert terminal[-1]["payload"]["status"] == "error"
    assert "scripted model failure" in terminal[-1]["payload"]["error"]
    assert terminal[-1]["payload"]["persistence_error"] == "canonical snapshot write failed"


class BlockingModel(DS4Model):
    """Hold the first model request until the test cancels the run."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.entered: asyncio.Event | None = None
        self.release: asyncio.Event | None = None

    @asynccontextmanager
    @override
    async def request_stream(
        self,
        messages: Any,
        model_settings: Any,
        model_request_parameters: Any,
        run_context: Any = None,
    ) -> AsyncGenerator[Any]:
        cast("Any", self.entered).set()
        await cast("Any", self.release).wait()
        async with super().request_stream(
            messages,
            model_settings,
            model_request_parameters,
            run_context,
        ) as stream:
            yield stream


class BlockingAfterStartModel(DS4Model):
    """Start the round, then hold its event stream until the test releases it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.entered: asyncio.Event | None = None
        self.release: asyncio.Event | None = None

    @asynccontextmanager
    @override
    async def request_stream(
        self,
        messages: Any,
        model_settings: Any,
        model_request_parameters: Any,
        run_context: Any = None,
    ) -> AsyncGenerator[Any]:
        async with super().request_stream(
            messages,
            model_settings,
            model_request_parameters,
            run_context,
        ) as stream:
            cast("Any", self.entered).set()
            await cast("Any", self.release).wait()
            yield stream


def test_cancelled_run_reports_interrupted_status(tmp_path: Path) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("answer")) as engine:
        model = BlockingModel(engine.session(), sampling=SamplingPolicy(seed=1))
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)

        async def scenario() -> Any:
            entered = asyncio.Event()
            release = asyncio.Event()
            model.entered = entered
            model.release = release
            task = asyncio.create_task(runtime.run("hello"))
            await asyncio.wait_for(entered.wait(), timeout=10)
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


def test_runtime_cancel_finishes_run_without_native_release(tmp_path: Path) -> None:
    """Cancelling native work alone must not leave the outer agent run alive."""
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("answer")) as engine:
        model = BlockingModel(engine.session(), sampling=SamplingPolicy(seed=1))
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)

        async def scenario() -> Any:
            entered = asyncio.Event()
            release = asyncio.Event()
            model.entered = entered
            model.release = release
            task = asyncio.create_task(runtime.run("hello"))
            await asyncio.wait_for(entered.wait(), timeout=10)

            runtime.cancel()
            done, _ = await asyncio.wait({task}, timeout=0.5)
            if not task.done():
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

            assert task in done, "runtime.cancel left the agent run waiting for its next model request"
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


def test_cancelled_stop_does_not_rearm_cancellation_for_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("first", "second")) as engine:
        model = BlockingAfterStartModel(engine.session(), sampling=SamplingPolicy(seed=1))
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        session = model.session
        entered = asyncio.Event()
        release = asyncio.Event()
        reached_reconcile = asyncio.Event()
        release_reconcile = asyncio.Event()
        original_cleanup = DS4StreamedResponse._cleanup

        async def gated_cleanup(response: Any) -> None:
            await original_cleanup(response)
            reached_reconcile.set()
            await release_reconcile.wait()

        monkeypatch.setattr(DS4StreamedResponse, "_cleanup", gated_cleanup)
        cancel_calls: list[str] = []
        original_cancel = runtime.cancel

        def tracked_cancel() -> None:
            cancel_calls.append("cancel")
            original_cancel()

        monkeypatch.setattr(runtime, "cancel", tracked_cancel)

        async def scenario() -> None:
            model.entered = entered
            model.release = release
            run = asyncio.create_task(runtime.run("first"))
            await asyncio.wait_for(entered.wait(), 10)
            stop = asyncio.create_task(runtime._stop_active())
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(reached_reconcile.wait(), 10)
            stop.cancel()
            await asyncio.sleep(0)
            assert not stop.done(), "cancelled stop abandoned the retained run"
            release_reconcile.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(stop, 5)
            await asyncio.gather(run, return_exceptions=True)
            assert cancel_calls == ["cancel"], "cancelled stop re-armed the worker signal"
            assert not session._worker.cancel_event.is_set(), "cancelled stop left a stale signal"
            events: list[SessionEvent] = []
            runtime.on_event(events.append)
            model.entered = asyncio.Event()
            next_release = asyncio.Event()
            next_release.set()
            model.release = next_release
            await asyncio.wait_for(runtime.run("second"), 10)
            assert not [event for event in events if event.kind == "Interrupted"]

        try:
            asyncio.run(scenario())
        finally:
            asyncio.run(runtime.close())


def test_load_prefix_seeds_history_the_next_run_renders(tmp_path: Path) -> None:
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

            assert [type(message) for message in messages] == [
                ModelRequest,
                ModelResponse,
            ]
            assert [cast("Any", part).content for m in messages for part in m.parts] == [
                "start the story",
                "The door opened",
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
            assert [cast("Any", m.parts[0]).content for m in runtime.messages[:2]] == [
                "start the story",
                "The door opened",
            ]
        finally:
            asyncio.run(runtime.close())


def test_terminal_success_record_failure_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        terminal_events = []
        runtime.on_event(terminal_events.append, kinds={"run_end"})
        original_write = cast("Any", runtime._record)._write
        blocked = tmp_path / "terminal-events"
        blocked.mkdir()

        def fail_terminal_success(record: Any) -> Any:
            if record["kind"] == "run_end" and record["payload"].get("status") == "success":
                path = cast("Any", runtime._record).path
                cast("Any", runtime._record).path = blocked
                try:
                    return original_write(record)
                finally:
                    cast("Any", runtime._record).path = path
            return original_write(record)

        monkeypatch.setattr(cast("Any", runtime._record), "_write", fail_terminal_success)
        try:
            with pytest.raises(EngineError, match="run record failed"):
                asyncio.run(runtime.run("hello"))
            assert runtime.status == "error"
            events = read_events(store.session_dir(runtime.session_id) / "events.jsonl")
            terminal = [line for line in events if line["kind"] == "run_end"]
            assert terminal[-1]["payload"]["status"] == "error"
            assert [event.payload["status"] for event in terminal_events] == ["error"]

            monkeypatch.setattr(cast("Any", runtime._record), "_write", original_write)
            asyncio.run(runtime.run("again"))
            assert runtime.status == "success"
            events = read_events(store.session_dir(runtime.session_id) / "events.jsonl")
            terminal = [line for line in events if line["kind"] == "run_end"]
            assert [line["payload"]["status"] for line in terminal] == [
                "error",
                "success",
            ]
            assert len({line["run_id"] for line in terminal}) == 2
            assert [event.payload["status"] for event in terminal_events] == [
                "error",
                "success",
            ]
        finally:
            asyncio.run(runtime.close())


def test_terminal_success_close_failure_rolls_back_before_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("answer")) as engine:
        runtime = AgentRuntime(make_model(engine), project, store=store, clock=fixed_clock)
        terminal_events = []
        runtime.on_event(terminal_events.append, kinds={"run_end"})
        original_write = cast("Any", runtime._record)._write

        def fail_terminal_close(record: Any) -> Any:
            if record["kind"] == "run_end" and record["payload"].get("status") == "success":
                original_write(record)
                error = OSError("terminal close failed")
                cast("Any", runtime._record).failure = error
                raise error
            return original_write(record)

        monkeypatch.setattr(cast("Any", runtime._record), "_write", fail_terminal_close)
        try:
            with pytest.raises(EngineError, match="run record failed"):
                asyncio.run(runtime.run("hello"))
            assert runtime.status == "error"
            events = read_events(store.session_dir(runtime.session_id) / "events.jsonl")
            terminal = [line for line in events if line["kind"] == "run_end"]
            assert [line["payload"]["status"] for line in terminal] == ["error"]
            assert terminal[0]["payload"]["error"] == "terminal close failed"
            assert [event.payload["status"] for event in terminal_events] == ["error"]
        finally:
            asyncio.run(runtime.close())


def test_run_record_failure_fails_the_run_instead_of_reporting_success(
    tmp_path: Path,
) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        blocked = store.session_dir(runtime.session_id) / "events.jsonl"
        blocked.mkdir(parents=True)
        try:
            with pytest.raises(OSError, match=r"events\.jsonl") as failed:
                asyncio.run(runtime.run("hello"))
            assert runtime.status == "error"
            assert "events.jsonl" in str(failed.value)
        finally:
            asyncio.run(runtime.close())


def test_prefill_trace_failure_fails_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("answer")) as engine:
        runtime = AgentRuntime(make_model(engine), project_dir(tmp_path), store=store)
        trace = tmp_path / "missing" / "trace.jsonl"
        original_open = Path.open

        def fail_trace_open(path: Path, *args: Any, **kwargs: Any) -> Any:
            if path == trace:
                raise OSError("trace write failed")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fail_trace_open)
        try:
            with (
                PrefillTraceWriter(runtime.model.session, trace, think=engine.think),
                pytest.raises(EngineError, match="prefill trace failed"),
            ):
                asyncio.run(runtime.run("hello"))
            assert runtime.status == "error"
            events = read_events(store.session_dir(runtime.session_id) / "events.jsonl")
            terminal = [line for line in events if line["kind"] == "run_end"]
            assert [line["payload"]["status"] for line in terminal] == ["error"]
        finally:
            asyncio.run(runtime.close())


def test_snapshot_history_failure_fails_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("answer")) as engine:
        runtime = AgentRuntime(make_model(engine), project_dir(tmp_path), store=store)

        async def fail_history(*args: Any, **kwargs: Any) -> None:
            raise ValueError("corrupt snapshot history")

        monkeypatch.setattr(runtime._holder().current, "latest_snapshot", fail_history)
        try:
            with pytest.raises(ValueError, match="corrupt snapshot history"):
                asyncio.run(runtime.run("hello"))
            assert runtime.status == "error"
            events = read_events(store.session_dir(runtime.session_id) / "events.jsonl")
            terminal = [line for line in events if line["kind"] == "run_end"]
            assert [line["payload"]["status"] for line in terminal] == ["error"]
        finally:
            asyncio.run(runtime.close())


@pytest.mark.parametrize("identity", ["new", "switch"])
@pytest.mark.parametrize("custom", [False, True])
def test_identity_then_manual_compact_and_exact_same_model_continuation(
    tmp_path: Path, identity: str, custom: bool
) -> None:
    with fake_engine(script=scripted_rounds("answer")) as engine:
        model = make_model(engine)
        selected = RecordingOwner(model.binder) if custom else None
        runtime = AgentRuntime(
            model,
            project_dir(tmp_path),
            store=SessionStore(tmp_path / "store"),
            clock=fixed_clock,
            context=ContextComposition(selected),
        )
        owner = model.context_owner
        compactor = runtime.components.compactor
        assert owner is (selected if custom else compactor)
        assert isinstance(compactor, LLMCompaction)

        async def seed() -> None:
            await runtime.run("old " * 3000)
            await runtime.run("recent")

        async def scenario() -> None:
            await seed()
            record = await runtime.save()
            await runtime.new()
            if identity == "switch":
                await runtime.switch(record.session_id)
                # Loaded KV is usable, but old named system checkpoints were cleared.
                assert model.session._checkpoint_tokens == {}
                ordinary = await runtime.run("ordinary continuation")
                assert ordinary.output == "answer"
                assert ordinary.usage.input_tokens == 46  # 21 payload + 8 user wrapper + 17 assistant opener
            else:
                await seed()
            assert model.context_owner is owner
            old_tokens = model.session.transcript.tokens
            syncs: list[Any] = []
            unsubscribe = model.session.on_event(lambda event: syncs.append(event.payload["event"]), kinds={"Synced"})
            result = await compactor.compact_messages(runtime._messages, reason="manual")
            unsubscribe()
            assert result.dropped >= 2
            compacted = model.session.transcript.tokens
            assert len(compacted) < len(old_tokens)
            binder_records = model.binder.records()
            assert len(syncs) == 1  # only replacement sync is public; summary uses a private native sync
            plan = syncs[0].plan
            if identity == "switch":
                assert plan.reused == 0
                assert plan.to_prefill == len(compacted)
            else:
                assert plan.checkpoint == "system"
                assert plan.reused == binder_records[0].start
                assert plan.to_prefill == len(compacted) - binder_records[0].start
            continuation = await runtime.run("continue")
            assert continuation.output == "answer"
            assert continuation.usage.output_tokens == 6
            assert continuation.usage.input_tokens == 33  # 8 payload + 8 user wrapper + 17 assistant opener
            assert model.binder.records()[: len(binder_records)] == binder_records
            assert model.session.transcript.tokens[: len(compacted)] == compacted
            assert model.context_owner is owner
            await runtime.close()

        asyncio.run(scenario())


@pytest.mark.parametrize("identity", ["new", "switch"])
def test_custom_owner_receives_real_tool_and_reserve_hooks_after_identity(tmp_path: Path, identity: str) -> None:
    from dataclasses import replace

    from pydantic_ai.models import ModelRequestParameters

    from ds4py.engine.prompt import PromptBuilder
    from ds4py.errors import ContextExhausted

    with fake_engine(script=scripted_rounds("answer")) as engine:
        model = make_model(engine)
        owner = RecordingOwner(model.binder, room=RoomResult(False))
        runtime = AgentRuntime(
            model,
            project_dir(tmp_path),
            store=SessionStore(tmp_path / "store"),
            clock=fixed_clock,
            context=ContextComposition(owner),
        )

        async def scenario() -> None:
            await runtime.run("seed")
            record = await runtime.save()
            await runtime.new()
            if identity == "switch":
                await runtime.switch(record.session_id)

            # Register one genuinely oversized tool, beyond read_file's output limit.
            @runtime.components.agent.tool_plain
            def huge() -> str:
                return "z" * 100000

            cast("FakeNative", engine.native).script = scripted_rounds(
                "<tool_call><function=huge></function></tool_call>", "done"
            )
            result = await runtime.run("read huge")
            assert result.output == "done"
            assert owner.requests[-1].reason == "tool_result"
            assert any(
                isinstance(p, ToolReturnPart) and p.content == "Tool error: tool result still does not fit"
                for m in result.all_messages()
                for p in m.parts
            )
            assert model.context_owner is owner
            assert [c for c in runtime.components.components if c is owner] == [owner]
            cast("FakeNative", engine.native).script = scripted_rounds("abc")
            async with model.request_stream(
                [*runtime.messages, ModelRequest([UserPromptPart("reserve")])], None, ModelRequestParameters()
            ) as stream:
                prefix = len(PromptBuilder(engine).assistant_prefix(model.session.transcript.tokens))
                projected = len(model.session.transcript) + prefix
                engine.config = replace(engine.config, ctx=projected)
                model.session._worker.call(lambda e, native: setattr(native, "_ctx", projected + 2)).result()
                with pytest.raises(ContextExhausted):
                    async for _ in stream:
                        pass
            assert [request.reason for request in owner.requests] == ["tool_result", "reserve"]
            assert owner.requests[-1] == ContextRequest(1, "reserve")
            await runtime.close()

        asyncio.run(scenario())


def test_automatic_compaction_persists_history_and_counts_summary_once(tmp_path: Path) -> None:
    with fake_engine(script=scripted_rounds("answer", "answer", "private summary", "final answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(
            model, project_dir(tmp_path), system="", store=SessionStore(tmp_path / "store"), clock=fixed_clock
        )
        events = []
        runtime.on_event(events.append, kinds={"before_compact", "compacted"})

        async def scenario() -> None:
            await runtime.run("x" * 40000)
            await runtime.run("r" * (56000 - len(model.session.transcript)))
            retained_key = (runtime.messages[2].metadata or {})["ds4.key"]
            result = await runtime.run("finish")
            assert result.output == "final answer"
            assert result.usage.output_tokens == 27  # 15 summary + 12 public output
            assert len(runtime.messages) == 5  # summary, retained pair, pending user, final response
            assert (runtime.messages[1].metadata or {})["ds4.key"] == retained_key
            assert not any(
                isinstance(p, UserPromptPart) and p.content == "x" * 40000 for m in runtime.messages for p in m.parts
            )
            assert [event.kind for event in events] == ["before_compact", "compacted"]
            assert events[1].payload["generated_tokens"] == 15
            await runtime.close()

        asyncio.run(scenario())
