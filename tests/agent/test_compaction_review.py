"""Review reproductions for close ownership and automatic summary accounting."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ToolCallPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
from test_compaction_async import Gate
from test_compaction_capability import script
from test_context_commands import make_runtime, seed
from test_context_strategy import RecordingOwner

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative, FakeSession
from ds4py.capabilities.compaction import CompactionResult
from ds4py.capabilities.context import RoomResult
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.sampling import SamplingPolicy
from ds4py.engine.transcript import TranscriptReplacement
from ds4py.errors import EngineError
from ds4py.model import DS4Model


async def cancel_at_replacement(task: asyncio.Task[None], gate: Gate) -> None:
    try:
        await asyncio.wait_for(gate.entered.wait(), 5)
        assert not gate.timed_out
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    finally:
        gate.release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.mark.parametrize("close_error", [False, True])
def test_close_rejects_followon_compaction_from_terminal_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    close_error: bool,
) -> None:
    # The event schedules a successor before close's drain waiter wakes. A
    # snapshot of _manual_done alone permits work to enter the closing Session.
    async def scenario() -> None:
        with ds4py.open("f.gguf", ctx=16384, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
            runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
            seed(runtime)
            session = runtime.model.session
            gate = Gate()
            argmax = FakeSession.argmax
            original_close = session.close
            close_calls = 0
            attempts: list[asyncio.Task[CompactionResult]] = []
            events: list[SessionEvent] = []

            def summary(native: FakeSession) -> int:
                gate.hold()
                return argmax(native)

            def followon(event: SessionEvent) -> None:
                events.append(event)
                if event.kind == "compaction_finished" and not attempts:
                    attempts.append(asyncio.create_task(runtime.compact()))

            def close_session() -> None:
                nonlocal close_calls
                close_calls += 1
                original_close()
                if close_error:
                    raise EngineError("state close failed")

            runtime.on_event(followon)
            monkeypatch.setattr(FakeSession, "argmax", summary)
            monkeypatch.setattr(session, "close", close_session)
            compact = asyncio.create_task(runtime.compact())
            closes: list[asyncio.Task[None]] = []
            try:
                await asyncio.wait_for(gate.entered.wait(), 5)
                close = asyncio.create_task(runtime.close())
                closes.append(close)
                await asyncio.sleep(0)
                for operation in (
                    runtime.run("overlap"),
                    runtime.compact(),
                    runtime.save(),
                    runtime.new(),
                    runtime.switch("missing"),
                ):
                    with pytest.raises(EngineError, match="runtime is closing"):
                        await operation
                concurrent = asyncio.create_task(runtime.close())
                closes.append(concurrent)
                gate.release.set()
                results = await asyncio.wait_for(asyncio.gather(compact, close, concurrent, return_exceptions=True), 5)
                assert results[2] is None
                assert isinstance(results[0], asyncio.CancelledError)
                assert isinstance(results[1], EngineError) if close_error else results[1] is None
                assert len(attempts) == 1
                with pytest.raises(EngineError, match=r"runtime is (closing|closed)"):
                    await attempts[0]
                assert sum(e.kind == "before_compact" for e in events) == 1
                assert runtime._active is None
                assert session.history.operations.closed
                await runtime.close()
                assert close_calls == 1
            finally:
                gate.release.set()
                await asyncio.gather(compact, *attempts, *closes, return_exceptions=True)
                monkeypatch.setattr(session, "close", original_close)
                await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("entry", ["threshold", "tool", "pending"])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_automatic_summary_usage_survives_failure_and_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    outcome: str,
) -> None:
    # Missing event observation loses emitted output when no result is returned;
    # adding the result blindly after observation instead double charges success.
    async def scenario() -> None:
        with ds4py.open("f.gguf", ctx=16384, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
            runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
            seed(runtime)
            model = runtime.model
            owner = runtime.components.compactor
            owner.soft = 0 if entry == "threshold" else 1
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=runtime._messages)
            request = ModelRequestContext(
                model=model,
                messages=ctx.messages,
                model_settings=None,
                model_request_parameters=ModelRequestParameters(),
            )
            events: list[SessionEvent] = []
            runtime.on_event(events.append)
            replace = model.session.replace_transcript
            gate = Gate()

            def terminate(value: TranscriptReplacement, *, reason: str) -> object:
                if outcome == "failure":
                    raise EngineError("replacement failed")
                gate.hold()
                return replace(value, reason=reason)

            if outcome != "success":
                monkeypatch.setattr(model.session, "replace_transcript", terminate)

            async def invoke() -> None:
                if entry == "threshold":
                    await owner.before_model_request(ctx, request)
                elif entry == "tool":
                    await owner.after_tool_execute(
                        ctx,
                        call=ToolCallPart("big", {}, tool_call_id="call"),
                        tool_def=ToolDefinition(name="big"),
                        args={},
                        result="x" * 15000,
                    )
                else:
                    ctx.messages.append(ModelRequest([ToolReturnPart("big", "x" * 15000, tool_call_id="call")]))
                    await owner._fit_pending_results(ctx, request)

            try:
                if outcome == "success":
                    await invoke()
                elif outcome == "cancel":
                    await cancel_at_replacement(asyncio.create_task(invoke()), gate)
                else:
                    with pytest.raises(EngineError):
                        await invoke()
                assert sum(e.kind == "SummaryToken" for e in events) == 13
                assert ctx.usage.output_tokens == 13
                assert [e.payload["outcome"] for e in events if e.kind == "compaction_finished"] == [
                    {"success": "compacted", "failure": "failed", "cancel": "cancelled"}[outcome]
                ]
            finally:
                monkeypatch.setattr(model.session, "replace_transcript", replace)
                await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("generated", [0, 7])
def test_custom_tool_owner_keeps_result_based_usage_without_events(tmp_path: Path, generated: int) -> None:
    async def scenario() -> None:
        with ds4py.open("f.gguf", ctx=16384, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
            runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
            seed(runtime)
            owner = RecordingOwner(runtime.model.binder, room=RoomResult(True, generated))
            ctx = RunContext(deps=None, model=runtime.model, usage=RunUsage(), messages=runtime._messages)
            try:
                await owner.after_tool_execute(
                    ctx,
                    call=ToolCallPart("big", {}, tool_call_id="call"),
                    tool_def=ToolDefinition(name="big"),
                    args={},
                    result="x" * 15000,
                )
                assert len(owner.requests) == 1
                assert ctx.usage.output_tokens == generated
            finally:
                await runtime.close()

    asyncio.run(scenario())
