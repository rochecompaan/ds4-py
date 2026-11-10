"""A cancelled close waiter must retain teardown through automatic compaction."""

import asyncio
from pathlib import Path

import pytest
from test_compaction_async import Gate
from test_compaction_capability import script
from test_context_commands import make_runtime, seed

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative, FakeSession
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.sampling import SamplingPolicy
from ds4py.errors import EngineError
from ds4py.model import DS4Model


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("close_error", [False, True])
def test_cancelled_close_retains_one_teardown_until_run_drains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repeat: bool,
    close_error: bool,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    async def scenario() -> None:
        with ds4py.open("f.gguf", ctx=16384, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
            runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
            seed(runtime)
            runtime.components.compactor.soft = 0
            session = runtime.model.session
            gate = Gate()
            argmax = FakeSession.argmax
            close = session.close
            closed_tokens: list[tuple[int, ...]] = []
            events: list[SessionEvent] = []
            runtime.on_event(events.append)

            def summary(native: FakeSession) -> int:
                gate.hold()
                return argmax(native)

            def close_session() -> None:
                closed_tokens.append(session._worker.call(lambda e, native: tuple(native.tokens())).result())
                close()
                if close_error:
                    raise EngineError("state close failed")

            monkeypatch.setattr(FakeSession, "argmax", summary)
            monkeypatch.setattr(session, "close", close_session)
            run = asyncio.create_task(runtime.run("next"))
            closes: list[asyncio.Task[None]] = []
            try:
                await asyncio.wait_for(gate.entered.wait(), 5)
                first = asyncio.create_task(runtime.close())
                closes.append(first)
                await asyncio.sleep(0)
                second = asyncio.create_task(runtime.close())
                closes.append(second)
                await asyncio.sleep(0)
                first.cancel()
                await asyncio.sleep(0)
                if repeat:
                    first.cancel()
                await asyncio.sleep(0)
                assert not second.done(), "concurrent close returned before teardown"
                assert not first.done(), "cancelled close abandoned its teardown owner"
                assert not run.done() and not session._operations.closed
                assert runtime._active is not None and runtime._active.kind == "close"
                with pytest.raises(EngineError, match="runtime is closing"):
                    await runtime.compact()
                gate.release.set()
                with pytest.raises(asyncio.CancelledError) as cancelled:
                    await asyncio.wait_for(first, 5)
                if close_error:
                    assert isinstance(cancelled.value.__cause__, EngineError)
                    assert str(cancelled.value.__cause__) == "state close failed"
                await asyncio.wait_for(second, 5)
                result = await asyncio.gather(run, return_exceptions=True)
                assert isinstance(result[0], asyncio.CancelledError)
                assert runtime._active is None
                assert runtime._closed and runtime.status == "closed" and session._operations.closed
                assert len(closed_tokens) == 1
                assert session.transcript.tokens[: len(closed_tokens[0])] == closed_tokens[0]
                assert [e.payload["outcome"] for e in events if e.kind == "compaction_finished"] == ["cancelled"]
                assert len([e for e in events if e.kind == "run_end"]) == 1
                assert not session._operations.active_round and session._operations.kind is None
                assert session._worker._poisoned_by is None
                stable = session.transcript.tokens, list(runtime.messages), len(events), list(closed_tokens)
                await runtime.close()
                await asyncio.sleep(0)
                assert (session.transcript.tokens, runtime.messages, len(events), closed_tokens) == stable
            finally:
                gate.release.set()
                await asyncio.gather(run, *closes, return_exceptions=True)
                await runtime.close()
                monkeypatch.setattr(session, "close", close)

    asyncio.run(scenario())
