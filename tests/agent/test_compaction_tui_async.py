"""The real input application must paint and accept controls during compaction."""

import asyncio
from pathlib import Path

import pytest
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from test_compaction_async import Gate
from test_compaction_capability import script
from test_context_commands import make_runtime, seed

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative, FakeSession
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.sampling import SamplingPolicy
from ds4py.model import DS4Model
from ds4py.tui import TUI
from ds4py.tui_render import FooterModel


@pytest.mark.parametrize("cancel", [False, True])
def test_real_prompt_paints_compacting_and_keeps_one_input_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    # An inactive PromptSession silently ignores invalidate(); rendering a
    # FooterModel in a synthetic consumer alone cannot catch that failure.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    async def scenario() -> None:
        with ds4py.open("f.gguf", ctx=16384, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
            runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
            seed(runtime)
            gate = Gate()
            argmax = FakeSession.argmax
            painted = asyncio.Event()
            idle_painted = asyncio.Event()
            busy = asyncio.Event()
            signalled = asyncio.Event()
            compact_done = asyncio.Event()
            run_done = asyncio.Event()
            events: list[SessionEvent] = []
            starts: asyncio.Queue[int] = asyncio.Queue()
            active = maximum = count = 0
            render = FooterModel.render

            def paint(footer: FooterModel, width: int) -> StyleAndTextTuples:
                fragments = render(footer, width)
                if footer.status == "compacting":
                    painted.set()
                elif footer.status == "idle" and compact_done.is_set():
                    idle_painted.set()
                return fragments

            def summary(native: FakeSession) -> int:
                gate.hold()
                return argmax(native)

            def record(event: SessionEvent) -> None:
                events.append(event)
                if event.kind == "compaction_finished":
                    compact_done.set()
                elif event.kind == "run_end":
                    run_done.set()

            runtime.on_event(record)
            cancel_operation = runtime.cancel

            def signal() -> None:
                signalled.set()
                cancel_operation()

            monkeypatch.setattr(runtime, "cancel", signal)
            monkeypatch.setattr(FakeSession, "argmax", summary)
            monkeypatch.setattr(FooterModel, "render", paint)
            with create_pipe_input() as input_:
                tui = TUI(runtime, input=input_, output=DummyOutput())
                prompt = tui.session.prompt_async

                async def tracked_prompt(message: str) -> str:
                    nonlocal active, maximum, count
                    active += 1
                    maximum = max(maximum, active)
                    count += 1
                    starts.put_nowait(count)
                    try:
                        answer = await prompt(message)
                        assert isinstance(answer, str)
                        return answer
                    finally:
                        active -= 1

                def show(message: object) -> None:
                    if "already active" in str(message):
                        busy.set()

                monkeypatch.setattr(tui.session, "prompt_async", tracked_prompt)
                monkeypatch.setattr(tui, "show_message", show)
                task = asyncio.create_task(tui.run("/compact"))
                try:
                    await asyncio.wait_for(gate.entered.wait(), 5)
                    await asyncio.wait_for(painted.wait(), 1)
                    assert not gate.timed_out
                    assert tui.session.app.is_running
                    assert await asyncio.wait_for(starts.get(), 1) == 1
                    input_.send_text("/compact\n")
                    await asyncio.wait_for(busy.wait(), 1)
                    assert await asyncio.wait_for(starts.get(), 1) == 2
                    assert sum(e.kind == "before_compact" for e in events) == 1
                    if cancel:
                        input_.send_bytes(b"\x03")
                        await asyncio.wait_for(signalled.wait(), 1)
                    else:
                        input_.send_text("continue\n")
                        assert await asyncio.wait_for(starts.get(), 1) == 3
                        assert not runtime._running
                    gate.release.set()
                    await asyncio.wait_for(compact_done.wait(), 5)
                    if not cancel:
                        await asyncio.wait_for(run_done.wait(), 5)
                    await asyncio.wait_for(idle_painted.wait(), 5)
                    assert not task.done()
                    assert maximum == active == 1
                    assert tui._prompt_task is not None and not tui._prompt_task.done()
                    assert sum(e.kind == "run_start" for e in events) == (0 if cancel else 1)
                    assert [e.payload["outcome"] for e in events if e.kind == "compaction_finished"] == [
                        "cancelled" if cancel else "compacted"
                    ]
                    input_.send_bytes(b"\x04")
                    assert await asyncio.wait_for(task, 5) == 0
                    assert active == 0
                finally:
                    gate.release.set()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    tui.close()
                    await runtime.close()

    asyncio.run(scenario())
