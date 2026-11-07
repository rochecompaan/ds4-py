import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.shortcuts import PromptSession
from test_runtime import fake_engine, fixed_clock, make_model, project_dir

from ds4py._native.fake import FAKE_EOS
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.engine.events import Stopped
from ds4py.engine.lifecycle import SessionEvent
from ds4py.runtime import AgentRuntime
from ds4py.store import SessionStore
from ds4py.tui import TUI


@pytest.mark.parametrize("prompt_exception", [False, True], ids=["key", "prompt-exception"])
def test_tui_ctrl_c_interrupts_generation_and_allows_follow_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt_exception: bool
) -> None:
    """Losing Ctrl+C, the interrupted terminal event, or the follow-up must fail this test."""
    second_token = threading.Event()
    round_number = 0
    repeated = tokenize_rendered("a")[0]
    follow_up_tokens = tokenize_rendered("follow-up answer")

    def script(_prompt: Any, generated: Any) -> Any:
        nonlocal round_number
        time.sleep(0.005)
        if not generated:
            round_number += 1
        if round_number == 1:
            if len(generated) == 2:
                second_token.set()
            return repeated
        if len(generated) < len(follow_up_tokens):
            return follow_up_tokens[len(generated)]
        return FAKE_EOS

    events: list[SessionEvent] = []

    async def wait_for_prompt(tui: TUI) -> None:
        async with asyncio.timeout(2):
            while not tui.session.app.is_running:  # noqa: ASYNC110 - prompt-toolkit has no async readiness event.
                await asyncio.sleep(0.01)

    with fake_engine(script=script) as engine:
        runtime = AgentRuntime(
            make_model(engine),
            project_dir(tmp_path),
            store=SessionStore(tmp_path / "home"),
            clock=fixed_clock,
        )

        if prompt_exception:
            original_prompt_async = PromptSession.prompt_async
            raised = False

            async def interrupt_prompt(self: PromptSession[str], *args: Any, **kwargs: Any) -> str:
                nonlocal raised
                if not raised:
                    raised = True
                    reached = await asyncio.to_thread(second_token.wait, 2)
                    assert reached, "fake engine did not reach two generated tokens"
                    runtime.cancel()
                    raise KeyboardInterrupt
                line = await original_prompt_async(self, *args, **kwargs)
                assert isinstance(line, str)
                return line

            monkeypatch.setattr(PromptSession, "prompt_async", interrupt_prompt)

        async def scenario() -> int:
            loop = asyncio.get_running_loop()
            run_ended = asyncio.Event()

            def record_event(event: SessionEvent) -> None:
                events.append(event)
                if event.kind == "run_end":
                    loop.call_soon_threadsafe(run_ended.set)

            unsubscribe = runtime.on_event(record_event)
            try:
                with create_pipe_input() as input_:
                    tui = TUI(runtime, input=input_, output=DummyOutput())
                    task = asyncio.create_task(tui.run("initial"))
                    if not prompt_exception:
                        reached_generation = await asyncio.to_thread(second_token.wait, 2)
                        assert reached_generation, "fake engine did not reach two generated tokens"
                        await wait_for_prompt(tui)
                        input_.send_text("\x03")

                    await asyncio.wait_for(run_ended.wait(), timeout=2)
                    run_ended.clear()
                    input_.send_text("follow up\n")
                    await asyncio.wait_for(run_ended.wait(), timeout=2)
                    input_.send_text("/exit\n")
                    return await task
            finally:
                unsubscribe()

        try:
            result = asyncio.run(asyncio.wait_for(scenario(), timeout=5))
        finally:
            asyncio.run(runtime.close())

    stopped = [event.payload["event"] for event in events if event.kind == "Stopped"]
    run_ends = [event for event in events if event.kind == "run_end"]
    assert result == 0
    assert sum(isinstance(event, Stopped) and event.reason == "interrupted" for event in stopped) == 1
    assert len(run_ends) == 2
    assert run_ends[0].payload["status"] == "interrupted"
    assert run_ends[-1].payload["status"] == "success"
    assert run_ends[-1].payload["run_id"] != run_ends[0].payload["run_id"]
