import asyncio
import time

from ds4py.capabilities.shell_jobs import DS4ShellJobs
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness import Shell


class MockSession:
    def __init__(self):
        self.states = {}

    def state(self, key, factory):
        if key not in self.states:
            self.states[key] = factory()
        return self.states[key]

    def close_states(self):
        for state in self.states.values():
            if hasattr(state, "close"):
                state.close()
        self.states.clear()


def _has_tool_return(messages):
    return any(
        isinstance(p, ToolReturnPart)
        for m in messages for p in getattr(m, "parts", [])
    )


def test_start_command_tracked_and_stopped(tmp_path):
    session = MockSession()
    jobs = DS4ShellJobs(session, tmp_path)

    def scripted(messages, info):
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("start_command", {"command": "sleep 60"}, tool_call_id="call-1"),
        ])

    agent = Agent(
        FunctionModel(scripted),
        capabilities=[LocalWorkspace(tmp_path), Shell(), jobs],
    )
    asyncio.run(agent.run("start"))

    state = session.state("shell_jobs", lambda: None)
    assert len(state.job_ids) == 1
    job_id = next(iter(state.job_ids))
    handle = tmp_path / ".pydantic-ai-harness" / "shell" / job_id / "handle"
    assert handle.exists()

    session.close_states()
    time.sleep(0.5)
    status = tmp_path / ".pydantic-ai-harness" / "shell" / job_id / "status.json"
    if status.exists():
        assert '"exit_code": null' not in status.read_text()


def test_cross_prompt_job_visible(tmp_path):
    session = MockSession()
    jobs = DS4ShellJobs(session, tmp_path)
    run_count = [0]

    def scripted(messages, info):
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        if run_count[0] == 0:
            run_count[0] += 1
            return ModelResponse(parts=[
                ToolCallPart("start_command", {"command": "sleep 60"}, tool_call_id="call-1"),
            ])
        job_id = next(iter(session.states["shell_jobs"].job_ids))
        return ModelResponse(parts=[
            ToolCallPart("check_command", {"command_id": job_id}, tool_call_id="call-2"),
        ])

    agent = Agent(
        FunctionModel(scripted),
        capabilities=[LocalWorkspace(tmp_path), Shell(), jobs],
    )
    asyncio.run(agent.run("start"))
    asyncio.run(agent.run("check"))
    assert run_count[0] == 1
    session.close_states()


def test_close_kills_a_term_ignoring_process_group(tmp_path):
    import os
    import signal

    session = MockSession()
    jobs = DS4ShellJobs(session, tmp_path)

    def scripted(messages, info):
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart(
                "start_command",
                {"command": "trap '' TERM; : > ready; while :; do sleep 1; done"},
                tool_call_id="call-1",
            ),
        ])

    agent = Agent(
        FunctionModel(scripted),
        capabilities=[LocalWorkspace(tmp_path), Shell(), jobs],
    )
    asyncio.run(agent.run("start"))
    assert _wait_ready(tmp_path)

    job_id = next(iter(session.states["shell_jobs"].job_ids))
    handle = tmp_path / ".pydantic-ai-harness" / "shell" / job_id / "handle"
    pid_text, group_text = handle.read_text().split()
    pid = int(pid_text)
    group = int(group_text)
    assert group > 0

    try:
        session.close_states()

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("process group survived close")

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError("supervisor process survived close")
    finally:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_switch_and_exit_stop_a_term_ignoring_job(tmp_path):
    import os
    import signal
    from dataclasses import replace

    import ds4py
    from ds4py import ThinkMode
    from ds4py._native.fake import FakeNative

    info = replace(
        FakeNative()._info,
        name="fake-ds4",
        family="deepseek4",
        is_deepseek41=False,
        is_glm_dsa=False,
        is_qwen4=False,
    )
    engine = ds4py.open("f.gguf", ctx=1024, think=ThinkMode.NONE, native=FakeNative(info=info))
    session = engine.session()
    jobs = DS4ShellJobs(session, tmp_path)
    command = "trap '' TERM; : > ready; while :; do sleep 1; done"

    def scripted(messages, info):
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("start_command", {"command": command}, tool_call_id="call-1"),
        ])

    def start():
        agent = Agent(
            FunctionModel(scripted),
            capabilities=[LocalWorkspace(tmp_path), Shell(), jobs],
        )
        asyncio.run(agent.run("start"))
        assert _wait_ready(tmp_path)
        job_id = next(iter(session.state("shell_jobs", lambda: None).job_ids))
        handle = tmp_path / ".pydantic-ai-harness" / "shell" / job_id / "handle"
        pid_text, group_text = handle.read_text().split()
        return int(pid_text), int(group_text)

    def wait_gone(pid, group):
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        raise AssertionError("process group survived session lifecycle close")

    groups = []
    try:
        pid, group = start()
        groups.append((pid, group))
        session.switch("b" * 32)
        wait_gone(pid, group)

        pid, group = start()
        groups.append((pid, group))
        session.close()
        wait_gone(pid, group)
    finally:
        for _, group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        engine.close()


def _wait_ready(workspace, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (workspace / "ready").exists():
            return True
        time.sleep(0.02)
    return False


def test_close_kills_background_child_after_wrapper_publishes_exit(tmp_path):
    import json
    import os
    import signal

    session = MockSession()
    jobs = DS4ShellJobs(session, tmp_path)
    command = (
        "(trap '' TERM; : > child-ready; while :; do sleep 1; done) & "
        "trap 'exit 0' TERM; : > ready; while :; do sleep 1; done"
    )

    def scripted(messages, info):
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("start_command", {"command": command}, tool_call_id="call-1"),
        ])

    agent = Agent(
        FunctionModel(scripted),
        capabilities=[LocalWorkspace(tmp_path), Shell(), jobs],
    )
    asyncio.run(agent.run("start"))
    assert _wait_ready(tmp_path)
    deadline = time.monotonic() + 5.0
    while not (tmp_path / "child-ready").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (tmp_path / "child-ready").exists()

    job_id = next(iter(session.states["shell_jobs"].job_ids))
    base = tmp_path / ".pydantic-ai-harness" / "shell" / job_id
    pid_text, group_text = (base / "handle").read_text().split()
    pid = int(pid_text)
    group = int(group_text)

    try:
        session.close_states()

        status = base / "status.json"
        assert status.exists()
        assert json.loads(status.read_text()).get("exit_code") is not None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("orphaned background child survived close")
    finally:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
