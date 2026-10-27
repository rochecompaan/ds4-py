import asyncio

import pytest

from ds4py.capabilities.approvals import DS4Approvals, ApprovalDecision, _MENU
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel


def _has_tool_return(messages):
    return any(
        isinstance(p, ToolReturnPart)
        for m in messages for p in getattr(m, "parts", [])
    )


def test_deny_skips_tool(tmp_path):
    calls = []

    def scripted(messages, info):
        for m in messages:
            for p in getattr(m, "parts", []):
                if isinstance(p, ToolReturnPart):
                    return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("run_command", {"command": "echo hi"}, tool_call_id="call-1")])

    def run_command(command: str):
        calls.append(command)
        return "done"

    async def answerer(req):
        return "n"

    agent = Agent(
        FunctionModel(scripted),
        tools=[run_command],
        capabilities=[DS4Approvals(answerer, project_dir=tmp_path)],
    )
    result = asyncio.run(agent.run("do it"))
    assert calls == []
    denied_parts = [
        p for m in result.all_messages() for p in getattr(m, "parts", [])
        if isinstance(p, ToolReturnPart) and getattr(p, "outcome", None) == "denied"
    ]
    assert len(denied_parts) == 1


def test_yolo_bypasses(tmp_path):
    calls = []

    def scripted(messages, info):
        for m in messages:
            for p in getattr(m, "parts", []):
                if isinstance(p, ToolReturnPart):
                    return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("run_command", {"command": "echo hi"}, tool_call_id="call-1")])

    def run_command(command: str):
        calls.append(command)
        return "done"

    async def answerer(req):
        raise AssertionError("should not ask")

    agent = Agent(
        FunctionModel(scripted),
        tools=[run_command],
        capabilities=[DS4Approvals(answerer, yolo=True, project_dir=tmp_path)],
    )
    asyncio.run(agent.run("do it"))
    assert calls == ["echo hi"]


def test_outside_path_asks(tmp_path):
    outside = tmp_path.parent / "outside.txt"

    def scripted(messages, info):
        for m in messages:
            for p in getattr(m, "parts", []):
                if isinstance(p, ToolReturnPart):
                    return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("write_file", {"path": str(outside), "content": "x"}, tool_call_id="call-1"),
        ])

    def write_file(path: str, content: str):
        return "ok"

    asked = []

    async def answerer(req):
        asked.append(req)
        return "y"

    agent = Agent(
        FunctionModel(scripted),
        tools=[write_file],
        capabilities=[DS4Approvals(answerer, project_dir=tmp_path)],
    )
    asyncio.run(agent.run("write"))
    assert len(asked) == 1
    assert asked[0].path == outside.resolve()


def test_inside_path_does_not_ask(tmp_path):
    inside = tmp_path / "a.txt"

    def scripted(messages, info):
        for m in messages:
            for p in getattr(m, "parts", []):
                if isinstance(p, ToolReturnPart):
                    return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("write_file", {"path": str(inside), "content": "x"}, tool_call_id="call-1"),
        ])

    def write_file(path: str, content: str):
        return "ok"

    async def answerer(req):
        raise AssertionError("should not ask")

    agent = Agent(
        FunctionModel(scripted),
        tools=[write_file],
        capabilities=[DS4Approvals(answerer, project_dir=tmp_path)],
    )
    asyncio.run(agent.run("write"))


def test_menu_mapping():
    assert _MENU["y"] == ApprovalDecision.ONCE
    assert _MENU["n"] == ApprovalDecision.DENY
    assert _MENU["t"] == ApprovalDecision.TOOL
    assert _MENU["a"] == ApprovalDecision.PATTERN
    assert _MENU["s"] == ApprovalDecision.STOP


def test_eof_denies(tmp_path):
    def scripted(messages, info):
        for m in messages:
            for p in getattr(m, "parts", []):
                if isinstance(p, ToolReturnPart):
                    return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("run_command", {"command": "echo hi"}, tool_call_id="call-1")])

    def run_command(command: str):
        return "done"

    def answerer(req):
        raise EOFError

    agent = Agent(
        FunctionModel(scripted),
        tools=[run_command],
        capabilities=[DS4Approvals(answerer, project_dir=tmp_path)],
    )
    result = asyncio.run(agent.run("do it"))
    denied_parts = [
        p for m in result.all_messages() for p in getattr(m, "parts", [])
        if isinstance(p, ToolReturnPart) and getattr(p, "outcome", None) == "denied"
    ]
    assert len(denied_parts) == 1


def test_yolo_still_respects_read_only_workspace(tmp_path):
    from pydantic_ai.capabilities import LocalWorkspace
    from pydantic_ai.workspaces import ReadOnlyWorkspace
    from pydantic_ai_harness import FileSystem, Shell

    class VisibleReadOnlyWorkspace(ReadOnlyWorkspace):
        # Keep the real FileSystem/Shell tools visible so their calls reach the
        # read-only backend instead of being rejected as unknown tool names.
        @property
        def read_only(self):
            return False

    class VisibleReadOnlyLocalWorkspace(LocalWorkspace):
        def get_workspace(self, ctx, *, ref=None):
            workspace = super().get_workspace(ctx, ref=ref)
            return VisibleReadOnlyWorkspace(workspace.wrapped) if workspace is not None else None

    def scripted(messages, info):
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("write_file", {"path": "a.txt", "content": "x"}, tool_call_id="call-1"),
            ToolCallPart(
                "run_command",
                {"command": "echo leaked > leaked.txt"},
                tool_call_id="call-2",
            ),
        ])

    async def answerer(req):
        raise AssertionError("should not ask")

    agent = Agent(
        FunctionModel(scripted),
        capabilities=[
            VisibleReadOnlyLocalWorkspace(tmp_path, read_only=True),
            FileSystem(root_dir=tmp_path, content_hashes=True),
            Shell(),
            DS4Approvals(answerer, yolo=True, project_dir=tmp_path),
        ],
    )
    result = asyncio.run(agent.run("write and run"))

    returned = [
        p for m in result.all_messages() for p in getattr(m, "parts", [])
        if isinstance(p, ToolReturnPart) and p.tool_name in ("write_file", "run_command")
    ]
    assert {p.tool_name for p in returned} == {"write_file", "run_command"}
    assert all(p.outcome == "failed" for p in returned)
    assert all("read-only" in str(p.content) for p in returned)
    assert not (tmp_path / "a.txt").exists()
    assert not (tmp_path / "leaked.txt").exists()


def test_tool_decision_remembers_tool(tmp_path):
    asked = []

    def scripted(messages, info):
        for m in messages:
            for p in getattr(m, "parts", []):
                if isinstance(p, ToolReturnPart):
                    return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("run_command", {"command": "echo hi"}, tool_call_id="call-1")])

    def run_command(command: str):
        return "done"

    async def answerer(req):
        asked.append(req)
        return "t"

    agent = Agent(
        FunctionModel(scripted),
        tools=[run_command],
        capabilities=[DS4Approvals(answerer, project_dir=tmp_path)],
    )
    asyncio.run(agent.run("do it"))
    assert len(asked) == 1
    # Second run should not ask for the same tool.
    asyncio.run(agent.run("do it again"))
    assert len(asked) == 1


def test_pattern_decision_remembers_first_word(tmp_path):
    asked = []

    def scripted(messages, info):
        for m in messages:
            for p in getattr(m, "parts", []):
                if isinstance(p, ToolReturnPart):
                    return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("run_command", {"command": "git status"}, tool_call_id="call-1")])

    def run_command(command: str):
        return "done"

    async def answerer(req):
        asked.append(req)
        return "a"

    agent = Agent(
        FunctionModel(scripted),
        tools=[run_command],
        capabilities=[DS4Approvals(answerer, project_dir=tmp_path)],
    )
    asyncio.run(agent.run("run git"))
    assert len(asked) == 1
    asyncio.run(agent.run("run git again"))
    assert len(asked) == 1


def test_stop_cancels_session(tmp_path):
    from pydantic_ai.exceptions import RunCancelled

    calls = []

    def scripted(messages, info):
        for m in messages:
            for p in getattr(m, "parts", []):
                if isinstance(p, ToolReturnPart):
                    return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("run_command", {"command": "echo hi"}, tool_call_id="call-1")])

    def run_command(command: str):
        calls.append(command)
        return "done"

    async def answerer(req):
        return "s"

    agent = Agent(
        FunctionModel(scripted),
        tools=[run_command],
        capabilities=[DS4Approvals(answerer, project_dir=tmp_path)],
    )
    with pytest.raises(RunCancelled):
        asyncio.run(agent.run("do it"))
    assert calls == []


def test_relative_write_inside_project_does_not_ask(tmp_path):
    from pydantic_ai.capabilities import LocalWorkspace
    from pydantic_ai_harness import FileSystem

    def scripted(messages, info):
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("write_file", {"path": "a.txt", "content": "hello"}, tool_call_id="call-1"),
        ])

    asked = []

    async def answerer(req):
        asked.append(req)
        return "n"

    agent = Agent(
        FunctionModel(scripted),
        capabilities=[
            LocalWorkspace(tmp_path),
            FileSystem(root_dir=tmp_path, cwd=tmp_path, content_hashes=True),
            DS4Approvals(answerer, project_dir=tmp_path),
        ],
    )
    asyncio.run(agent.run("write"))

    assert asked == []
    assert (tmp_path / "a.txt").read_text() == "hello"


def test_default_answerer_denies_safely(tmp_path):
    from ds4py.agent import build_agent_components

    class MockSession:
        def __init__(self):
            self.states = {}
            self.listeners = []

        def state(self, key, factory):
            if key not in self.states:
                self.states[key] = factory()
            return self.states[key]

        def on_event(self, listener, kinds=None):
            self.listeners.append((listener, kinds))

    def scripted(messages, info):
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart(
                "run_command",
                {"command": "echo leaked > leaked.txt"},
                tool_call_id="call-1",
            ),
        ])

    async def stream_function(messages, info):
        if _has_tool_return(messages):
            yield "done"
        else:
            yield {0: DeltaToolCall(name="run_command", json_args='{"command": "echo leaked > leaked.txt"}')}

    model = FunctionModel(scripted, stream_function=stream_function)
    model.session = MockSession()
    components = build_agent_components(model, tmp_path)
    agent = Agent(
        FunctionModel(scripted, stream_function=stream_function),
        capabilities=components.components,
    )

    result = asyncio.run(agent.run("run it"))

    denied = [
        p for m in result.all_messages() for p in getattr(m, "parts", [])
        if isinstance(p, ToolReturnPart) and getattr(p, "outcome", None) == "denied"
    ]
    assert len(denied) == 1
    assert not (tmp_path / "leaked.txt").exists()


def test_malformed_quoting_asks_instead_of_aborting(tmp_path):
    asked = []
    calls = []
    command = 'echo "unterminated'

    def scripted(messages, info):
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("run_command", {"command": command}, tool_call_id="call-1"),
        ])

    def run_command(command: str):
        calls.append(command)
        return "done"

    async def answerer(req):
        asked.append(req)
        return "y"

    agent = Agent(
        FunctionModel(scripted),
        tools=[run_command],
        capabilities=[DS4Approvals(answerer, project_dir=tmp_path)],
    )
    asyncio.run(agent.run("do it"))
    assert len(asked) == 1
    assert calls == [command]


def test_malformed_quoting_does_not_set_a_reusable_pattern(tmp_path):
    asked = []
    command = 'echo "unterminated'

    def scripted(messages, info):
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("run_command", {"command": command}, tool_call_id="call-1"),
        ])

    def run_command(command: str):
        return "done"

    async def answerer(req):
        asked.append(req)
        return "a"

    agent = Agent(
        FunctionModel(scripted),
        tools=[run_command],
        capabilities=[DS4Approvals(answerer, project_dir=tmp_path)],
    )
    asyncio.run(agent.run("do it"))
    asyncio.run(agent.run("do it again"))
    assert len(asked) == 2
