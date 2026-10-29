import asyncio

from ds4py.agent import build_agent_components
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai_harness import AskUser


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


def test_build_agent_components_behavior(tmp_path):
    (tmp_path / "hello.txt").write_text("world")

    def scripted(messages, info):
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("read_file", {"path": "hello.txt"}, tool_call_id="call-1"),
        ])

    async def stream_function(messages, info):
        if _has_tool_return(messages):
            yield "done"
        else:
            yield {0: DeltaToolCall(name="read_file", json_args='{"path": "hello.txt"}')}

    base_model = FunctionModel(scripted, stream_function=stream_function)
    base_model.session = MockSession()
    components = build_agent_components(base_model, tmp_path, interactive=False)

    result = asyncio.run(components.agent.run("read hello"))

    values = [
        p for m in result.all_messages() for p in getattr(m, "parts", [])
        if isinstance(p, ToolReturnPart)
    ]
    assert any("world" in str(v.content) for v in values)

    events_file = tmp_path / "events.jsonl"
    assert events_file.exists()
    assert len(events_file.read_text().strip().splitlines()) > 0


def test_factory_wires_yolo_and_persistent_instructions(tmp_path):
    from ds4py.capabilities.approvals import DS4Approvals
    from pydantic_ai_harness import StepPersistence

    def scripted(messages, info):
        return ModelResponse(parts=[TextPart("done")])

    async def stream_function(messages, info):
        yield "done"

    model = FunctionModel(scripted, stream_function=stream_function)
    model.session = MockSession()
    components = build_agent_components(
        model, tmp_path, yolo=True, system="persistent rules",
    )

    approvals = next(c for c in components.components if isinstance(c, DS4Approvals))
    assert not any(isinstance(c, StepPersistence) for c in components.components)
    result = asyncio.run(components.agent.run("hello"))

    assert approvals.yolo is True
    assert result.all_messages()[0].instructions == "persistent rules"


def test_no_ask_user_unless_interactive(tmp_path):
    def scripted(messages, info):
        return ModelResponse(parts=[TextPart("done")])

    async def stream_function(messages, info):
        yield "done"

    model = FunctionModel(scripted, stream_function=stream_function)
    model.session = MockSession()
    components = build_agent_components(model, tmp_path, interactive=False)
    assert not any(isinstance(c, AskUser) for c in components.components)

    model2 = FunctionModel(scripted, stream_function=stream_function)
    model2.session = MockSession()
    components2 = build_agent_components(
        model2, tmp_path, interactive=True, answerer=lambda req: "y",
    )
    assert any(isinstance(c, AskUser) for c in components2.components)
