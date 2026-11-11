import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai_harness import AskUser
from pydantic_ai_harness.ask_user import (
    AskUserAnswer,
    AskUserRequest,
    AskUserResponse,
    Question,
    QuestionOption,
)
from test_context_strategy import RecordingOwner, StubBinder

import ds4py
from ds4py._native.fake import FakeNative
from ds4py.agent import build_agent_components
from ds4py.capabilities.compaction import LLMCompaction
from ds4py.capabilities.context import ContextComposition, ContextStrategy
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.sampling import SamplingPolicy
from ds4py.model import DS4Model


class MockSession:
    def __init__(self) -> None:
        self.states = {}
        self.listeners = []

    def state(self, key: Any, factory: Any) -> Any:
        if key not in self.states:
            self.states[key] = factory()
        return self.states[key]

    def on_event(self, listener: Any, kinds: Any = None) -> None:
        self.listeners.append((listener, kinds))

    def emit(self, kind: str, **payload: Any) -> None:
        event = SessionEvent(kind, "mock-session", payload)
        for listener, kinds in tuple(self.listeners):
            if kinds is None or kind in kinds:
                listener(event)

    def close_states(self) -> None:
        for state in self.states.values():
            if hasattr(state, "close"):
                state.close()
        self.states.clear()


def _has_tool_return(messages: Any) -> bool:
    return any(isinstance(p, ToolReturnPart) for m in messages for p in getattr(m, "parts", []))


def test_build_agent_components_behavior(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("world")

    def scripted(messages: Any, info: Any) -> ModelResponse:
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart("read_file", {"path": "hello.txt"}, tool_call_id="call-1"),
            ]
        )

    async def stream_function(messages: Any, info: Any) -> AsyncIterator[Any]:
        if _has_tool_return(messages):
            yield "done"
        else:
            yield {0: DeltaToolCall(name="read_file", json_args='{"path": "hello.txt"}')}

    base_model: Any = FunctionModel(scripted, stream_function=stream_function)
    base_model.session = MockSession()
    base_model.binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
    components = build_agent_components(base_model, tmp_path, interactive=False)

    result = asyncio.run(components.agent.run("read hello"))

    values = [p for m in result.all_messages() for p in getattr(m, "parts", []) if isinstance(p, ToolReturnPart)]
    assert any("world" in str(v.content) for v in values)

    events_file = tmp_path / "events.jsonl"
    assert events_file.exists()
    assert len(events_file.read_text().strip().splitlines()) > 0


def test_factory_runs_shell_commands_without_an_approval_policy(tmp_path: Path) -> None:
    def scripted(messages: Any, info: Any) -> ModelResponse:
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "run_command",
                    {"command": "printf unblocked > command-ran.txt"},
                    tool_call_id="call-1",
                )
            ]
        )

    async def stream_function(messages: Any, info: Any) -> AsyncIterator[Any]:
        if _has_tool_return(messages):
            yield "done"
        else:
            yield {
                0: DeltaToolCall(
                    name="run_command",
                    json_args='{"command": "printf unblocked > command-ran.txt"}',
                )
            }

    model: Any = FunctionModel(scripted, stream_function=stream_function)
    model.session = MockSession()
    model.binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
    components = build_agent_components(model, tmp_path)

    asyncio.run(components.agent.run("run it"))

    assert (tmp_path / "command-ran.txt").read_text() == "unblocked"


def test_factory_does_not_expose_list_directory(tmp_path: Path) -> None:
    exposed: set[str] = set()

    def scripted(messages: Any, info: Any) -> ModelResponse:
        exposed.update(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart("done")])

    async def stream_function(messages: Any, info: Any) -> AsyncIterator[Any]:
        exposed.update(tool.name for tool in info.function_tools)
        yield "done"

    model: Any = FunctionModel(scripted, stream_function=stream_function)
    model.session = MockSession()
    model.binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
    components = build_agent_components(model, tmp_path)

    asyncio.run(components.agent.run("inspect tools"))

    assert "list_directory" not in exposed
    assert {
        "read_file",
        "write_file",
        "edit_file",
        "search_files",
        "find_files",
        "create_directory",
        "file_info",
        "run_command",
    } <= exposed


def test_factory_shell_lists_hidden_files(tmp_path: Path) -> None:
    (tmp_path / ".meta").mkdir()
    (tmp_path / ".meta" / "example.py").write_text("example = True\n")

    def scripted(messages: Any, info: Any) -> ModelResponse:
        if _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "run_command",
                    {"command": "ls -la .meta"},
                    tool_call_id="call-1",
                )
            ]
        )

    async def stream_function(messages: Any, info: Any) -> AsyncIterator[Any]:
        if _has_tool_return(messages):
            yield "done"
        else:
            yield {
                0: DeltaToolCall(
                    name="run_command",
                    json_args='{"command": "ls -la .meta"}',
                )
            }

    model: Any = FunctionModel(scripted, stream_function=stream_function)
    model.session = MockSession()
    model.binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
    components = build_agent_components(model, tmp_path)

    result = asyncio.run(components.agent.run("list hidden examples"))

    listings = [
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "run_command"
    ]
    assert any("example.py" in listing for listing in listings), listings


def test_factory_applies_default_coding_instructions(tmp_path: Path) -> None:
    from ds4py.agent_prompt import DEFAULT_CODING_AGENT_INSTRUCTIONS

    def scripted(messages: Any, info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    async def stream_function(messages: Any, info: Any) -> AsyncIterator[Any]:
        yield "done"

    model: Any = FunctionModel(scripted, stream_function=stream_function)
    model.session = MockSession()
    model.binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
    components = build_agent_components(model, tmp_path)

    result = asyncio.run(components.agent.run("hello"))

    first_message = result.all_messages()[0]
    assert isinstance(first_message, ModelRequest)
    assert first_message.instructions == DEFAULT_CODING_AGENT_INSTRUCTIONS


def test_factory_applies_persistent_instructions(tmp_path: Path) -> None:
    from pydantic_ai_harness import StepPersistence

    def scripted(messages: Any, info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    async def stream_function(messages: Any, info: Any) -> AsyncIterator[Any]:
        yield "done"

    model: Any = FunctionModel(scripted, stream_function=stream_function)
    model.session = MockSession()
    model.binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
    components = build_agent_components(
        model,
        tmp_path,
        system="persistent rules",
    )

    assert not any(isinstance(c, StepPersistence) for c in components.components)
    result = asyncio.run(components.agent.run("hello"))

    first_message = result.all_messages()[0]
    assert isinstance(first_message, ModelRequest)
    assert first_message.instructions == "persistent rules"


def test_no_ask_user_unless_interactive(tmp_path: Path) -> None:
    def scripted(messages: Any, info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    async def stream_function(messages: Any, info: Any) -> AsyncIterator[Any]:
        yield "done"

    model: Any = FunctionModel(scripted, stream_function=stream_function)
    model.session = MockSession()
    model.binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
    components = build_agent_components(model, tmp_path, interactive=False)
    assert not any(isinstance(c, AskUser) for c in components.components)

    model2: Any = FunctionModel(scripted, stream_function=stream_function)
    model2.session = MockSession()
    model2.binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)

    async def question_answerer(request: AskUserRequest) -> AskUserResponse:
        return AskUserResponse(
            answers=tuple(
                AskUserAnswer(header=question.header, selected=(question.options[0].label,))
                for question in request.questions
            )
        )

    components2 = build_agent_components(
        model2,
        tmp_path,
        interactive=True,
        question_answerer=question_answerer,
    )
    assert any(isinstance(c, AskUser) for c in components2.components)


def test_question_callback_has_a_typed_contract(tmp_path: Path) -> None:
    def scripted(messages: Any, info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    async def stream_function(messages: Any, info: Any) -> AsyncIterator[Any]:
        yield "done"

    model: Any = FunctionModel(scripted, stream_function=stream_function)
    model.session = MockSession()
    model.binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)

    async def question_answerer(request: AskUserRequest) -> AskUserResponse:
        assert isinstance(request, AskUserRequest)
        return AskUserResponse(
            answers=tuple(
                AskUserAnswer(header=question.header, selected=(question.options[0].label,))
                for question in request.questions
            )
        )

    components = build_agent_components(
        model,
        tmp_path,
        interactive=True,
        question_answerer=question_answerer,
    )
    questions = next(c for c in components.components if isinstance(c, AskUser))
    request = AskUserRequest(
        questions=(
            Question(
                header="next",
                question="What next?",
                options=(QuestionOption(label="one"), QuestionOption(label="two")),
            ),
        )
    )

    response = asyncio.run(cast("Any", questions.answerer)(request))

    assert questions.answerer is question_answerer
    assert isinstance(response, AskUserResponse)
    assert response.answers[0].selected == ("one",)


def test_factory_registers_exactly_one_selected_owner(tmp_path: Path) -> None:
    with ds4py.open("f.gguf", ctx=65536, native=FakeNative()) as engine:
        model = DS4Model(engine.session(), sampling=SamplingPolicy(seed=1))
        components = build_agent_components(model, tmp_path)
        owners = [item for item in components.components if isinstance(item, ContextStrategy)]
        assert owners == [components.compactor]
        assert model.context_owner is components.compactor
        assert isinstance(components.compactor, LLMCompaction)


def test_custom_owner_replaces_only_automatic_compaction(tmp_path: Path) -> None:
    with ds4py.open("f.gguf", ctx=65536, native=FakeNative()) as engine:
        model = DS4Model(engine.session(), sampling=SamplingPolicy(seed=1))
        owner = RecordingOwner(model.binder)
        components = build_agent_components(model, tmp_path, context=ContextComposition(owner))
        assert model.context_owner is owner
        assert [item for item in components.components if isinstance(item, ContextStrategy)] == [owner]
        assert isinstance(components.compactor, LLMCompaction)
        assert components.compactor is not owner
