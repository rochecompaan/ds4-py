import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.tools import ToolDefinition

from ds4py.agent import build_agent_components


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

    def close_states(self) -> None:
        for state in self.states.values():
            if hasattr(state, "close"):
                state.close()
        self.states.clear()


def _has_tool_return(messages: Any) -> bool:
    return any(
        isinstance(part, ToolReturnPart)
        for message in messages
        for part in getattr(message, "parts", [])
    )


def _run_agent(
    tmp_path: Path,
    call: tuple[str, dict[str, Any]] | None = None,
    *,
    read_only: bool = False,
) -> tuple[Any, dict[str, ToolDefinition]]:
    exposed: dict[str, ToolDefinition] = {}

    def response(messages: Any, info: Any) -> ModelResponse:
        if call is None or _has_tool_return(messages):
            return ModelResponse(parts=[TextPart("done")])
        name, args = call
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id="call-1")])

    def scripted(messages: Any, info: Any) -> ModelResponse:
        exposed.update((tool.name, tool) for tool in info.function_tools)
        return response(messages, info)

    async def stream_function(messages: Any, info: Any) -> AsyncIterator[Any]:
        exposed.update((tool.name, tool) for tool in info.function_tools)
        if call is None or _has_tool_return(messages):
            yield "done"
        else:
            name, args = call
            yield {0: DeltaToolCall(name=name, json_args=json.dumps(args))}

    model: Any = FunctionModel(scripted, stream_function=stream_function)
    model.session = MockSession()
    components = build_agent_components(model, tmp_path, read_only=read_only)
    result = asyncio.run(components.agent.run("run the requested tool"))
    return result, exposed


def _read_result(tmp_path: Path, args: dict[str, Any]) -> str:
    result, _ = _run_agent(tmp_path, ("read_file", args))
    contents = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "read_file"
    ]
    assert len(contents) == 1
    assert isinstance(contents[0], str)
    return contents[0]


def test_read_schema_uses_configured_default() -> None:
    from ds4py.capabilities.file_system import prepare_file_tool_definitions

    definition = ToolDefinition(
        name="read_file",
        description="Read a file.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "default": 0},
                "limit": {
                    "type": "integer",
                    "default": None,
                    "description": "Maximum number of lines to return (default: 2000).",
                },
            },
            "required": ["path"],
        },
    )
    other = ToolDefinition(name="write_file", parameters_json_schema={"type": "object"})

    prepared = prepare_file_tool_definitions([definition, other], default_read_lines=77)

    schema = prepared[0].parameters_json_schema
    assert set(schema["properties"]) == {"path", "offset", "limit"}
    assert schema["required"] == ["path"]
    assert schema["properties"]["limit"]["default"] is None
    assert schema["properties"]["limit"]["description"] == (
        "Maximum number of lines to return (default: 77)."
    )
    assert prepared[1] is other


def test_agent_exposes_configured_read_schema(tmp_path: Path) -> None:
    _, exposed = _run_agent(tmp_path)

    schema = exposed["read_file"].parameters_json_schema
    assert set(schema["properties"]) == {"path", "offset", "limit"}
    assert schema["required"] == ["path"]
    assert schema["properties"]["limit"]["default"] is None
    assert schema["properties"]["limit"]["description"] == (
        "Maximum number of lines to return (default: 160)."
    )


def test_default_read_returns_160_lines_and_next_offset(tmp_path: Path) -> None:
    (tmp_path / "lines.txt").write_text(
        "".join(f"line-{number:03}\n" for number in range(1, 166))
    )

    content = _read_result(tmp_path, {"path": "lines.txt"})

    assert "line-160" in content
    assert "line-161" not in content
    assert "Use offset=160" in content


def test_explicit_read_limit_overrides_default(tmp_path: Path) -> None:
    (tmp_path / "lines.txt").write_text(
        "".join(f"line-{number:03}\n" for number in range(1, 166))
    )

    content = _read_result(tmp_path, {"path": "lines.txt", "limit": 20})

    assert "line-020" in content
    assert "line-021" not in content
    assert "Use offset=20" in content


def test_read_retains_50000_character_cap(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text(
        "".join(f"{number:03}-" + "x" * 1000 + "\n" for number in range(120))
    )

    content = _read_result(tmp_path, {"path": "large.txt"})

    assert len(content) <= 50_000
    assert "Use offset=" in content


def test_read_only_workspace_keeps_writable_tools_hidden(tmp_path: Path) -> None:
    _, exposed = _run_agent(tmp_path, read_only=True)

    assert "read_file" in exposed
    assert {"write_file", "edit_file", "create_directory"}.isdisjoint(exposed)
