import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.filesystem import FileReadEvent
from test_context_strategy import StubBinder

from ds4py.agent import build_agent_components
from ds4py.agent_prompt_pi import PI_CODING_AGENT_INSTRUCTIONS


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
        from ds4py.engine.lifecycle import SessionEvent

        event = SessionEvent(kind, "mock-session", payload)
        for listener, kinds in tuple(self.listeners):
            if kinds is None or kind in kinds:
                listener(event)


def run_profile(
    tmp_path: Path,
    calls: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    read_only: bool = False,
    system: str | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    shell_timeouts: list[Any] | None = None,
    events: list[Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    from pydantic_ai_harness.shell import ShellToolset

    turns = [*(calls or []), "done"]
    remaining = iter(turns)
    exposed = {}

    def next_response() -> ModelResponse:
        turn = next(remaining)
        if isinstance(turn, str):
            return ModelResponse(parts=[TextPart(turn)])
        name, args = turn
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=f"call-{len(exposed)}")])

    def scripted(messages: Any, info: Any) -> ModelResponse:
        exposed.update((tool.name, tool) for tool in info.function_tools)
        return next_response()

    async def stream_function(messages: Any, info: Any) -> AsyncIterator[Any]:
        exposed.update((tool.name, tool) for tool in info.function_tools)
        turn = next(remaining)
        if isinstance(turn, str):
            yield turn
        else:
            name, args = turn
            yield {0: DeltaToolCall(name=name, json_args=json.dumps(args))}

    if shell_timeouts is not None:
        assert monkeypatch is not None
        original = ShellToolset.run_command

        async def spy(self: Any, ctx: RunContext[object], command: str, **kwargs: Any) -> Any:
            shell_timeouts.append(kwargs.get("timeout_seconds"))
            return await original(self, ctx, command, **kwargs)

        monkeypatch.setattr(ShellToolset, "run_command", spy)

    model: Any = FunctionModel(scripted, stream_function=stream_function)
    model.session = MockSession()
    model.binder = StubBinder([0] * 100, transcript_tokens=0, ctx=65536)
    if events is not None:
        model.session.on_event(events.append)
    components = build_agent_components(
        model,
        tmp_path,
        read_only=read_only,
        system=system,
        tool_profile="pi",
    )
    result = asyncio.run(components.agent.run("run the requested tools"))
    return result, exposed


def returns(result: Any) -> list[Any]:
    return [
        part.content for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
    ]


def test_pi_definitions_match_reference_exactly(tmp_path: Path) -> None:
    _, exposed = run_profile(tmp_path)
    fixture = json.loads((Path(__file__).parents[1] / "fixtures" / "pi_profile.json").read_text())
    assert list(exposed) == ["read", "bash", "edit", "write"]
    observed = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_json_schema,
            "strict": tool.strict,
        }
        for tool in exposed.values()
    ]
    assert observed == fixture
    assert all(tool.tool_kind is None for tool in exposed.values())


def test_pi_instructions_name_only_the_four_tools(tmp_path: Path) -> None:
    result, _ = run_profile(tmp_path)
    first = result.all_messages()[0]
    assert isinstance(first, ModelRequest)
    assert first.instructions == PI_CODING_AGENT_INSTRUCTIONS
    for name in ("read_file", "write_file", "edit_file", "run_command", "start_command"):
        assert name not in PI_CODING_AGENT_INSTRUCTIONS
    for name in ("read", "bash", "edit", "write"):
        assert name in PI_CODING_AGENT_INSTRUCTIONS


def test_pi_system_override_wins(tmp_path: Path) -> None:
    result, _ = run_profile(tmp_path, system="custom rules")
    first = result.all_messages()[0]
    assert isinstance(first, ModelRequest)
    assert first.instructions == "custom rules"


@pytest.mark.parametrize(
    ("offset", "shown", "hidden"), [(0, "line-1", "line-2"), (1, "line-1", "line-2"), (2, "line-2", "line-1")]
)
def test_pi_read_offset_conversion(tmp_path: Path, offset: int, shown: str, hidden: str) -> None:
    (tmp_path / "lines.txt").write_text("line-1\nline-2\n")
    result, _ = run_profile(tmp_path, [("read", {"path": "lines.txt", "offset": offset, "limit": 1})])
    content = returns(result)[0]
    assert shown in content
    assert hidden not in content


def test_pi_read_defaults_to_2000_lines(tmp_path: Path) -> None:
    (tmp_path / "lines.txt").write_text("".join(f"line-{number}\n" for number in range(1, 2006)))
    result, _ = run_profile(tmp_path, [("read", {"path": "lines.txt"})])
    content = returns(result)[0]
    assert "line-2000" in content
    assert "line-2001" not in content
    assert "offset=2000" in content


def test_pi_write_and_nonincremental_batch_edit_round_trip(tmp_path: Path) -> None:
    result, _ = run_profile(
        tmp_path,
        [
            ("write", {"path": "data.txt", "content": "a\nb\nc\n"}),
            (
                "edit",
                {
                    "path": "data.txt",
                    "edits": [
                        {"oldText": "a\n", "newText": "b\n"},
                        {"oldText": "b\n", "newText": "z\n"},
                    ],
                },
            ),
        ],
    )
    assert (tmp_path / "data.txt").read_text() == "b\nz\nc\n"
    assert len(returns(result)) == 2


@pytest.mark.parametrize(
    ("original", "edits"),
    [
        ("a\na\n", [{"oldText": "a\n", "newText": "x\n"}]),
        (
            "abc",
            [
                {"oldText": "ab", "newText": "x"},
                {"oldText": "bc", "newText": "y"},
            ],
        ),
    ],
    ids=["ambiguous", "overlap"],
)
def test_pi_edit_refuses_ambiguity_and_overlap_without_writing(
    tmp_path: Path, original: str, edits: list[dict[str, str]]
) -> None:
    (tmp_path / "data.txt").write_text(original)
    run_profile(tmp_path, [("edit", {"path": "data.txt", "edits": edits})])
    assert (tmp_path / "data.txt").read_text() == original


def test_pi_bash_round_trip_and_timeout_forwarding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timeouts = []
    result, _ = run_profile(
        tmp_path,
        [("bash", {"command": "printf first"}), ("bash", {"command": "printf second", "timeout": 2})],
        monkeypatch=monkeypatch,
        shell_timeouts=timeouts,
    )
    assert timeouts == [None, 2]
    content = "\n".join(returns(result))
    assert "first" in content and "second" in content


def test_pi_read_only_hides_write_and_edit(tmp_path: Path) -> None:
    _, exposed = run_profile(tmp_path, read_only=True)
    assert list(exposed) == ["read"]


def test_pi_workspace_refuses_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.unlink(missing_ok=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_profile(workspace, [("write", {"path": "../outside.txt", "content": "no"})])
    assert not outside.exists()


def test_pi_uses_existing_result_text_and_file_events(tmp_path: Path) -> None:
    (tmp_path / "lines.txt").write_text("contents\n")
    events = []
    result, _ = run_profile(tmp_path, [("read", {"path": "lines.txt"})], events=events)
    content = returns(result)[0]
    assert "[lines.txt | 1 lines" in content
    assert "contents" in content
    assert any(isinstance(event.payload.get("event"), FileReadEvent) for event in events)
