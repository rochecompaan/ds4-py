from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ds4py.context_paging.history import PAGING_TOOL_NAMES, HistoryProjection, UserHistoryItem, project_history
from ds4py.context_paging.public_history import (
    PublicModelTurn,
    PublicText,
    PublicToolCall,
    PublicUser,
    project_public_history,
)
from ds4py.context_paging.recovery_parts import public_item_value
from ds4py.message_metadata import DS4_KEY

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class UnknownPart:
    content: str


def _turn(parts: Sequence[object], results: Sequence[ModelRequest] | None = None) -> tuple[ModelMessage, ...]:
    response = ModelResponse(cast("Any", parts), metadata={DS4_KEY: "turn"})
    messages: list[ModelMessage] = [response, *(results or [])]
    return tuple(messages)


def _result(call_id: str, name: str = "read", **kwargs: Any) -> ModelRequest:
    return ModelRequest([ToolReturnPart(name, f"result for {call_id}", tool_call_id=call_id, **kwargs)])


def _public_turn(messages: tuple[ModelMessage, ...]) -> PublicModelTurn:
    public = project_public_history(project_history(messages))
    assert len(public) == 1
    item = public[0]
    assert isinstance(item, PublicModelTurn)
    return item


def test_multipart_user_text_and_image_preserve_baseline_blocks() -> None:
    request = ModelRequest(
        [
            UserPromptPart(
                ["Inspect this", BinaryContent(b"IMAGEDATA", media_type="image/png")],
            )
        ],
        metadata={DS4_KEY: "user-image"},
    )

    history = HistoryProjection((request,), (UserHistoryItem("user-image", 0, request, 0, ""),))
    public = project_public_history(history)

    assert len(public) == 1
    item = public[0]
    assert isinstance(item, PublicUser)
    assert item.content == (
        {"type": "text", "text": "Inspect this"},
        {"type": "image", "data": "SU1BR0VEQVRB", "mimeType": "image/png"},
    )
    assert public_item_value(item)["userMessage"] == {
        "role": "user",
        "content": list(item.content),
        "timestamp": item.message_timestamp,
    }


def test_public_content_preserves_original_indices_and_removes_provider_private_values() -> None:
    ordinary = ToolCallPart(
        "read",
        {"path": "src/public.ts", "thinking": "application-owned-value", "nested": {"file_path": "src/nested.ts"}},
        tool_call_id="ordinary",
        provider_details={"thoughtSignature": "private-tool-signature"},
    )
    messages = _turn(
        [
            ThinkingPart("private-reasoning", signature="private-signature"),
            TextPart("Public answer", provider_details={"signature": "private-text-signature"}),
            ordinary,
            ToolCallPart("load_history", {"hidden": "paging-argument"}, tool_call_id="paging"),
            UnknownPart("private-unknown-part"),
        ],
        [
            _result("ordinary"),
            _result("paging", "load_history"),
        ],
    )
    before = deepcopy(messages)

    turn = _public_turn(messages)

    assert turn.content == (
        None,
        PublicText("Public answer"),
        PublicToolCall("ordinary", "read", ordinary.args, 2),
        None,
        None,
    )
    assert tuple(result.tool_call_id for result in turn.tool_results) == ("ordinary",)
    assert turn.tools == ("read",)
    assert turn.files == ("src/public.ts", "src/nested.ts")
    assert turn.failed is False
    public_text = repr(turn)
    for private in (
        "private-reasoning",
        "private-signature",
        "private-text-signature",
        "private-tool-signature",
        "paging-argument",
        "result for paging",
    ):
        assert private not in public_text
    assert "application-owned-value" in public_text
    assert messages == before


def test_public_arguments_are_isolated_from_canonical_messages() -> None:
    messages = _turn(
        [ToolCallPart("read", {"nested": {"path": "before"}}, tool_call_id="call")],
        [_result("call")],
    )

    turn = _public_turn(messages)
    call = turn.content[0]
    assert isinstance(call, PublicToolCall)
    arguments = cast("dict[str, Any]", call.arguments)
    nested = cast("dict[str, str]", arguments["nested"])
    nested["path"] = "after"

    response = cast("ModelResponse", messages[0])
    original_call = cast("ToolCallPart", response.parts[0])
    assert original_call.args == {"nested": {"path": "before"}}


def test_paging_only_turns_disappear() -> None:
    calls = [ToolCallPart(name, {}, tool_call_id=f"call-{name}") for name in PAGING_TOOL_NAMES]
    results = [_result(f"call-{name}", name) for name in PAGING_TOOL_NAMES]

    assert project_public_history(project_history(_turn(calls, results))) == ()


def test_mixed_turn_retains_ordinary_work_and_recomputes_failure_metadata() -> None:
    nested = {"complete": True, "calls": [{"id": "nested/1", "name": "load_history", "status": "ok"}]}
    messages = _turn(
        [
            ThinkingPart("private"),
            ToolCallPart("codemode", {"code": "hiddenScript()"}, tool_call_id="wrapper"),
            TextPart("Public decision"),
            ToolCallPart("read", {"path": "src/app.ts"}, tool_call_id="ordinary"),
        ],
        [
            _result("wrapper", "codemode", metadata={"nestedCalls": nested}, outcome="failed"),
            _result("ordinary"),
        ],
    )

    turn = _public_turn(messages)

    assert [type(part).__name__ if part is not None else None for part in turn.content] == [
        None,
        None,
        "PublicText",
        "PublicToolCall",
    ]
    assert turn.tools == ("read",)
    assert turn.files == ("src/app.ts",)
    assert turn.failed is False
    assert "hiddenScript" not in repr(turn)
    assert "result for wrapper" not in repr(turn)
    assert "result for ordinary" in repr(turn)


@pytest.mark.parametrize(
    "nested",
    [
        {"complete": True, "calls": [{"id": "n1", "name": "load_history", "status": "ok"}]},
        {"complete": True, "calls": [{"id": "n1", "name": "search_history", "status": "error"}]},
        {"complete": True, "calls": [{"id": "n1", "name": "browse_history", "status": "unfinished"}]},
    ],
)
def test_complete_all_paging_nested_wrappers_disappear(nested: dict[str, object]) -> None:
    messages = _turn(
        [ToolCallPart("custom-wrapper", {"code": "hiddenScript()"}, tool_call_id="wrapper")],
        [_result("wrapper", "custom-wrapper", metadata={"nestedCalls": nested})],
    )

    assert project_public_history(project_history(messages)) == ()


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"nestedCalls": None},
        {"nestedCalls": {"complete": False, "calls": [{"id": "n1", "name": "load_history", "status": "ok"}]}},
        {"nestedCalls": {"complete": True, "calls": []}},
        {
            "nestedCalls": {
                "complete": True,
                "calls": [
                    {"id": "n1", "name": "load_history", "status": "ok"},
                    {"id": "n2", "name": "read", "status": "ok"},
                ],
            }
        },
        {"nestedCalls": {"complete": True, "calls": [{"id": "n1", "name": "load_history", "status": "future"}]}},
    ],
)
def test_malformed_mixed_or_absent_nested_provenance_stays_public(metadata: object) -> None:
    messages = _turn(
        [ToolCallPart("codemode", {"code": "dynamicScript()"}, tool_call_id="wrapper")],
        [_result("wrapper", "codemode", metadata=metadata)],
    )

    turn = _public_turn(messages)

    assert isinstance(turn.content[0], PublicToolCall)
    assert turn.tool_results[0].content == [{"type": "text", "text": "result for wrapper"}]


@pytest.mark.parametrize("outcome", ["failed", "denied", "interrupted"])
def test_tool_failure_outcomes_are_public_failures(outcome: str) -> None:
    messages = _turn(
        [ToolCallPart("read", {}, tool_call_id="call")],
        [_result("call", outcome=cast("Any", outcome))],
    )

    assert _public_turn(messages).failed is True


def test_retry_prompt_is_a_public_failure() -> None:
    response = ModelResponse([ToolCallPart("read", {}, tool_call_id="call")], metadata={DS4_KEY: "turn"})
    retry = ModelRequest([RetryPromptPart("Retry prompt: correct input", tool_name="read", tool_call_id="call")])

    turn = _public_turn((response, retry))

    assert turn.failed is True
    assert turn.tool_results[0].content == [{"type": "text", "text": "Retry prompt: correct input"}]
