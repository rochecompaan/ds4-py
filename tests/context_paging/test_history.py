import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from ds4py.context_paging.history import HistoryProjectionError, ModelTurnHistoryItem, project_history
from ds4py.context_paging.public_history import PublicModelTurn, project_public_history
from ds4py.context_paging.recovery_parts import public_item_value
from ds4py.message_metadata import DS4_KEY, DS4_TIMESTAMP

FIXTURES = json.loads((Path(__file__).parents[1] / "fixtures/context_paging/history.json").read_text())["fixtures"]


def _response_content(block: dict[str, Any]) -> object:
    kind = block["type"]
    if kind == "text":
        return TextPart(block["text"], provider_details={"signature": block.get("textSignature")})
    if kind == "thinking":
        return ThinkingPart(
            block["thinking"],
            signature=block.get("thinkingSignature"),
            provider_details={"redacted": block.get("redacted")},
        )
    if kind == "toolCall":
        return ToolCallPart(
            block["name"],
            block.get("arguments", {}),
            tool_call_id=block["id"],
            provider_details={"thoughtSignature": block.get("thoughtSignature")},
        )
    return object()


def _timestamp(milliseconds: int) -> datetime:
    return datetime.fromtimestamp(milliseconds / 1000, UTC)


def _user_content(content: object) -> object:
    if isinstance(content, str):
        return content
    return [
        part["text"]
        if part["type"] == "text"
        else BinaryContent(base64.b64decode(part["data"]), media_type=part["mimeType"])
        for part in cast("list[dict[str, Any]]", content)
    ]


def _message(entry: dict[str, Any]) -> ModelRequest | ModelResponse:
    raw = entry["message"]
    role = raw["role"]
    metadata = {DS4_KEY: entry["id"], DS4_TIMESTAMP: entry["timestamp"]}
    timestamp = _timestamp(raw["timestamp"])
    if role == "user":
        return ModelRequest(
            [UserPromptPart(cast("Any", _user_content(raw["content"])), timestamp=timestamp)],
            metadata=metadata,
            timestamp=timestamp,
        )
    if role == "assistant":
        parts = [_response_content(block) for block in raw["content"]]
        stop_reason = raw["stopReason"]
        finish_reason = {
            "stop": "stop",
            "toolUse": "tool_call",
            "length": "length",
            "aborted": "error",
            "error": "error",
        }[stop_reason]
        usage = raw["usage"]
        return ModelResponse(
            cast("Any", parts),
            metadata=metadata,
            finish_reason=cast("Any", finish_reason),
            state="incomplete" if stop_reason in {"aborted", "length"} else "complete",
            timestamp=timestamp,
            provider_name=raw["provider"],
            model_name=raw["model"],
            usage=RequestUsage(
                input_tokens=usage["input"],
                output_tokens=usage["output"],
                cache_read_tokens=usage["cacheRead"],
                cache_write_tokens=usage["cacheWrite"],
            ),
        )
    if role == "toolResult":
        content = [part.get("text", "") for part in raw["content"]]
        text = "\n".join(content)
        call_id = raw["toolCallId"]
        if text.startswith("Retry prompt:"):
            part: object = RetryPromptPart(text, tool_name=raw["toolName"], tool_call_id=call_id, timestamp=timestamp)
        else:
            outcome = "denied" if "denied" in text.lower() else "failed" if raw["isError"] else "success"
            result_metadata = {key: raw[key] for key in ("nestedCalls", "details") if key in raw}
            part = ToolReturnPart(
                raw["toolName"],
                content,
                tool_call_id=call_id,
                outcome=cast("Any", outcome),
                metadata=result_metadata or None,
                timestamp=timestamp,
            )
        return ModelRequest([cast("Any", part)], metadata=metadata, timestamp=timestamp)
    raise AssertionError(f"unsupported fixture role {role}")


def _messages(fixture: dict[str, Any]) -> list[ModelRequest | ModelResponse]:
    return [_message(entry) for entry in fixture["input"]["entries"]]


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture["id"])
def test_generated_history_contract(fixture: dict[str, Any]) -> None:
    messages = _messages(fixture)
    expected = fixture["expected"]
    options = fixture["input"]["options"]
    restore_boundary = "switched" if options.get("recoverIncomplete") else None

    if "error" in expected["projection"]:
        with pytest.raises(HistoryProjectionError) as caught:
            project_history(messages, restore_boundary=cast("Any", restore_boundary))
        assert caught.value.code == expected["projection"]["error"]["code"]
        return

    projection = project_history(messages, restore_boundary=cast("Any", restore_boundary))
    public = project_public_history(projection)
    expected_public = expected["public"]
    assert [public_item_value(item) for item in public] == [item for item in expected_public if item is not None]
    assert [item.history_id for item in public] == [item["id"] for item in expected_public if item is not None]
    assert [item.failed for item in public if isinstance(item, PublicModelTurn)] == [
        item["metadata"]["failed"] for item in expected_public if item is not None and item["kind"] == "modelTurn"
    ]


def test_newest_incomplete_turn_is_retained_for_selection_but_not_public() -> None:
    fixture = next(item for item in FIXTURES if item["id"] == "history-newest-incomplete-tail-omitted")

    projection = project_history(_messages(fixture))
    turn = projection.items[-1]

    assert isinstance(turn, ModelTurnHistoryItem)
    assert turn.complete is False
    assert turn.failed is False
    assert turn.omit_from_public is True
    assert turn.exchanges[0].result is None
    assert all(item.history_id != turn.history_id for item in project_public_history(projection))


@pytest.mark.parametrize("restore_boundary", ["switched", "forked"])
def test_restore_boundaries_classify_saved_incomplete_exchanges(restore_boundary: str) -> None:
    fixture = next(item for item in FIXTURES if item["id"] == "history-restore-incomplete-retained")

    projection = project_history(_messages(fixture), restore_boundary=cast("Any", restore_boundary))
    turn = projection.items[0]

    assert isinstance(turn, ModelTurnHistoryItem)
    assert turn.interrupted_by_restore is True
    assert turn.failed is True


def test_live_refresh_does_not_classify_an_older_incomplete_exchange_as_restored() -> None:
    fixture = next(item for item in FIXTURES if item["id"] == "history-restore-incomplete-retained")

    with pytest.raises(HistoryProjectionError) as caught:
        project_history(_messages(fixture), restore_boundary=cast("Any", "refreshed"))

    assert caught.value.code == "INCOMPLETE_TOOL_RESULTS"


def test_length_capped_incomplete_response_is_not_a_failure() -> None:
    response = ModelResponse(
        [TextPart("truncated")],
        metadata={DS4_KEY: "turn-length"},
        finish_reason="length",
        state="incomplete",
    )

    projection = project_history([response])
    turn = projection.items[0]

    assert isinstance(turn, ModelTurnHistoryItem)
    assert turn.complete is True
    assert turn.failed is False
    public = project_public_history(projection)[0]
    assert isinstance(public, PublicModelTurn)
    assert public.failed is False
