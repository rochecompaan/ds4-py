"""Public, provider-independent DTOs for paged history and recovery."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)

from ds4py.context_paging.history import (
    PAGING_TOOL_NAMES,
    HistoryProjection,
    ModelTurnHistoryItem,
    ToolResultPart,
    UserHistoryItem,
)
from ds4py.message_metadata import DS4_TIMESTAMP

if TYPE_CHECKING:
    from datetime import datetime

_FILE_ARGUMENT_KEYS = {"path", "file", "files", "filepath", "file_path"}
_NESTED_STATUSES = {"ok", "error", "unfinished"}


@dataclass(frozen=True, slots=True)
class PublicText:
    text: str


@dataclass(frozen=True, slots=True)
class PublicToolCall:
    tool_call_id: str
    tool_name: str
    arguments: object
    original_index: int


@dataclass(frozen=True, slots=True)
class PublicToolResult:
    tool_call_id: str
    tool_name: str
    content: object
    failed: bool
    message_index: int
    part_index: int
    timestamp: int = 0
    nested_calls: object = None
    details: object = None


@dataclass(frozen=True, slots=True)
class PublicUser:
    history_id: str
    sequence: int
    message_index: int
    content: tuple[object, ...]
    timestamp: str = ""
    message_timestamp: int = 0
    content_is_list: bool = False


def _empty_usage() -> dict[str, object]:
    return {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }


@dataclass(frozen=True, slots=True)
class PublicAssistantDetails:
    api: str = "openai-completions"
    provider: str = ""
    model: str = ""
    usage: dict[str, object] = field(default_factory=_empty_usage)
    stop_reason: str = "stop"
    timestamp: int = 0


@dataclass(frozen=True, slots=True)
class PublicModelTurn:
    history_id: str
    sequence: int
    response_index: int
    content: tuple[PublicText | PublicToolCall | None, ...]
    tool_results: tuple[PublicToolResult, ...]
    tools: tuple[str, ...]
    files: tuple[str, ...]
    failed: bool
    interrupted_by_restore: bool
    timestamp: str = ""
    assistant: PublicAssistantDetails = field(default_factory=PublicAssistantDetails)


type PublicHistoryItem = PublicUser | PublicModelTurn


def _has_only_nested_paging_calls(result: ToolResultPart) -> bool:
    if not isinstance(result, ToolReturnPart) or not isinstance(result.metadata, Mapping):
        return False
    nested = result.metadata.get("nestedCalls")
    if not isinstance(nested, Mapping) or nested.get("complete") is not True:
        return False
    calls = nested.get("calls")
    if not isinstance(calls, list) or not calls:
        return False
    return all(
        isinstance(call, Mapping)
        and isinstance(call.get("id"), str)
        and bool(call.get("id"))
        and call.get("name") in PAGING_TOOL_NAMES
        and call.get("status") in _NESTED_STATUSES
        for call in calls
    )


def paging_tool_call_ids(turn: ModelTurnHistoryItem) -> set[str]:
    excluded = {
        exchange.call.tool_call_id for exchange in turn.exchanges if exchange.call.tool_name in PAGING_TOOL_NAMES
    }
    for exchange in turn.exchanges:
        result = exchange.result
        if isinstance(result, ToolReturnPart) and (
            result.tool_name in PAGING_TOOL_NAMES or _has_only_nested_paging_calls(result)
        ):
            excluded.add(exchange.call.tool_call_id)
    return excluded


def _has_public_work(turn: ModelTurnHistoryItem, excluded: set[str]) -> bool:
    if any(isinstance(part, TextPart) for part in turn.response.parts):
        return True
    if not excluded:
        return True
    return any(exchange.call.tool_call_id not in excluded for exchange in turn.exchanges)


def _collect_files(value: object, files: list[str], seen: set[str]) -> None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _collect_files(item, files, seen)
        return
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        if str(key).lower() in _FILE_ARGUMENT_KEYS:
            candidates = [child] if isinstance(child, str) else child if isinstance(child, list) else []
            for candidate in candidates:
                if isinstance(candidate, str) and candidate not in seen:
                    seen.add(candidate)
                    files.append(candidate)
        _collect_files(child, files, seen)


def _visible_calls(turn: ModelTurnHistoryItem, excluded: set[str]) -> list[tuple[int, ToolCallPart]]:
    return [
        (index, part)
        for index, part in enumerate(turn.response.parts)
        if isinstance(part, ToolCallPart) and part.tool_call_id not in excluded
    ]


def _url_content(value: ImageUrl | AudioUrl | DocumentUrl | VideoUrl) -> dict[str, object]:
    mime_type = value.media_type or "application/octet-stream"
    if value.url.lower().startswith("data:"):
        return {
            "type": "image" if isinstance(value, ImageUrl) else "bytes",
            "mimeType": mime_type,
            "embedded": True,
        }
    return {
        "type": value.kind.removesuffix("-url"),
        "url": value.url,
        "mimeType": mime_type,
    }


def _media_content(value: object) -> dict[str, object] | None:
    if isinstance(value, (BinaryContent, BinaryImage)) and value.media_type.startswith("image/"):
        return {
            "type": "image",
            "data": base64.b64encode(value.data).decode("ascii"),
            "mimeType": value.media_type,
        }
    if isinstance(value, (BinaryContent, BinaryImage)):
        return {"type": "bytes", "bytes": len(value.data), "mimeType": value.media_type}
    if isinstance(value, (ImageUrl, AudioUrl, DocumentUrl, VideoUrl)):
        return _url_content(value)
    if isinstance(value, UploadedFile):
        return {
            "type": "file",
            "fileId": value.file_id,
            "providerName": value.provider_name,
            "mimeType": value.media_type or "application/octet-stream",
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "bytes": len(value)}
    return None


def _json_safe_content(value: object) -> object:
    media = _media_content(value)
    if media is not None:
        return media
    if isinstance(value, Mapping):
        return {key: _json_safe_content(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_safe_content(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("Recovery part contains a value that is not public JSON data.")


def _milliseconds(value: datetime | None) -> int:
    return int(value.timestamp() * 1000) if value is not None else 0


def _source_timestamp(message: ModelRequest | ModelResponse) -> str:
    metadata = message.metadata
    if isinstance(metadata, Mapping) and isinstance(metadata.get(DS4_TIMESTAMP), str):
        return str(metadata[DS4_TIMESTAMP])
    return message.timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z") if message.timestamp else ""


def _public_tool_content(content: object) -> object:
    safe = _json_safe_content(content)
    if isinstance(safe, str):
        return [{"type": "text", "text": safe}]
    if isinstance(safe, Sequence) and not isinstance(safe, (bytes, bytearray)):
        return [{"type": "text", "text": child} if isinstance(child, str) else child for child in safe]
    return safe


def _assistant_details(response: ModelResponse) -> PublicAssistantDetails:
    usage = response.usage
    finish_reason = response.finish_reason
    stop_reason = {
        "stop": "stop",
        "tool_call": "toolUse",
        "length": "length",
    }.get(finish_reason, "aborted" if response.state == "incomplete" else "error")
    return PublicAssistantDetails(
        provider=response.provider_name or "",
        model=response.model_name or "",
        usage={
            "input": usage.input_tokens or 0,
            "output": usage.output_tokens or 0,
            "cacheRead": usage.cache_read_tokens or 0,
            "cacheWrite": usage.cache_write_tokens or 0,
            "totalTokens": usage.total_tokens or 0,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
        },
        stop_reason=stop_reason,
        timestamp=_milliseconds(response.timestamp),
    )


def _user_block(content: object) -> object:
    if isinstance(content, str):
        return {"type": "text", "text": content}
    if isinstance(content, BinaryContent) and content.media_type.startswith("image/"):
        return {
            "type": "image",
            "data": base64.b64encode(content.data).decode("ascii"),
            "mimeType": content.media_type,
        }
    return _json_safe_content(content)


def _public_result(
    result: ToolResultPart,
    message_index: int,
    part_index: int,
    call_names: Mapping[str, str],
) -> PublicToolResult:
    failed = isinstance(result, RetryPromptPart) or (isinstance(result, ToolReturnPart) and result.outcome != "success")
    tool_name = result.tool_name or call_names.get(result.tool_call_id, "")
    metadata = getattr(result, "metadata", None)
    return PublicToolResult(
        result.tool_call_id,
        tool_name,
        _public_tool_content(result.content),
        failed,
        message_index,
        part_index,
        _milliseconds(result.timestamp),
        deepcopy(metadata.get("nestedCalls")) if isinstance(metadata, Mapping) else None,
        deepcopy(metadata.get("details")) if isinstance(metadata, Mapping) else None,
    )


def _public_model_turn(turn: ModelTurnHistoryItem) -> PublicModelTurn | None:
    excluded = paging_tool_call_ids(turn)
    if not _has_public_work(turn, excluded):
        return None
    visible_calls = _visible_calls(turn, excluded)
    call_names = {exchange.call.tool_call_id: exchange.call.tool_name for exchange in turn.exchanges}
    content = tuple(
        PublicText(part.content)
        if isinstance(part, TextPart)
        else PublicToolCall(part.tool_call_id, part.tool_name, _json_safe_content(part.args), index)
        if isinstance(part, ToolCallPart) and part.tool_call_id not in excluded
        else None
        for index, part in enumerate(turn.response.parts)
    )
    results = tuple(
        _public_result(part, message_index, part_index, call_names)
        for message, message_index in zip(turn.result_messages, turn.result_message_indices, strict=True)
        for part_index, part in enumerate(message.parts)
        if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id not in excluded
    )
    files: list[str] = []
    seen: set[str] = set()
    for _, call in visible_calls:
        _collect_files(call.args, files, seen)
    failed = turn.interrupted_by_restore or turn.response_failed or any(result.failed for result in results)
    return PublicModelTurn(
        turn.history_id,
        turn.sequence,
        turn.response_index,
        content,
        results,
        tuple(call.tool_name for _, call in visible_calls),
        tuple(files),
        failed,
        turn.interrupted_by_restore,
        _source_timestamp(turn.response),
        _assistant_details(turn.response),
    )


def _public_user(item: UserHistoryItem) -> PublicUser:
    parts = [part for part in item.message.parts if isinstance(part, UserPromptPart)]
    content_is_list = len(parts) != 1 or not isinstance(parts[0].content, str)
    content: list[object] = []
    for part in parts:
        if isinstance(part.content, str) and not content_is_list:
            content.append(part.content)
        elif isinstance(part.content, str):
            content.append(_user_block(part.content))
        else:
            content.extend(_user_block(child) for child in part.content)
    return PublicUser(
        item.history_id,
        item.sequence,
        item.message_index,
        tuple(content),
        _source_timestamp(item.message),
        _milliseconds(parts[0].timestamp) if parts else _milliseconds(item.message.timestamp),
        content_is_list,
    )


def project_public_history(history: HistoryProjection) -> tuple[PublicHistoryItem, ...]:
    """Exclude private and paging-only content while retaining ordinary work."""

    public: list[PublicHistoryItem] = []
    for item in history.items:
        if isinstance(item, UserHistoryItem):
            public.append(_public_user(item))
            continue
        if item.omit_from_public:
            continue
        projected = _public_model_turn(item)
        if projected is not None:
            public.append(projected)
    return tuple(public)


__all__ = [
    "PublicHistoryItem",
    "PublicModelTurn",
    "PublicText",
    "PublicToolCall",
    "PublicToolResult",
    "PublicUser",
    "paging_tool_call_ids",
    "project_public_history",
]
