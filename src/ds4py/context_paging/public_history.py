"""Public, provider-independent DTOs for paged history and recovery."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

from pydantic_ai.messages import (
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ds4py.context_paging.history import (
    PAGING_TOOL_NAMES,
    HistoryProjection,
    ModelTurnHistoryItem,
    ToolResultPart,
    UserHistoryItem,
)

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


@dataclass(frozen=True, slots=True)
class PublicUser:
    history_id: str
    sequence: int
    message_index: int
    content: tuple[str, ...]


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


def _excluded_call_ids(turn: ModelTurnHistoryItem) -> set[str]:
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
    if not excluded:
        return True
    return any(
        exchange.call.tool_call_id not in excluded
        or (exchange.result is not None and exchange.call.tool_call_id not in excluded)
        for exchange in turn.exchanges
    )


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


def _public_result(
    result: ToolResultPart,
    message_index: int,
    part_index: int,
    call_names: Mapping[str, str],
) -> PublicToolResult:
    failed = isinstance(result, RetryPromptPart) or (isinstance(result, ToolReturnPart) and result.outcome != "success")
    tool_name = result.tool_name or call_names.get(result.tool_call_id, "")
    return PublicToolResult(
        result.tool_call_id,
        tool_name,
        deepcopy(result.content),
        failed,
        message_index,
        part_index,
    )


def _public_model_turn(turn: ModelTurnHistoryItem) -> PublicModelTurn | None:
    excluded = _excluded_call_ids(turn)
    if not _has_public_work(turn, excluded):
        return None
    visible_calls = _visible_calls(turn, excluded)
    call_names = {exchange.call.tool_call_id: exchange.call.tool_name for exchange in turn.exchanges}
    content = tuple(
        PublicText(part.content)
        if isinstance(part, TextPart)
        else PublicToolCall(part.tool_call_id, part.tool_name, deepcopy(part.args), index)
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
    )


def _public_user(item: UserHistoryItem) -> PublicUser:
    return PublicUser(
        item.history_id,
        item.sequence,
        item.message_index,
        tuple(
            part.content
            for part in item.message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        ),
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
    "project_public_history",
]
