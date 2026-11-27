"""Canonical user/model-turn history and atomic tool-exchange projection."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)

from ds4py.binding_provenance import message_content_hash, validate_messages
from ds4py.errors import EngineError
from ds4py.message_metadata import DS4_KEY

if TYPE_CHECKING:
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet

    from ds4py.binding_provenance import PreparedRequestViewProvenance
    from ds4py.context_paging.selection import ContextCutFrontier

PAGING_TOOL_NAMES = ("search_history", "browse_history", "load_history", "read_context_output")

type HistoryProjectionErrorCode = Literal[
    "DUPLICATE_TOOL_CALL",
    "ORPHAN_TOOL_RESULT",
    "DUPLICATE_TOOL_RESULT",
    "INCOMPLETE_TOOL_RESULTS",
    "MISMATCHED_TOOL_RESULT",
]
type ToolResultPart = ToolReturnPart | RetryPromptPart
type RestoreBoundary = Literal["switched", "forked"]


class HistoryProjectionError(EngineError):
    """Canonical messages do not form valid atomic tool exchanges."""

    def __init__(self, code: HistoryProjectionErrorCode) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class HistoryExchange:
    """One assistant call and its optional canonical result."""

    call: ToolCallPart
    call_index: int
    result: ToolResultPart | None
    result_message_index: int | None
    result_part_index: int | None

    @property
    def failed(self) -> bool:
        if isinstance(self.result, RetryPromptPart):
            return True
        return isinstance(self.result, ToolReturnPart) and self.result.outcome != "success"


@dataclass(frozen=True, slots=True)
class UserHistoryItem:
    """One standalone canonical request."""

    history_id: str
    sequence: int
    message: ModelRequest
    message_index: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class ModelTurnHistoryItem:
    """One response and all immediately following canonical tool results."""

    history_id: str
    sequence: int
    response: ModelResponse
    response_index: int
    result_messages: tuple[ModelRequest, ...]
    result_message_indices: tuple[int, ...]
    exchanges: tuple[HistoryExchange, ...]
    content_hash: str | None
    omit_from_public: bool = False
    interrupted_by_restore: bool = False

    @property
    def complete(self) -> bool:
        return all(exchange.result is not None for exchange in self.exchanges)

    @property
    def response_failed(self) -> bool:
        return self.response.finish_reason == "error" or (
            self.response.state == "incomplete" and self.response.finish_reason != "length"
        )

    @property
    def failed(self) -> bool:
        return (
            self.interrupted_by_restore or self.response_failed or any(exchange.failed for exchange in self.exchanges)
        )


type HistoryItem = UserHistoryItem | ModelTurnHistoryItem


@dataclass(frozen=True, slots=True)
class HistoryProjection:
    """References to canonical messages plus validated exchange provenance."""

    canonical_messages: tuple[ModelMessage, ...]
    items: tuple[HistoryItem, ...]


def _history_id(message: ModelMessage) -> str:
    key = (message.metadata or {}).get(DS4_KEY)
    if not isinstance(key, str) or not key:
        raise EngineError("canonical history message keys must be non-empty strings")
    return key


def _result_parts(message: ModelRequest) -> list[tuple[int, ToolResultPart]]:
    return [
        (index, part) for index, part in enumerate(message.parts) if isinstance(part, (ToolReturnPart, RetryPromptPart))
    ]


def _is_result_message(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and bool(_result_parts(message))


def _response_interrupted(response: ModelResponse) -> bool:
    return response.finish_reason == "error" or (response.state == "incomplete" and response.finish_reason != "length")


def _response_content_hash(response: ModelResponse) -> str | None:
    if not all(isinstance(part, (TextPart, ThinkingPart, ToolCallPart)) for part in response.parts):
        return None
    return message_content_hash(response)


def _project_model_turn(
    messages: tuple[ModelMessage, ...],
    index: int,
    sequence: int,
    restore_boundary: RestoreBoundary | None,
    restore_ids: AbstractSet[str],
) -> tuple[ModelTurnHistoryItem, int]:
    response = messages[index]
    if not isinstance(response, ModelResponse):
        raise TypeError("model turn projection requires a response")
    calls = [(part_index, part) for part_index, part in enumerate(response.parts) if isinstance(part, ToolCallPart)]
    expected = {call.tool_call_id: (part_index, call) for part_index, call in calls}
    if len(expected) != len(calls):
        raise HistoryProjectionError("DUPLICATE_TOOL_CALL")

    seen: dict[str, tuple[ToolResultPart, int, int]] = {}
    result_messages: list[ModelRequest] = []
    result_message_indices: list[int] = []
    result_index = index + 1
    while result_index < len(messages) and _is_result_message(messages[result_index]):
        result_message = messages[result_index]
        if not isinstance(result_message, ModelRequest):
            raise TypeError("tool result projection requires a request")
        validate_messages([result_message])
        result_messages.append(result_message)
        result_message_indices.append(result_index)
        for part_index, result in _result_parts(result_message):
            call_id = result.tool_call_id
            if call_id not in expected:
                raise HistoryProjectionError("MISMATCHED_TOOL_RESULT")
            if call_id in seen:
                raise HistoryProjectionError("DUPLICATE_TOOL_RESULT")
            seen[call_id] = (result, result_index, part_index)
        result_index += 1

    exchanges = tuple(
        HistoryExchange(
            call,
            part_index,
            *seen.get(call.tool_call_id, (None, None, None)),
        )
        for part_index, call in calls
    )
    incomplete = len(seen) != len(expected)
    response_id = _history_id(response)
    interrupted_by_restore = (
        incomplete
        and not _response_interrupted(response)
        and (restore_boundary in ("switched", "forked") or response_id in restore_ids)
    )
    omit_from_public = False
    if incomplete and not _response_interrupted(response) and not interrupted_by_restore:
        if result_index != len(messages):
            raise HistoryProjectionError("INCOMPLETE_TOOL_RESULTS")
        omit_from_public = True

    item = ModelTurnHistoryItem(
        response_id,
        sequence,
        response,
        index,
        tuple(result_messages),
        tuple(result_message_indices),
        exchanges,
        _response_content_hash(response),
        omit_from_public,
        interrupted_by_restore,
    )
    return item, result_index


def project_history(
    messages: Sequence[ModelMessage],
    *,
    restore_boundary: RestoreBoundary | None = None,
    restore_ids: AbstractSet[str] = frozenset(),
) -> HistoryProjection:
    """Validate canonical exchanges without mutating or synthesizing messages."""

    canonical = tuple(messages)
    items: list[HistoryItem] = []
    index = 0
    while index < len(canonical):
        message = canonical[index]
        if isinstance(message, ModelResponse):
            item, index = _project_model_turn(canonical, index, len(items), restore_boundary, restore_ids)
            items.append(item)
            continue
        if _is_result_message(message):
            raise HistoryProjectionError("ORPHAN_TOOL_RESULT")
        if not isinstance(message, ModelRequest):
            raise TypeError(f"unsupported canonical message type: {type(message).__name__}")
        validate_messages([message])
        items.append(UserHistoryItem(_history_id(message), len(items), message, index, message_content_hash(message)))
        index += 1
    return HistoryProjection(canonical, tuple(items))


def derive_frontier(
    canonical_messages: Sequence[ModelMessage],
    prepared_view: PreparedRequestViewProvenance,
) -> ContextCutFrontier | None:
    """Derive a frontier only after canonical view provenance has been validated."""
    from ds4py.context_paging.selection import derive_validated_frontier  # noqa: PLC0415 - Break the history/selection module cycle.

    return derive_validated_frontier(canonical_messages, prepared_view)


__all__ = [
    "PAGING_TOOL_NAMES",
    "HistoryExchange",
    "HistoryItem",
    "HistoryProjection",
    "HistoryProjectionError",
    "ModelTurnHistoryItem",
    "UserHistoryItem",
    "derive_frontier",
    "project_history",
]
