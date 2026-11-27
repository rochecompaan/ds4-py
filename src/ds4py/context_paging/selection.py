"""Stable rolling-window cuts measured through exact request projections."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart

from ds4py.context_paging.history import (
    HistoryExchange,
    HistoryItem,
    HistoryProjection,
    ModelTurnHistoryItem,
    UserHistoryItem,
    project_history,
)
from ds4py.context_paging.notice import (
    EvictedConversation,
    PagingNotice,
    ToolRecoveryReference,
    build_paging_notice,
    paging_notice_key,
    protected_overflow_notice,
    recovery_notice,
)
from ds4py.context_paging.public_history import paging_tool_call_ids, project_public_history
from ds4py.context_paging.settings import RollingWindowSettings, resolve_token_limits
from ds4py.errors import ContextSelectionError, EngineError
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY, DS4_VIEW_OF

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage

    from ds4py.binding_projection import RequestProjection
    from ds4py.binding_provenance import PreparedRequestViewProvenance
    from ds4py.context_paging.public_history import PublicHistoryItem

type CutKind = Literal["prefix", "completedTurn", "partialTurn"]
type CutFallbackReason = Literal[
    "raw-history-unavailable",
    "cut-key-unresolved",
    "keyless-no-key",
    "keyless-snap-over-budget",
    "frontier-not-in-groups",
]
type SelectionMode = Literal["within-budget", "paged", "protected-overflow", "recovery"]
type OverflowMode = Literal["normal", "protected-overflow", "recovery"]


class SelectionMeter(Protocol):
    def measure(
        self,
        *,
        canonical_messages: Sequence[ModelMessage],
        candidate_messages: Sequence[ModelMessage],
    ) -> RequestProjection: ...


@dataclass(frozen=True, slots=True)
class ContextCutKey:
    history_id: str
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ContextCutFrontier:
    kind: CutKind
    last_evicted: ContextCutKey
    user_history_id: str | None = None


@dataclass(frozen=True, slots=True)
class SelectionState:
    frontier: ContextCutFrontier
    notice: PagingNotice | None


@dataclass(frozen=True, slots=True)
class SelectionResult:
    candidate_messages: tuple[ModelMessage, ...]
    measurement: RequestProjection
    effective_budget: int
    effective_target: int
    mode: SelectionMode
    state: SelectionState | None
    notice: PagingNotice | None
    fallback_reason: CutFallbackReason | None = None
    warning: str | None = None

    @property
    def frontier(self) -> ContextCutFrontier | None:
        return self.state.frontier if self.state else None


@dataclass(frozen=True, slots=True)
class _CompletedUnit:
    items: tuple[HistoryItem, ...]
    origin: int


@dataclass(frozen=True, slots=True)
class _Grouped:
    prefixes: tuple[tuple[HistoryItem, ...], ...]
    completed: tuple[_CompletedUnit, ...]
    active_request: UserHistoryItem | None
    active_exchanges: tuple[ModelTurnHistoryItem, ...]
    resident: tuple[ModelMessage, ...]


@dataclass(frozen=True, slots=True)
class _EvictionUnit:
    kind: CutKind
    items: tuple[HistoryItem, ...]
    origin: int = -1


@dataclass(frozen=True, slots=True)
class _View:
    grouped: _Grouped
    units: tuple[_EvictionUnit, ...]
    active_start: int


@dataclass(frozen=True, slots=True)
class _Boundary:
    count: int
    frontier: ContextCutFrontier
    notice: PagingNotice | None
    candidate: tuple[ModelMessage, ...]
    measurement: RequestProjection


@dataclass(frozen=True, slots=True)
class _RunResult:
    candidate: tuple[ModelMessage, ...]
    measurement: RequestProjection
    mode: SelectionMode
    normal_notice: PagingNotice | None
    removed_count: int
    last_keyed: _Boundary | None
    last_was_keyless: bool


_WARNED_LIMITS: set[tuple[int, int]] = set()


def _warning(budget: int, target: int, *, explicit: bool) -> str | None:
    pair = (budget, target)
    if not explicit or pair in _WARNED_LIMITS:
        return None
    _WARNED_LIMITS.add(pair)
    return (
        f"Explicit trim target is at or above the token budget; using the effective budget {budget} as the trim target."
    )


def _item_messages(item: HistoryItem) -> tuple[ModelMessage, ...]:
    if isinstance(item, UserHistoryItem):
        return (item.message,)
    return (item.response, *item.result_messages)


def _eligible(item: HistoryItem) -> bool:
    return isinstance(item, UserHistoryItem) or not (item.response_failed or item.interrupted_by_restore)


def _opaque_raw_item(message: ModelMessage, index: int) -> HistoryItem:
    if isinstance(message, ModelRequest):
        return UserHistoryItem("", index, message, index, "")
    return ModelTurnHistoryItem("", index, message, index, (), (), (), None)


def _raw_turn(messages: Sequence[ModelMessage], start: int, end: int) -> ModelTurnHistoryItem:
    response = messages[start]
    if not isinstance(response, ModelResponse):
        raise TypeError("A raw fallback turn must start with a response.")
    result_messages = tuple(cast("ModelRequest", message) for message in messages[start + 1 : end])
    exchanges: list[HistoryExchange] = []
    for call_index, call in enumerate(response.parts):
        if not isinstance(call, ToolCallPart):
            continue
        for message_offset, message in enumerate(result_messages):
            for part_index, result in enumerate(message.parts):
                if isinstance(result, ToolReturnPart) and result.tool_call_id == call.tool_call_id:
                    exchanges.append(HistoryExchange(call, call_index, result, start + 1 + message_offset, part_index))
    return ModelTurnHistoryItem(
        "",
        start,
        response,
        start,
        result_messages,
        tuple(range(start + 1, end)),
        tuple(exchanges),
        None,
    )


def _raw_fallback(history: HistoryProjection) -> tuple[HistoryProjection, _Grouped]:
    messages = history.canonical_messages
    active_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], ModelRequest)
            and any(isinstance(part, UserPromptPart) for part in messages[index].parts)
        ),
        None,
    )
    items: list[HistoryItem] = []
    active_request: UserHistoryItem | None = None
    index = 0
    while index < len(messages):
        message = messages[index]
        if index == active_index and isinstance(message, ModelRequest):
            active_request = UserHistoryItem("", index, message, index, "")
            items.append(active_request)
            index += 1
            continue
        if isinstance(message, ModelResponse):
            end = index + 1
            while end < len(messages):
                following = messages[end]
                if not isinstance(following, ModelRequest) or not any(
                    isinstance(part, ToolReturnPart) for part in following.parts
                ):
                    break
                end += 1
            items.append(_raw_turn(messages, index, end))
            index = end
            continue
        items.append(_opaque_raw_item(message, index))
        index += 1
    active_position = items.index(active_request) if active_request is not None else len(items)
    grouped = _Grouped(
        tuple((item,) for item in items[:active_position]),
        (),
        active_request,
        tuple(item for item in items[active_position + 1 :] if isinstance(item, ModelTurnHistoryItem)),
        (),
    )
    return HistoryProjection(messages, tuple(items)), grouped


def _group(history: HistoryProjection) -> _Grouped:
    items = [item for item in history.items if _eligible(item)]
    users = [index for index, item in enumerate(items) if isinstance(item, UserHistoryItem)]
    if not users:
        return _Grouped(tuple((item,) for item in items), (), None, (), history.resident_messages)
    prefixes = tuple((item,) for item in items[: users[0]])
    completed = tuple(_CompletedUnit(tuple(items[start:end]), items[start].sequence) for start, end in pairwise(users))
    active_items = items[users[-1] :]
    active_request = active_items[0]
    if not isinstance(active_request, UserHistoryItem) or not all(
        isinstance(item, ModelTurnHistoryItem) for item in active_items[1:]
    ):
        raise ContextSelectionError(
            "INVALID_MESSAGE_STRUCTURE", "Canonical active turn has an unsupported message structure."
        )
    return _Grouped(
        prefixes,
        completed,
        active_request,
        tuple(item for item in active_items[1:] if isinstance(item, ModelTurnHistoryItem)),
        history.resident_messages,
    )


def _view(grouped: _Grouped) -> _View:
    units = [
        *(_EvictionUnit("prefix", items) for items in grouped.prefixes),
        *(_EvictionUnit("completedTurn", unit.items, unit.origin) for unit in grouped.completed),
    ]
    active_start = len(units)
    units.extend(_EvictionUnit("partialTurn", (exchange,)) for exchange in grouped.active_exchanges)
    return _View(grouped, tuple(units), active_start)


def _candidate(
    view: _View,
    count: int,
    notice: PagingNotice | None,
    replacement: tuple[ModelTurnHistoryItem, tuple[ModelRequest, ...]] | None = None,
) -> tuple[ModelMessage, ...]:
    messages: list[ModelMessage] = list(view.grouped.resident)
    if notice is not None:
        messages.append(notice.as_message())
    for unit in view.units[count : view.active_start]:
        for item in unit.items:
            messages.extend(_item_messages(item))
    if view.grouped.active_request is not None:
        messages.append(view.grouped.active_request.message)
    for unit in view.units[max(count, view.active_start) :]:
        for item in unit.items:
            if replacement is not None and item is replacement[0]:
                messages.extend((item.response, *replacement[1]))
            else:
                messages.extend(_item_messages(item))
    return tuple(messages)


def _measure(
    history: HistoryProjection,
    meter: SelectionMeter,
    candidate: Sequence[ModelMessage],
) -> RequestProjection:
    return meter.measure(canonical_messages=history.canonical_messages, candidate_messages=candidate)


def _matches(item: HistoryItem, key: ContextCutKey) -> bool:
    if item.history_id != key.history_id:
        return False
    return key.tool_call_id is None or (
        isinstance(item, ModelTurnHistoryItem)
        and any(exchange.call.tool_call_id == key.tool_call_id for exchange in item.exchanges)
    )


def _raw_frontier_resolves(history: HistoryProjection, frontier: ContextCutFrontier) -> bool:
    unique: dict[str, HistoryItem | None] = {}
    for item in history.items:
        unique[item.history_id] = None if item.history_id in unique else item
    anchor = unique.get(frontier.last_evicted.history_id)
    if anchor is None or not _matches(anchor, frontier.last_evicted):
        return False
    if frontier.kind != "partialTurn":
        return True
    if not isinstance(anchor, ModelTurnHistoryItem):
        return False
    return frontier.user_history_id is None or isinstance(unique.get(frontier.user_history_id), UserHistoryItem)


def _apply_frontier(view: _View, frontier: ContextCutFrontier) -> _View | None:  # noqa: PLR0911 - Frontier variants have distinct failure and success results.
    grouped = view.grouped
    if frontier.kind in ("prefix", "completedTurn"):
        units = grouped.prefixes if frontier.kind == "prefix" else tuple(unit.items for unit in grouped.completed)
        indexes = [
            index for index, items in enumerate(units) if any(_matches(item, frontier.last_evicted) for item in items)
        ]
        if len(indexes) != 1:
            return None
        index = indexes[0]
        if frontier.kind == "prefix":
            return _view(
                _Grouped(
                    grouped.prefixes[index + 1 :],
                    grouped.completed,
                    grouped.active_request,
                    grouped.active_exchanges,
                    grouped.resident,
                )
            )
        return _view(
            _Grouped(
                (), grouped.completed[index + 1 :], grouped.active_request, grouped.active_exchanges, grouped.resident
            )
        )

    completed_candidates: list[tuple[int, int]] = []
    for unit_index, unit in enumerate(grouped.completed):
        for item_index, item in enumerate(unit.items):
            if isinstance(item, ModelTurnHistoryItem) and _matches(item, frontier.last_evicted):
                completed_candidates.append((unit_index, item_index))
    active_candidates = [
        index for index, item in enumerate(grouped.active_exchanges) if _matches(item, frontier.last_evicted)
    ]
    if len(completed_candidates) + len(active_candidates) != 1:
        return None
    if active_candidates:
        active = grouped.active_request
        if frontier.user_history_id is not None and (active is None or active.history_id != frontier.user_history_id):
            return None
        index = active_candidates[0]
        return _view(_Grouped((), (), active, grouped.active_exchanges[index + 1 :], grouped.resident))

    unit_index, item_index = completed_candidates[0]
    original = grouped.completed[unit_index]
    request = original.items[0]
    if not isinstance(request, UserHistoryItem):
        return None
    if frontier.user_history_id is not None and request.history_id != frontier.user_history_id:
        return None
    remainder = _CompletedUnit((request, *original.items[item_index + 1 :]), original.origin)
    return _view(
        _Grouped(
            (),
            (remainder, *grouped.completed[unit_index + 1 :]),
            grouped.active_request,
            grouped.active_exchanges,
            grouped.resident,
        )
    )


def _cut_key(item: HistoryItem) -> ContextCutKey | None:
    if not item.history_id:
        return None
    call_id = item.exchanges[0].call.tool_call_id if isinstance(item, ModelTurnHistoryItem) and item.exchanges else None
    return ContextCutKey(item.history_id, call_id)


def _frontier(unit: _EvictionUnit, view: _View) -> ContextCutFrontier | None:
    for item in reversed(unit.items):
        key = _cut_key(item)
        if key is None:
            continue
        if unit.kind != "partialTurn":
            return ContextCutFrontier(unit.kind, key)
        owner = view.grouped.active_request
        return ContextCutFrontier("partialTurn", key, owner.history_id if owner and owner.history_id else None)
    return None


def _protected_exchange(view: _View, history: HistoryProjection) -> ModelTurnHistoryItem | None:
    exchanges = view.grouped.active_exchanges
    if not exchanges or not history.items or history.items[-1] is not exchanges[-1]:
        return None
    exchange = exchanges[-1]
    return exchange if exchange.exchanges and exchange.complete and not exchange.omit_from_public else None


def _error(code: str, diagnostic: str, resident: int, estimated: int, budget: int) -> ContextSelectionError:
    return ContextSelectionError(code, diagnostic, resident, estimated, budget)


def _recovered_protected_exchange(
    turn: ModelTurnHistoryItem,
    recoverable_ids: set[str],
) -> tuple[ModelTurnHistoryItem, tuple[ModelRequest, ...]]:
    excluded = paging_tool_call_ids(turn)
    replacements: list[ModelRequest] = []
    for message in turn.result_messages:
        clone = deepcopy(message)
        original_key = (message.metadata or {}).get(DS4_KEY)
        if not isinstance(original_key, str) or not original_key:
            raise EngineError("protected recovery requires canonical tool-result message keys")
        clone.metadata = {
            **(clone.metadata or {}),
            DS4_KEY: f"{original_key}:recovery",
            DS4_CONTEXT_PAGING: "recovery",
            DS4_VIEW_OF: original_key,
        }
        for part in clone.parts:
            if not isinstance(part, ToolReturnPart):
                continue
            tool_part = cast("ToolReturnPart", part)
            if part.tool_call_id in excluded:
                tool_part.content = (
                    "[Context paging recovery — generated by the extension]\n"
                    "This oversized paging-tool result is excluded from recovery.\n"
                    "Use search_history or browse_history to find the original history items instead."
                )
            elif turn.history_id in recoverable_ids:
                handle = json.dumps(
                    {"historyId": turn.history_id, "source": "toolResult", "toolCallId": part.tool_call_id},
                    separators=(",", ":"),
                )
                tool_part.content = (
                    "[Context paging recovery — generated by the extension]\n"
                    "This tool result exceeded the active model context window.\n"
                    f"Read the exact stored result with read_context_output({handle})."
                )
        replacements.append(clone)
    return turn, tuple(replacements)


def select_view(  # noqa: C901, PLR0911, PLR0913, PLR0915 - The public FIFO selector owns one explicit state machine and its complete input.
    history: HistoryProjection,
    *,
    settings: RollingWindowSettings,
    model_context: int,
    meter: SelectionMeter,
    remembered: SelectionState | None = None,
    overflow: OverflowMode = "normal",
) -> SelectionResult:
    """Select a stable request view without mutating canonical history or live state."""
    limits = resolve_token_limits(settings, model_context)
    budget = limits.budget_tokens
    warning = _warning(budget, limits.trim_to_tokens, explicit=limits.explicit_target_at_budget)
    raw_grouped = _group(history)
    raw_view = _view(raw_grouped)
    raw_measurement = _measure(history, meter, _candidate(raw_view, 0, None))
    if raw_measurement.resident_tokens > budget:
        resident = raw_measurement.resident_tokens
        raise _error(
            "RESIDENT_INPUT_TOO_LARGE",
            f"Resident input estimate {resident} exceeds budget estimate {budget}.",
            resident,
            resident,
            budget,
        )
    public_items = project_public_history(history)
    public_by_id = {item.history_id: item for item in public_items}
    recoverable_ids = set(public_by_id)

    def public_for_unit(unit: _EvictionUnit) -> tuple[PublicHistoryItem, ...]:
        return tuple(public_by_id[item.history_id] for item in unit.items if item.history_id in public_by_id)

    def latest_tool_reference(item: HistoryItem) -> ToolRecoveryReference | None:
        if not isinstance(item, ModelTurnHistoryItem) or item.history_id not in recoverable_ids:
            return None
        excluded = paging_tool_call_ids(item)
        reference = None
        for exchange in item.exchanges:
            if isinstance(exchange.result, ToolReturnPart) and exchange.call.tool_call_id not in excluded:
                reference = ToolRecoveryReference(item.history_id, exchange.call.tool_call_id, exchange.call.tool_name)
        return reference

    def run(view: _View, destination: int, previous: SelectionState | None, *, stateful: bool) -> _RunResult:  # noqa: C901, PLR0912, PLR0915 - FIFO transitions stay in one executor so their safety rules cannot drift.
        count = 0
        notice = previous.notice if previous else None
        candidate = _candidate(view, count, notice)
        measurement = _measure(history, meter, candidate)
        evicted_id = (
            previous.frontier.last_evicted.history_id
            if previous and previous.frontier.last_evicted.history_id in recoverable_ids
            else None
        )
        tool_reference: ToolRecoveryReference | None = None
        if previous is not None and previous.notice is None:
            # A restored frontier has no serialized notice. Replay the evicted
            # tool scan so its rebuilt notice keeps the same recovery reference.
            for item in history.items:
                reference = latest_tool_reference(item)
                if reference is not None:
                    tool_reference = reference
                if item.history_id == previous.frontier.last_evicted.history_id:
                    break
        notice_key: str | None = None
        last_keyed = (
            _Boundary(0, previous.frontier, notice, candidate, measurement)
            if previous is not None and notice is not None
            else None
        )
        last_was_keyless = False
        protected = _protected_exchange(view, history)
        conversation = EvictedConversation()
        retained_origins = {unit.origin for unit in view.units[: view.active_start] if unit.kind == "completedTurn"}
        for unit in raw_grouped.completed:
            if unit.origin not in retained_origins:
                conversation.add_turn(public_for_unit(_EvictionUnit("completedTurn", unit.items, unit.origin)))
        floor_candidate: list[ModelMessage] = []
        if view.grouped.active_request is not None:
            floor_candidate.append(view.grouped.active_request.message)
        if protected is not None:
            floor_candidate.extend(_item_messages(protected))
        floor = _measure(history, meter, floor_candidate)
        notice_characters = max(0, budget - floor.total_tokens) * 4

        def bounded_notice(
            history_id: str | None,
            tool: ToolRecoveryReference | None,
            current_view: _View,
            count: int,
        ) -> tuple[PagingNotice, tuple[ModelMessage, ...], RequestProjection]:
            notice = build_paging_notice(budget, history_id, tool, conversation, notice_characters)
            candidate = _candidate(current_view, count, notice)
            measurement = _measure(history, meter, candidate)
            if measurement.total_tokens > budget:
                header_only = build_paging_notice(budget, history_id, tool)
                if header_only.content != notice.content:
                    notice = header_only
                    candidate = _candidate(current_view, count, notice)
                    measurement = _measure(history, meter, candidate)
            return notice, candidate, measurement

        if previous is not None and notice is None:
            notice, candidate, measurement = bounded_notice(evicted_id, tool_reference, view, count)
            last_keyed = _Boundary(count, previous.frontier, notice, candidate, measurement)

        if measurement.total_tokens <= budget and overflow != "recovery":
            return _RunResult(
                candidate,
                measurement,
                "paged" if previous else "within-budget",
                notice,
                count,
                last_keyed,
                last_was_keyless,
            )
        if view.grouped.active_request is not None:
            active_only = _measure(history, meter, (view.grouped.active_request.message,))
            if active_only.total_tokens > budget:
                estimated = active_only.total_tokens
                raise _error(
                    "ACTIVE_REQUEST_TOO_LARGE",
                    f"Resident plus active request estimate {estimated} exceeds budget estimate {budget}.",
                    active_only.resident_tokens,
                    estimated,
                    budget,
                )

        while measurement.total_tokens > min(destination, budget) or (overflow == "recovery" and count == 0):
            unit = view.units[count] if count < len(view.units) else None
            if unit is None or (protected is not None and unit.items == (protected,)):
                if count == 0 and previous is not None and measurement.total_tokens > budget:
                    notice, candidate, measurement = bounded_notice(evicted_id, tool_reference, view, count)
                if measurement.total_tokens <= budget:
                    return _RunResult(candidate, measurement, "paged", notice, count, last_keyed, last_was_keyless)
                if protected is not None:
                    if overflow != "recovery":
                        overflow_notice = protected_overflow_notice(budget)
                        overflow_candidate = _candidate(view, count, overflow_notice)
                        overflow_measurement = _measure(history, meter, overflow_candidate)
                        if overflow_measurement.total_tokens <= limits.model_limit:
                            return _RunResult(
                                overflow_candidate,
                                overflow_measurement,
                                "protected-overflow",
                                notice,
                                count,
                                last_keyed,
                                last_was_keyless,
                            )
                    if overflow != "protected-overflow":
                        replacement = _recovered_protected_exchange(protected, recoverable_ids)
                        recovery = recovery_notice()
                        recovery_candidate = _candidate(view, count, recovery, replacement)
                        recovery_measurement = _measure(history, meter, recovery_candidate)
                        if recovery_measurement.total_tokens <= limits.model_limit:
                            return _RunResult(
                                recovery_candidate,
                                recovery_measurement,
                                "recovery",
                                notice,
                                count,
                                last_keyed,
                                last_was_keyless,
                            )
                        estimated = recovery_measurement.total_tokens
                        raise _error(
                            "ACTIVE_REQUEST_TOO_LARGE",
                            f"Protected exchange recovery estimate {estimated} exceeds model context estimate {model_context}.",
                            recovery_measurement.resident_tokens,
                            estimated,
                            budget,
                        )
                estimated = measurement.total_tokens
                code = "ACTIVE_REQUEST_TOO_LARGE" if view.grouped.active_request else "RESIDENT_INPUT_TOO_LARGE"
                raise _error(
                    code,
                    f"Resident and retained request estimate {estimated} exceeds budget estimate {budget}.",
                    measurement.resident_tokens,
                    estimated,
                    budget,
                )

            count += 1
            for item in unit.items:
                if item.history_id in recoverable_ids:
                    evicted_id = item.history_id
                reference = latest_tool_reference(item)
                if reference is not None:
                    tool_reference = reference
            conversation_grew = unit.kind == "completedTurn" and conversation.add_turn(public_for_unit(unit))
            next_key = paging_notice_key(evicted_id, tool_reference)
            if conversation_grew or next_key != notice_key or notice is None:
                notice, candidate, measurement = bounded_notice(evicted_id, tool_reference, view, count)
                notice_key = next_key
            else:
                candidate = _candidate(view, count, notice)
                measurement = _measure(history, meter, candidate)
            if stateful:
                frontier = _frontier(unit, view)
                last_was_keyless = frontier is None
                if frontier is not None:
                    last_keyed = _Boundary(count, frontier, notice, candidate, measurement)

        return _RunResult(candidate, measurement, "paged", notice, count, last_keyed, last_was_keyless)

    def fallback(reason: CutFallbackReason) -> SelectionResult:
        result = run(raw_view, budget, None, stateful=False)
        return SelectionResult(
            result.candidate,
            result.measurement,
            budget,
            budget,
            result.mode,
            None,
            result.normal_notice,
            reason,
            warning,
        )

    if history.canonical_messages and not history.items:
        # Public recovery and keyed frontiers remain empty during raw fallback.
        history, raw_grouped = _raw_fallback(history)
        raw_view = _view(raw_grouped)
        return fallback("raw-history-unavailable")
    view = raw_view
    if remembered is not None:
        if not _raw_frontier_resolves(history, remembered.frontier):
            return fallback("cut-key-unresolved")
        retained = _apply_frontier(view, remembered.frontier)
        if retained is None:
            return fallback("frontier-not-in-groups")
        view = retained
    result = run(view, limits.trim_to_tokens, remembered, stateful=True)
    if result.removed_count == 0:
        state = SelectionState(remembered.frontier, result.normal_notice) if remembered else None
        return SelectionResult(
            result.candidate,
            result.measurement,
            budget,
            limits.trim_to_tokens,
            result.mode,
            state,
            result.normal_notice if result.mode in ("within-budget", "paged") else None,
            None,
            warning,
        )
    boundary = result.last_keyed
    if boundary is None:
        return fallback("keyless-no-key")
    if result.last_was_keyless:
        if boundary.measurement.total_tokens > budget:
            return fallback("keyless-snap-over-budget")
        state = remembered if boundary.count == 0 and remembered else SelectionState(boundary.frontier, boundary.notice)
        return SelectionResult(
            boundary.candidate,
            boundary.measurement,
            budget,
            limits.trim_to_tokens,
            "paged",
            state,
            boundary.notice,
            None,
            warning,
        )
    state = SelectionState(boundary.frontier, boundary.notice)
    return SelectionResult(
        result.candidate,
        result.measurement,
        budget,
        limits.trim_to_tokens,
        result.mode,
        state,
        result.normal_notice if result.mode == "paged" else None,
        None,
        warning,
    )


def derive_validated_frontier(
    canonical_messages: Sequence[ModelMessage],
    prepared_view: PreparedRequestViewProvenance,
) -> ContextCutFrontier | None:
    """Derive a saved cut by replaying legal groups against validated identities."""
    positions = {(message.metadata or {}).get(DS4_KEY): index for index, message in enumerate(canonical_messages)}
    targets = [identity.canonical_key for identity in prepared_view.messages if identity.canonical_key is not None]
    if not targets:
        raise EngineError("a saved request view must retain at least one canonical message")
    last_retained = max(positions[target] for target in targets)
    grouped = _group(project_history(canonical_messages[: last_retained + 1]))
    view = _view(grouped)
    for count in range(len(view.units) + 1):
        candidate = _candidate(view, count, None)
        candidate_keys = [(message.metadata or {}).get(DS4_KEY) for message in candidate]
        if candidate_keys != targets:
            continue
        if count == 0:
            return None
        frontier = _frontier(view.units[count - 1], view)
        if frontier is not None:
            return frontier
    raise EngineError("the saved request view does not replay to one legal canonical frontier")


__all__ = [
    "ContextCutFrontier",
    "ContextCutKey",
    "OverflowMode",
    "RollingWindowSettings",
    "SelectionMeter",
    "SelectionResult",
    "SelectionState",
    "select_view",
]
