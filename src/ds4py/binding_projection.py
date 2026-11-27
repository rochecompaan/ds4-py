"""Exact request tokens and canonical unit spans derived from binding plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolReturnPart,
)

from ds4py.binding_provenance import is_resident_message, message_content_hash
from ds4py.engine.prompt import PromptBuilder
from ds4py.errors import EngineError
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY, DS4_VIEW_OF

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic_ai.messages import ModelMessage

    from ds4py.binding_plan import BindingChange, BindingPlan
    from ds4py.binding_provenance import BoundMessage
    from ds4py.engine.session import Engine


@dataclass(frozen=True, slots=True)
class RequestUnitSpan:
    """One completed turn or trailing request-only unit in projected tokens."""

    unit_key: str
    start: int
    end: int
    request_keys: tuple[str, ...]
    response_key: str | None


@dataclass(frozen=True, slots=True)
class RequestProjection:
    """The exact request passed to Session.plan, with separated token costs."""

    tokens: tuple[int, ...]
    total_tokens: int
    change: BindingChange
    resident_tokens: int
    notice_tokens: int
    units: tuple[RequestUnitSpan, ...]


def _message_key(message: ModelMessage) -> str:
    key = (message.metadata or {}).get(DS4_KEY)
    if not isinstance(key, str) or not key:
        raise EngineError("projected request messages require non-empty canonical keys")
    return key


def _canonical_identities(messages: Sequence[ModelMessage]) -> dict[str, str]:
    identities: dict[str, str] = {}
    for message in messages:
        key = _message_key(message)
        if key in identities:
            raise EngineError(f"canonical history has duplicate message key {key!r}")
        identities[key] = message_content_hash(message)
    return identities


def _validate_view_identities(
    canonical: Mapping[str, str],
    request_view: Sequence[ModelMessage],
) -> None:
    for message in request_view:
        metadata = message.metadata or {}
        key = _message_key(message)
        marker = metadata.get(DS4_CONTEXT_PAGING)
        target = metadata.get(DS4_VIEW_OF)
        if marker == "notice":
            if DS4_VIEW_OF in metadata:
                raise EngineError("a paging notice must not have a canonical target")
            continue
        if marker == "recovery":
            if not isinstance(target, str) or target not in canonical:
                raise EngineError("a recovery replacement must target canonical content")
            if key in canonical:
                raise EngineError("a recovery replacement requires a fresh view key")
            continue
        if marker is not None or DS4_VIEW_OF in metadata:
            raise EngineError("projected request has invalid paging metadata")
        if canonical.get(key) != message_content_hash(message):
            raise EngineError("an unmarked view message must resolve to canonical content with the same key and hash")


def _canonical_key(message: ModelMessage) -> str:
    metadata = message.metadata or {}
    target = metadata.get(DS4_VIEW_OF)
    if isinstance(target, str) and target:
        return target
    return _message_key(message)


def _is_result(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, (ToolReturnPart, RetryPromptPart)) for part in message.parts
    )


def _is_notice(message: ModelMessage) -> bool:
    return bool((message.metadata or {}).get(DS4_CONTEXT_PAGING) == "notice")


def _is_resident(message: ModelMessage) -> bool:
    return is_resident_message(message)


def _unit_groups(messages: Sequence[ModelMessage]) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if _is_notice(message) or _is_resident(message):
            index += 1
            continue
        current.append(index)
        index += 1
        if isinstance(message, ModelResponse):
            while index < len(messages) and (_is_result(messages[index]) or _is_notice(messages[index])):
                if not _is_notice(messages[index]):
                    current.append(index)
                index += 1
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _unit_span(
    group: Sequence[int],
    messages: Sequence[ModelMessage],
    records: Mapping[str, BoundMessage],
) -> RequestUnitSpan:
    bound = []
    request_keys: list[str] = []
    response_key: str | None = None
    for index in group:
        message = messages[index]
        key = _message_key(message)
        try:
            bound.append(records[key])
        except KeyError as error:
            raise EngineError(f"projection has no bound token range for message key {key!r}") from error
        if isinstance(message, ModelResponse):
            response_key = _canonical_key(message)
        else:
            request_keys.append(_canonical_key(message))
    unit_key = response_key or (request_keys[0] if request_keys else None)
    if unit_key is None:
        raise EngineError("a projected history unit requires a request or response identity")
    return RequestUnitSpan(
        unit_key,
        min(record.start for record in bound),
        max(record.end for record in bound),
        tuple(request_keys),
        response_key,
    )


def _planned_tokens(engine: Engine, plan: BindingPlan, live_tokens: Sequence[int]) -> tuple[int, ...]:
    prefix_length = len(live_tokens) if plan.truncate_to is None else plan.truncate_to
    tokens = list(live_tokens[:prefix_length])
    tokens.extend(plan.initial.appends.tokens)
    if plan.reset_chat:
        for segment in PromptBuilder(engine).begin(plan.definitions, plan.instructions):
            tokens.extend(segment.tokens)
    for boundary in plan.boundaries:
        tokens.extend(boundary.appends.tokens)
    return tuple(tokens)


def build_request_projection(
    *,
    engine: Engine,
    live_tokens: Sequence[int],
    canonical_messages: Sequence[ModelMessage],
    plan: BindingPlan,
) -> RequestProjection:
    """Reconstruct the scratch request and derive spans from planned records."""

    canonical = _canonical_identities(canonical_messages)
    _validate_view_identities(canonical, plan.messages)
    records = plan.boundaries[-1].records if plan.boundaries else plan.initial.records
    record_by_key = {record.key: record for record in records}
    units = tuple(_unit_span(group, plan.messages, record_by_key) for group in _unit_groups(plan.messages))
    notice_tokens = sum(
        record_by_key[_message_key(message)].end - record_by_key[_message_key(message)].start
        for message in plan.messages
        if _is_notice(message)
    )

    history_tokens = _planned_tokens(engine, plan, live_tokens)
    assistant_prefix = PromptBuilder(engine).assistant_prefix(history_tokens)
    tokens = (*history_tokens, *assistant_prefix)
    if len(tokens) != plan.projected_tokens:
        raise AssertionError("reconstructed projection must match the planner's exact token count")
    unit_tokens = sum(unit.end - unit.start for unit in units)
    resident_tokens = len(history_tokens) - notice_tokens - unit_tokens
    if resident_tokens < 0:
        raise AssertionError("projected unit and notice ranges must not exceed the request")
    return RequestProjection(tokens, len(tokens), plan.change, resident_tokens, notice_tokens, units)


__all__ = ["RequestProjection", "RequestUnitSpan", "build_request_projection"]
