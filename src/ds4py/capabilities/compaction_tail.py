"""Choose a whole-message, whole-tool-exchange tail from bound token ranges."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.messages import RetryPromptPart, ToolCallPart, ToolReturnPart

from ds4py.engine.transcript import Segment
from ds4py.errors import EngineError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage

    from ds4py.binding import MessageBinder


@dataclass(frozen=True)
class CompactionTail:
    messages: list[ModelMessage]
    bottom: int
    tokens: tuple[int, ...]
    segments: tuple[Segment, ...]
    system_end: int
    dropped: int
    budget: int
    open_start: int | None
    snapshot: list[ModelMessage]


def tail_boundary(bottom: int, system: int, budget: int, users: Sequence[int]) -> int:
    """C's nearest user boundary within the inclusive 2x scan, else target."""
    earliest = max(system, bottom - 2 * budget)
    return max((start for start in users if earliest <= start < bottom), default=max(system, bottom - budget))


def _exchange_start(messages: Sequence[ModelMessage], cut: int) -> int:
    calls: dict[str, int] = {
        part.tool_call_id: index
        for index, message in enumerate(messages)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    }
    while True:
        dependencies: list[int] = [cut]
        for message in messages[cut:]:
            for part in message.parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id:
                    dependencies.append(calls.get(part.tool_call_id, cut))
        start = min(dependencies)
        if start >= cut:
            return cut
        cut = start


def plan_tail(binder: MessageBinder, messages: Sequence[ModelMessage]) -> CompactionTail:
    """Plan without changing messages, provenance, or native state.

    Adoption on copies recovers keys lost by processed harness requests. Tokens
    are always sliced from the real transcript, never re-rendered from text.
    """
    snapshot = binder.adopt(deepcopy(messages))
    records = binder.records()
    transcript = binder.session.transcript
    tokens, segments = transcript.tokens, transcript.segments
    positions = {(message.metadata or {}).get("ds4.key"): index for index, message in enumerate(snapshot)}
    if any(record.key not in positions for record in records):
        raise EngineError("compaction history does not cover the bound transcript")
    system_end = 0
    for segment in segments:
        if segment.kind != "system" or segment.meta.get("key"):
            break
        system_end = segment.end
    budget = max(1, min(binder.session.engine.config.ctx // 10, 50_000))
    users = [segment.start for segment in segments if segment.kind == "user"]
    target = tail_boundary(len(tokens), system_end, budget, users)
    cut = len(snapshot)
    for record in records:
        if record.end > target:
            cut = positions[record.key]
            break
    # Unrendered pending requests are always retained.
    cut = min(cut, positions[records[-1].key] + 1) if records else 0
    cut = _exchange_start(snapshot, cut)
    end = records[-1].end if records else system_end
    open_start = next(
        (
            segment.start
            for segment in segments
            if segment.start >= end and segment.kind == "assistant_prefix" and not segment.meta.get("key")
        ),
        None,
    )
    starts = {record.key: record.start for record in records}
    bottom = next(
        (
            starts[key]
            for message in snapshot[cut:]
            if isinstance(key := (message.metadata or {}).get("ds4.key"), str) and key in starts
        ),
        open_start if open_start is not None else len(tokens),
    )
    # Include notes immediately preceding the retained message, not a fragment.
    for segment in reversed(segments):
        if segment.end == bottom and segment.kind == "note":
            bottom = segment.start
    return CompactionTail(
        snapshot[cut:],
        bottom,
        tokens[bottom:],
        tuple(
            Segment(segment.start - bottom, segment.end - bottom, segment.kind, segment.meta)
            for segment in segments
            if segment.start >= bottom
        ),
        system_end,
        cut,
        budget,
        open_start,
        snapshot,
    )
