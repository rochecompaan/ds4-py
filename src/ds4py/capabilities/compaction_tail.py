"""Choose a whole-message, whole-tool-exchange tail from bound token ranges."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.messages import RetryPromptPart, ToolCallPart, ToolReturnPart

from ds4py.engine.transcript import TranscriptReplacement
from ds4py.errors import EngineError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic_ai.messages import ModelMessage

    from ds4py.binding import MessageBinder
    from ds4py.binding_provenance import BoundMessage
    from ds4py.engine.transcript import Segment


@dataclass(frozen=True)
class CompactionTail:
    snapshot: tuple[ModelMessage, ...]
    cut_messages: int
    bottom_tokens: int
    system_end_tokens: int
    retained: TranscriptReplacement

    @property
    def messages(self) -> tuple[ModelMessage, ...]:
        return self.snapshot[self.cut_messages :]


def tail_boundary(bottom: int, system: int, budget: int, users: Sequence[int]) -> int:
    """C's nearest user boundary within the inclusive 2x scan, else target."""
    earliest = max(system, bottom - 2 * budget)
    return max((start for start in users if earliest <= start < bottom), default=max(system, bottom - budget))


def _system_end(segments: Sequence[Segment]) -> int:
    """Tokens of the leading keyless system segments that stay with the summary."""
    system_end = 0
    for segment in segments:
        if segment.kind != "system" or segment.meta.get("key"):
            break
        system_end = segment.end
    return system_end


def _cut_index(
    records: Sequence[BoundMessage], positions: Mapping[str, int], target_tokens: int, message_count: int
) -> int:
    """First message whose bound range reaches past the target."""
    cut = message_count
    for record in records:
        if record.end > target_tokens:
            cut = positions[record.key]
            break
    # Unrendered pending requests are always retained.
    return min(cut, positions[records[-1].key] + 1) if records else 0


def _bottom_tokens(
    messages: Sequence[ModelMessage],
    cut_messages: int,
    starts: Mapping[str, int],
    open_start_tokens: int | None,
    total_tokens: int,
) -> int:
    """Token start of the earliest retained bound message, else the open assistant."""
    return next(
        (
            starts[key]
            for message in messages[cut_messages:]
            if isinstance(key := (message.metadata or {}).get("ds4.key"), str) and key in starts
        ),
        open_start_tokens if open_start_tokens is not None else total_tokens,
    )


def _note_start(bottom_tokens: int, segments: Sequence[Segment]) -> int:
    """Include notes immediately preceding the retained message, not a fragment."""
    for segment in reversed(segments):
        if segment.end == bottom_tokens and segment.kind == "note":
            bottom_tokens = segment.start
    return bottom_tokens


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
    positions = {
        key: index
        for index, message in enumerate(snapshot)
        if isinstance(key := (message.metadata or {}).get("ds4.key"), str)
    }
    if any(record.key not in positions for record in records):
        raise EngineError("compaction history does not cover the bound transcript")
    system_end = _system_end(segments)
    budget = max(1, min(binder.session.engine.config.ctx // 10, 50_000))
    users = [segment.start for segment in segments if segment.kind == "user"]
    target = tail_boundary(len(tokens), system_end, budget, users)
    cut = _exchange_start(snapshot, _cut_index(records, positions, target, len(snapshot)))
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
    bottom = _note_start(_bottom_tokens(snapshot, cut, starts, open_start, len(tokens)), segments)
    return CompactionTail(
        tuple(snapshot),
        cut,
        bottom,
        system_end,
        TranscriptReplacement(tokens, segments).slice(bottom),
    )
