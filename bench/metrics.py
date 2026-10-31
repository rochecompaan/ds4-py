"""Strict comparison metrics extracted from bakeoff event and trace evidence."""

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Any

from ds4py.engine.replay_types import ReplayTrace


class MetricError(ValueError):
    """Evidence is missing or internally inconsistent."""


@dataclass(frozen=True)
class ToolAccounting:
    parsed: int = 0
    executed: int = 0
    refused: int = 0
    retried: int = 0

    @property
    def accounted(self) -> int:
        return self.executed + self.refused + self.retried

    @property
    def complete(self) -> bool:
        return self.accounted == self.parsed


@dataclass(frozen=True)
class EventSummary:
    wall_seconds: float
    progress_events: int
    tokens: int
    tool_calls: int
    avg_prefill_reuse: float | None
    non_extend_rounds: int
    invalid_non_extend_rounds: int = 0
    allowed_non_extend_rounds: int = 0
    non_extension_reasons: tuple[str, ...] = ()
    parser_errors: int = 0
    tool_accounting: ToolAccounting | None = ToolAccounting()
    reuse_ratios: tuple[float, ...] = ()


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MetricError(f"{name} must be a nonnegative integer")
    return value


def _wall(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricError("wall seconds must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise MetricError("wall seconds must be finite and nonnegative")
    return value


def _payload(
    record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        raise MetricError("event payload must be an object")
    nested = payload.get("event", payload)
    if not isinstance(nested, dict):
        raise MetricError("engine event payload must be an object")
    return payload, nested


def _canonical_tool_part(part: object) -> tuple[dict[str, Any], str]:
    if not isinstance(part, dict):
        raise MetricError("canonical tool event needs a part object")
    call_id = part.get("tool_call_id")
    if not isinstance(call_id, str) or not call_id:
        raise MetricError("canonical tool event needs a nonempty tool_call_id")
    return part, call_id


def _reuse(plan: object) -> tuple[str, int, int]:
    if not isinstance(plan, dict):
        raise MetricError("Synced event is missing its plan")
    kind = plan.get("kind")
    if kind not in {"extend", "restore", "rebuild"}:
        raise MetricError(f"unknown sync plan kind: {kind!r}")
    reused = _count(plan.get("reused"), "plan.reused")
    to_prefill = _count(plan.get("to_prefill"), "plan.to_prefill")
    denominator = reused + to_prefill
    if reused > denominator:  # Also documents the intended ratio invariant.
        raise MetricError("plan.reused exceeds its denominator")
    return kind, reused, denominator


def summarize_events(
    events: Iterable[dict[str, Any]], *, wall_seconds: float,
) -> EventSummary:
    """Summarize one attempt's JSONL records without inventing missing values."""
    records = list(events)
    if not records or any(not isinstance(record, dict) for record in records):
        raise MetricError("an attempt needs a nonempty list of event objects")
    starts = sum(record.get("kind") == "run_start" for record in records)
    ends = sum(record.get("kind") == "run_end" for record in records)
    if starts == 0 or starts != ends:
        raise MetricError(f"unbalanced run events: {starts} starts and {ends} ends")
    progress = tokens = parsed = executed = refused = retried = 0
    parser_errors = non_extend = invalid_non_extend = allowed_non_extend = 0
    ratios: list[float] = []
    reasons: list[str] = []
    seen_runs: set[str] = set()
    canonical_calls: set[str] = set()
    resolved_calls: set[str] = set()
    for record in records:
        kind = record.get("kind")
        if not isinstance(kind, str):
            raise MetricError("event kind must be a string")
        payload, nested = _payload(record)
        if kind == "Prefill":
            _count(nested.get("done"), "Prefill.done")
            _count(nested.get("total"), "Prefill.total")
            progress += 1
        elif kind == "Token":
            _count(nested.get("id"), "Token.id")
            tokens += 1
        elif kind == "Synced":
            plan_kind, reused, denominator = _reuse(nested.get("plan"))
            if denominator:
                ratios.append(reused / denominator)
            run_id = record.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise MetricError("Synced event needs a nonempty run_id")
            if run_id in seen_runs and plan_kind != "extend":
                non_extend += 1
                reason = payload.get("exception_reason", nested.get("exception_reason"))
                if reason in {"compaction", "strategy"}:
                    allowed_non_extend += 1
                    if reason not in reasons:
                        reasons.append(reason)
                else:
                    invalid_non_extend += 1
            seen_runs.add(run_id)
        elif kind == "Stopped":
            reason = nested.get("reason")
            if not isinstance(reason, str):
                raise MetricError("Stopped event needs a reason")
            parser_errors += reason == "parser_error"
        elif kind == "ToolCalls":
            calls = nested.get("calls")
            if not isinstance(calls, list):
                raise MetricError("ToolCalls event needs a calls list")
            parsed += len(calls)
        elif kind == "FunctionToolCallEvent":
            part, call_id = _canonical_tool_part(payload.get("part"))
            if part.get("part_kind") != "tool-call":
                raise MetricError("FunctionToolCallEvent needs a tool-call part")
            if call_id in canonical_calls:
                raise MetricError(f"duplicate canonical tool call: {call_id}")
            canonical_calls.add(call_id)
        elif kind in {"ToolResultEvent", "FunctionToolResultEvent"}:
            part = payload.get("part")
            if kind == "FunctionToolResultEvent":
                part, call_id = _canonical_tool_part(part)
                if call_id not in canonical_calls:
                    raise MetricError(f"result for unknown canonical tool call: {call_id}")
                if call_id in resolved_calls:
                    raise MetricError(f"duplicate canonical tool result: {call_id}")
                resolved_calls.add(call_id)
            if not isinstance(part, dict):
                raise MetricError("ToolResultEvent needs a part")
            part_kind = part.get("part_kind")
            if part_kind == "retry-prompt":
                retried += 1
            elif part_kind == "tool-return":
                outcome = part.get("outcome")
                if outcome == "denied":
                    refused += 1
                elif outcome in {"success", "failed", "interrupted"}:
                    executed += 1
                else:
                    raise MetricError(f"unknown tool outcome: {outcome!r}")
            else:
                raise MetricError(f"unknown tool result part: {part_kind!r}")
    unresolved = canonical_calls - resolved_calls
    if unresolved:
        raise MetricError(f"canonical tool calls without outcomes: {sorted(unresolved)!r}")
    accounting = ToolAccounting(parsed, executed, refused, retried)
    return EventSummary(
        wall_seconds=_wall(wall_seconds),
        progress_events=progress,
        tokens=tokens,
        tool_calls=parsed,
        avg_prefill_reuse=sum(ratios) / len(ratios) if ratios else None,
        non_extend_rounds=non_extend,
        invalid_non_extend_rounds=invalid_non_extend,
        allowed_non_extend_rounds=allowed_non_extend,
        non_extension_reasons=tuple(reasons),
        parser_errors=parser_errors,
        tool_accounting=accounting,
        reuse_ratios=tuple(ratios),
    )


def metrics_from_trace(
    trace: ReplayTrace, *, wall_seconds: float, progress_events: int, tool_calls: int,
) -> EventSummary:
    """Extract C metrics when no JSONL harness event stream is available."""
    rounds = list(trace.rounds)
    if not rounds:
        raise MetricError("a finished trace needs at least one round")
    ratios = []
    tokens = 0
    non_extend = 0
    previous_length = None
    for index, round_ in enumerate(rounds):
        prompt_length = len(round_.prompt)
        if prompt_length == 0:
            raise MetricError(f"round {index} has an empty prompt")
        cached = _count(round_.cached, f"round {index} cached")
        if cached > prompt_length:
            raise MetricError(f"round {index} cached prefix exceeds its prompt")
        ratios.append(cached / prompt_length)
        sampled = len(round_.sampled)
        tokens += sampled
        if previous_length is not None and cached != previous_length:
            non_extend += 1
        previous_length = prompt_length + sampled
    calls = _count(tool_calls, "tool_calls")
    return EventSummary(
        wall_seconds=_wall(wall_seconds),
        progress_events=_count(progress_events, "progress_events"),
        tokens=tokens,
        tool_calls=calls,
        avg_prefill_reuse=sum(ratios) / len(ratios),
        non_extend_rounds=non_extend,
        tool_accounting=None,
        reuse_ratios=tuple(ratios),
    )


def combine_summaries(summaries: Iterable[EventSummary]) -> EventSummary:
    """Sum attempt metrics and weight reuse by plans, not by attempt averages."""
    values = list(summaries)
    if not values:
        raise MetricError("cannot combine zero attempts")
    ratios = tuple(ratio for value in values for ratio in value.reuse_ratios)
    accounting = None
    if all(value.tool_accounting is not None for value in values):
        accounting = ToolAccounting(*(
            sum(value.tool_accounting.__dict__[field] for value in values)
            for field in ("parsed", "executed", "refused", "retried")
        ))
    reasons = tuple(dict.fromkeys(
        reason for value in values for reason in value.non_extension_reasons
    ))
    return EventSummary(
        wall_seconds=sum(value.wall_seconds for value in values),
        progress_events=sum(value.progress_events for value in values),
        tokens=sum(value.tokens for value in values),
        tool_calls=sum(value.tool_calls for value in values),
        avg_prefill_reuse=sum(ratios) / len(ratios) if ratios else None,
        non_extend_rounds=sum(value.non_extend_rounds for value in values),
        invalid_non_extend_rounds=sum(value.invalid_non_extend_rounds for value in values),
        allowed_non_extend_rounds=sum(value.allowed_non_extend_rounds for value in values),
        non_extension_reasons=reasons,
        parser_errors=sum(value.parser_errors for value in values),
        tool_accounting=accounting,
        reuse_ratios=ratios,
    )
