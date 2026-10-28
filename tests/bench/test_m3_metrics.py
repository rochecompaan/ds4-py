"""Synthetic event and trace metrics for M3 gates."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.metrics import MetricError, metrics_from_trace, summarize_events
from ds4py.engine.replay_types import ReplayRound, ReplayTrace


def event(kind, payload=None, run_id="run-1"):
    return {"kind": kind, "run_id": run_id, "payload": payload or {}}


def synced(kind="extend", reused=80, to_prefill=20, reason=None):
    plan = {"kind": kind, "reused": reused, "to_prefill": to_prefill}
    payload = {"event": {"pos": reused + to_prefill, "plan": plan, "prefilled": to_prefill}}
    if reason is not None:
        payload["exception_reason"] = reason
    return event("Synced", payload)


def result_part(part_kind="tool-return", outcome="success"):
    return event("ToolResultEvent", {
        "part": {"part_kind": part_kind, "outcome": outcome, "tool_call_id": "call-1"},
    })


def valid_events():
    return [
        event("run_start"),
        synced("rebuild", 0, 100),
        event("Prefill", {"event": {"done": 100, "total": 100}}),
        event("Token", {"event": {"id": 11, "text_bytes": "31"}}),
        event("ToolCalls", {"event": {"calls": [
            {"name": "read", "args": {"path": "a"}, "raw": "raw-a"},
            {"name": "write", "args": {"path": "b"}, "raw": "raw-b"},
        ]}}),
        result_part(),
        result_part("retry-prompt", None),
        synced(),
        event("Prefill", {"event": {"done": 20, "total": 20}}),
        event("Token", {"event": {"id": 12, "text_bytes": "32"}}),
        event("Stopped", {"event": {"reason": "stop_token"}}),
        event("run_end", {"status": "success"}),
    ]


def test_event_metrics_cover_six_comparison_values_and_tool_accounting():
    summary = summarize_events(valid_events(), wall_seconds=2.5)

    assert summary.wall_seconds == 2.5
    assert summary.progress_events == 2
    assert summary.tokens == 2
    assert summary.tool_calls == 2
    assert summary.avg_prefill_reuse == pytest.approx(0.4)
    assert summary.non_extend_rounds == 0
    assert summary.invalid_non_extend_rounds == 0
    assert summary.parser_errors == 0
    assert summary.tool_accounting.parsed == 2
    assert summary.tool_accounting.executed == 1
    assert summary.tool_accounting.retried == 1
    assert summary.tool_accounting.complete


def test_non_extension_rounds_require_recorded_compaction_or_strategy_reason():
    events = valid_events()
    events.insert(-2, synced("rebuild", 20, 80, "compaction"))
    events.insert(-2, synced("restore", 10, 90, "strategy"))
    events.insert(-2, synced("rebuild", 0, 100))

    summary = summarize_events(events, wall_seconds=1)

    assert summary.non_extend_rounds == 3
    assert summary.allowed_non_extend_rounds == 2
    assert summary.invalid_non_extend_rounds == 1
    assert summary.non_extension_reasons == ("compaction", "strategy")


def test_zero_reuse_denominator_is_the_only_excluded_reuse_sample():
    events = valid_events()
    events.insert(-2, synced(reused=0, to_prefill=0))
    summary = summarize_events(events, wall_seconds=1)
    assert summary.avg_prefill_reuse == pytest.approx(0.4)

    only_zero = [event("run_start"), synced("rebuild", 0, 0), event("run_end", {"status": "success"})]
    assert summarize_events(only_zero, wall_seconds=1).avg_prefill_reuse is None


def test_parser_error_and_bad_argument_retry_have_distinct_gate_semantics():
    events = valid_events()
    events[-2] = event("Stopped", {"event": {"reason": "parser_error"}})
    summary = summarize_events(events, wall_seconds=1)

    assert summary.parser_errors == 1
    assert summary.tool_accounting.retried == 1
    assert summary.tool_accounting.complete


def test_missing_tool_outcome_is_evidence_failure_not_a_valid_zero():
    events = [item for item in valid_events() if item["kind"] != "ToolResultEvent"]
    summary = summarize_events(events, wall_seconds=1)

    assert summary.tool_calls == 2
    assert not summary.tool_accounting.complete
    assert summary.tool_accounting.accounted == 0


@pytest.mark.parametrize("mutation", ["missing_end", "negative_reuse", "bad_token"])
def test_malformed_or_incomplete_events_raise(mutation):
    events = valid_events()
    if mutation == "missing_end":
        events = events[:-1]
    elif mutation == "negative_reuse":
        events[1] = synced("rebuild", -1, 100)
    else:
        events[2] = event("Token", {"event": {"id": "not-an-int"}})

    with pytest.raises(MetricError):
        summarize_events(events, wall_seconds=1)


def test_trace_metrics_use_committed_samples_and_prefix_extension():
    trace = ReplayTrace(None, None, (
        ReplayRound(0, tuple(range(10)), (10, 11), 0, 0, False),
        ReplayRound(1, tuple(range(15)), (15, 16, 17), 0, 12, False),
        ReplayRound(2, tuple(range(20)), (20,), 0, 5, False),
    ))

    summary = metrics_from_trace(
        trace, wall_seconds=3, progress_events=7, tool_calls=4,
    )

    assert summary.wall_seconds == 3
    assert summary.progress_events == 7
    assert summary.tokens == 6
    assert summary.tool_calls == 4
    assert summary.avg_prefill_reuse == pytest.approx((0.8 + 0.25) / 3)
    assert summary.non_extend_rounds == 1
    assert summary.invalid_non_extend_rounds == 0


def test_trace_reuse_rejects_inconsistent_cached_prefix():
    trace = ReplayTrace(None, None, (
        ReplayRound(0, (1, 2), (3,), 0, 3, False),
    ))
    with pytest.raises(MetricError):
        metrics_from_trace(trace, wall_seconds=1, progress_events=1, tool_calls=0)
