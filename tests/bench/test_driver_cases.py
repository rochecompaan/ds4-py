"""Manifest export and strict NDJSON boundaries for reference drivers."""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.driver_cases import (
    DriverEvidenceError,
    export_trace_cases,
    read_case_manifest,
    read_driver_ndjson,
)
from ds4py.engine.replay_types import ReplayRound, ReplayTrace


def test_export_preserves_round_order_steps_and_token_payloads(tmp_path: Path) -> None:
    trace = ReplayTrace(
        None,
        None,
        (
            ReplayRound(3, (10, 11, 12, 13), tuple(range(20, 40)), 75, 2, False),
            ReplayRound(4, (20, 21), (), 0, 0, False),
            ReplayRound(5, (30, 31, 32), (40, 41), 25, 3, False),
        ),
    )

    manifest = export_trace_cases(trace, task_id="task-a", attempt=2, destination=tmp_path)
    cases = read_case_manifest(manifest)

    assert [case.round_index for case in cases] == [3, 4, 5]
    assert [case.steps for case in cases] == [16, 0, 2]
    assert [case.split_at for case in cases] == [2, 0, 3]
    assert [case.think for case in cases] == [75, 0, 25]
    assert all(case.task_id == "task-a" and case.attempt == 2 for case in cases)
    first_tokens = [int(item) for item in cases[0].token_file.read_text().splitlines()]
    assert first_tokens == [10, 11, 12, 13, *range(20, 36)]
    assert cases[1].token_file.read_text() == "20\n21\n"


def test_export_rejects_cached_prefix_beyond_prompt_and_unsafe_ids(tmp_path: Path) -> None:
    invalid = ReplayTrace(None, None, (ReplayRound(0, (1, 2), (), 0, 3, False),))
    with pytest.raises(DriverEvidenceError, match="cached prefix"):
        export_trace_cases(invalid, task_id="task", attempt=1, destination=tmp_path / "invalid")

    valid = ReplayTrace(None, None, (ReplayRound(0, (1, 2), (), 0, 0, False),))
    with pytest.raises(DriverEvidenceError, match="task_id"):
        export_trace_cases(valid, task_id="bad\ttask", attempt=1, destination=tmp_path / "unsafe")


def valid_records() -> list[dict[str, Any]]:
    return [
        {
            "kind": "round_start",
            "source_trace": "/traces/c.trace",
            "task_id": "task-a",
            "attempt": 1,
            "round": 0,
            "token_file": "/cases/00000.tokens",
            "prompt_tokens": 3,
            "split_at": 1,
            "steps": 1,
            "think": 75,
        },
        {
            "kind": "sync",
            "label": "prefix",
            "input_tokens": 1,
            "cached_tokens": 0,
            "prefill_tokens": 1,
            "position": 1,
            "elapsed_seconds": 0.1,
        },
        {"kind": "prefill", "done": 1, "total": 1},
        {
            "kind": "sync",
            "label": "full",
            "input_tokens": 3,
            "cached_tokens": 1,
            "prefill_tokens": 2,
            "position": 3,
            "elapsed_seconds": 0.2,
        },
        {"kind": "prefill", "done": 2, "total": 2},
        {
            "kind": "step",
            "index": 0,
            "position": 3,
            "selected_token": 7,
            "top": [{"token": index, "logit": 6.0 - index} for index in range(12)],
        },
        {
            "kind": "round_end",
            "prefill_tokens": 3,
            "decode_steps": 1,
            "prefill_seconds": 0.3,
            "decode_seconds": 0.4,
            "terminal_reason": "stop_token",
        },
    ]


def write_records(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_read_driver_ndjson_returns_typed_ordered_evidence(tmp_path: Path) -> None:
    path = write_records(tmp_path / "evidence.ndjson", valid_records())

    rounds = read_driver_ndjson(path)

    assert len(rounds) == 1
    evidence = rounds[0]
    assert evidence.source_trace == Path("/traces/c.trace")
    assert evidence.case.round_index == 0
    assert [(call.label, call.position) for call in evidence.sync_calls] == [("prefix", 1), ("full", 3)]
    assert [(chunk.done, chunk.total) for chunk in evidence.progress_chunks] == [(1, 1), (2, 2)]
    assert len(evidence.steps) == 1
    assert evidence.steps[0].position == 3
    assert evidence.steps[0].selected_token == 7
    assert len(evidence.steps[0].top) == 12
    assert evidence.prefill_seconds == pytest.approx(0.3)
    assert evidence.decode_seconds == pytest.approx(0.4)
    assert evidence.terminal_reason == "stop_token"


@pytest.mark.parametrize("mutation", ["json", "duplicate_step", "short_top", "negative_time", "wrong_position"])
def test_read_driver_ndjson_rejects_malformed_evidence(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "invalid.ndjson"
    if mutation == "json":
        path.write_text("{not json}\n", encoding="utf-8")
    else:
        records = copy.deepcopy(valid_records())
        if mutation == "duplicate_step":
            records.insert(-1, copy.deepcopy(records[-2]))
            records[-1]["decode_steps"] = 2
        elif mutation == "short_top":
            records[-2]["top"] = records[-2]["top"][:11]
        elif mutation == "negative_time":
            records[-1]["decode_seconds"] = -0.1
        else:
            records[-2]["position"] = 4
        write_records(path, records)

    with pytest.raises(DriverEvidenceError):
        read_driver_ndjson(path)


def test_zero_token_round_has_sync_evidence_and_no_fabricated_step(tmp_path: Path) -> None:
    records = valid_records()
    records[0]["split_at"] = 0
    records[0]["steps"] = 0
    records = [
        records[0],
        {
            "kind": "sync",
            "label": "full",
            "input_tokens": 3,
            "cached_tokens": 0,
            "prefill_tokens": 3,
            "position": 3,
            "elapsed_seconds": 0.2,
        },
        {"kind": "prefill", "done": 3, "total": 3},
        {
            "kind": "round_end",
            "prefill_tokens": 3,
            "decode_steps": 0,
            "prefill_seconds": 0.2,
            "decode_seconds": 0.0,
            "terminal_reason": "max_tokens",
        },
    ]

    rounds = read_driver_ndjson(write_records(tmp_path / "zero.ndjson", records))

    assert len(rounds) == 1
    assert rounds[0].steps == ()
    assert len(rounds[0].sync_calls) == 1
    assert rounds[0].case.steps == 0
