"""Same-host reference-driver comparisons and Gate 6 composition."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.comparison import StepCapture, TopLogit, compare_rate
from bench.driver_cases import DriverCase, DriverRoundEvidence
from bench.driver_equivalence import compare_driver_rounds, rates_by_task
from bench.provenance import ProgressChunk, RoundProvenance, SyncCall, TraceProvenance
from bench.report import build_report
from bench.restore_check import RestoreSummary
from bench.self_replay import SelfReplayResult
from ds4py._json import is_json_object


def capture(*, logit_delta: float = 0.0, reorder: bool = False) -> StepCapture:
    entries = [TopLogit(index, 6.0 - index) for index in range(12)]
    if reorder:
        entries[0], entries[1] = entries[1], entries[0]
    entries[3] = replace(entries[3], logit=entries[3].logit + logit_delta)
    return StepCapture(3, tuple(entries), entries[0].token)


def evidence(
    *,
    round_index: int = 0,
    requested_steps: int = 1,
    captures: tuple[StepCapture, ...] | None = None,
) -> DriverRoundEvidence:
    case = DriverCase("task-a", 1, round_index, Path("/cases/0.tokens"), 3, 1, requested_steps, 75)
    steps = captures if captures is not None else (capture(),) if requested_steps else ()
    return DriverRoundEvidence(
        source_trace=Path("/traces/c.trace"),
        case=case,
        sync_calls=(
            SyncCall("prefix", 1, 0, 1, 1),
            SyncCall("full", 3, 1, 2, 3),
        ),
        progress_chunks=(ProgressChunk(1, 1), ProgressChunk(2, 2)),
        steps=steps,
        prefill_seconds=0.25,
        decode_seconds=0.5,
        terminal_reason="max_tokens",
    )


def provenance(round_: DriverRoundEvidence, **changes: object) -> TraceProvenance:
    values = {
        "bakeoff_session_id": "session-1",
        "instance_id": "instance-1",
        "gpu_model": "gpu",
        "gpu_driver_version": "driver",
        "ds4_revision": "ds4",
        "ds4py_revision": "python",
        "libds4_sha256": "library",
        "model_path": "/models/fake.gguf",
        "model_sha256": "model",
        "backend": "cpu",
        "ctx": 4096,
        "temperature": 0.0,
        "seed": 1,
        "response_budget": 8192,
        "reset_policy": "fresh-per-round",
        "trace_sha256": "trace",
        "source_trace_sha256": "source",
        "rounds": (
            RoundProvenance(
                round_.case.round_index,
                round_.case.think,
                "prompt",
                round_.sync_calls,
                round_.progress_chunks,
            ),
        ),
    }
    values.update(changes)
    return TraceProvenance(**values)  # type: ignore[arg-type]


def test_every_selected_round_and_ordered_top_logits_are_required() -> None:
    c_round = evidence()
    python_round = evidence(captures=(capture(logit_delta=1e-6),))
    result = compare_driver_rounds(
        [c_round],
        [python_round],
        c_provenance=provenance(c_round),
        ds4py_provenance=provenance(python_round),
    )

    assert result.passed
    assert result.valid
    assert result.decode_steps == 1
    assert result.rounds[0].steps[0].result.max_abs_delta == pytest.approx(1e-6)

    missing = compare_driver_rounds(
        [c_round],
        [],
        c_provenance=provenance(c_round),
        ds4py_provenance=provenance(c_round),
    )
    assert not missing.passed
    assert not missing.valid
    assert "missing" in missing.detail


def test_ranking_reorder_and_tolerance_boundary_are_numeric_failures() -> None:
    c_round = evidence()
    for python_round in (
        evidence(captures=(capture(reorder=True),)),
        evidence(captures=(capture(logit_delta=1e-5),)),
    ):
        result = compare_driver_rounds(
            [c_round],
            [python_round],
            c_provenance=provenance(c_round),
            ds4py_provenance=provenance(python_round),
        )
        assert not result.passed
        assert result.valid
        assert not result.rounds[0].steps[0].result.passed


@pytest.mark.parametrize(
    "changes",
    [
        {"instance_id": "other-instance"},
        {"bakeoff_session_id": "other-session"},
        {"libds4_sha256": "other-library"},
        {"reset_policy": "other-reset"},
    ],
)
def test_session_provenance_mismatch_is_invalid(changes: dict[str, object]) -> None:
    c_round = evidence()
    python_round = evidence()
    result = compare_driver_rounds(
        [c_round],
        [python_round],
        c_provenance=provenance(c_round),
        ds4py_provenance=provenance(python_round, **changes),
    )

    assert not result.passed
    assert not result.valid


def test_prefill_boundary_mismatch_is_invalid() -> None:
    c_round = evidence()
    python_round = evidence()
    changed_sync = SyncCall("full", 3, 0, 3, 3)
    python_round = replace(python_round, sync_calls=(python_round.sync_calls[0], changed_sync))

    result = compare_driver_rounds(
        [c_round],
        [python_round],
        c_provenance=provenance(c_round),
        ds4py_provenance=provenance(python_round),
    )

    assert not result.passed
    assert not result.valid


def test_zero_token_rounds_are_sync_only_but_set_needs_decode_coverage() -> None:
    zero_c = evidence(requested_steps=0)
    zero_python = evidence(requested_steps=0)
    only_zero = compare_driver_rounds(
        [zero_c],
        [zero_python],
        c_provenance=provenance(zero_c),
        ds4py_provenance=provenance(zero_python),
    )
    assert only_zero.rounds[0].passed
    assert only_zero.rounds[0].sync_only
    assert not only_zero.passed
    assert not only_zero.valid

    decode_c = evidence(round_index=1)
    decode_python = evidence(round_index=1)
    c_provenance = provenance(zero_c)
    python_provenance = provenance(zero_python)
    extra_round = RoundProvenance(1, 75, "prompt", decode_c.sync_calls, decode_c.progress_chunks)
    c_provenance = replace(c_provenance, rounds=(*c_provenance.rounds, extra_round))
    python_provenance = replace(python_provenance, rounds=(*python_provenance.rounds, extra_round))
    mixed = compare_driver_rounds(
        [zero_c, decode_c],
        [zero_python, decode_python],
        c_provenance=c_provenance,
        ds4py_provenance=python_provenance,
    )
    assert mixed.passed
    assert mixed.decode_steps == 1


def test_reference_driver_rates_feed_gate_three_without_another_run() -> None:
    rounds = [evidence(), evidence(round_index=1, requested_steps=0)]

    rates = rates_by_task(rounds)

    assert rates["task-a"].prefill_tokens == 6
    assert rates["task-a"].prefill_seconds == pytest.approx(0.5)
    assert rates["task-a"].decode_steps == 1
    assert rates["task-a"].decode_seconds == pytest.approx(1.0)
    assert compare_rate(rates["task-a"], rates["task-a"]).passed


def test_gate_six_requires_both_new_parts_not_c_diagnostic_or_tool_coverage() -> None:
    c_round = evidence()
    driver = compare_driver_rounds(
        [c_round],
        [c_round],
        c_provenance=provenance(c_round),
        ds4py_provenance=provenance(c_round),
    )
    self_result = SelfReplayResult("task-a", 0, True, "self replay", tool_call=False)

    report = build_report(
        [],
        RestoreSummary(),
        driver_equivalence=driver,
        self_replay_results=[self_result],
    )

    assert report.gates["gate-6"].passed
    assert "Part A" in report.gates["gate-6"].detail
    assert "Part B" in report.gates["gate-6"].detail
    data = report.to_dict()
    driver_json = data["driver_equivalence"]
    self_json = data["self_replay"]
    assert is_json_object(driver_json)
    assert driver_json["passed"] is True
    assert isinstance(self_json, list) and len(self_json) == 1
    first_self = self_json[0]
    assert is_json_object(first_self)
    assert first_self["passed"] is True
    assert data["replay_diagnostic"] is None

    missing_self = build_report([], RestoreSummary(), driver_equivalence=driver)
    assert not missing_self.gates["gate-6"].passed
    assert not missing_self.gates["gate-6"].valid
