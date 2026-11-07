"""Same-host reference-driver comparisons and Gate 6 composition."""

import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Self

import pytest

import ds4py
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench import ds4py_reference_driver
from bench.cap_parity import compare_cap_parity
from bench.comparison import StepCapture, TopLogit, compare_rate
from bench.driver_cases import (
    DriverCase,
    DriverRoundEvidence,
    export_round_zero_cap_case,
    load_case_tokens,
    read_case_manifest,
)
from bench.driver_equivalence import cached_fractions_by_task, compare_driver_rounds, rates_by_task
from bench.provenance import (
    Q8_F16_CACHE_ENV,
    Q8_F16_CACHE_MB,
    ProgressChunk,
    RoundProvenance,
    SyncCall,
    TraceProvenance,
)
from bench.report import build_report
from bench.restore_check import RestoreSummary
from bench.self_replay import SelfReplayResult
from ds4py._json import is_json_object
from ds4py.engine.replay_types import ReplayRound, ReplayTrace


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
        "q8_f16_cache_mb": 5120,
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
    return TraceProvenance(**values)  # ty: ignore[invalid-argument-type]


def test_cap_parity_compares_round_zero_cap_outcomes_only() -> None:
    capped = evidence(requested_steps=3, captures=(capture(), capture(), capture()))
    other_tokens = evidence(requested_steps=3, captures=(capture(logit_delta=1e-3), capture(), capture()))
    result = compare_cap_parity(
        [(capped, other_tokens)],
        provenance=[(provenance(capped), provenance(other_tokens))],
        response_budget=3,
    )
    assert result.passed and result.valid

    stopped = replace(capped, terminal_reason="stop_token", steps=(capture(),))
    also_stopped = replace(other_tokens, terminal_reason="tool_call_complete", steps=(capture(), capture()))
    stopped_result = compare_cap_parity(
        [(stopped, also_stopped)],
        provenance=[(provenance(stopped), provenance(also_stopped))],
        response_budget=3,
    )
    assert stopped_result.passed and stopped_result.valid

    mismatch = compare_cap_parity(
        [(capped, also_stopped)],
        provenance=[(provenance(capped), provenance(also_stopped))],
        response_budget=3,
    )
    assert not mismatch.passed and mismatch.valid

    later_round = evidence(round_index=1)
    invalid = compare_cap_parity(
        [(later_round, later_round)],
        provenance=[(provenance(later_round), provenance(later_round))],
        response_budget=1,
    )
    assert not invalid.passed and not invalid.valid


def test_cap_parity_uses_terminal_reason_for_budget_sized_stop() -> None:
    captures = (capture(), capture(), capture())
    capped = evidence(requested_steps=3, captures=captures)
    stopped = replace(capped, terminal_reason="stop_token")

    result = compare_cap_parity(
        [(capped, stopped)],
        provenance=[(provenance(capped), provenance(stopped))],
        response_budget=3,
    )

    assert not result.passed
    assert result.valid
    assert "C=cap, ds4-py=stop" in result.detail


def test_cap_case_export_pads_expected_tokens_and_excludes_later_rounds(tmp_path: Path) -> None:
    trace = ReplayTrace(
        ctx=100000,
        model="/model.gguf",
        rounds=(
            ReplayRound(0, (1, 2, 3), (4, 5), 75, 1, False),
            ReplayRound(1, (1, 2, 3, 4, 5), (6,), 75, 5, False),
        ),
    )

    manifest = export_round_zero_cap_case(
        trace,
        task_id="task-a",
        attempt=1,
        destination=tmp_path,
        response_budget=5,
    )

    cases = read_case_manifest(manifest)
    assert len(cases) == 1
    case = cases[0]
    assert case.round_index == 0 and case.steps == 5
    prompt, expected = load_case_tokens(case)
    assert prompt == (1, 2, 3)
    assert expected == (4, 5, 0, 0, 0)


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
        {"q8_f16_cache_mb": 4096},
        {"native_patches": ({"name": "other.patch", "sha256": "other-sha"},)},
        {"cuda_disable_host_register": "1"},
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


def test_python_reference_driver_ignores_tool_completion_until_native_budget() -> None:
    tool_tokens = tuple(
        tokenize_rendered("<tool_call><function=read><parameter=path>/tmp/x</parameter></function></tool_call>")
    )
    sampled = (*tool_tokens, *((121,) * 8))
    assert len(sampled) == 70

    def script(_prompt: Sequence[int], generated: Sequence[int]) -> int:
        return sampled[len(generated)] if len(generated) < len(sampled) else FAKE_EOS

    info = replace(FakeNative()._info, family="qwen", is_qwen4=True)
    prompt = tuple(tokenize_rendered("SYS"))
    case = DriverCase("task-a", 1, 0, Path("/cases/0.tokens"), len(prompt), 0, 70, 75)
    with ds4py.open("/models/fake.gguf", ctx=4096, native=FakeNative(info=info, script=script)) as engine:
        ds4py_reference_driver._sync(engine, prompt, "full")
        result = ds4py_reference_driver._decode(engine, case, prompt, sampled)

    assert len(result.steps) == 70
    assert result.terminal_reason == "max_tokens"


def test_python_reference_driver_pins_cache_before_engine_open(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, str | None] = {}

    class FakeEngine:
        def __enter__(self) -> Self:
            observed["cache_mb"] = os.environ.get(Q8_F16_CACHE_ENV)
            return self

        def __exit__(self, *exc: object) -> None:
            pass

    def fake_open(*args: object, **kwargs: object) -> FakeEngine:
        return FakeEngine()

    monkeypatch.setenv(Q8_F16_CACHE_ENV, "4096")
    monkeypatch.setattr(ds4py_reference_driver.ds4py, "open", fake_open)
    monkeypatch.setattr(ds4py_reference_driver, "read_case_manifest", lambda _: ())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ds4py_reference_driver.py",
            "--model",
            "model.gguf",
            "--cases",
            "cases.jsonl",
            "--source-trace",
            "trace.jsonl",
        ],
    )

    assert ds4py_reference_driver.main() == 0
    assert observed["cache_mb"] == str(Q8_F16_CACHE_MB)


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


def test_reference_driver_rates_and_cache_fractions_feed_gate_three_without_another_run() -> None:
    rounds = [evidence(), evidence(round_index=1, requested_steps=0)]

    rates = rates_by_task(rounds)
    cached_fractions = cached_fractions_by_task(rounds)

    assert rates["task-a"].prefill_tokens == 6
    assert rates["task-a"].prefill_seconds == pytest.approx(0.5)
    assert rates["task-a"].decode_steps == 1
    assert rates["task-a"].decode_seconds == pytest.approx(1.0)
    assert compare_rate(rates["task-a"], rates["task-a"]).passed
    # Each full sync reuses 1/3 of its prompt. The deliberate empty-cache
    # prefix-reset syncs are not independent cache-reuse observations.
    assert cached_fractions["task-a"] == pytest.approx(1 / 3)


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
