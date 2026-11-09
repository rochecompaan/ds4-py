"""Sequential bakeoff runner and gates 1–5 with fake executables."""

import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.comparison import RateEvidence, StepCapture, TopLogit
from bench.driver_cases import DriverCase, DriverRoundEvidence, read_case_manifest
from bench.driver_equivalence import DriverEquivalenceResult, DriverRoundResult
from bench.metrics import EventSummary, MetricError, ToolAccounting
from bench.provenance import ProgressChunk, RoundProvenance, SyncCall, TraceProvenance
from bench.report import GateResult, build_report
from bench.restore_check import CheckResult, RestoreResult, RestoreSummary
from bench.run_bakeoff import (
    AgentConfig,
    TaskRunResult,
    _attempt_summary,
    build_agent_command,
    run_bakeoff,
    run_reference_equivalence,
    run_task,
    run_tasks_sequentially,
)
from bench.self_replay import SelfReplayResult
from bench.session import BakeoffSession
from bench.workspaces import prepare_exercise
from ds4py._json import is_json_object
from ds4py.engine.replay_types import ReplayRound, ReplayTrace


def config(tmp_path: Path, agent: Any = "ds4py") -> Any:
    executable = tmp_path / f"{agent}-fake.py"
    if not executable.exists():
        executable.write_text(FAKE_AGENT)
    return AgentConfig(
        agent=agent,
        executable=(sys.executable, str(executable)),
        model="/model.gguf",
        backend="cpu",
        ctx=100000,
        temperature=0,
        seed=1,
        tokens=256,
    )


def result(task_id: Any = "task", agent: Any = "ds4py", **changes: Any) -> Any:
    save = Path("/saves") / task_id / agent
    trace = Path("/traces") / task_id / agent
    metric_values: dict[str, Any] = {
        "wall_seconds": 10,
        "progress_events": 3,
        "tokens": 20,
        "tool_calls": 1,
        "avg_prefill_reuse": 0.8,
        "non_extend_rounds": 0,
        "terminal_reasons": ("stop_token",) if agent == "ds4py" else ("recorded_boundary",),
        "tool_accounting": ToolAccounting(1, 1) if agent == "ds4py" else None,
    }
    for name in EventSummary.__dataclass_fields__:
        if name in changes:
            metric_values[name] = changes.pop(name)
    base = TaskRunResult(
        task_id=task_id,
        agent=agent,
        passed=True,
        attempts=1,
        summary=EventSummary(**metric_values),
        attempt_save_dirs=(save,) if agent == "ds4py" else (),
        attempt_trace_paths=(trace,),
        grader_outputs=("grader secret output",),
    )
    return replace(base, **changes)


def with_summary(item: Any, **changes: Any) -> Any:
    return replace(item, summary=replace(item.summary, **changes))


def test_task_result_preserves_summary_evidence_and_derives_last_attempt_paths() -> None:
    summary = EventSummary(
        wall_seconds=2,
        progress_events=3,
        tokens=4,
        tool_calls=1,
        avg_prefill_reuse=0.625,
        non_extend_rounds=1,
        allowed_non_extend_rounds=1,
        non_extension_reasons=("compaction",),
        terminal_reasons=("stop_token",),
        reuse_ratios=(0.5, 0.75),
    )
    first_save = Path("/saves/first")
    last_save = Path("/saves/last")
    first_trace = Path("/traces/first")
    last_trace = Path("/traces/last")
    run = TaskRunResult(
        task_id="task",
        agent="ds4py",
        passed=True,
        attempts=2,
        summary=summary,
        attempt_save_dirs=(first_save, last_save),
        attempt_trace_paths=(first_trace, last_trace),
    )

    assert run.summary is summary
    assert run.non_extension_reasons == ("compaction",)
    assert run.reuse_ratios == (0.5, 0.75)
    assert run.save_dir == last_save
    assert run.trace_path == last_trace
    data = build_report([run], RestoreSummary()).to_dict()
    assert is_json_object(data)
    tasks = data["tasks"]
    assert isinstance(tasks, list) and all(is_json_object(item) for item in tasks)
    first = tasks[0]
    if not is_json_object(first):
        raise AssertionError("task record must be an object")
    summary = first["summary"]
    if not is_json_object(summary):
        raise AssertionError("task summary must be an object")
    assert summary["non_extension_reasons"] == ["compaction"]
    assert summary["terminal_reasons"] == ["stop_token"]
    assert summary["reuse_ratios"] == [0.5, 0.75]


def healthy_restore(results: Any) -> Any:
    runs = tuple(
        RestoreResult(save_dir=save, passed=True)
        for item in results
        if item.agent == "ds4py"
        for save in item.attempt_save_dirs
    )
    return RestoreSummary(
        runs=runs,
        interruption=CheckResult(True, "resumed"),
    )


def provenance() -> TraceProvenance:
    round_ = RoundProvenance(
        round_index=0,
        think=75,
        prompt_sha256="prompt-sha",
        sync_calls=(SyncCall("full", 12, 8, 4, 12),),
        progress_chunks=(ProgressChunk(4, 4),),
        observed_think=75,
    )
    return TraceProvenance(
        bakeoff_session_id="session-1",
        instance_id="instance-1",
        gpu_model="NVIDIA H100",
        gpu_driver_version="580.00",
        ds4_revision="ds4-revision",
        ds4py_revision="ds4py-revision",
        libds4_sha256="library-sha",
        model_path="/models/ds4.gguf",
        model_sha256="model-sha",
        backend="cuda",
        ctx=100000,
        q8_f16_cache_mb=5120,
        temperature=0.0,
        seed=1,
        response_budget=8192,
        reset_policy="fresh-per-round",
        trace_sha256="c-output-sha",
        source_trace_sha256="source-sha",
        rounds=(round_,),
        observed_libds4_sha256="library-sha",
        observed_model_path="/models/ds4.gguf",
        observed_backend="cuda",
        observed_ctx=100000,
        observed_seed=1,
        observed_temperature=0.0,
    )


def bakeoff_session() -> BakeoffSession:
    return BakeoffSession(
        bakeoff_session_id="session-1",
        instance_id="instance-1",
        gpu_model="cpu",
        gpu_driver_version="not-applicable",
        ds4_revision="ds4-revision",
        ds4py_revision="ds4py-revision",
        libds4_sha256="library-sha",
        model_path="/model.gguf",
        model_sha256="model-sha",
        backend="cpu",
        ctx=100000,
        temperature=0.0,
        seed=1,
        response_budget=256,
        q8_f16_cache_mb=5120,
    )


def test_session_environment_inherits_native_patch_set_and_host_register_value() -> None:
    session = replace(
        bakeoff_session(),
        native_patches=(
            SimpleNamespace(name="docs/patches/cuda-no-copy-host-register-opt-out.patch", sha256="patch-sha"),
        ),
        cuda_disable_host_register="1",
    )

    environment = session.environment()

    assert environment["DS4_BAKEOFF_NATIVE_PATCHES"] == "docs/patches/cuda-no-copy-host-register-opt-out.patch"
    assert environment["DS4_CUDA_DISABLE_HOST_REGISTER"] == "1"


def paired_evidence(results: Any) -> dict[str, Any]:
    task_ids = sorted({item.task_id for item in results if item.agent == "c"})
    rates = {
        task_id: (
            RateEvidence(task_id, 100, 2.0, 40, 4.0),
            RateEvidence(task_id, 100, 2.0, 40, 4.0),
        )
        for task_id in task_ids
    }
    c_provenance = provenance()
    ds4py_provenance = replace(c_provenance, trace_sha256="ds4py-output-sha")
    cached_fractions = dict.fromkeys(task_ids, (1 / 3, 1 / 3))
    return {
        "rate_evidence": rates,
        "cached_fractions": cached_fractions,
        "provenance": ((c_provenance, ds4py_provenance),),
    }


def test_agent_commands_share_model_sampling_and_task_prompt(tmp_path: Path) -> None:
    prompt = "solve this exact task"
    trace = tmp_path / "trace.jsonl"
    c_command = build_agent_command(config(tmp_path, "c"), prompt, trace)
    python = build_agent_command(config(tmp_path), prompt, trace)

    for command in (c_command, python):
        for flag, value in (
            ("--model", "/model.gguf"),
            ("--backend", "cpu"),
            ("--ctx", "100000"),
            ("--temp", "0"),
            ("--seed", "1"),
            ("--tokens", "256"),
            ("--prompt", prompt),
            ("--trace", str(trace)),
        ):
            assert command[command.index(flag) + 1] == value
    assert "--non-interactive" in c_command and "--yolo" not in c_command
    assert "--non-interactive" in python and "--yolo" not in python
    assert "--request-limit" not in c_command
    assert python[python.index("--request-limit") + 1] == "100"


def test_request_limit_failure_is_named_in_task_errors(tmp_path: Path) -> None:
    source = make_exercise(tmp_path)
    workspace = prepare_exercise(source, tmp_path / "staged", max_attempts=1)
    executable = tmp_path / "request-limit.py"
    executable.write_text(
        FAKE_AGENT
        + '\nimport sys\nsys.stderr.write("UsageLimitExceeded: request_limit=1\\n")\n'
        + "raise SystemExit(1)\n"
    )
    agent = replace(
        config(tmp_path),
        executable=(sys.executable, str(executable)),
    )

    completed = run_task(
        workspace,
        agent,
        tmp_path / "results",
        grade_runner=lambda *args: subprocess.CompletedProcess([], 0, stdout="pass", stderr=""),
    )

    assert "attempt 1 request_limit" in completed.errors


@pytest.mark.parametrize(
    ("reason", "sampled", "budget", "message"),
    [
        ("max_tokens", (1,), 2, "max_tokens has 1 Token events, expected response budget 2"),
        ("stop_token", (1, 2), 1, "has 2 Token events, exceeding response budget 1"),
    ],
)
def test_ds4py_event_metrics_enforce_response_budget(
    tmp_path: Path,
    reason: str,
    sampled: tuple[int, ...],
    budget: int,
    message: str,
) -> None:
    events_path = tmp_path / "events.jsonl"
    records = [
        {"kind": "run_start", "run_id": "run-1", "payload": {}},
        *({"kind": "Token", "run_id": "run-1", "payload": {"event": {"id": token}}} for token in sampled),
        {"kind": "Stopped", "run_id": "run-1", "payload": {"event": {"reason": reason}}},
        {"kind": "run_end", "run_id": "run-1", "payload": {"status": "success"}},
    ]
    events_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "session_id": "session-1",
                "run_id": "run-1",
                "step_index": 0,
                "round_index": 0,
                "prompt": [1, 2],
                "think": 0,
                "cached": 0,
                "prefill_suffix": [1, 2],
                "sampled": list(sampled),
            }
        )
        + "\n"
    )

    with pytest.raises(MetricError, match=message):
        _attempt_summary(
            replace(config(tmp_path), tokens=budget),
            events_path,
            trace_path,
            "",
            1.0,
        )


def test_expected_max_tokens_exit_is_not_lifecycle_error(tmp_path: Path) -> None:
    source = make_exercise(tmp_path)
    workspace = prepare_exercise(source, tmp_path / "staged", max_attempts=1)
    executable = tmp_path / "max-tokens.py"
    executable.write_text(
        FAKE_AGENT.replace('{"reason": "stop_token"}', '{"reason": "max_tokens"}')
        + '\nimport sys\nsys.stderr.write("ds4py: model response ended with max_tokens\\n")\n'
        + "raise SystemExit(1)\n"
    )
    agent = replace(
        config(tmp_path),
        executable=(sys.executable, str(executable)),
        tokens=1,
    )

    completed = run_task(
        workspace,
        agent,
        tmp_path / "results",
        grade_runner=lambda *args: subprocess.CompletedProcess([], 0, stdout="pass", stderr=""),
    )

    assert completed.terminal_reasons == ("max_tokens",)
    assert completed.errors == ()
    report = build_report([completed], healthy_restore([completed]))
    assert report.gates["gate-2"].passed
    assert report.caps.totals.ds4py == 1


def test_failed_first_attempt_feeds_grader_output_to_exactly_one_retry(
    tmp_path: Path,
) -> None:
    source = make_exercise(tmp_path)
    workspace = prepare_exercise(source, tmp_path / "staged")
    outcomes = iter(
        [
            subprocess.CompletedProcess([], 1, stdout="hidden failure", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="hidden pass", stderr=""),
        ]
    )

    completed = run_task(
        workspace,
        config(tmp_path),
        tmp_path / "results",
        grade_runner=lambda *args: next(outcomes),
    )

    assert completed.attempts == 2 and completed.passed
    assert completed.grader_outputs == ("hidden failure", "hidden pass")
    assert len(completed.attempt_save_dirs) == 2
    assert all(path.is_dir() for path in completed.attempt_save_dirs)
    assert completed.tokens == 2
    invocations = read_invocations(tmp_path)
    assert len(invocations) == 2
    assert "hidden failure" not in invocations[0]["prompt"]
    assert "hidden failure" in invocations[1]["prompt"]


def test_relative_results_root_keeps_attempt_artifacts_at_invocation_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_exercise(tmp_path)
    workspace = prepare_exercise(source, tmp_path / "staged")
    monkeypatch.chdir(tmp_path)

    completed = run_task(
        workspace,
        config(tmp_path),
        Path("results"),
        grade_runner=lambda command, cwd, env: subprocess.CompletedProcess(command, 0, "pass", ""),
    )

    assert completed.errors == ()
    assert completed.trace_path is not None
    assert completed.trace_path.is_absolute()
    assert completed.trace_path.is_file()
    assert completed.trace_path.is_relative_to(tmp_path / "results")
    assert not (workspace.agent_dir / "results").exists()


def test_c_and_python_run_sequentially_on_equivalent_tasks(tmp_path: Path) -> None:
    source = make_exercise(tmp_path)
    staged = prepare_exercise(source, tmp_path / "staged")

    results = run_tasks_sequentially(
        [staged],
        [config(tmp_path, "c"), config(tmp_path)],
        tmp_path / "results",
        grade_runner=lambda command, cwd, env: subprocess.CompletedProcess(command, 0, "pass", ""),
    )

    assert [(item.task_id, item.agent) for item in results] == [
        ("tiny", "c"),
        ("tiny", "ds4py"),
    ]
    assert all(item.passed for item in results)
    invocations = read_invocations(tmp_path)
    assert [item["agent"] for item in invocations] == ["c", "ds4py"]
    assert results[0].tool_accounting is None
    assert not (tmp_path / "concurrency-violation").exists()
    assert not (results[0].save_dir or Path("missing")).exists() if results[0].save_dir else True


def test_sequential_runner_reports_each_task_start_and_finish(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = make_exercise(tmp_path)
    staged = prepare_exercise(source, tmp_path / "staged")

    run_tasks_sequentially(
        [staged],
        [config(tmp_path, "c"), config(tmp_path)],
        tmp_path / "results",
        grade_runner=lambda command, cwd, env: subprocess.CompletedProcess(command, 0, "pass", ""),
    )

    assert capsys.readouterr().out.splitlines() == [
        "[bakeoff] start c/tiny",
        "[bakeoff] finish c/tiny: PASS",
        "[bakeoff] start ds4py/tiny",
        "[bakeoff] finish ds4py/tiny: PASS",
    ]


def test_c_metrics_failure_leaves_tool_accounting_unavailable(tmp_path: Path) -> None:
    source = make_exercise(tmp_path)
    workspace = prepare_exercise(source, tmp_path / "staged")
    executable = tmp_path / "c-without-evidence.py"
    executable.write_text("")
    c_config = replace(
        config(tmp_path, "c"),
        executable=(sys.executable, str(executable)),
    )

    completed = run_task(
        workspace,
        c_config,
        tmp_path / "results",
        grade_runner=lambda command, cwd, env: subprocess.CompletedProcess(command, 0, "pass", ""),
    )

    assert completed.errors
    assert completed.tool_accounting is None


def test_gates_one_to_five_pass_with_complete_evidence() -> None:
    c = result(agent="c", wall_seconds=10)
    python = result(wall_seconds=15, non_extend_rounds=2)
    python = with_summary(python, allowed_non_extend_rounds=2)
    results = [c, python]
    report = build_report(results, healthy_restore(results), **paired_evidence(results))

    assert all(report.gates[f"gate-{number}"].passed for number in range(1, 6))
    assert not report.gates["gate-6"].passed
    assert "not run" in report.gates["gate-6"].detail


def test_gate_one_allows_exactly_one_task_of_pass_slack() -> None:
    c = [result(f"task-{index}", "c") for index in range(3)]
    python = [result(f"task-{index}", passed=index != 0) for index in range(3)]
    assert build_report(c + python, healthy_restore(c + python)).gates["gate-1"].passed
    python[1] = replace(python[1], passed=False)
    report = build_report(c + python, healthy_restore(c + python))
    assert not report.gates["gate-1"].passed
    assert "attempt" in report.gates["gate-1"].detail


def test_gate_two_allows_bad_argument_retry_but_not_parser_error_or_missing_outcome() -> None:
    c = result(agent="c")
    python = with_summary(result(), tool_accounting=ToolAccounting(2, 1, 0, 1))
    assert build_report([c, python], healthy_restore([c, python])).gates["gate-2"].passed

    parser_error = with_summary(python, parser_errors=1)
    assert not build_report([c, parser_error], healthy_restore([c, parser_error])).gates["gate-2"].passed
    missing = with_summary(python, tool_accounting=ToolAccounting(2, 1, 0, 0))
    invalid = build_report([c, missing], healthy_restore([c, missing])).gates["gate-2"]
    assert not invalid.passed
    assert not invalid.valid


def test_gate_two_rejects_missing_unknown_and_inconsistent_terminal_evidence() -> None:
    c = result(agent="c")
    python = result()
    results = [c, python]
    evidence = healthy_restore(results)

    for reasons in ((), ("mystery",)):
        changed = with_summary(python, terminal_reasons=reasons)
        gate = build_report([c, changed], evidence).gates["gate-2"]
        assert not gate.passed
        assert not gate.valid

    inconsistent = with_summary(
        python,
        terminal_reasons=("tool_call_complete",),
        tool_accounting=ToolAccounting(),
    )
    gate = build_report([c, inconsistent], evidence).gates["gate-2"]
    assert not gate.passed
    assert not gate.valid


def test_gate_two_accepts_max_tokens_and_reports_caps_without_parity() -> None:
    c = result(agent="c")
    python = with_summary(result(), terminal_reasons=("max_tokens", "max_tokens"))
    results = [c, python]

    report = build_report(results, healthy_restore(results))

    gate = report.gates["gate-2"]
    assert gate.passed and gate.valid
    assert report.caps.tasks["task"].c == 0
    assert report.caps.tasks["task"].ds4py == 2
    assert report.caps.totals.ds4py == 2
    assert report.caps.parity.detail == "not run"


def test_gate_two_enforces_explicit_cap_parity() -> None:
    c = result(agent="c")
    python = result()
    results = [c, python]

    report = build_report(
        results,
        healthy_restore(results),
        cap_parity=GateResult(False, "cap outcomes differ"),
    )

    assert not report.gates["gate-2"].passed
    assert "cap outcomes differ" in report.gates["gate-2"].detail


def test_cap_phase_runs_both_drivers_for_each_arm_round_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bench.phases as phase_runner
    import bench.run_bakeoff as runner

    c_trace = tmp_path / "c.trace"
    ds4py_trace = tmp_path / "ds4py.trace"
    c_trace.write_text("c")
    ds4py_trace.write_text("ds4py")
    monkeypatch.setattr(phase_runner, "select_c_traces", lambda _results: [("task", c_trace)])
    monkeypatch.setattr(phase_runner, "select_ds4py_traces", lambda _results: [("task", ds4py_trace)])
    trace = ReplayTrace(100000, None, (ReplayRound(0, (1, 2), (3,), 75, 0, False),))
    monkeypatch.setattr(phase_runner, "read_replay", lambda _path: trace)
    calls = []

    def fake_driver(command: list[str], output_path: Path, log_path: Path) -> Any:
        case = read_case_manifest(Path(command[command.index("--cases") + 1]))[0]
        source_trace = Path(command[command.index("--source-trace") + 1])
        output_path.write_text("{}")
        log_path.write_text("")
        calls.append((command[0], source_trace, case))
        return (
            (
                DriverRoundEvidence(
                    source_trace=source_trace,
                    case=case,
                    sync_calls=(SyncCall("full", 2, 0, 2, 2),),
                    progress_chunks=(ProgressChunk(2, 2),),
                    steps=(),
                    prefill_seconds=0.0,
                    decode_seconds=0.0,
                    terminal_reason="stop_token",
                    observed_libds4_sha256="library-sha",
                    observed_model_path="/model.gguf",
                    observed_backend="cpu",
                    observed_ctx=100000,
                    observed_seed=1,
                    observed_temperature=0.0,
                    observed_think=case.think,
                ),
            ),
            (),
        )

    monkeypatch.setattr(phase_runner, "_run_driver_command", fake_driver)
    c_driver = tmp_path / "c-driver"

    result = runner._run_cap_parity([], config(tmp_path), c_driver, bakeoff_session(), tmp_path / "results")

    assert result.passed, result.detail
    assert len(calls) == 4
    assert {call[1] for call in calls} == {c_trace, ds4py_trace}
    assert [call[0] for call in calls].count(str(c_driver)) == 2
    assert [call[0] for call in calls].count(sys.executable) == 2
    assert all(call[2].round_index == 0 and call[2].steps == 256 for call in calls)


def test_reference_selection_failure_invalidates_explicit_cap_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bench.phases as phase_runner
    import bench.run_bakeoff as runner
    from bench.trace_selection import ReplaySelectionError

    def fail_selection(_results: Any) -> Any:
        raise ReplaySelectionError("missing C traces")

    monkeypatch.setattr(phase_runner, "select_c_traces", fail_selection)

    phase = runner.run_reference_equivalence(
        [],
        config(tmp_path),
        tmp_path / "c-driver",
        bakeoff_session(),
        tmp_path / "results",
        cap_parity=True,
    )

    assert phase.cap_parity is not None
    assert not phase.cap_parity.passed and not phase.cap_parity.valid
    assert "missing C traces" in phase.cap_parity.detail


def test_gate_two_rejects_unexpected_task_interruption() -> None:
    c = result(agent="c")
    python = with_summary(result(), terminal_reasons=("interrupted",))

    gate = build_report([c, python], healthy_restore([c, python])).gates["gate-2"]

    assert not gate.passed
    assert gate.valid


def test_gate_three_ignores_wall_time_ratio_with_valid_paired_evidence() -> None:
    c = result(agent="c", wall_seconds=10)
    python = result(wall_seconds=100)
    results = [c, python]

    gate = build_report(results, healthy_restore(results), **paired_evidence(results)).gates["gate-3"]

    assert gate.passed and gate.valid


def test_gate_three_enforces_rate_and_cached_fraction_tolerances() -> None:
    c = result(agent="c")
    python = result()
    results = [c, python]
    evidence = paired_evidence(results)
    c_rate, ds4py_rate = evidence["rate_evidence"]["task"]
    evidence["rate_evidence"]["task"] = (c_rate, replace(ds4py_rate, prefill_seconds=2.70))

    bad_rate = build_report(results, healthy_restore(results), **evidence).gates["gate-3"]
    assert not bad_rate.passed
    assert bad_rate.valid

    evidence = paired_evidence(results)
    evidence["cached_fractions"]["task"] = (0.9, 0.5)
    bad_cache = build_report(results, healthy_restore(results), **evidence).gates["gate-3"]
    assert not bad_cache.passed
    assert bad_cache.valid


def test_gate_three_cache_parity_ignores_unpaired_live_trajectories() -> None:
    c = with_summary(result(agent="c"), avg_prefill_reuse=0.9)
    python = with_summary(result(), avg_prefill_reuse=0.5)
    results = [c, python]

    gate = build_report(results, healthy_restore(results), **paired_evidence(results)).gates["gate-3"]

    assert gate.passed and gate.valid


def test_gate_three_rejects_unexplained_non_extensions() -> None:
    c = result(agent="c")
    python = with_summary(result(), invalid_non_extend_rounds=1, non_extend_rounds=1)
    results = [c, python]

    gate = build_report(results, healthy_restore(results), **paired_evidence(results)).gates["gate-3"]

    assert not gate.passed
    assert gate.valid


def test_gate_three_missing_rates_or_bad_provenance_is_invalid() -> None:
    c = result(agent="c")
    python = result()
    results = [c, python]
    restore = healthy_restore(results)

    missing = build_report(results, restore)
    assert not missing.gates["gate-3"].passed
    assert not missing.gates["gate-3"].valid
    assert "gate-3: INVALID" in missing.compact_table()

    evidence = paired_evidence(results)
    c_provenance, ds4py_provenance = evidence["provenance"][0]
    evidence["provenance"] = ((c_provenance, replace(ds4py_provenance, instance_id="other-instance")),)
    mismatched = build_report(results, restore, **evidence).gates["gate-3"]
    assert not mismatched.passed
    assert not mismatched.valid


def test_gate_four_requires_interrupt_resume() -> None:
    results = [result(agent="c"), result()]
    restore = healthy_restore(results)
    restore = replace(restore, interruption=CheckResult(False, "no resume"))
    report = build_report(results, restore)
    assert not report.gates["gate-4"].passed
    assert "no resume" in report.gates["gate-4"].detail


def test_gate_five_requires_restore_for_every_attempt() -> None:
    python = replace(result(attempts=2), attempts=2)
    saves = (Path("/saves/first"), Path("/saves/second"))
    python = replace(python, attempts=2, attempt_save_dirs=saves)
    c = result(agent="c")
    restore = healthy_restore([c, python])
    restore = replace(restore, runs=restore.runs[:1])

    assert not build_report([c, python], restore).gates["gate-5"].passed


def test_gate_five_rejects_an_attempt_with_no_save_directory() -> None:
    c = result(agent="c")
    save = Path("/saves/only-one")
    python = replace(result(), attempts=2, attempt_save_dirs=(save,))
    restore = healthy_restore([c, python])

    report = build_report([c, python], restore)

    assert not report.gates["gate-5"].passed
    assert "attempt" in report.gates["gate-5"].detail


def test_report_json_has_attempt_feedback_and_table_has_deltas_not_logs() -> None:
    c = result(agent="c", wall_seconds=10, progress_events=2, tokens=10)
    python = result(wall_seconds=12, progress_events=5, tokens=14)
    results = [c, python]
    report = build_report(results, healthy_restore(results), **paired_evidence(results))

    data = report.to_dict()
    assert is_json_object(data)
    assert "rates" in data and "caps" in data and "provenance" in data
    rates = data["rates"]
    assert is_json_object(rates) and "task" in rates
    caps = data["caps"]
    assert is_json_object(caps) and "totals" in caps
    tasks = data["tasks"]
    assert isinstance(tasks, list) and all(is_json_object(item) for item in tasks)
    python_data = next(item for item in tasks if is_json_object(item) and item["agent"] == "ds4py")
    if not is_json_object(python_data):
        raise AssertionError("ds4py task record must be an object")
    assert python_data["grader_outputs"] == ["grader secret output"]
    table = report.compact_table()
    for metric in (
        "wall_seconds",
        "progress_events",
        "tokens",
        "tool_calls",
        "avg_prefill_reuse",
        "non_extend_rounds",
        "max_tokens_exits",
        "paired_rates",
    ):
        assert metric in table
    assert "delta" in table and "grader secret output" not in table


def make_exercise(root: Any) -> Any:
    source = root / "benchmark/python/exercises/practice/tiny"
    (source / ".docs").mkdir(parents=True)
    (source / ".meta").mkdir()
    (source / ".docs/instructions.md").write_text("Return good.\n")
    (source / ".meta/config.json").write_text(
        json.dumps(
            {
                "files": {
                    "solution": ["solution.py"],
                    "test": ["tiny_test.py"],
                    "example": [".meta/example.py"],
                }
            }
        )
    )
    (source / ".meta/example.py").write_text("def value():\n    return 'good'\n")
    (source / "solution.py").write_text("def value():\n    return 'good'\n")
    (source / "tiny_test.py").write_text("def test_value():\n    assert True\n")
    return source


def read_invocations(root: Any) -> Any:
    path = root / "invocations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


FAKE_AGENT = r"""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

args = sys.argv[1:]
def option(name):
    return args[args.index(name) + 1]

root = Path(os.environ["BAKEOFF_ATTEMPT_ROOT"])
root.mkdir(parents=True, exist_ok=True)
agent = os.environ["BAKEOFF_AGENT_NAME"]
lock_path = Path(os.environ["BAKEOFF_LOCK_PATH"])
lock_path.parent.mkdir(parents=True, exist_ok=True)
with lock_path.open("w") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_path.with_name("concurrency-violation").write_text("overlap")
        raise
    time.sleep(0.02)
    events = [
        {"kind": "run_start", "run_id": "run-1", "payload": {}},
        {"kind": "Synced", "run_id": "run-1", "payload": {"event": {
            "plan": {"kind": "rebuild", "reused": 0, "to_prefill": 2}
        }}},
        {"kind": "Prefill", "run_id": "run-1", "payload": {"event": {"done": 2, "total": 2}}},
        {"kind": "Token", "run_id": "run-1", "payload": {"event": {"id": 1}}},
        {"kind": "Stopped", "run_id": "run-1", "payload": {"event": {"reason": "stop_token"}}},
        {"kind": "run_end", "run_id": "run-1", "payload": {"status": "success"}},
    ]
    events_text = "\n".join(json.dumps(item) for item in events) + "\n"
    Path(os.environ["BAKEOFF_EVENTS_PATH"]).write_text(events_text)
    trace = {
        "session_id": "fake", "run_id": "run-1", "step_index": 0,
        "round_index": 0, "prompt": [1, 2], "think": 0, "cached": 0,
        "prefill_suffix": [1, 2], "sampled": [1],
    }
    Path(option("--trace")).write_text(json.dumps(trace) + "\n")
    if agent == "ds4py":
        save = Path(os.environ["DS4PY_HOME"]) / "sessions" / ("save-" + os.environ["BAKEOFF_ATTEMPT"])
        save.mkdir(parents=True)
        (save / "save.json").write_text("{}")
        (save / "events.jsonl").write_text(events_text)
    invocation = {"agent": agent, "prompt": option("--prompt"), "argv": args}
    with Path(os.environ["BAKEOFF_INVOCATIONS"]).open("a") as stream:
        stream.write(json.dumps(invocation) + "\n")
    fcntl.flock(lock, fcntl.LOCK_UN)
"""


def test_complete_runner_rejects_missing_attempt_library_before_task_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bench.run_bakeoff as runner

    staged = [SimpleNamespace(task_id="repo-glm-close")]
    monkeypatch.setattr(runner, "load_manifest", lambda _path: object())
    monkeypatch.setattr(runner, "prepare_manifest", lambda *args: staged)
    monkeypatch.setattr(
        runner,
        "run_tasks_sequentially",
        lambda *args, **kwargs: pytest.fail("a task arm started without a resolvable attempt library"),
    )
    monkeypatch.delenv("DS4_LIB", raising=False)
    monkeypatch.setenv("DS4PY_HOME", str(tmp_path / "native"))

    with pytest.raises(RuntimeError, match="Set DS4_LIB"):
        run_bakeoff(
            tmp_path / "manifest",
            tmp_path / "benchmark",
            tmp_path,
            (config(tmp_path),),
            tmp_path / "results",
            session=bakeoff_session(),
            c_driver=tmp_path / "c-driver",
        )


def test_complete_runner_preserves_missing_attempt_trace_as_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bench.phases as phase_runner
    import bench.run_bakeoff as runner
    import bench.self_replay as replay_module

    missing = tmp_path / "missing-attempt.jsonl"
    task = result(
        task_id="repo-glm-close",
        passed=False,
        attempt_save_dirs=(),
        attempt_trace_paths=(missing,),
        errors=("attempt trace is unavailable",),
    )
    staged = [SimpleNamespace(task_id="repo-glm-close")]
    library = tmp_path / "libds4.so"
    library.write_bytes(b"native")
    monkeypatch.setenv("DS4_LIB", str(library))
    monkeypatch.setattr(runner, "load_manifest", lambda _path: object())
    monkeypatch.setattr(runner, "prepare_manifest", lambda *args: staged)
    monkeypatch.setattr(runner, "run_tasks_sequentially", lambda *args, **kwargs: (task,))
    monkeypatch.setattr(phase_runner, "check_result_saves", lambda *args, **kwargs: ())
    monkeypatch.setattr(runner, "clone_workspace", lambda *args: object())
    monkeypatch.setattr(runner, "interruption_acceptance", lambda *args: CheckResult(True, "ok"))

    @contextmanager
    def fake_open(*args: Any, **kwargs: Any) -> Any:
        yield object()

    monkeypatch.setattr(replay_module.ds4py, "open", fake_open)

    report = run_bakeoff(
        tmp_path / "manifest",
        tmp_path / "benchmark",
        tmp_path,
        (config(tmp_path),),
        tmp_path / "results",
        session=bakeoff_session(),
        c_driver=tmp_path / "c-driver",
    )

    assert report.self_replay is not None and len(report.self_replay) == 1
    evidence = report.self_replay[0]
    assert not evidence.valid
    assert str(missing) in evidence.detail
    assert not report.gates["gate-6"].valid
    assert evidence.detail in report.gates["gate-6"].detail
    sidecar = json.loads(missing.with_name(f"{missing.name}.provenance.json").read_text())
    assert sidecar["invalid_reason"] == evidence.detail
    written = json.loads((tmp_path / "results/bakeoff.json").read_text())
    assert written["gates"]["gate-6"]["valid"] is False
    assert written["self_replay"][0]["detail"] == evidence.detail
    assert "gate-6: INVALID" in (tmp_path / "results/report.txt").read_text()


def test_complete_runner_writes_gate_six_after_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bench.phases as phase_runner
    import bench.run_bakeoff as runner

    order = []
    staged = [SimpleNamespace(task_id="repo-glm-close")]
    session = bakeoff_session()
    library = tmp_path / "libds4.so"
    library.write_bytes(b"native")
    monkeypatch.setenv("DS4_LIB", str(library))
    monkeypatch.setattr(runner, "load_manifest", lambda _path: object())
    monkeypatch.setattr(runner, "prepare_manifest", lambda *args: staged)
    monkeypatch.setattr(
        runner,
        "run_tasks_sequentially",
        lambda *args, **kwargs: order.append("tasks") or (),
    )
    monkeypatch.setattr(runner, "record_task_provenance", lambda *args: {})
    monkeypatch.setattr(
        phase_runner,
        "check_result_saves",
        lambda *args, **kwargs: order.append("restores") or (),
    )
    monkeypatch.setattr(runner, "clone_workspace", lambda *args: object())
    monkeypatch.setattr(
        runner,
        "interruption_acceptance",
        lambda *args: order.append("interruption") or CheckResult(True, "ok"),
    )
    driver = DriverEquivalenceResult(
        rounds=(DriverRoundResult("repo-glm-close", 1, 0, True, "reference logits match"),),
        decode_steps=1,
        passed=True,
        detail="reference drivers match",
    )
    phase = SimpleNamespace(
        result=driver,
        rate_evidence={},
        cached_fractions={},
        provenance=(),
        cap_parity=GateResult(True, "round-0 cap outcomes match"),
    )
    reference_options = {}

    def fake_reference(*args: Any, **kwargs: Any) -> Any:
        reference_options.update(kwargs)
        order.append("reference")
        return phase

    monkeypatch.setattr(runner, "run_reference_equivalence", fake_reference)
    monkeypatch.setattr(
        runner,
        "replay_ds4py_task_results",
        lambda *args, **kwargs: (
            order.append("self-replay") or (SelfReplayResult("repo-glm-close", 0, True, "self replay match"),)
        ),
    )

    report = run_bakeoff(
        tmp_path / "manifest",
        tmp_path / "benchmark",
        tmp_path,
        (config(tmp_path),),
        tmp_path / "results",
        session=session,
        c_driver=tmp_path / "c-driver",
        cap_parity=True,
    )

    assert reference_options["cap_parity"] is True
    assert report.caps.parity.passed
    assert report.caps.parity.detail == "round-0 cap outcomes match"
    assert order == ["tasks", "restores", "interruption", "reference", "self-replay"]
    assert report.gates["gate-6"].passed
    assert report.schema_version == 2
    assert report.certification == "targeted"
    assert report.bakeoff_session_id == "session-1"
    assert report.instance_id == "instance-1"
    written = json.loads((tmp_path / "results/bakeoff.json").read_text())
    assert written["schema_version"] == 2
    assert written["certification"] == "targeted"
    assert written["bakeoff_session_id"] == "session-1"
    assert written["instance_id"] == "instance-1"
    assert "gate-6: PASS" in (tmp_path / "results/report.txt").read_text()


def test_reference_phase_runs_sequentially_and_records_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bench.phases as phase_runner

    session = bakeoff_session()
    trace = tmp_path / "c.trace"
    trace.write_text("raw C trace")
    token_file = tmp_path / "case.tokens"
    token_file.write_text("1\n2\n3\n4\n")
    task = replace(result(agent="c"), attempt_trace_paths=(trace,))
    capture = StepCapture(3, tuple(TopLogit(index, 12.0 - index) for index in range(12)), 0)
    evidence = DriverRoundEvidence(
        source_trace=trace,
        case=DriverCase("task", 1, 0, token_file, 3, 1, 1, 2),
        sync_calls=(SyncCall("prefix", 1, 0, 1, 1), SyncCall("full", 3, 1, 2, 3)),
        progress_chunks=(ProgressChunk(1, 1), ProgressChunk(2, 2)),
        steps=(capture,),
        prefill_seconds=1.0,
        decode_seconds=2.0,
        terminal_reason="recorded_boundary",
        observed_libds4_sha256="library-sha",
        observed_model_path="/model.gguf",
        observed_backend="cpu",
        observed_ctx=100000,
        observed_seed=1,
        observed_temperature=0.0,
        observed_think=2,
    )
    calls = []
    monkeypatch.setattr(
        phase_runner,
        "read_replay",
        lambda _path: SimpleNamespace(model=session.model_path, ctx=session.ctx, rounds=(object(),)),
    )

    def export(*args: Any, **kwargs: Any) -> Path:
        destination = cast("Path", kwargs["destination"])
        manifest = destination / "cases.tsv"
        manifest.write_text("cases")
        return manifest

    def drive(command: list[str], output_path: Path, log_path: Path) -> Any:
        label = "c" if command[0].endswith("c-driver") else "ds4py"
        calls.append(label)
        output_path.write_text("{}\n")
        log_path.write_text("")
        return (replace(evidence, terminal_reason="recorded_boundary" if label == "c" else "stop_token"),), ()

    monkeypatch.setattr(phase_runner, "export_trace_cases", export)
    monkeypatch.setattr(phase_runner, "_run_driver_command", drive)

    phase = run_reference_equivalence(
        [task],
        config(tmp_path),
        tmp_path / "c-driver",
        session,
        tmp_path / "reference",
    )

    assert calls == ["c", "ds4py"]
    assert phase.result.passed, phase.result.detail
    assert phase.result.valid
    assert phase.result.decode_steps == 1
    assert phase.rate_evidence["task"][0].decode_steps == 1
    assert phase.rate_evidence["task"][1].decode_steps == 1
    assert phase.cached_fractions["task"] == pytest.approx((1 / 3, 1 / 3))
    assert len(phase.provenance) == 1
    assert len(list((tmp_path / "reference").rglob("*.provenance.json"))) == 2
