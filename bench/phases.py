"""Bakeoff phase orchestration outside the task-arm loop."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from bench.comparison import RateEvidence
    from bench.driver_cases import DriverRoundEvidence
    from bench.models import AgentConfig, TaskRunResult
    from bench.provenance import TraceProvenance
    from bench.restore_check import CheckResult, RestoreResult
    from bench.self_replay import SelfReplayResult
    from bench.session import BakeoffSession
    from bench.workspaces import Workspace
    from ds4py.engine.replay_types import ReplayTrace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.cap_parity import compare_cap_parity
from bench.driver_cases import DriverEvidenceError, read_driver_ndjson
from bench.driver_equivalence import (
    DriverEquivalenceResult,
    DriverRoundResult,
    cached_fractions_by_task,
    combine_driver_results,
    compare_driver_rounds,
    rates_by_task,
)
from bench.provenance import write_trace_provenance
from bench.pty_driver import drive_interruption_pty
from bench.report import GateResult
from bench.restore_check import check_result_saves
from bench.self_replay import SelfReplayContext, replay_ds4py_task_results
from bench.trace_selection import ReplaySelectionError


def run_restore_phase(
    task_results: Sequence[TaskRunResult],
    config: AgentConfig,
    results_root: Path,
    c_driver: str | Path,
    *,
    check: Callable[..., tuple[RestoreResult, ...]] = check_result_saves,
) -> tuple[RestoreResult, ...]:
    """Restore every ds4-py save through the existing Gate 5 driver."""
    return check(
        task_results,
        config,
        results_root / "restores",
        c_driver=c_driver,
    )


@dataclass(frozen=True)
class ReferenceOperations:
    select_c: Callable[..., list[tuple[str, Path]]]
    select_ds4py: Callable[..., list[tuple[str, Path]]]
    read_trace: Callable[..., ReplayTrace]
    export_cases: Callable[..., Path]
    export_cap_case: Callable[..., Path]
    run_driver: Callable[..., tuple[tuple[DriverRoundEvidence, ...], tuple[str, ...]]]


def _acceptance_environment(home: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["DS4PY_HOME"] = str(home)
    return environment


def interruption_acceptance(
    config: AgentConfig,
    workspace: Workspace,
    results_root: str | Path,
    *,
    build_command: Callable[..., list[str]],
    drive: Callable[..., CheckResult] = drive_interruption_pty,
) -> CheckResult:
    """Interrupt visible generation, submit a next prompt, then exit cleanly."""
    root = Path(results_root) / "gate-4" / "interruption"
    root.mkdir(parents=True, exist_ok=True)
    prompt = (workspace.agent_dir / "TASK.md").read_text()
    command = build_command(config, prompt, root / "trace.jsonl")
    command.remove("--non-interactive")
    result = drive(
        command,
        home=root / "home",
        follow_up="State the next concrete action in one sentence, then stop.",
        cwd=workspace.agent_dir,
        env=_acceptance_environment(root / "home"),
        timeout=180,
    )
    (root / "interactive.log").write_text(result.detail)
    return result


@dataclass(frozen=True)
class ReferencePhase:
    result: DriverEquivalenceResult
    rate_evidence: dict[str, tuple[RateEvidence, RateEvidence]]
    provenance: tuple[tuple[TraceProvenance, TraceProvenance], ...]
    cap_parity: GateResult | None = None
    cached_fractions: dict[str, tuple[float, float]] = field(default_factory=dict)


def _reference_failure(task_id: str, attempt: int, detail: str) -> DriverEquivalenceResult:
    return DriverEquivalenceResult(
        rounds=(DriverRoundResult(task_id, attempt, -1, passed=False, detail=detail, valid=False),),
        decode_steps=0,
        passed=False,
        detail=detail,
        valid=False,
    )


def _run_driver_command(
    command: list[str],
    output_path: Path,
    log_path: Path,
) -> tuple[tuple[DriverRoundEvidence, ...], tuple[str, ...]]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output_path.write_text(completed.stdout)
    log_path.write_text(completed.stderr)
    issues = [f"driver exited {completed.returncode}"] if completed.returncode else []
    try:
        rounds = read_driver_ndjson(output_path)
    except (OSError, DriverEvidenceError) as error:
        return (), (*issues, f"invalid NDJSON evidence: {error}")
    return rounds, tuple(issues)


def _reference_commands(
    config: AgentConfig,
    c_driver: str | Path,
    manifest: Path,
    trace_path: Path,
) -> tuple[list[str], list[str]]:
    common = [
        "--model",
        config.model,
        "--cases",
        str(manifest),
        "--source-trace",
        str(trace_path),
        "--backend",
        config.backend,
        "--ctx",
        str(config.ctx),
    ]
    return (
        [str(c_driver), *common],
        [sys.executable, str(Path(__file__).with_name("ds4py_reference_driver.py")), *common],
    )


def _validate_reference_trace(trace: ReplayTrace, config: AgentConfig) -> None:
    if (
        trace.model is not None
        and Path(trace.model).expanduser().resolve() != Path(config.model).expanduser().resolve()
    ):
        raise DriverEvidenceError("C trace model differs from the reference model")
    if trace.ctx is not None and trace.ctx != config.ctx:
        raise DriverEvidenceError("C trace context differs from the reference context")


def _run_cap_parity(  # noqa: PLR0913 - Original inputs and patchable runner operations stay explicit.
    task_results: Sequence[TaskRunResult],
    config: AgentConfig,
    c_driver: str | Path,
    session: BakeoffSession,
    results_root: str | Path,
    *,
    operations: ReferenceOperations,
) -> GateResult:
    """Run both reference drivers on each arm's round-0 prompt."""
    try:
        selected = [
            *(("c", task_id, path) for task_id, path in operations.select_c(task_results)),
            *(("ds4py", task_id, path) for task_id, path in operations.select_ds4py(task_results)),
        ]
    except (ReplaySelectionError, OSError, ValueError) as error:
        return GateResult(passed=False, detail=f"cap trace selection failed: {error}", valid=False)

    pairs = []
    provenance_pairs = []
    issues = []
    attempts: dict[tuple[str, str], int] = {}
    for arm, task_id, trace_path in selected:
        key = (arm, task_id)
        attempts[key] = attempts.get(key, 0) + 1
        attempt = attempts[key]
        label = f"{arm} {task_id} attempt {attempt}"
        destination = Path(results_root) / "cap-parity" / f"{arm}-{task_id}-attempt-{attempt}"
        try:
            trace = operations.read_trace(trace_path)
            _validate_reference_trace(trace, config)
            manifest = operations.export_cap_case(
                trace,
                task_id=task_id,
                attempt=attempt,
                destination=destination,
                response_budget=config.tokens,
            )
            c_output = destination / "c.ndjson"
            ds4py_output = destination / "ds4py.ndjson"
            c_command, ds4py_command = _reference_commands(config, c_driver, manifest, trace_path)
            c_rounds, c_issues = operations.run_driver(c_command, c_output, destination / "c.log")
            ds4py_rounds, ds4py_issues = operations.run_driver(
                ds4py_command,
                ds4py_output,
                destination / "ds4py.log",
            )
            issues.extend(f"{label} C driver: {issue}" for issue in c_issues)
            issues.extend(f"{label} ds4-py driver: {issue}" for issue in ds4py_issues)
            if len(c_rounds) != 1 or len(ds4py_rounds) != 1:
                issues.append(f"{label}: expected one round per driver")
                continue
            c_provenance = session.reference_provenance(c_output, trace_path, c_rounds)
            ds4py_provenance = session.reference_provenance(ds4py_output, trace_path, ds4py_rounds)
            write_trace_provenance(c_output, c_provenance)
            write_trace_provenance(ds4py_output, ds4py_provenance)
            pairs.append((c_rounds[0], ds4py_rounds[0]))
            provenance_pairs.append((c_provenance, ds4py_provenance))
        except (OSError, ValueError, DriverEvidenceError) as error:
            issues.append(f"{label}: {error}")
    return compare_cap_parity(
        pairs,
        provenance=provenance_pairs,
        response_budget=config.tokens,
        diagnostic_issues=issues,
    )


def run_reference_equivalence(  # noqa: PLR0913 - Task evidence, driver setup, and the optional phase are independent inputs.
    task_results: Sequence[TaskRunResult],
    config: AgentConfig,
    c_driver: str | Path,
    session: BakeoffSession,
    results_root: str | Path,
    *,
    cap_parity: bool = False,
    operations: ReferenceOperations,
) -> ReferencePhase:
    """Run reference drivers for every C attempt and optional cap prompts."""
    try:
        selected = operations.select_c(task_results)
    except ReplaySelectionError as error:
        cap_result = (
            GateResult(passed=False, detail=f"cap trace selection failed: {error}", valid=False) if cap_parity else None
        )
        return ReferencePhase(_reference_failure("<selection>", 1, str(error)), {}, (), cap_result)
    comparisons: list[DriverEquivalenceResult] = []
    provenance_pairs: list[tuple[TraceProvenance, TraceProvenance]] = []
    c_evidence: list[DriverRoundEvidence] = []
    ds4py_evidence: list[DriverRoundEvidence] = []
    attempts: dict[str, int] = {}
    for task_id, trace_path in selected:
        attempts[task_id] = attempts.get(task_id, 0) + 1
        attempt = attempts[task_id]
        destination = Path(results_root) / f"{len(comparisons):03d}-{task_id}-attempt-{attempt}"
        try:
            trace = operations.read_trace(trace_path)
            _validate_reference_trace(trace, config)
            destination.mkdir(parents=True, exist_ok=True)
            manifest = operations.export_cases(
                trace,
                task_id=task_id,
                attempt=attempt,
                destination=destination,
            )
            c_output = destination / "c.ndjson"
            ds4py_output = destination / "ds4py.ndjson"
            c_command, ds4py_command = _reference_commands(config, c_driver, manifest, trace_path)
            c_rounds, c_issues = operations.run_driver(c_command, c_output, destination / "c.log")
            ds4py_rounds, ds4py_issues = operations.run_driver(
                ds4py_command,
                ds4py_output,
                destination / "ds4py.log",
            )
            c_provenance = session.reference_provenance(c_output, trace_path, c_rounds)
            ds4py_provenance = session.reference_provenance(ds4py_output, trace_path, ds4py_rounds)
            write_trace_provenance(c_output, c_provenance)
            write_trace_provenance(ds4py_output, ds4py_provenance)
            comparisons.append(
                compare_driver_rounds(
                    c_rounds,
                    ds4py_rounds,
                    c_provenance=c_provenance,
                    ds4py_provenance=ds4py_provenance,
                    diagnostic_issues=(
                        *(f"C driver: {issue}" for issue in c_issues),
                        *(f"ds4-py driver: {issue}" for issue in ds4py_issues),
                    ),
                )
            )
            c_evidence.extend(c_rounds)
            ds4py_evidence.extend(ds4py_rounds)
            provenance_pairs.append((c_provenance, ds4py_provenance))
        except (OSError, ValueError, DriverEvidenceError) as error:
            comparisons.append(_reference_failure(task_id, attempt, str(error)))
    c_rates = rates_by_task(c_evidence)
    ds4py_rates = rates_by_task(ds4py_evidence)
    rates = {
        task_id: (c_rates[task_id], ds4py_rates[task_id]) for task_id in sorted(c_rates.keys() & ds4py_rates.keys())
    }
    c_cached = cached_fractions_by_task(c_evidence)
    ds4py_cached = cached_fractions_by_task(ds4py_evidence)
    cached_fractions = {
        task_id: (c_cached[task_id], ds4py_cached[task_id]) for task_id in sorted(c_cached.keys() & ds4py_cached.keys())
    }
    cap_result = (
        _run_cap_parity(task_results, config, c_driver, session, results_root, operations=operations)
        if cap_parity
        else None
    )
    return ReferencePhase(
        combine_driver_results(comparisons),
        rates,
        tuple(provenance_pairs),
        cap_result,
        cached_fractions,
    )


def run_self_replay_phase(  # noqa: PLR0913 - Provenance, session, and patchable replay operations stay explicit.
    task_results: Sequence[TaskRunResult],
    config: AgentConfig,
    *,
    provenance: Mapping[Path, TraceProvenance],
    session: BakeoffSession,
    replay: Callable[..., tuple[SelfReplayResult, ...]] = replay_ds4py_task_results,
    context: type[SelfReplayContext] = SelfReplayContext,
) -> tuple[SelfReplayResult, ...]:
    """Replay ds4-py against itself with the recorded task provenance."""
    return replay(
        task_results,
        config,
        provenance=provenance,
        context=context(
            bakeoff_session_id=session.bakeoff_session_id,
            instance_id=session.instance_id,
            model_sha256=session.model_sha256,
            libds4_sha256=session.libds4_sha256,
        ),
    )
