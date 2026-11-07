"""Sequential bakeoff agent runner and per-task evidence collection."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.driver_cases import export_round_zero_cap_case, export_trace_cases
from bench.metrics import (
    EventSummary,
    MetricError,
    combine_summaries,
    metrics_from_trace,
    summarize_events,
)
from bench.models import AgentConfig, TaskRunResult
from bench.phases import (
    ReferenceOperations,
    ReferencePhase,
    _run_driver_command,
    run_restore_phase,
    run_self_replay_phase,
)
from bench.phases import (
    _run_cap_parity as _phase_cap_parity,
)
from bench.phases import (
    interruption_acceptance as _phase_interruption_acceptance,
)
from bench.phases import (
    run_reference_equivalence as _phase_reference_equivalence,
)
from bench.pty_driver import drive_interruption_pty, drive_pty  # noqa: F401 - Compatibility re-export.
from bench.report import BakeoffReport, GateResult, build_report
from bench.restore_check import CheckResult, RestoreSummary, check_result_saves
from bench.self_replay import SelfReplayContext, replay_ds4py_task_results, select_ds4py_traces
from bench.session import BakeoffSession, record_task_provenance
from bench.trace_selection import select_c_traces
from bench.workspaces import (
    CommandRunner,
    GradeResult,
    Workspace,
    clone_workspace,
    grade_workspace,
    load_manifest,
    prepare_manifest,
)
from ds4py._json import JsonValue, is_json_object
from ds4py._native.loader import resolve_lib_path
from ds4py.engine.replay import read_replay

FULL_TASK_COUNT = 12

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from ds4py.engine.replay_types import ReplayTrace


def _option_number(value: float) -> str:
    return str(int(value)) if isinstance(value, int) or value.is_integer() else str(value)


def build_agent_command(
    config: AgentConfig,
    prompt: str,
    trace_path: str | Path,
) -> list[str]:
    """Build equivalent C and ds4-py sampling/task invocations."""
    command = [*config.executable, "--non-interactive"]
    if config.agent == "ds4py":
        command.extend(["--request-limit", str(config.request_limit)])
    command.extend(
        [
            "--model",
            config.model,
            "--backend",
            config.backend,
            "--ctx",
            str(config.ctx),
            "--temp",
            _option_number(config.temperature),
            "--seed",
            str(config.seed),
            "--tokens",
            str(config.tokens),
            "--trace",
            str(trace_path),
            "--prompt",
            prompt,
        ]
    )
    return command


def load_events(path: str | Path) -> list[dict[str, JsonValue]]:
    path = Path(path)
    if not path.is_file():
        raise MetricError(f"missing event stream: {path}")
    records: list[dict[str, JsonValue]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise MetricError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not is_json_object(record):
            raise MetricError(f"{path}:{line_number}: event must be an object")
        records.append(record)
    if not records:
        raise MetricError(f"empty event stream: {path}")
    return records


def _empty_summary(wall_seconds: float) -> EventSummary:
    return EventSummary(wall_seconds, 0, 0, 0, None, 0, tool_accounting=None)


def _trace_tokens(path: str | Path) -> tuple[ReplayTrace, int]:
    trace = read_replay(path)
    return trace, sum(len(round_.sampled) for round_ in trace.rounds)


def _c_tool_calls(log_text: str) -> int:
    return sum(bool(re.search(r"\btool[_ -]call(?:ed|s)?\b", line, re.IGNORECASE)) for line in log_text.splitlines())


def _attempt_summary(
    config: AgentConfig,
    events_path: Path,
    trace_path: Path,
    log_text: str,
    wall_seconds: float,
) -> EventSummary:
    if events_path.is_file():
        summary = summarize_events(load_events(events_path), wall_seconds=wall_seconds)
        if config.agent == "c":
            summary = replace(summary, tool_accounting=None)
        _trace, tokens = _trace_tokens(trace_path)
        if tokens != summary.tokens:
            raise MetricError(f"trace has {tokens} committed tokens but events have {summary.tokens}")
        return summary
    if config.agent == "ds4py":
        raise MetricError(f"ds4py attempt has no event stream: {events_path}")
    trace, _tokens = _trace_tokens(trace_path)
    return metrics_from_trace(
        trace,
        wall_seconds=wall_seconds,
        progress_events=len(trace.rounds),
        tool_calls=_c_tool_calls(log_text),
        response_budget=config.tokens,
    )


def _expected_max_tokens_exit(
    config: AgentConfig,
    returncode: int,
    log_text: str,
    summary: EventSummary,
) -> bool:
    """The CLI intentionally exits 1 after a complete max_tokens response."""
    return (
        config.agent == "ds4py"
        and returncode == 1
        and summary.terminal_reasons[-1:] == ("max_tokens",)
        and log_text.rstrip().endswith("ds4py: model response ended with max_tokens")
    )


def _record_process_errors(
    errors: list[str],
    attempt: int,
    config: AgentConfig,
    completed: subprocess.CompletedProcess[str],
    summary: EventSummary,
) -> None:
    if not completed.returncode:
        return
    log_text = (completed.stdout or "") + (completed.stderr or "")
    if not _expected_max_tokens_exit(config, completed.returncode, log_text, summary):
        errors.append(f"attempt {attempt} exited {completed.returncode}")
    if "request_limit" in log_text:
        errors.append(f"attempt {attempt} request_limit")


def _save_dir(home: Path) -> Path:
    sessions = home / "sessions"
    saves = sorted(path.parent for path in sessions.glob("*/save.json")) if sessions.is_dir() else []
    if len(saves) != 1:
        raise MetricError(f"expected one saved session under {sessions}, found {len(saves)}")
    return saves[0]


def run_task(
    workspace: Workspace,
    config: AgentConfig,
    results_root: str | Path,
    *,
    grade_runner: CommandRunner | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> TaskRunResult:
    """Run one agent on one staged task, with at most two sequential attempts."""
    results_root = Path(results_root).resolve()
    base_prompt = (workspace.agent_dir / "TASK.md").read_text()
    summaries: list[EventSummary] = []
    grades: list[GradeResult] = []
    save_dirs: list[Path] = []
    trace_paths: list[Path] = []
    errors: list[str] = []
    feedback = ""
    for attempt in range(1, workspace.max_attempts + 1):
        attempt_root = results_root / workspace.task_id / config.agent / f"attempt-{attempt}"
        attempt_root.mkdir(parents=True, exist_ok=True)
        trace_path = attempt_root / ("trace.jsonl" if config.agent == "ds4py" else "trace.txt")
        events_path = attempt_root / "events.jsonl"
        home = attempt_root / "home"
        prompt = base_prompt
        if feedback:
            prompt += (
                "\n\nThe previous attempt failed hidden grading. Fix the remaining issues.\nGrader output:\n" + feedback
            )
        command = build_agent_command(config, prompt, trace_path)
        environment = os.environ.copy()
        environment.update(
            {
                "DS4PY_HOME": str(home),
                "BAKEOFF_ATTEMPT_ROOT": str(attempt_root),
                "BAKEOFF_EVENTS_PATH": str(events_path),
                "BAKEOFF_AGENT_NAME": config.agent,
                "BAKEOFF_ATTEMPT": str(attempt),
                "BAKEOFF_INVOCATIONS": str(results_root.parent / "invocations.jsonl"),
                "BAKEOFF_LOCK_PATH": str(results_root.parent / "agent.lock"),
            }
        )
        started = clock()
        completed = subprocess.run(
            command,
            cwd=workspace.agent_dir,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        wall_seconds = clock() - started
        log_text = (completed.stdout or "") + (completed.stderr or "")
        (attempt_root / "agent.log").write_text(log_text)
        if config.agent == "ds4py":
            try:
                save_dir = _save_dir(home)
                save_dirs.append(save_dir)
                if (save_dir / "events.jsonl").is_file():
                    events_path = save_dir / "events.jsonl"
            except MetricError as error:
                errors.append(str(error))
        trace_paths.append(trace_path)
        try:
            summary = _attempt_summary(
                config,
                events_path,
                trace_path,
                log_text,
                wall_seconds,
            )
        except (MetricError, OSError, ValueError) as error:
            errors.append(f"attempt {attempt} metrics: {error}")
            summary = _empty_summary(wall_seconds)
        _record_process_errors(errors, attempt, config, completed, summary)
        summaries.append(summary)
        grade = grade_workspace(workspace, runner=grade_runner)
        grades.append(grade)
        if grade.passed:
            break
        feedback = grade.output
    summary = combine_summaries(summaries)
    return TaskRunResult(
        task_id=workspace.task_id,
        agent=config.agent,
        passed=grades[-1].passed,
        attempts=len(grades),
        summary=summary,
        grader_outputs=tuple(grade.output for grade in grades),
        attempt_save_dirs=tuple(save_dirs),
        attempt_trace_paths=tuple(trace_paths),
        errors=tuple(errors),
        workspace=workspace.agent_dir.resolve(),
    )


def run_tasks_sequentially(
    workspaces: Sequence[Workspace],
    configs: Sequence[AgentConfig],
    results_root: str | Path,
    *,
    grade_runner: CommandRunner | None = None,
) -> list[TaskRunResult]:
    """Close every agent process before cloning or starting the next run."""
    results: list[TaskRunResult] = []
    for config in configs:
        for staged in workspaces:
            label = f"{config.agent}/{staged.task_id}"
            print(f"[bakeoff] start {label}", flush=True)
            workspace = clone_workspace(
                staged,
                Path(results_root) / "workspaces" / staged.task_id / config.agent,
            )
            completed = run_task(
                workspace,
                config,
                results_root,
                grade_runner=grade_runner,
            )
            results.append(completed)
            outcome = "PASS" if completed.passed else "FAIL"
            print(f"[bakeoff] finish {label}: {outcome}", flush=True)
    return results


def interruption_acceptance(
    config: AgentConfig,
    workspace: Workspace,
    results_root: str | Path,
) -> CheckResult:
    """Interrupt visible generation, submit a next prompt, then exit cleanly."""
    return _phase_interruption_acceptance(
        config,
        workspace,
        results_root,
        build_command=build_agent_command,
        drive=drive_interruption_pty,
    )


def _reference_operations() -> ReferenceOperations:
    return ReferenceOperations(
        select_c=select_c_traces,
        select_ds4py=select_ds4py_traces,
        read_trace=read_replay,
        export_cases=export_trace_cases,
        export_cap_case=export_round_zero_cap_case,
        run_driver=_run_driver_command,
    )


def _run_cap_parity(
    task_results: Sequence[TaskRunResult],
    config: AgentConfig,
    c_driver: str | Path,
    session: BakeoffSession,
    results_root: str | Path,
) -> GateResult:
    """Run both reference drivers on each arm's round-0 prompt."""
    return _phase_cap_parity(
        task_results,
        config,
        c_driver,
        session,
        results_root,
        operations=_reference_operations(),
    )


def run_reference_equivalence(  # noqa: PLR0913 - Task evidence, driver setup, and the optional phase are independent inputs.
    task_results: Sequence[TaskRunResult],
    config: AgentConfig,
    c_driver: str | Path,
    session: BakeoffSession,
    results_root: str | Path,
    *,
    cap_parity: bool = False,
) -> ReferencePhase:
    """Run reference drivers for every C attempt and optional cap prompts."""
    return _phase_reference_equivalence(
        task_results,
        config,
        c_driver,
        session,
        results_root,
        cap_parity=cap_parity,
        operations=_reference_operations(),
    )


@contextmanager
def _session_environment(session: BakeoffSession) -> Generator[None]:
    values = session.environment()
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _validate_ds4py_attempt_library(
    configs: Sequence[AgentConfig],
    staged: Sequence[Workspace],
    results_root: Path,
) -> None:
    """Fail before task arms when their native library cannot be resolved."""
    if not staged or not any(config.agent == "ds4py" for config in configs):
        return
    attempt_home = results_root / staged[0].task_id / "ds4py" / "attempt-1" / "home"
    previous_home = os.environ.get("DS4PY_HOME")
    os.environ["DS4PY_HOME"] = str(attempt_home)
    try:
        candidate = resolve_lib_path(None)
    finally:
        if previous_home is None:
            os.environ.pop("DS4PY_HOME", None)
        else:
            os.environ["DS4PY_HOME"] = previous_home
    try:
        library = candidate.resolve(strict=True)
        if not library.is_file():
            raise FileNotFoundError(candidate)
    except OSError as error:
        raise RuntimeError(
            f"ds4-py attempt library does not resolve: {candidate}. Set DS4_LIB to libds4.so before starting the bakeoff."
        ) from error
    print(f"[bakeoff] ds4-py attempt library {library}", flush=True)


def run_bakeoff(  # noqa: PLR0913 - The manifest and output layout are independent required inputs.
    manifest_path: str | Path,
    benchmark: str | Path,
    repo_root: str | Path,
    configs: Sequence[AgentConfig],
    results_root: str | Path,
    *,
    session: BakeoffSession,
    c_driver: str | Path,
    grade_runner: CommandRunner | None = None,
    cap_parity: bool = False,
) -> BakeoffReport:
    """Run one-session tasks, acceptance checks, and both Gate 6 parts."""
    results_root = Path(results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=False)
    print(f"[bakeoff] session {session.bakeoff_session_id}", flush=True)
    manifest = load_manifest(manifest_path)
    staged = prepare_manifest(manifest, results_root / "staged", benchmark, repo_root)
    _validate_ds4py_attempt_library(configs, staged, results_root)
    with _session_environment(session):
        task_results = run_tasks_sequentially(
            staged,
            configs,
            results_root,
            grade_runner=grade_runner,
        )
        task_provenance = record_task_provenance(task_results, session)
        ds4_config = next(config for config in configs if config.agent == "ds4py")
        restore_runs = run_restore_phase(
            task_results,
            ds4_config,
            results_root,
            c_driver,
            check=check_result_saves,
        )
        interruption_workspace = clone_workspace(
            staged[0],
            results_root / "acceptance" / "interruption",
        )
        interruption = interruption_acceptance(ds4_config, interruption_workspace, results_root)
        restore = RestoreSummary(restore_runs, interruption)
        reference = run_reference_equivalence(
            task_results,
            ds4_config,
            c_driver,
            session,
            results_root / "driver-equivalence",
            cap_parity=cap_parity,
        )
        self_results = run_self_replay_phase(
            task_results,
            ds4_config,
            provenance=task_provenance,
            session=session,
            replay=replay_ds4py_task_results,
            context=SelfReplayContext,
        )
    report = build_report(
        task_results,
        restore,
        rate_evidence=reference.rate_evidence,
        cached_fractions=reference.cached_fractions,
        provenance=reference.provenance,
        cap_parity=reference.cap_parity,
        driver_equivalence=reference.result,
        self_replay_results=self_results,
        certification="full-12-task" if len(staged) == FULL_TASK_COUNT else "targeted",
        bakeoff_session_id=session.bakeoff_session_id,
        instance_id=session.instance_id,
    )
    report.write_json(results_root / "bakeoff.json")
    (results_root / "report.txt").write_text(report.compact_table())
    return report


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if value is None or value == "" else Path(value)


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("bakeoff.yaml"))
    parser.add_argument("--benchmark", type=Path, default=root.parent / "benchmark")
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--agent", required=True, help="C reference agent executable")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--instance-id", default=os.environ.get("DS4_BAKEOFF_INSTANCE_ID"))
    parser.add_argument(
        "--c-driver",
        type=Path,
        default=_environment_path("DS4_C_DRIVER"),
    )
    parser.add_argument("--cap-parity", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.instance_id:
        parser.error("--instance-id or DS4_BAKEOFF_INSTANCE_ID is required")
    if args.c_driver is None:
        parser.error("--c-driver or DS4_C_DRIVER is required")
    if not args.c_driver.is_file():
        parser.error(f"C driver does not exist: {args.c_driver}")
    if args.results.exists():
        parser.error(f"result root already exists: {args.results}")
    manifest = load_manifest(args.manifest)
    configs = [
        AgentConfig(
            "c",
            (args.agent,),
            model=args.model,
            backend=args.backend,
            ctx=manifest.ctx,
            temperature=manifest.temperature,
            seed=manifest.seed,
            tokens=args.tokens,
        ),
        AgentConfig(
            "ds4py",
            (
                sys.executable,
                str(Path(__file__).with_name("restore_check.py")),
                "attempt",
            ),
            model=args.model,
            backend=args.backend,
            ctx=manifest.ctx,
            temperature=manifest.temperature,
            seed=manifest.seed,
            tokens=args.tokens,
        ),
    ]
    session = BakeoffSession.capture(
        configs,
        args.instance_id,
        repo_root=args.repo_root,
    )
    report = run_bakeoff(
        args.manifest,
        args.benchmark,
        args.repo_root,
        configs,
        args.results,
        session=session,
        c_driver=args.c_driver,
        cap_parity=args.cap_parity,
    )
    print(report.compact_table(), end="")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
