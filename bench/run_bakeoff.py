"""Sequential bakeoff agent runner and per-task evidence collection."""

import argparse
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import pty
import re
import select
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.workspaces import (
    CommandRunner, GradeResult, Workspace, clone_workspace, grade_workspace,
    load_manifest, prepare_manifest,
)
from bench.replay_c_traces import replay_task_results
from bench.report import BakeoffReport, build_report
from bench.restore_check import CheckResult, RestoreSummary, check_result_saves
from bench.metrics import (
    EventSummary,
    MetricError,
    combine_summaries,
    metrics_from_trace,
    summarize_events,
)
from ds4py.engine.replay import read_replay
from ds4py.engine.replay_types import ReplayTrace


@dataclass(frozen=True)
class AgentConfig:
    agent: str
    executable: tuple[str, ...]
    model: str
    backend: str
    ctx: int
    temperature: float
    seed: int
    tokens: int

    def __post_init__(self) -> None:
        if self.agent not in {"c", "ds4py"}:
            raise ValueError("agent must be 'c' or 'ds4py'")
        object.__setattr__(self, "executable", tuple(str(item) for item in self.executable))


_SUMMARY_FIELDS = frozenset(EventSummary.__dataclass_fields__)


@dataclass(frozen=True)
class TaskRunResult:
    task_id: str
    agent: str
    passed: bool
    attempts: int
    summary: EventSummary
    grader_outputs: tuple[str, ...] = ()
    attempt_save_dirs: tuple[Path, ...] = ()
    attempt_trace_paths: tuple[Path, ...] = ()
    errors: tuple[str, ...] = ()
    workspace: Path | None = None

    def __post_init__(self) -> None:
        if self.attempts not in {1, 2}:
            raise ValueError("attempts must be one or two")
        for name in ("progress_events", "tokens", "tool_calls", "non_extend_rounds"):
            value = getattr(self.summary, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not math.isfinite(self.summary.wall_seconds) or self.summary.wall_seconds < 0:
            raise ValueError("wall_seconds must be finite and nonnegative")
        reuse = self.summary.avg_prefill_reuse
        if reuse is not None and not 0 <= reuse <= 1:
            raise ValueError("avg_prefill_reuse must be in [0,1]")

    def __getattr__(self, name: str) -> Any:
        if name in _SUMMARY_FIELDS:
            return getattr(self.summary, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    @property
    def save_dir(self) -> Path | None:
        return self.attempt_save_dirs[-1] if self.attempt_save_dirs else None

    @property
    def trace_path(self) -> Path | None:
        return self.attempt_trace_paths[-1] if self.attempt_trace_paths else None


def _option_number(value: int | float) -> str:
    return str(int(value)) if isinstance(value, int) or value.is_integer() else str(value)


def build_agent_command(
    config: AgentConfig, prompt: str, trace_path: str | Path,
) -> list[str]:
    """Build equivalent C and ds4-py sampling/task invocations."""
    command = [*config.executable, "--non-interactive"]
    command.extend([
        "--model", config.model,
        "--backend", config.backend,
        "--ctx", str(config.ctx),
        "--temp", _option_number(config.temperature),
        "--seed", str(config.seed),
        "--tokens", str(config.tokens),
        "--trace", str(trace_path),
        "--prompt", prompt,
    ])
    return command


def drive_pty(
    command: Sequence[str], *, steps: Iterable[tuple[str, str | None]],
    cwd: str | Path | None = None, env: Mapping[str, str] | None = None,
    timeout: float = 30,
) -> CheckResult:
    """Drive timed expect/send steps through a controlling pseudo-terminal."""
    deadline = time.monotonic() + timeout
    output = bytearray()
    pid, master = pty.fork()
    if pid == 0:
        try:
            if cwd is not None:
                os.chdir(cwd)
            os.execvpe(command[0], list(command), os.environ.copy() if env is None else env)
        except BaseException as error:
            print(f"PTY exec failed: {error}", file=sys.stderr)
            os._exit(127)
    wait_status = None

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("PTY timed out")
        return left

    def read_available(wait: float) -> bool:
        ready, _, _ = select.select([master], [], [], min(wait, 0.1))
        if not ready:
            return True
        try:
            data = os.read(master, 8192)
        except OSError:
            return False
        if not data:
            return False
        output.extend(data)
        return True

    try:
        cursor = 0
        for pattern, response in steps:
            regex = re.compile(pattern, re.IGNORECASE)
            while True:
                text = output.decode("utf-8", errors="replace")
                match = regex.search(text, cursor)
                if match:
                    cursor = match.end()
                    break
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    wait_status = status
                    raise RuntimeError(f"process exited while waiting for {pattern!r}")
                read_available(remaining())
            if response is not None:
                os.write(master, response.encode("utf-8"))
        while wait_status is None:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                wait_status = status
                break
            if not read_available(remaining()):
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    wait_status = status
        # Drain anything written between the final expectation and process exit.
        while read_available(0):
            pass
        code = os.waitstatus_to_exitcode(wait_status)
        passed = code == 0
    except (OSError, RuntimeError, TimeoutError) as error:
        output.extend(f"\nPTY driver: {error}".encode())
        passed = False
        if wait_status is None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                _, wait_status = os.waitpid(pid, 0)
            except ChildProcessError:
                pass
    finally:
        os.close(master)
    detail = output.decode("utf-8", errors="replace")[-8000:]
    return CheckResult(passed, detail)


def load_events(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise MetricError(f"missing event stream: {path}")
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise MetricError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(record, dict):
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
    return sum(
        bool(re.search(r"\btool[_ -]call(?:ed|s)?\b", line, re.IGNORECASE))
        for line in log_text.splitlines()
    )


def _attempt_summary(
    config: AgentConfig, events_path: Path, trace_path: Path,
    log_text: str, wall_seconds: float,
) -> EventSummary:
    if events_path.is_file():
        summary = summarize_events(load_events(events_path), wall_seconds=wall_seconds)
        if config.agent == "c":
            summary = replace(summary, tool_accounting=None)
        _trace, tokens = _trace_tokens(trace_path)
        if tokens != summary.tokens:
            raise MetricError(
                f"trace has {tokens} committed tokens but events have {summary.tokens}"
            )
        return summary
    if config.agent == "ds4py":
        raise MetricError(f"ds4py attempt has no event stream: {events_path}")
    trace, _tokens = _trace_tokens(trace_path)
    return metrics_from_trace(
        trace, wall_seconds=wall_seconds,
        progress_events=len(trace.rounds), tool_calls=_c_tool_calls(log_text),
    )


def _save_dir(home: Path) -> Path:
    sessions = home / "sessions"
    saves = sorted(path.parent for path in sessions.glob("*/save.json")) if sessions.is_dir() else []
    if len(saves) != 1:
        raise MetricError(f"expected one saved session under {sessions}, found {len(saves)}")
    return saves[0]


def run_task(
    workspace: Workspace, config: AgentConfig, results_root: str | Path, *,
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
                "\n\nThe previous attempt failed hidden grading. Fix the remaining issues.\n"
                "Grader output:\n" + feedback
            )
        command = build_agent_command(config, prompt, trace_path)
        environment = os.environ.copy()
        environment.update({
            "DS4PY_HOME": str(home),
            "BAKEOFF_ATTEMPT_ROOT": str(attempt_root),
            "BAKEOFF_EVENTS_PATH": str(events_path),
            "BAKEOFF_AGENT_NAME": config.agent,
            "BAKEOFF_ATTEMPT": str(attempt),
            "BAKEOFF_INVOCATIONS": str(results_root.parent / "invocations.jsonl"),
            "BAKEOFF_LOCK_PATH": str(results_root.parent / "agent.lock"),
        })
        started = clock()
        completed = subprocess.run(
            command, cwd=workspace.agent_dir, env=environment,
            capture_output=True, text=True,
        )
        wall_seconds = clock() - started
        log_text = (completed.stdout or "") + (completed.stderr or "")
        (attempt_root / "agent.log").write_text(log_text)
        if completed.returncode:
            errors.append(f"attempt {attempt} exited {completed.returncode}")
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
                config, events_path, trace_path, log_text, wall_seconds,
            )
        except (MetricError, OSError, ValueError) as error:
            errors.append(f"attempt {attempt} metrics: {error}")
            summary = _empty_summary(wall_seconds)
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
        errors=tuple(errors), workspace=workspace.agent_dir.resolve(),
    )


def run_tasks_sequentially(
    workspaces: Sequence[Workspace], configs: Sequence[AgentConfig],
    results_root: str | Path, *, grade_runner: CommandRunner | None = None,
) -> list[TaskRunResult]:
    """Close every agent process before cloning or starting the next run."""
    results: list[TaskRunResult] = []
    for config in configs:
        for staged in workspaces:
            label = f"{config.agent}/{staged.task_id}"
            print(f"[bakeoff] start {label}", flush=True)
            workspace = clone_workspace(
                staged, Path(results_root) / "workspaces" / staged.task_id / config.agent,
            )
            completed = run_task(
                workspace, config, results_root, grade_runner=grade_runner,
            )
            results.append(completed)
            outcome = "PASS" if completed.passed else "FAIL"
            print(f"[bakeoff] finish {label}: {outcome}", flush=True)
    return results


def _acceptance_environment(home: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["DS4PY_HOME"] = str(home)
    return environment


def interruption_acceptance(
    config: AgentConfig, workspace: Workspace, results_root: str | Path,
) -> CheckResult:
    """Interrupt visible generation, submit a next prompt, then exit cleanly."""
    root = Path(results_root) / "gate-4" / "interruption"
    root.mkdir(parents=True, exist_ok=True)
    prompt = (workspace.agent_dir / "TASK.md").read_text()
    command = build_agent_command(config, prompt, root / "trace.jsonl")
    command.remove("--non-interactive")
    result = drive_pty(
        command, cwd=workspace.agent_dir,
        env=_acceptance_environment(root / "home"),
        steps=[
            (r"prefill|tokens/s", "\x03"),
            (r"\bidle\b", "State the next concrete action in one sentence, then stop.\n"),
            (r"\bidle\b", "/exit\n"),
        ],
        timeout=180,
    )
    (root / "interactive.log").write_text(result.detail)
    return result


def run_bakeoff(
    manifest_path: str | Path, benchmark: str | Path, repo_root: str | Path,
    configs: Sequence[AgentConfig], results_root: str | Path, *,
    grade_runner: CommandRunner | None = None,
) -> BakeoffReport:
    """Run all tasks and acceptance checks, then write all six gate results."""
    results_root = Path(results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest(manifest_path)
    staged = prepare_manifest(manifest, results_root / "staged", benchmark, repo_root)
    task_results = run_tasks_sequentially(
        staged, configs, results_root, grade_runner=grade_runner,
    )
    ds4_config = next(config for config in configs if config.agent == "ds4py")
    restore_runs = check_result_saves(task_results, ds4_config, results_root / "restores")
    interruption_workspace = clone_workspace(
        staged[0], results_root / "acceptance" / "interruption",
    )
    interruption = interruption_acceptance(ds4_config, interruption_workspace, results_root)
    restore = RestoreSummary(restore_runs, interruption)
    replay = replay_task_results(task_results, ds4_config)
    report = build_report(task_results, restore, replay)
    report.write_json(results_root / "bakeoff.json")
    (results_root / "report.txt").write_text(report.compact_table())
    return report


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
    parser.add_argument("--tokens", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    common = dict(
        model=args.model, backend=args.backend, ctx=manifest.ctx,
        temperature=manifest.temperature, seed=manifest.seed, tokens=args.tokens,
    )
    configs = [
        AgentConfig("c", (args.agent,), **common),
        AgentConfig(
            "ds4py", (sys.executable, str(Path(__file__).with_name("restore_check.py")), "attempt"),
            **common,
        ),
    ]
    report = run_bakeoff(
        args.manifest, args.benchmark, args.repo_root, configs, args.results,
    )
    print(report.compact_table(), end="")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
