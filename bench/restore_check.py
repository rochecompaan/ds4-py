"""Save validation, deterministic restore checks, and fresh-process drivers."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from collections.abc import Awaitable, Callable, Iterator, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from bench.run_bakeoff import TaskRunResult

from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.replay import read_replay
from ds4py.errors import Ds4pyError
from ds4py.runtime import AgentRuntime
from ds4py.store import SaveRecord, SessionStore


FOLLOWUP = "Continue the previous task: state the next concrete action, then stop."


class RestoreError(RuntimeError):
    """Save or restore evidence is absent, malformed, or inconsistent."""


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class RestoreRun:
    trace_path: Path
    terminal_reason: str
    switch_prefill: int
    followup_plan: str
    followup_prefill: int = 0


@dataclass(frozen=True)
class SaveEvidence:
    record: SaveRecord
    events_path: Path
    snapshot_path: Path


@dataclass(frozen=True)
class RestoreResult:
    save_dir: Path
    passed: bool
    detail: str = ""
    case1: CheckResult = CheckResult(False, "not run")
    case2: CheckResult = CheckResult(False, "not run")
    stripped_save_dir: Path | None = None
    trace_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class RestoreSummary:
    runs: tuple[RestoreResult, ...] = ()
    interruption: CheckResult = CheckResult(False, "not run")


RestoreRunner = Callable[[Path, str, int, Path], RestoreRun]


class RestoreConfig(Protocol):
    @property
    def executable(self) -> tuple[str, ...]: ...

    @property
    def model(self) -> str: ...

    @property
    def backend(self) -> str: ...

    @property
    def ctx(self) -> int: ...

    @property
    def temperature(self) -> float: ...

    @property
    def seed(self) -> int: ...


def _events(path: Path) -> Path:
    if not path.is_file():
        raise RestoreError(f"missing event stream: {path}")
    kinds = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RestoreError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
            raise RestoreError(f"{path}:{line_number}: malformed event")
        kinds.append(value["kind"])
    if "run_start" not in kinds or "run_end" not in kinds:
        raise RestoreError(f"{path}: events do not contain a complete run")
    return path


def validate_save(save_dir: str | Path) -> SaveEvidence:
    """Validate one record, exact KV payload, events, and named complete snapshot."""
    save_dir = Path(save_dir)
    try:
        store = SessionStore()
        record = store.read_record(save_dir / "save.json")
        payload = store.read_kv_header(save_dir / "session.kv")
        if payload != record.payload_bytes:
            raise RestoreError(
                f"{save_dir}: KV payload {payload} does not match record {record.payload_bytes}"
            )
        events_path = _events(save_dir / "events.jsonl")
        steps_root = save_dir / "steps"
        if not steps_root.is_dir():
            raise RestoreError(f"missing step store: {steps_root}")
        from pydantic_ai_harness.step_persistence import FileStepStore

        snapshots = FileStepStore(steps_root)
        snapshot = asyncio.run(snapshots.latest_snapshot(
            run_id=record.run_id, include_interrupted=True,
        ))
        if snapshot is None or snapshot.state != "complete" or snapshot.step_index != record.step_index:
            raise RestoreError(f"{save_dir}: named complete snapshot is missing")
        snapshot_path = None
        for path in sorted((steps_root / record.run_id / "snapshots").glob("*.json")):
            value = json.loads(path.read_text())
            if value.get("step_index") == record.step_index and value.get("state") == "complete":
                snapshot_path = path
                break
        if snapshot_path is None:
            raise RestoreError(f"{save_dir}: snapshot JSON for step {record.step_index} is missing")
        return SaveEvidence(record, events_path, snapshot_path)
    except RestoreError:
        raise
    except (Ds4pyError, OSError, ValueError, TypeError) as error:
        raise RestoreError(f"{save_dir}: invalid save evidence: {error}") from error


def make_stripped_copy(save_dir: str | Path, home: str | Path) -> Path:
    """Copy a save into a fresh home and replace only its KV payload with zero bytes."""
    save_dir = Path(save_dir)
    record = SessionStore().read_record(save_dir / "save.json")
    destination = Path(home) / "sessions" / record.session_id
    if destination.exists():
        raise RestoreError(f"stripped destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(save_dir, destination)
    SessionStore().write_kv_atomic(destination / "session.kv", b"")
    return destination


def _samples(trace_path: str | Path) -> tuple[int, ...]:
    trace = read_replay(trace_path)
    return tuple(token for round_ in trace.rounds for token in round_.sampled)


def _compare_restore(
    case1: RestoreRun, case2: RestoreRun,
) -> tuple[list[str], tuple[int, ...], tuple[int, ...]]:
    problems: list[str] = []
    if case1.switch_prefill != 0:
        problems.append(f"case 1 prefilled {case1.switch_prefill} tokens during switch")
    if case2.switch_prefill != 0:
        problems.append(f"case 2 prefilled {case2.switch_prefill} tokens during switch")
    if case2.followup_prefill <= 0:
        problems.append("case 2 did not prefill during follow-up")
    if case1.followup_plan != "extend":
        problems.append(f"case 1 follow-up plan was {case1.followup_plan}, not extend")
    if case2.followup_plan != "rebuild":
        problems.append(f"case 2 follow-up plan was {case2.followup_plan}, not rebuild")
    first = _samples(case1.trace_path)
    second = _samples(case2.trace_path)
    for index in range(min(len(first), len(second), 256)):
        if first[index] != second[index]:
            problems.append(
                f"token {index} differs: case 1={first[index]} case 2={second[index]}"
            )
            break
    else:
        if len(first) != len(second) and min(len(first), len(second)) < 256:
            problems.append(f"token {min(len(first), len(second))} is missing from the shorter restore")
        elif len(first) < 256 and len(second) < 256 and case1.terminal_reason != case2.terminal_reason:
            problems.append(
                f"terminal reasons differ: {case1.terminal_reason} vs {case2.terminal_reason}"
            )
    return problems, first, second


def check_restore(
    save_dir: str | Path, work_dir: str | Path, *, runner: RestoreRunner,
) -> RestoreResult:
    """Run original and stripped saves through two new driver processes/calls."""
    save_dir = Path(save_dir)
    work_dir = Path(work_dir)
    case1 = case2 = None
    stripped = None
    trace_paths: list[Path] = []
    try:
        evidence = validate_save(save_dir)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        original_home = save_dir.parent.parent
        stripped = make_stripped_copy(save_dir, work_dir / "stripped-home")
        case1_trace = work_dir / "case-1.jsonl"
        case1 = runner(original_home, evidence.record.session_id, 1, case1_trace)
        trace_paths.append(case1_trace)
        case2_trace = work_dir / "case-2.jsonl"
        case2 = runner(stripped.parent.parent, evidence.record.session_id, 2, case2_trace)
        trace_paths.append(case2_trace)
        problems, first, second = _compare_restore(case1, case2)
        passed = not problems
        detail = "; ".join(problems) if problems else (
            f"256-token-capped deterministic match ({min(256, len(first), len(second))} tokens); "
            "case 1 loaded KV and case 2 rebuilt"
        )
        return RestoreResult(
            save_dir, passed, detail,
            CheckResult(case1.switch_prefill == 0 and case1.followup_plan == "extend", "case 1"),
            CheckResult(
                case2.switch_prefill == 0 and case2.followup_prefill > 0
                and case2.followup_plan == "rebuild", "case 2",
            ),
            stripped, tuple(trace_paths),
        )
    except (Ds4pyError, RestoreError, OSError, ValueError, subprocess.SubprocessError) as error:
        return RestoreResult(save_dir, False, str(error), stripped_save_dir=stripped,
                             trace_paths=tuple(trace_paths))


async def run_attempt_with_save(
    runtime: AgentRuntime, prompt: str, *,
    operation: Callable[[], Awaitable[int]] | None = None,
) -> int:
    """Save exactly once; the standard noninteractive operation owns its finally."""
    if operation is None:
        from ds4py.noninteractive import run_noninteractive

        return await run_noninteractive(runtime, prompt)
    try:
        return await operation()
    finally:
        await runtime.save()


async def switch_via_command(
    runtime: AgentRuntime, session_id: str,
) -> object | None:
    """Use the public interactive slash-command path, not a direct runtime switch."""
    from ds4py.tui_commands import SlashCommandRegistry, install_builtin_commands

    registry = SlashCommandRegistry()
    install_builtin_commands(registry, runtime, request_exit=lambda: None)
    return await registry.execute(f"/switch {session_id}")


@contextmanager
def _runtime(
    args: argparse.Namespace, *, read_only: bool = False,
) -> Iterator[AgentRuntime]:
    from ds4py.agent_cli import open_runtime

    with open_runtime(args, Path.cwd(), read_only=read_only) as runtime:
        yield runtime


@contextmanager
def _trace(
    runtime: AgentRuntime, args: argparse.Namespace,
) -> Iterator[None]:
    from ds4py.prefill_trace import PrefillTraceWriter

    trace = (
        PrefillTraceWriter(runtime.model.session, args.trace, think=runtime.model.session.engine.think)
        if args.trace is not None else nullcontext()
    )
    with trace:
        yield


def _prompt(args: argparse.Namespace) -> str | None:
    if args.prompt_file is not None:
        return args.prompt_file.read_bytes().decode("utf-8")
    return args.prompt


def attempt_command(argv: Sequence[str]) -> int:
    from ds4py.agent_cli import parser

    args = parser().parse_args(argv)
    prompt = _prompt(args)
    if prompt is None:
        raise RestoreError("attempt driver requires --prompt or --prompt-file")
    metadata = Path(os.environ.get(
        "BAKEOFF_ATTEMPT_METADATA", str(args.trace or "attempt") + ".attempt.json",
    ))
    save_dir = None
    status = 1
    try:
        with _runtime(args) as runtime:
            save_dir = runtime.store.session_dir(runtime.session_id)
            with _trace(runtime, args):
                if args.non_interactive:
                    status = asyncio.run(run_attempt_with_save(runtime, prompt))
                else:
                    async def interactive() -> int:
                        from ds4py.agent_cli import _run_tui

                        return await _run_tui(runtime, prompt)

                    status = asyncio.run(run_attempt_with_save(
                        runtime, prompt, operation=interactive,
                    ))
    finally:
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps({
            "status": status, "save_dir": str(save_dir) if save_dir else None,
        }) + "\n")
    return status


async def _restore_operation(
    runtime: AgentRuntime, session_id: str,
) -> dict[str, Any]:
    state = {
        "followup": False, "switch_prefill": 0, "followup_prefill": 0,
        "plan": None, "reason": None,
    }

    def listener(event: SessionEvent) -> None:
        if event.kind == "Synced":
            plan = event.payload["event"].plan
            if state["followup"]:
                if state["plan"] is None:
                    state["plan"] = plan.kind
                    state["followup_prefill"] = plan.to_prefill
            else:
                state["switch_prefill"] += plan.to_prefill
        elif event.kind == "Stopped" and state["followup"]:
            state["reason"] = event.payload["event"].reason

    runtime.model.session.on_event(listener)
    await switch_via_command(runtime, session_id)
    state["followup"] = True
    await runtime.run(FOLLOWUP)
    return state


def restore_command(argv: Sequence[str]) -> int:
    from ds4py.agent_cli import parser

    custom = argparse.ArgumentParser()
    custom.add_argument("--session-id", required=True)
    custom.add_argument("--metadata", type=Path, required=True)
    own, remaining = custom.parse_known_args(argv)
    args = parser().parse_args(remaining)
    # Attempts run non-interactively. Restore must render the same prompt-visible
    # capabilities; read_only below independently enforces restore safety.
    args.non_interactive = True
    with _runtime(args, read_only=True) as runtime:
        with _trace(runtime, args):
            state = asyncio.run(_restore_operation(runtime, own.session_id))
    own.metadata.write_text(json.dumps({
        "switch_prefill": state["switch_prefill"], "followup_plan": state["plan"],
        "followup_prefill": state["followup_prefill"],
        "terminal_reason": state["reason"], "trace_path": str(args.trace),
    }) + "\n")
    return 0


def process_restore_runner(
    config: RestoreConfig, *, workspace: Path | None = None,
) -> RestoreRunner:
    """Return a runner that starts each restore from its task workspace."""
    def run(
        home: Path, session_id: str, case: int, trace_path: Path,
    ) -> RestoreRun:
        metadata = trace_path.with_suffix(".json")
        executable = config.executable[:-1] if config.executable[-1] == "attempt" else config.executable
        command = [
            *executable, "restore", "--session-id", session_id,
            "--metadata", str(metadata), "--model", config.model,
            "--backend", config.backend, "--ctx", str(config.ctx),
            "--temp", str(config.temperature), "--seed", str(config.seed),
            "--tokens", "256", "--trace", str(trace_path),
        ]
        environment = os.environ.copy()
        environment["DS4PY_HOME"] = str(home)
        completed = subprocess.run(
            command, env=environment, capture_output=True, text=True, cwd=workspace,
        )
        trace_path.with_suffix(".log").write_text(
            (completed.stdout or "") + (completed.stderr or "")
        )
        if completed.returncode:
            raise RestoreError(f"restore case {case} exited {completed.returncode}")
        value = json.loads(metadata.read_text())
        return RestoreRun(
            trace_path=trace_path, terminal_reason=value["terminal_reason"],
            switch_prefill=value["switch_prefill"], followup_plan=value["followup_plan"],
            followup_prefill=value["followup_prefill"],
        )
    return run


def check_result_saves(
    task_results: Sequence[TaskRunResult], config: RestoreConfig, results_root: str | Path,
) -> tuple[RestoreResult, ...]:
    """Restore every ds4-py attempt save in a fresh process pair."""
    results: list[RestoreResult] = []
    for item in task_results:
        if item.agent != "ds4py":
            continue
        for index, save_dir in enumerate(item.attempt_save_dirs, 1):
            results.append(check_restore(
                save_dir,
                Path(results_root) / item.task_id / f"attempt-{index}" / "restore",
                runner=process_restore_runner(
                    config, workspace=getattr(item, "workspace", None),
                ),
            ))
    return tuple(results)


def check_command(argv: Sequence[str]) -> int:
    """Recheck every ds4-py save referenced by a written bakeoff report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--ctx", type=int, default=100000)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    data = json.loads(args.report.read_text())
    tasks = cast(
        list[TaskRunResult],
        [SimpleNamespace(**item) for item in data.get("tasks", [])],
    )
    config = cast(RestoreConfig, SimpleNamespace(
        executable=(sys.executable, str(Path(__file__).resolve()), "attempt"),
        model=args.model, backend=args.backend, ctx=args.ctx,
        temperature=args.temp, seed=args.seed,
    ))
    results = check_result_saves(
        tasks, config, args.results or args.report.parent / "restores",
    )
    if not results:
        raise RestoreError("report contains no ds4-py attempt saves")
    passed = all(result.passed for result in results)
    output = args.output or args.report.parent / "restore-check.json"
    output.write_text(json.dumps({
        "passed": passed, "results": [asdict(result) for result in results],
    }, indent=2, sort_keys=True, default=str) + "\n")
    for result in results:
        print(f"{'PASS' if result.passed else 'FAIL'} {result.save_dir}: {result.detail}")
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"attempt", "restore", "check"}:
        raise SystemExit("usage: restore_check.py attempt|restore|check [options]")
    command, arguments = argv[0], argv[1:]
    try:
        if command == "attempt":
            return attempt_command(arguments)
        if command == "restore":
            return restore_command(arguments)
        return check_command(arguments)
    except (RestoreError, OSError, ValueError) as error:
        print(f"Bakeoff driver: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
