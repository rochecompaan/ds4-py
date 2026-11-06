"""Save validation, deterministic restore checks, and fresh-process drivers."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import heapq
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, TypeIs

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from ds4py.engine.lifecycle import SessionEvent
    from ds4py.engine.logits import LogitsView
    from ds4py.runtime import AgentRuntime

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.comparison import TOP_K, StepCapture, TopLogit, compare_step
from bench.models import AgentConfig, TaskRunResult
from bench.provenance import Q8_F16_CACHE_ENV, Q8_F16_CACHE_MB, TraceProvenance, validate_session_pair
from bench.restore_c_reference import prepare_c_rebuild, write_c_rebuild_metadata
from bench.restore_history import same_history_command
from bench.session import report_provenance_environment
from ds4py._json import is_json_object
from ds4py.engine.events import Stopped, Synced
from ds4py.engine.replay import read_replay
from ds4py.errors import Ds4pyError
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
    followup_position: int
    first_step: StepCapture | None
    provenance_path: Path | None
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
    position: CheckResult = CheckResult(False, "not run")
    restore_fidelity: CheckResult = CheckResult(False, "not run")
    rebuild_equivalence: CheckResult = CheckResult(False, "not run")
    stripped_save_dir: Path | None = None
    trace_paths: tuple[Path, ...] = ()
    valid: bool = True


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

    @property
    def request_limit(self) -> int: ...

    @property
    def tokens(self) -> int: ...


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


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        raise RestoreError(message)


def validate_save(save_dir: str | Path) -> SaveEvidence:
    """Validate one record, exact KV payload, events, and named complete snapshot."""
    save_dir = Path(save_dir)
    try:
        store = SessionStore()
        record = store.read_record(save_dir / "save.json")
        payload = store.read_kv_header(save_dir / "session.kv")
        _require(
            condition=payload == record.payload_bytes,
            message=f"{save_dir}: KV payload {payload} does not match record {record.payload_bytes}",
        )
        events_path = _events(save_dir / "events.jsonl")
        steps_root = save_dir / "steps"
        _require(condition=steps_root.is_dir(), message=f"missing step store: {steps_root}")
        from pydantic_ai_harness.step_persistence import FileStepStore

        snapshots = FileStepStore(steps_root)
        snapshot = asyncio.run(
            snapshots.latest_snapshot(
                run_id=record.run_id,
                include_interrupted=True,
            )
        )
        _require(
            condition=snapshot is not None
            and snapshot.state == "complete"
            and snapshot.step_index == record.step_index,
            message=f"{save_dir}: named complete snapshot is missing",
        )
        snapshot_path = None
        for path in sorted((steps_root / record.run_id / "snapshots").glob("*.json")):
            value = json.loads(path.read_text())
            if value.get("step_index") == record.step_index and value.get("state") == "complete":
                snapshot_path = path
                break
        if snapshot_path is None:
            raise RestoreError(f"{save_dir}: snapshot JSON for step {record.step_index} is missing")  # noqa: TRY301
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


@dataclass(frozen=True)
class _RestoreMetadata:
    provenance: TraceProvenance
    save_record_sha256: str
    followup_prompt_sha256: str


def _restore_metadata(run: RestoreRun, label: str) -> _RestoreMetadata:
    if run.provenance_path is None:
        raise RestoreError(f"{label}: restore provenance is missing")
    try:
        value = json.loads(run.provenance_path.read_text())
        provenance_value = value["provenance"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RestoreError(f"{label}: malformed restore provenance: {error}") from error
    if not isinstance(provenance_value, dict):
        raise RestoreError(f"{label}: malformed restore provenance")
    try:
        provenance_value["rounds"] = ()
        provenance_value["native_patches"] = tuple(provenance_value.get("native_patches", ()))
        return _RestoreMetadata(
            TraceProvenance(**provenance_value),
            value["save_record_sha256"],
            value["followup_prompt_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RestoreError(f"{label}: malformed restore provenance: {error}") from error


def validate_restore_pair(case1: RestoreRun, case2: RestoreRun) -> tuple[str, ...]:
    """Validate shared session, original save, and fixed follow-up evidence."""
    first = _restore_metadata(case1, "case 1")
    second = _restore_metadata(case2, "case 2")
    issues = list(validate_session_pair(first.provenance, second.provenance))
    if not first.save_record_sha256 or first.save_record_sha256 != second.save_record_sha256:
        issues.append("save-record hash differs or is missing")
    expected_followup = hashlib.sha256(FOLLOWUP.encode()).hexdigest()
    for label, metadata in (("case 1", first), ("case 2", second)):
        if metadata.followup_prompt_sha256 != expected_followup:
            issues.append(f"{label} follow-up prompt hash differs")
    return tuple(issues)


def _compare_restore(
    case1: RestoreRun,
    case2: RestoreRun,
) -> tuple[list[str], tuple[int, ...], tuple[int, ...], bool, CheckResult]:
    problems: list[str] = []
    valid = True
    structural = [
        (case1.switch_prefill != 0, f"case 1 prefilled {case1.switch_prefill} tokens during switch"),
        (case2.switch_prefill != 0, f"case 2 prefilled {case2.switch_prefill} tokens during switch"),
        (case2.followup_prefill <= 0, "case 2 did not prefill during follow-up"),
        (case1.followup_plan != "extend", f"case 1 follow-up plan was {case1.followup_plan}, not extend"),
        (case2.followup_plan != "rebuild", f"case 2 follow-up plan was {case2.followup_plan}, not rebuild"),
    ]
    for failed, detail in structural:
        if failed:
            problems.append(detail)
            valid = False
    provenance_issues = validate_restore_pair(case1, case2)
    problems.extend(provenance_issues)
    valid = valid and not provenance_issues

    first_trace = read_replay(case1.trace_path)
    second_trace = read_replay(case2.trace_path)
    first = tuple(token for round_ in first_trace.rounds for token in round_.sampled)
    second = tuple(token for round_ in second_trace.rounds for token in round_.sampled)
    for label, run, trace in (("case 1", case1, first_trace), ("case 2", case2, second_trace)):
        # Restore metadata captures the first fixed-follow-up sync. Later model
        # rounds have new prompts and must not be compared with that position.
        prompt_position = len(trace.rounds[0].prompt)
        if run.followup_position != prompt_position:
            problems.append(f"{label} follow-up position {run.followup_position} != prompt length {prompt_position}")
            valid = False
        if run.first_step is None:
            problems.append(f"{label} first-step evidence is missing")
            valid = False
        elif run.first_step.position != run.followup_position:
            problems.append(f"{label} first-step position differs from Synced.pos")
            valid = False
    positions_match = (
        case1.first_step is not None
        and case2.first_step is not None
        and case1.first_step.position == case2.first_step.position
    )
    if not positions_match:
        problems.append("5.2 position: case 1 and case 2 first-step positions differ or are missing")
        valid = False
    position = CheckResult(
        valid and positions_match,
        (
            f"5.2 position: both cases captured step 0 at {case1.followup_position}"
            if valid and positions_match
            else "5.2 position: invalid or unequal positions"
        ),
    )
    return problems, first, second, valid, position


def _compare_reference(
    run: RestoreRun,
    reference: RestoreRun,
    *,
    label: str,
    expected_plan: str,
) -> tuple[CheckResult, list[str], bool]:
    """Compare one restore path with a reference that holds its KV history constant."""
    problems: list[str] = []
    valid = True
    provenance_issues = validate_restore_pair(run, reference)
    if provenance_issues:
        problems.extend(f"{label}: {issue}" for issue in provenance_issues)
        valid = False
    if reference.followup_plan != expected_plan:
        problems.append(f"{label}: reference plan was {reference.followup_plan}, not {expected_plan}")
        valid = False
    if reference.followup_prefill <= 0:
        problems.append(f"{label}: reference did not prefill the follow-up")
        valid = False
    if run.first_step is None or reference.first_step is None:
        problems.append(f"{label}: first-step evidence is missing")
        valid = False
        return CheckResult(False, "; ".join(problems)), problems, valid
    comparison = compare_step(run.first_step, reference.first_step)
    if not comparison.passed:
        problems.append(f"{label}: {comparison.detail}")
    valid = valid and comparison.valid
    detail = "; ".join(problems) if problems else f"{label}: {comparison.detail}"
    result = CheckResult(valid and comparison.passed, detail)
    return result, problems, valid


def check_restore(
    save_dir: str | Path,
    work_dir: str | Path,
    *,
    runner: RestoreRunner,
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
        same_history_trace = work_dir / "case-3.jsonl"
        same_history = runner(original_home, evidence.record.session_id, 3, same_history_trace)
        trace_paths.append(same_history_trace)
        c_rebuild_trace = work_dir / "case-4.ndjson"
        c_rebuild = runner(stripped.parent.parent, evidence.record.session_id, 4, c_rebuild_trace)
        trace_paths.append(c_rebuild_trace)
        problems, first, second, valid, position = _compare_restore(case1, case2)
        restore_fidelity, fidelity_problems, fidelity_valid = _compare_reference(
            case1,
            same_history,
            label="5.3 restore fidelity",
            expected_plan="same-history",
        )
        rebuild_equivalence, rebuild_problems, rebuild_valid = _compare_reference(
            case2,
            c_rebuild,
            label="5.4 rebuild equivalence",
            expected_plan="c-rebuild",
        )
        problems.extend(fidelity_problems)
        problems.extend(rebuild_problems)
        valid = valid and fidelity_valid and rebuild_valid
        passed = valid and not problems
        detail = (
            "; ".join(problems)
            if problems
            else (
                f"{position.detail}; {restore_fidelity.detail}; {rebuild_equivalence.detail}; "
                "case 1 loaded KV and case 2 rebuilt; "
                f"diagnostic tokens case 1={len(first)} case 2={len(second)}; "
                f"terminal case 1={case1.terminal_reason} case 2={case2.terminal_reason}"
            )
        )
        return RestoreResult(
            save_dir=save_dir,
            passed=passed,
            detail=detail,
            case1=CheckResult(case1.switch_prefill == 0 and case1.followup_plan == "extend", "5.1 case 1"),
            case2=CheckResult(
                case2.switch_prefill == 0 and case2.followup_prefill > 0 and case2.followup_plan == "rebuild",
                "5.1 case 2",
            ),
            position=position,
            restore_fidelity=restore_fidelity,
            rebuild_equivalence=rebuild_equivalence,
            stripped_save_dir=stripped,
            trace_paths=tuple(trace_paths),
            valid=valid,
        )
    except (
        Ds4pyError,
        RestoreError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        return RestoreResult(
            save_dir,
            False,
            str(error),
            stripped_save_dir=stripped,
            trace_paths=tuple(trace_paths),
            valid=False,
        )


async def run_attempt_with_save(
    runtime: AgentRuntime,
    prompt: str,
    *,
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
    runtime: AgentRuntime,
    session_id: str,
) -> object | None:
    """Use the public interactive slash-command path, not a direct runtime switch."""
    from ds4py.tui_commands import SlashCommandRegistry, install_builtin_commands

    registry = SlashCommandRegistry()
    install_builtin_commands(registry, runtime, request_exit=lambda: None)
    return await registry.execute(f"/switch {session_id}")


def _relocate_local_workspace(runtime: AgentRuntime) -> None:
    """Point restored continuation metadata at the configured workspace copy."""
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.workspaces import WorkspaceRef

    response = next((message for message in reversed(runtime.messages) if isinstance(message, ModelResponse)), None)
    if response is None or response.workspace_ref is None or response.workspace_ref.provider != "local":
        return
    # Workspace references are run metadata and are never rendered into the prompt.
    response.workspace_ref = WorkspaceRef(provider="local", id=str(runtime.project_dir.resolve()))


@contextmanager
def _runtime(
    args: argparse.Namespace,
    *,
    read_only: bool = False,
) -> Generator[AgentRuntime]:
    from ds4py.agent_cli import open_runtime

    with open_runtime(args, Path.cwd(), read_only=read_only) as runtime:
        yield runtime


@contextmanager
def _trace(
    runtime: AgentRuntime,
    args: argparse.Namespace,
) -> Generator[None]:
    from ds4py.prefill_trace import PrefillTraceWriter

    trace = (
        PrefillTraceWriter(runtime.model.session, args.trace, think=runtime.model.session.engine.think)
        if args.trace is not None
        else nullcontext()
    )
    with trace:
        yield


def _prompt(args: argparse.Namespace) -> str | None:
    prompt: object = args.prompt_file.read_bytes().decode("utf-8") if args.prompt_file is not None else args.prompt
    if prompt is None:
        return None
    if not isinstance(prompt, str):
        raise RestoreError("prompt must be text")
    return prompt


def attempt_command(argv: Sequence[str]) -> int:
    from ds4py.agent_cli import parser

    args = parser().parse_args(argv)
    prompt = _prompt(args)
    if prompt is None:
        raise RestoreError("attempt driver requires --prompt or --prompt-file")
    metadata = Path(
        os.environ.get(
            "BAKEOFF_ATTEMPT_METADATA",
            str(args.trace or "attempt") + ".attempt.json",
        )
    )
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

                    status = asyncio.run(
                        run_attempt_with_save(
                            runtime,
                            prompt,
                            operation=interactive,
                        )
                    )
    finally:
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            json.dumps(
                {
                    "status": status,
                    "save_dir": str(save_dir) if save_dir else None,
                }
            )
            + "\n"
        )
    return status


class _RestoreState(TypedDict):
    followup: bool
    switch_prefill: int
    followup_prefill: int
    followup_position: int
    first_step: StepCapture | None
    plan: str | None
    reason: str | None


async def _restore_operation(
    runtime: AgentRuntime,
    session_id: str,
) -> _RestoreState:
    state: _RestoreState = {
        "followup": False,
        "switch_prefill": 0,
        "followup_prefill": 0,
        "followup_position": -1,
        "first_step": None,
        "plan": None,
        "reason": None,
    }

    def listener(event: SessionEvent) -> None:
        inner = event.payload["event"]
        if event.kind == "Synced" and isinstance(inner, Synced):
            plan = inner.plan
            if state["followup"]:
                if state["plan"] is None:
                    state["plan"] = plan.kind
                    state["followup_prefill"] = plan.to_prefill
                    state["followup_position"] = inner.pos
            else:
                state["switch_prefill"] += plan.to_prefill
        elif event.kind == "Stopped" and state["followup"] and isinstance(inner, Stopped):
            state["reason"] = inner.reason

    def capture_first_step(view: LogitsView) -> None:
        if state["first_step"] is not None:
            return
        entries = heapq.nlargest(TOP_K, ((float(value), index) for index, value in enumerate(view.logits())))
        top = tuple(TopLogit(token, logit) for logit, token in entries)
        state["first_step"] = StepCapture(view.position, top, top[0].token)

    runtime.model.session.on_event(listener)
    await switch_via_command(runtime, session_id)
    _relocate_local_workspace(runtime)
    state["followup"] = True
    runtime.model.session.add_logits_hook(capture_first_step)
    from ds4py.model import ResponseTerminalError

    try:
        await runtime.run(FOLLOWUP)
    except ResponseTerminalError as error:
        if error.reason != "max_tokens":
            raise
        # Gate 5 needs the first pre-sample call, not a complete continuation.
        state["reason"] = error.reason
    return state


def restore_command(argv: Sequence[str]) -> int:
    from ds4py.agent_cli import parser

    custom = argparse.ArgumentParser()
    custom.add_argument("--session-id", required=True)
    custom.add_argument("--metadata", type=Path, required=True)
    own, remaining = custom.parse_known_args(argv)
    args = parser().parse_args(remaining)
    # Attempts run non-interactively. Restore must render the same prompt-visible
    # capabilities so a saved KV prefix remains valid.
    args.non_interactive = True
    with _runtime(args) as runtime, _trace(runtime, args):
        state = asyncio.run(_restore_operation(runtime, own.session_id))
    if args.trace is None:
        raise RestoreError("restore evidence requires --trace")
    trace_path = Path(args.trace)
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    save_path = Path(os.environ.get("DS4PY_HOME", "")) / "sessions" / own.session_id / "save.json"
    save_record_sha256 = hashlib.sha256(save_path.read_bytes()).hexdigest() if save_path.is_file() else ""
    provenance = TraceProvenance(
        bakeoff_session_id=os.environ.get("BAKEOFF_SESSION_ID", "standalone"),
        instance_id=os.environ.get("DS4_BAKEOFF_INSTANCE_ID", "unknown"),
        gpu_model=os.environ.get("DS4_BAKEOFF_GPU_MODEL", "unknown"),
        gpu_driver_version=os.environ.get("DS4_BAKEOFF_GPU_DRIVER_VERSION", "unknown"),
        ds4_revision=os.environ.get("DS4_BAKEOFF_DS4_REVISION", "unknown"),
        ds4py_revision=os.environ.get("DS4_BAKEOFF_DS4PY_REVISION", "unknown"),
        libds4_sha256=os.environ.get("DS4_BAKEOFF_LIBDS4_SHA256", "unknown"),
        model_path=str(Path(args.model).expanduser().resolve()),
        model_sha256=os.environ.get("DS4_BAKEOFF_MODEL_SHA256", "unknown"),
        backend=args.backend,
        ctx=args.ctx,
        q8_f16_cache_mb=int(os.environ.get(Q8_F16_CACHE_ENV, str(Q8_F16_CACHE_MB))),
        temperature=args.temp,
        seed=args.seed,
        response_budget=args.tokens,
        reset_policy="restore-switch",
        trace_sha256=trace_sha256,
        source_trace_sha256=trace_sha256,
        rounds=(),
    )
    own.metadata.write_text(
        json.dumps(
            {
                "switch_prefill": state["switch_prefill"],
                "followup_plan": state["plan"],
                "followup_prefill": state["followup_prefill"],
                "followup_position": state["followup_position"],
                "first_step": asdict(state["first_step"]) if state["first_step"] is not None else None,
                "terminal_reason": state["reason"],
                "trace_path": str(trace_path),
                "provenance": asdict(provenance),
                "save_record_sha256": save_record_sha256,
                "followup_prompt_sha256": hashlib.sha256(FOLLOWUP.encode()).hexdigest(),
            }
        )
        + "\n"
    )
    return 0


class _TopLogitEvidence(TypedDict):
    token: int
    logit: float


class _StepEvidence(TypedDict):
    position: int
    top: list[_TopLogitEvidence]
    selected_token: int


def _is_step_evidence(value: object) -> TypeIs[_StepEvidence]:
    if not is_json_object(value):
        return False
    if type(value.get("position")) is not int or type(value.get("selected_token")) is not int:
        return False
    top = value.get("top")
    return isinstance(top, list) and all(
        is_json_object(entry) and type(entry.get("token")) is int and type(entry.get("logit")) in {int, float}
        for entry in top
    )


def _step_from_metadata(value: object) -> StepCapture | None:
    if value is None:
        return None
    if not _is_step_evidence(value):
        raise RestoreError("malformed first-step restore evidence")
    top = tuple(TopLogit(entry["token"], float(entry["logit"])) for entry in value["top"])
    return StepCapture(value["position"], top, value["selected_token"])


def process_restore_runner(
    config: RestoreConfig,
    *,
    workspace: Path | None = None,
    provenance_environment: Mapping[str, str] | None = None,
    c_driver: str | Path | None = None,
) -> RestoreRunner:
    """Return a runner that starts each restore from a disposable task-workspace copy."""

    def run(
        home: Path,
        session_id: str,
        case: int,
        trace_path: Path,
    ) -> RestoreRun:
        home = home.resolve()
        trace_path = trace_path.resolve()
        metadata = trace_path.with_suffix(".json")
        source_workspace = (workspace if workspace is not None else Path.cwd()).resolve()
        executable = config.executable[:-1] if config.executable[-1] == "attempt" else config.executable
        case2_trace = trace_path.with_name("case-2.jsonl")
        if case == 4:
            if c_driver is None:
                raise RestoreError("Gate 5 rebuild equivalence requires the C replay driver")
            command = prepare_c_rebuild(config, c_driver, case2_trace, trace_path)
        else:
            command = [
                *executable,
                "same-history" if case == 3 else "restore",
                "--session-id",
                session_id,
                "--metadata",
                str(metadata),
                "--model",
                config.model,
                "--backend",
                config.backend,
                "--ctx",
                str(config.ctx),
                "--temp",
                str(config.temperature),
                "--seed",
                str(config.seed),
                "--request-limit",
                str(config.request_limit),
                "--tokens",
                str(config.tokens),
                "--trace",
                str(trace_path),
            ]
            if case == 3:
                command.extend(["--restore-trace", str(trace_path.with_name("case-1.jsonl"))])
        environment = os.environ.copy()
        if provenance_environment is not None:
            environment.update(provenance_environment)
        environment["DS4PY_HOME"] = str(home)
        with tempfile.TemporaryDirectory(prefix=f"ds4py-restore-case-{case}-") as temporary:
            restore_workspace = Path(temporary) / "workspace"
            shutil.copytree(source_workspace, restore_workspace)
            completed = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                cwd=restore_workspace,
                check=False,
            )
        trace_path.with_suffix(".log").write_text((completed.stdout or "") + (completed.stderr or ""))
        if completed.returncode:
            raise RestoreError(f"restore case {case} exited {completed.returncode}")
        if case == 4:
            trace_path.write_text(completed.stdout or "")
            write_c_rebuild_metadata(case2_trace, trace_path, metadata)
        value = json.loads(metadata.read_text())
        return RestoreRun(
            trace_path=trace_path,
            terminal_reason=value["terminal_reason"],
            switch_prefill=value["switch_prefill"],
            followup_plan=value["followup_plan"],
            followup_position=value["followup_position"],
            first_step=_step_from_metadata(value["first_step"]),
            provenance_path=metadata,
            followup_prefill=value["followup_prefill"],
        )

    return run


def check_result_saves(
    task_results: Sequence[TaskRunResult],
    config: RestoreConfig,
    results_root: str | Path,
    *,
    provenance_environment: Mapping[str, str] | None = None,
    c_driver: str | Path | None = None,
) -> tuple[RestoreResult, ...]:
    """Restore every ds4-py attempt save with same-history and C references."""
    results: list[RestoreResult] = []
    for item in task_results:
        if item.agent != "ds4py":
            continue
        for index, save_dir in enumerate(item.attempt_save_dirs, 1):
            results.append(
                check_restore(
                    save_dir,
                    Path(results_root) / item.task_id / f"attempt-{index}" / "restore",
                    runner=process_restore_runner(
                        config,
                        workspace=getattr(item, "workspace", None),
                        provenance_environment=provenance_environment,
                        c_driver=c_driver,
                    ),
                )
            )
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
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--request-limit", type=int, default=100)
    parser.add_argument("--c-driver", default=os.environ.get("DS4_C_DRIVER"))
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    data = json.loads(args.report.read_text())
    tasks = [TaskRunResult.from_dict(item) for item in data.get("tasks", [])]
    config = AgentConfig(
        agent="ds4py",
        executable=(sys.executable, str(Path(__file__).resolve()), "attempt"),
        model=args.model,
        backend=args.backend,
        ctx=args.ctx,
        temperature=args.temp,
        seed=args.seed,
        tokens=args.tokens,
        request_limit=args.request_limit,
    )
    environment = report_provenance_environment(data)
    options = {"c_driver": args.c_driver} if args.c_driver is not None else {}
    if environment:
        options["provenance_environment"] = environment
    results = check_result_saves(
        tasks,
        config,
        args.results or args.report.parent / "restores",
        **options,
    )
    if not results:
        raise RestoreError("report contains no ds4-py attempt saves")
    passed = all(result.passed for result in results)
    output = args.output or args.report.parent / "restore-check.json"
    output.write_text(
        json.dumps(
            {
                "passed": passed,
                "results": [asdict(result) for result in results],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    for result in results:
        print(f"{'PASS' if result.passed else 'FAIL'} {result.save_dir}: {result.detail}")
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    os.environ[Q8_F16_CACHE_ENV] = str(Q8_F16_CACHE_MB)
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"attempt", "restore", "same-history", "check"}:
        raise SystemExit("usage: restore_check.py attempt|restore|same-history|check [options]")
    command, arguments = argv[0], argv[1:]
    try:
        if command == "attempt":
            return attempt_command(arguments)
        if command == "restore":
            return restore_command(arguments)
        if command == "same-history":
            return same_history_command(arguments)
        return check_command(arguments)
    except (RestoreError, OSError, ValueError) as error:
        print(f"Bakeoff driver: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
