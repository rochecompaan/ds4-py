"""Save evidence, restore execution, and Gate 5.1 plan capture."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import heapq
import json
import os
import shutil
import sys
from collections.abc import Awaitable, Callable, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    from collections.abc import Generator

    from ds4py.engine.lifecycle import SessionEvent
    from ds4py.engine.logits import LogitsView
    from ds4py.runtime import AgentRuntime
    from ds4py.store import SaveRecord

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.comparison import TOP_K, StepCapture, TopLogit
from bench.provenance import Q8_F16_CACHE_ENV, Q8_F16_CACHE_MB, TraceProvenance
from ds4py.engine.events import Stopped, Synced
from ds4py.errors import Ds4pyError
from ds4py.store import SessionStore

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
