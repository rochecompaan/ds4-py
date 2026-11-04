"""Replay complete ds4-py traces under their recorded session conditions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bench.replay_c_traces import ReplayMismatch, ReplayResult, ReplaySelectionError, _parse_c, replay_round
from ds4py.engine.replay import read_replay
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.worker import ResetJob
from ds4py.errors import Ds4pyError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from bench.models import TaskRunResult
    from bench.provenance import TraceProvenance
    from ds4py.engine.replay_types import ReplayTrace
    from ds4py.engine.session import Engine


@dataclass(frozen=True)
class SelfReplayContext:
    bakeoff_session_id: str
    instance_id: str
    model_sha256: str
    libds4_sha256: str
    reset_policy: str = "sequential"


@dataclass(frozen=True)
class SelfReplayResult:
    task_id: str
    round_index: int
    passed: bool
    detail: str
    tokens: int = 0
    tool_call: bool = False
    terminal_reason: str | None = None
    trace_path: Path | None = None
    valid: bool = True


def select_ds4py_traces(task_results: Iterable[TaskRunResult]) -> list[tuple[str, Path]]:
    """Select every ds4-py attempt in manifest order, without a 12-task rule."""
    tasks = [item for item in task_results if item.agent == "ds4py"]
    identifiers = [item.task_id for item in tasks]
    if not tasks:
        raise ReplaySelectionError("self-replay needs at least one ds4-py task")
    if len(set(identifiers)) != len(identifiers):
        raise ReplaySelectionError("self-replay task identifiers must be distinct")
    selected = []
    for item in tasks:
        paths = item.attempt_trace_paths
        if len(paths) != item.attempts or any(path is None for path in paths):
            raise ReplaySelectionError(
                f"{item.task_id}: {len(paths)}/{item.attempts} ds4-py attempt traces are present"
            )
        selected.extend((item.task_id, Path(path)) for path in paths)
    return selected


def _provenance_issues(
    engine: Engine,
    path: Path,
    trace: ReplayTrace,
    provenance: TraceProvenance,
    context: SelfReplayContext,
) -> tuple[str, ...]:
    expected = {
        "bakeoff_session_id": context.bakeoff_session_id,
        "instance_id": context.instance_id,
        "model_sha256": context.model_sha256,
        "libds4_sha256": context.libds4_sha256,
        "reset_policy": context.reset_policy,
        "backend": engine.config.backend,
        "ctx": engine.config.ctx,
        "temperature": 0.0,
        "seed": 1,
    }
    issues = [
        f"{name} differs: recorded={getattr(provenance, name)!r}, expected={value!r}"
        for name, value in expected.items()
        if getattr(provenance, name) != value
    ]
    model = Path(provenance.model_path).expanduser().resolve()
    if model != Path(engine.config.model).expanduser().resolve():
        issues.append(f"model path differs: recorded={model}, expected={engine.config.model}")
    if trace.model is not None and Path(trace.model).expanduser().resolve() != model:
        issues.append("trace model differs from provenance")
    if trace.ctx is not None and trace.ctx != engine.config.ctx:
        issues.append("trace context differs from engine")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if provenance.trace_sha256 != digest or provenance.source_trace_sha256 != digest:
        issues.append("trace hash differs from the selected ds4-py trace")
    if len(provenance.rounds) != len(trace.rounds):
        issues.append(f"provenance has {len(provenance.rounds)} rounds for {len(trace.rounds)} trace rounds")
    for recorded, round_ in zip(provenance.rounds, trace.rounds, strict=False):
        if recorded.round_index != round_.number or recorded.think != round_.think:
            issues.append(f"round {round_.number} provenance does not match the trace")
    return tuple(issues)


def _invalid(task_id: str, path: Path, detail: str) -> SelfReplayResult:
    return SelfReplayResult(task_id, -1, passed=False, detail=detail, trace_path=path, valid=False)


def _from_replay(result: ReplayResult, path: Path, *, valid: bool = True) -> SelfReplayResult:
    return SelfReplayResult(
        result.task_id,
        result.round_index,
        result.passed,
        result.detail,
        tokens=result.tokens,
        tool_call=result.tool_call,
        terminal_reason=result.terminal_reason,
        trace_path=path,
        valid=valid,
    )


def replay_ds4py_with_engine(
    engine: Engine,
    selected: Sequence[tuple[str, Path]],
    *,
    provenance: Mapping[Path, TraceProvenance],
    context: SelfReplayContext,
) -> tuple[SelfReplayResult, ...]:
    """Reset per attempt, preserve rounds within each attempt, and execute no tools."""
    if not selected:
        return (SelfReplayResult("<selection>", -1, passed=False, detail="no ds4-py traces selected", valid=False),)
    results: list[SelfReplayResult] = []
    for task_id, selected_path in selected:
        path = Path(selected_path)
        recorded = provenance.get(path)
        if recorded is None:
            results.append(_invalid(task_id, path, "missing ds4-py trace provenance"))
            continue
        try:
            trace = read_replay(path)
            issues = _provenance_issues(engine, path, trace, recorded, context)
        except (OSError, ValueError) as error:
            results.append(_invalid(task_id, path, f"malformed ds4-py trace {path}: {error}"))
            continue
        if issues:
            results.append(_invalid(task_id, path, "invalid self-replay provenance: " + "; ".join(issues)))
            continue
        try:
            list(engine._worker.submit(ResetJob()))
        except Ds4pyError as error:
            results.append(_invalid(task_id, path, f"could not reset ds4-py session: {error}"))
            continue
        policy = SamplingPolicy.for_engine(engine.info, temperature=0.0, seed=1)
        rng = Rng(1)
        for ordinal, round_ in enumerate(trace.rounds):
            try:
                result = replay_round(engine, policy, rng, task_id, ordinal, round_)
                results.append(_from_replay(result, path))
            except (Ds4pyError, ReplayMismatch) as error:
                data = b"".join(engine.token_text(token) for token in round_.sampled)
                snapshot = _parse_c(engine, round_, [data])
                failed = ReplayResult(
                    task_id,
                    ordinal,
                    passed=False,
                    detail=str(error),
                    tool_call=snapshot.state == "DONE" and bool(snapshot.calls),
                    trace_path=path,
                )
                results.append(_from_replay(failed, path, valid=isinstance(error, ReplayMismatch)))
    return tuple(results)
