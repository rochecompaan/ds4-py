"""Six-gate bakeoff report with per-task diagnostics and no log dumping."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bench.replay_c_traces import ReplayResult
    from bench.restore_check import RestoreResult, RestoreSummary
    from bench.run_bakeoff import TaskRunResult


METRICS = (
    "wall_seconds",
    "progress_events",
    "tokens",
    "tool_calls",
    "avg_prefill_reuse",
    "non_extend_rounds",
)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    detail: str


@dataclass(frozen=True)
class BakeoffReport:
    tasks: tuple[TaskRunResult, ...]
    restore: RestoreSummary
    gates: dict[str, GateResult]
    replay: tuple[ReplayResult, ...] | None = None

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates.values())

    def to_dict(self) -> dict[str, Any]:
        return _json(
            {
                "tasks": self.tasks,
                "restore": self.restore,
                "gates": self.gates,
                "replay": self.replay,
            }
        )

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    def compact_table(self) -> str:
        lines = ["GATES"]
        lines.extend(
            f"{name}: {'PASS' if gate.passed else 'FAIL'} — {gate.detail}" for name, gate in self.gates.items()
        )
        lines.append("")
        lines.append("task | metric | C | ds4py | delta")
        lines.append("--- | --- | ---: | ---: | ---:")
        for task_id in _common_tasks(self.tasks)[0]:
            c = _task(self.tasks, task_id, "c")
            python = _task(self.tasks, task_id, "ds4py")
            for metric in METRICS:
                left, right = getattr(c, metric), getattr(python, metric)
                delta = right - left if left is not None and right is not None else None
                lines.append(f"{task_id} | {metric} | {_number(left)} | {_number(right)} | {_number(delta)}")
        return "\n".join(lines) + "\n"


RunReport = BakeoffReport


def _json(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _task(
    tasks: Sequence[TaskRunResult],
    task_id: str,
    agent: str,
) -> TaskRunResult:
    return next(item for item in tasks if item.task_id == task_id and item.agent == agent)


def _common_tasks(
    tasks: Sequence[TaskRunResult],
) -> tuple[list[str], set[str], set[str]]:
    c_ids = {item.task_id for item in tasks if item.agent == "c"}
    python_ids = {item.task_id for item in tasks if item.agent == "ds4py"}
    return sorted(c_ids & python_ids), c_ids, python_ids


def _gate(name: str, passed: bool, detail: str) -> tuple[str, GateResult]:
    return name, GateResult(bool(passed), detail)


def replay_gate(replay_results: Iterable[ReplayResult] | None) -> GateResult:
    """Gate 6 requires complete per-round evidence and a real tool-call round."""
    values = None if replay_results is None else tuple(replay_results)
    coverage = any(item.tool_call for item in values or ())
    passed = bool(values) and all(item.passed for item in values) and coverage
    detail = (
        "not run"
        if values is None
        else (
            f"{sum(item.passed for item in values)}/{len(values)} replay checks passed; "
            f"tool-call round {'present' if coverage else 'missing'}"
        )
    )
    return GateResult(passed, detail)


def update_replay_report(
    path: str | Path,
    replay_results: Iterable[ReplayResult],
) -> GateResult:
    """Replace only replay evidence and gate 6 in an existing six-gate report."""
    path = Path(path)
    data = json.loads(path.read_text())
    values = tuple(replay_results)
    gate = replay_gate(values)
    data["replay"] = _json(values)
    data["gates"]["gate-6"] = asdict(gate)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return gate


def _restore_runs(restore: RestoreSummary) -> dict[Path, RestoreResult]:
    return {Path(item.save_dir): item for item in restore.runs}


def build_report(
    task_results: Iterable[TaskRunResult],
    restore_results: RestoreSummary,
    replay_results: Iterable[ReplayResult] | None = None,
) -> BakeoffReport:
    """Build gates 1–5 and reserve gate 6 for the C replay phase."""
    tasks = tuple(task_results)
    common, c_ids, python_ids = _common_tasks(tasks)
    c = [item for item in tasks if item.agent == "c"]
    python = [item for item in tasks if item.agent == "ds4py"]
    c_passes = sum(item.passed for item in c)
    python_passes = sum(item.passed for item in python)
    complete_pairs = bool(common) and c_ids == python_ids and len(c) == len(python) == len(common)
    gate1 = complete_pairs and python_passes >= c_passes - 1
    gates = [
        _gate(
            "gate-1",
            gate1,
            f"passes C={c_passes}/{len(c)} ds4py={python_passes}/{len(python)}; "
            f"attempts C={sum(item.attempts for item in c)} "
            f"ds4py={sum(item.attempts for item in python)}; common tasks={len(common)}",
        )
    ]

    tool_failures = []
    for item in python:
        accounting = item.tool_accounting
        reasons = []
        if item.parser_errors:
            reasons.append(f"parser_error stops={item.parser_errors}")
        if accounting is None:
            reasons.append("tool outcome accounting unavailable")
        elif not accounting.complete:
            reasons.append(f"tool outcomes {accounting.accounted}/{accounting.parsed} accounted")
        if item.errors:
            reasons.extend(item.errors)
        if reasons:
            tool_failures.append(f"{item.task_id}: {'; '.join(reasons)}")
    gates.append(
        _gate(
            "gate-2",
            bool(python) and not tool_failures,
            "all parsed calls have terminal outcomes; bad-argument retries are allowed"
            if not tool_failures
            else " | ".join(tool_failures),
        )
    )

    timing_failures = []
    for task_id in common:
        left, right = _task(tasks, task_id, "c"), _task(tasks, task_id, "ds4py")
        if left.wall_seconds <= 0 or right.avg_prefill_reuse is None or left.avg_prefill_reuse is None:
            timing_failures.append(f"{task_id}: incomplete timing/reuse evidence")
        elif right.wall_seconds > 1.5 * left.wall_seconds:
            timing_failures.append(
                f"{task_id}: ds4py {right.wall_seconds:.3f}s exceeds 1.5x C {left.wall_seconds:.3f}s"
            )
        if right.invalid_non_extend_rounds:
            timing_failures.append(f"{task_id}: {right.invalid_non_extend_rounds} unexplained non-extension rounds")
    gates.append(
        _gate(
            "gate-3",
            complete_pairs and not timing_failures,
            "per-task time is within 1.5x and later syncs extend KV or name an exception"
            if not timing_failures
            else " | ".join(timing_failures),
        )
    )

    interruption = restore_results.interruption
    gates.append(
        _gate(
            "gate-4",
            interruption.passed,
            f"interruption: {interruption.detail}",
        )
    )

    expected = [save for item in python for save in item.attempt_save_dirs]
    restored = _restore_runs(restore_results)
    missing = [str(save) for save in expected if save not in restored or not restored[save].passed]
    missing.extend(
        f"{item.task_id}: attempt without save ({len(item.attempt_save_dirs)}/{item.attempts})"
        for item in python
        if len(item.attempt_save_dirs) != item.attempts
    )
    gates.append(
        _gate(
            "gate-5",
            bool(python) and bool(expected) and not missing,
            f"case-1/case-2 restore passed for all {len(expected)} attempts"
            if not missing
            else "missing or failed restore evidence: " + ", ".join(missing),
        )
    )

    replay_values = None if replay_results is None else tuple(replay_results)
    gates.append(("gate-6", replay_gate(replay_values)))
    return BakeoffReport(tasks, restore_results, dict(gates), replay_values)
