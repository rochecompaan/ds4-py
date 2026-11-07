"""Six-gate bakeoff report with per-task diagnostics and no log dumping."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bench.comparison import compare_cached_fraction, compare_rate
from bench.provenance import validate_pair
from ds4py._json import is_json_object
from ds4py.engine.events import STOP_REASONS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from bench.comparison import RateEvidence
    from bench.driver_equivalence import DriverEquivalenceResult
    from bench.models import TaskRunResult
    from bench.provenance import TraceProvenance
    from bench.restore_check import RestoreResult, RestoreSummary
    from bench.self_replay import SelfReplayResult
    from bench.trace_selection import ReplayResult
    from ds4py._json import JsonValue


REPORT_SCHEMA_VERSION = 2

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
    valid: bool = True


@dataclass(frozen=True)
class CapCounts:
    c: int
    ds4py: int


@dataclass(frozen=True)
class CapReport:
    tasks: dict[str, CapCounts]
    totals: CapCounts
    parity: GateResult


def _empty_caps() -> CapReport:
    return CapReport({}, CapCounts(0, 0), GateResult(passed=False, detail="not run"))


@dataclass(frozen=True)
class BakeoffReport:
    tasks: tuple[TaskRunResult, ...]
    restore: RestoreSummary
    gates: dict[str, GateResult]
    replay_diagnostic: tuple[ReplayResult, ...] | None = None
    driver_equivalence: DriverEquivalenceResult | None = None
    self_replay: tuple[SelfReplayResult, ...] | None = None
    rate_evidence: dict[str, tuple[RateEvidence, RateEvidence]] = field(default_factory=dict)
    cached_fractions: dict[str, tuple[float, float]] = field(default_factory=dict)
    caps: CapReport = field(default_factory=_empty_caps)
    provenance: tuple[tuple[TraceProvenance, TraceProvenance], ...] = ()
    schema_version: int = REPORT_SCHEMA_VERSION
    certification: str = "targeted"
    bakeoff_session_id: str = ""
    instance_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError("benchmark report schema_version must be 2")
        if self.certification not in {"targeted", "full-12-task"}:
            raise ValueError("certification must be targeted or full-12-task")

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates.values())

    def to_dict(self) -> dict[str, JsonValue]:
        value = _json(
            {
                "schema_version": self.schema_version,
                "certification": self.certification,
                "bakeoff_session_id": self.bakeoff_session_id,
                "instance_id": self.instance_id,
                "tasks": self.tasks,
                "restore": self.restore,
                "gates": self.gates,
                "replay_diagnostic": self.replay_diagnostic,
                "driver_equivalence": self.driver_equivalence,
                "self_replay": self.self_replay,
                "rates": {
                    task_id: {
                        "c": pair[0],
                        "ds4py": pair[1],
                        "result": compare_rate(pair[0], pair[1]),
                    }
                    for task_id, pair in self.rate_evidence.items()
                },
                "cached_prefix": {
                    task_id: {
                        "c": pair[0],
                        "ds4py": pair[1],
                        "result": compare_cached_fraction(pair[0], pair[1]),
                    }
                    for task_id, pair in self.cached_fractions.items()
                },
                "caps": self.caps,
                "provenance": tuple(
                    {"c": c, "ds4py": ds4py, "issues": validate_pair(c, ds4py)} for c, ds4py in self.provenance
                ),
            }
        )
        if not is_json_object(value):
            raise AssertionError("report root must be an object")
        return value

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    def compact_table(self) -> str:
        lines = ["GATES"]
        lines.extend(f"{name}: {_gate_status(gate)} — {gate.detail}" for name, gate in self.gates.items())
        lines.append(f"cap_parity: {self.caps.parity.detail}")
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
            caps = self.caps.tasks.get(task_id, CapCounts(0, 0))
            lines.append(f"{task_id} | max_tokens_exits | {caps.c} | {caps.ds4py} | {caps.ds4py - caps.c}")
        lines.append("")
        lines.append("PAIRED RATES")
        for task_id in _common_tasks(self.tasks)[0]:
            pair = self.rate_evidence.get(task_id)
            if pair is None:
                lines.append(f"{task_id} | paired_rates | INVALID — missing paired rate evidence")
                continue
            result = compare_rate(pair[0], pair[1])
            status = "INVALID" if not result.valid else "PASS" if result.passed else "FAIL"
            lines.append(f"{task_id} | paired_rates | {status} — {result.detail}")
        lines.append("")
        lines.append("PAIRED CACHED-PREFIX FRACTIONS")
        for task_id in _common_tasks(self.tasks)[0]:
            pair = self.cached_fractions.get(task_id)
            if pair is None:
                lines.append(f"{task_id} | cached_prefix | INVALID — missing paired cache evidence")
                continue
            result = compare_cached_fraction(pair[0], pair[1])
            status = "INVALID" if not result.valid else "PASS" if result.passed else "FAIL"
            lines.append(
                f"{task_id} | cached_prefix | C={_number(pair[0])} ds4py={_number(pair[1])} {status} — {result.detail}"
            )
        return "\n".join(lines) + "\n"


RunReport = BakeoffReport


def _gate_status(gate: GateResult) -> str:
    if not gate.valid:
        return "INVALID"
    return "PASS" if gate.passed else "FAIL"


def _json(value: object) -> JsonValue:
    if is_dataclass(value) and not isinstance(value, type):
        return _json(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported report value: {type(value).__name__}")


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


def _gate(name: str, passed: bool, detail: str, *, valid: bool = True) -> tuple[str, GateResult]:
    return name, GateResult(bool(passed) and valid, detail, valid)


def _cap_report(tasks: Sequence[TaskRunResult], parity: GateResult | None) -> CapReport:
    counts = {
        task_id: CapCounts(
            c=sum(
                item.terminal_reasons.count("max_tokens")
                for item in tasks
                if item.task_id == task_id and item.agent == "c"
            ),
            ds4py=sum(
                item.terminal_reasons.count("max_tokens")
                for item in tasks
                if item.task_id == task_id and item.agent == "ds4py"
            ),
        )
        for task_id in sorted({item.task_id for item in tasks})
    }
    totals = CapCounts(sum(count.c for count in counts.values()), sum(count.ds4py for count in counts.values()))
    return CapReport(counts, totals, parity or GateResult(passed=False, detail="not run"))


def _gate_six(
    driver: DriverEquivalenceResult | None,
    self_results: tuple[SelfReplayResult, ...] | None,
) -> GateResult:
    part_a = (
        GateResult(passed=False, detail="Part A: not run", valid=False)
        if driver is None
        else GateResult(driver.passed, f"Part A: {driver.detail}", driver.valid)
    )
    if self_results is None or len(self_results) == 0:
        part_b = GateResult(passed=False, detail="Part B: not run", valid=False)
    else:
        valid = all(item.valid for item in self_results)
        passed = valid and all(item.passed for item in self_results)
        part_b = GateResult(
            passed,
            f"Part B: {sum(item.passed for item in self_results)}/{len(self_results)} self-replay rounds passed",
            valid,
        )
    return GateResult(
        part_a.passed and part_b.passed,
        f"{part_a.detail}; {part_b.detail}",
        part_a.valid and part_b.valid,
    )


def _restore_runs(restore: RestoreSummary) -> dict[Path, RestoreResult]:
    return {Path(item.save_dir): item for item in restore.runs}


def _gate_two(python: Sequence[TaskRunResult], cap_parity: GateResult | None) -> tuple[str, GateResult]:
    tool_failures = []
    lifecycle_valid = True
    for item in python:
        accounting = item.tool_accounting
        reasons = []
        if not item.terminal_reasons:
            lifecycle_valid = False
            reasons.append("terminal evidence missing")
        unknown = set(item.terminal_reasons) - set(STOP_REASONS)
        if unknown:
            lifecycle_valid = False
            reasons.append(f"unknown terminal reasons={sorted(unknown)}")
        parser_stops = item.terminal_reasons.count("parser_error")
        if item.parser_errors or parser_stops:
            reasons.append(f"parser_error stops={max(item.parser_errors, parser_stops)}")
        if "interrupted" in item.terminal_reasons:
            reasons.append("unexpected interrupted model round")
        tool_terminals = item.terminal_reasons.count("tool_call_complete")
        if tool_terminals and (accounting is None or accounting.parsed < tool_terminals):
            lifecycle_valid = False
            reasons.append("tool_call_complete without a parsed call")
        if accounting is None:
            lifecycle_valid = False
            reasons.append("tool outcome accounting unavailable")
        elif not accounting.complete:
            lifecycle_valid = False
            reasons.append(f"tool outcomes {accounting.accounted}/{accounting.parsed} accounted")
        if item.errors:
            reasons.extend(item.errors)
        if reasons:
            tool_failures.append(f"{item.task_id}: {'; '.join(reasons)}")
    if cap_parity is not None and not cap_parity.passed:
        tool_failures.append(f"cap parity: {cap_parity.detail}")
    valid = lifecycle_valid and (cap_parity is None or cap_parity.valid)
    return _gate(
        "gate-2",
        bool(python) and not tool_failures,
        "all parsed calls have terminal outcomes; bad-argument retries and max_tokens are allowed"
        if not tool_failures
        else " | ".join(tool_failures),
        valid=valid,
    )


def _task_rate_failures(
    task_id: str,
    right: TaskRunResult,
    pair: tuple[RateEvidence, RateEvidence] | None,
    cached_pair: tuple[float, float] | None,
) -> tuple[list[str], bool]:
    failures = []
    valid = True
    if pair is None:
        valid = False
        failures.append(f"{task_id}: missing paired rate evidence")
    else:
        rate_result = compare_rate(pair[0], pair[1])
        if not rate_result.valid:
            valid = False
        if not rate_result.passed:
            failures.append(f"{task_id}: {rate_result.detail}")
    if cached_pair is None:
        valid = False
        failures.append(f"{task_id}: incomplete paired cached-prefix evidence")
    else:
        cached_result = compare_cached_fraction(*cached_pair)
        if not cached_result.valid:
            valid = False
        if not cached_result.passed:
            failures.append(f"{task_id}: {cached_result.detail}")
    if right.invalid_non_extend_rounds:
        failures.append(f"{task_id}: {right.invalid_non_extend_rounds} unexplained non-extension rounds")
    return failures, valid


def _gate_three(  # noqa: PLR0913, PLR0917 - Rate, cache, and provenance evidence have separate contracts.
    tasks: Sequence[TaskRunResult],
    common: Sequence[str],
    complete_pairs: bool,
    rates: Mapping[str, tuple[RateEvidence, RateEvidence]],
    cached_fractions: Mapping[str, tuple[float, float]],
    provenance_pairs: tuple[tuple[TraceProvenance, TraceProvenance], ...],
) -> tuple[str, GateResult]:
    failures = []
    valid = complete_pairs
    if not complete_pairs:
        failures.append("incomplete C/ds4-py task pairs")
    if not provenance_pairs:
        valid = False
        failures.append("missing paired driver provenance")
    for c_provenance, ds4py_provenance in provenance_pairs:
        issues = validate_pair(c_provenance, ds4py_provenance)
        if issues:
            valid = False
            failures.extend(issues)
    for task_id in common:
        right = _task(tasks, task_id, "ds4py")
        task_failures, task_valid = _task_rate_failures(
            task_id,
            right,
            rates.get(task_id),
            cached_fractions.get(task_id),
        )
        failures.extend(task_failures)
        valid = valid and task_valid
    return _gate(
        "gate-3",
        complete_pairs and not failures,
        "paired rates are within 0.25, cached-prefix fractions within 0.15, and later syncs extend KV"
        if not failures
        else " | ".join(failures),
        valid=valid,
    )


def build_report(  # noqa: PLR0913 - Each evidence input has a distinct live-run contract.
    task_results: Iterable[TaskRunResult],
    restore_results: RestoreSummary,
    replay_results: Iterable[ReplayResult] | None = None,
    *,
    rate_evidence: Mapping[str, tuple[RateEvidence, RateEvidence]] | None = None,
    cached_fractions: Mapping[str, tuple[float, float]] | None = None,
    provenance: Iterable[tuple[TraceProvenance, TraceProvenance]] | None = None,
    cap_parity: GateResult | None = None,
    driver_equivalence: DriverEquivalenceResult | None = None,
    self_replay_results: Iterable[SelfReplayResult] | None = None,
    certification: str = "targeted",
    bakeoff_session_id: str = "",
    instance_id: str = "",
) -> BakeoffReport:
    """Build all six gates from their independent evidence sections."""
    tasks = tuple(task_results)
    rates = dict(rate_evidence or {})
    cached = dict(cached_fractions or {})
    provenance_pairs = tuple(provenance or ())
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

    gates.append(_gate_two(python, cap_parity))
    gates.append(_gate_three(tasks, common, complete_pairs, rates, cached, provenance_pairs))

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
    restore_valid = (
        bool(python)
        and bool(expected)
        and all(save in restored and restored[save].valid for save in expected)
        and all(len(item.attempt_save_dirs) == item.attempts for item in python)
    )
    gates.append(
        _gate(
            "gate-5",
            restore_valid and not missing,
            f"case-1/case-2 first-step logits passed for all {len(expected)} attempts"
            if not missing
            else "missing or failed restore evidence: " + ", ".join(missing),
            valid=restore_valid,
        )
    )

    replay_values = None if replay_results is None else tuple(replay_results)
    self_values = None if self_replay_results is None else tuple(self_replay_results)
    gates.append(("gate-6", _gate_six(driver_equivalence, self_values)))
    return BakeoffReport(
        tasks=tasks,
        restore=restore_results,
        gates=dict(gates),
        replay_diagnostic=replay_values,
        driver_equivalence=driver_equivalence,
        self_replay=self_values,
        rate_evidence=rates,
        cached_fractions=cached,
        caps=_cap_report(tasks, cap_parity),
        provenance=provenance_pairs,
        certification=certification,
        bakeoff_session_id=bakeoff_session_id,
        instance_id=instance_id,
    )
