"""Compare same-host C and ds4-py reference-driver evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bench.comparison import RateEvidence, compare_step
from bench.provenance import validate_pair

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bench.comparison import ComparisonResult
    from bench.driver_cases import DriverRoundEvidence
    from bench.provenance import TraceProvenance


@dataclass(frozen=True)
class DriverStepResult:
    index: int
    result: ComparisonResult


@dataclass(frozen=True)
class DriverRoundResult:
    task_id: str
    attempt: int
    round_index: int
    passed: bool
    detail: str
    valid: bool = True
    sync_only: bool = False
    steps: tuple[DriverStepResult, ...] = ()


@dataclass(frozen=True)
class DriverEquivalenceResult:
    rounds: tuple[DriverRoundResult, ...]
    decode_steps: int
    passed: bool
    detail: str
    valid: bool = True


def _key(round_: DriverRoundEvidence) -> tuple[str, int, int]:
    return round_.case.task_id, round_.case.attempt, round_.case.round_index


def _provenance_issues(
    provenance: TraceProvenance,
    round_: DriverRoundEvidence,
    label: str,
) -> list[str]:
    matches = [item for item in provenance.rounds if item.round_index == round_.case.round_index]
    if len(matches) != 1:
        return [f"{label} provenance has {len(matches)} entries for round {round_.case.round_index}"]
    expected = matches[0]
    issues = []
    if expected.think != round_.case.think:
        issues.append(f"{label} think mode differs from provenance")
    if expected.sync_calls != round_.sync_calls:
        issues.append(f"{label} sync evidence differs from provenance")
    if expected.progress_chunks != round_.progress_chunks:
        issues.append(f"{label} progress evidence differs from provenance")
    return issues


def _missing_round(key: tuple[str, int, int], detail: str) -> DriverRoundResult:
    task_id, attempt, round_index = key
    return DriverRoundResult(task_id, attempt, round_index, passed=False, detail=detail, valid=False)


def _compare_round(
    c_round: DriverRoundEvidence,
    ds4py_round: DriverRoundEvidence,
    c_provenance: TraceProvenance,
    ds4py_provenance: TraceProvenance,
    shared_issues: tuple[str, ...],
) -> DriverRoundResult:
    case = c_round.case
    issues = list(shared_issues)
    issues.extend(_provenance_issues(c_provenance, c_round, "C"))
    issues.extend(_provenance_issues(ds4py_provenance, ds4py_round, "ds4-py"))
    if c_round.source_trace != ds4py_round.source_trace:
        issues.append("source trace paths differ")
    if case != ds4py_round.case:
        issues.append("case manifest fields differ")
    if c_round.sync_calls != ds4py_round.sync_calls:
        issues.append("prefill boundaries differ")
    if c_round.progress_chunks != ds4py_round.progress_chunks:
        issues.append("progress chunks differ")
    if len(c_round.steps) != len(ds4py_round.steps):
        issues.append(f"decode step counts differ: C={len(c_round.steps)}, ds4-py={len(ds4py_round.steps)}")
    if len(c_round.steps) != case.steps or len(ds4py_round.steps) != case.steps:
        issues.append(f"decode evidence does not cover requested {case.steps} steps")

    comparisons = tuple(
        DriverStepResult(index, compare_step(c_step, ds4py_step))
        for index, (c_step, ds4py_step) in enumerate(zip(c_round.steps, ds4py_round.steps, strict=False))
    )
    numeric_failures = [f"step {item.index}: {item.result.detail}" for item in comparisons if not item.result.passed]
    valid = not issues and all(item.result.valid for item in comparisons)
    passed = valid and not numeric_failures
    detail = "ok" if passed else "; ".join([*issues, *numeric_failures]) or "invalid comparison"
    return DriverRoundResult(
        case.task_id,
        case.attempt,
        case.round_index,
        passed,
        detail,
        valid=valid,
        sync_only=case.steps == 0,
        steps=comparisons,
    )


def compare_driver_rounds(
    c_rounds: Sequence[DriverRoundEvidence],
    ds4py_rounds: Sequence[DriverRoundEvidence],
    *,
    c_provenance: TraceProvenance,
    ds4py_provenance: TraceProvenance,
    diagnostic_issues: Sequence[str] = (),
) -> DriverEquivalenceResult:
    """Compare every selected round and retain numeric and invalid failures."""
    c_by_key = {_key(round_): round_ for round_ in c_rounds}
    ds4py_by_key = {_key(round_): round_ for round_ in ds4py_rounds}
    c_duplicates = len(c_by_key) != len(c_rounds)
    ds4py_duplicates = len(ds4py_by_key) != len(ds4py_rounds)
    shared_issues = (*validate_pair(c_provenance, ds4py_provenance), *diagnostic_issues)
    keys = list(c_by_key)
    keys.extend(key for key in ds4py_by_key if key not in c_by_key)
    results = []
    for key in keys:
        c_round = c_by_key.get(key)
        ds4py_round = ds4py_by_key.get(key)
        if c_round is None:
            results.append(_missing_round(key, "unexpected ds4-py round; missing C round"))
            continue
        if ds4py_round is None:
            results.append(_missing_round(key, "missing ds4-py round"))
            continue
        duplicate_issues = []
        if c_duplicates:
            duplicate_issues.append("duplicate C round keys")
        if ds4py_duplicates:
            duplicate_issues.append("duplicate ds4-py round keys")
        results.append(
            _compare_round(
                c_round,
                ds4py_round,
                c_provenance,
                ds4py_provenance,
                (*shared_issues, *duplicate_issues),
            )
        )

    decode_steps = sum(len(round_.steps) for round_ in results)
    valid = bool(results) and decode_steps > 0 and all(round_.valid for round_ in results)
    passed = valid and all(round_.passed for round_ in results)
    detail = f"{sum(round_.passed for round_ in results)}/{len(results)} rounds passed; {decode_steps} decode steps"
    if not results:
        detail = "no selected C rounds"
        if shared_issues:
            detail += "; " + "; ".join(shared_issues)
    elif decode_steps == 0:
        detail += "; complete set has no decode coverage"
    first_failure = next((round_.detail for round_ in results if not round_.passed), None)
    if first_failure is not None:
        detail += f"; first failure: {first_failure}"
    return DriverEquivalenceResult(tuple(results), decode_steps, passed, detail, valid)


def combine_driver_results(results: Sequence[DriverEquivalenceResult]) -> DriverEquivalenceResult:
    """Combine independent trace comparisons without dropping failed rounds."""
    rounds = tuple(round_ for result in results for round_ in result.rounds)
    decode_steps = sum(result.decode_steps for result in results)
    valid = bool(results) and decode_steps > 0 and all(result.valid for result in results)
    passed = valid and all(result.passed for result in results)
    detail = f"{sum(round_.passed for round_ in rounds)}/{len(rounds)} rounds passed; {decode_steps} decode steps"
    first_failure = next((result.detail for result in results if not result.passed), None)
    if first_failure is not None:
        detail += f"; first failure: {first_failure}"
    return DriverEquivalenceResult(rounds, decode_steps, passed, detail, valid)


def rates_by_task(rounds: Sequence[DriverRoundEvidence]) -> dict[str, RateEvidence]:
    """Aggregate the same reference-driver timing evidence used by Part A."""
    totals: dict[str, tuple[int, float, int, float]] = {}
    for round_ in rounds:
        task_id = round_.case.task_id
        prefill_tokens, prefill_seconds, decode_steps, decode_seconds = totals.get(task_id, (0, 0.0, 0, 0.0))
        totals[task_id] = (
            prefill_tokens + sum(call.prefill_tokens for call in round_.sync_calls),
            prefill_seconds + round_.prefill_seconds,
            decode_steps + len(round_.steps),
            decode_seconds + round_.decode_seconds,
        )
    return {task_id: RateEvidence(task_id, *values) for task_id, values in totals.items()}
