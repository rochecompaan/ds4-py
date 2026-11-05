"""Optional paired round-0 response-budget checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bench.provenance import validate_pair
from bench.report import GateResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bench.driver_cases import DriverRoundEvidence
    from bench.provenance import TraceProvenance

_CAP_TERMINALS = {"max_tokens", "stop_token", "tool_call_complete"}


def _issues(
    c_round: DriverRoundEvidence,
    ds4py_round: DriverRoundEvidence,
    c_provenance: TraceProvenance,
    ds4py_provenance: TraceProvenance,
    response_budget: int,
) -> list[str]:
    issues = list(validate_pair(c_provenance, ds4py_provenance))
    case = c_round.case
    if case != ds4py_round.case:
        issues.append("driver case fields differ")
    if c_round.source_trace != ds4py_round.source_trace:
        issues.append("source trace paths differ")
    if case.round_index != 0:
        issues.append(f"round {case.round_index} is not round 0")
    if case.steps != response_budget:
        issues.append(f"driver budget {case.steps} differs from response budget {response_budget}")
    if c_round.sync_calls != ds4py_round.sync_calls:
        issues.append("sync evidence differs")
    for label, round_ in (("C", c_round), ("ds4-py", ds4py_round)):
        if round_.terminal_reason not in _CAP_TERMINALS:
            issues.append(f"{label} terminal reason is invalid for cap parity: {round_.terminal_reason!r}")
        if len(round_.steps) > response_budget:
            issues.append(f"{label} decode steps exceed the response budget")
        if round_.terminal_reason == "max_tokens" and len(round_.steps) != response_budget:
            issues.append(f"{label} max_tokens evidence does not reach the response budget")
    return issues


def compare_cap_parity(
    pairs: Sequence[tuple[DriverRoundEvidence, DriverRoundEvidence]],
    *,
    provenance: Sequence[tuple[TraceProvenance, TraceProvenance]],
    response_budget: int,
    diagnostic_issues: Sequence[str] = (),
) -> GateResult:
    """Compare cap outcomes for identical round-0 prompts without comparing tokens."""
    issues = list(diagnostic_issues)
    if not pairs:
        issues.append("no paired round-0 prompts")
    if response_budget < 1:
        issues.append("response budget must be positive")
    if len(pairs) != len(provenance):
        issues.append("cap evidence and provenance counts differ")
        return GateResult(passed=False, detail="; ".join(issues), valid=False)

    differences = []
    keys = set()
    for (c_round, ds4py_round), (c_provenance, ds4py_provenance) in zip(pairs, provenance, strict=True):
        issues.extend(_issues(c_round, ds4py_round, c_provenance, ds4py_provenance, response_budget))
        case = c_round.case
        key = (c_round.source_trace, case.task_id, case.attempt)
        if key in keys:
            issues.append(f"duplicate cap prompt: {case.task_id} attempt {case.attempt}")
        keys.add(key)
        c_capped = len(c_round.steps) == response_budget
        ds4py_capped = len(ds4py_round.steps) == response_budget
        if c_capped != ds4py_capped:
            differences.append(
                f"{case.task_id} attempt {case.attempt}: "
                f"C={'cap' if c_capped else 'stop'}, ds4-py={'cap' if ds4py_capped else 'stop'}"
            )
    valid = not issues
    passed = valid and not differences
    detail = (
        f"round-0 cap outcomes match for {len(pairs)} prompts"
        if passed
        else "; ".join([*issues, *differences]) or "cap outcomes differ"
    )
    return GateResult(passed, detail, valid)
