"""Parse first-step evidence and compare fixed-history restore fidelity."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, TypeIs

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.comparison import StepCapture, TopLogit, compare_step
from bench.restore_plan import CheckResult, RestoreError, RestoreRun
from bench.restore_position import validate_restore_pair
from ds4py._json import is_json_object

if TYPE_CHECKING:
    from bench.restore_plan import RestoreReference


def _compare_reference(
    run: RestoreRun,
    reference: RestoreReference,
    *,
    label: str,
    expected_plan: str | None,
) -> tuple[CheckResult, list[str], bool]:
    """Compare one restore path with a reference that holds its KV history constant."""
    problems: list[str] = []
    valid = True
    provenance_issues = validate_restore_pair(run, reference)
    if provenance_issues:
        problems.extend(f"{label}: {issue}" for issue in provenance_issues)
        valid = False
    if expected_plan is not None:
        if not isinstance(reference, RestoreRun):
            raise RestoreError(f"{label}: Python reference evidence is missing")
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
