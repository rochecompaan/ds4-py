"""Validate restore provenance and Gate 5.2 positions."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.provenance import TraceProvenance, validate_session_pair
from bench.restore_plan import FOLLOWUP, CheckResult, RestoreError
from ds4py.engine.replay import read_replay

if TYPE_CHECKING:
    from bench.restore_plan import RestoreReference, RestoreRun


@dataclass(frozen=True)
class _RestoreMetadata:
    provenance: TraceProvenance
    save_record_sha256: str
    followup_prompt_sha256: str


def _restore_metadata(run: RestoreReference, label: str) -> _RestoreMetadata:
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


def validate_restore_pair(case1: RestoreRun, case2: RestoreReference) -> tuple[str, ...]:
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
