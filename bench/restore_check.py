"""Orchestrate Gates 5.1-5.4 and preserve the restore driver interface."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.models import AgentConfig, TaskRunResult
from bench.provenance import Q8_F16_CACHE_ENV, Q8_F16_CACHE_MB
from bench.restore_equivalence import RestoreProvenance, process_restore_runner
from bench.restore_fidelity import _compare_reference
from bench.restore_history import same_history_command
from bench.restore_plan import (
    FOLLOWUP,
    CheckResult,
    CReferenceRun,
    RestoreConfig,
    RestoreError,
    RestoreOperations,
    RestoreResult,
    RestoreRun,
    RestoreSummary,
    SaveEvidence,
    attempt_command,
    make_stripped_copy,
    restore_command,
    run_attempt_with_save,
    switch_via_command,
    validate_save,
)
from bench.restore_position import _compare_restore, validate_restore_pair
from bench.session import report_bakeoff_session, report_provenance_environment
from ds4py.errors import Ds4pyError

__all__ = [
    "FOLLOWUP",
    "CReferenceRun",
    "CheckResult",
    "RestoreConfig",
    "RestoreError",
    "RestoreOperations",
    "RestoreResult",
    "RestoreRun",
    "RestoreSummary",
    "SaveEvidence",
    "attempt_command",
    "check_command",
    "check_restore",
    "check_result_saves",
    "main",
    "make_stripped_copy",
    "process_restore_runner",
    "restore_command",
    "run_attempt_with_save",
    "same_history_command",
    "switch_via_command",
    "validate_restore_pair",
    "validate_save",
]


def check_restore(
    save_dir: str | Path,
    work_dir: str | Path,
    *,
    operations: RestoreOperations,
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
        case1 = operations.restore(original_home, evidence.record.session_id, case1_trace)
        trace_paths.append(case1_trace)
        case2_trace = work_dir / "case-2.jsonl"
        case2 = operations.restore(stripped.parent.parent, evidence.record.session_id, case2_trace)
        trace_paths.append(case2_trace)
        same_history_trace = work_dir / "case-3.jsonl"
        same_history = operations.replay_same_history(
            original_home,
            evidence.record.session_id,
            same_history_trace,
            case1.trace_path,
        )
        trace_paths.append(same_history_trace)
        c_rebuild_trace = work_dir / "case-4.ndjson"
        c_rebuild = operations.c_rebuild(stripped.parent.parent, case2.trace_path, c_rebuild_trace)
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
            expected_plan=None,
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


def check_result_saves(
    task_results: Sequence[TaskRunResult],
    config: RestoreConfig,
    results_root: str | Path,
    *,
    provenance: RestoreProvenance | None = None,
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
                    operations=process_restore_runner(
                        config,
                        session=provenance.session if provenance else None,
                        workspace=getattr(item, "workspace", None),
                        provenance_environment=provenance.environment if provenance else None,
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
    session = report_bakeoff_session(data)
    if session is not None or environment:
        options["provenance"] = RestoreProvenance(session=session, environment=environment or None)
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
