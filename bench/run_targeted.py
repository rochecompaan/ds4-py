"""Run selected ds4-py bakeoff tasks in independent repetitions."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.run_bakeoff import AgentConfig, run_task
from bench.workspaces import clone_workspace, load_manifest, prepare_manifest


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("bakeoff.yaml"))
    parser.add_argument("--benchmark", type=Path, default=root.parent / "benchmark")
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--tokens", type=_positive_int, default=8192)
    parser.add_argument("--request-limit", type=_positive_int, default=100)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results_root = args.results.resolve()
    results_root.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest(args.manifest)
    staged = prepare_manifest(manifest, results_root / "staged", args.benchmark, args.repo_root)
    workspaces = {workspace.task_id: workspace for workspace in staged}
    missing = sorted(set(args.task) - workspaces.keys())
    if missing:
        raise ValueError(f"unknown bakeoff tasks: {', '.join(missing)}")

    config = AgentConfig(
        "ds4py",
        (sys.executable, str(Path(__file__).with_name("restore_check.py")), "attempt"),
        model=args.model,
        backend=args.backend,
        ctx=manifest.ctx,
        temperature=manifest.temperature,
        seed=manifest.seed,
        tokens=args.tokens,
        request_limit=args.request_limit,
    )
    records: list[dict[str, object]] = []
    failed = False
    for task_id in args.task:
        for repetition in range(1, args.repetitions + 1):
            run_root = results_root / task_id / f"run-{repetition}"
            workspace = replace(
                clone_workspace(workspaces[task_id], run_root / "workspace"),
                max_attempts=1,
            )
            completed = run_task(workspace, config, run_root)
            record = asdict(completed)
            record["repetition"] = repetition
            records.append(record)
            passed = completed.passed and not completed.errors
            failed = failed or not passed
            print(
                f"[targeted] {task_id} run-{repetition}: {'PASS' if passed else 'FAIL'}",
                flush=True,
            )

    (results_root / "targeted.json").write_text(json.dumps(records, indent=2, sort_keys=True, default=str) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
