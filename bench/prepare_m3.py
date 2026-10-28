"""Prepare the twelve pinned M3 tasks from existing local checkouts."""

import argparse
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.m3 import (  # noqa: E402
    BenchError,
    load_manifest,
    prepare_manifest,
    validate_manifest_sources,
)


def benchmark_checkout(argument=None):
    value = argument or os.environ.get("POLYGLOT_BENCHMARK")
    if not value:
        raise BenchError(
            "set POLYGLOT_BENCHMARK to the pinned Aider polyglot-benchmark checkout"
        )
    return Path(value).expanduser().resolve()


def main(argv=None):
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=script.with_name("m3.yaml"))
    parser.add_argument("--benchmark", help="defaults to POLYGLOT_BENCHMARK")
    parser.add_argument(
        "--output", type=Path, default=script.parent / "results" / "workspaces",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=script.parents[1],
        help="ds4-py checkout whose HEAD supplies both repository tasks",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="verify the pinned checkout and exercise inputs without writing workspaces",
    )
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        benchmark = benchmark_checkout(args.benchmark)
        if args.check_only:
            validate_manifest_sources(manifest, benchmark)
            print(
                f"verified {len(manifest.exercises) + len(manifest.repository_tasks)} "
                f"pinned tasks; benchmark {manifest.commit}"
            )
        else:
            workspaces = prepare_manifest(
                manifest, args.output, benchmark, args.repo_root,
            )
            print(f"prepared {len(workspaces)} tasks under {args.output}")
    except BenchError as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
