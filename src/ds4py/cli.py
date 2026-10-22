"""Command-line entry point for ds4py."""

import argparse
from pathlib import Path
import sys

from ds4py._native.loader import default_lib_dir
from ds4py.abi.build import build_library, default_backend
from ds4py.abi.check import check_library, format_report
from ds4py.errors import Ds4pyError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ds4py")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build a checked libds4")
    build.add_argument("--src", type=Path, required=True)
    build.add_argument("--backend", choices=("cpu", "metal", "cuda", "rocm"), default=default_backend())
    build.add_argument("--out", type=Path, default=default_lib_dir())
    build.add_argument("-j", type=int)
    check = commands.add_parser("check", help="check a libds4 ABI without opening a model")
    check.add_argument("--lib", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        report = check_library(args.lib)
        print(format_report(report))
        return 1 if report.error is not None else 0
    try:
        installed = build_library(args.src, backend=args.backend, out_dir=args.out, jobs=args.j)
        print(installed)
    except Ds4pyError as exc:
        print(f"ds4py: {exc}", file=sys.stderr)
        return 1
    return 0
