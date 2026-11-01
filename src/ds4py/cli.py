"""Command-line entry point for ds4py."""

import argparse
from pathlib import Path
import sys

from ds4py._native.loader import default_lib_dir
from ds4py.abi.build import build_library, default_backend
from ds4py.abi.check import check_library, check_live, format_report
from ds4py.errors import Ds4pyError


def _core_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ds4py")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build a checked libds4")
    build.add_argument("--src", type=Path, required=True)
    build.add_argument("--backend", choices=("cpu", "metal", "cuda", "rocm"), default=default_backend())
    build.add_argument("--out", type=Path, default=default_lib_dir())
    build.add_argument("-j", type=int)
    check = commands.add_parser("check", help="check a libds4 ABI without opening a model")
    check.add_argument("--lib", type=Path)
    check.add_argument("--live", action="store_true")
    check.add_argument("--model", type=Path)
    check.add_argument("--ctx", type=int, default=4096)
    raw = commands.add_parser("raw", help="generate from an untemplated prompt")
    prompts = raw.add_mutually_exclusive_group(required=True)
    prompts.add_argument("prompt", nargs="?")
    prompts.add_argument("-p", dest="prompt_text")
    prompts.add_argument("--prompt-file", type=Path)
    raw.add_argument("-m", dest="model", default="ds4flash.gguf")
    raw.add_argument("-c", dest="ctx", type=int, default=100_000)
    raw.add_argument("-n", dest="tokens", type=int, default=50_000)
    raw.add_argument("--temp", type=float)
    raw.add_argument("--top-p", type=float)
    raw.add_argument("--min-p", type=float)
    raw.add_argument("--seed", type=int)
    backends = raw.add_mutually_exclusive_group()
    backends.add_argument("--backend")
    for backend in ("cpu", "metal", "cuda"):
        backends.add_argument(f"--{backend}", dest="backend", action="store_const", const=backend)
    raw.add_argument("-t", dest="threads", type=int, default=0)
    raw.add_argument("--trace", type=Path)
    raw.add_argument("--lib", type=Path)
    args = parser.parse_args(argv)
    if args.command == "raw":
        from ds4py.raw import command
        try:
            return command(args)
        except (Ds4pyError, OSError, UnicodeError) as exc:
            print(f"ds4py: {exc}", file=sys.stderr)
            return 1
    if args.command == "check":
        if args.live and args.model is None:
            parser.error("--live needs --model")
        try:
            report = (check_live(args.lib, args.model, ctx=args.ctx) if args.live
                      else check_library(args.lib))
        except (Ds4pyError, OSError) as exc:
            print(f"ds4py: {exc}", file=sys.stderr)
            return 1
        print(format_report(report))
        sync_failed = (report.manifest is not None and
                       report.manifest.live_checks.get("sync_plans", {}).get("ok") is False)
        return 1 if report.error is not None or sync_failed else 0
    try:
        installed = build_library(args.src, backend=args.backend, out_dir=args.out, jobs=args.j)
        print(installed)
    except Ds4pyError as exc:
        print(f"ds4py: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"build", "check", "raw"}:
        return _core_main(args)
    if "--raw" in args:
        args.remove("--raw")
        return _core_main(["raw", *args])
    from ds4py.agent_cli import main as agent_main
    return agent_main(args)
