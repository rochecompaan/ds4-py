"""Agent command-line parsing and in-process runtime composition."""

import argparse
import asyncio
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys

from ds4py import ThinkMode
from ds4py.errors import Ds4pyError


_THINK_MODES = {
    "none": ThinkMode.NONE,
    "high": ThinkMode.HIGH,
    "max": ThinkMode.MAX,
    "low": ThinkMode.LOW,
    "medium": ThinkMode.MEDIUM,
}


def _level(value: str) -> int:
    level = int(value)
    if not 1 <= level <= 100:
        raise argparse.ArgumentTypeError("think level must be between 1 and 100")
    return level


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="ds4py")
    result.add_argument("--non-interactive", action="store_true")
    prompts = result.add_mutually_exclusive_group()
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    result.add_argument("--system")
    result.add_argument("--prefix-file", type=Path)
    result.add_argument("--trace", type=Path)
    result.add_argument("--read-only", action="store_true")
    result.add_argument("--yolo", action="store_true")
    result.add_argument("-m", "--model", default="ds4flash.gguf")
    result.add_argument("-c", "--ctx", type=int, default=100_000)
    result.add_argument("-n", "--tokens", type=int, default=50_000)
    result.add_argument("--temp", type=float)
    result.add_argument("--top-p", type=float)
    result.add_argument("--min-p", type=float)
    result.add_argument("--seed", type=int)
    think = result.add_mutually_exclusive_group()
    think.add_argument("--think", nargs="?", const="high", choices=tuple(_THINK_MODES))
    think.add_argument("--think-level", type=_level)
    think.add_argument("--think-max", dest="think_mode", action="store_const",
                       const=ThinkMode.MAX)
    think.add_argument("--nothink", dest="think_mode", action="store_const",
                       const=ThinkMode.NONE)
    backends = result.add_mutually_exclusive_group()
    backends.add_argument("--backend")
    for backend in ("cpu", "metal", "cuda"):
        backends.add_argument(f"--{backend}", dest="backend", action="store_const",
                              const=backend)
    result.add_argument("-t", "--threads", type=int, default=0)
    result.add_argument("--lib", type=Path)
    return result


def _think_mode(args) -> int:
    if args.think_mode is not None:
        return args.think_mode
    if args.think_level is not None:
        return 1000 + args.think_level
    if args.think is not None:
        return _THINK_MODES[args.think]
    return ThinkMode.HIGH


def _approval_answerer(request):
    detail = request.command or request.path or request.tool_name
    print(f"Approve {request.tool_name}: {detail}? [y/N]", file=sys.stderr, flush=True)
    answer = sys.stdin.readline()
    if answer == "":
        raise EOFError("approval input reached EOF")
    return answer


@contextmanager
def open_runtime(args, project_dir, *, read_only=False):
    """Open the canonical agent stack with an explicit CLI interaction policy."""
    from ds4py import open as open_engine
    from ds4py.engine.sampling import SamplingPolicy
    from ds4py.model import DS4Model
    from ds4py.runtime import AgentRuntime

    with open_engine(
        args.model, ctx=args.ctx, backend=args.backend, think=_think_mode(args),
        seed=args.seed, threads=args.threads, lib=args.lib,
    ) as engine:
        session = engine.session()
        sampling = SamplingPolicy.for_engine(
            engine.info, temperature=args.temp, top_p=args.top_p,
            min_p=args.min_p, seed=args.seed,
        )
        model = DS4Model(session, sampling=sampling, max_tokens=args.tokens)
        runtime = AgentRuntime(
            model, project_dir, read_only=read_only, yolo=args.yolo,
            system=args.system, interactive=not args.non_interactive,
            answerer=None if args.yolo else _approval_answerer,
        )
        try:
            yield runtime
        finally:
            asyncio.run(runtime.close())


def command(args) -> int:
    from ds4py.engine.prefix import load_prefix
    from ds4py.noninteractive import run_noninteractive
    from ds4py.prefill_trace import PrefillTraceWriter

    prompt = args.prompt
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_bytes().decode("utf-8")
    with open_runtime(args, Path.cwd(), read_only=args.read_only) as runtime:
        if args.prefix_file is not None:
            runtime.load_prefix(load_prefix(args.prefix_file))
        trace = (
            PrefillTraceWriter(
                runtime.model.session, args.trace,
                think=runtime.model.session.engine.think,
            )
            if args.trace is not None else nullcontext()
        )
        with trace:
            if args.non_interactive:
                return asyncio.run(run_noninteractive(runtime, prompt))
            return asyncio.run(_run_tui(runtime, prompt))


async def _run_tui(runtime, prompt):
    from ds4py.tui import TUI

    tui = TUI(runtime)
    if prompt is not None:
        return await tui.run(prompt)
    return await tui.run()


def main(argv: list[str]) -> int:
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if args.non_interactive and args.prompt is None and args.prompt_file is None:
        arg_parser.error("--non-interactive requires --prompt or --prompt-file")
    try:
        return command(args)
    except (Ds4pyError, OSError, UnicodeError, ValueError) as exc:
        print(f"ds4py: {exc}", file=sys.stderr)
        return 1
