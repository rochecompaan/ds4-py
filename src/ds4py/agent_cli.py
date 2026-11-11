"""Agent command-line parsing and in-process runtime composition."""

import argparse
import asyncio
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

from ds4py import ThinkMode
from ds4py._native.protocol import THINK_LEVEL_BASE
from ds4py.agent_composition import AgentExtension, DefaultAgentExtension, EngineOpenOptions
from ds4py.errors import Ds4pyError

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from pydantic_ai_harness.ask_user import AskUserRequest, AskUserResponse

    from ds4py.runtime import AgentRuntime
    from ds4py.tui_commands import SlashCommandRegistry


_MAX_THINK_LEVEL = 100

_THINK_MODES = {
    "none": ThinkMode.NONE,
    "high": ThinkMode.HIGH,
    "max": ThinkMode.MAX,
    "low": ThinkMode.LOW,
    "medium": ThinkMode.MEDIUM,
}


def _level(value: str) -> int:
    level = int(value)
    if not 1 <= level <= _MAX_THINK_LEVEL:
        raise argparse.ArgumentTypeError("think level must be between 1 and 100")
    return level


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def parser(extension: AgentExtension | None = None) -> argparse.ArgumentParser:
    extension = DefaultAgentExtension() if extension is None else extension
    result = argparse.ArgumentParser(prog=extension.program)
    result.add_argument("--non-interactive", action="store_true")
    prompts = result.add_mutually_exclusive_group()
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    result.add_argument("--system")
    result.add_argument("--prefix-file", type=Path)
    result.add_argument("--trace", type=Path)
    result.add_argument("--read-only", action="store_true")
    result.add_argument("-m", "--model", default="ds4flash.gguf")
    result.add_argument("-c", "--ctx", type=int, default=100_000)
    result.add_argument("-n", "--tokens", type=int, default=50_000)
    result.add_argument("--request-limit", type=_positive_int, default=100)
    result.add_argument("--temp", type=float)
    result.add_argument("--top-k", type=int)
    result.add_argument("--top-p", type=float)
    result.add_argument("--min-p", type=float)
    result.add_argument("--seed", type=int)
    think = result.add_mutually_exclusive_group()
    think.add_argument("--think", nargs="?", const="high", choices=tuple(_THINK_MODES))
    think.add_argument("--think-level", type=_level)
    think.add_argument("--think-max", dest="think_mode", action="store_const", const=ThinkMode.MAX)
    think.add_argument("--nothink", dest="think_mode", action="store_const", const=ThinkMode.NONE)
    backends = result.add_mutually_exclusive_group()
    backends.add_argument("--backend")
    for backend in ("cpu", "metal", "cuda"):
        backends.add_argument(f"--{backend}", dest="backend", action="store_const", const=backend)
    result.add_argument("-t", "--threads", type=int, default=0)
    result.add_argument("--lib", type=Path)
    extension.configure_parser(result)
    return result


def _think_mode(args: argparse.Namespace) -> int:
    if args.think_mode is not None:
        return int(args.think_mode)
    if args.think_level is not None:
        return THINK_LEVEL_BASE + int(args.think_level)
    if args.think is not None:
        return _THINK_MODES[args.think]
    return ThinkMode.HIGH


async def _question_answerer(request: AskUserRequest) -> AskUserResponse:
    """Read AskUser choices and return its typed response."""
    from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserResponse

    from ds4py.question_answers import selected_labels

    answers = []
    for question in request.questions:
        print(question.question, file=sys.stderr, flush=True)
        for index, option in enumerate(question.options, 1):
            description = f" — {option.description}" if bool(option.description) else ""
            print(f"  {index}. {option.label}{description}", file=sys.stderr, flush=True)
        answer = sys.stdin.readline()
        if answer == "":
            return AskUserResponse(cancelled=True)
        selected = selected_labels(question, answer)
        if selected is None:
            return AskUserResponse(cancelled=True)
        answers.append(AskUserAnswer(header=question.header, selected=selected))
    return AskUserResponse(answers=tuple(answers))


@contextmanager
def open_runtime(
    args: argparse.Namespace,
    project_dir: str | Path,
    *,
    read_only: bool = False,
    extension: AgentExtension | None = None,
    extension_config: object = None,
) -> Generator[AgentRuntime]:
    """Open the canonical agent stack with its CLI interaction services."""
    from ds4py.model import DS4Model
    from ds4py.runtime import AgentRuntime

    hooks_configured = extension is not None
    extension = DefaultAgentExtension() if extension is None else extension
    project_dir = Path(project_dir)
    options = EngineOpenOptions(
        model=args.model,
        ctx=args.ctx,
        backend=args.backend,
        think=_think_mode(args),
        seed=args.seed,
        threads=args.threads,
        lib=args.lib,
        tokens=args.tokens,
        interactive=not args.non_interactive,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
    )
    with extension.open_session(options, project_dir, extension_config) as configured:
        model = DS4Model(configured.session, sampling=configured.sampling, max_tokens=options.tokens)
        runtime = AgentRuntime(
            model,
            project_dir,
            read_only=read_only,
            system=args.system,
            interactive=options.interactive,
            question_answerer=_question_answerer if options.interactive else None,
            request_limit=args.request_limit,
            before_identity_change=extension.before_identity_change if hooks_configured else None,
            after_identity_change=extension.after_identity_change if hooks_configured else None,
        )
        try:
            yield runtime
        finally:
            asyncio.run(runtime.close())


def command(
    args: argparse.Namespace,
    *,
    extension: AgentExtension | None = None,
    extension_config: object = None,
    project_dir: Path | None = None,
) -> int:
    from ds4py.engine.prefix import load_prefix
    from ds4py.noninteractive import run_noninteractive
    from ds4py.prefill_trace import PrefillTraceWriter

    extension_configured = extension is not None
    extension = DefaultAgentExtension() if extension is None else extension
    project_dir = Path.cwd() if project_dir is None else project_dir
    prompt = args.prompt
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_bytes().decode("utf-8")
    with open_runtime(
        args,
        project_dir,
        read_only=args.read_only,
        extension=extension if extension_configured else None,
        extension_config=extension_config,
    ) as runtime:
        if args.prefix_file is not None:
            runtime.load_prefix(load_prefix(args.prefix_file))
        trace = (
            PrefillTraceWriter(
                runtime.model.session,
                args.trace,
                think=runtime.model.session.engine.think,
            )
            if args.trace is not None
            else nullcontext()
        )
        with trace:
            if args.non_interactive:
                return asyncio.run(run_noninteractive(runtime, prompt, program=extension.program))
            configure_commands = extension.configure_commands if extension_configured else None
            return asyncio.run(_run_tui(runtime, prompt, configure_commands=configure_commands))


async def _run_tui(
    runtime: AgentRuntime,
    prompt: str | None,
    *,
    configure_commands: Callable[[SlashCommandRegistry, AgentRuntime], None] | None = None,
) -> int:
    from ds4py.tui import TUI

    tui = TUI(runtime, configure_commands=configure_commands)
    from pydantic_ai_harness import AskUser

    components = getattr(getattr(runtime, "components", None), "components", ())
    questions = next(
        (item for item in components if isinstance(item, AskUser)),
        None,
    )
    if questions is not None:
        questions.answerer = tui.ask
    if prompt is not None:
        return await tui.run(prompt)
    return await tui.run()


def main(argv: list[str], *, extension: AgentExtension | None = None) -> int:
    active_extension = DefaultAgentExtension() if extension is None else extension
    arg_parser = parser(active_extension)
    args = arg_parser.parse_args(argv)
    if args.non_interactive and args.prompt is None and args.prompt_file is None:
        arg_parser.error("--non-interactive requires --prompt or --prompt-file")
    project_dir = Path.cwd()
    try:
        extension_config = active_extension.resolve_extension_config(args, project_dir)
        return command(
            args,
            extension=extension,
            extension_config=extension_config,
            project_dir=project_dir,
        )
    except (Ds4pyError, OSError, UnicodeError, ValueError) as exc:
        print(f"{active_extension.program}: {exc}", file=sys.stderr)
        return 1
