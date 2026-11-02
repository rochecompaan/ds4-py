"""Slash-command registration and the built-in interactive commands."""

from __future__ import annotations

import inspect
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Never

from prompt_toolkit.completion import Completion

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ds4py.runtime import AgentRuntime
    from ds4py.store import SaveRecord


class CommandError(ValueError):
    """A user-correctable slash-command error."""


@dataclass(frozen=True)
class _Command:
    handler: Callable[..., object]
    description: str
    usage: str


class SlashCommandRegistry:
    """Extension seam for parsing, completing, and executing slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, _Command] = {}

    def register(
        self,
        name: str,
        handler: Callable[..., object],
        *,
        description: str,
        usage: str | None = None,
    ) -> None:
        name = name.lstrip("/")
        if not name or any(char.isspace() for char in name):
            raise ValueError("a command name must be one nonempty word")
        if name in self._commands:
            raise ValueError(f"command already registered: /{name}")
        self._commands[name] = _Command(
            handler,
            description,
            usage or f"/{name}",
        )

    async def execute(self, line: str) -> object | None:
        try:
            parts = shlex.split(line)
        except ValueError as error:
            raise CommandError(str(error)) from error
        if not parts or not parts[0].startswith("/"):
            raise CommandError("a slash command must start with '/'")
        name = parts[0][1:]
        command = self._commands.get(name)
        if command is None:
            raise CommandError(f"unknown command: /{name}")
        result = command.handler(*parts[1:])
        return await result if inspect.isawaitable(result) else result

    def completions(self, prefix: str) -> list[Completion]:
        if not prefix.startswith("/") or any(char.isspace() for char in prefix):
            return []
        return [
            Completion(
                f"/{name} ",
                start_position=-len(prefix),
                display_meta=command.description,
            )
            for name, command in sorted(self._commands.items())
            if f"/{name}".startswith(prefix)
        ]


def _usage(usage: str) -> Never:
    raise CommandError(f"usage: {usage}")


def _record(action: str, record: SaveRecord) -> str:
    title = record.title or "(untitled)"
    return f"{action} {record.session_id} — {title} ({record.tokens} tokens)"


def install_builtin_commands(
    registry: SlashCommandRegistry,
    runtime: AgentRuntime,
    *,
    request_exit: Callable[[], None],
) -> None:
    """Install exactly the commands supported by the interactive shell."""

    def no_args(args: Sequence[str], usage: str) -> None:
        if args:
            _usage(usage)

    def exit_(*args: str) -> None:
        no_args(args, "/exit")
        request_exit()

    async def new(*args: str) -> str:
        no_args(args, "/new")
        return f"new session {await runtime.new()}"

    async def save(*args: str) -> str:
        no_args(args, "/save")
        return _record("saved", await runtime.save())

    async def switch(*args: str) -> str:
        if len(args) != 1:
            _usage("/switch <prefix>")
        return _record("switched to", await runtime.switch(args[0]))

    def hints(*args: str) -> str:
        if len(args) != 1 or args[0] not in {"on", "off"}:
            _usage("/hints on|off")
        enabled = args[0] == "on"
        runtime.set_hints(enabled)
        return f"hints {args[0]}"

    for name, handler, description, usage in (
        ("exit", exit_, "Exit the shell", "/exit"),
        ("new", new, "Start a new session", "/new"),
        ("save", save, "Save the current session", "/save"),
        ("switch", switch, "Restore a saved session", "/switch <prefix>"),
        ("hints", hints, "Toggle programming hints", "/hints on|off"),
    ):
        registry.register(name, handler, description=description, usage=usage)
