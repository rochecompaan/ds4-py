"""Slash-command registration and the built-in interactive commands."""

from __future__ import annotations

import inspect
import shlex
from dataclasses import dataclass

from prompt_toolkit.completion import Completion


class CommandError(ValueError):
    """A user-correctable slash-command error."""


@dataclass(frozen=True)
class _Command:
    handler: object
    description: str
    usage: str


class SlashCommandRegistry:
    """Extension seam for parsing, completing, and executing slash commands."""

    def __init__(self):
        self._commands = {}

    def register(self, name, handler, *, description, usage=None):
        name = name.lstrip("/")
        if not name or any(char.isspace() for char in name):
            raise ValueError("a command name must be one nonempty word")
        if name in self._commands:
            raise ValueError(f"command already registered: /{name}")
        self._commands[name] = _Command(
            handler, description, usage or f"/{name}",
        )

    async def execute(self, line):
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

    def completions(self, prefix):
        if not prefix.startswith("/") or any(char.isspace() for char in prefix):
            return []
        return [
            Completion(
                f"/{name} ", start_position=-len(prefix),
                display_meta=command.description,
            )
            for name, command in sorted(self._commands.items())
            if f"/{name}".startswith(prefix)
        ]


def _usage(usage):
    raise CommandError(f"usage: {usage}")


def _record(action, record):
    title = record.title or "(untitled)"
    return f"{action} {record.session_id} — {title} ({record.tokens} tokens)"


def install_builtin_commands(registry, runtime, *, request_exit):
    """Install exactly the commands supported by the interactive shell."""

    def no_args(args, usage):
        if args:
            _usage(usage)

    def exit_(*args):
        no_args(args, "/exit")
        request_exit()

    async def new(*args):
        no_args(args, "/new")
        return f"new session {await runtime.new()}"

    async def save(*args):
        no_args(args, "/save")
        return _record("saved", await runtime.save())

    async def switch(*args):
        if len(args) != 1:
            _usage("/switch <prefix>")
        return _record("switched to", await runtime.switch(args[0]))

    def hints(*args):
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
