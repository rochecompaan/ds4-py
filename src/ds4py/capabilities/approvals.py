"""Approval capability for tools that mutate the workspace or run commands."""

from dataclasses import dataclass
from enum import Enum
import inspect
import shlex
from pathlib import Path

from pydantic_ai import ToolDenied
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import SkipToolExecution


class ApprovalDecision(Enum):
    ONCE = "once"
    DENY = "deny"
    TOOL = "tool"
    PATTERN = "pattern"
    STOP = "stop"


@dataclass
class ApprovalRequest:
    tool_name: str
    args: dict
    command: str | None = None
    path: Path | None = None


_MENU = {
    "y": ApprovalDecision.ONCE,
    "n": ApprovalDecision.DENY,
    "t": ApprovalDecision.TOOL,
    "a": ApprovalDecision.PATTERN,
    "s": ApprovalDecision.STOP,
}

_COMMAND_TOOLS = frozenset(("run_command", "start_command"))
_PATH_TOOLS = frozenset(("write_file", "edit_file", "create_directory"))


class DS4Approvals(AbstractCapability):
    """Ask before commands and writes outside the project unless bypassed."""

    def __init__(self, answerer, *, yolo=False, project_dir):
        self.answerer = answerer
        self.yolo = yolo
        self.project_dir = Path(project_dir).resolve()
        self._approved_tools = set()
        self._approved_patterns = set()

    async def before_tool_execute(self, ctx, *, call, tool_def, args):
        if self.yolo:
            return args

        tool_name = call.tool_name
        if tool_name not in _COMMAND_TOOLS and tool_name not in _PATH_TOOLS:
            return args

        if tool_name in self._approved_tools:
            return args

        command = None
        path = None
        pattern = None
        if tool_name in _COMMAND_TOOLS:
            command = args.get("command", "")
            if not isinstance(command, str) or not command:
                raise SkipToolExecution(
                    ToolDenied(f"Tool {tool_name} denied: unrecognized arguments.")
                )
            try:
                parts = shlex.split(command)
            except ValueError:
                parts = []
            pattern = parts[0] if parts else None
        else:
            path_value = args.get("path") or args.get("file_path")
            if not path_value:
                raise SkipToolExecution(
                    ToolDenied(f"Tool {tool_name} denied: unrecognized arguments.")
                )
            candidate = Path(path_value)
            if not candidate.is_absolute():
                candidate = self.project_dir / candidate
            path = candidate.resolve()
            if path.is_relative_to(self.project_dir):
                return args
            pattern = str(path.parent)

        if pattern and (tool_name, pattern) in self._approved_patterns:
            return args

        decision = await self._ask(ApprovalRequest(tool_name, dict(args), command, path))
        if decision == ApprovalDecision.ONCE:
            return args
        if decision == ApprovalDecision.TOOL:
            self._approved_tools.add(tool_name)
            return args
        if decision == ApprovalDecision.PATTERN:
            if pattern:
                self._approved_patterns.add((tool_name, pattern))
            return args
        if decision == ApprovalDecision.STOP:
            ctx.cancel()
            raise SkipToolExecution(ToolDenied(f"Tool {tool_name} denied; session stopped."))
        raise SkipToolExecution(ToolDenied(f"Tool {tool_name} denied."))

    async def _ask(self, request):
        if self.answerer is None:
            raise SkipToolExecution(
                ToolDenied(
                    "Tool denied: no approval answerer is configured; "
                    "run interactively or enable yolo mode.",
                ),
            )
        try:
            result = self.answerer(request)
            if inspect.isawaitable(result):
                result = await result
        except (EOFError, StopIteration):
            raise SkipToolExecution(
                ToolDenied("Tool denied: noninteractive input reached EOF; run interactively to approve."),
            )
        if isinstance(result, str):
            return _MENU.get(result.strip().lower(), ApprovalDecision.DENY)
        if isinstance(result, ApprovalDecision):
            return result
        if result is None:
            raise SkipToolExecution(ToolDenied("Tool denied: no answer was supplied."))
        return ApprovalDecision.DENY
