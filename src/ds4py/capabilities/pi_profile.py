"""Pi-compatible tool schemas and adapters over the canonical toolsets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NotRequired, Required, TypedDict, override

from pydantic import TypeAdapter
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, ToolsetTool, WrapperToolset
from pydantic_ai.workspaces import WorkspaceError
from pydantic_ai_harness import Shell
from pydantic_ai_harness.filesystem import Replacement

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from ds4py.agent_composition import ToolProfile

from .file_system import DS4FileSystem

_READ_DESCRIPTION = "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50,000 characters (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete."
_BASH_DESCRIPTION = "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds."
_EDIT_DESCRIPTION = "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes."
_WRITE_DESCRIPTION = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories."

_SCHEMAS = {
    "read": {
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
            "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "number", "description": "Maximum number of lines to read"},
        },
    },
    "bash": {
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, defaults to the configured ds4-py timeout of 30 seconds)",
            },
        },
    },
    "edit": {
        "type": "object",
        "required": ["path", "edits"],
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["oldText", "newText"],
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call.",
                        },
                        "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
                    },
                },
                "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.",
            },
        },
    },
    "write": {
        "type": "object",
        "required": ["path", "content"],
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
            "content": {"type": "string", "description": "Content to write to the file"},
        },
    },
}
_DESCRIPTIONS = {
    "read": _READ_DESCRIPTION,
    "bash": _BASH_DESCRIPTION,
    "edit": _EDIT_DESCRIPTION,
    "write": _WRITE_DESCRIPTION,
}
_NAMES = ("read", "bash", "edit", "write")
_ORIGINAL_NAMES = {"read": "read_file", "bash": "run_command", "edit": "edit_file", "write": "write_file"}


class ReadArgs(TypedDict):
    path: Required[str]
    offset: NotRequired[int | float]
    limit: NotRequired[int | float]


class BashArgs(TypedDict):
    command: Required[str]
    timeout: NotRequired[int | float]


class EditItem(TypedDict):
    oldText: str
    newText: str


class EditArgs(TypedDict):
    path: str
    edits: list[EditItem]


class WriteArgs(TypedDict):
    path: str
    content: str


def resolve_tool_profile(requested: ToolProfile, *, family: str) -> Literal["native", "pi"]:
    """Resolve auto as a development hypothesis pending the tool comparison."""
    if requested == "auto":
        return "pi" if family == "hermes" else "native"
    return requested


_VALIDATORS = {
    "read": TypeAdapter(ReadArgs).validator,
    "bash": TypeAdapter(BashArgs).validator,
    "edit": TypeAdapter(EditArgs).validator,
    "write": TypeAdapter(WriteArgs).validator,
}


def _string(args: Mapping[str, object], name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str):
        raise ModelRetry(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not float(value).is_integer():
        raise ModelRetry(f"{name} must be an integer")
    return int(value)


@dataclass
class PiToolset(WrapperToolset[object]):
    """Map four Pi calls onto the original ctx-bearing tools."""

    @override
    async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
        original = await self.wrapped.get_tools(ctx)
        tools = {}
        for name in _NAMES:
            if ctx.workspace.read_only and name in ("edit", "write"):
                continue
            original_name = _ORIGINAL_NAMES[name]
            if original_name not in original:
                continue
            definition = ToolDefinition(
                name=name,
                description=_DESCRIPTIONS[name],
                parameters_json_schema=_SCHEMAS[name],
                strict=False,
            )
            tools[name] = replace(
                original[original_name],
                toolset=self,
                tool_def=definition,
                args_validator=_VALIDATORS[name],
            )
        return tools

    def _read_args(self, args: Mapping[str, object]) -> dict[str, object]:
        offset = _integer(args.get("offset", 1), "offset")
        if offset < 0:
            raise ModelRetry("offset must be non-negative")
        limit = _integer(args.get("limit", 2000), "limit")
        if limit < 1:
            raise ModelRetry("limit must be at least 1")
        return {"path": _string(args, "path"), "offset": max(0, offset - 1), "limit": limit}

    def _bash_args(self, args: Mapping[str, object]) -> dict[str, object]:
        return {
            "command": _string(args, "command"),
            "timeout_seconds": args.get("timeout"),
        }

    async def _edit_args(self, args: Mapping[str, object], ctx: RunContext[object]) -> dict[str, object]:
        path = _string(args, "path")
        edits = args.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ModelRetry("edits must be a non-empty array")
        try:
            text = (await ctx.workspace.read_bytes(path)).decode()
        except (OSError, UnicodeDecodeError) as error:
            raise ModelRetry(f"Cannot edit {path}: {error}") from error
        except WorkspaceError as error:
            raise ToolFailed(f"Cannot edit {path}: {error}") from error
        replacements = []
        ranges = []
        for item in edits:
            if not isinstance(item, Mapping):
                raise ModelRetry("each edit must be an object")
            string_item = {}
            for key, value in item.items():
                if not isinstance(key, str):
                    raise ModelRetry("edit keys must be strings")
                string_item[key] = value
            old = _string(string_item, "oldText")
            new = _string(string_item, "newText")
            if not old:
                raise ModelRetry("oldText must not be empty")
            start = text.find(old)
            if start < 0 or text.find(old, start + 1) >= 0:
                raise ModelRetry("every oldText must match exactly once in the original file")
            ranges.append((start, start + len(old)))
            replacements.append((start, new))
        ordered = sorted(zip(ranges, replacements, strict=True))
        for index in range(1, len(ordered)):
            if ordered[index][0][0] < ordered[index - 1][0][1]:
                raise ModelRetry("oldText regions must not overlap")
        output = []
        cursor = 0
        for (start, end), (_, new) in ordered:
            output.extend((text[cursor:start], new))
            cursor = end
        output.append(text[cursor:])
        final = "".join(output)
        replacement = Replacement(old_text=text, new_text=final)
        return {"path": path, "replacements": [replacement]}

    @override
    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, object],
        ctx: RunContext[object],
        tool: ToolsetTool[object],
    ) -> object:
        original_name = _ORIGINAL_NAMES[name]
        original_tool = (await self.wrapped.get_tools(ctx))[original_name]
        if name == "read":
            converted = self._read_args(tool_args)
        elif name == "bash":
            converted = self._bash_args(tool_args)
        elif name == "edit":
            converted = await self._edit_args(tool_args, ctx)
        else:
            converted = {"path": _string(tool_args, "path"), "content": _string(tool_args, "content")}
        return await self.wrapped.call_tool(original_name, converted, ctx, original_tool)


class PiProfile(AbstractCapability[object]):
    """Expose only the four Pi operations through the canonical toolsets."""

    def __init__(self, project_dir: str | Path) -> None:
        self._filesystem = DS4FileSystem(
            tools=("read_file", "write_file", "edit_file"),
            root_dir=project_dir,
            content_hashes=True,
            max_read_lines=2000,
            max_read_chars=50_000,
        )
        self._shell = Shell(tools=("run_command",), default_timeout=30)
        self._toolset: PiToolset | None = None

    @override
    def get_toolset(self) -> AbstractToolset[object]:
        if self._toolset is None:
            self._toolset = PiToolset(CombinedToolset([self._filesystem.get_toolset(), self._shell.get_toolset()]))
        return self._toolset
