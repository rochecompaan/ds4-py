"""Configure harness filesystem tools for the ds4-py coding agent."""

import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, override

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.toolsets import AbstractToolset, PreparedToolset
from pydantic_ai.workspaces import WorkspaceError

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.tools import RunContext, ToolDefinition
    from pydantic_ai.toolsets import ToolsetTool
from pydantic_ai_harness import FileSystem

_MAX_LINE_COMPARISONS = 10_000
_MAX_ERROR_LINE_CHARS = 200
_READ_HASH = re.compile(r"\[(?P<path>.+) \| \d+ lines \| hash:(?P<hash>[0-9a-f]{12})\]")


def prepare_file_tool_definitions(tool_defs: list[ToolDefinition], *, default_read_lines: int) -> list[ToolDefinition]:
    """Describe the configured read default without changing tool execution."""
    prepared = []
    for tool_def in tool_defs:
        if tool_def.name != "read_file":
            prepared.append(tool_def)
            continue
        schema = deepcopy(tool_def.parameters_json_schema)
        schema["properties"]["limit"]["description"] = (
            f"Maximum number of lines to return (default: {default_read_lines})."
        )
        prepared.append(replace(tool_def, parameters_json_schema=schema))
    return prepared


def _nearest_line(text: str, old_text: str) -> tuple[int, str] | None:
    old_lines = [line[:_MAX_ERROR_LINE_CHARS] for line in old_text.splitlines() if line.strip()]
    best: tuple[int, str] | None = None
    best_score = -1.0
    comparisons = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        for old_line in old_lines:
            if comparisons >= _MAX_LINE_COMPARISONS:
                return best
            comparisons += 1
            score = SequenceMatcher(None, old_line, line[:_MAX_ERROR_LINE_CHARS], autojunk=False).ratio()
            if score > best_score:
                best_score = score
                best = (number, line[:_MAX_ERROR_LINE_CHARS])
    return best


def _last_read_hash(messages: list[ModelMessage], path: str) -> str | None:
    for message in reversed(messages):
        for part in reversed(message.parts):
            if (
                not isinstance(part, ToolReturnPart)
                or part.tool_name != "read_file"
                or not isinstance(part.content, str)
            ):
                continue
            match = _READ_HASH.search(part.content)
            if match is not None:
                matched_path = match["path"]
                matched_hash = match["hash"]
                if isinstance(matched_path, str) and isinstance(matched_hash, str) and matched_path == path:
                    return matched_hash
    return None


async def _missing_old_text_message(
    error: ModelRetry,
    tool_args: dict[str, object],
    ctx: RunContext[object],
) -> str | None:
    original = str(error)
    path = tool_args.get("path")
    old_text = tool_args.get("old_text")
    if (
        not original.startswith("old_text not found in ")
        or not original.endswith(". No changes were written.")
        or not isinstance(path, str)
        or not isinstance(old_text, str)
    ):
        return None
    try:
        raw = await ctx.workspace.read_bytes(path)
        text = raw.decode("utf-8")
    except OSError, UnicodeDecodeError, WorkspaceError:
        return None
    nearest = _nearest_line(text, old_text)
    if nearest is None:
        return None

    number, line = nearest
    parts = [original, f"Nearest match: line {number}:\n{line}"]
    if (last_hash := _last_read_hash(ctx.messages, path)) is not None:
        current_hash = hashlib.sha256(raw).hexdigest()[:12]
        if current_hash == last_hash:
            parts.append(
                "The file has not changed since the model last read it; use the nearest line to correct old_text."
            )
        else:
            parts.append("The file has changed since the model last read it; re-read it before retrying.")
    return "\n".join(parts)


class _DS4FileToolset(PreparedToolset[object]):
    @override
    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, object],
        ctx: RunContext[object],
        tool: ToolsetTool[object],
    ) -> object:
        try:
            return await super().call_tool(name, tool_args, ctx, tool)
        except ModelRetry as error:
            if name != "edit_file" or (message := await _missing_old_text_message(error, tool_args, ctx)) is None:
                raise
            raise ModelRetry(message) from error


@dataclass
class DS4FileSystem(FileSystem):
    """A filesystem capability whose schema reflects its configured read default."""

    _prepared_toolset: PreparedToolset[object] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @override
    def get_toolset(self) -> AbstractToolset[object]:  # ty: ignore[invalid-method-override] - PreparedToolset is supported by FileSystem but missing from its return annotation.
        if self._prepared_toolset is None:
            self._prepared_toolset = _DS4FileToolset(
                super().get_toolset(),
                lambda _ctx, tool_defs: prepare_file_tool_definitions(
                    tool_defs, default_read_lines=self.max_read_lines
                ),
            )
        return self._prepared_toolset
