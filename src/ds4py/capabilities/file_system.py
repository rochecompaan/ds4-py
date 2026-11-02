"""Configure harness filesystem tools for the ds4-py coding agent."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, PreparedToolset
from pydantic_ai_harness import FileSystem


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


@dataclass
class DS4FileSystem(FileSystem):
    """A filesystem capability whose schema reflects its configured read default."""

    _prepared_toolset: PreparedToolset[Any] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    # FileSystem's declared return union excludes its supported toolset wrappers.
    def get_toolset(self) -> AbstractToolset[Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self._prepared_toolset is None:
            self._prepared_toolset = PreparedToolset(
                super().get_toolset(),
                lambda _ctx, tool_defs: prepare_file_tool_definitions(
                    tool_defs, default_read_lines=self.max_read_lines
                ),
            )
        return self._prepared_toolset
