"""Immutable tool-parser results and internal argument values."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    """One parsed call; raw is its canonical stanza, not the enclosing block.

    DSML raw contains only its invoke stanza; GLM/Qwen include tool_call tags.
    Payload spelling is preserved, while args contains unescaped/typed values.
    """

    name: str
    args: Mapping[str, object]
    raw: str


@dataclass(frozen=True)
class ToolArgument:
    """Internal parsed argument value and its string/JSON classification."""

    name: str
    value: object
    is_string: bool


@dataclass(frozen=True)
class Snapshot:
    """Immutable parser snapshot, including the entire canonical block as bytes."""

    state: str
    raw: bytes
    calls: tuple[ToolCall, ...]
    error: str | None
    param_close_prefix: bool
    in_think: bool
    dsml_in_think: bool
