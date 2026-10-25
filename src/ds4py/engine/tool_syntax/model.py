"""Immutable tool-parser results and internal argument values."""
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Mapping[str, object]


@dataclass(frozen=True)
class ToolArgument:
    name: str
    value: object
    is_string: bool


@dataclass(frozen=True)
class Snapshot:
    state: str
    raw: bytes
    calls: tuple[ToolCall, ...]
    error: str | None
    param_close_prefix: bool
    in_think: bool
    dsml_in_think: bool
