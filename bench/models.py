"""Dependency-neutral benchmark configuration and report models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _field(data: Mapping[str, Any], name: str) -> Any:
    try:
        return data[name]
    except KeyError as error:
        raise ValueError(f"report record is missing {name!r}") from error


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name) for item in _sequence(value, name))


def _paths(value: object, name: str) -> tuple[Path, ...]:
    return tuple(Path(_text(item, name)) for item in _sequence(value, name))


@dataclass(frozen=True)
class AgentConfig:
    agent: str
    executable: tuple[str, ...]
    model: str
    backend: str
    ctx: int
    temperature: float
    seed: int
    tokens: int
    request_limit: int = 100

    def __post_init__(self) -> None:
        if self.agent not in {"c", "ds4py"}:
            raise ValueError("agent must be 'c' or 'ds4py'")
        object.__setattr__(self, "executable", tuple(str(item) for item in self.executable))


@dataclass(frozen=True)
class ToolAccounting:
    parsed: int = 0
    executed: int = 0
    refused: int = 0
    retried: int = 0

    @classmethod
    def from_dict(cls, value: object) -> "ToolAccounting":
        data = _mapping(value, "tool_accounting")
        return cls(
            parsed=_field(data, "parsed"),
            executed=_field(data, "executed"),
            refused=_field(data, "refused"),
            retried=_field(data, "retried"),
        )

    @property
    def accounted(self) -> int:
        return self.executed + self.refused + self.retried

    @property
    def complete(self) -> bool:
        return self.accounted == self.parsed


@dataclass(frozen=True)
class EventSummary:
    wall_seconds: float
    progress_events: int
    tokens: int
    tool_calls: int
    avg_prefill_reuse: float | None
    non_extend_rounds: int
    invalid_non_extend_rounds: int = 0
    allowed_non_extend_rounds: int = 0
    non_extension_reasons: tuple[str, ...] = ()
    parser_errors: int = 0
    tool_accounting: ToolAccounting | None = ToolAccounting()
    reuse_ratios: tuple[float, ...] = ()

    @classmethod
    def from_dict(cls, value: object) -> "EventSummary":
        data = _mapping(value, "summary")
        accounting = _field(data, "tool_accounting")
        return cls(
            wall_seconds=_field(data, "wall_seconds"),
            progress_events=_field(data, "progress_events"),
            tokens=_field(data, "tokens"),
            tool_calls=_field(data, "tool_calls"),
            avg_prefill_reuse=_field(data, "avg_prefill_reuse"),
            non_extend_rounds=_field(data, "non_extend_rounds"),
            invalid_non_extend_rounds=data.get("invalid_non_extend_rounds", 0),
            allowed_non_extend_rounds=data.get("allowed_non_extend_rounds", 0),
            non_extension_reasons=_strings(
                data.get("non_extension_reasons", ()), "non_extension_reasons",
            ),
            parser_errors=data.get("parser_errors", 0),
            tool_accounting=(
                None if accounting is None else ToolAccounting.from_dict(accounting)
            ),
            reuse_ratios=tuple(
                float(item) for item in _sequence(
                    data.get("reuse_ratios", ()), "reuse_ratios",
                )
            ),
        )


_SUMMARY_FIELDS = frozenset(EventSummary.__dataclass_fields__)


@dataclass(frozen=True)
class TaskRunResult:
    task_id: str
    agent: str
    passed: bool
    attempts: int
    summary: EventSummary
    grader_outputs: tuple[str, ...] = ()
    attempt_save_dirs: tuple[Path, ...] = ()
    attempt_trace_paths: tuple[Path, ...] = ()
    errors: tuple[str, ...] = ()
    workspace: Path | None = None

    def __post_init__(self) -> None:
        if self.attempts not in {1, 2}:
            raise ValueError("attempts must be one or two")
        for name in ("progress_events", "tokens", "tool_calls", "non_extend_rounds"):
            value = getattr(self.summary, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not math.isfinite(self.summary.wall_seconds) or self.summary.wall_seconds < 0:
            raise ValueError("wall_seconds must be finite and nonnegative")
        reuse = self.summary.avg_prefill_reuse
        if reuse is not None and not 0 <= reuse <= 1:
            raise ValueError("avg_prefill_reuse must be in [0,1]")

    @classmethod
    def from_dict(cls, value: object) -> "TaskRunResult":
        data = _mapping(value, "task")
        workspace = data.get("workspace")
        return cls(
            task_id=_text(_field(data, "task_id"), "task_id"),
            agent=_text(_field(data, "agent"), "agent"),
            passed=_boolean(_field(data, "passed"), "passed"),
            attempts=_field(data, "attempts"),
            summary=EventSummary.from_dict(_field(data, "summary")),
            grader_outputs=_strings(data.get("grader_outputs", ()), "grader_outputs"),
            attempt_save_dirs=_paths(
                data.get("attempt_save_dirs", ()), "attempt_save_dirs",
            ),
            attempt_trace_paths=_paths(
                data.get("attempt_trace_paths", ()), "attempt_trace_paths",
            ),
            errors=_strings(data.get("errors", ()), "errors"),
            workspace=(
                None if workspace is None else Path(_text(workspace, "workspace"))
            ),
        )

    def __getattr__(self, name: str) -> Any:
        if name in _SUMMARY_FIELDS:
            return getattr(self.summary, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    @property
    def save_dir(self) -> Path | None:
        return self.attempt_save_dirs[-1] if self.attempt_save_dirs else None

    @property
    def trace_path(self) -> Path | None:
        return self.attempt_trace_paths[-1] if self.attempt_trace_paths else None
