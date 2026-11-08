"""Dependency-neutral benchmark configuration and report models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ds4py._json import is_json_object


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not is_json_object(value):
        raise ValueError(f"{name} must be an object")
    return value


def _field(data: Mapping[str, object], name: str) -> object:
    try:
        return data[name]
    except KeyError as error:
        raise ValueError(f"report record is missing {name!r}") from error


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")  # noqa: TRY004 - Report validation uses ValueError.
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")  # noqa: TRY004 - Report validation uses ValueError.
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004 - Report validation uses ValueError.
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")  # noqa: TRY004 - Report validation uses ValueError.
    return float(value)


def _optional_number(value: object, name: str) -> float | None:
    return None if value is None else _number(value, name)


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")  # noqa: TRY004 - Report validation uses ValueError.
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
    def from_dict(cls, value: object) -> ToolAccounting:
        data = _mapping(value, "tool_accounting")
        return cls(
            parsed=_integer(_field(data, "parsed"), "parsed"),
            executed=_integer(_field(data, "executed"), "executed"),
            refused=_integer(_field(data, "refused"), "refused"),
            retried=_integer(_field(data, "retried"), "retried"),
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
    terminal_reasons: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: object) -> EventSummary:
        data = _mapping(value, "summary")
        accounting = _field(data, "tool_accounting")
        return cls(
            wall_seconds=_number(_field(data, "wall_seconds"), "wall_seconds"),
            progress_events=_integer(_field(data, "progress_events"), "progress_events"),
            tokens=_integer(_field(data, "tokens"), "tokens"),
            tool_calls=_integer(_field(data, "tool_calls"), "tool_calls"),
            avg_prefill_reuse=_optional_number(_field(data, "avg_prefill_reuse"), "avg_prefill_reuse"),
            non_extend_rounds=_integer(_field(data, "non_extend_rounds"), "non_extend_rounds"),
            invalid_non_extend_rounds=_integer(data.get("invalid_non_extend_rounds", 0), "invalid_non_extend_rounds"),
            allowed_non_extend_rounds=_integer(data.get("allowed_non_extend_rounds", 0), "allowed_non_extend_rounds"),
            non_extension_reasons=_strings(
                data.get("non_extension_reasons", ()),
                "non_extension_reasons",
            ),
            parser_errors=_integer(data.get("parser_errors", 0), "parser_errors"),
            tool_accounting=(None if accounting is None else ToolAccounting.from_dict(accounting)),
            reuse_ratios=tuple(
                _number(item, "reuse_ratios")
                for item in _sequence(
                    data.get("reuse_ratios", ()),
                    "reuse_ratios",
                )
            ),
            terminal_reasons=_strings(_field(data, "terminal_reasons"), "terminal_reasons"),
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
    def from_dict(cls, value: object) -> TaskRunResult:
        data = _mapping(value, "task")
        workspace = data.get("workspace")
        return cls(
            task_id=_text(_field(data, "task_id"), "task_id"),
            agent=_text(_field(data, "agent"), "agent"),
            passed=_boolean(_field(data, "passed"), "passed"),
            attempts=_integer(_field(data, "attempts"), "attempts"),
            summary=EventSummary.from_dict(_field(data, "summary")),
            grader_outputs=_strings(data.get("grader_outputs", ()), "grader_outputs"),
            attempt_save_dirs=_paths(
                data.get("attempt_save_dirs", ()),
                "attempt_save_dirs",
            ),
            attempt_trace_paths=_paths(
                data.get("attempt_trace_paths", ()),
                "attempt_trace_paths",
            ),
            errors=_strings(data.get("errors", ()), "errors"),
            workspace=(None if workspace is None else Path(_text(workspace, "workspace"))),
        )

    @property
    def wall_seconds(self) -> float:
        return self.summary.wall_seconds

    @property
    def progress_events(self) -> int:
        return self.summary.progress_events

    @property
    def tokens(self) -> int:
        return self.summary.tokens

    @property
    def tool_calls(self) -> int:
        return self.summary.tool_calls

    @property
    def avg_prefill_reuse(self) -> float | None:
        return self.summary.avg_prefill_reuse

    @property
    def non_extend_rounds(self) -> int:
        return self.summary.non_extend_rounds

    @property
    def invalid_non_extend_rounds(self) -> int:
        return self.summary.invalid_non_extend_rounds

    @property
    def allowed_non_extend_rounds(self) -> int:
        return self.summary.allowed_non_extend_rounds

    @property
    def non_extension_reasons(self) -> tuple[str, ...]:
        return self.summary.non_extension_reasons

    @property
    def parser_errors(self) -> int:
        return self.summary.parser_errors

    @property
    def tool_accounting(self) -> ToolAccounting | None:
        return self.summary.tool_accounting

    @property
    def reuse_ratios(self) -> tuple[float, ...]:
        return self.summary.reuse_ratios

    @property
    def terminal_reasons(self) -> tuple[str, ...]:
        return self.summary.terminal_reasons

    @property
    def save_dir(self) -> Path | None:
        return self.attempt_save_dirs[-1] if self.attempt_save_dirs else None

    @property
    def trace_path(self) -> Path | None:
        return self.attempt_trace_paths[-1] if self.attempt_trace_paths else None
