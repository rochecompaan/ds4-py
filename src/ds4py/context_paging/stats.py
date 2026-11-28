"""Read-only RollingWindow statistics from media-aware saved snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic_ai.messages import (
    BinaryImage,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ds4py.binding_provenance import message_content_hash
from ds4py.context_paging.estimates import TEXT_UNITS_PER_TOKEN, estimate_image_tokens
from ds4py.context_paging.text_units import utf16_length
from ds4py.message_metadata import (
    DS4_CONTEXT_PAGING,
    DS4_KEY,
    DS4_PAGING_INPUT_TOKENS,
    DS4_REQUEST_SYNC_COUNT,
)
from ds4py.message_rendering import stable_text

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord

_MAX_SAFE_INTEGER = 2**53 - 1
_KIB = 1024
_MISSING = object()


class StatisticsStore(Protocol):
    """The public async store reads used by statistics; FileStepStore satisfies it."""

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]: ...

    async def list_snapshots(
        self,
        *,
        run_id: str,
        include_interrupted: bool = False,
    ) -> list[ContinuableSnapshot]: ...


@dataclass(frozen=True, slots=True)
class RollingWindowStatisticsState:
    """The live values a report needs, copied without owner/runtime references."""

    enabled: bool
    token_budget: int
    model_window: int
    input_tokens: int | None

    def __post_init__(self) -> None:
        if _counter(self.token_budget) is None or self.token_budget == 0:
            raise ValueError("statistics token budget must be a positive safe integer")
        if _counter(self.model_window) is None or self.model_window == 0:
            raise ValueError("statistics model window must be a positive safe integer")
        if self.input_tokens is not None and _counter(self.input_tokens) is None:
            raise ValueError("statistics input tokens must be a non-negative safe integer")


@dataclass(frozen=True, slots=True)
class InputTokenReading:
    """One selected request's input and the evidence supporting its value."""

    provenance: Literal["exact", "reconstructible", "unavailable"]
    tokens: int | None

    def __post_init__(self) -> None:
        if self.provenance not in {"exact", "reconstructible", "unavailable"}:
            raise ValueError("unknown input-token provenance")
        if self.provenance == "unavailable":
            if self.tokens is not None:
                raise ValueError("unavailable input tokens must be None")
        elif _counter(self.tokens) is None:
            raise ValueError("available input tokens must be a non-negative safe integer")


@dataclass(frozen=True, slots=True)
class ContextPagingStats:
    """Complete report data; only history tokens are display estimates."""

    enabled: bool
    stored_bytes: int
    history_tokens: int
    input: InputTokenReading
    token_budget: int
    model_window: int
    cache_read_tokens: int | None
    cache_write_tokens: int | None

    @property
    def effective_budget(self) -> int:
        return min(self.token_budget, self.model_window)

    def render(self) -> str:
        budget_note = "   (inactive)"
        if self.enabled and self.input.tokens is not None:
            budget_note = f"   ({self.input.tokens / self.effective_budget * 100:.1f}% used)"
        elif self.enabled:
            budget_note = ""
        return "\n".join(
            [
                f"Context paging: {'on' if self.enabled else 'off'}",
                "",
                "SESSION — complete saved history",
                _row("Stored size", _bytes(self.stored_bytes)),
                _row("History tokens", f"~{_number(self.history_tokens)}"),
                "",
                "CONTEXT — latest model request",
                _row("Input tokens", _number(self.input.tokens)),
                _row("Paging budget", f"{_number(self.effective_budget)}{budget_note}"),
                _row("Model window", _number(self.model_window)),
                "",
                "CACHE — whole-session totals",
                _row("Tokens read", _number(self.cache_read_tokens)),
                _row("Tokens written", _number(self.cache_write_tokens)),
            ]
        )


def _counter(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_SAFE_INTEGER else None


def _sum(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return _counter(left + right)


def read_request_input(response: ModelResponse) -> InputTokenReading:
    """Read exact selected input, or reconstruct only a proven single-sync request."""
    metadata = response.metadata if isinstance(response.metadata, Mapping) else {}
    exact = metadata.get(DS4_PAGING_INPUT_TOKENS, _MISSING)
    if exact is not _MISSING:
        tokens = _counter(exact)
        return InputTokenReading("exact", tokens) if tokens is not None else InputTokenReading("unavailable", None)
    if _counter(metadata.get(DS4_REQUEST_SYNC_COUNT)) != 1:
        return InputTokenReading("unavailable", None)
    counters = [
        _counter(getattr(response.usage, name, None))
        for name in ("input_tokens", "cache_read_tokens", "cache_write_tokens")
    ]
    if any(counter is None for counter in counters):
        return InputTokenReading("unavailable", None)
    total = sum(counter for counter in counters if counter is not None)
    tokens = _counter(total)
    return (
        InputTokenReading("reconstructible", tokens) if tokens is not None else InputTokenReading("unavailable", None)
    )


def _stored_bytes(session_dir: Path) -> int:
    steps = session_dir / "steps"
    if not steps.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in steps.rglob("*")
        if path.name != "events.jsonl" and not path.is_symlink() and path.is_file()
    )


def _json_arguments(arguments: object) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _content_units(content: object) -> int:
    if isinstance(content, str):
        return utf16_length(content)
    if isinstance(content, (BinaryImage, ImageUrl)):
        return estimate_image_tokens()
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        return sum(_content_units(item) for item in content)
    return utf16_length(stable_text(content))


def _message_units(message: ModelRequest | ModelResponse) -> int:
    units = 0
    for part in message.parts:
        if isinstance(part, (UserPromptPart, ToolReturnPart, RetryPromptPart)):
            units += _content_units(part.content)
        elif isinstance(part, (TextPart, ThinkingPart)):
            units += utf16_length(part.content)
        elif isinstance(part, ToolCallPart):
            units += utf16_length(part.tool_name) + utf16_length(_json_arguments(part.args))
    return units


def _history_tokens(message: ModelRequest | ModelResponse) -> int:
    units = _message_units(message)
    return (units + TEXT_UNITS_PER_TOKEN - 1) // TEXT_UNITS_PER_TOKEN


def _usage_field(usage: object, name: str) -> int | None:
    if isinstance(usage, Mapping):
        aliases = {
            "cache_read_tokens": ("cache_read_tokens", "cacheRead"),
            "cache_write_tokens": ("cache_write_tokens", "cacheWrite"),
        }
        for alias in aliases[name]:
            if alias in usage:
                return _counter(usage.get(alias))
        return None
    return _counter(getattr(usage, name, None))


def _tool_usages(message: ModelRequest) -> list[object]:
    usages = []
    for part in message.parts:
        if isinstance(part, ToolReturnPart) and isinstance(part.metadata, Mapping):
            usage = part.metadata.get("usage")
            if usage is not None:
                usages.append(usage)
    return usages


async def collect_statistics(
    *,
    store: StatisticsStore,
    session_dir: Path,
    current: RollingWindowStatisticsState,
) -> ContextPagingStats:
    """Collect saved history through the store's public media-aware snapshot reads."""
    seen: set[tuple[str, str]] = set()
    history_tokens = 0
    cache_read: int | None = 0
    cache_write: int | None = 0
    latest_response: ModelResponse | None = None

    runs = await store.list_runs()
    for run in runs:
        snapshots = await store.list_snapshots(run_id=run.run_id, include_interrupted=True)
        for snapshot in snapshots:
            for message in snapshot.messages:
                if isinstance(message, ModelResponse):
                    latest_response = message
                metadata = message.metadata if isinstance(message.metadata, Mapping) else {}
                if metadata.get(DS4_CONTEXT_PAGING) == "notice":
                    continue
                key = metadata.get(DS4_KEY)
                identity = (key, message_content_hash(message)) if isinstance(key, str) and key else None
                if identity is not None:
                    if identity in seen:
                        continue
                    seen.add(identity)
                history_tokens += _history_tokens(message)
                usages: list[object] = [message.usage] if isinstance(message, ModelResponse) else []
                if isinstance(message, ModelRequest):
                    usages.extend(_tool_usages(message))
                for usage in usages:
                    cache_read = _sum(cache_read, _usage_field(usage, "cache_read_tokens"))
                    cache_write = _sum(cache_write, _usage_field(usage, "cache_write_tokens"))

    input_reading = (
        InputTokenReading("exact", current.input_tokens)
        if current.input_tokens is not None
        else read_request_input(latest_response)
        if latest_response is not None
        else InputTokenReading("unavailable", None)
    )
    return ContextPagingStats(
        enabled=current.enabled,
        stored_bytes=_stored_bytes(session_dir),
        history_tokens=history_tokens,
        input=input_reading,
        token_budget=current.token_budget,
        model_window=current.model_window,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def _number(value: int | None) -> str:
    return "unavailable" if value is None else f"{value:,}"


def _bytes(value: int) -> str:
    if value < _KIB:
        return f"{value} B"
    if value < _KIB * _KIB:
        return f"{value / _KIB:.2f} KiB"
    return f"{value / (_KIB * _KIB):.2f} MiB"


def _row(label: str, value: str) -> str:
    return f"  {label:<21}{value}"


__all__ = [
    "ContextPagingStats",
    "InputTokenReading",
    "RollingWindowStatisticsState",
    "StatisticsStore",
    "collect_statistics",
    "read_request_input",
]
