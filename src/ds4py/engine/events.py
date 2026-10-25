"""Events emitted by engine rounds."""
from dataclasses import dataclass
from typing import Callable

from ds4py.errors import EngineError

from .plan import Plan
from .tool_syntax.model import ToolCall


@dataclass(frozen=True)
class TextDelta:
    text: str
    thinking: bool


@dataclass(frozen=True)
class Prefill:
    done: int
    total: int
    cached_prefix: int
    tps: float


@dataclass(frozen=True)
class Synced:
    pos: int
    plan: Plan
    prefilled: int


@dataclass(frozen=True)
class Interrupted:
    pos: int


@dataclass(frozen=True)
class Checkpointed:
    key: str
    tokens: int
    bytes: int


@dataclass(frozen=True)
class Dropped:
    key: str


@dataclass(frozen=True)
class Token:
    id: int
    text_bytes: bytes


@dataclass(frozen=True)
class Stopped:
    reason: str


class ReserveReached:
    """A one-shot decision that keeps a chat assistant round open."""

    def __init__(self, needed: int, *, on_resume: Callable[[int], None],
                 on_fail: Callable[[], None]):
        self.needed = needed
        self._on_resume = on_resume
        self._on_fail = on_fail
        self._decision: str | None = None

    def _claim(self, decision: str) -> None:
        if self._decision is not None:
            raise EngineError("reserve pause already has a decision")
        self._decision = decision

    def resume(self, reserve: int) -> None:
        if isinstance(reserve, bool) or not isinstance(reserve, int):
            raise TypeError("reserve must be an integer")
        if reserve < 0:
            raise ValueError("reserve must be non-negative")
        self._claim("resume")
        self._on_resume(reserve)

    def fail(self) -> None:
        self._claim("fail")
        self._on_fail()


@dataclass(frozen=True)
class Thinking:
    on: bool


@dataclass(frozen=True)
class ToolSyntax:
    state: str


@dataclass(frozen=True)
class ToolCalls:
    calls: tuple[ToolCall, ...]


STOP_REASONS = (
    "stop_token", "tool_call_complete", "max_tokens", "context_reserve",
    "interrupted", "parser_error", "context_exhausted",
)
Event = Prefill | Synced | Interrupted | Checkpointed | Dropped | Token | Stopped | ReserveReached | TextDelta | Thinking | ToolSyntax | ToolCalls
