"""Immutable events emitted by the engine worker."""
from dataclasses import dataclass

from .plan import Plan


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


STOP_REASONS = (
    "stop_token", "tool_call_complete", "max_tokens", "context_reserve",
    "interrupted", "parser_error", "context_exhausted",
)
Event = Prefill | Synced | Interrupted | Checkpointed | Dropped | Token | Stopped
