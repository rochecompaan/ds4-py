"""Shared replay values for C text and ds4-py JSONL traces."""

from dataclasses import dataclass

from ds4py import ThinkMode

_THINK_NAMES = {
    "none": ThinkMode.NONE,
    "high": ThinkMode.HIGH,
    "max": ThinkMode.MAX,
    "low": ThinkMode.LOW,
    "medium": ThinkMode.MEDIUM,
}


class ReplayError(ValueError):
    """A trace cannot be reconstructed without guessing at its token stream."""


@dataclass(frozen=True)
class ReplayRound:
    """One rebuilt prompt and sampled-ID sequence with traced boundary policy."""

    number: int
    prompt: tuple[int, ...]
    sampled: tuple[int, ...]
    think: int
    cached: int
    context_limited: bool
    prefix_only: bool = False


@dataclass(frozen=True)
class ReplayTrace:
    """A chat trace with at least one replayable round.

    C traces carry model and context metadata. ds4-py JSONL traces intentionally
    leave those values unset rather than pretending their session IDs are models.
    """

    ctx: int | None
    model: str | None
    rounds: tuple[ReplayRound, ...]
