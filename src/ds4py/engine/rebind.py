"""Validate a transcript replacement and rebind it through the engine worker.

Replacement never writes history: `Session` commits the validated tokens and
segments only after the worker has restored a checkpoint or rebuilt from zero.
"""

from typing import TYPE_CHECKING

from ds4py.errors import EngineError

from .events import Synced
from .worker import SyncJob

if TYPE_CHECKING:
    from .plan import Plan
    from .session import Session
    from .transcript import TranscriptReplacement


def validate(replacement: TranscriptReplacement) -> None:
    """Require segments to cover the tokens contiguously from token zero."""
    expected = 0
    for segment in replacement.segments:
        if segment.start != expected or segment.end <= segment.start:
            raise ValueError("replacement segments must cover the tokens without gaps from token zero")
        expected = segment.end
    if expected != len(replacement.tokens):
        raise ValueError(
            f"replacement segments cover {expected} tokens, not the {len(replacement.tokens)} tokens present"
        )


def replace_transcript(
    session: Session,
    replacement: TranscriptReplacement,
    *,
    reason: str,
) -> Plan:
    """Validate, then sync the replacement tokens; return the worker's plan."""
    validate(replacement)
    plan: Plan | None = None
    for event in session._job_events(SyncJob(replacement.tokens)):
        if isinstance(event, Synced):
            plan = event.plan
    if plan is None:
        raise EngineError(f"transcript replacement ({reason}) did not complete")
    return plan
