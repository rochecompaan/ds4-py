"""Validate a transcript replacement and rebind it through the engine worker.

Replacement never writes history: `Session` commits the validated tokens and
segments only after the worker has restored a checkpoint or rebuilt from zero.
"""

from typing import TYPE_CHECKING

from ds4py.errors import EngineError

from .events import Synced
from .plan import Plan
from .transcript import validate_segment_coverage
from .worker import ResetJob, SyncJob

if TYPE_CHECKING:
    from .session import Session
    from .transcript import TranscriptReplacement


def replace_transcript(
    session: Session,
    replacement: TranscriptReplacement,
    *,
    reason: str,
) -> Plan:
    """Validate, then sync the replacement tokens; return the worker's plan."""
    validate_segment_coverage(replacement.segments, len(replacement.tokens))
    if not replacement.tokens:
        # Native sync rejects empty prompts; use the established reset path.
        list(session._job_events(ResetJob()))
        session._checkpoint_tokens.clear()
        return Plan("rebuild", 0, 0, fits=True, checkpoint=None)
    plan: Plan | None = None
    for event in session._job_events(SyncJob(replacement.tokens)):
        if isinstance(event, Synced):
            plan = event.plan
    if plan is None:
        raise EngineError(f"transcript replacement ({reason}) did not complete")
    return plan
