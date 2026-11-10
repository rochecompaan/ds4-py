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
    from ds4py._native.protocol import SessionHandle

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
    original, valid = session._worker.call(
        lambda e, native: (tuple(native.tokens()), native.checkpoint_valid())
    ).result()
    try:
        plan: Plan | None = None
        for event in session._job_events(SyncJob(replacement.tokens)):
            if isinstance(event, Synced):
                plan = event.plan
    except BaseException:
        _restore_native(session, original, valid=valid)
        raise
    if plan is None:
        _restore_native(session, original, valid=valid)
        raise EngineError(f"transcript replacement ({reason}) did not complete")
    return plan


def _restore_native(session: Session, original: tuple[int, ...], *, valid: bool) -> None:
    # A failed sync may already have invalidated/rebuilt part of KV. Restore the
    # prior native prefix, not necessarily all Python tokens: a pending EOS may
    # not have been evaluated yet. Ignore cancellation and emit no usage sync.
    def restore(native: SessionHandle) -> None:
        if not original:
            native.rewind(0)
        elif len(original) == native.ctx():
            native.sync(original[:-1])
            native.eval(original[-1])
        else:
            native.sync(original)
        if not valid:
            native.invalidate()

    session._worker.call(lambda e, native: restore(native)).result()
