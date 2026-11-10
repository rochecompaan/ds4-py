"""Validate a transcript replacement and rebind it through the engine worker.

Replacement never writes history: `SessionHistory` commits the validated tokens
and segments only after the worker has restored a checkpoint or rebuilt from
zero.
"""

from typing import TYPE_CHECKING

from ds4py.errors import EngineError

from .events import Synced
from .native_restore import restore_native_tokens
from .plan import Plan
from .transcript import validate_segment_coverage
from .worker import ResetJob, SyncJob

if TYPE_CHECKING:
    from .history import SessionHistory
    from .session import Session
    from .transcript import TranscriptReplacement


def replace_transcript(
    session: Session,
    history: SessionHistory,
    replacement: TranscriptReplacement,
    *,
    reason: str,
) -> Plan:
    """Validate, then sync the replacement tokens; return the worker's plan."""
    validate_segment_coverage(replacement.segments, len(replacement.tokens))
    original, valid = session._worker.call(
        lambda e, native: (tuple(native.tokens()), native.checkpoint_valid())
    ).result()
    if not replacement.tokens:
        # Native sync rejects empty prompts; reset also disposes checkpoints.
        try:
            list(session._job_events(ResetJob()))
        except BaseException as error:
            _restore_after_cleanup_failure(session, original, valid=valid, error=error)
            raise
        history.clear_checkpoints()
        return Plan("rebuild", 0, 0, fits=True, checkpoint=None)
    try:
        plan = _sync_replacement(session, replacement, reason=reason)
    except BaseException:
        _restore_native(session, original, valid=valid)
        raise
    try:
        for key, length in history.checkpoint_tokens.items():
            if length > len(replacement.tokens):
                history.dispose_checkpoint(key)
                history.forget_checkpoint(key)
    except BaseException as error:
        _restore_after_cleanup_failure(session, original, valid=valid, error=error)
        raise
    return plan


def _restore_after_cleanup_failure(
    session: Session, original: tuple[int, ...], *, valid: bool, error: BaseException
) -> None:
    _restore_native(session, original, valid=valid)
    # Disposal may already have removed checkpoints or failed to release a
    # native resource. Restored tokens alone do not make that worker reusable.
    session.poison(error)


def _sync_replacement(session: Session, replacement: TranscriptReplacement, *, reason: str) -> Plan:
    plan: Plan | None = None
    for event in session._job_events(SyncJob(replacement.tokens)):
        if isinstance(event, Synced):
            plan = event.plan
    if plan is None:
        raise EngineError(f"transcript replacement ({reason}) did not complete")
    return plan


def _restore_native(session: Session, original: tuple[int, ...], *, valid: bool) -> None:
    # A failed sync may already have invalidated/rebuilt part of KV. Restore the
    # prior native prefix, not necessarily all Python tokens: a pending EOS may
    # not have been evaluated yet. Ignore cancellation and emit no usage sync.
    session._worker.call(
        lambda e, native: restore_native_tokens(native, original, valid=valid, poison=session._worker._poison)
    ).result()
