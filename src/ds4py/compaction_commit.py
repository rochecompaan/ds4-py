"""Commit a compaction: prepare provenance, replace natively, then assign state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ds4py.async_session import run_session_operation
from ds4py.binding_provenance import prepare_provenance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage

    from ds4py.binding import MessageBinder
    from ds4py.engine.plan import Plan
    from ds4py.engine.reserve import ReserveOperations
    from ds4py.engine.transcript import TranscriptReplacement


async def commit_compaction(
    binder: MessageBinder,
    messages: list[ModelMessage],
    compacted_messages: Sequence[ModelMessage],
    replacement: TranscriptReplacement,
    *,
    operations: ReserveOperations | None = None,
) -> Plan:
    """Commit one compaction so a native failure needs no rollback.

    Preparation is pure and deep-copies the records it will install. Native
    replacement is the only fallible mutation: when it raises, neither the
    binder's records nor ``messages`` have changed. The remaining steps assign
    the already prepared values.
    """
    prepared = prepare_provenance(
        compacted_messages,
        replacement.segments,
        loaded=len(replacement.tokens),
    )
    committed: list[Plan] = []
    # Validate the strict owner on its loop, not inside an inherited executor.
    authority = operations.issue_authority("replacement") if operations is not None else None

    def replace() -> Plan:
        plan = (
            binder.session.replace_transcript(replacement, reason="compaction", authority=authority)
            if authority is not None
            else binder.session.replace_transcript(replacement, reason="compaction")
        )
        committed.append(plan)
        return plan

    try:
        return await run_session_operation(binder.session, replace)
    finally:
        # run_session_operation drains before raising cancellation. A successful
        # native commit must publish its matching prepared state even then; a
        # failed replacement publishes nothing. There is no checkpoint here.
        if committed:
            binder.apply_compaction(prepared)
            messages[:] = compacted_messages
