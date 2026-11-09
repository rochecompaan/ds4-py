"""Commit a compaction: prepare provenance, replace natively, then assign state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ds4py.binding_provenance import prepare_provenance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage

    from ds4py.binding import MessageBinder
    from ds4py.engine.plan import Plan
    from ds4py.engine.transcript import TranscriptReplacement


def commit_compaction(
    binder: MessageBinder,
    messages: list[ModelMessage],
    compacted_messages: Sequence[ModelMessage],
    replacement: TranscriptReplacement,
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
    plan = binder.session.replace_transcript(replacement, reason="compaction")
    binder.apply_compaction(prepared)
    messages[:] = compacted_messages
    return plan
