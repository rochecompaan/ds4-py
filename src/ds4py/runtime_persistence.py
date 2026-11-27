"""Pure construction of save snapshots and records from explicit values."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from pydantic_ai.messages import BinaryContent, BinaryImage, ImageUrl, ToolReturnPart, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot

from ds4py.errors import EngineError
from ds4py.store import SaveRecord, token_digest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage

    from ds4py.store import SegmentProvenance


def _contains_image(content: object) -> bool:
    if isinstance(content, (BinaryContent, BinaryImage)):
        return content.media_type.startswith("image/")
    if isinstance(content, ImageUrl):
        return True
    if isinstance(content, dict):
        return any(_contains_image(value) for value in content.values())
    if isinstance(content, (list, tuple)):
        return any(_contains_image(value) for value in content)
    return False


def validate_save_messages(*message_groups: Sequence[ModelMessage]) -> None:
    """Keep the existing M5 image-save refusal ahead of every save artifact."""
    for messages in message_groups:
        for message in messages:
            for part in message.parts:
                if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                    raise EngineError("image input requires M5 vision support")
                if isinstance(part, ToolReturnPart) and _contains_image(part.content):
                    raise EngineError("image input requires M5 vision support")


def build_save_snapshot(
    *,
    run_id: str,
    session_id: str,
    messages: Sequence[ModelMessage],
) -> ContinuableSnapshot:
    """Build the canonical harness snapshot without retaining caller message objects."""
    return ContinuableSnapshot(
        run_id=run_id,
        step_index=0,
        messages=deepcopy(list(messages)),
        conversation_id=session_id,
        agent_name="ds4py",
    )


def build_save_record(  # noqa: PLR0913 - The pure builder mirrors the save record's explicit fields.
    *,
    snapshot: ContinuableSnapshot,
    session_id: str,
    model_id: int,
    quant_bits: int,
    ctx: int,
    title: str,
    created_at: str,
    last_used: str,
    payload_bytes: int,
    tokens: Sequence[int],
    segments: Sequence[SegmentProvenance],
    request_view: Sequence[ModelMessage] | None = None,
) -> SaveRecord:
    """Build one save record, computing only token count and digest."""
    return SaveRecord(
        session_id=session_id,
        conversation_id=session_id,
        run_id=snapshot.run_id,
        step_index=snapshot.step_index,
        model_id=model_id,
        quant_bits=quant_bits,
        ctx=ctx,
        title=title,
        created_at=created_at,
        last_used=last_used,
        payload_bytes=payload_bytes,
        tokens=len(tokens),
        tokens_sha256=token_digest(tokens),
        segments=tuple(segments),
        request_view=None if request_view is None else tuple(request_view),
    )
