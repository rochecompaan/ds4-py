"""Complete-reply sizing for context recovery tools."""

import json
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

MAXIMUM_RECOVERY_REPLY_BYTES = 16_384
MAXIMUM_INLINE_PART_BYTES = 4_096
MAXIMUM_PAGE_LIMIT = MAXIMUM_RECOVERY_REPLY_BYTES
MAXIMUM_HISTORY_ID_CHARACTERS = 128
MAXIMUM_PART_CHARACTERS = 512


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class RecoveryTextReply(TypedDict):
    content: list[TextBlock]


def compact_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def text_reply(texts: Sequence[str]) -> RecoveryTextReply:
    return {"content": [{"type": "text", "text": text} for text in texts]}


def fits_recovery_text(texts: Sequence[str]) -> bool:
    return len(compact_json(text_reply(texts)).encode("utf-8")) <= MAXIMUM_RECOVERY_REPLY_BYTES


def recovery_text_reply(texts: Sequence[str]) -> RecoveryTextReply:
    if not fits_recovery_text(texts):
        raise ValueError("Context paging text reply exceeds the 16 KiB byte limit.")
    return text_reply(texts)


def fits_recovery_response(payload: object) -> bool:
    return fits_recovery_text([compact_json(payload)])


def recovery_response(payload: object) -> RecoveryTextReply:
    return recovery_text_reply([compact_json(payload)])


__all__ = [
    "MAXIMUM_HISTORY_ID_CHARACTERS",
    "MAXIMUM_INLINE_PART_BYTES",
    "MAXIMUM_PAGE_LIMIT",
    "MAXIMUM_PART_CHARACTERS",
    "MAXIMUM_RECOVERY_REPLY_BYTES",
    "RecoveryTextReply",
    "TextBlock",
    "compact_json",
    "fits_recovery_response",
    "fits_recovery_text",
    "recovery_response",
    "recovery_text_reply",
    "text_reply",
]
