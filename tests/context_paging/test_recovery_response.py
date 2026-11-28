import json
from collections.abc import Mapping

import pytest

from ds4py.context_paging.recovery_response import (
    MAXIMUM_RECOVERY_REPLY_BYTES,
    fits_recovery_text,
    recovery_response,
    recovery_text_reply,
)


def _reply_bytes(reply: Mapping[str, object]) -> int:
    return len(json.dumps(reply, ensure_ascii=False, separators=(",", ":")).encode())


def test_complete_reply_counts_escaping_and_utf8_bytes() -> None:
    text = 'quote " slash \\ emoji 😀\n'
    reply = recovery_text_reply([text])

    assert _reply_bytes(reply) <= MAXIMUM_RECOVERY_REPLY_BYTES
    assert fits_recovery_text([text])
    assert not fits_recovery_text(["😀" * MAXIMUM_RECOVERY_REPLY_BYTES])
    assert "details" not in reply


def test_recovery_response_wraps_exact_json_without_details() -> None:
    reply = recovery_response({"text": "hello", "nested": [1, True, None]})

    content = reply["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    assert json.loads(str(block["text"])) == {"text": "hello", "nested": [1, True, None]}
    assert "details" not in reply


def test_oversized_response_raises_instead_of_truncating_utf8() -> None:
    with pytest.raises(ValueError, match="16 KiB"):
        recovery_text_reply(["x" * MAXIMUM_RECOVERY_REPLY_BYTES])
