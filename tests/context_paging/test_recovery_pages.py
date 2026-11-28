import json

import pytest

from ds4py.context_paging.public_history import PublicModelTurn, PublicText, PublicToolCall, PublicToolResult
from ds4py.context_paging.recovery_parts import read_context_output
from ds4py.context_paging.recovery_response import MAXIMUM_RECOVERY_REPLY_BYTES, fits_recovery_text
from ds4py.context_paging.text_units import utf16_length, utf16_slice


def _turn(text: str = "A😀B") -> PublicModelTurn:
    return PublicModelTurn(
        "turn-1",
        0,
        0,
        (PublicText(text), PublicToolCall("call-1", "read", {}, 1)),
        (PublicToolResult("call-1", "read", "result text", False, 1, 0),),
        ("read",),
        (),
        False,
        False,
    )


def test_unique_legacy_defaults_and_explicit_handles_resolve_same_part() -> None:
    items = [_turn()]
    implicit = read_context_output(items, {"historyId": "turn-1", "source": "assistant"})
    explicit = read_context_output(items, {"historyId": "turn-1", "part": "/assistantMessage/content/0"})
    result = read_context_output(items, {"historyId": "turn-1", "source": "toolResult"})

    assert implicit.header["part"] == explicit.header["part"]
    assert implicit.text == "A😀B"
    assert json.loads(result.text)["content"] == "result text"
    assert result.header["part"] == "/toolResults/0"
    assert result.header["format"] == "json"


def test_ambiguous_assistant_output_lists_public_choices() -> None:
    turn = PublicModelTurn("turn-1", 0, 0, (PublicText("one"), PublicText("two")), (), (), (), False, False)

    with pytest.raises(ValueError, match="contentIndex") as error:
        read_context_output([turn], {"historyId": "turn-1", "source": "assistant"})
    assert "0 (text)" in str(error.value)
    assert "1 (text)" in str(error.value)


def test_astral_pages_use_utf16_offsets_and_never_split_pairs() -> None:
    first = read_context_output(
        [_turn()],
        {"historyId": "turn-1", "source": "assistant", "offset": 1, "limit": 2},
    )

    assert first.text == "😀"
    assert first.header["offset"] == 1
    assert first.header["nextOffset"] == 3
    assert first.header["totalCharacters"] == 4
    assert first.header["next"] == {
        "historyId": "turn-1",
        "part": "/assistantMessage/content/0",
        "offset": 3,
        "limit": 2,
    }
    following = read_context_output([_turn()], dict(first.header["next"]))
    assert following.text == "B"
    assert following.header["nextOffset"] is None

    with pytest.raises(ValueError, match="splits a Unicode character"):
        read_context_output([_turn()], {"historyId": "turn-1", "source": "assistant", "offset": 2, "limit": 2})


def test_large_pages_shrink_to_complete_utf8_reply_and_continue_exactly() -> None:
    text = ("ascii 😀\n" * 3000) + "tail"
    turn = _turn(text)
    first = read_context_output([turn], {"historyId": "turn-1", "source": "assistant"})

    blocks = [json.dumps(first.header, separators=(",", ":")), first.text]
    assert fits_recovery_text(blocks)
    encoded = json.dumps(
        {"content": [{"type": "text", "text": block} for block in blocks]},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert len(encoded) <= MAXIMUM_RECOVERY_REPLY_BYTES
    assert first.header["nextOffset"] is not None
    assert first.header["totalCharacters"] == utf16_length(text)

    remaining = text.encode("utf-16-le")[first.header["nextOffset"] * 2 :].decode("utf-16-le")
    second = read_context_output([turn], dict(first.header["next"]))
    assert second.text == utf16_slice(remaining, 0, utf16_length(second.text))


def test_explicit_part_cannot_be_combined_with_legacy_selectors() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        read_context_output(
            [_turn()],
            {
                "historyId": "turn-1",
                "source": "assistant",
                "part": "/assistantMessage/content/0",
            },
        )


def test_non_text_modes_and_out_of_range_offsets_fail_closed() -> None:
    with pytest.raises(ValueError, match="text mode"):
        read_context_output([_turn()], {"historyId": "turn-1", "source": "assistant", "mode": "image"})
    with pytest.raises(ValueError, match="exceeds"):
        read_context_output([_turn()], {"historyId": "turn-1", "source": "assistant", "offset": 5})
