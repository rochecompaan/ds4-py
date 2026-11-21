import json

import pytest

from ds4py.context_paging.public_history import (
    PublicModelTurn,
    PublicText,
    PublicToolCall,
    PublicToolResult,
    PublicUser,
)
from ds4py.context_paging.recovery_parts import (
    describe_history_part,
    history_parts,
    public_history_item,
    recovery_part_text,
    select_history_part,
    serialize_recovery_part,
)


def _turn(*, image: bool = False) -> PublicModelTurn:
    result_content: object = [{"type": "text", "text": "stored result"}]
    if image:
        result_content = [
            {"type": "text", "text": "before image"},
            {"type": "image", "data": "AAECAwQ=", "mimeType": "image/png"},
        ]
    return PublicModelTurn(
        "turn-1",
        1,
        0,
        (
            PublicText("public reply"),
            None,
            PublicToolCall("call-1", "read", {"path": "a/b"}, 2),
        ),
        (PublicToolResult("call-1", "read", result_content, False, 1, 0),),
        ("read",),
        ("a/b",),
        False,
        False,
    )


def test_public_parts_include_content_metadata_and_root_without_details() -> None:
    item = _turn()
    parts = history_parts(item)

    assert [part.part for part in parts] == [
        "/assistantMessage/content/0",
        "/assistantMessage/content/2",
        "/toolResults/0/content/0",
        "/metadata",
        "",
    ]
    root = json.loads(serialize_recovery_part(item, ""))
    assert root["id"] == "turn-1"
    assert root["assistantMessage"]["content"][1] is None
    assert "details" not in root
    assert "details" not in root["toolResults"][0]


def test_json_pointer_selects_only_public_fields_and_rejects_null_slots() -> None:
    item = _turn()

    assert select_history_part(item, "/assistantMessage/content/2/arguments/path") == "a/b"
    assert recovery_part_text(item, "/assistantMessage/content/0") == {"format": "text", "text": "public reply"}
    with pytest.raises(ValueError, match="Unknown public recovery part"):
        select_history_part(item, "/assistantMessage/content/1")
    with pytest.raises(ValueError, match="Invalid recovery part pointer"):
        select_history_part(item, "assistantMessage")
    with pytest.raises(ValueError, match="Unknown public recovery part"):
        select_history_part(item, "/private")


def test_user_root_and_plain_text_use_public_string_content() -> None:
    user = PublicUser("user-1", 0, 0, ("hello",))

    assert select_history_part(user, "/userMessage/content") == "hello"
    assert recovery_part_text(user, "/userMessage/content") == {"format": "text", "text": "hello"}


def test_image_serialization_keeps_only_metadata_and_native_read_handle() -> None:
    item = _turn(image=True)

    encoded = serialize_recovery_part(item, "/toolResults/0/content")
    parsed = json.loads(encoded)
    image = parsed[1]

    assert image == {
        "type": "image",
        "mimeType": "image/png",
        "bytes": 5,
        "read": {"historyId": "turn-1", "part": "/toolResults/0/content/1", "mode": "image"},
    }
    assert "AAECAwQ=" not in encoded
    assert '"data"' not in encoded


def test_image_fields_cannot_be_read_as_text_or_json() -> None:
    item = _turn(image=True)

    with pytest.raises(ValueError, match="image handles"):
        select_history_part(item, "/toolResults/0/content/1/data")


def test_part_descriptions_have_bounded_previews_and_exact_handles() -> None:
    item = PublicModelTurn(
        "turn-long",
        0,
        0,
        (PublicText("😀" * 200),),
        (),
        (),
        (),
        False,
        False,
    )
    part = history_parts(item)[0]

    description = describe_history_part(item, part)

    assert description["type"] == "text"
    assert description["bytes"] == 800
    assert description["read"] == {"historyId": "turn-long", "part": "/assistantMessage/content/0"}
    assert "😀" in str(description["preview"])
    assert "data" not in description


def test_public_history_lookup_does_not_return_private_or_unknown_items() -> None:
    assert public_history_item([_turn()], "turn-1") == _turn()
    with pytest.raises(ValueError, match="Unknown history ID"):
        public_history_item([_turn()], "private")
