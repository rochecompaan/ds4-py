import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import SystemPromptPart, UserPromptPart
from test_history import _messages

from ds4py.context_paging.history import project_history
from ds4py.context_paging.notice import (
    ENTRY_CHARACTERS,
    MINIMUM_CLIPPED_CHARACTERS,
    EvictedConversation,
    PublicModelTurn,
    PublicText,
    PublicUser,
    _clipped_entry,
    _reply_entry,
    build_paging_notice,
    conversation_character_limit,
    paging_notice_key,
)
from ds4py.context_paging.public_history import project_public_history

FIXTURES = json.loads((Path(__file__).parents[1] / "fixtures/context_paging/notice.json").read_text())["fixtures"]


def _user(turn: int, text: str | None = None) -> PublicUser:
    return PublicUser(f"u{turn}", turn * 2, turn * 2, (text or f"request {turn}",))


def _reply(turn: int, text: str | None = None) -> PublicModelTurn:
    return PublicModelTurn(
        f"r{turn}",
        turn * 2 + 1,
        turn * 2 + 1,
        (PublicText(text or f"reply {turn}"),),
        (),
        (),
        (),
        False,
        False,
    )


def _conversation(turns: int) -> EvictedConversation:
    conversation = EvictedConversation()
    for turn in range(1, turns + 1):
        conversation.add_turn([_user(turn), _reply(turn)])
    return conversation


def _fixture_notice_value(fixture: dict[str, Any]) -> dict[str, object]:
    conversation = EvictedConversation()
    turn: list[Any] = []
    for item in project_public_history(project_history(_messages(fixture))):
        if isinstance(item, PublicUser) and turn:
            conversation.add_turn(turn)
            turn = []
        turn.append(item)
    if turn:
        conversation.add_turn(turn)
    recent = next(
        entry["id"] for entry in reversed(fixture["input"]["entries"]) if entry["message"]["role"] == "assistant"
    )
    options = {"max_characters": fixture["input"]["maxCharacters"]} if "maxCharacters" in fixture["input"] else {}
    notice = build_paging_notice(
        fixture["input"]["effectiveBudgetTokens"],
        recent,
        conversation=conversation,
        **options,
    )
    part = notice.as_message().parts[0]
    assert isinstance(part, (SystemPromptPart, UserPromptPart))
    return {
        "role": "user" if isinstance(part, UserPromptPart) else "system",
        "content": part.content,
        "timestamp": int(part.timestamp.timestamp() * 1000),
    }


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture["id"])
def test_generated_notice_contract(fixture: dict[str, Any]) -> None:
    assert _fixture_notice_value(fixture) == fixture["expected"]["value"]


def test_only_newest_three_evicted_turn_replies_are_retained() -> None:
    notice = build_paging_notice(100_000, "r4", conversation=_conversation(4))

    for expected in ("request 1", "request 2", "request 3", "request 4", "reply 2", "reply 3", "reply 4"):
        assert expected in notice.content
    assert "reply 1" not in notice.content
    assert 'Recent evicted historyId: "r4".' in notice.content


def test_newest_first_selection_is_displayed_oldest_first() -> None:
    conversation = _conversation(3)
    full = build_paging_notice(100_000, "r3", conversation=conversation).content
    limited = build_paging_notice(100_000, "r3", conversation=conversation, max_characters=1050).content

    assert "reply 3" in limited
    assert "request 3" in limited
    assert "request 1" not in limited
    assert limited.index("request 3") < limited.index("reply 3")
    assert "earlier user message" in limited
    assert "not shown" in limited
    assert len(limited) <= 1050
    assert len(full) < 1050


def test_only_final_public_reply_and_full_user_request_contribute() -> None:
    conversation = EvictedConversation()
    first = _reply(1, "intermediate reply")
    final = PublicModelTurn(
        "r1-final",
        2,
        2,
        (None, PublicText("first line"), PublicText("final line")),
        (),
        (),
        (),
        False,
        False,
    )
    conversation.add_turn([_user(1, "full request"), first, final])

    notice = build_paging_notice(100_000, "r1-final", conversation=conversation).content

    assert "full request" in notice
    assert "intermediate reply" not in notice
    assert "first line\nfinal line" in notice
    assert "read_context_output({" not in notice


def test_clipped_entries_have_exact_original_part_handles() -> None:
    conversation = EvictedConversation()
    conversation.add_turn([_user(1, "x" * 7000), _reply(1)])

    notice = build_paging_notice(100_000, "r1", conversation=conversation).content

    assert "Clipped after 6,000 of 7,000 characters" in notice
    handle = {"historyId": "u1", "part": "/userMessage/content"}
    assert f"read_context_output({json.dumps(handle, separators=(',', ':'))})" in notice


def test_reply_handles_use_original_assistant_indices() -> None:
    single = _reply_entry(
        PublicModelTurn("r", 0, 0, (None, PublicText("one"), PublicText("two")), (), (), (), False, False)
    )
    assert single is not None
    assert single.part == "/assistantMessage/content"
    one = _reply_entry(PublicModelTurn("r", 0, 0, (None, None, PublicText("one")), (), (), (), False, False))
    assert one is not None
    assert one.part == "/assistantMessage/content/2"


def test_entry_clip_limit_and_minimum_useful_clip() -> None:
    entry = _reply_entry(_reply(1, "x" * 7000))
    assert entry is not None

    full = _clipped_entry(entry, 10_000)
    too_small = _clipped_entry(entry, 1)

    assert full is not None
    assert "Clipped after 6,000" in full
    assert too_small is None
    assert ENTRY_CHARACTERS == 6000
    assert MINIMUM_CLIPPED_CHARACTERS == 500


def test_conversation_cap_uses_effective_budget_and_remaining_room() -> None:
    assert conversation_character_limit(200_000) == 32_000
    assert conversation_character_limit(10_000) == 4_000

    conversation = _conversation(1)
    small = build_paging_notice(10_000, "r1", conversation=conversation, max_characters=500).content
    large = build_paging_notice(200_000, "r1", conversation=conversation).content

    assert len(small) <= 500 + 100
    assert len(large) > len(small)


def test_notice_identity_and_message_metadata_are_stable() -> None:
    first = build_paging_notice(1000, "r1")
    second = build_paging_notice(1000, "r1")
    message = first.as_message()

    assert first == second
    assert first.key == paging_notice_key("r1", None)
    assert message.metadata == {"ds4.key": "context-paging-notice", "ds4.context_paging": "notice"}
    assert isinstance(message.parts[0], UserPromptPart)
    assert message.parts[0].content == first.content
