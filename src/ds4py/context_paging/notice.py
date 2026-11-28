"""Frozen paging notices with bounded copied conversation and exact handles."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic_ai.messages import ModelRequest, UserPromptPart

from ds4py.context_paging.public_history import PublicHistoryItem, PublicModelTurn, PublicText, PublicUser
from ds4py.context_paging.text_units import clip_utf16, utf16_length
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY

RECENT_REPLY_TURNS = 3
ENTRY_CHARACTERS = 6_000
SECTION_CHARACTERS = 32_000
MINIMUM_CLIPPED_CHARACTERS = 500
NOTICE_VIEW_KEY = "context-paging-notice"

_SECTION_HEADING = (
    "Earlier conversation, copied from session history, oldest first. These messages were already handled. "
    "They are not new requests, but they may still define the current task."
)


@dataclass(frozen=True, slots=True)
class ToolRecoveryReference:
    history_id: str
    tool_call_id: str
    tool_name: str


@dataclass(frozen=True, slots=True)
class PagingNotice:
    content: str
    key: str

    def as_message(self) -> ModelRequest:
        return ModelRequest(
            [UserPromptPart(self.content, timestamp=datetime.fromtimestamp(0, UTC))],
            metadata={DS4_KEY: NOTICE_VIEW_KEY, DS4_CONTEXT_PAGING: "notice"},
        )


@dataclass(frozen=True, slots=True)
class ConversationEntry:
    kind: Literal["user message", "assistant reply"]
    history_id: str
    text: str
    part: str


@dataclass(slots=True)
class _RecordedEntry:
    entry: ConversationEntry
    turn: int
    whole: str | None = None


def _request_entry(message: PublicUser) -> ConversationEntry | None:
    text = "\n".join(
        content
        if isinstance(content, str)
        else str(content.get("text", ""))
        if isinstance(content, Mapping) and content.get("type") == "text"
        else ""
        for content in message.content
    )
    if not text.strip():
        return None
    return ConversationEntry("user message", message.history_id, text, "/userMessage/content")


def _reply_entry(message: PublicModelTurn) -> ConversationEntry | None:
    indexed = [
        (index, part.text)
        for index, part in enumerate(message.content)
        if isinstance(part, PublicText) and part.text.strip()
    ]
    if not indexed:
        return None
    part = f"/assistantMessage/content/{indexed[0][0]}" if len(indexed) == 1 else "/assistantMessage/content"
    return ConversationEntry("assistant reply", message.history_id, "\n".join(text for _, text in indexed), part)


def _header(entry: ConversationEntry) -> str:
    history_id = json.dumps(entry.history_id, ensure_ascii=False)
    return f"--- {entry.kind}, historyId {history_id} ---"


def _clip_marker(entry: ConversationEntry, shown: int) -> str:
    handle = json.dumps({"historyId": entry.history_id, "part": entry.part}, ensure_ascii=False, separators=(",", ":"))
    return (
        f"[Clipped after {shown:,} of {utf16_length(entry.text):,} characters. "
        f"Read the full text with read_context_output({handle}).]"
    )


def _format_entry(entry: ConversationEntry, limit: int) -> str:
    text, shown = clip_utf16(entry.text, limit)
    lines = [_header(entry), text]
    if shown < utf16_length(entry.text):
        lines.append(_clip_marker(entry, shown))
    return "\n".join(lines)


def _clipped_entry(entry: ConversationEntry, room: int) -> str | None:
    overhead = utf16_length(_header(entry)) + utf16_length(_clip_marker(entry, ENTRY_CHARACTERS)) + 2
    limit = min(ENTRY_CHARACTERS, room - overhead)
    return _format_entry(entry, limit) if limit >= MINIMUM_CLIPPED_CHARACTERS else None


def _omitted_line(omitted: int, oldest_shown_id: str | None) -> str:
    count = "1 earlier user message is" if omitted == 1 else f"{omitted} earlier user messages are"
    if oldest_shown_id is None:
        return f"{count} not shown. Find them with search_history or browse_history."
    handle = json.dumps(
        {"historyId": oldest_shown_id, "direction": "backward"},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{count} not shown. Find them with browse_history({handle})."


class EvictedConversation:
    """Whole evicted turns, newest-three replies, and bounded section rendering."""

    def __init__(self) -> None:
        self._requests: list[_RecordedEntry] = []
        self._replies: list[_RecordedEntry] = []
        self._turn_count = 0
        self._longest_id = ""

    def add_turn(self, items: list[PublicHistoryItem] | tuple[PublicHistoryItem, ...]) -> bool:
        turn = self._turn_count
        self._turn_count += 1
        first = items[0] if items else None
        request = _request_entry(first) if isinstance(first, PublicUser) else None
        public_reply = next(
            (item for item in reversed(items) if isinstance(item, PublicModelTurn) and _reply_entry(item) is not None),
            None,
        )
        reply = _reply_entry(public_reply) if public_reply is not None else None
        if request is not None:
            self._requests.append(_RecordedEntry(request, turn))
        if reply is not None:
            self._replies.append(_RecordedEntry(reply, turn))
        while self._replies and self._replies[0].turn < self._turn_count - RECENT_REPLY_TURNS:
            self._replies.pop(0)
        for entry in (request, reply):
            if entry is not None and utf16_length(entry.history_id) > utf16_length(self._longest_id):
                self._longest_id = entry.history_id
        return request is not None or reply is not None

    def _newest_first(self) -> list[_RecordedEntry]:
        entries: list[_RecordedEntry] = []
        request = len(self._requests) - 1
        reply = len(self._replies) - 1
        while request >= 0 or reply >= 0:
            reply_is_newer = reply >= 0 and (request < 0 or self._replies[reply].turn >= self._requests[request].turn)
            if reply_is_newer:
                entries.append(self._replies[reply])
                reply -= 1
            else:
                entries.append(self._requests[request])
                request -= 1
        return entries

    def section_lines(self, max_characters: int) -> list[str]:
        room = (
            max_characters
            - utf16_length(_SECTION_HEADING)
            - utf16_length(_omitted_line(len(self._requests), self._longest_id))
            - 5
        )
        blocks: list[str] = []
        shown_requests = 0
        oldest_shown: ConversationEntry | None = None
        for recorded in self._newest_first():
            whole = recorded.whole
            if whole is None:
                whole = _format_entry(recorded.entry, ENTRY_CHARACTERS)
                recorded.whole = whole
            block = whole if utf16_length(whole) <= room else _clipped_entry(recorded.entry, room)
            if block is None:
                break
            blocks.append(block)
            room -= utf16_length(block) + 2
            oldest_shown = recorded.entry
            if recorded.entry.kind == "user message":
                shown_requests += 1
            if block != whole:
                break
        if not blocks:
            return []
        omitted = len(self._requests) - shown_requests
        lines = ["", _SECTION_HEADING]
        if omitted > 0:
            lines.append(_omitted_line(omitted, oldest_shown.history_id if oldest_shown else None))
        lines.extend(["", "\n\n".join(reversed(blocks))])
        return lines


def conversation_character_limit(budget_tokens: int) -> int:
    return min(SECTION_CHARACTERS, budget_tokens * 2 // 5)


def paging_notice_key(history_id: str | None, tool_reference: ToolRecoveryReference | None) -> str:
    return json.dumps(
        [
            history_id,
            tool_reference.history_id if tool_reference else None,
            tool_reference.tool_call_id if tool_reference else None,
            tool_reference.tool_name if tool_reference else None,
        ],
        separators=(",", ":"),
    )


def _notice_header(
    budget_tokens: int,
    history_id: str | None,
    tool_reference: ToolRecoveryReference | None,
    *,
    with_conversation: bool,
) -> list[str]:
    lines = [
        "[Context paging notice — generated by the extension]",
        "This is context paging, not compaction. Nothing was summarized.",
        f"Older context left the {budget_tokens:,}-token rolling window. Raw session history is unchanged.",
        "Continue the current task. Do not stop or ask the user for direction because of this notice.",
        (
            "To recover detail, first use the earlier conversation below, then plan or notes files, "
            "then search_history or browse_history."
            if with_conversation
            else "To recover detail, first use plan or notes files, then search_history or browse_history."
        ),
        "Use load_history for exact items or read_context_output for exact output pages.",
    ]
    if history_id is not None:
        lines.append(f"Recent evicted historyId: {json.dumps(history_id, ensure_ascii=False)}.")
    if tool_reference is not None:
        handle = json.dumps(
            {
                "historyId": tool_reference.history_id,
                "source": "toolResult",
                "toolCallId": tool_reference.tool_call_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        lines.extend(
            [
                f"Tool name: {tool_reference.tool_name}.",
                f"Read evicted tool output with read_context_output({handle}).",
            ]
        )
    return lines


def build_paging_notice(
    budget_tokens: int,
    history_id: str | None,
    tool_reference: ToolRecoveryReference | None = None,
    conversation: EvictedConversation | None = None,
    max_characters: int = SECTION_CHARACTERS * 1000,
) -> PagingNotice:
    header = _notice_header(budget_tokens, history_id, tool_reference, with_conversation=True)
    room = min(conversation_character_limit(budget_tokens), max_characters - utf16_length("\n".join(header)))
    conversation_lines = conversation.section_lines(room) if conversation is not None and room > 0 else []
    lines = (
        [*header, *conversation_lines]
        if conversation_lines
        else _notice_header(budget_tokens, history_id, tool_reference, with_conversation=False)
    )
    return PagingNotice("\n".join(lines), paging_notice_key(history_id, tool_reference))


def protected_overflow_notice(budget_tokens: int) -> PagingNotice:
    content = "\n".join(
        [
            "[Context paging notice — generated by the extension]",
            "The newest tool-result exchange is present in full for this follow-up call.",
            f"The normal rolling budget is {budget_tokens:,} estimated tokens.",
            "This exchange can leave context after this call.",
            "Use search_history or browse_history, then load_history or read_context_output, to recover it.",
        ]
    )
    return PagingNotice(content, "protected-overflow")


def recovery_notice() -> PagingNotice:
    content = (
        "[Context paging notice — generated by the extension]\n"
        "The newest tool-result payloads were replaced with recovery references or exclusion notices "
        "because they exceeded the active model context window.\n"
        "For recoverable results, use read_context_output with the exact arguments in their replacements."
    )
    return PagingNotice(content, "recovery")


__all__ = [
    "ENTRY_CHARACTERS",
    "MINIMUM_CLIPPED_CHARACTERS",
    "EvictedConversation",
    "PagingNotice",
    "ToolRecoveryReference",
    "build_paging_notice",
    "conversation_character_limit",
    "paging_notice_key",
    "protected_overflow_notice",
    "recovery_notice",
]
