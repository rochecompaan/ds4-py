"""Public recovery DTOs, JSON Pointer reads, and bounded text pages."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict, TypeIs

from ds4py.context_paging.public_history import (
    PublicHistoryItem,
    PublicModelTurn,
    PublicText,
    PublicToolCall,
    PublicUser,
)
from ds4py.context_paging.recovery_response import (
    MAXIMUM_HISTORY_ID_CHARACTERS,
    MAXIMUM_PAGE_LIMIT,
    MAXIMUM_PART_CHARACTERS,
    compact_json,
    fits_recovery_text,
)
from ds4py.context_paging.text_units import utf16_length, utf16_slice

_MAXIMUM_CHOICE_CHARACTERS = 160
_MAXIMUM_CHOICES_BYTES = 1_536
_PARTS_HINT = 'Use read_context_output with mode "parts" to list every public part with read handles.'
_PLAIN_TEXT_FIELD_COUNT = 2


class RecoveryInputError(Exception):
    """A model-correctable recovery request that does not fit legacy selectors."""


@dataclass(frozen=True, slots=True)
class HistoryPart:
    part: str
    value: object


class PartReadHandle(TypedDict):
    historyId: str
    part: str
    mode: NotRequired[Literal["image"]]


class ImageMetadata(TypedDict):
    type: Literal["image"]
    mimeType: str
    bytes: int
    read: PartReadHandle


class HistoryPartDescription(TypedDict):
    part: str
    type: str
    bytes: int
    preview: NotRequired[str]
    read: PartReadHandle


class RecoveryPartText(TypedDict):
    format: Literal["text", "json"]
    text: str


class ContextOutputNext(TypedDict):
    historyId: str
    part: str
    offset: int
    limit: NotRequired[int]


class ContextOutputHeader(TypedDict):
    historyId: str
    part: str
    format: Literal["text", "json"]
    offset: int
    nextOffset: int | None
    totalCharacters: int
    next: NotRequired[ContextOutputNext]


@dataclass(frozen=True, slots=True)
class ContextOutputPage:
    header: ContextOutputHeader
    text: str


def is_recovery_image(value: object) -> TypeIs[Mapping[str, object]]:
    return (
        isinstance(value, Mapping)
        and value.get("type") == "image"
        and isinstance(value.get("data"), str)
        and isinstance(value.get("mimeType"), str)
    )


def public_history_item(items: Sequence[PublicHistoryItem], history_id: str) -> PublicHistoryItem:
    item = next((item for item in items if item.history_id == history_id), None)
    if item is None:
        raise ValueError(f"Unknown history ID {history_id}.")
    return item


def _text_value(part: PublicText) -> dict[str, object]:
    return {"type": "text", "text": part.text}


def _call_value(part: PublicToolCall) -> dict[str, object]:
    return {
        "type": "toolCall",
        "id": part.tool_call_id,
        "name": part.tool_name,
        "arguments": part.arguments,
    }


def _user_content_value(item: PublicUser) -> object:
    if item.content_is_list or len(item.content) != 1:
        return list(item.content)
    return item.content[0]


def public_item_value(item: PublicHistoryItem) -> dict[str, object]:
    """Convert the public DTO to stable JSON-shaped values, not Pydantic internals."""
    if isinstance(item, PublicUser):
        return {
            "id": item.history_id,
            "kind": "user",
            "sequence": item.sequence,
            "timestamp": item.timestamp,
            "userMessage": {
                "role": "user",
                "content": _user_content_value(item),
                "timestamp": item.message_timestamp,
            },
        }
    tool_results = []
    for result in item.tool_results:
        value: dict[str, object] = {
            "role": "toolResult",
            "toolCallId": result.tool_call_id,
            "toolName": result.tool_name,
            "content": result.content,
            "isError": result.failed,
            "timestamp": result.timestamp,
        }
        if result.nested_calls is not None:
            value["nestedCalls"] = result.nested_calls
        if result.details is not None:
            value["details"] = result.details
        tool_results.append(value)
    value = {
        "id": item.history_id,
        "kind": "modelTurn",
        "sequence": item.sequence,
        "timestamp": item.timestamp,
        "assistantMessage": {
            "role": "assistant",
            "api": item.assistant.api,
            "provider": item.assistant.provider,
            "model": item.assistant.model,
            "content": [
                None if part is None else _text_value(part) if isinstance(part, PublicText) else _call_value(part)
                for part in item.content
            ],
            "usage": item.assistant.usage,
            "stopReason": item.assistant.stop_reason,
            "timestamp": item.assistant.timestamp,
        },
        "toolResults": tool_results,
        "metadata": {
            "tools": list(item.tools),
            "files": list(item.files),
            "failed": item.failed,
        },
    }
    if item.interrupted_by_restore:
        value["interruptedByRestore"] = True
    return value


def _escape_key(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def clip_value(value: str) -> str:
    if utf16_length(value) <= _MAXIMUM_CHOICE_CHARACTERS:
        return value
    return f"{utf16_slice(value, 0, _MAXIMUM_CHOICE_CHARACTERS - 1)}…"


def list_choices(choices: Sequence[str], limit: int = 20) -> str:
    if not choices:
        return "none"
    shown: list[str] = []
    size = 0
    for choice in choices[:limit]:
        clipped = clip_value(choice)
        size += len(compact_json(clipped).encode("utf-8"))
        if size > _MAXIMUM_CHOICES_BYTES:
            break
        shown.append(clipped)
    omitted = len(choices) - len(shown)
    return f"{', '.join(shown)}, and {omitted} more" if omitted > 0 else ", ".join(shown)


def _unknown_part(item: PublicHistoryItem) -> ValueError:
    pointers = [compact_json(part.part) for part in history_parts(item)]
    return ValueError(
        f"Unknown public recovery part. Public parts: {list_choices(pointers)}. "
        'Use read_context_output with mode "parts" to list them with read handles.'
    )


def select_history_part(item: PublicHistoryItem, pointer: str) -> object:
    if not isinstance(pointer, str) or (pointer != "" and not pointer.startswith("/")) or "~2" in pointer:
        raise ValueError("Invalid recovery part pointer.")
    value: object = public_item_value(item)
    segments = [] if pointer == "" else pointer[1:].split("/")
    for segment in segments:
        if "~" in segment and segment.replace("~1", "").replace("~0", "").find("~") >= 0:
            raise ValueError("Invalid recovery part pointer.")
        key = segment.replace("~1", "/").replace("~0", "~")
        if is_recovery_image(value):
            raise ValueError("Read image attachments through their image handles, not image fields.")
        if isinstance(value, Mapping) and key in value:
            value = value[key]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and key.isdigit():
            index = int(key)
            if index >= len(value):
                raise _unknown_part(item)
            value = value[index]
        else:
            raise _unknown_part(item)
    if value is None and pointer.startswith("/assistantMessage/content/"):
        raise ValueError(str(_unknown_part(item)))
    return value


def history_parts(item: PublicHistoryItem) -> list[HistoryPart]:
    parts: list[HistoryPart] = []
    value = public_item_value(item)

    def add_content(content: object, path: str) -> None:
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            for index, child in enumerate(content):
                if child is not None:
                    parts.append(HistoryPart(f"{path}/{index}", child))
        else:
            parts.append(HistoryPart(path, content))

    if isinstance(item, PublicUser):
        add_content(_user_content_value(item), "/userMessage/content")
    else:
        content = [
            None if part is None else _text_value(part) if isinstance(part, PublicText) else _call_value(part)
            for part in item.content
        ]
        add_content(content, "/assistantMessage/content")
        for index, result in enumerate(item.tool_results):
            add_content(result.content, f"/toolResults/{index}/content")
        parts.append(HistoryPart("/metadata", value["metadata"]))
    parts.append(HistoryPart("", value))
    return parts


def _image_metadata(image: Mapping[str, object], history_id: str, part: str) -> ImageMetadata:
    data = image["data"]
    mime_type = image["mimeType"]
    if not isinstance(data, str) or not isinstance(mime_type, str):
        raise TypeError("Image data and MIME type must be strings.")
    padding = 2 if data.endswith("==") else 1 if data.endswith("=") else 0
    return {
        "type": "image",
        "mimeType": mime_type,
        "bytes": max(0, utf16_length(data) * 3 // 4 - padding),
        "read": {"historyId": history_id, "part": part, "mode": "image"},
    }


def _image_safe(value: object, history_id: str, path: str) -> object:
    if is_recovery_image(value):
        return _image_metadata(value, history_id, path)
    if isinstance(value, Mapping):
        return {
            str(key): _image_safe(child, history_id, f"{path}/{_escape_key(str(key))}") for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_image_safe(child, history_id, f"{path}/{index}") for index, child in enumerate(value)]
    if isinstance(value, (bytes, bytearray)):
        raise TypeError("Binary payloads require an image handle and cannot be serialized as JSON.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("Recovery part contains a value that is not public JSON data.")


def serialize_recovery_part(item: PublicHistoryItem, pointer: str) -> str:
    value = select_history_part(item, pointer)
    return compact_json(_image_safe(value, item.history_id, pointer))


def _plain_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and len(value) == _PLAIN_TEXT_FIELD_COUNT and value.get("type") == "text":
        text = value.get("text")
        return text if isinstance(text, str) else None
    return None


def recovery_part_text(item: PublicHistoryItem, pointer: str) -> RecoveryPartText:
    text = _plain_text(select_history_part(item, pointer))
    if text is not None:
        return {"format": "text", "text": text}
    return {"format": "json", "text": serialize_recovery_part(item, pointer)}


def describe_history_part(item: PublicHistoryItem, part: HistoryPart) -> HistoryPartDescription:
    if is_recovery_image(part.value):
        metadata = _image_metadata(part.value, item.history_id, part.part)
        return {
            "part": part.part,
            "type": "image",
            "bytes": metadata["bytes"],
            "read": metadata["read"],
        }
    recovery = recovery_part_text(item, part.part)
    value_type = "json"
    if recovery["format"] == "text":
        value_type = "text"
    elif isinstance(part.value, Mapping) and part.value.get("type") in ("text", "toolCall"):
        value_type = str(part.value.get("type"))
    preview = utf16_slice(recovery["text"], 0, 160)
    return {
        "part": part.part,
        "type": value_type,
        "bytes": len(recovery["text"].encode("utf-8")),
        "preview": preview,
        "read": {"historyId": item.history_id, "part": part.part},
    }


def context_output_texts(page: ContextOutputPage) -> tuple[str, ...]:
    header = compact_json(page.header)
    return (header,) if page.text == "" else (header, page.text)


def _safe_integer(value: object) -> TypeIs[int]:
    return isinstance(value, int) and not isinstance(value, bool) and abs(value) <= 2**53 - 1


def _assistant_part(item: PublicModelTurn, content_index: object) -> str:
    blocks = [(index, part) for index, part in enumerate(item.content) if part is not None]
    if content_index is None:
        texts = [(index, part) for index, part in blocks if isinstance(part, PublicText)]
        only = texts[0] if len(texts) == 1 else blocks[0] if not texts and len(blocks) == 1 else None
        if only is None:
            choices = [f"{index} ({'text' if isinstance(part, PublicText) else 'toolCall'})" for index, part in blocks]
            raise ValueError(
                "Assistant output needs contentIndex unless the turn has exactly one public text block or one public block. "
                f"Public blocks: {list_choices(choices)}. {_PARTS_HINT}"
            )
        return f"/assistantMessage/content/{only[0]}"
    if not _safe_integer(content_index) or int(content_index) < 0:
        raise ValueError("Assistant output contentIndex must be a nonnegative integer.")
    index = int(content_index)
    if index >= len(item.content):
        raise ValueError(f"Unknown assistant content index {index}.")
    if item.content[index] is None:
        raise ValueError(f"Assistant content index {index} is not public recovery output.")
    return f"/assistantMessage/content/{index}"


def _tool_result_part(item: PublicModelTurn, tool_call_id: object) -> str:
    if tool_call_id is None:
        if len(item.tool_results) == 1:
            return "/toolResults/0"
        choices = [f"{result.tool_call_id} ({result.tool_name})" for result in item.tool_results]
        raise ValueError(
            "Tool-result output needs toolCallId unless the turn has exactly one public tool result. "
            f"Public tool results: {list_choices(choices)}. {_PARTS_HINT}"
        )
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError("Tool-result output toolCallId must be a nonempty string.")
    for index, result in enumerate(item.tool_results):
        if result.tool_call_id == tool_call_id:
            return f"/toolResults/{index}"
    raise ValueError(f"Unknown or non-public tool-result toolCallId {clip_value(tool_call_id)}.")


def _selected_part(
    items: Sequence[PublicHistoryItem], input_value: Mapping[str, object], history_id: str
) -> tuple[PublicHistoryItem, str]:
    part = input_value.get("part")
    if part is not None:
        if not isinstance(part, str):
            raise ValueError("Invalid recovery part pointer.")
        if any(input_value.get(key) is not None for key in ("source", "contentIndex", "toolCallId")):
            raise ValueError("A part handle cannot be combined with legacy output selectors.")
        return public_history_item(items, history_id), part
    source = input_value.get("source")
    item = public_history_item(items, history_id)
    if not isinstance(item, PublicModelTurn):
        raise RecoveryInputError(
            f"History ID {history_id} is not a model turn; use an exact part handle for this item."
        )
    if source == "assistant":
        if input_value.get("toolCallId") is not None:
            raise ValueError("Assistant output does not accept toolCallId.")
        return item, _assistant_part(item, input_value.get("contentIndex"))
    if source == "toolResult":
        if input_value.get("contentIndex") is not None:
            raise ValueError("Tool-result output does not accept contentIndex.")
        return item, _tool_result_part(item, input_value.get("toolCallId"))
    raise ValueError("Output requires a part handle or an assistant/toolResult source.")


def _page_at(  # noqa: PLR0913, PLR0917 - A page echoes the complete continuation context.
    item: PublicHistoryItem,
    part: str,
    recovery: RecoveryPartText,
    offset: int,
    limit: int | None,
    end: int,
) -> ContextOutputPage:
    text = utf16_slice(recovery["text"], offset, end)
    actual_end = offset + utf16_length(text)
    total = utf16_length(recovery["text"])
    next_offset = actual_end if actual_end < total else None
    header = ContextOutputHeader(
        historyId=item.history_id,
        part=part,
        format=recovery["format"],
        offset=offset,
        nextOffset=next_offset,
        totalCharacters=total,
    )
    if next_offset is not None:
        next_handle = ContextOutputNext(historyId=item.history_id, part=part, offset=next_offset)
        if limit is not None:
            next_handle["limit"] = limit
        header["next"] = next_handle
    return ContextOutputPage(header, text)


def read_context_output(  # noqa: PLR0912 - Input validation and page fitting have distinct failure branches.
    items: Sequence[PublicHistoryItem], input_value: Mapping[str, object]
) -> ContextOutputPage:
    """Read one exact public part page with UTF-16 offsets and a 16 KiB reply cap."""
    history_id = input_value.get("historyId")
    if not isinstance(history_id, str):
        raise TypeError("historyId must be a string.")
    part = input_value.get("part")
    if utf16_length(history_id) > MAXIMUM_HISTORY_ID_CHARACTERS:
        raise ValueError(f"historyId must be at most {MAXIMUM_HISTORY_ID_CHARACTERS} characters.")
    if isinstance(part, str) and utf16_length(part) > MAXIMUM_PART_CHARACTERS:
        raise ValueError(f"part must be at most {MAXIMUM_PART_CHARACTERS} characters.")
    if input_value.get("mode") not in (None, "text"):
        raise ValueError("Text output requires text mode.")
    offset = input_value.get("offset", 0)
    limit = input_value.get("limit")
    if not _safe_integer(offset) or int(offset) < 0:
        raise ValueError("Output offset must be a nonnegative integer.")
    if limit is not None and (not _safe_integer(limit) or not 1 <= int(limit) <= MAXIMUM_PAGE_LIMIT):
        raise ValueError(f"Output limit must be an integer from 1 to {MAXIMUM_PAGE_LIMIT}.")
    offset = int(offset)
    page_limit = int(limit) if limit is not None else None
    item, selected = _selected_part(items, input_value, history_id)
    recovery = recovery_part_text(item, selected)
    total = utf16_length(recovery["text"])
    if offset > total:
        raise ValueError("Output offset exceeds the output length.")
    if utf16_length(utf16_slice(recovery["text"], 0, offset)) != offset:
        raise ValueError("Output offset splits a Unicode character.")

    def page(end: int) -> ContextOutputPage:
        return _page_at(item, selected, recovery, offset, page_limit, end)

    def fits(end: int) -> bool:
        return fits_recovery_text(context_output_texts(page(end)))

    low = offset
    end = min(offset + (page_limit or MAXIMUM_PAGE_LIMIT), total)
    high = end if fits(end) else end - 1
    if high == end:
        low = end
    while low < high:
        middle = math.ceil((low + high) / 2)
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    result = page(low)
    if not result.text and offset < total:
        raise ValueError("Output limit is too small for the next Unicode character. Increase limit.")
    return result


__all__ = [
    "ContextOutputHeader",
    "ContextOutputNext",
    "ContextOutputPage",
    "HistoryPart",
    "HistoryPartDescription",
    "PartReadHandle",
    "RecoveryPartText",
    "clip_value",
    "context_output_texts",
    "describe_history_part",
    "history_parts",
    "is_recovery_image",
    "list_choices",
    "public_history_item",
    "public_item_value",
    "read_context_output",
    "recovery_part_text",
    "select_history_part",
    "serialize_recovery_part",
]
