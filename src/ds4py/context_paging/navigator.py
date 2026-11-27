"""Public history references, BM25 search, and stable navigation."""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Never, TypedDict

from ds4py.context_paging.public_history import PublicHistoryItem, PublicModelTurn, PublicText, PublicUser
from ds4py.context_paging.recovery_response import compact_json
from ds4py.context_paging.text_units import utf16_length, utf16_slice

_K1 = 1.2
_B = 0.75
_MAX_FILTER_ITEMS = 10
_MAX_LOAD_ITEMS = 3
_TERM_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[\w./:-]+", re.UNICODE)


class HistoryReference(TypedDict):
    historyId: str
    sequence: int
    kind: Literal["user", "modelTurn"]
    timestamp: str
    preview: str
    tools: list[str]
    files: list[str]
    failed: bool
    previousHistoryId: str | None
    nextHistoryId: str | None


@dataclass(frozen=True, slots=True)
class _SearchDocument:
    text: str
    terms: dict[str, int]
    length: int


class HistoryNavigatorError(ValueError):
    def __init__(
        self,
        code: Literal["INVALID_HISTORY_NAVIGATOR_INPUT", "UNKNOWN_HISTORY_ANCHOR", "UNKNOWN_HISTORY_ID"],
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


def _invalid(message: str) -> Never:
    raise HistoryNavigatorError("INVALID_HISTORY_NAVIGATOR_INPUT", message)


def _validate_string(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str) or utf16_length(value) > maximum:
        _invalid(f"{label} must be a string of at most {maximum} characters")
    return value


def _validate_strings(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) > _MAX_FILTER_ITEMS:
        _invalid(f"{label} must contain at most 10 strings")
    return [_validate_string(item, f"{label} item", 200) for item in value]


def _bounded_integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        _invalid(f"{label} must be an integer from {minimum} to {maximum}")
    return value


def _normalize_term(term: str) -> str:
    return term.strip("./:-")


def _tokenize(text: str) -> list[str]:
    return [normalized for match in _TERM_PATTERN.finditer(text.lower()) if (normalized := _normalize_term(match[0]))]


def _search_safe(value: object) -> object:
    if isinstance(value, Mapping):
        if value.get("type") == "image" and isinstance(value.get("data"), str):
            return {"type": "image", "mimeType": value.get("mimeType")}
        return {str(key): _search_safe(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_search_safe(child) for child in value]
    return value


def _serialized(value: object) -> str:
    try:
        return compact_json(_search_safe(value))
    except (TypeError, ValueError) as error:
        raise ValueError("Recovery part contains a value that is not public JSON data.") from error


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        return _serialized(content)
    segments: list[str] = []
    for block in content:
        if block is None:
            continue
        if isinstance(block, Mapping) and block.get("type") == "image":
            segments.append(f"[Image: {block.get('mimeType')}]")
        elif isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str):
            segments.append(str(block.get("text")))
        else:
            segments.append(_serialized(block))
    return " ".join(segments)


def _assistant_values(item: PublicModelTurn) -> list[object]:
    values: list[object] = []
    for part in item.content:
        if part is None:
            values.append(None)
        elif isinstance(part, PublicText):
            values.append({"type": "text", "text": part.text})
        else:
            values.append(
                {
                    "type": "toolCall",
                    "id": part.tool_call_id,
                    "name": part.tool_name,
                    "arguments": part.arguments,
                }
            )
    return values


def _user_content(item: PublicUser) -> object:
    if item.content_is_list or len(item.content) != 1:
        return list(item.content)
    return item.content[0]


def _item_text(item: PublicHistoryItem) -> str:
    if isinstance(item, PublicUser):
        return _content_text(_user_content(item))
    assistant = _assistant_values(item)
    calls = [part for part in item.content if part is not None and not isinstance(part, PublicText)]
    file_segments = [segment for file in item.files for segment in re.split(r"[\\/]", file)]
    return " ".join(
        [
            _content_text(assistant),
            *[text for call in calls for text in (call.tool_name, _serialized(call.arguments))],
            *[_content_text(result.content) for result in item.tool_results],
            *item.files,
            *file_segments,
        ]
    )


def _preview(item: PublicHistoryItem) -> str:
    content: object = _user_content(item) if isinstance(item, PublicUser) else _assistant_values(item)
    compact = " ".join(_content_text(content).split())
    return utf16_slice(compact, 0, 160)


def _excerpt(text: str, match_start: int, match_end: int) -> str:
    maximum = 160
    if utf16_length(text) <= maximum:
        return text
    width = maximum - 2
    start_units = utf16_length(text[:match_start])
    end_units = utf16_length(text[:match_end])
    before = max(0, (width - (end_units - start_units)) // 2)
    start = min(max(0, start_units - before), utf16_length(text) - width)
    end = min(utf16_length(text), start + width)
    excerpt = utf16_slice(text, start, end)
    return f"{'…' if start > 0 else ''}{excerpt}{'…' if end < utf16_length(text) else ''}"


def _search_preview(text: str, query_terms: Sequence[str]) -> str | None:
    compact = " ".join(text.split())
    for term in query_terms:
        matches = list(re.finditer(re.escape(term), compact, re.IGNORECASE))
        if matches:
            match = matches[-1]
            return _excerpt(compact, match.start(), match.end())
    return None


def _directional_indexes(length: int, anchor: int | None, direction: str, count: int, stride: int) -> list[int]:
    step = -stride if direction == "backward" else stride
    current = length - 1 if direction == "backward" else 0
    if anchor is not None:
        current = anchor + step
    indexes: list[int] = []
    while len(indexes) < count and 0 <= current < length:
        indexes.append(current)
        current += step
    return indexes


def _around_indexes(length: int, anchor: int, count: int, stride: int) -> list[int]:
    candidates = [anchor]
    candidates[:0] = list(range(anchor - stride, -1, -stride))[::-1]
    candidates.extend(range(anchor + stride, length, stride))
    position = candidates.index(anchor)
    before = (count - 1) // 2
    start = max(0, min(position - before, max(0, len(candidates) - count)))
    return candidates[start : start + count]


class HistoryNavigator:
    """Index only public items; unchanged visible IDs retain the search corpus."""

    def __init__(self, items: Sequence[PublicHistoryItem] = ()) -> None:
        self._items: tuple[PublicHistoryItem, ...] = ()
        self._references: tuple[HistoryReference, ...] = ()
        self._by_id: dict[str, PublicHistoryItem] = {}
        self._by_sequence: dict[int, PublicHistoryItem] = {}
        self._documents: tuple[_SearchDocument, ...] | None = None
        self._document_frequencies: dict[str, int] = {}
        self._average_document_length = 0.0
        self.cache_identity = 0
        self.rebuild(items)

    @property
    def items(self) -> tuple[PublicHistoryItem, ...]:
        return self._items

    def rebuild(self, items: Sequence[PublicHistoryItem]) -> None:
        if tuple(item.history_id for item in items) == tuple(item.history_id for item in self._items):
            return
        self.cache_identity += 1
        self._items = tuple(replace(item, sequence=index) for index, item in enumerate(items))
        self._by_id = {item.history_id: item for item in self._items}
        self._by_sequence = {item.sequence: item for item in self._items}
        self._references = tuple(self._reference(item, index) for index, item in enumerate(self._items))
        self._documents = None
        self._document_frequencies = {}
        self._average_document_length = 0.0

    def _reference(self, item: PublicHistoryItem, index: int) -> HistoryReference:
        return {
            "historyId": item.history_id,
            "sequence": item.sequence,
            "kind": "user" if isinstance(item, PublicUser) else "modelTurn",
            "timestamp": item.timestamp,
            "preview": _preview(item),
            "tools": list(item.tools) if isinstance(item, PublicModelTurn) else [],
            "files": list(item.files) if isinstance(item, PublicModelTurn) else [],
            "failed": item.failed if isinstance(item, PublicModelTurn) else False,
            "previousHistoryId": self._items[index - 1].history_id if index > 0 else None,
            "nextHistoryId": self._items[index + 1].history_id if index + 1 < len(self._items) else None,
        }

    def search(self, input_value: Mapping[str, object]) -> list[HistoryReference]:
        query = _validate_string(input_value.get("query"), "query", 200)
        files = _validate_strings(input_value["files"], "files") if input_value.get("files") is not None else None
        tools = _validate_strings(input_value["tools"], "tools") if input_value.get("tools") is not None else None
        failed = input_value.get("failed")
        if failed is not None and not isinstance(failed, bool):
            _invalid("failed must be a boolean")
        limit = _bounded_integer(input_value.get("limit", 5), "limit", 1, 10)
        query_terms = list(dict.fromkeys(_tokenize(query)))
        if query and not query_terms:
            return []
        documents = self._build_search_index()
        scored = [
            (index, score)
            for index, item in enumerate(self._items)
            if (score := self._score(documents[index], query_terms, len(documents))) is not None
            and self._matches(item=item, files=files, tools=tools, failed=failed)
        ]
        scored.sort(key=lambda pair: (-pair[1], self._items[pair[0]].sequence))
        results: list[HistoryReference] = []
        for index, _ in scored[:limit]:
            reference = self._references[index].copy()
            if query_terms:
                reference["preview"] = _search_preview(documents[index].text, query_terms) or reference["preview"]
            results.append(reference)
        return results

    def browse(self, input_value: Mapping[str, object]) -> list[HistoryReference]:
        direction = input_value.get("direction")
        if direction not in ("backward", "forward", "around"):
            _invalid("direction must be backward, forward, or around")
        history_id = input_value.get("historyId")
        sequence = input_value.get("sequence")
        if history_id is not None and sequence is not None:
            raise HistoryNavigatorError("UNKNOWN_HISTORY_ANCHOR", "Specify historyId or sequence, not both")
        if history_id is not None:
            history_id = _validate_string(history_id, "historyId", 128)
        if sequence is not None:
            sequence = _bounded_integer(sequence, "sequence", 0, 1_000_000)
        count = _bounded_integer(input_value.get("count", 5), "count", 1, 10)
        stride = _bounded_integer(input_value.get("stride", 1), "stride", 1, 10)
        anchor: PublicHistoryItem | None = None
        if history_id is not None:
            anchor = self._by_id.get(history_id)
        elif sequence is not None:
            anchor = self._by_sequence.get(sequence)
        if (history_id is not None or sequence is not None) and anchor is None:
            raise HistoryNavigatorError("UNKNOWN_HISTORY_ANCHOR", "Unknown history anchor")
        if not self._items and anchor is None:
            return []
        if direction == "around":
            indexes = _around_indexes(
                len(self._items), anchor.sequence if anchor else len(self._items) - 1, count, stride
            )
        else:
            indexes = _directional_indexes(
                len(self._items), anchor.sequence if anchor else None, str(direction), count, stride
            )
        return [self._references[index].copy() for index in indexes]

    def load(self, history_ids: Sequence[str]) -> list[PublicHistoryItem]:
        if not isinstance(history_ids, list) or not 1 <= len(history_ids) <= _MAX_LOAD_ITEMS:
            _invalid("historyIds must contain one to three IDs")
        ids = [_validate_string(history_id, "history ID", 128) for history_id in history_ids]
        loaded = [self._by_id.get(history_id) for history_id in ids]
        for index, item in enumerate(loaded):
            if item is None:
                raise HistoryNavigatorError("UNKNOWN_HISTORY_ID", f"Unknown history ID {ids[index]}")
        return [item for item in loaded if item is not None]

    def _build_search_index(self) -> tuple[_SearchDocument, ...]:
        if self._documents is not None:
            return self._documents
        documents: list[_SearchDocument] = []
        frequencies: dict[str, int] = {}
        for item in self._items:
            terms: dict[str, int] = {}
            for term in _tokenize(_item_text(item)):
                terms[term] = terms.get(term, 0) + 1
            for term in terms:
                frequencies[term] = frequencies.get(term, 0) + 1
            documents.append(_SearchDocument(_item_text(item), terms, sum(terms.values())))
        self._documents = tuple(documents)
        self._document_frequencies = frequencies
        self._average_document_length = (
            sum(document.length for document in documents) / len(documents) if documents else 0.0
        )
        return self._documents

    def _score(self, document: _SearchDocument, query_terms: Sequence[str], total_documents: int) -> float | None:
        if not query_terms:
            return 0.0
        score = 0.0
        matched = False
        for term in query_terms:
            frequency = document.terms.get(term, 0)
            if frequency == 0:
                continue
            matched = True
            document_frequency = self._document_frequencies.get(term, 0)
            idf = math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
            ratio = document.length / self._average_document_length if self._average_document_length else 0.0
            score += idf * (frequency * (_K1 + 1)) / (frequency + _K1 * (1 - _B + _B * ratio))
        return score if matched else None

    @staticmethod
    def _matches(
        *, item: PublicHistoryItem, files: list[str] | None, tools: list[str] | None, failed: bool | None
    ) -> bool:
        item_files = list(item.files) if isinstance(item, PublicModelTurn) else []
        item_tools = list(item.tools) if isinstance(item, PublicModelTurn) else []
        item_failed = item.failed if isinstance(item, PublicModelTurn) else False
        return (
            (files is None or all(file in item_files for file in files))
            and (tools is None or not tools or any(tool in item_tools for tool in tools))
            and (failed is None or failed == item_failed)
        )


__all__ = ["HistoryNavigator", "HistoryNavigatorError", "HistoryReference"]
