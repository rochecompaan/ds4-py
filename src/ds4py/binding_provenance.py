"""Validate saved provenance and prepare pure binder state swaps."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast
from uuid import uuid4

from pydantic_ai.messages import (
    InstructionBaselinePart,
    InstructionDeltaPart,
    InstructionPart,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ds4py._json import is_json_object, is_string_key_mapping
from ds4py.engine.transcript import Segment, validate_segment_coverage
from ds4py.errors import EngineError
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY, DS4_SOURCE, DS4_VIEW_OF, PagingMarker
from ds4py.message_rendering import stable_text

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from ds4py._json import JsonValue
    from ds4py.store import SegmentProvenance


class _ProjectionPart(TypedDict, total=False):
    kind: str
    content: object
    name: str
    args: dict[str, JsonValue]
    id: str
    raw: str


class _MessageProjection(TypedDict):
    kind: str
    parts: list[_ProjectionPart]
    instructions: NotRequired[str]


@dataclass(frozen=True)
class BoundMessage:
    """One keyed range whose snapshot content hash matched the saved record."""

    key: str
    content_hash: str
    start: int
    end: int


@dataclass(frozen=True)
class PreparedProvenance:
    """Validated state a binder can install and assign without further work."""

    installed: tuple[Segment, ...]
    records: tuple[BoundMessage, ...]
    bound_messages: tuple[ModelMessage, ...]


@dataclass(frozen=True)
class RequestViewMessageProvenance:
    """One validated request-view identity and its optional canonical target."""

    view_key: str
    canonical_key: str | None
    marker: PagingMarker | None


@dataclass(frozen=True)
class PreparedRequestViewProvenance:
    """Bound view state and its ordered, immutable canonical identity mapping."""

    prepared: PreparedProvenance
    messages: tuple[RequestViewMessageProvenance, ...]


def _arguments(args: str | Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
    value: object = json.loads(args) if isinstance(args, str) else args
    if value is None:
        return {}
    if not is_string_key_mapping(value):
        raise ValueError("tool call arguments must be a mapping")
    converted = dict(value)
    if not is_json_object(converted):
        raise ValueError("tool call arguments must contain JSON values")
    return converted


def _json_safe_content(value: object) -> object:
    """Project restored media without embedding bytes in provenance JSON."""
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        return {str(key): _json_safe_content(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_content(child) for child in value]
    return value


def _projection(message: ModelMessage) -> _MessageProjection:
    dumped = ModelMessagesTypeAdapter.dump_python([message], warnings=False)[0]
    projected: _MessageProjection = {"kind": dumped["kind"], "parts": []}
    if isinstance(message, ModelRequest) and message.instructions is not None:
        projected["instructions"] = message.instructions
    for part in dumped["parts"]:
        kind = part["part_kind"]
        if kind in ("system-prompt", "instruction", "user-prompt", "text", "thinking"):
            projected["parts"].append({"kind": kind, "content": _json_safe_content(part["content"])})
        elif kind == "tool-call":
            item: _ProjectionPart = {
                "kind": kind,
                "name": part["tool_name"],
                "args": _arguments(part["args"]),
                "id": part["tool_call_id"],
            }
            details = part.get("provider_details") or {}
            raw = details.get("ds4py.raw")
            if (message.metadata or {}).get(DS4_SOURCE) == "sampled" and isinstance(raw, str):
                item["raw"] = raw
            projected["parts"].append(item)
        elif kind in {"tool-return", "retry-prompt"}:
            projected["parts"].append(
                {
                    "kind": kind,
                    "name": part["tool_name"],
                    "content": stable_text(_json_safe_content(part["content"])),
                    "id": part["tool_call_id"],
                }
            )
    return projected


def _digest(value: object) -> str:
    encoded = str(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ensure_key(message: ModelMessage) -> str:
    """Return the message's stable key, assigning one when absent."""
    metadata = dict(message.metadata or {})
    key = metadata.get(DS4_KEY)
    if key is not None and not isinstance(key, str):
        raise TypeError("message ds4.key must be a string or None")
    if key is None or key == "":
        key = uuid4().hex
        metadata[DS4_KEY] = key
    message.metadata = metadata
    return key


def ensure_message_keys(messages: Sequence[ModelMessage]) -> None:
    """Assign stable keys to canonical messages without rendering or native mutation."""
    for message in messages:
        _ensure_key(message)


def keyed_messages(
    messages: Sequence[ModelMessage],
    records: Sequence[BoundMessage],
) -> list[tuple[str, str]]:
    """Pair each message with its key and content hash.

    The harness renders requests from processed copies, so a key written on
    an earlier request can be missing from the message handed back here. The
    transcript records are the authority: a message at the same position with
    the same content is the message they were rendered from, so it keeps
    their key instead of forcing a rewrite.
    """
    keyed = []
    for index, message in enumerate(messages):
        content_hash = _content_hash(message)
        key = (message.metadata or {}).get(DS4_KEY)
        if key is None and index < len(records) and records[index].content_hash == content_hash:
            key = records[index].key
            metadata = dict(message.metadata or {})
            metadata[DS4_KEY] = key
            message.metadata = metadata
        else:
            key = _ensure_key(message)
        keyed.append((key, content_hash))
    return keyed


def _content_hash(message: ModelMessage) -> str:
    return _digest(_projection(message))


def message_content_hash(message: ModelMessage) -> str:
    """Return the canonical content hash without changing the message."""
    return _content_hash(message)


def _key_recovery_hash(message: ModelMessage) -> str:
    """Match processed copies that gained request instructions but lost their key."""
    projection = _projection(message)
    projection.pop("instructions", None)
    return _digest(projection)


def _validate(messages: Sequence[ModelMessage]) -> None:
    for message in messages:
        parts = message.parts
        for part in parts:
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                raise EngineError("image input requires M5 vision support")
            if isinstance(message, ModelRequest) and not isinstance(
                part,
                (
                    SystemPromptPart,
                    InstructionPart,
                    InstructionBaselinePart,
                    InstructionDeltaPart,
                    UserPromptPart,
                    ToolReturnPart,
                    RetryPromptPart,
                ),
            ):
                raise EngineError(f"unsupported harness request part: {type(part).__name__}")
            if isinstance(message, ModelResponse) and not isinstance(part, (TextPart, ThinkingPart, ToolCallPart)):
                raise EngineError(f"unsupported harness response part: {type(part).__name__}")


def validate_messages(messages: Sequence[ModelMessage]) -> None:
    """Validate supported parts and reject unsupported media at the model boundary."""
    _validate(messages)


def _installed_segment(provenance: SegmentProvenance) -> Segment:
    kind = provenance.kind
    meta = {
        "key": provenance.key,
        "content_hash": provenance.content_hash,
        "source": provenance.source,
    }
    if kind == "note" or kind.startswith("system_note"):
        subtype = kind.split(":", 1)[1] if ":" in kind else ""
        if subtype:
            meta["subtype"] = subtype
        if provenance.text is not None:
            meta["text"] = provenance.text
        kind = "note"
    return Segment(provenance.start, provenance.end, kind, MappingProxyType(meta))


def _installable(provenance: Segment | SegmentProvenance) -> tuple[Segment, str, str]:
    """Return the installable segment and the message identity it names."""
    if isinstance(provenance, Segment):
        return (
            provenance,
            str(provenance.meta.get("key", "")),
            str(provenance.meta.get("content_hash", "")),
        )
    return _installed_segment(provenance), provenance.key, provenance.content_hash


def prepare_provenance(  # noqa: C901, PLR0912, PLR0915 - Restore validation covers each persisted contract.
    messages: Sequence[ModelMessage],
    segments: Sequence[Segment | SegmentProvenance],
    *,
    loaded: int | None,
) -> PreparedProvenance:
    """Validate saved provenance against a snapshot and copy the state to install.

    ``segments`` is save-record provenance or transcript segments whose metadata
    carries the same identity. Snapshot messages are matched to keyed ranges in
    order and by content hash: a record and its snapshot are written by one
    save, so the order is fixed, and the hash proves the snapshot still
    describes the tokens. ``loaded`` is the length of a payload already
    installed in the transcript, or ``None`` to skip that check.
    """
    snapshot = list(messages)
    _validate(snapshot)
    installed = []
    entries: list[tuple[Segment | SegmentProvenance, str]] = []
    ranges: dict[str, list[int]] = {}
    keys: list[str] = []
    hashes: dict[str, str] = {}
    for provenance in segments:
        installed_segment, key, content_hash = _installable(provenance)
        installed.append(installed_segment)
        entries.append((provenance, key))
        if not key:
            continue
        if key in ranges:
            if content_hash != hashes[key]:
                raise EngineError(f"save record ranges for message key {key!r} disagree on the content hash")
            ranges[key][1] = provenance.end
        else:
            ranges[key] = [provenance.start, provenance.end]
            keys.append(key)
            hashes[key] = content_hash
    token_count = loaded if loaded is not None else (installed[-1].end if installed else 0)
    try:
        validate_segment_coverage(installed, token_count)
    except ValueError as error:
        raise EngineError(f"save record does not continue the loaded transcript: {error}") from error

    # A response's range starts at its assistant prefix, which has no key.
    pending_prefix = None
    for provenance, key in entries:
        if not key:
            if provenance.kind == "assistant_prefix":
                pending_prefix = provenance.start
            continue
        if pending_prefix is not None and pending_prefix < ranges[key][0]:
            ranges[key][0] = pending_prefix
        pending_prefix = None

    positions: dict[str, int] = {}
    for index, message in enumerate(snapshot):
        key = (message.metadata or {}).get(DS4_KEY)
        if key is None:
            continue
        if key in positions:
            raise EngineError(f"the snapshot has duplicate message key {key!r}")
        positions[key] = index

    records = []
    bound_messages = []
    previous = -1
    for key in keys:
        position = positions.get(key)
        if position is None:
            raise EngineError(f"the save record message key {key!r} is absent from the snapshot")
        if position <= previous:
            raise EngineError(f"the save record message key {key!r} is out of snapshot order")
        message = snapshot[position]
        content_hash = _content_hash(message)
        if content_hash != hashes[key]:
            raise EngineError(f"the save record content hash for message key {key!r} does not match the snapshot")
        records.append(BoundMessage(key, content_hash, ranges[key][0], ranges[key][1]))
        bound_messages.append(message)
        previous = position

    return PreparedProvenance(
        installed=tuple(installed),
        records=tuple(records),
        bound_messages=tuple(deepcopy(message) for message in bound_messages),
    )


def _canonical_identities(messages: Sequence[ModelMessage]) -> dict[str, tuple[ModelMessage, str]]:
    validate_messages(messages)
    identities: dict[str, tuple[ModelMessage, str]] = {}
    for message in messages:
        key = (message.metadata or {}).get(DS4_KEY)
        if key is None:
            continue
        if not isinstance(key, str) or not key:
            raise EngineError("canonical message keys must be non-empty strings")
        if key in identities:
            raise EngineError(f"canonical snapshot has duplicate message key {key!r}")
        identities[key] = (message, message_content_hash(message))
    return identities


def validate_canonical_snapshot(messages: Sequence[ModelMessage]) -> None:
    """Validate canonical schema, media, and stable key identity without token segments."""
    _canonical_identities(messages)


def _request_view_identity(
    message: ModelMessage,
    canonical: Mapping[str, tuple[ModelMessage, str]],
) -> RequestViewMessageProvenance:
    metadata = message.metadata or {}
    key = metadata.get(DS4_KEY)
    if not isinstance(key, str) or not key:
        raise EngineError("request view message keys must be non-empty strings")

    raw_marker = metadata.get(DS4_CONTEXT_PAGING)
    if DS4_CONTEXT_PAGING in metadata and raw_marker not in ("notice", "recovery"):
        raise EngineError(f"invalid paging marker {raw_marker!r}")
    marker = cast("PagingMarker | None", raw_marker)
    has_target = DS4_VIEW_OF in metadata
    target = metadata.get(DS4_VIEW_OF)

    if marker is not None and key in canonical:
        raise EngineError(f"paging message {key!r} must have a fresh view key")
    if marker == "notice":
        if has_target:
            raise EngineError("a paging notice must not have a canonical target")
        return RequestViewMessageProvenance(key, None, marker)
    if marker == "recovery":
        if not isinstance(target, str) or not target or target not in canonical:
            raise EngineError("a recovery view message target must name an existing canonical message")
        return RequestViewMessageProvenance(key, target, marker)
    if has_target:
        raise EngineError("an unmarked view message must not have a canonical target")

    identity = canonical.get(key)
    if identity is None or identity[1] != message_content_hash(message):
        raise EngineError("an unmarked view message must resolve to canonical content with the same key and hash")
    return RequestViewMessageProvenance(key, key, None)


def prepare_request_view_provenance(
    *,
    canonical_messages: Sequence[ModelMessage],
    request_view: Sequence[ModelMessage],
    segments: Sequence[Segment | SegmentProvenance],
    loaded: int | None,
) -> PreparedRequestViewProvenance:
    """Validate a view against its token records and full canonical snapshot."""
    canonical = _canonical_identities(canonical_messages)
    prepared = prepare_provenance(request_view, segments, loaded=loaded)
    identities = tuple(_request_view_identity(message, canonical) for message in request_view)
    return PreparedRequestViewProvenance(prepared, identities)
