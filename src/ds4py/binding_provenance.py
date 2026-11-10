"""Validate saved provenance and prepare pure binder state swaps."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, NotRequired, TypedDict

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
from ds4py.message_rendering import stable_text

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic_ai.messages import ModelMessage

    from ds4py._json import JsonValue
    from ds4py.store import SegmentProvenance


class _ProjectionPart(TypedDict, total=False):
    kind: str
    content: str
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


def _projection(message: ModelMessage) -> _MessageProjection:
    dumped = ModelMessagesTypeAdapter.dump_python([message], warnings=False)[0]
    projected: _MessageProjection = {"kind": dumped["kind"], "parts": []}
    if isinstance(message, ModelRequest) and message.instructions is not None:
        projected["instructions"] = message.instructions
    for part in dumped["parts"]:
        kind = part["part_kind"]
        if kind in ("system-prompt", "instruction", "user-prompt", "text", "thinking"):
            projected["parts"].append({"kind": kind, "content": part["content"]})
        elif kind == "tool-call":
            item: _ProjectionPart = {
                "kind": kind,
                "name": part["tool_name"],
                "args": _arguments(part["args"]),
                "id": part["tool_call_id"],
            }
            details = part.get("provider_details") or {}
            raw = details.get("ds4py.raw")
            if (message.metadata or {}).get("ds4.source") == "sampled" and isinstance(raw, str):
                item["raw"] = raw
            projected["parts"].append(item)
        elif kind in {"tool-return", "retry-prompt"}:
            projected["parts"].append(
                {
                    "kind": kind,
                    "name": part["tool_name"],
                    "content": stable_text(part["content"]),
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


def _content_hash(message: ModelMessage) -> str:
    return _digest(_projection(message))


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
        key = (message.metadata or {}).get("ds4.key")
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
