"""Bind harness message history to the engine's single token transcript."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypedDict
from uuid import uuid4

from pydantic_ai.messages import (
    InstructionBaselinePart,
    InstructionDeltaPart,
    InstructionPart,
    ModelMessage,
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

from ds4py.engine import chat as chat_ops
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.tool_prompt import ToolDefinition
from ds4py.engine.transcript import Segment, Transcript, TranscriptView
from ds4py.errors import EngineError
from ds4py.message_rendering import MessageRenderer, stable_text

if TYPE_CHECKING:
    from ds4py.capabilities.system_notes import DS4SystemNotes
    from ds4py.engine.session import Engine, Session


class _ProjectionPart(TypedDict, total=False):
    kind: str
    content: str
    name: str
    args: dict[str, Any]
    id: str
    raw: str


class _MessageProjection(TypedDict):
    kind: str
    parts: list[_ProjectionPart]
    instructions: NotRequired[str]


class _Provenance(Protocol):
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def kind(self) -> str: ...
    @property
    def key(self) -> str: ...
    @property
    def content_hash(self) -> str: ...
    @property
    def source(self) -> str: ...


@dataclass(frozen=True)
class BindingChange:
    kind: str
    start: int
    continuation: bool


@dataclass(frozen=True)
class SegmentProvenance:
    start: int
    end: int
    kind: str
    key: str
    content_hash: str
    source: str


@dataclass(frozen=True)
class _BoundMessage:
    key: str
    content_hash: str
    start: int
    end: int


class _ScratchSession:
    """Rendering target with no native session/KV state."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._transcript = Transcript()
        self.transcript = TranscriptView(self._transcript)

    def start_chat(
        self, tools: Iterable[ToolDefinition] = (), instructions: str | None = None,
    ) -> None:
        for segment in PromptBuilder(self._engine).begin(tools, instructions):
            self._transcript.append(segment.tokens, segment.kind)
        self._transcript.set_chat()

    def append_tokens(
        self, tokens: Sequence[int], kind: str, meta: Mapping[str, object] | None = None,
    ) -> Segment:
        return self._transcript.append(tokens, kind, meta)


def _definitions(tools: Iterable[Any]) -> tuple[ToolDefinition, ...]:
    return tuple(
        ToolDefinition(
            tool.name,
            tool.description or "",
            tool.parameters_json_schema if hasattr(tool, "parameters_json_schema")
            else tool.parameters,
        )
        for tool in tools
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tool_digest(tools: Iterable[ToolDefinition]) -> str:
    return _digest([
        {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
        for tool in tools
    ])


def _ensure_key(message: ModelMessage) -> str:
    metadata = dict(message.metadata or {})
    key = metadata.get("ds4.key")
    if not key:
        key = uuid4().hex
        metadata["ds4.key"] = key
    message.metadata = metadata
    return key


def _arguments(args: str | Mapping[str, Any] | None) -> dict[str, Any]:
    value: Any = json.loads(args) if isinstance(args, str) else args
    return dict(value or {})


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
        elif kind == "tool-return":
            projected["parts"].append({
                "kind": kind,
                "name": part["tool_name"],
                "content": stable_text(part["content"]),
                "id": part["tool_call_id"],
            })
        elif kind == "retry-prompt":
            projected["parts"].append({
                "kind": kind,
                "name": part["tool_name"],
                "content": stable_text(part["content"]),
                "id": part["tool_call_id"],
            })
    return projected


def _content_hash(message: ModelMessage) -> str:
    return _digest(_projection(message))


def _validate(messages: Iterable[ModelMessage]) -> None:
    for message in messages:
        parts = message.parts
        for part in parts:
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                raise EngineError("image input requires M5 vision support")
            if isinstance(message, ModelRequest) and not isinstance(part, (
                SystemPromptPart, InstructionPart, InstructionBaselinePart,
                InstructionDeltaPart, UserPromptPart, ToolReturnPart, RetryPromptPart,
            )):
                raise EngineError(f"unsupported harness request part: {type(part).__name__}")
            if isinstance(message, ModelResponse) and not isinstance(
                part, (TextPart, ThinkingPart, ToolCallPart)
            ):
                raise EngineError(f"unsupported harness response part: {type(part).__name__}")


class MessageBinder:
    """Longest-valid-prefix binding keyed by persistent message key and content hash.

    The binder belongs to one session's consuming thread. ``prepare`` requires
    an idle native round. ``bind_response`` runs during response finalization,
    before that round closes.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self._renderer = MessageRenderer(session.engine)
        self.note_provider = None
        self._records: list[_BoundMessage] = []
        self._bound_messages: list[ModelMessage] = []
        self._signature = None
        self._restored = False

    def forget(self) -> None:
        """Drop binding state so the next request renders the message list fresh."""
        self._records = []
        self._bound_messages = []
        self._signature = None
        self._restored = False

    def adopt(self, messages: Sequence[ModelMessage]) -> list[ModelMessage]:
        """Keep final harness messages only where they match the bound transcript."""
        candidates = list(messages)
        _validate(candidates)
        adopted = [
            candidate
            if index >= len(self._records)
            or _content_hash(candidate) == self._records[index].content_hash
            else deepcopy(self._bound_messages[index])
            for index, candidate in enumerate(candidates)
        ]
        self._keyed(adopted)
        return adopted

    @staticmethod
    def _installed_segment(provenance: _Provenance) -> Segment:
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
            kind = "note"
        return Segment(provenance.start, provenance.end, kind, MappingProxyType(meta))

    def _restored_preamble(
        self, definitions: Sequence[ToolDefinition], instructions: str | None,
    ) -> bool:
        """True when the loaded transcript still starts with the current preamble."""
        if not self.session.transcript.chat:
            return False
        segments = PromptBuilder(self.session.engine).begin(definitions, instructions)
        preamble = tuple(token for segment in segments for token in segment.tokens)
        loaded = self.session.transcript.tokens
        return bool(preamble) and len(preamble) <= len(loaded) and loaded[:len(preamble)] == preamble

    def validate_restore(
        self, messages: Sequence[ModelMessage], segments: Sequence[_Provenance],
    ) -> None:
        """Validate saved history without installing a native transcript."""
        self._restore_state(messages, segments, loaded=None)

    def restore(
        self, messages: Sequence[ModelMessage], segments: Sequence[_Provenance],
    ) -> None:
        """Bind a loaded native payload to a saved message history.

        ``segments`` is a save record's provenance (values with ``start``,
        ``end``, ``kind``, ``key``, ``content_hash`` and ``source``). The
        transcript already holds the loaded tokens, so this installs the saved
        ranges and rebuilds the prefix records without rendering anything.
        Snapshot messages are matched to keyed ranges in order and by content
        hash: a record and its snapshot are written by one save, so the order is
        fixed, and the hash proves the snapshot still describes the tokens.
        """
        installed, records = self._restore_state(
            messages, segments, loaded=len(self.session.transcript),
        )
        self.session._transcript.install(installed)
        self._records = records
        self._bound_messages = deepcopy(list(messages))
        self._signature = None
        self._restored = True

    def _restore_state(
        self,
        messages: Sequence[ModelMessage],
        segments: Sequence[_Provenance],
        loaded: int | None,
    ) -> tuple[list[Segment], list[_BoundMessage]]:
        messages = list(messages)
        _validate(messages)
        installed = []
        ranges: dict[str, list[int]] = {}
        keys: list[str] = []
        hashes: dict[str, str] = {}
        expected = 0
        for provenance in segments:
            if provenance.start != expected or provenance.end <= provenance.start:
                raise EngineError(
                    f"save record segment {provenance.start}:{provenance.end} "
                    "does not continue the loaded transcript"
                )
            expected = provenance.end
            installed.append(self._installed_segment(provenance))
            if not provenance.key:
                continue
            if provenance.key in ranges:
                if provenance.content_hash != hashes[provenance.key]:
                    raise EngineError(
                        f"save record ranges for message key {provenance.key!r} "
                        "disagree on the content hash"
                    )
                ranges[provenance.key][1] = provenance.end
            else:
                ranges[provenance.key] = [provenance.start, provenance.end]
                keys.append(provenance.key)
                hashes[provenance.key] = provenance.content_hash
        if loaded is not None and expected != loaded:
            raise EngineError(
                f"save record covers {expected} tokens, the loaded payload has {loaded}"
            )

        # A response's range starts at its assistant prefix, which has no key.
        pending_prefix = None
        for provenance in segments:
            if not provenance.key:
                if provenance.kind == "assistant_prefix":
                    pending_prefix = provenance.start
                continue
            if pending_prefix is not None and pending_prefix < ranges[provenance.key][0]:
                ranges[provenance.key][0] = pending_prefix
            pending_prefix = None

        if len(keys) != len(messages):
            raise EngineError(
                f"the save record holds {len(keys)} keyed messages, "
                f"the snapshot has {len(messages)}"
            )
        records = []
        for message, key in zip(messages, keys):
            metadata = dict(message.metadata or {})
            existing = metadata.get("ds4.key")
            if existing != key:
                raise EngineError(
                    f"the snapshot message key {existing!r} is not the save "
                    f"record key {key!r}"
                )
            content_hash = _content_hash(message)
            if content_hash != hashes[key]:
                raise EngineError(
                    f"the save record content hash for message key {key!r} "
                    "does not match the snapshot"
                )
            records.append(_BoundMessage(key, content_hash, ranges[key][0], ranges[key][1]))

        return installed, records

    @staticmethod
    def _trailing_request(messages: Sequence[ModelMessage]) -> bool:
        if not messages or not isinstance(messages[-1], ModelRequest):
            return False
        return any(
            isinstance(part, (UserPromptPart, ToolReturnPart, RetryPromptPart))
            for part in messages[-1].parts
        )

    @staticmethod
    def _continuation(messages: Sequence[ModelMessage]) -> bool:
        return bool(
            messages and isinstance(messages[-1], ModelRequest)
            and messages[-1].parts
            and isinstance(messages[-1].parts[-1], (ToolReturnPart, RetryPromptPart))
        )

    def _retire_paused(self) -> None:
        if self.session._paused_stream is None:
            return
        previous = self._records[-1]
        start = len(self.session.transcript)
        chat_ops.retire_paused(self.session)
        end = len(self.session.transcript)
        self.session._transcript.annotate(
            start,
            kinds=("eos",),
            meta={
                "key": previous.key,
                "content_hash": previous.content_hash,
                "source": "sampled",
            },
        )
        self._records[-1] = _BoundMessage(
            previous.key, previous.content_hash, previous.start, end,
        )

    def _sync_notes(self, provider: DS4SystemNotes | None) -> None:
        sync = getattr(provider, "resync_from_segments", None)
        if sync is None:
            sync = getattr(provider, "sync_from_segments", None)
        if sync is not None:
            sync(self.session.transcript.segments)

    def _keyed(self, messages: Sequence[ModelMessage]) -> list[tuple[str, str]]:
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
            key = (message.metadata or {}).get("ds4.key")
            if (
                key is None and index < len(self._records)
                and self._records[index].content_hash == content_hash
            ):
                key = self._records[index].key
                metadata = dict(message.metadata or {})
                metadata["ds4.key"] = key
                message.metadata = metadata
            else:
                key = _ensure_key(message)
            keyed.append((key, content_hash))
        return keyed

    def prepare(
        self,
        messages: Sequence[ModelMessage],
        *,
        tools: Iterable[Any] = (),
        instructions: str | None = None,
        note_provider: DS4SystemNotes | None = None,
    ) -> BindingChange:
        messages = list(messages)
        _validate(messages)
        keyed = self._keyed(messages)
        definitions = _definitions(tools)
        signature = (_tool_digest(definitions), instructions, int(self.session.engine.think))
        had_binding = self._signature is not None
        system_changed = self._signature != signature or not self.session.transcript.chat
        provider = note_provider if note_provider is not None else self.note_provider
        if self._restored:
            # A restore adopts a loaded KV without knowing the request that will
            # extend it; adopt the signature only when the preamble really is the
            # one already loaded, otherwise fall back to a rewrite.
            if system_changed and self._restored_preamble(definitions, instructions):
                self._signature = signature
                system_changed = False
            self._restored = False

        if system_changed:
            self._renderer.preflight(messages)
            start = 0
            removed_notes = any(
                segment.kind == "note" for segment in self.session.transcript.segments
            )
            if len(self.session.transcript):
                self.session.truncate(0)
            self._records = []
            self._bound_messages = []
            self.session.start_chat(definitions, instructions)
            self._signature = signature
            if removed_notes:
                self._sync_notes(provider)
            common = 0
            kind = "rewrite" if had_binding else "append"
        else:
            common = 0
            while (
                common < len(messages) and common < len(self._records)
                and keyed[common] == (self._records[common].key, self._records[common].content_hash)
            ):
                common += 1
            if common == len(messages) == len(self._records):
                pending = getattr(provider, "pending", None)
                if not (
                    pending is not None
                    and self._trailing_request(messages)
                    and pending(messages)
                ):
                    return BindingChange(
                        "unchanged", len(self.session.transcript), self._continuation(messages),
                    )
                # A required note is pending; rewrite the trailing request so the
                # note lands at its boundary instead of returning unchanged.
                common = len(messages) - 1
            self._renderer.preflight(messages[common:])
            if common < len(self._records):
                start = self._records[common].start
                removed_notes = any(
                    segment.kind == "note" and segment.end > start
                    for segment in self.session.transcript.segments
                )
                self.session.truncate(start)
                self._records = self._records[:common]
                self._bound_messages = self._bound_messages[:common]
                kind = "rewrite"
                if removed_notes:
                    self._sync_notes(provider)
            else:
                self._retire_paused()
                start = len(self.session.transcript)
                kind = "append"

        for offset, (message, (key, content_hash)) in enumerate(
            zip(messages[common:], keyed[common:]),
        ):
            if provider is not None and isinstance(message, ModelRequest):
                for text, subtype in provider.notes(messages[:common + offset + 1]):
                    chat_ops.append_system_note(self.session, text, subtype)
            message_start, end = self._renderer.append(
                self.session, message, key, content_hash,
            )
            self._records.append(_BoundMessage(key, content_hash, message_start, end))
            self._bound_messages.append(deepcopy(message))
        return BindingChange(kind, start, self._continuation(messages))

    def count_tokens(
        self,
        messages: Sequence[ModelMessage],
        *,
        tools: Iterable[Any] = (),
        instructions: str | None = None,
    ) -> int:
        messages = list(messages)
        _validate(messages)
        scratch = _ScratchSession(self.session.engine)
        scratch.start_chat(_definitions(tools), instructions)
        for message in messages:
            key = (message.metadata or {}).get("ds4.key", "")
            self._renderer.append(scratch, message, key, _content_hash(message))
        prefix = PromptBuilder(self.session.engine).assistant_prefix(scratch.transcript.tokens)
        if prefix:
            scratch.append_tokens(prefix, "assistant_prefix")
        return len(scratch.transcript)

    def bind_response(self, response: ModelResponse) -> None:
        key = _ensure_key(response)
        assert response.metadata is not None  # _ensure_key initializes it
        response.metadata["ds4.source"] = "sampled"
        content_hash = _content_hash(response)
        segments = self.session.transcript.segments
        prefix_index = max(
            (index for index, segment in enumerate(segments) if segment.kind == "assistant_prefix"),
            default=None,
        )
        trailing = [
            segment for segment in segments[(prefix_index + 1) if prefix_index is not None else 0:]
            if segment.kind in ("assistant", "eos")
        ]
        sampled_start = trailing[0].start if trailing else len(self.session.transcript)
        self.session._transcript.annotate(
            sampled_start, kinds=("assistant", "eos"),
            meta={"key": key, "content_hash": content_hash, "source": "sampled"},
        )
        start = segments[prefix_index].start if prefix_index is not None else sampled_start
        record = _BoundMessage(key, content_hash, start, len(self.session.transcript))
        if self._records and self._records[-1].key == key and self._records[-1].start == start:
            self._records[-1] = record
            self._bound_messages[-1] = deepcopy(response)
        else:
            self._records.append(record)
            self._bound_messages.append(deepcopy(response))

    def export_segments(self) -> tuple[SegmentProvenance, ...]:
        exported = []
        for segment in self.session.transcript.segments:
            kind = segment.kind
            if kind == "note":
                subtype = segment.meta.get("subtype")
                kind = f"system_note:{subtype}" if subtype else "system_note"
            exported.append(SegmentProvenance(
                segment.start,
                segment.end,
                kind,
                str(segment.meta.get("key", "")),
                str(segment.meta.get("content_hash", "")),
                str(segment.meta.get("source", "rendered")),
            ))
        return tuple(exported)
