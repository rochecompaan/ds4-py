"""Bind harness message history to the engine's single token transcript."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolReturnPart,
    UserPromptPart,
)

from ds4py._json import is_string_key_mapping
from ds4py.binding_provenance import (
    BoundMessage,
    _arguments,  # noqa: F401 - Re-exported for tool rendering tests.
    _content_hash,
    _digest,
    _key_recovery_hash,
    _validate,
    prepare_provenance,
)
from ds4py.binding_request import count_bound_request
from ds4py.engine import chat as chat_ops
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.tool_prompt import ToolDefinition
from ds4py.engine.transcript import Segment, Transcript, TranscriptView
from ds4py.message_rendering import MessageRenderer

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ds4py.binding_provenance import PreparedProvenance, _Provenance
    from ds4py.capabilities.system_notes import DS4SystemNotes
    from ds4py.engine.session import Engine, Session


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
    text: str | None = None


class _ScratchSession:
    """Rendering target with no native session/KV state."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._transcript = Transcript()
        self.transcript = TranscriptView(self._transcript)

    def start_chat(
        self,
        tools: Iterable[ToolDefinition] = (),
        instructions: str | None = None,
    ) -> None:
        for segment in PromptBuilder(self._engine).begin(tools, instructions):
            self._transcript.append(segment.tokens, segment.kind)
        self._transcript.set_chat()

    def append_tokens(
        self,
        tokens: Sequence[int],
        kind: str,
        meta: Mapping[str, object] | None = None,
    ) -> Segment:
        return self._transcript.append(tokens, kind, meta)


def _definitions(tools: Iterable[object]) -> tuple[ToolDefinition, ...]:
    definitions = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if not isinstance(name, str):
            raise TypeError("tool name must be a string")
        description = getattr(tool, "description", None)
        if description is not None and not isinstance(description, str):
            raise TypeError("tool description must be a string or None")
        if hasattr(tool, "parameters_json_schema"):
            parameters = tool.parameters_json_schema
        else:
            parameters = getattr(tool, "parameters", None)
        if not is_string_key_mapping(parameters):
            raise TypeError("tool parameters must be a string-keyed mapping")
        definitions.append(ToolDefinition(name, description or "", parameters))
    return tuple(definitions)


def _tool_digest(tools: Iterable[ToolDefinition]) -> str:
    return _digest(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]
    )


def _ensure_key(message: ModelMessage) -> str:
    metadata = dict(message.metadata or {})
    key = metadata.get("ds4.key")
    if key is not None and not isinstance(key, str):
        raise TypeError("message ds4.key must be a string or None")
    if key is None or key == "":
        key = uuid4().hex
        metadata["ds4.key"] = key
    message.metadata = metadata
    return key


class MessageBinder:
    """Longest-valid-prefix binding keyed by persistent message key and content hash.

    The binder belongs to one session's consuming thread. ``prepare`` requires
    an idle native round. ``bind_response`` runs during response finalization,
    before that round closes.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self._renderer = MessageRenderer(session.engine)
        self.note_provider: DS4SystemNotes | None = None
        self._records: list[BoundMessage] = []
        self._bound_messages: list[ModelMessage] = []
        self._signature = None
        self._restored = False
        self._restored_notes: dict[str, list[tuple[str, str | None]]] = {}

    def forget(self) -> None:
        """Drop binding state so the next request renders the message list fresh."""
        self._records = []
        self._bound_messages = []
        self._signature = None
        self._restored = False
        self._restored_notes = {}

    def restore_notes(self, segments: Sequence[_Provenance]) -> None:
        """Schedule exact saved notes before the message keys that followed them."""
        restored: dict[str, list[tuple[str, str | None]]] = {}
        pending: list[tuple[str, str | None]] = []
        for segment in segments:
            if segment.kind == "system_note" or segment.kind.startswith("system_note:"):
                text = getattr(segment, "text", None)
                if not isinstance(text, str):
                    raise ValueError("saved note text is required for rendered restore")
                subtype = segment.kind.partition(":")[2] or None
                pending.append((text, subtype))
            elif segment.key and pending:
                restored.setdefault(segment.key, []).extend(pending)
                pending = []
        if pending:
            raise ValueError("saved note has no following message boundary")
        self._restored_notes = restored

    def adopt(self, messages: Sequence[ModelMessage]) -> list[ModelMessage]:
        """Reconcile bound content by message key without truncating harness history.

        Recover exact merged request boundaries first, just as for model input.
        The incoming sequence is authoritative: it may contain history omitted
        from the model's request view. A candidate with the same stable key can
        inherit its bound counterpart's content (for example, a partial response
        that the harness later completed). An unkeyed processed copy can recover
        its key only from a unique match that differs at most by request instructions.
        """
        candidates = self.canonicalize(list(messages))
        _validate(candidates)
        bound = {record.key: index for index, record in enumerate(self._records)}
        recoverable: dict[str, list[int]] = {}
        for index, message in enumerate(self._bound_messages):
            recoverable.setdefault(_key_recovery_hash(message), []).append(index)
        adopted = []
        for candidate in candidates:
            adopted_candidate = candidate
            key = (candidate.metadata or {}).get("ds4.key")
            index = bound.get(key) if key is not None else None
            if index is None and not key:
                matches = recoverable.get(_key_recovery_hash(candidate), [])
                if len(matches) == 1:
                    index = matches[0]
            if index is not None and _content_hash(candidate) != self._records[index].content_hash:
                adopted_candidate = deepcopy(self._bound_messages[index])
            elif index is not None:
                sampled = (self._bound_messages[index].metadata or {}).get("ds4.sampled")
                if sampled is not None:
                    metadata = dict(adopted_candidate.metadata or {})
                    metadata["ds4.sampled"] = deepcopy(sampled)
                    adopted_candidate.metadata = metadata
            adopted.append(adopted_candidate)
        self._keyed(adopted)
        return adopted

    def adopt_failed(self, messages: Sequence[ModelMessage]) -> list[ModelMessage]:
        """Drop an unexecuted suffix while preserving bound trailing requests."""
        bound_keys = {record.key for record in self._records}
        adopted = self.adopt(messages)
        bound_indices = [
            index for index, message in enumerate(adopted) if (message.metadata or {}).get("ds4.key") in bound_keys
        ]
        if not bound_indices:
            return []
        return adopted[: bound_indices[-1] + 1]

    def canonicalize(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Undo only a unique exact harness merge of adjacent bound requests.

        The harness normalizes after capability hooks, discarding application
        keys when merging requests. Recover known part boundaries, never an
        unrelated or deliberately changed request-only view.
        """
        canonical: list[ModelMessage] = []
        changed = False
        for message in messages:
            matches: list[list[ModelMessage]] = []
            if isinstance(message, ModelRequest) and not (message.metadata or {}).get("ds4.key"):
                for start, first in enumerate(self._bound_messages):
                    if not isinstance(first, ModelRequest):
                        continue
                    parts = list(first.parts)
                    instructions = first.instructions
                    for end in range(start + 1, len(self._bound_messages)):
                        following = self._bound_messages[end]
                        if not isinstance(following, ModelRequest):
                            break
                        if (
                            bool(instructions)
                            and bool(following.instructions)
                            and instructions != following.instructions
                        ):
                            break
                        instructions = instructions or following.instructions
                        parts.extend(following.parts)
                        if parts == message.parts and instructions == message.instructions:
                            matches.append(self._bound_messages[start : end + 1])
            if len(matches) == 1:
                canonical.extend(deepcopy(matches[0]))
                changed = True
            else:
                canonical.append(message)
        return canonical if changed else messages

    def records(self) -> tuple[BoundMessage, ...]:
        """The keyed ranges currently bound to the transcript."""
        return tuple(self._records)

    def apply_restore(self, prepared: PreparedProvenance) -> None:
        """Assign restored state after the session installs a loaded payload."""
        self._records = list(prepared.records)
        self._bound_messages = list(prepared.bound_messages)
        self._signature = None
        self._restored = True

    def apply_compaction(self, prepared: PreparedProvenance) -> None:
        """Assign prepared records while keeping the current request signature."""
        self._records = list(prepared.records)
        self._bound_messages = list(prepared.bound_messages)

    def _restored_preamble(
        self,
        definitions: Sequence[ToolDefinition],
        instructions: str | None,
    ) -> bool:
        """True when the loaded transcript still starts with the current preamble."""
        if not self.session.transcript.chat:
            return False
        segments = PromptBuilder(self.session.engine).begin(definitions, instructions)
        preamble = tuple(token for segment in segments for token in segment.tokens)
        loaded = self.session.transcript.tokens
        return bool(preamble) and len(preamble) <= len(loaded) and loaded[: len(preamble)] == preamble

    def validate_restore(
        self,
        messages: Sequence[ModelMessage],
        segments: Sequence[_Provenance],
    ) -> None:
        """Validate saved history without installing a native transcript."""
        prepare_provenance(messages, segments, loaded=None)

    def restore(
        self,
        messages: Sequence[ModelMessage],
        segments: Sequence[_Provenance],
    ) -> None:
        """Bind a loaded native payload to a saved message history.

        ``segments`` is a save record's provenance (values with ``start``,
        ``end``, ``kind``, ``key``, ``content_hash`` and ``source``). The
        transcript already holds the loaded tokens, so preparation rebuilds the
        prefix records without rendering anything, the session installs the
        saved ranges, and the binder then assigns the prepared records. Snapshot
        messages are matched to keyed ranges in order and by content hash: a
        record and its snapshot are written by one save, so the order is fixed,
        and the hash proves the snapshot still describes the tokens.
        """
        prepared = prepare_provenance(
            messages,
            segments,
            loaded=len(self.session.transcript),
        )
        self.session.install_segments(prepared.installed)
        self.apply_restore(prepared)

    @staticmethod
    def _trailing_request(messages: Sequence[ModelMessage]) -> bool:
        if not messages or not isinstance(messages[-1], ModelRequest):
            return False
        return any(isinstance(part, (UserPromptPart, ToolReturnPart, RetryPromptPart)) for part in messages[-1].parts)

    @staticmethod
    def _continuation(messages: Sequence[ModelMessage]) -> bool:
        return bool(
            messages
            and isinstance(messages[-1], ModelRequest)
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
        self._records[-1] = BoundMessage(
            previous.key,
            previous.content_hash,
            previous.start,
            end,
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
            if key is None and index < len(self._records) and self._records[index].content_hash == content_hash:
                key = self._records[index].key
                metadata = dict(message.metadata or {})
                metadata["ds4.key"] = key
                message.metadata = metadata
            else:
                key = _ensure_key(message)
            keyed.append((key, content_hash))
        return keyed

    def prepare(  # noqa: C901, PLR0912, PLR0915 - Binding coordinates each history transition.
        self,
        messages: Sequence[ModelMessage],
        *,
        tools: Iterable[object] = (),
        instructions: str | None = None,
        note_provider: DS4SystemNotes | None = None,
    ) -> BindingChange:
        messages = list(messages)
        _validate(messages)
        keyed = self._keyed(messages)
        definitions = _definitions(tools)
        signature = (
            _tool_digest(definitions),
            instructions,
            int(self.session.engine.think),
        )
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
            removed_notes = any(segment.kind == "note" for segment in self.session.transcript.segments)
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
                common < len(messages)
                and common < len(self._records)
                and keyed[common] == (self._records[common].key, self._records[common].content_hash)
            ):
                common += 1
            if common == len(messages) == len(self._records):
                pending = getattr(provider, "pending", None)
                if not (pending is not None and self._trailing_request(messages) and pending(messages)):
                    return BindingChange(
                        "unchanged",
                        len(self.session.transcript),
                        self._continuation(messages),
                    )
                # A required note is pending; rewrite the trailing request so the
                # note lands at its boundary instead of returning unchanged.
                common = len(messages) - 1
            self._renderer.preflight(messages[common:])
            if common < len(self._records):
                start = self._records[common].start
                removed_notes = any(
                    segment.kind == "note" and segment.end > start for segment in self.session.transcript.segments
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
            if isinstance(message, ModelRequest):
                for text, subtype in self._restored_notes.get(key, ()):
                    chat_ops.append_system_note(self.session, text, subtype)
                if provider is not None:
                    for text, subtype in provider.notes(messages[: common + offset + 1]):
                        chat_ops.append_system_note(self.session, text, subtype)
            message_start, end = self._renderer.append(
                self.session,
                message,
                key,
                content_hash,
            )
            self._records.append(BoundMessage(key, content_hash, message_start, end))
            self._bound_messages.append(deepcopy(message))
        return BindingChange(kind, start, self._continuation(messages))

    def count_tokens(
        self,
        messages: Sequence[ModelMessage],
        *,
        tools: Iterable[object] = (),
        instructions: str | None = None,
        preserve_bound: bool = False,
    ) -> int:
        """Count a request, including its assistant prefix, without native sync.

        ``preserve_bound`` uses the current native preamble and exact sampled
        spans for pending results; the default renders the supplied configuration.
        """
        messages = list(messages)
        _validate(messages)
        scratch = _ScratchSession(self.session.engine)
        if preserve_bound and self._records:
            end = self._records[0].start
            for segment in self.session.transcript.segments:
                if segment.end <= end:
                    scratch.append_tokens(
                        self.session.transcript.tokens[segment.start : segment.end], segment.kind, segment.meta
                    )
            return count_bound_request(self, scratch, messages)
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
        if response.metadata is None:
            raise AssertionError("_ensure_key must initialize metadata")
        response.metadata["ds4.source"] = "sampled"
        content_hash = _content_hash(response)
        segments = self.session.transcript.segments
        prefix_index = max(
            (index for index, segment in enumerate(segments) if segment.kind == "assistant_prefix"),
            default=None,
        )
        trailing = [
            segment
            for segment in segments[(prefix_index + 1) if prefix_index is not None else 0 :]
            if segment.kind in ("assistant", "eos")
        ]
        sampled_start = trailing[0].start if trailing else len(self.session.transcript)
        response.metadata["ds4.sampled"] = {
            "content_hash": content_hash,
            "segments": [
                {
                    "kind": segment.kind,
                    "tokens": list(self.session.transcript.tokens[segment.start : segment.end]),
                }
                for segment in trailing
            ],
        }
        self.session._transcript.annotate(
            sampled_start,
            kinds=("assistant", "eos"),
            meta={"key": key, "content_hash": content_hash, "source": "sampled"},
        )
        start = segments[prefix_index].start if prefix_index is not None else sampled_start
        record = BoundMessage(key, content_hash, start, len(self.session.transcript))
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
            text = segment.meta.get("text") if kind == "note" else None
            if kind == "note":
                subtype = segment.meta.get("subtype")
                kind = f"system_note:{subtype}" if subtype else "system_note"
            exported.append(
                SegmentProvenance(
                    segment.start,
                    segment.end,
                    kind,
                    str(segment.meta.get("key", "")),
                    str(segment.meta.get("content_hash", "")),
                    str(segment.meta.get("source", "rendered")),
                    text=str(text) if text is not None else None,
                )
            )
        return tuple(exported)
