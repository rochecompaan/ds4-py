"""One pure request transition, applied to either live or scratch history."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import Model

from ds4py.binding_provenance import BoundMessage, _validate
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.transcript import Segment, Transcript, TranscriptReplacement, TranscriptView
from ds4py.errors import EngineError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import ModelRequestParameters

    from ds4py.binding import MessageBinder, PreparedMessages
    from ds4py.capabilities.system_notes import DS4SystemNotes, NoteState
    from ds4py.engine.session import Engine
    from ds4py.engine.tool_prompt import ToolDefinition

type Signature = tuple[str, str | None, int]
type SavedNotes = tuple[tuple[str, tuple[tuple[str, str | None], ...]], ...]


def request_instructions(messages: Sequence[ModelMessage], params: ModelRequestParameters) -> str | None:
    """Resolve the same effective instructions for projection and preparation."""
    parts = Model._get_instruction_parts(messages, params)
    if not bool(parts):
        return None
    return "\n\n".join(part.content for part in parts) or None


@dataclass(frozen=True)
class BindingChange:
    kind: str
    start: int
    continuation: bool


class ScratchSession:
    """Rendering target with no native session/KV state."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._transcript = Transcript()
        self.transcript = TranscriptView(self._transcript)

    def start_chat(self, tools: Iterable[ToolDefinition] = (), instructions: str | None = None) -> None:
        for segment in PromptBuilder(self._engine).begin(tools, instructions):
            self._transcript.append(segment.tokens, segment.kind)
        self._transcript.set_chat()

    def append_tokens(self, tokens: Sequence[int], kind: str, meta: Mapping[str, object] | None = None) -> Segment:
        return self._transcript.append(tokens, kind, meta)


@dataclass(frozen=True)
class BindingBoundary:
    """An atomic append group and the state valid only after its complete apply.

    ``start_messages``/``end_messages`` bound the newly committed candidate
    range, so publication copies only those targets and earlier prefixes are
    never rewritten.
    """

    appends: TranscriptReplacement
    records: tuple[BoundMessage, ...]
    bound_messages: tuple[ModelMessage, ...]
    notes: NoteState | None
    restored_notes: SavedNotes
    start_messages: int
    end_messages: int
    retire_paused: bool = False


@dataclass(frozen=True)
class BindingPlan:
    change: BindingChange
    messages: tuple[ModelMessage, ...]
    initial: BindingBoundary
    boundaries: tuple[BindingBoundary, ...]
    signature: Signature | None
    reset_chat: bool
    truncate_to: int | None
    definitions: tuple[ToolDefinition, ...]
    instructions: str | None
    projected_tokens: int


def _replacement(scratch: ScratchSession, start: int = 0) -> TranscriptReplacement:
    return TranscriptReplacement(
        scratch.transcript.tokens[start:],
        tuple(
            replace(s, start=s.start - start, end=s.end - start)
            for s in scratch.transcript.segments
            if s.start >= start
        ),
    )


@dataclass(frozen=True)
class BindingConfig:
    definitions: tuple[ToolDefinition, ...]
    instructions: str | None
    signature: Signature | None


def configured_plan(
    binder: MessageBinder,
    prepared: PreparedMessages,
    config: BindingConfig,
    provider: DS4SystemNotes | None,
) -> BindingPlan:
    changed = binder._signature != config.signature or not binder.session.transcript.chat
    if binder._restored and binder._restored_preamble(config.definitions, config.instructions):
        changed = False
    return _transition(binder, prepared, config, provider, system_changed=changed)


def require_bound_chat(binder: MessageBinder) -> None:
    """Reject bound projection before any request preprocessing runs."""
    if not binder.session.transcript.chat:
        raise EngineError("bound request requires an initialized chat")


def bound_plan(binder: MessageBinder, prepared: PreparedMessages) -> BindingPlan:
    require_bound_chat(binder)
    return _transition(
        binder, prepared, BindingConfig((), None, binder._signature), binder.note_provider, system_changed=False
    )


def _transition(  # noqa: PLR0912, PLR0915 - One traversal owns every binding transition.
    binder: MessageBinder,
    prepared: PreparedMessages,
    config: BindingConfig,
    provider: DS4SystemNotes | None,
    *,
    system_changed: bool,
) -> BindingPlan:
    candidates = list(prepared.candidates)
    _validate(candidates)
    keyed = binder._keyed(candidates)
    preview = provider.preview() if provider is not None else None
    restored = deepcopy(binder._restored_notes)
    transcript = binder.session.transcript
    scratch = ScratchSession(binder.session.engine)
    common = 0
    truncate = None
    retire = False
    if system_changed:
        binder._renderer.preflight(candidates)
        start = 0
        truncate = 0 if len(transcript) or transcript.chat else None
        scratch.start_chat(config.definitions, config.instructions)
        kind = "rewrite" if binder._signature is not None else "append"
    else:
        records = binder.records()
        while (
            common < len(candidates)
            and common < len(records)
            and keyed[common] == (records[common].key, records[common].content_hash)
        ):
            common += 1
        unchanged = common == len(candidates) == len(records)
        if unchanged and preview is not None and binder._trailing_request(candidates) and preview.pending(candidates):
            common -= 1
            unchanged = False
        binder._renderer.preflight(candidates[common:])
        if unchanged:
            start = len(transcript)
            kind = "unchanged"
        elif common < len(records):
            start = truncate = records[common].start
            kind = "rewrite"
        else:
            start = len(transcript)
            kind = "append"
            retire = binder.session.paused
        end = truncate if truncate is not None else len(transcript)
        scratch._transcript.replace(transcript.tokens, transcript.segments)
        scratch._transcript.truncate(end)
    if preview is not None and any(
        s.kind == "note" and s.end > (truncate if truncate is not None else len(transcript))
        for s in transcript.segments
    ):
        preview.resync_from_segments(scratch.transcript.segments)
    records = list(binder.records()[:common])
    bound = deepcopy(binder._bound_messages[:common])

    published = 0

    def boundary(start: int, *, retire: bool = False) -> BindingBoundary:
        nonlocal published
        appended = BindingBoundary(
            _replacement(scratch, start),
            tuple(records),
            tuple(bound),
            preview.checkpoint() if preview is not None else None,
            tuple((key, tuple(notes)) for key, notes in restored.items()),
            published,
            len(records),
            retire,
        )
        published = len(records)
        return appended

    initial = boundary(len(scratch.transcript))
    boundaries = []
    if retire:
        previous = records[-1]
        tokens = PromptBuilder(binder.session.engine).append_assistant_end()
        if tokens:
            scratch.append_tokens(
                tokens,
                "eos",
                {
                    "key": previous.key,
                    "content_hash": previous.content_hash,
                    "source": "sampled",
                },
            )
        records[-1] = replace(previous, end=len(scratch.transcript))
        boundaries.append(boundary(start, retire=True))
    for index in range(common, len(candidates)):
        message = candidates[index]
        key, content_hash = keyed[index]
        boundary_start = len(scratch.transcript)
        if isinstance(message, ModelRequest):
            notes = list(restored.pop(key, ()))
            if preview is not None:
                notes.extend(preview.notes(candidates[: index + 1]))
            for text, subtype in notes:
                tokens = PromptBuilder(binder.session.engine).message(scratch.transcript.tokens, "system", "\n" + text)
                meta = {"text": text}
                if bool(subtype):
                    meta["subtype"] = subtype
                scratch.append_tokens(tokens, "note", meta)
        message_start, end = binder._renderer.append(scratch, message, key, content_hash)
        records.append(BoundMessage(key, content_hash, message_start, end))
        bound.append(message)
        boundaries.append(boundary(boundary_start))
    return BindingPlan(
        BindingChange(kind, start, binder._continuation(candidates)),
        tuple(candidates),
        initial,
        tuple(boundaries),
        config.signature,
        system_changed,
        truncate,
        config.definitions,
        config.instructions,
        len(scratch.transcript) + len(PromptBuilder(binder.session.engine).assistant_prefix(scratch.transcript.tokens)),
    )
