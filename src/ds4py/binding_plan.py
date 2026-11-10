"""One pure request transition, applied to either live or scratch history."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelRequest, RetryPromptPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import Model

from ds4py.binding_provenance import BoundMessage, _validate, keyed_messages
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.transcript import Segment, Transcript, TranscriptReplacement, TranscriptView

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import ModelRequestParameters

    from ds4py.capabilities.system_notes import DS4SystemNotes, NotePreview, NoteState
    from ds4py.engine.session import Engine, Session
    from ds4py.engine.tool_prompt import ToolDefinition
    from ds4py.message_rendering import MessageRenderer

type Signature = tuple[str, str | None, int]
type SavedNotes = tuple[tuple[str, tuple[tuple[str, str | None], ...]], ...]

_RESERVE_CONTINUATION = (ToolReturnPart, RetryPromptPart)


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
class PreparedMessages:
    """One canonicalized request: publication targets and planning copies.

    ``targets`` receives published keys and is the response view handed back to
    the harness. ``candidates`` are independent copies because the plan writes
    recovered keys and rendered notes without mutating the request targets.
    """

    targets: tuple[ModelMessage, ...]
    candidates: tuple[ModelMessage, ...]


@dataclass(frozen=True)
class BindingState:
    """One immutable observation of the binder state used for planning."""

    records: tuple[BoundMessage, ...]
    bound_messages: tuple[ModelMessage, ...]
    signature: Signature | None
    restored: bool
    restored_notes: Mapping[str, Sequence[tuple[str, str | None]]]
    session: Session
    renderer: MessageRenderer


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


@dataclass(frozen=True)
class BindingConfig:
    definitions: tuple[ToolDefinition, ...]
    instructions: str | None
    signature: Signature | None


@dataclass(frozen=True)
class _SelectedPrefix:
    scratch: ScratchSession
    common: int
    start: int
    truncate: int | None
    retire: bool
    kind: str


def _replacement(scratch: ScratchSession, start: int = 0) -> TranscriptReplacement:
    return TranscriptReplacement(scratch.transcript.tokens, scratch.transcript.segments).slice(start)


class _BoundaryBuilder:
    """Build consecutive boundaries over one accumulating scratch transcript."""

    def __init__(
        self,
        scratch: ScratchSession,
        records: list[BoundMessage],
        bound: list[ModelMessage],
        preview: NotePreview | None,
        restored: dict[str, list[tuple[str, str | None]]],
    ) -> None:
        self.scratch = scratch
        self.records = records
        self.bound = bound
        self.preview = preview
        self.restored = restored
        self._published = 0

    def initial(self) -> BindingBoundary:
        return self.build(len(self.scratch.transcript))

    def build(self, start: int, *, retire: bool = False) -> BindingBoundary:
        boundary = BindingBoundary(
            _replacement(self.scratch, start),
            tuple(self.records),
            tuple(self.bound),
            self.preview.checkpoint() if self.preview is not None else None,
            tuple((key, tuple(notes)) for key, notes in self.restored.items()),
            self._published,
            len(self.records),
            retire,
        )
        self._published = len(self.records)
        return boundary


def _trailing_request(messages: Sequence[ModelMessage]) -> bool:
    if not messages or not isinstance(messages[-1], ModelRequest):
        return False
    return any(isinstance(part, (UserPromptPart, ToolReturnPart, RetryPromptPart)) for part in messages[-1].parts)


def _continuation(messages: Sequence[ModelMessage]) -> bool:
    return bool(
        messages
        and isinstance(messages[-1], ModelRequest)
        and messages[-1].parts
        and isinstance(messages[-1].parts[-1], _RESERVE_CONTINUATION)
    )


def _system_prefix(
    state: BindingState,
    config: BindingConfig,
    candidates: Sequence[ModelMessage],
    scratch: ScratchSession,
) -> _SelectedPrefix:
    transcript = state.session.transcript
    state.renderer.preflight(candidates)
    scratch.start_chat(config.definitions, config.instructions)
    truncate = 0 if len(transcript) or transcript.chat else None
    kind = "rewrite" if state.signature is not None else "append"
    return _SelectedPrefix(scratch, 0, 0, truncate, retire=False, kind=kind)


def _incremental_prefix(
    state: BindingState,
    scratch: ScratchSession,
    candidates: Sequence[ModelMessage],
    keyed: Sequence[tuple[str, str]],
    preview: NotePreview | None,
) -> _SelectedPrefix:
    transcript = state.session.transcript
    records = state.records
    common = 0
    while (
        common < len(candidates)
        and common < len(records)
        and keyed[common] == (records[common].key, records[common].content_hash)
    ):
        common += 1
    unchanged = common == len(candidates) == len(records)
    if unchanged and preview is not None and _trailing_request(candidates) and preview.pending(candidates):
        common -= 1
        unchanged = False
    state.renderer.preflight(candidates[common:])
    if unchanged:
        prefix = _SelectedPrefix(scratch, common, len(transcript), None, retire=False, kind="unchanged")
    elif common < len(records):
        prefix = _SelectedPrefix(
            scratch,
            common,
            records[common].start,
            records[common].start,
            retire=False,
            kind="rewrite",
        )
    else:
        prefix = _SelectedPrefix(
            scratch,
            common,
            len(transcript),
            None,
            retire=state.session.paused,
            kind="append",
        )
    end = len(transcript) if prefix.truncate is None else prefix.truncate
    scratch._transcript.replace(transcript.tokens, transcript.segments)
    scratch._transcript.truncate(end)
    return prefix


def _append_notes(
    state: BindingState,
    builder: _BoundaryBuilder,
    key: str,
    preceding: Sequence[ModelMessage],
) -> None:
    notes = list(builder.restored.pop(key, ()))
    if builder.preview is not None:
        notes.extend(builder.preview.notes(preceding))
    for text, subtype in notes:
        tokens = PromptBuilder(state.session.engine).message(builder.scratch.transcript.tokens, "system", "\n" + text)
        meta = {"text": text}
        if bool(subtype):
            meta["subtype"] = subtype
        builder.scratch.append_tokens(tokens, "note", meta)


def _append_message(
    state: BindingState,
    builder: _BoundaryBuilder,
    candidates: Sequence[ModelMessage],
    keyed: Sequence[tuple[str, str]],
    index: int,
) -> BindingBoundary:
    message = candidates[index]
    key, content_hash = keyed[index]
    boundary_start = len(builder.scratch.transcript)
    if isinstance(message, ModelRequest):
        _append_notes(state, builder, key, candidates[: index + 1])
    message_start, end = state.renderer.append(builder.scratch, message, key, content_hash)
    builder.records.append(BoundMessage(key, content_hash, message_start, end))
    builder.bound.append(message)
    return builder.build(boundary_start)


def _retire_response(builder: _BoundaryBuilder, engine: Engine, previous: BoundMessage) -> BoundMessage:
    tokens = PromptBuilder(engine).append_assistant_end()
    if tokens:
        builder.scratch.append_tokens(
            tokens,
            "eos",
            {
                "key": previous.key,
                "content_hash": previous.content_hash,
                "source": "sampled",
            },
        )
    return replace(previous, end=len(builder.scratch.transcript))


def plan_transition(
    state: BindingState,
    prepared: PreparedMessages,
    config: BindingConfig,
    provider: DS4SystemNotes | None,
    *,
    system_changed: bool,
) -> BindingPlan:
    """Plan one transition from an immutable binder snapshot."""
    candidates = list(prepared.candidates)
    _validate(candidates)
    keyed = keyed_messages(candidates, state.records)
    preview = provider.preview() if provider is not None else None
    restored = {key: list(notes) for key, notes in state.restored_notes.items()}
    scratch = ScratchSession(state.session.engine)
    if system_changed:
        prefix = _system_prefix(state, config, candidates, scratch)
    else:
        prefix = _incremental_prefix(state, scratch, candidates, keyed, preview)
    transcript = state.session.transcript
    end = len(transcript) if prefix.truncate is None else prefix.truncate
    if preview is not None and any(s.kind == "note" and s.end > end for s in transcript.segments):
        preview.resync_from_segments(prefix.scratch.transcript.segments)
    records = list(state.records[: prefix.common])
    bound = list(deepcopy(state.bound_messages[: prefix.common]))
    builder = _BoundaryBuilder(prefix.scratch, records, bound, preview, restored)
    initial = builder.initial()
    boundaries: list[BindingBoundary] = []
    if prefix.retire:
        records[-1] = _retire_response(builder, state.session.engine, records[-1])
        boundaries.append(builder.build(prefix.start, retire=True))
    for index in range(prefix.common, len(candidates)):
        boundaries.append(_append_message(state, builder, candidates, keyed, index))
    engine = state.session.engine
    projected = len(prefix.scratch.transcript) + len(
        PromptBuilder(engine).assistant_prefix(prefix.scratch.transcript.tokens)
    )
    return BindingPlan(
        BindingChange(prefix.kind, prefix.start, _continuation(candidates)),
        tuple(candidates),
        initial,
        tuple(boundaries),
        config.signature,
        system_changed,
        prefix.truncate,
        config.definitions,
        config.instructions,
        projected,
    )
