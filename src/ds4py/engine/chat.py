"""Session-level chat helpers; these functions own transcript mutations."""
from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

from ds4py.errors import EngineError
from .decode import GenerateJob
from .events import Event
from .pause import resolve_control
from .prefix import PrefixTurn
from .prompt import PromptBuilder
from .sampling import SamplingPolicy
from .stream import RoundStream
from .tool_prompt import ToolDefinition
from .transcript import Segment
from .worker import SyncJob

if TYPE_CHECKING:
    from .session import Session


def _append(
    session: "Session", tokens: Sequence[int], kind: str,
    meta: Mapping[str, object] | None = None,
) -> Segment | None:
    if not tokens:
        return None
    return session._transcript.append(tokens, kind, meta)


def start_chat(
    session: "Session", tools: Iterable[ToolDefinition] = (),
    instructions: str | None = None,
) -> None:
    if session._transcript.chat:
        raise EngineError("chat is already initialized")
    segments = PromptBuilder(session._engine).begin(tools, instructions)
    for segment in segments:
        _append(session, segment.tokens, segment.kind)
    session._transcript.set_chat()


def _start_if_empty(session: "Session") -> None:
    if session._transcript.chat:
        return
    if len(session._transcript):
        raise EngineError("chat is not initialized")
    start_chat(session)


def _require_chat(session: "Session") -> None:
    if not session._transcript.chat:
        raise EngineError("chat is not initialized")


def append_user(
    session: "Session", text: str, images: Sequence[object] = (),
) -> Segment:
    if images:
        raise EngineError("image input requires M5 vision support")
    _start_if_empty(session)
    retire_paused(session)
    builder = PromptBuilder(session._engine)
    return cast(Segment, _append(
        session, builder.message(session._transcript.tokens, "user", text), "user",
    ))


def append_prefix(session: "Session", turns: Iterable[PrefixTurn]) -> None:
    turns = tuple(turns)
    if any(turn.role not in ("user", "assistant") for turn in turns):
        raise ValueError("prefix turns must have user or assistant roles")
    _start_if_empty(session)
    retire_paused(session)
    builder = PromptBuilder(session._engine)
    for turn in turns:
        tokens = builder.message(session._transcript.tokens, turn.role, turn.content)
        _append(session, tokens, turn.role)
        if turn.role == "assistant":
            append_assistant_end(session)


def run_round(
    session: "Session", max_tokens: int, sampling: SamplingPolicy | None,
    stop_sequences: Sequence[bytes] = (),
) -> Generator[Event, None, None]:
    stream = session._paused_stream
    resuming = stream is not None
    control = resolve_control(stream.control if stream is not None else None, stop_sequences)
    if not resuming:
        append_assistant_prefix(session)
    try:
        think = session._engine.think
        if stream is None:
            stream = RoundStream(session._engine.info, think, control=control)
        remaining = max_tokens
        reserve = 1
        last_tokens: tuple[int, ...] | None = None
        while True:
            tokens = session._transcript.tokens
            if tokens != last_tokens or len(tokens) < session._engine.config.ctx:
                if (yield from session._job_events(SyncJob(tokens))):
                    return
            # An unchanged full session is already synced. Generate again so a
            # zero reserve reports another pause instead of failing the sync.
            sampled_before = len(stream.sampled_tokens)
            action = yield from session._job_events(GenerateJob(
                sampling or session._engine.sampling_defaults, session._rng,
                remaining, session._transcript, reserve=reserve,
                think_mode=think, stream=stream,
                logits_hooks=tuple(session._logits_hooks),
                control=control, resume=resuming,
            ), pause_on_reserve=True)
            last_tokens = session._transcript.tokens
            if not isinstance(action, tuple) or action[0] != "resume":
                return
            remaining -= len(stream.sampled_tokens) - sampled_before
            reserve = action[1]
            resuming = True
    finally:
        if stream is not None and stream.stop_sequence is not None:
            session._paused_stream = stream
            session._paused_control = stream.control
        else:
            session._paused_stream = session._paused_control = None
            append_assistant_end(session)


def append_assistant_prefix(session: "Session") -> Segment | None:
    _require_chat(session)
    builder = PromptBuilder(session._engine)
    tokens = builder.assistant_prefix(session._transcript.tokens)
    return _append(session, tokens, "assistant_prefix")


def append_assistant_end(session: "Session") -> Segment | None:
    _require_chat(session)
    return _append(session, PromptBuilder(session._engine).append_assistant_end(), "eos")


def append_tool_results(session: "Session", results: Iterable[str]) -> Segment:
    _require_chat(session)
    retire_paused(session)
    builder = PromptBuilder(session._engine)
    tokens = builder.tool_results(session._transcript.tokens, results)
    return cast(Segment, _append(session, tokens, "tool"))


def append_system_note(
    session: "Session", text: str, subtype: str | None = None,
) -> Segment:
    _require_chat(session)
    retire_paused(session)
    builder = PromptBuilder(session._engine)
    tokens = builder.message(session._transcript.tokens, "system", "\n" + text)
    meta = {"subtype": subtype} if subtype else None
    return cast(Segment, _append(session, tokens, "note", meta))


def retire_paused(session: "Session") -> None:
    """Close a paused assistant before accepting a different conversation turn."""
    if session._paused_stream is not None:
        session._paused_stream = session._paused_control = None
        append_assistant_end(session)
