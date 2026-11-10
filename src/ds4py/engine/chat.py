"""Session-level chat helpers; these functions own transcript mutations."""

from typing import TYPE_CHECKING, cast

from ds4py.errors import EngineError

from .decode import GenerateJob
from .pause import resolve_control
from .prompt import PromptBuilder
from .stream import RoundStream
from .worker import SyncJob

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping, Sequence

    from .events import Event
    from .prefix import PrefixTurn
    from .sampling import SamplingPolicy
    from .session import Session
    from .transcript import Segment


def _append(
    session: Session,
    tokens: Sequence[int],
    kind: str,
    meta: Mapping[str, object] | None = None,
) -> Segment | None:
    if not tokens:
        return None
    return session.history.append_segment(tokens, kind, meta)


def _start_if_empty(session: Session) -> None:
    history = session.history
    if history.transcript.chat:
        return
    if len(history.transcript):
        raise EngineError("chat is not initialized")
    session.start_chat()


def _require_chat(session: Session) -> None:
    if not session.history.transcript.chat:
        raise EngineError("chat is not initialized")


def append_message(
    session: Session,
    role: str,
    text: str,
    *,
    images: Sequence[object] = (),
    meta: Mapping[str, object] | None = None,
) -> Segment:
    """Append one plain-payload chat message, initializing an empty chat if needed."""
    if images:
        raise EngineError("image input requires M5 vision support")
    _start_if_empty(session)
    history = session.history
    with history.edit():
        retire_paused(session)
        tokens = PromptBuilder(session.engine).message(history.transcript.tokens, role, text)
        return history.append_segment(tokens, role, meta)


def append_user(
    session: Session,
    text: str,
    images: Sequence[object] = (),
) -> Segment:
    if images:
        raise EngineError("image input requires M5 vision support")
    _start_if_empty(session)
    history = session.history
    with history.edit():
        retire_paused(session)
        builder = PromptBuilder(session.engine)
        return cast(
            "Segment",
            _append(
                session,
                builder.message(history.transcript.tokens, "user", text),
                "user",
            ),
        )


def append_prefix(session: Session, turns: Iterable[PrefixTurn]) -> None:
    turns = tuple(turns)
    if any(turn.role not in ("user", "assistant") for turn in turns):
        raise ValueError("prefix turns must have user or assistant roles")
    _start_if_empty(session)
    history = session.history
    with history.edit():
        retire_paused(session)
        builder = PromptBuilder(session.engine)
        for turn in turns:
            tokens = builder.message(history.transcript.tokens, turn.role, turn.content)
            _append(session, tokens, turn.role)
            if turn.role == "assistant":
                append_assistant_end(session)


def run_round(
    session: Session,
    max_tokens: int,
    sampling: SamplingPolicy | None,
    stop_sequences: Sequence[bytes] = (),
) -> Generator[Event]:
    history = session.history
    stream = history.paused_stream
    resuming = stream is not None
    control = resolve_control(stream.control if stream is not None else None, stop_sequences)
    if not resuming:
        append_assistant_prefix(session)
    try:
        think = session.engine.think
        if stream is None:
            stream = RoundStream(session.engine.info, think, control=control)
        remaining = max_tokens
        reserve = 1
        last_tokens: tuple[int, ...] | None = None
        while True:
            tokens = history.transcript.tokens
            if (tokens != last_tokens or len(tokens) < session.engine.config.ctx) and bool(
                (yield from session._job_events(SyncJob(tokens)))
            ):
                return
            # An unchanged full session is already synced. Generate again so a
            # zero reserve reports another pause instead of failing the sync.
            sampled_before = len(stream.sampled_tokens)
            action = yield from session._job_events(
                GenerateJob(
                    sampling or session.engine.sampling_defaults,
                    session._rng,
                    remaining,
                    history.mutable_transcript,
                    reserve=reserve,
                    think_mode=think,
                    stream=stream,
                    logits_hooks=tuple(session._logits_hooks),
                    control=control,
                    resume=resuming,
                    decode_step=session._decode_step,
                ),
                pause_on_reserve=True,
            )
            last_tokens = history.transcript.tokens
            if not isinstance(action, tuple) or action[0] != "resume":
                return
            remaining -= len(stream.sampled_tokens) - sampled_before
            reserve = action[1]
            resuming = True
    finally:
        if stream.stop_sequence is not None:
            history.pause_stream(stream)
        else:
            history.clear_stream_pause()
            append_assistant_end(session)


def append_assistant_prefix(session: Session) -> Segment | None:
    _require_chat(session)
    builder = PromptBuilder(session.engine)
    tokens = builder.assistant_prefix(session.history.transcript.tokens)
    return _append(session, tokens, "assistant_prefix")


def append_assistant_end(session: Session) -> Segment | None:
    _require_chat(session)
    return _append(session, PromptBuilder(session.engine).append_assistant_end(), "eos")


def append_tool_results(session: Session, results: Iterable[str]) -> Segment:
    _require_chat(session)
    history = session.history
    with history.edit():
        retire_paused(session)
        builder = PromptBuilder(session.engine)
        tokens = builder.tool_results(history.transcript.tokens, results)
        return cast("Segment", _append(session, tokens, "tool"))


def append_system_note(
    session: Session,
    text: str,
    subtype: str | None = None,
) -> Segment:
    _require_chat(session)
    history = session.history
    with history.edit():
        retire_paused(session)
        builder = PromptBuilder(session.engine)
        tokens = builder.message(history.transcript.tokens, "system", "\n" + text)
        meta: dict[str, str] = {"text": text}
        if bool(subtype):
            meta["subtype"] = subtype
        return cast("Segment", _append(session, tokens, "note", meta))


def retire_paused(session: Session) -> None:
    """Close a paused assistant before accepting a different conversation turn."""
    history = session.history
    if history.paused_stream is not None:
        history.clear_stream_pause()
        append_assistant_end(session)
