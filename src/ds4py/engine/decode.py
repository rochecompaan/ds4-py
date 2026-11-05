"""Token acceptance and commit policy separated from native decode steps."""

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, cast

from ds4py.errors import EngineError

from .events import Event, Token
from .logits import LogitsView
from .pause import ControlSpan

if TYPE_CHECKING:
    from collections.abc import Callable

    from ds4py._native.protocol import EngineHandle, SessionHandle

    from .sampling import Rng, SamplingPolicy
    from .stream import RoundStream
    from .transcript import Transcript


class Verdict(Enum):
    CONTINUE = auto()
    STOP_BEFORE = auto()
    STOP_AFTER = auto()


@dataclass(frozen=True)
class GenerateJob:
    sampling: SamplingPolicy
    rng: Rng
    max_tokens: int
    transcript: Transcript
    reserve: int = 1
    think_mode: int | None = None
    stream: RoundStream | None = None
    logits_hooks: tuple[Callable[[LogitsView], None], ...] = ()
    control: ControlSpan = field(default_factory=ControlSpan)
    resume: bool = False
    sampled_tokens: list[int] = field(default_factory=list)
    decode_step: DecodeStep | None = None

    def __post_init__(self) -> None:
        if self.stream is not None and self.stream.control is not self.control:
            raise ValueError("stream and job control must be identical")


@dataclass
class _DecodeOverlay:
    block_position: int
    stream: RoundStream | None
    control: ControlSpan
    sampled_tokens: list[int]
    accepted: int
    reason: str | None
    finalized: bool
    finish_parser: bool | None
    verdict: Verdict
    token_ids: list[int] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)


class DecodeContext:
    def __init__(  # noqa: PLR0913 - Decode context mirrors the native round dependencies.
        self,
        engine: EngineHandle,
        job: GenerateJob,
        *,
        limit: int,
        limit_reason: str,
        cancelled: Callable[[], bool],
        emit: Callable[[Event], None],
        session: SessionHandle | None = None,
    ) -> None:
        self.rng = job.rng.state
        self.reason: str | None = None
        self._engine, self._job = engine, job
        self._limit, self._limit_reason = limit, limit_reason
        self._cancelled, self._emit = cancelled, emit
        self._session = session
        self._accepted = 0
        self._committed = False
        self._sampled_tokens = job.stream.sampled_tokens if job.stream is not None else job.sampled_tokens
        self._logits_view = (
            LogitsView(cast("SessionHandle", session), engine.info().vocab_size) if job.logits_hooks else None
        )
        self._verdict = Verdict.CONTINUE
        self._event_buffer: list[Event] | None = None
        self._overlay: _DecodeOverlay | None = None
        self.finalized = False
        self._finish_parser: bool | None = None

    @property
    def staging(self) -> bool:
        return self._overlay is not None

    def _require_staging(self) -> _DecodeOverlay:
        if self._overlay is None:
            raise EngineError("decode staging block is not active")
        return self._overlay

    def _reject_staging(self) -> None:
        if self.staging:
            raise EngineError("direct decode operation is not allowed while staging")

    def begin_staging(self) -> None:
        if self.staging:
            raise EngineError("decode staging block is already active")
        if self._session is None:
            raise EngineError("decode staging requires a native session")
        stream = self._job.stream.fork() if self._job.stream is not None else None
        control = stream.control if stream is not None else self._job.control.fork()
        sampled_tokens = stream.sampled_tokens if stream is not None else list(self._sampled_tokens)
        self._overlay = _DecodeOverlay(
            block_position=self._session.pos(),
            stream=stream,
            control=control,
            sampled_tokens=sampled_tokens,
            accepted=self._accepted,
            reason=self.reason,
            finalized=self.finalized,
            finish_parser=self._finish_parser,
            verdict=self._verdict,
        )

    def discard(self) -> None:
        overlay = self._require_staging()
        self._accepted = overlay.accepted
        self.reason = overlay.reason
        self.finalized = overlay.finalized
        self._finish_parser = overlay.finish_parser
        self._verdict = overlay.verdict
        self._overlay = None

    def buffer_events(self) -> None:
        self._event_buffer = []

    def flush_events(self) -> None:
        events, self._event_buffer = self._event_buffer, None
        for event in events or ():
            self._emit(event)

    def _publish(self, event: Event) -> None:
        if self._overlay is not None:
            self._overlay.events.append(event)
        elif self._event_buffer is None:
            self._emit(event)
        else:
            self._event_buffer.append(event)

    def before_sample(self) -> None:
        if self._logits_view is not None:
            if self._session is None:
                raise AssertionError("logits hooks require a native session")
            overlay = self._overlay
            stream = overlay.stream if overlay is not None else self._job.stream
            sampled_tokens = overlay.sampled_tokens if overlay is not None else self._sampled_tokens
            position = overlay.block_position + len(overlay.token_ids) if overlay is not None else self._session.pos()
            self._logits_view._update(
                position=position,
                in_think=bool(stream and stream.snapshot.in_think),
                think_tokens=stream.think_tokens if stream is not None else 0,
                parser_state=stream.snapshot.state if stream is not None else None,
                sampled_tokens=sampled_tokens,
            )
            for hook in self._job.logits_hooks:
                hook(self._logits_view)

    def is_stop(self, token: int) -> bool:
        if self._job.think_mode is None:
            return self._engine.is_stop(token)
        return self._engine.is_stop_for_think_mode(token, self._job.think_mode)

    def sampling(self) -> SamplingPolicy:
        policy = self._job.sampling
        overlay = self._overlay
        stream = overlay.stream if overlay is not None else self._job.stream
        if stream is not None and stream.parser.wants_greedy_sampling():
            return replace(policy, temperature=0.0, top_k=0, top_p=1.0, min_p=0.0)
        return policy

    def cancelled(self) -> bool:
        return self._cancelled()

    def accept(self, token: int) -> Verdict:
        if self.finalized:
            return Verdict.STOP_AFTER
        if self.is_stop(token):
            self.finish("stop_token")
            self._verdict = Verdict.STOP_BEFORE
            return self._verdict
        self._accepted += 1
        if self._accepted >= self._limit:
            self.reason = self._limit_reason
            self._verdict = Verdict.STOP_AFTER
        else:
            self._verdict = Verdict.CONTINUE
        return self._verdict

    def _observe_token(
        self,
        token: int,
        verdict: Verdict,
        *,
        stream: RoundStream | None,
        control: ControlSpan,
        sampled_tokens: list[int],
    ) -> Verdict:
        text = self._engine.token_text(token)
        self._publish(Token(token, text))
        if stream is not None:
            stream.observe(token, text, self._publish)
        elif control.sequences:
            control.feed(text)
        # Stop sequences can pause a raw round before any hook is registered.
        if stream is None and (self._logits_view is not None or control.sequences):
            sampled_tokens.append(token)
        if stream is not None and stream.malformed:
            self.finish("parser_error")
            verdict = Verdict.STOP_AFTER
        elif stream is not None and stream.complete:
            self.finish("tool_call_complete", finish_parser=False)
            verdict = Verdict.STOP_AFTER
        elif control.stop_sequence is not None:
            self.finish("stop_sequence")
            verdict = Verdict.STOP_AFTER
        self._verdict = verdict
        return verdict

    def _append_transcript(self, tokens: list[int]) -> None:
        if not tokens:
            return
        if self._committed:
            self._job.transcript.extend(tokens)
        else:
            segments = self._job.transcript.segments
            if self._job.resume and segments and segments[-1].kind == "assistant":
                self._job.transcript.extend(tokens)
            else:
                self._job.transcript.append(tokens, "assistant")
            self._committed = True

    def commit(self, token: int, verdict: Verdict | None = None) -> Verdict:
        """Publish an evaluated token, then observe its streaming state."""
        self._reject_staging()
        if self.finalized:
            return Verdict.STOP_AFTER
        verdict = self._verdict if verdict is None else verdict
        self._append_transcript([token])
        return self._observe_token(
            token,
            verdict,
            stream=self._job.stream,
            control=self._job.control,
            sampled_tokens=self._sampled_tokens,
        )

    def stage(self, token: int, verdict: Verdict | None = None) -> Verdict:
        overlay = self._require_staging()
        if self.finalized:
            return Verdict.STOP_AFTER
        verdict = self._verdict if verdict is None else verdict
        overlay.token_ids.append(token)
        return self._observe_token(
            token,
            verdict,
            stream=overlay.stream,
            control=overlay.control,
            sampled_tokens=overlay.sampled_tokens,
        )

    def publish(self) -> None:
        overlay = self._require_staging()
        self._append_transcript(overlay.token_ids)
        if self._job.stream is not None:
            if overlay.stream is None:
                raise AssertionError("chat staging requires a stream overlay")
            self._job.stream.restore(overlay.stream.capture())
        else:
            self._job.control.restore(overlay.control.capture())
            self._sampled_tokens[:] = overlay.sampled_tokens
        events = overlay.events
        self._overlay = None
        for event in events:
            self._emit(event)

    def finish(self, reason: str, *, finish_parser: bool = True) -> None:
        """Set the terminal reason without changing accepted-token counters."""
        self.reason = reason
        self.finalized = True
        self._finish_parser = finish_parser
        self._verdict = Verdict.STOP_AFTER

    def resume(self) -> None:
        self._reject_staging()
        if self._job.stream is not None:
            self._job.stream.resume()
        else:
            self._job.control.resume()
        self.reason = None
        self.finalized = False
        self._finish_parser = None
        self._verdict = Verdict.CONTINUE

    def finalize(self) -> None:
        self._reject_staging()
        stream = self._job.stream
        if stream is None or self.reason in (
            "interrupted",
            "context_reserve",
            "stop_sequence",
        ):
            return
        finish_parser = self._finish_parser
        if finish_parser is None:
            finish_parser = self.reason in ("stop_token", "max_tokens")
        stream.finalize(parser=finish_parser, emit=self._publish)
        if stream.malformed:
            self.reason = "parser_error"
        elif stream.complete:
            self.reason = "tool_call_complete"
        elif stream.stop_sequence is not None:
            self.reason = "stop_sequence"
        stream.emit_calls(self._publish)


class DecodeStep(Protocol):
    hooks_every_sample: bool

    def run(self, session: SessionHandle, ctx: DecodeContext) -> None: ...


class PlainStep:
    hooks_every_sample = True

    def run(self, session: SessionHandle, ctx: DecodeContext) -> None:
        while not ctx.cancelled():
            ctx.before_sample()
            p = ctx.sampling()
            token = session.sample(p.temperature, p.top_k, p.top_p, p.min_p, ctx.rng)
            if token < 0:
                raise EngineError("cannot sample: the session state is not valid; sync first")
            verdict = ctx.accept(token)
            if verdict is Verdict.STOP_BEFORE:
                return
            session.eval(token)
            verdict = ctx.commit(token)
            if verdict is Verdict.STOP_AFTER:
                return


def generate(  # noqa: PLR0913 - Generation accepts the complete native round context.
    engine: EngineHandle,
    session: SessionHandle,
    job: GenerateJob,
    *,
    cancelled: Callable[[], bool],
    emit: Callable[[Event], None],
    step: DecodeStep | None = None,
) -> str:
    room = session.ctx() - session.pos()
    limit = min(job.max_tokens, max(room - job.reserve, 0))
    limit_reason = "max_tokens" if job.max_tokens <= room - job.reserve else "context_reserve"
    decode_step = step or PlainStep()
    if job.logits_hooks and not decode_step.hooks_every_sample:
        decode_step = PlainStep()
    ctx = DecodeContext(
        engine,
        job,
        limit=limit,
        limit_reason=limit_reason,
        cancelled=cancelled,
        emit=emit,
        session=session,
    )
    if job.resume:
        ctx.resume()
    if limit <= 0:
        ctx.reason = limit_reason
    elif not ctx.finalized:
        try:
            decode_step.run(session, ctx)
        except BaseException:
            if ctx.staging:
                ctx.discard()
            raise
        if ctx.staging:
            ctx.discard()
            raise EngineError("decode returned with an active staging block")
        if ctx.reason is None:
            if cancelled():
                ctx.reason = "interrupted"
            else:
                raise EngineError("decode returned no tokens")
    ctx.finalize()
    if ctx.reason is None:
        raise AssertionError("finalized decode requires a stop reason")
    return ctx.reason
