"""Token acceptance and commit policy separated from native decode steps."""
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Callable, Protocol

from ds4py._native.protocol import EngineHandle, SessionHandle
from ds4py.errors import EngineError
from .events import Event, Token
from .hooks import LogitHookSet
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
    hooks: LogitHookSet | None = None
    stop_sequences: list[bytes] = field(default_factory=list)


class DecodeContext:
    def __init__(self, engine: EngineHandle, job: GenerateJob, *, limit: int,
                 limit_reason: str, cancelled: Callable[[], bool],
                 emit: Callable[[Event], None], session: SessionHandle | None = None,
                 hooks_every_sample: bool = True):
        self.rng = job.rng.state
        self.reason: str | None = None
        self._engine, self._job = engine, job
        self._limit, self._limit_reason = limit, limit_reason
        self._cancelled, self._emit = cancelled, emit
        self._session = session
        self._hooks_every_sample = hooks_every_sample
        self._accepted = 0
        self._committed = False
        self._verdict = Verdict.CONTINUE
        self._event_buffer = None

    def buffer_events(self):
        self._event_buffer = []

    def flush_events(self):
        events, self._event_buffer = self._event_buffer, None
        for event in events or ():
            self._emit(event)

    def _publish(self, event):
        if self._event_buffer is None:
            self._emit(event)
        else:
            self._event_buffer.append(event)

    def _prepare_hooks(self) -> None:
        hooks = self._job.hooks
        if hooks is None:
            return
        stream = self._job.stream
        thinking = bool(stream and stream.decoder.thinking)
        hooks.before_sample(self._session, self.sampling(), thinking=thinking)

    def before_sample(self) -> None:
        if self._hooks_every_sample:
            self._prepare_hooks()

    def before_block(self) -> None:
        self._prepare_hooks()

    def is_stop(self, token: int) -> bool:
        if self._job.think_mode is None:
            return self._engine.is_stop(token)
        return self._engine.is_stop_for_think_mode(token, self._job.think_mode)

    def sampling(self) -> SamplingPolicy:
        policy = self._job.sampling
        if self._job.stream is not None and self._job.stream.parser.wants_greedy_sampling():
            return replace(policy, temperature=0.0, top_k=0, top_p=1.0, min_p=0.0)
        return policy

    def cancelled(self) -> bool:
        return self._cancelled()

    def accept(self, token: int) -> Verdict:
        if self.is_stop(token):
            self.reason = "stop_token"
            self._verdict = Verdict.STOP_BEFORE
            return self._verdict
        self._accepted += 1
        if self._accepted >= self._limit:
            self.reason = self._limit_reason
            self._verdict = Verdict.STOP_AFTER
        else:
            self._verdict = Verdict.CONTINUE
        return self._verdict

    def commit(self, token: int, verdict: Verdict | None = None) -> Verdict:
        """Publish an evaluated token, then observe its streaming state."""
        verdict = self._verdict if verdict is None else verdict
        text = self._engine.token_text(token)
        if self._committed:
            self._job.transcript.extend([token])
        else:
            self._job.transcript.append([token], "assistant")
            self._committed = True
        self._publish(Token(token, text))
        stream = self._job.stream
        if stream is not None:
            stream.observe(token, text, self._publish)
            if stream.malformed:
                self.reason = "parser_error"
                verdict = Verdict.STOP_AFTER
            elif stream.complete:
                self.reason = "tool_call_complete"
                verdict = Verdict.STOP_AFTER
        self._verdict = verdict
        return verdict

    def finalize(self) -> None:
        stream = self._job.stream
        if stream is None or self.reason in ("interrupted", "context_reserve"):
            return
        finish_parser = self.reason in ("stop_token", "max_tokens")
        stream.finalize(parser=finish_parser, emit=self._publish)
        if stream.malformed:
            self.reason = "parser_error"
        elif stream.complete:
            self.reason = "tool_call_complete"
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


def generate(engine: EngineHandle, session: SessionHandle, job: GenerateJob, *,
             cancelled: Callable[[], bool], emit: Callable[[Event], None],
             step: DecodeStep | None = None) -> str:
    room = session.ctx() - session.pos()
    limit = min(job.max_tokens, max(room - job.reserve, 0))
    limit_reason = "max_tokens" if job.max_tokens <= room - job.reserve else "context_reserve"
    decode_step = step or PlainStep()
    ctx = DecodeContext(
        engine, job, limit=limit, limit_reason=limit_reason,
        cancelled=cancelled, emit=emit, session=session,
        hooks_every_sample=decode_step.hooks_every_sample,
    )
    registered = False
    try:
        if job.hooks is not None:
            session.set_logit_hooks(job.hooks)
            registered = True
        if limit <= 0:
            ctx.reason = limit_reason
        else:
            decode_step.run(session, ctx)
            if ctx.reason is None:
                if cancelled():
                    ctx.reason = "interrupted"
                else:
                    raise EngineError("decode returned no tokens")
        ctx.finalize()
        return ctx.reason
    finally:
        if registered:
            session.clear_logit_hooks()
