"""Token acceptance and commit policy separated from native decode steps."""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Protocol

from ds4py._native.protocol import EngineHandle, SessionHandle
from ds4py.errors import EngineError
from .events import Event, Token
from .sampling import Rng, SamplingPolicy
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


class DecodeContext:
    def __init__(self, engine: EngineHandle, job: GenerateJob, *, limit: int,
                 limit_reason: str, cancelled: Callable[[], bool],
                 emit: Callable[[Event], None]):
        self.rng = job.rng.state
        self.reason: str | None = None
        self._engine, self._job = engine, job
        self._limit, self._limit_reason = limit, limit_reason
        self._cancelled, self._emit = cancelled, emit
        self._accepted = 0
        self._committed = False

    def before_sample(self) -> None:
        pass

    def sampling(self) -> SamplingPolicy:
        return self._job.sampling

    def cancelled(self) -> bool:
        return self._cancelled()

    def accept(self, token: int) -> Verdict:
        if self._engine.is_stop(token):
            self.reason = "stop_token"
            return Verdict.STOP_BEFORE
        self._accepted += 1
        if self._accepted >= self._limit:
            self.reason = self._limit_reason
            return Verdict.STOP_AFTER
        return Verdict.CONTINUE

    def commit(self, token: int) -> None:
        """Publish an accepted token only after the step has evaluated it."""
        text = self._engine.token_text(token)
        if self._committed:
            self._job.transcript.extend([token])
        else:
            self._job.transcript.append([token], "assistant")
            self._committed = True
        self._emit(Token(token, text))


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
            ctx.commit(token)
            if verdict is Verdict.STOP_AFTER:
                return


def generate(engine: EngineHandle, session: SessionHandle, job: GenerateJob, *,
             cancelled: Callable[[], bool], emit: Callable[[Event], None],
             step: DecodeStep | None = None) -> str:
    room = session.ctx() - session.pos()
    limit = min(job.max_tokens, max(room - job.reserve, 0))
    limit_reason = "max_tokens" if job.max_tokens <= room - job.reserve else "context_reserve"
    if limit == 0:
        return limit_reason
    ctx = DecodeContext(engine, job, limit=limit, limit_reason=limit_reason,
                        cancelled=cancelled, emit=emit)
    if step is None:
        step = PlainStep()
    step.run(session, ctx)
    return ctx.reason or "interrupted"
