"""Private summary generation; message selection and replacement belong to callers."""

from array import array
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ds4py._native.protocol import SyncStatus, ThinkMode
from ds4py.errors import EngineError

from .decode import PlainStep, Verdict
from .events import SummaryToken

if TYPE_CHECKING:
    from collections.abc import Callable

    from ds4py._native.protocol import EngineHandle, SessionHandle

    from .events import Event

# agent_compact_make_prompt, antirez/ds4 commit 5379c1c (without a reason).
SUMMARY_PROMPT = (
    "Internal ds4-agent context compaction request. This is not a user request.\n"
    "Write a durable task-state summary of the conversation so far. Preserve only facts that matter for continuing the work:\n"
    "- user goals, constraints, and preferences\n"
    "- files inspected or edited\n"
    "- commands run and important results\n"
    "- decisions, rejected approaches, known bugs, and pending next steps\n"
    "- reloadable bulky data with exact paths/ranges/commands when available\n\n"
    "Do not invent facts. Do not include generic narration. Do not include raw file contents unless they were essential to a conclusion.\n"
    "Do not solve unfinished tasks, calculate new results, or finish incomplete code in the summary. "
    "Record the user's requirements and what actually happened, leaving unfinished work pending.\n"
    "After the summary, stop. Do not continue the user task, do not call tools, and do not output thinking tags or DSML markup.\n"
    "Output only the compact summary.\n"
)
_DSML_TEXT = "\uff5cDSML\uff5c"
_DSML = _DSML_TEXT.encode()
_MIN_SUMMARY_TOKENS = 256
_TOOL_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
)


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class SummaryJob:
    """An already-rendered caller prefix plus private summary suffix."""

    tokens: tuple[int, ...]
    result: Future[SummaryResult] = field(default_factory=Future)


class _SummaryDecode:
    """PlainStep policy: greedy, tool-masked, with no response parser/history."""

    def __init__(
        self,
        engine: EngineHandle,
        session: SessionHandle,
        limit: int,
        cancelled: Callable[[], bool],
    ) -> None:
        self._engine, self._session = engine, session
        self._limit, self._cancelled = limit, cancelled
        self._tokens: list[SummaryToken] = []
        self._text = bytearray()
        self._dsml_ids = engine.tokenize(_DSML_TEXT, rendered=True)
        # Only dedicated vocabulary tokens are maskable, not plain spelling.
        self._masked = {ids[0] for marker in _TOOL_MARKERS if len(ids := engine.tokenize(marker, rendered=True)) == 1}
        self._logits = array("f", [0.0]) * engine.info().vocab_size

    def cancelled(self) -> bool:
        return self._cancelled()

    def before_sample(self) -> None:
        if self._masked:
            self._session.copy_logits(self._logits)
            for token in self._masked:
                self._logits[token] = float("-inf")
            self._session.set_logits(self._logits)

    def sample(self, session: SessionHandle) -> int:
        return session.argmax()

    def _truncate(self, end: int) -> None:
        if end and self._text[end - 1] == ord("<"):
            end -= 1
        del self._text[end:]
        remaining = end
        kept: list[SummaryToken] = []
        for token in self._tokens:
            if remaining <= 0:
                break
            text = token.text_bytes[:remaining]
            kept.append(SummaryToken(token.id, text))
            remaining -= len(text)
        self._tokens = kept

    def accept(self, token: int) -> Verdict:
        if len(self._dsml_ids) == 1 and token == self._dsml_ids[0]:
            self._truncate(len(self._text))
            return Verdict.STOP_BEFORE
        if self._engine.is_stop_for_think_mode(token, ThinkMode.NONE):
            return Verdict.STOP_BEFORE
        return Verdict.CONTINUE

    def commit(self, token: int) -> Verdict:
        text = self._engine.token_text(token)
        self._tokens.append(SummaryToken(token, text))
        self._text.extend(text)
        marker = self._text.find(_DSML)
        if marker >= 0:
            self._truncate(marker)
            return Verdict.STOP_AFTER
        return Verdict.STOP_AFTER if len(self._tokens) >= self._limit else Verdict.CONTINUE

    def result(self, emit: Callable[[Event], None]) -> SummaryResult:
        if self.cancelled():
            raise EngineError("summary generation interrupted")
        if not self._text:
            raise EngineError("compaction summary was empty")
        result = SummaryResult(self._text.decode("utf-8", errors="replace"), tuple(t.id for t in self._tokens))
        # Buffer until control-marker trimming is complete: even private events
        # must not expose DSML or the '<' that precedes its dedicated token.
        for token in self._tokens:
            emit(token)
        return result


def summarize(  # noqa: PLR0913 - Summary execution needs native handles and worker control callbacks.
    engine: EngineHandle,
    session: SessionHandle,
    job: SummaryJob,
    *,
    cancelled: Callable[[], bool],
    emit: Callable[[Event], None],
    poison: Callable[[BaseException], None],
) -> SummaryResult:
    """Borrow native state, run the shared decoder, and restore before returning."""
    room = session.ctx() - len(job.tokens) - 1
    if room < _MIN_SUMMARY_TOKENS:
        raise EngineError("not enough context left to request compaction summary")
    limit = min(room, max(_MIN_SUMMARY_TOKENS, min(session.ctx() // 8, 4096)))
    prior = session.tokens()
    snapshot = session.save_snapshot() if session.checkpoint_valid() else None
    try:
        if session.sync(job.tokens, cancel=cancelled) is SyncStatus.INTERRUPTED:
            raise EngineError("summary generation interrupted")
        ctx = _SummaryDecode(engine, session, limit, cancelled)
        PlainStep().run(session, ctx)
        return ctx.result(emit)
    finally:
        try:
            if snapshot is not None:
                session.load_snapshot(snapshot)
            else:
                # Invalid/empty states cannot be snapshotted. Rebuild without
                # cancellation, then reinstate invalid semantics, never private KV.
                session.invalidate()
                if prior:
                    session.sync(prior)
                else:
                    session.rewind(0)
                session.invalidate()
        except BaseException as error:
            poison(error)
            raise
        finally:
            if snapshot is not None:
                snapshot.close()
