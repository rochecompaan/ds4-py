"""Private summary generation; message selection and replacement belong to callers."""

from array import array
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ds4py._native.protocol import SyncStatus, ThinkMode
from ds4py.errors import EngineError

from .decode import PlainStep, Verdict
from .events import SummaryToken
from .native_restore import restore_native_tokens
from .prompt import PromptBuilder

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ds4py._native.protocol import EngineHandle, SessionHandle

    from .events import Event
    from .session import Engine

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
SUMMARY_SPARE_TOKENS = 1
_QWEN_TOOL_MARKERS = ("<function=", "</function>", "<parameter=", "</parameter>")
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


def summary_budget(ctx: int) -> int:
    """Shared output budget, independent of trigger and execution room policy."""
    return max(_MIN_SUMMARY_TOKENS, min(ctx // 8, 4096))


def summary_suffix(engine: Engine, prefix: Sequence[int], *, prompt: str = SUMMARY_PROMPT) -> tuple[int, ...]:
    """Render a private user prompt and no-think opener against the caller prefix."""
    tokens = PromptBuilder(engine).message(prefix, "user", prompt)
    tokens.extend(engine._assistant_prefix([*prefix, *tokens], ThinkMode.NONE))
    return tuple(tokens)


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
        markers = _TOOL_MARKERS + (_QWEN_TOOL_MARKERS if engine.info().family == "qwen" else ())
        self._literal_markers = (_DSML, *(marker.encode() for marker in markers))
        # Mask dedicated tokens and sanitize ordinary spellings of the same set.
        self._masked = {ids[0] for marker in markers if len(ids := engine.tokenize(marker, rendered=True)) == 1}
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

    def _truncate(self, end: int, *, dsml: bool = True) -> None:
        if dsml and end and self._text[end - 1] == ord("<"):
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
        boundaries = [
            (position, marker == _DSML)
            for marker in self._literal_markers
            if (position := self._text.find(marker)) >= 0
        ]
        if boundaries:
            position, dsml = min(boundaries)
            self._truncate(position, dsml=dsml)
            return Verdict.STOP_AFTER
        return Verdict.STOP_AFTER if len(self._tokens) >= self._limit else Verdict.CONTINUE

    def result(self, emit: Callable[[Event], None]) -> SummaryResult:
        if self.cancelled():
            raise EngineError("summary generation interrupted")
        if not self._text:
            raise EngineError("compaction summary was empty")
        result = SummaryResult(self._text.decode("utf-8", errors="replace"), tuple(t.id for t in self._tokens))
        # Buffer until all control-marker trimming is complete: even private
        # events must not expose tool syntax, DSML, or its preceding '<'.
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
    room = session.ctx() - len(job.tokens) - SUMMARY_SPARE_TOKENS
    if room < _MIN_SUMMARY_TOKENS:
        raise EngineError("not enough context left to request compaction summary")
    limit = min(room, summary_budget(session.ctx()))
    prior = session.tokens()
    snapshot = session.save_snapshot() if session.checkpoint_valid() else None
    try:
        if session.sync(job.tokens, cancel=cancelled) is SyncStatus.INTERRUPTED:
            raise EngineError("summary generation interrupted")
        ctx = _SummaryDecode(engine, session, limit, cancelled)
        PlainStep().run(session, ctx)
        return ctx.result(emit)
    finally:
        if snapshot is None:
            restore_native_tokens(session, prior, valid=False, poison=poison)
        else:
            try:
                try:
                    session.load_snapshot(snapshot)
                finally:
                    snapshot.close()
            except BaseException as error:
                poison(error)
                raise
