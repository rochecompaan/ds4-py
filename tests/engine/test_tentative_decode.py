from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from ds4py import ThinkMode
from ds4py._native.fake import FakeNative
from ds4py._native.protocol import EngineInfo
from ds4py.engine.decode import DecodeContext, GenerateJob, Verdict, generate
from ds4py.engine.events import Event, TextDelta, Token, ToolSyntax
from ds4py.engine.logits import LogitsView
from ds4py.engine.pause import ControlSpan
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.stream import RoundStream
from ds4py.engine.transcript import Transcript
from ds4py.errors import EngineError

if TYPE_CHECKING:
    from ds4py._native.protocol import EngineHandle, SessionHandle

BASE_INFO = FakeNative()._info


class TransactionEngine:
    def __init__(self, info: EngineInfo) -> None:
        self._info = info
        self.texts = {7: b"a", 8: b"<tool_call>", 9: b"<function=", 10: b"discard", 11: b"guard"}

    def info(self) -> EngineInfo:
        return self._info

    def token_text(self, token: int) -> bytes:
        return self.texts.get(token, bytes([token]))

    def is_stop(self, token: int) -> bool:
        return token == 0

    def is_stop_for_think_mode(self, token: int, mode: int) -> bool:
        return token == 0


class TransactionSession:
    def __init__(self) -> None:
        self.tokens = [5, 6]
        self.calls: list[tuple[str, int]] = []

    def pos(self) -> int:
        return len(self.tokens)

    def ctx(self) -> int:
        return 100

    def eval(self, token: int) -> None:
        self.tokens.append(token)
        self.calls.append(("eval", token))

    def rewind(self, position: int) -> None:
        del self.tokens[position:]
        self.calls.append(("rewind", position))


def make_context(
    chat: bool,
    *,
    stop_sequences: tuple[bytes, ...] = (),
    texts: dict[int, bytes] | None = None,
    limit: int = 10,
) -> tuple[DecodeContext, TransactionSession, GenerateJob, list[Event], list[LogitsView]]:
    info = replace(BASE_INFO, family="qwen", is_qwen4=True) if chat else BASE_INFO
    engine = TransactionEngine(info)
    if texts is not None:
        engine.texts.update(texts)
    session = TransactionSession()
    transcript = Transcript()
    transcript.append([5, 6], "user")
    control = ControlSpan(stop_sequences)
    stream = RoundStream(info, ThinkMode.NONE, control=control) if chat else None
    seen: list[LogitsView] = []
    job = GenerateJob(
        SamplingPolicy(temperature=0.7, top_k=4, top_p=0.8, min_p=0.1),
        Rng(1),
        20,
        transcript,
        think_mode=ThinkMode.NONE if chat else None,
        stream=stream,
        control=control,
        logits_hooks=(seen.append,),
    )
    events: list[Event] = []
    ctx = DecodeContext(
        cast("EngineHandle", engine),
        job,
        limit=limit,
        limit_reason="max_tokens",
        cancelled=lambda: False,
        emit=events.append,
        session=cast("SessionHandle", session),
    )
    assert ctx.accept(7) is Verdict.CONTINUE
    session.eval(7)
    assert ctx.commit(7) is Verdict.CONTINUE
    return ctx, session, job, events, seen


@pytest.mark.parametrize("chat", [False, True], ids=["raw", "chat"])
def test_staging_publishes_exact_prefix_and_preserves_canonical_identities(chat: bool) -> None:
    ctx, session, job, events, seen = make_context(chat)
    transcript_prefix = (job.transcript.tokens, job.transcript.segments)
    canonical_stream = job.stream
    canonical_control = job.control
    sampled = canonical_stream.sampled_tokens if canonical_stream is not None else job.sampled_tokens
    prefix_events = list(events)

    ctx.begin_staging()
    assert ctx.staging
    for token in (8, 9):
        verdict = ctx.accept(token)
        assert ctx.stage(token, verdict) is Verdict.CONTINUE
        session.eval(token)

    assert (job.transcript.tokens, job.transcript.segments) == transcript_prefix
    assert sampled == [7]
    assert events == prefix_events
    assert canonical_control.stop_sequence is None
    ctx.before_sample()
    assert seen[-1].position == 5
    assert seen[-1].sampled_tokens == (7, 8, 9)
    assert seen[-1].parser_state == ("STRUCTURAL" if chat else None)
    if chat:
        assert ctx.sampling() == SamplingPolicy(temperature=0.0, top_k=0, top_p=1.0, min_p=0.0)
    else:
        assert ctx.sampling() == job.sampling

    ctx.publish()
    assert not ctx.staging
    assert job.stream is canonical_stream
    assert job.control is canonical_control
    assert canonical_stream is None or canonical_stream.control is canonical_control
    current_sampled = canonical_stream.sampled_tokens if canonical_stream is not None else job.sampled_tokens
    assert current_sampled is sampled
    assert sampled == [7, 8, 9]
    assert job.transcript.tokens == (5, 6, 7, 8, 9)
    expected_tail: list[Event] = [Token(8, b"<tool_call>"), Token(9, b"<function=")]
    if chat:
        expected_tail = [
            Token(8, b"<tool_call>"),
            ToolSyntax("STRUCTURAL"),
            TextDelta("<tool_call>", False),
            Token(9, b"<function="),
            TextDelta("<function=", False),
        ]
    assert events == [*prefix_events, *expected_tail]

    ctx.before_sample()
    assert seen[-1].position == 5
    assert seen[-1].sampled_tokens == (7, 8, 9)

    canonical_after_publish = (job.transcript.tokens, job.transcript.segments, list(sampled), list(events))
    ctx.begin_staging()
    verdict = ctx.accept(10)
    ctx.stage(10, verdict)
    ctx.discard()
    assert (job.transcript.tokens, job.transcript.segments, sampled, events) == canonical_after_publish


@pytest.mark.parametrize("eval_before_stage", [False, True], ids=["stage-before-eval", "eval-before-stage"])
def test_staged_position_does_not_depend_on_native_eval_order(eval_before_stage: bool) -> None:
    ctx, session, _job, _events, seen = make_context(False)
    ctx.begin_staging()
    verdict = ctx.accept(8)
    if eval_before_stage:
        session.eval(8)
    ctx.stage(8, verdict)
    if not eval_before_stage:
        session.eval(8)
    ctx.before_sample()
    assert seen[-1].position == 4
    assert seen[-1].sampled_tokens == (7, 8)


def test_staging_rejects_nested_and_direct_canonical_operations() -> None:
    ctx, session, job, events, _seen = make_context(False)
    for operation in (lambda: ctx.stage(8), ctx.publish, ctx.discard):
        with pytest.raises(EngineError):
            operation()

    ctx.begin_staging()
    before = (session.pos(), job.transcript.tokens, job.transcript.segments, list(job.sampled_tokens), list(events))
    with pytest.raises(EngineError):
        ctx.begin_staging()
    for operation in (lambda: ctx.commit(11), ctx.resume, ctx.finalize):
        with pytest.raises(EngineError):
            operation()
        assert (session.pos(), job.transcript.tokens, job.transcript.segments, job.sampled_tokens, events) == before
    ctx.discard()


@pytest.mark.parametrize(
    ("reason", "chat", "text", "stop_sequences", "limit"),
    [
        ("max_tokens", False, b"last", (), 2),
        ("stop_sequence", False, b"stop", (b"stop",), 10),
        ("stop_sequence", True, b"stop", (b"stop",), 10),
        (
            "tool_call_complete",
            True,
            b"<tool_call><function=echo><parameter=value>x</parameter></function></tool_call>!",
            (),
            10,
        ),
        ("parser_error", True, b"<tool_call>bad", (), 10),
    ],
)
def test_staged_terminal_matches_direct_finalization(
    reason: str,
    chat: bool,
    text: bytes,
    stop_sequences: tuple[bytes, ...],
    limit: int,
) -> None:
    reference, reference_session, reference_job, reference_events, _reference_seen = make_context(
        chat,
        stop_sequences=stop_sequences,
        texts={8: text},
        limit=limit,
    )
    verdict = reference.accept(8)
    assert verdict is (Verdict.STOP_AFTER if reason == "max_tokens" else Verdict.CONTINUE)
    reference_session.eval(8)
    assert reference.commit(8, verdict) is Verdict.STOP_AFTER
    reference.finalize()

    ctx, session, job, events, _seen = make_context(
        chat,
        stop_sequences=stop_sequences,
        texts={8: text},
        limit=limit,
    )
    canonical_events = list(events)
    canonical_stream_snapshot = job.stream.snapshot if job.stream is not None else None
    ctx.begin_staging()
    verdict = ctx.accept(8)
    assert verdict is (Verdict.STOP_AFTER if reason == "max_tokens" else Verdict.CONTINUE)
    assert ctx.stage(8, verdict) is Verdict.STOP_AFTER
    assert events == canonical_events
    assert job.control.stop_sequence is None
    assert job.stream is None or (not job.stream.finalized and job.stream.snapshot == canonical_stream_snapshot)
    session.eval(8)
    ctx.publish()
    ctx.finalize()

    assert ctx.reason == reference.reason == reason
    assert ctx._finish_parser == reference._finish_parser
    assert job.transcript.tokens == reference_job.transcript.tokens
    assert job.transcript.segments == reference_job.transcript.segments
    assert events == reference_events
    assert job.sampled_tokens == reference_job.sampled_tokens
    assert job.control.stop_sequence == reference_job.control.stop_sequence
    assert session.tokens == reference_session.tokens
    if job.stream is not None and reference_job.stream is not None:
        assert job.stream.sampled_tokens == reference_job.stream.sampled_tokens
        assert job.stream.snapshot == reference_job.stream.snapshot
        assert job.stream.finalized == reference_job.stream.finalized


def test_generate_discards_an_abandoned_staging_block() -> None:
    original, session, job, _prefix_events, _seen = make_context(False)
    before = (job.transcript.tokens, job.transcript.segments, list(job.sampled_tokens), session.pos())

    class AbandoningStep:
        hooks_every_sample = True

        def run(self, session: SessionHandle, ctx: DecodeContext) -> None:
            ctx.begin_staging()
            verdict = ctx.accept(8)
            ctx.stage(8, verdict)

    events: list[Event] = []
    with pytest.raises(EngineError, match=r"^decode returned with an active staging block$"):
        generate(
            original._engine,
            cast("SessionHandle", session),
            job,
            cancelled=lambda: False,
            emit=events.append,
            step=AbandoningStep(),
        )
    assert (job.transcript.tokens, job.transcript.segments, job.sampled_tokens, session.pos()) == before
    assert events == []
