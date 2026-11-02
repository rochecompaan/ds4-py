from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import pytest
from fake_steps import FakeMultiTokenStep

from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative, FakeSession
from ds4py._native.fake_chat import tokenize_rendered
from ds4py._native.protocol import EngineConfig, EngineInfo
from ds4py.engine.decode import DecodeContext, GenerateJob, Verdict, generate
from ds4py.engine.events import Event, TextDelta, Token, ToolSyntax
from ds4py.engine.logits import LogitsView
from ds4py.engine.pause import ControlSpan
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.stream import RoundStream
from ds4py.engine.transcript import Transcript
from ds4py.engine.worker import EngineWorker, SyncJob
from ds4py.errors import EngineError

if TYPE_CHECKING:
    from collections.abc import Callable

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


@dataclass
class EngineRun:
    reason: str
    events: list[Event]
    transcript: Transcript
    job: GenerateJob
    native_position: int
    native_tokens: list[int]
    native_calls: list[tuple[object, ...]]
    samples: list[tuple[float, int, float, float]]
    token_positions: list[tuple[int, int, int]]
    step: FakeMultiTokenStep | None


def token_script(tokens: Sequence[int]) -> Callable[[Sequence[int], Sequence[int]], int]:
    def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
        return tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS

    return script


def draft_blocks(tokens: Sequence[int], size: int = 17) -> list[list[int]]:
    return [list(tokens[start + 1 : start + size]) for start in range(0, len(tokens), size)]


def run_engine_case(
    tokens: Sequence[int],
    *,
    info: EngineInfo = BASE_INFO,
    blocks: Sequence[Sequence[int]] | None = None,
    hooked: bool = False,
    hook: Callable[[LogitsView], None] | None = None,
    stop_sequences: tuple[bytes, ...] = (),
    think_mode: int = ThinkMode.NONE,
    chat: bool = True,
) -> EngineRun:
    worker = EngineWorker(
        FakeNative(script=token_script(tokens), info=info),
        EngineConfig(Path("f"), ctx=1024),
    )
    try:
        transcript = Transcript()
        transcript.append([5, 6], "user")
        list(worker.submit(SyncJob((5, 6))))
        control = ControlSpan(stop_sequences)
        stream = RoundStream(info, think_mode, control=control) if chat else None
        hooks = (hook,) if hook is not None else (((lambda view: None),) if hooked else None)
        job = GenerateJob(
            SamplingPolicy(),
            Rng(1),
            1000,
            transcript,
            think_mode=think_mode if chat else None,
            stream=stream,
            control=control,
            logits_hooks=() if hooks is None else hooks,
        )
        step = FakeMultiTokenStep(blocks or ()) if blocks is not None else None
        events: list[Event] = []
        token_positions: list[tuple[int, int, int]] = []

        def execute(engine: EngineHandle, session: SessionHandle) -> str:
            def emit(event: Event) -> None:
                events.append(event)
                if isinstance(event, Token):
                    token_positions.append((event.id, session.pos(), len(transcript)))

            return generate(engine, session, job, cancelled=lambda: False, emit=emit, step=step)

        reason = worker.call(execute).result()
        native_position, native_tokens, native_calls, samples = worker.call(
            lambda engine, session: (
                session.pos(),
                session.tokens(),
                list(cast("FakeSession", session).calls),
                list(cast("FakeSession", session).samples),
            )
        ).result()
        return EngineRun(
            reason,
            events,
            transcript,
            job,
            native_position,
            native_tokens,
            native_calls,
            samples,
            token_positions,
            step,
        )
    finally:
        worker.close()


def family_stanza(info: EngineInfo) -> str:
    if info.is_glm_dsa:
        return '<tool_call>echo<arg_key>value</arg_key><arg_value>{"nested":[1,true]}</arg_value></tool_call>'
    if info.is_qwen4:
        return '<tool_call><function=echo><parameter=value>{"nested":[1,true]}</parameter></function></tool_call>'
    calls, invoke, parameter = (
        (" calls", " invoke", " parameter") if info.is_deepseek41 else ("tool_calls", "invoke", "parameter")
    )
    return (
        f'<｜DSML｜{calls}><｜DSML｜{invoke} name="echo">'
        f'<｜DSML｜{parameter} name="value">{{"nested":[1,true]}}</｜DSML｜{parameter}>'
        f"</｜DSML｜{invoke}></｜DSML｜{calls}>"
    )


FAMILY_INFOS = [
    BASE_INFO,
    replace(BASE_INFO, family="deepseek41", is_deepseek41=True),
    replace(BASE_INFO, family="glm", is_glm_dsa=True),
    replace(BASE_INFO, family="qwen", is_qwen4=True),
]


@pytest.mark.parametrize("info", FAMILY_INFOS, ids=lambda info: info.family)
@pytest.mark.parametrize("scenario", ["chunks", "think", "stop", "malformed", "tool"])
def test_hooked_serial_and_staged_family_parity(info: EngineInfo, scenario: str) -> None:
    stanza = family_stanza(info)
    stop_sequences: tuple[bytes, ...] = ()
    if scenario == "chunks":
        text = "prefix" + stanza + "tail"
        block_size = 3
    elif scenario == "think":
        text = "<think>reason</think>answer"
        block_size = 4
    elif scenario == "stop":
        text = "stop-tail"
        stop_sequences = (b"stop",)
        block_size = 3
    elif scenario == "malformed":
        text = "|DSML|" if not info.is_glm_dsa and not info.is_qwen4 else "<tool_call>bad"
        block_size = 4
    else:
        text = stanza if not info.is_glm_dsa and not info.is_qwen4 else stanza + "!"
        block_size = 5
    tokens = tokenize_rendered(text)
    blocks = draft_blocks(tokens, block_size)

    think_mode = ThinkMode.HIGH if scenario == "think" else ThinkMode.NONE
    reference = run_engine_case(tokens, info=info, hooked=True, stop_sequences=stop_sequences, think_mode=think_mode)
    staged = run_engine_case(
        tokens,
        info=info,
        blocks=blocks,
        hooked=True,
        stop_sequences=stop_sequences,
        think_mode=think_mode,
    )

    assert staged.events == reference.events
    assert staged.reason == reference.reason
    assert staged.transcript.tokens == reference.transcript.tokens
    assert staged.transcript.segments == reference.transcript.segments
    assert staged.job.sampled_tokens == reference.job.sampled_tokens
    staged_stream = staged.job.stream
    reference_stream = reference.job.stream
    assert staged_stream is not None and reference_stream is not None
    assert staged_stream.sampled_tokens == reference_stream.sampled_tokens
    assert staged.native_position == reference.native_position
    assert staged.native_tokens == reference.native_tokens
    assert staged.job.rng.state.value == reference.job.rng.state.value


def test_qwen_staged_opener_makes_the_next_verify_row_greedy() -> None:
    info = FAMILY_INFOS[-1]
    tokens = tokenize_rendered(family_stanza(info))
    run = run_engine_case(tokens, info=info, blocks=draft_blocks(tokens), hooked=True)
    opener_length = len(b"<tool_call>")
    assert run.samples[opener_length] == (0.0, 0, 1.0, 0.0)


def test_staged_hook_metadata_matches_every_serial_verify_row() -> None:
    info = FAMILY_INFOS[-1]
    text = "<think>reason</think>" + family_stanza(info)
    tokens = tokenize_rendered(text)
    serial_seen: list[tuple[int, str | None, bool, int, tuple[int, ...]]] = []
    staged_seen: list[tuple[int, str | None, bool, int, tuple[int, ...]]] = []

    def record(seen: list[tuple[int, str | None, bool, int, tuple[int, ...]]]) -> Callable[[LogitsView], None]:
        def hook(view: LogitsView) -> None:
            seen.append((view.position, view.parser_state, view.in_think, view.think_tokens, view.sampled_tokens))

        return hook

    reference = run_engine_case(tokens, info=info, hook=record(serial_seen), think_mode=ThinkMode.HIGH)
    staged = run_engine_case(
        tokens,
        info=info,
        blocks=draft_blocks(tokens, 4),
        hook=record(staged_seen),
        think_mode=ThinkMode.HIGH,
    )
    assert staged_seen == serial_seen
    assert [row[0] for row in staged_seen] == list(range(2, 2 + len(tokens) + 1))
    assert [row[4] for row in staged_seen] == [tuple(tokens[:index]) for index in range(len(tokens) + 1)]
    assert staged.reason == reference.reason


def test_terminal_staged_block_has_no_later_native_rows_and_publishes_after_commit() -> None:
    info = FAMILY_INFOS[-1]
    tokens = tokenize_rendered(family_stanza(info) + "!TAIL")
    kept = tokenize_rendered(family_stanza(info) + "!")
    run = run_engine_case(tokens, info=info, blocks=draft_blocks(tokens))
    assert run.reason == "tool_call_complete"
    assert run.native_tokens == [5, 6, *kept]
    assert [call for call in run.native_calls if call[0] == "eval"] == [("eval", token) for token in kept]
    assert [event.id for event in run.events if isinstance(event, Token)] == kept
    assert run.token_positions[-1] == (kept[-1], run.native_position, len(run.transcript))


def splitmix_after(seed: int, count: int) -> int:
    value = seed
    mask = (1 << 64) - 1
    for _ in range(count):
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value = value ^ (value >> 31)
    return value


def test_scripted_draft_blocks_advance_rng_only_for_anchors() -> None:
    tokens = [98, 99, 100, 0]
    run = run_engine_case(tokens, blocks=[[99, 100]], chat=False)
    assert run.job.rng.state.value == splitmix_after(1, 2)
    assert len(run.samples) == 2
    assert [event.id for event in run.events if isinstance(event, Token)] == tokens[:-1]


class FaultSession(TransactionSession):
    def __init__(
        self,
        samples: Sequence[int],
        *,
        eval_failure: tuple[int, EngineError, bool] | None = None,
        rewind_failure: EngineError | None = None,
    ) -> None:
        super().__init__()
        self.samples_to_return = list(samples)
        self.eval_failure = eval_failure
        self.rewind_failure = rewind_failure

    def sample(self, temperature: float, top_k: int, top_p: float, min_p: float, rng: object) -> int:
        return self.samples_to_return.pop(0)

    @override
    def eval(self, token: int) -> None:
        if self.eval_failure is not None and token == self.eval_failure[0]:
            _, failure, partial = self.eval_failure
            if partial:
                self.tokens.append(token)
            raise failure
        super().eval(token)

    @override
    def rewind(self, position: int) -> None:
        if self.rewind_failure is not None:
            raise self.rewind_failure
        super().rewind(position)


def run_fault_step(
    session: FaultSession,
    *,
    blocks: Sequence[Sequence[int]],
    hook: Callable[[LogitsView], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    observed: list[tuple[GenerateJob, list[Event]]] | None = None,
) -> tuple[GenerateJob, list[Event]]:
    engine = TransactionEngine(BASE_INFO)
    transcript = Transcript()
    transcript.append([5, 6], "user")
    job = GenerateJob(
        SamplingPolicy(),
        Rng(1),
        20,
        transcript,
        logits_hooks=() if hook is None else (hook,),
    )
    events: list[Event] = []
    if observed is not None:
        observed.append((job, events))
    generate(
        cast("EngineHandle", engine),
        cast("SessionHandle", session),
        job,
        cancelled=cancelled,
        emit=events.append,
        step=FakeMultiTokenStep(blocks),
    )
    return job, events


@pytest.mark.parametrize("fail_call", [1, 2], ids=["before-native", "after-native-prefix"])
def test_hook_failure_publishes_only_an_already_accepted_prefix(fail_call: int) -> None:
    failure = EngineError("hook failed")
    calls = 0

    def hook(view: LogitsView) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_call:
            raise failure

    session = FaultSession([98, 99, 0])
    observed: list[tuple[GenerateJob, list[Event]]] = []
    with pytest.raises(EngineError) as caught:
        run_fault_step(session, blocks=[[99]], hook=hook, observed=observed)
    assert caught.value is failure
    expected = [] if fail_call == 1 else [98]
    job, events = observed[0]
    assert session.tokens == [5, 6, *expected]
    assert job.transcript.tokens == (5, 6, *expected)
    assert [event.id for event in events if isinstance(event, Token)] == expected
    assert job.sampled_tokens == expected


@pytest.mark.parametrize("partial", [False, True], ids=["before-side-effect", "after-side-effect"])
def test_native_eval_failure_never_publishes_uncertain_tokens(partial: bool) -> None:
    failure = EngineError("eval failed")
    session = FaultSession([98, 99, 100], eval_failure=(99, failure, partial))
    observed: list[tuple[GenerateJob, list[Event]]] = []
    with pytest.raises(EngineError) as caught:
        run_fault_step(session, blocks=[[99, 100]], observed=observed)
    assert caught.value is failure
    assert session.tokens == [5, 6, 98, *([99] if partial else [])]
    job, events = observed[0]
    assert job.transcript.tokens == (5, 6)
    assert events == []
    assert job.sampled_tokens == []


def test_rewind_failure_does_not_publish_the_staged_prefix() -> None:
    failure = EngineError("rewind failed")
    session = FaultSession([98, 99, 100], rewind_failure=failure)
    checks = iter([False, False, True])
    observed: list[tuple[GenerateJob, list[Event]]] = []
    with pytest.raises(EngineError) as caught:
        run_fault_step(session, blocks=[[99, 100]], cancelled=lambda: next(checks, True), observed=observed)
    assert caught.value is failure
    assert session.tokens[:2] == [5, 6]
    job, events = observed[0]
    assert job.transcript.tokens == (5, 6)
    assert events == []
    assert job.sampled_tokens == []
