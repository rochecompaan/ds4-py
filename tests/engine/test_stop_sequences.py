from typing import Any, cast

import pytest
from fake_steps import FakeMultiTokenStep

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative, FakeSession
from ds4py._native.protocol import EngineInfo
from ds4py.engine.decode import DecodeContext, GenerateJob, Verdict, generate
from ds4py.engine.events import Stopped, Token
from ds4py.engine.pause import ControlSpan
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.stream import RoundStream
from ds4py.engine.transcript import Transcript

INFO = EngineInfo("fake", 0, "deepseek41", True, False, False, False, 0, 0, 1000)


class ScriptedSession:
    def __init__(self, tokens: Any, *, ctx: Any = 100) -> None:
        self.tokens_to_sample = list(tokens)
        self.tokens = []
        self.rewinds = []
        self.sample_calls = 0
        self._ctx = ctx

    def sample(self, *args: Any) -> Any:
        self.sample_calls += 1
        return self.tokens_to_sample.pop(0)

    def eval(self, token: Any) -> None:
        self.tokens.append(token)

    def rewind(self, pos: Any) -> None:
        self.rewinds.append(pos)
        del self.tokens[pos:]

    def pos(self) -> int:
        return len(self.tokens)

    def ctx(self) -> Any:
        return self._ctx


class TextEngine:
    def __init__(self, texts: Any = None) -> None:
        self.texts = texts or {}

    def token_text(self, token: Any) -> bytes:
        return bytes(self.texts.get(token, b"" if token == 0 else bytes([token])))

    def is_stop(self, token: Any) -> Any:
        return token == 0

    def is_stop_for_think_mode(self, token: Any, mode: Any) -> Any:
        return token == 0


def decode(
    tokens: Any,
    stop_sequences: Any,
    *,
    texts: Any = None,
    stream: Any = None,
    control: Any = None,
    max_tokens: Any = 20,
    resume: Any = False,
    step: Any = None,
    ctx: Any = 100,
) -> Any:
    session = ScriptedSession(tokens, ctx=ctx)
    transcript = Transcript()
    control = control or ControlSpan(stop_sequences)
    if stream is not None:
        stream.control = control
    job = GenerateJob(
        SamplingPolicy(),
        Rng(1),
        max_tokens,
        transcript,
        stream=stream,
        control=control,
        resume=resume,
    )
    reason = generate(
        cast("Any", TextEngine(texts)),
        cast("Any", session),
        job,
        cancelled=lambda: False,
        emit=lambda event: None,
        step=step,
    )
    return reason, session, transcript, control


def test_generate_job_rejects_a_different_stream_controller() -> None:
    stream = RoundStream(INFO, ThinkMode.NONE, control=ControlSpan([b"stop"]))
    original = stream.control
    with pytest.raises(ValueError, match="identical"):
        GenerateJob(
            SamplingPolicy(),
            Rng(1),
            5,
            Transcript(),
            stream=stream,
            control=ControlSpan([b"other"]),
        )
    assert stream.control is original
    assert stream.control.sequences == (b"stop",)


def test_stop_sequence_matches_across_tokens_and_stops_after_matching_token() -> None:
    reason, session, transcript, control = decode(
        [ord("s"), ord("t"), ord("o"), ord("p"), 5, 0],
        [b"stop"],
    )
    assert reason == "stop_sequence" and transcript.tokens == tuple(b"stop")
    assert session.tokens == list(b"stop") and session.tokens_to_sample == [5, 0]
    assert control.stop_sequence == b"stop"


def test_stop_sequence_never_matches_thinking_but_matches_following_control_text() -> None:
    texts = {1: b"<think>", 2: b"stop", 3: b"</think>", 4: b"stop"}
    stream = RoundStream(INFO, ThinkMode.HIGH)
    reason, _, transcript, _ = decode(
        [1, 2, 3, 4, 0],
        [b"stop"],
        texts=texts,
        stream=stream,
    )
    assert reason == "stop_sequence" and transcript.tokens == (1, 2, 3, 4)
    assert stream.sampled_tokens == [1, 2, 3, 4]


def test_active_sequences_require_a_matching_prefix_after_region_start() -> None:
    stream = RoundStream(INFO, ThinkMode.NONE)
    assert stream.active_stop_sequences == []
    reason, _, _, control = decode([ord("x"), ord("s"), 0], [b"stop"], stream=stream, max_tokens=2)
    assert reason == "max_tokens" and control.active_stop_sequences == [b"stop"]


def test_context_limit_does_not_turn_an_unmatched_prefix_into_a_stop() -> None:
    reason, session, transcript, control = decode(
        [ord("s"), ord("t"), 0],
        [b"stop"],
        ctx=3,
    )
    assert reason == "context_reserve" and control.stop_sequence is None
    assert control.matcher.pending == 2 and session.sample_calls == 2
    assert transcript.tokens == (ord("s"), ord("t"))


def test_max_tokens_counts_accepted_sampled_tokens_not_characters() -> None:
    reason, _, transcript, _ = decode(
        [1, 2, 0],
        [b"never"],
        texts={1: "é".encode(), 2: b"abcdef"},
        max_tokens=2,
    )
    assert reason == "max_tokens" and transcript.tokens == (1, 2)


def test_resume_retains_matcher_state_and_clears_only_stop_field() -> None:
    control = ControlSpan([b"stop"])
    reason, _, _, _ = decode(
        [ord("s"), ord("t"), ord("o"), ord("p"), 0],
        [b"stop"],
        control=control,
    )
    assert reason == "stop_sequence"
    control.matcher.feed(b"st")
    reason, _, _, _ = decode(
        [ord("o"), ord("p"), 0],
        [b"stop"],
        control=control,
        resume=True,
    )
    assert reason == "stop_sequence" and control.stop_sequence == b"stop"


def test_finish_guards_accepted_token_counters_and_later_commits() -> None:
    transcript = Transcript()
    job = GenerateJob(SamplingPolicy(), Rng(1), 10, transcript)
    ctx = DecodeContext(
        cast("Any", TextEngine()),
        job,
        limit=10,
        limit_reason="max_tokens",
        cancelled=lambda: False,
        emit=lambda event: None,
    )
    ctx.accept(1)
    assert ctx.commit(1) is Verdict.CONTINUE
    ctx.finish("pause", finish_parser=False)
    assert ctx.accept(2) is Verdict.STOP_AFTER
    assert ctx.commit(2) is Verdict.STOP_AFTER
    assert ctx._accepted == 1 and transcript.tokens == (1,)


def test_multi_token_step_rewinds_unkept_tail_and_stops_at_sequence() -> None:
    tokens = [ord("s"), ord("t"), ord("o"), ord("p"), 5, 0]
    step = FakeMultiTokenStep([[ord("t"), ord("o"), ord("p"), 5]])
    reason, session, transcript, _ = decode(tokens, [b"stop"], step=step)
    assert reason == "stop_sequence" and transcript.tokens == tuple(b"stop")
    assert session.tokens == list(b"stop") and session.rewinds == [4]
    assert session.sample_calls == 1


def test_public_chat_stop_sequence_leaves_open_and_next_round_resumes() -> None:
    stop = tuple(byte + 1 for byte in b"stop")

    def script(prompt: Any, generated: Any) -> Any:
        source = (
            b"next"
            if any(tuple(prompt[index : index + len(stop)]) == stop for index in range(len(prompt) - len(stop) + 1))
            else b"stop"
        )
        return source[len(generated)] + 1 if len(generated) < len(source) else 0

    with ds4py.open("f.gguf", think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
        session = engine.session()
        session.append_user("hi")
        first = list(session.run_round(stop_sequences=[b"stop"]))
        assert first[-1] == Stopped("stop_sequence")
        assert session.transcript.segments[-1].kind == "assistant"
        second = list(session.run_round(stop_sequences=[b"stop"]))
        assert second[-1] == Stopped("stop_token")
        assert session.transcript.segments[-1].kind == "eos"
        tokens = session.transcript.tokens
        expected = tuple(byte + 1 for byte in b"stopnext")
        assert any(
            tokens[index : index + len(expected)] == expected for index in range(len(tokens) - len(expected) + 1)
        )


@pytest.mark.parametrize("chat", [False, True], ids=["raw", "chat"])
def test_paused_round_rejects_changed_stop_sequences(chat: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    stop = tuple(byte + 1 for byte in b"stop")

    def script(prompt: Any, generated: Any) -> Any:
        source = (
            b"next"
            if any(tuple(prompt[index : index + len(stop)]) == stop for index in range(len(prompt) - len(stop) + 1))
            else b"stop"
        )
        return source[len(generated)] + 1 if len(generated) < len(source) else 0

    sample_calls = 0
    original_sample = FakeSession.sample

    def counting_sample(self: Any, *args: Any) -> Any:
        nonlocal sample_calls
        sample_calls += 1
        return original_sample(self, *args)

    monkeypatch.setattr(FakeSession, "sample", counting_sample)
    with ds4py.open("f.gguf", think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
        session = engine.session()
        if chat:
            session.append_user("hi")
        else:
            session.append_tokens([5, 6], "user")
        assert list(session.run_round(stop_sequences=[b"stop"]))[-1] == Stopped("stop_sequence")
        sampled_before = sample_calls
        with pytest.raises(
            ValueError,
            match=r"^stop_sequences cannot change while resuming a round$",
        ):
            list(session.run_round(stop_sequences=[b"other"]))
        assert sample_calls == sampled_before
        assert list(session.run_round(stop_sequences=[b"stop"]))[-1] == Stopped("stop_token")
        assert sample_calls > sampled_before


def test_terminal_marker_prefix_is_matched_independently_of_display() -> None:
    stream = RoundStream(INFO, ThinkMode.NONE)
    reason, _, transcript, control = decode(
        [60],
        [b"<"],
        stream=stream,
        max_tokens=1,
    )
    assert reason == "stop_sequence"
    assert transcript.tokens == (60,) and control.stop_sequence == b"<"


def test_new_user_retires_paused_stream_and_starts_fresh_assistant() -> None:
    def script(prompt: Any, generated: Any) -> Any:
        return b"stop"[len(generated)] + 1 if len(generated) < 4 else 0

    with ds4py.open("f.gguf", think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
        session = engine.session()
        session.append_user("first")
        first = list(session.run_round(stop_sequences=[b"stop"]))
        assert first[-1] == Stopped("stop_sequence")
        old_stream = session._paused_stream
        session.append_user("second")
        assert session._paused_stream is None
        assert [s.kind for s in session.transcript.segments][-2:] == ["eos", "user"]
        second = list(session.run_round(stop_sequences=[b"stop"]))
        assert second[-1] == Stopped("stop_sequence")
        assert [event.id for event in first if isinstance(event, Token)] == [116, 117, 112, 113]
        assert [event.id for event in second if isinstance(event, Token)] == [116, 117, 112, 113]
        assert [s.kind for s in session.transcript.segments][-6:] == [
            "assistant_prefix",
            "assistant",
            "eos",
            "user",
            "assistant_prefix",
            "assistant",
        ]
        assert session._paused_stream is not old_stream
        assert len(cast("Any", session._paused_stream).sampled_tokens) == 4
        assert [s.kind for s in session.transcript.segments].count("assistant_prefix") == 2


@pytest.mark.parametrize(
    ("mutation", "kind"),
    [
        (lambda session: session.append_tool_results(["result"]), "tool"),
        (lambda session: session.append_system_note("note"), "note"),
    ],
)
def test_message_mutations_close_a_paused_assistant(mutation: Any, kind: Any) -> None:
    with ds4py.open("f.gguf", think=ThinkMode.NONE, native=FakeNative()) as engine:
        session = engine.session()
        session.append_user("first")
        assert list(session.run_round(stop_sequences=[b"ok"]))[-1] == Stopped("stop_sequence")
        mutation(session)
        assert session._paused_stream is None
        assert [segment.kind for segment in session.transcript.segments][-2:] == ["eos", kind]
