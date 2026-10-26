from dataclasses import replace
from pathlib import Path

import pytest

from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py._native.protocol import EngineConfig, ThinkMode
from ds4py.engine.decode import GenerateJob, generate
from ds4py.engine.events import (
    Stopped, TextDelta, Thinking, Token, ToolCalls, ToolSyntax,
)
from ds4py.engine.pause import ControlSpan
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.stream import RoundStream
from ds4py.engine.transcript import Transcript
from ds4py.engine.worker import EngineWorker, SyncJob

DSML = ("<｜DSML｜tool_calls><｜DSML｜invoke name=\"read\">"
        "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/x</｜DSML｜parameter>"
        "</｜DSML｜invoke></｜DSML｜tool_calls>")
GLM_TWO = ("<tool_call>first<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
           "<tool_call>second<arg_key>y</arg_key><arg_value>2</arg_value></tool_call>")
QWEN = ("<tool_call><function=read><parameter=path>/tmp/x</parameter>"
        "</function></tool_call>")


def scripted(tokens):
    tokens = tuple(tokens)
    return lambda prompt, generated: tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS


def native_for(family, text):
    info = replace(
        FakeNative()._info,
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )
    return FakeNative(info=info, script=scripted(tokenize_rendered(text)))


@pytest.fixture
def workers():
    created = []
    yield created
    for worker in created:
        worker.close()


def run_native(workers, native, *, chat=True, max_tokens=1000, ctx=1024,
               think=ThinkMode.NONE, policy=None):
    worker = EngineWorker(native, EngineConfig(Path("f"), ctx=ctx))
    workers.append(worker)
    transcript = Transcript()
    transcript.append([5, 6], "user")
    list(worker.submit(SyncJob((5, 6))))
    stream = RoundStream(worker.info, think) if chat else None
    job = GenerateJob(
        policy or SamplingPolicy(), Rng(1), max_tokens, transcript,
        think_mode=think if chat else None, stream=stream,
        control=stream.control if stream is not None else ControlSpan(),
    )
    events = list(worker.submit(job))
    return events, transcript, stream, worker


def event_calls(events):
    return [event for event in events if isinstance(event, ToolCalls)]


@pytest.mark.parametrize("family,text,name,path", [
    ("deepseek4", DSML, "read", "/tmp/x"),
    ("glm", GLM_TWO, "first", "1"),
    ("qwen", QWEN, "read", "/tmp/x"),
])
def test_chat_tool_calls_stop_and_emit_before_stopped(workers, family, text, name, path):
    events, transcript, stream, _ = run_native(workers, native_for(family, text))
    calls = event_calls(events)
    assert len(calls) == 1 and events[-1] == Stopped("tool_call_complete")
    assert events.index(calls[0]) == len(events) - 2
    assert calls[0].calls[0].name == name
    assert calls[0].calls[0].args["path" if family != "glm" else "x"] == path
    if family == "glm":
        assert [call.name for call in calls[0].calls] == ["first", "second"]
    assert stream.finalized and not stream.parser.wants_greedy_sampling()


def test_parser_terminal_overrides_simultaneous_caller_limit(workers):
    ids = tokenize_rendered(DSML)
    events, _, _, _ = run_native(workers, native_for("deepseek4", DSML), max_tokens=len(ids))
    assert events[-1] == Stopped("tool_call_complete")
    assert len(event_calls(events)) == 1


def test_context_reserve_does_not_finalize_stream(workers):
    events, _, stream, _ = run_native(
        workers, native_for("qwen", QWEN), ctx=10, max_tokens=100,
    )
    assert events[-1] == Stopped("context_reserve")
    assert not stream.finalized and stream.snapshot.state != "ERROR"
    assert event_calls(events) == []


def test_structural_tokens_switch_actual_sampling_and_values_restore_it(workers):
    policy = SamplingPolicy(temperature=0.6, top_k=4, top_p=0.9, min_p=0.03)
    events, _, _, worker = run_native(
        workers, native_for("deepseek4", DSML), policy=policy,
    )
    samples = worker.call(lambda e, s: s.samples).result()
    normal = (0.6, 4, 0.9, 0.03)
    greedy = (0.0, 0, 1.0, 0.0)
    assert events[-1] == Stopped("tool_call_complete")
    assert samples[0] == normal and greedy in samples
    first_greedy = samples.index(greedy)
    assert normal in samples[first_greedy + 1:]


@pytest.mark.parametrize("family,text", [("deepseek4", DSML), ("glm", GLM_TWO), ("qwen", QWEN)])
def test_markup_in_thinking_is_parser_error_without_calls(workers, family, text):
    events, _, _, _ = run_native(
        workers, native_for(family, text), think=ThinkMode.HIGH,
    )
    assert events[-1] == Stopped("parser_error")
    assert event_calls(events) == []


def test_tool_markup_after_think_close_is_allowed(workers):
    for family, text in (("deepseek4", DSML), ("glm", GLM_TWO), ("qwen", QWEN)):
        events, _, _, _ = run_native(
            workers, native_for(family, "</think>" + text), think=ThinkMode.HIGH,
        )
        assert events[-1] == Stopped("tool_call_complete")
        assert len(event_calls(events)) == 1


def test_eos_finalizes_an_incomplete_call_as_parser_error(workers):
    text = "<tool_call><function=read><parameter=path>unfinished"
    events, _, stream, _ = run_native(workers, native_for("qwen", text))
    assert events[-1] == Stopped("parser_error")
    assert stream.snapshot.state == "ERROR" and event_calls(events) == []


def test_raw_round_has_no_parser_or_display_events(workers):
    events, _, stream, _ = run_native(workers, native_for("glm", GLM_TWO), chat=False)
    assert stream is None and events[-1] == Stopped("stop_token")
    assert all(isinstance(event, (Token, Stopped)) for event in events)
    assert event_calls(events) == []


class StopEngine:
    def __init__(self, texts=None):
        self.info = FakeNative()._info
        self.texts = texts or {}
        self.raw_stops = []
        self.chat_stops = []

    def is_stop(self, token):
        self.raw_stops.append(token)
        return token == 0

    def is_stop_for_think_mode(self, token, mode):
        self.chat_stops.append((token, mode))
        return token == 0

    def token_text(self, token):
        return self.texts.get(token, b"x")


class StopSession:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.seen = []

    def ctx(self): return 100
    def pos(self): return 10
    def sample(self, *args): return self.tokens.pop(0)
    def eval(self, token): self.seen.append(("eval", token))


def test_raw_and_chat_use_their_respective_stop_bindings():
    for chat in (False, True):
        engine, session, transcript = StopEngine(), StopSession([0]), Transcript()
        stream = RoundStream(engine.info, ThinkMode.HIGH) if chat else None
        reason = generate(
            engine, session,
            GenerateJob(SamplingPolicy(), Rng(1), 5, transcript,
                        think_mode=ThinkMode.HIGH if chat else None, stream=stream,
                        control=stream.control if stream is not None else ControlSpan()),
            cancelled=lambda: False, emit=lambda event: None,
        )
        assert reason == "stop_token"
        assert bool(engine.chat_stops) == chat
        assert bool(engine.raw_stops) != chat
        if chat:
            assert engine.chat_stops == [(0, ThinkMode.HIGH)]


def test_commit_order_is_eval_transcript_token_then_stream_events():
    engine = StopEngine({7: b"<think>", 8: b"x"})
    session, transcript, observed = StopSession([7, 8, 0]), Transcript(), []
    stream = RoundStream(engine.info, ThinkMode.NONE)
    reason = generate(
        engine, session,
        GenerateJob(SamplingPolicy(), Rng(1), 5, transcript,
                    think_mode=ThinkMode.NONE, stream=stream, control=stream.control),
        cancelled=lambda: False,
        emit=lambda event: observed.append((event, transcript.tokens)),
    )
    assert reason == "stop_token"
    assert [type(event) for event, _ in observed] == [Token, Thinking, Token, TextDelta]
    assert observed[0][1] == (7,) and observed[1][1] == (7,)
    assert observed[2][1] == (7, 8) and observed[3][1] == (7, 8)
    assert observed[1][0] == Thinking(True)
    assert observed[3][0] == TextDelta("x", True)
    assert session.seen == [("eval", 7), ("eval", 8)]


def test_interruption_does_not_finalize_stream_or_publish_calls():
    engine, session, transcript = StopEngine(), StopSession([7, 8, 0]), Transcript()
    stream = RoundStream(engine.info, ThinkMode.NONE)
    answers = iter([False, False, True, True])
    events = []
    reason = generate(
        engine, session,
        GenerateJob(SamplingPolicy(), Rng(1), 5, transcript,
                    think_mode=ThinkMode.NONE, stream=stream, control=stream.control),
        cancelled=lambda: next(answers, True), emit=events.append,
    )
    assert reason == "interrupted" and not stream.finalized
    assert stream.snapshot.state != "ERROR"
    assert event_calls(events) == []
