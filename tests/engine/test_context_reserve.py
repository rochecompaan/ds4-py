import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.engine.events import (
    Prefill,
    ReserveReached,
    Stopped,
    SummaryToken,
    Synced,
    Thinking,
    Token,
    ToolSyntax,
)
from ds4py.engine.transcript import Segment, TranscriptReplacement
from ds4py.errors import EngineError

DSML = (
    '<｜DSML｜tool_calls><｜DSML｜invoke name="read">'
    '<｜DSML｜parameter name="path" string="true">/tmp/x</｜DSML｜parameter>'
    "</｜DSML｜invoke></｜DSML｜tool_calls>"
)


def scripted(*tokens: Any) -> Any:
    index = 0

    def script(prompt: Any, generated: Any) -> Any:
        nonlocal index
        token = tokens[index] if index < len(tokens) else FAKE_EOS
        index += 1
        return token

    return script


@contextmanager
def chat_session(*tokens: Any, ctx: Any = 673, cancel: Any = None, think: Any = ThinkMode.NONE) -> Generator[Any]:
    # The default fake chat prompt plus assistant prefix is 671 tokens.
    with ds4py.open(
        "f.gguf",
        ctx=ctx,
        think=think,
        cancel=cancel,
        native=FakeNative(script=scripted(*tokens)),
    ) as engine:
        session = engine.session()
        session.append_user("hi")
        yield session


def next_pause(events: Any, seen: Any) -> Any:
    for event in events:
        seen.append(event)
        if isinstance(event, ReserveReached):
            return event
    raise AssertionError("chat context reserve did not pause for a decision")


def make_room(session: Any, ctx: Any = 1024) -> None:
    session._worker.call(lambda engine, native: setattr(native, "_ctx", ctx)).result()


def content_events(events: Any) -> Any:
    return [event for event in events if not isinstance(event, (Prefill, Synced, ReserveReached))]


def test_raw_round_still_reports_context_reserve_directly() -> None:
    with ds4py.open("f.gguf", ctx=4, native=FakeNative()) as engine:
        session = engine.session()
        session.append_tokens([5, 6], "user")
        assert list(session.run_round())[-1] == Stopped("context_reserve")


def test_chat_pause_waits_without_eos_and_fail_closes_once() -> None:
    with chat_session(98) as session:
        events = []
        round_ = session.run_round()
        pause = next_pause(round_, events)
        assert pause.needed == 1
        assert not any(isinstance(event, Stopped) for event in events)
        kinds = [segment.kind for segment in session.transcript.segments]
        assert kinds.count("assistant_prefix") == 1
        assert kinds[-1] == "assistant" and "eos" not in kinds

        pause.fail()
        assert list(round_) == [Stopped("context_exhausted")]
        kinds = [segment.kind for segment in session.transcript.segments]
        assert kinds.count("eos") == 1


def test_resume_matches_unpaused_content_and_extends_one_assistant_segment() -> None:
    tokens = tuple(tokenize_rendered("<think>a</think>" + DSML))
    with chat_session(*tokens, ctx=1024, think=ThinkMode.HIGH) as session:
        expected_events = list(session.run_round(max_tokens=100))
        expected_tokens = session.transcript.tokens
        expected_kinds = [segment.kind for segment in session.transcript.segments]

    with chat_session(*tokens, think=ThinkMode.HIGH) as session:
        events = []
        round_ = session.run_round(max_tokens=100)
        pause = next_pause(round_, events)
        make_room(session)
        pause.resume(0)
        events.extend(round_)

        assert any(isinstance(event, Thinking) for event in events)
        assert any(isinstance(event, ToolSyntax) for event in events)
        assert content_events(events) == content_events(expected_events)
        assert session.transcript.tokens == expected_tokens
        kinds = [segment.kind for segment in session.transcript.segments]
        assert kinds == expected_kinds
        assert kinds.count("assistant") == 1
        assert kinds.count("assistant_prefix") == 1


def test_resume_preserves_the_remaining_caller_token_budget() -> None:
    with chat_session(98, 99, 100) as session:
        events = []
        round_ = session.run_round(max_tokens=2)
        pause = next_pause(round_, events)
        make_room(session)
        pause.resume(0)
        events.extend(round_)

        assert [event.id for event in events if isinstance(event, Token)] == [98, 99]
        assert events[-1] == Stopped("max_tokens")


def test_repeated_pauses_and_zero_reserve_require_one_decision_each() -> None:
    with chat_session(98, 99, 100) as session:
        round_ = session.run_round(max_tokens=10)
        first_events = []
        first = next_pause(round_, first_events)
        first.resume(0)

        second_events = []
        second = next_pause(round_, second_events)
        assert [event.id for event in second_events if isinstance(event, Token)] == [99]
        second.resume(0)

        third_events = []
        third = next_pause(round_, third_events)
        assert not any(isinstance(event, Token) for event in third_events)
        third.fail()
        assert list(round_) == [Stopped("context_exhausted")]


def test_advancing_without_a_decision_fails_and_releases_the_round() -> None:
    with chat_session(98) as session:
        events = []
        round_ = session.run_round()
        next_pause(round_, events)
        with pytest.raises(EngineError, match="decision"):
            next(round_)
        assert [segment.kind for segment in session.transcript.segments].count("eos") == 1
        session.append_user("next")


def test_second_decision_cannot_change_an_inflight_continuation() -> None:
    with chat_session(98) as session:
        events = []
        round_ = session.run_round()
        pause = next_pause(round_, events)
        make_room(session)
        pause.resume(0)
        next(round_)
        with pytest.raises(EngineError, match=r"already|decision"):
            pause.fail()
        remainder = list(round_)
        assert remainder[-1] == Stopped("stop_token")


def test_reserve_validation_does_not_consume_the_decision() -> None:
    with chat_session(98) as session:
        events = []
        round_ = session.run_round()
        pause = next_pause(round_, events)
        with pytest.raises(TypeError, match="integer"):
            pause.resume(True)
        with pytest.raises(ValueError, match="non-negative"):
            pause.resume(-1)
        make_room(session)
        pause.resume(0)
        assert list(round_)[-1] == Stopped("stop_token")


def test_closing_a_paused_round_drains_and_preserves_worker_ownership() -> None:
    cancel = threading.Event()
    with chat_session(98, cancel=cancel) as session:
        events = []
        round_ = session.run_round()
        next_pause(round_, events)
        make_room(session)
        round_.close()

        assert not cancel.is_set()
        assert [segment.kind for segment in session.transcript.segments].count("eos") == 1
        session.append_user("next")
        assert list(session.run_round(max_tokens=0))[-1] == Stopped("max_tokens")


def test_replacement_is_allowed_only_at_the_exact_reserve_boundary() -> None:
    with chat_session(98) as session:
        round_ = session.run_round()
        try:
            next(round_)
            replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
            with pytest.raises(EngineError, match="round is open"):
                session.replace_transcript(replacement, reason="compaction")
            pause = next_pause(round_, [])
            make_room(session)
            replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
            session.replace_transcript(replacement, reason="compaction")
            assert session._active_round and session._reserve_boundary
            pause.resume(1)
            next(round_)
            with pytest.raises(EngineError, match="round is open"):
                session.replace_transcript(replacement, reason="compaction")
            assert list(round_)[-1] == Stopped("stop_token")
        finally:
            round_.close()
        assert not session._active_round and not session._reserve_boundary
        replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
        session.replace_transcript(replacement, reason="compaction")


def test_summary_ownership_blocks_replacement_even_if_outer_round_closes() -> None:
    with chat_session(98, 98) as session:
        outer = session.run_round()
        next_pause(outer, [])
        make_room(session, 4096)
        rejected = []

        def replace_during_summary(event: object) -> None:
            replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
            with pytest.raises(EngineError, match="round is open"):
                session.replace_transcript(replacement, reason="compaction")
            outer.close()
            with pytest.raises(EngineError, match="round is open"):
                session.replace_transcript(replacement, reason="compaction")
            rejected.append(True)

        session.on_event(replace_during_summary, kinds={SummaryToken.__name__})
        assert session.prepare_summary(0).summary == "a"
        assert rejected == [True]
        assert not session._active_round and not session._reserve_boundary
        replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
        session.replace_transcript(replacement, reason="compaction")


def test_reserve_replacement_drops_checkpoints_beyond_the_new_end() -> None:
    with chat_session(98) as session:
        session.checkpoint("history")
        outer = session.run_round()
        try:
            next_pause(outer, [])
            replacement = TranscriptReplacement(session.transcript.tokens[:100], (Segment(0, 100, "system", {}),))
            session.replace_transcript(replacement, reason="compaction")
            assert "history" not in session._checkpoint_tokens
            assert session.transcript.tokens == replacement.tokens
            assert session._active_round and session._reserve_boundary
        finally:
            outer.close()
