"""Session identity, lifecycle ordering, event delivery, reset, and think changes."""

import re
from typing import Any, Never, cast
from uuid import UUID

import pytest

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeEngine, FakeNative, FakeSession, reply
from ds4py._native.protocol import SyncStatus
from ds4py.engine.events import Stopped, Synced
from ds4py.engine.lifecycle import SessionEvent
from ds4py.errors import EmptyPromptError, EngineError


class EmptySyncSession(FakeSession):
    """A fake session that accepts the empty reset prompt."""

    def __init__(self, engine: Any, ctx: Any) -> None:
        super().__init__(engine, ctx)
        self.reset_syncs = []

    def sync(self, tokens: Any, *, progress: Any = None, cancel: Any = None) -> Any:
        if not tokens:
            self.reset_syncs.append(())
            cast("Any", self._engine._native).record.append("reset")
            self._tokens = []
            self._prompt_length = 0
            self._valid = True
            self._logits = None
            return SyncStatus.DONE
        return super().sync(tokens, progress=progress, cancel=cancel)


class ResetNative(FakeNative):
    def __init__(self, *, record: Any = None, **fake: Any) -> None:
        super().__init__(**fake)
        self.sessions = []
        self.record = [] if record is None else record

    def open_engine(self, config: Any) -> Any:
        engine = FakeEngine(self)
        if not isinstance(engine, FakeEngine):  # pragma: no cover - defensive
            raise AssertionError("FakeEngine subclass expected")
        original = engine.create_session

        def create_session(ctx: Any) -> Any:
            session = EmptySyncSession(engine, ctx)
            engine._sessions.add(session)
            self.sessions.append(session)
            return session

        engine.create_session = create_session
        return engine


def reset_engine(
    *, ctx: Any = 1024, family: Any = "deepseek4", think: Any = ThinkMode.NONE, record: Any = None, **fake: Any
) -> Any:
    from dataclasses import replace

    info = replace(
        FakeNative()._info,
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )
    native = ResetNative(info=info, record=record, **fake)
    return ds4py.open("f.gguf", ctx=ctx, think=think, native=native), native


EMPTY_PROMPT_REJECTIONS = (
    "empty prompt",
    "prompt must be nonempty and shorter than context",
)


class RejectingResetSession(FakeSession):
    """A strict native session that rejects the empty reset prompt."""

    def __init__(self, engine: Any, ctx: Any, rejection: Any) -> None:
        super().__init__(engine, ctx)
        self.rejection = rejection
        self.fallbacks = []

    def sync(self, tokens: Any, *, progress: Any = None, cancel: Any = None) -> Any:
        if not tokens:
            raise self.rejection
        return super().sync(tokens, progress=progress, cancel=cancel)

    def invalidate(self) -> None:
        self.fallbacks.append("invalidate")
        super().invalidate()

    def rewind(self, pos: Any) -> None:
        self.fallbacks.append(("rewind", pos))
        super().rewind(pos)


class RejectingResetNative(FakeNative):
    def __init__(self, rejection: Any, **fake: Any) -> None:
        super().__init__(**fake)
        self.rejection = rejection
        self.sessions = []

    def open_engine(self, config: Any) -> Any:
        engine = FakeEngine(self)

        def create_session(ctx: Any) -> Any:
            session = RejectingResetSession(engine, ctx, self.rejection)
            engine._sessions.add(session)
            self.sessions.append(session)
            return session

        engine.create_session = create_session
        return engine


def rejecting_engine(rejection: Any, *, script: Any = None) -> Any:
    native = RejectingResetNative(rejection, script=script or reply(b"ok"))
    return ds4py.open("f.gguf", ctx=1024, think=ThinkMode.NONE, native=native), native


class CloseCounter:
    def __init__(self, record: Any = None) -> None:
        self.closes = 0
        self._record = record

    def close(self) -> None:
        self.closes += 1
        if self._record is not None:
            self._record.append("close")


def kinds(items: Any) -> Any:
    return [
        item if isinstance(item, str) else item.kind if isinstance(item, SessionEvent) else type(item).__name__
        for item in items
    ]


def test_session_id_is_a_uuid4_hex_and_stays_across_prompts() -> None:
    engine, _ = reset_engine(script=reply(b"ok"))
    with engine:
        session = engine.session()
        first = session.id
        assert re.fullmatch(r"[0-9a-f]{32}", first)
        assert UUID(first).version == 4
        session.append_user("one")
        assert list(session.run_round(max_tokens=2))[-1] == Stopped("max_tokens")
        assert session.id == first
        session.append_user("two")
        assert list(session.run_round(max_tokens=2))[-1] == Stopped("max_tokens")
        assert session.id == first


def test_new_publishes_end_then_closes_state_then_resets_then_starts() -> None:
    record = []
    engine, native = reset_engine(script=reply(b"ok"), record=record)
    with engine:
        session = engine.session()
        state = CloseCounter(record)
        assert session.state("jobs", lambda: state) is state
        session.on_event(lambda event: record.append(event))
        session.append_user("hello")
        list(session.run_round(max_tokens=2))
        session.checkpoint("turn")
        assert "turn" in session._checkpoint_tokens
        sampling = session.engine.sampling_defaults
        worker = session.engine._worker
        old = session.id
        record.clear()

        new = session.new()

        assert kinds(record) == ["session_end", "close", "reset", "session_start"]
        assert isinstance(record[0], SessionEvent) and record[0].session_id == old
        assert record[0].payload == {"old": old}
        assert state.closes == 1
        assert isinstance(record[3], SessionEvent) and record[3].session_id == new
        assert record[3].payload == {"old": old, "new": new}
        assert new != old and session.id == new
        assert session.transcript.tokens == () and session.transcript.segments == ()
        assert session.ctx_used == 0
        assert session._checkpoint_tokens == {}
        assert session.engine._worker._checkpoints == {}
        assert native.sessions[0].reset_syncs == [()]
        assert session.engine.sampling_defaults is sampling
        assert session.engine._worker is worker
        replacement = session.state("jobs", lambda: CloseCounter())
        assert replacement is not state and state.closes == 1


@pytest.mark.parametrize("message", EMPTY_PROMPT_REJECTIONS)
def test_new_rewinds_when_the_native_rejects_the_empty_reset_prompt(message: Any) -> None:
    engine, native = rejecting_engine(EmptyPromptError(message))
    with engine:
        session = engine.session()
        session.append_user("hello")
        list(session.run_round(max_tokens=2))
        session.checkpoint("turn")
        native_session = native.sessions[0]
        old = session.id
        events = []
        session.on_event(lambda event: events.append(event))

        new = session.new()

        assert new == session.id != old
        assert kinds(events) == ["session_end", "session_start"]
        assert events[0].session_id == old
        assert events[1].session_id == new
        assert native_session.fallbacks == ["invalidate", ("rewind", 0)]
        assert native_session._tokens == [] and not native_session._valid
        assert session.transcript.tokens == () and session.transcript.segments == ()
        assert session._checkpoint_tokens == {}
        assert session.engine._worker._checkpoints == {}
        # The rewound native session rebuilds from position zero on the next sync.
        session.append_user("after")
        assert list(session.run_round(max_tokens=1))[-1] == Stopped("max_tokens")


@pytest.mark.parametrize(
    "failure",
    [
        EngineError("native device failure during sync"),
        EngineError("empty prompt"),
    ],
)
def test_new_propagates_an_unrelated_native_sync_error_without_starting_a_session(failure: Any) -> None:
    engine, native = rejecting_engine(failure)
    with engine:
        session = engine.session()
        session.append_user("hello")
        list(session.run_round(max_tokens=2))
        checkpointed = session.checkpoint("turn")
        native_session = native.sessions[0]
        native_tokens = tuple(native_session._tokens)
        _, snapshot = session.engine._worker._checkpoints["turn"]
        facade_checkpoints = dict(session._checkpoint_tokens)
        transcript = session.transcript.tokens
        segments = session.transcript.segments
        worker = session.engine._worker
        old = session.id
        events = []
        session.on_event(lambda event: events.append(event))

        with pytest.raises(EngineError) as caught:
            session.new()

        assert caught.value is failure
        assert session.id == old
        assert session.transcript.tokens == transcript
        assert session.transcript.segments == segments
        assert tuple(native_session._tokens) == native_tokens
        assert native_session.fallbacks == []
        assert kinds(events) == ["session_end"]
        assert session.engine._worker is worker
        # The failed reset must not mutate checkpoint ownership: both mappings
        # and the native snapshot survive, so checkpoint operations still work.
        assert facade_checkpoints == {"turn": checkpointed.tokens}
        assert session._checkpoint_tokens == facade_checkpoints
        assert session.engine._worker._checkpoints == {"turn": (native_tokens, snapshot)}
        assert not snapshot.closed
        # The worker owns a still-usable session after the failed reset.
        session.append_user("after")
        assert list(session.run_round(max_tokens=1))[-1] == Stopped("max_tokens")
        session.truncate(len(native_tokens))
        assert session.plan().kind == "restore" and session.plan().checkpoint == "turn"
        session.drop_checkpoint("turn")
        assert snapshot.closed
        assert session._checkpoint_tokens == {}
        assert session.engine._worker._checkpoints == {}


def test_switch_publishes_exact_order_and_runs_restore_while_idle() -> None:
    engine, _ = reset_engine(script=reply(b"ok"))
    with engine:
        session = engine.session()
        record = []
        state = CloseCounter(record)
        session.state("jobs", lambda: state)
        session.append_user("kept")
        tokens = session.transcript.tokens
        old = session.id
        session.on_event(lambda event: record.append(event))
        target = "a" * 32

        def restore() -> None:
            record.append("restore")

        returned = session.switch(target, restore)

        assert returned == target == session.id
        assert kinds(record) == [
            "switching",
            "session_end",
            "close",
            "restore",
            "session_start",
            "switched",
        ]
        assert record[0].session_id == old and record[0].payload == {"old": old, "new": target}
        assert record[1].session_id == old and record[1].payload == {"old": old}
        assert record[3] == "restore"
        assert record[4].session_id == target
        assert record[5].session_id == target
        assert state.closes == 1
        assert session.transcript.tokens == tokens
        assert session.state("jobs", lambda: CloseCounter()) is not state


def test_switch_restore_failure_still_starts_the_target_session() -> None:
    engine, _ = reset_engine(script=reply(b"ok"))
    with engine:
        session = engine.session()
        record = []
        state = CloseCounter(record)
        session.state("jobs", lambda: state)
        old = session.id
        target = "b" * 32
        session.on_event(lambda event: record.append(event))
        failure = EngineError("restore rejected")

        def restore() -> Never:
            record.append("restore")
            raise failure

        with pytest.raises(EngineError) as caught:
            session.switch(target, restore)

        assert caught.value is failure
        assert session.id == target
        assert kinds(record) == [
            "switching",
            "session_end",
            "close",
            "restore",
            "session_start",
        ]
        assert record[-1].session_id == target
        assert record[-1].payload == {
            "old": old,
            "new": target,
            "status": "error",
            "error": "restore rejected",
        }
        assert state.closes == 1
        assert session.state("jobs", lambda: CloseCounter()) is not state


def test_fork_emits_only_forked_and_keeps_transcript_kv_checkpoints_and_state() -> None:
    engine, native = reset_engine(script=reply(b"ok"))
    with engine:
        session = engine.session()
        state = CloseCounter()
        session.state("jobs", lambda: state)
        session.append_user("kept")
        list(session.run_round(max_tokens=2))
        session.checkpoint("turn")
        tokens = session.transcript.tokens
        checkpoints = dict(session._checkpoint_tokens)
        calls = list(native.sessions[0].calls)
        worker = session.engine._worker
        old = session.id
        record = []
        session.on_event(lambda event: record.append(event))

        new = session.fork()

        assert kinds(record) == ["forked"]
        assert record[0].session_id == old
        assert record[0].payload == {"old": old, "new": new}
        assert session.id == new and new != old
        assert session.transcript.tokens == tokens
        assert session._checkpoint_tokens == checkpoints
        assert native.sessions[0].calls == calls
        assert session.engine._worker is worker
        assert state.closes == 0
        assert session.state("jobs", lambda: CloseCounter()) is state


def test_engine_events_are_published_before_the_consumer_receives_them() -> None:
    engine, _ = reset_engine(script=reply(b"ok"))
    with engine:
        session = engine.session()
        session.append_tokens([5, 6], "user")
        published = []

        def listener(event: Any) -> None:
            published.append(event)

        session.on_event(listener, kinds={"Prefill", "Synced", "Token", "Stopped"})
        consumer = list(session.run_round(max_tokens=5))

        assert kinds(published) == kinds(consumer)
        assert all(
            bus_kind == event.payload["event"].__class__.__name__ and event.payload["event"] is delivered_event
            for bus_kind, event, delivered_event in zip(kinds(published), published, consumer)
        )
        assert consumer[-1] == Stopped("stop_token")


def test_on_event_filters_and_unsubscribes_idempotently() -> None:
    engine, _ = reset_engine()
    with engine:
        session = engine.session()
        seen = []
        unsubscribe = session.on_event(
            lambda event: seen.append(event),
            kinds=("status",),
        )
        session.emit("status", text="ready")
        session.emit("other", text="hidden")
        unsubscribe()
        unsubscribe()
        session.emit("status", text="late")

        assert [(event.kind, event.payload) for event in seen] == [
            ("status", {"text": "ready"}),
        ]
        assert all(isinstance(event, SessionEvent) for event in seen)


def test_listener_failure_is_reported_without_hiding_events_or_corrupting_native_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine, _ = reset_engine(script=reply(b"ok"))
    with engine:
        session = engine.session()
        session.append_tokens([5, 6], "user")
        observed = []

        def broken(event: Any) -> None:
            if event.kind == "Synced":
                raise RuntimeError("listener exploded")

        session.on_event(broken)
        session.on_event(lambda event: observed.append(event.kind))
        events = list(session.run_round(max_tokens=5))

        assert observed == kinds(events)
        assert events[-1] == Stopped("stop_token")
        assert session.ctx_used == len(session.transcript)
        session.append_tokens([7], "user")
        assert list(session.run_round(max_tokens=0))[-1] == Stopped("max_tokens")
        assert "listener exploded" in capsys.readouterr().err


def test_set_think_invalidates_the_bound_prefix_without_mutating_a_live_round() -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    from ds4py.binding import MessageBinder

    engine, _ = reset_engine(script=reply(b"ok"), ctx=8192, family="deepseek41")
    with engine:
        session = engine.session()
        binder = MessageBinder(session)
        messages = [ModelRequest(parts=[UserPromptPart("hello")])]
        binder.prepare(messages)
        before = session.transcript.tokens

        session.set_think(ThinkMode.MAX)

        assert session.engine.think == ThinkMode.HIGH
        change = binder.prepare(messages)
        assert change.kind == "rewrite" and change.start == 0
        assert session.transcript.tokens != before
        assert b"Reasoning Effort" in b"".join(session.engine.token_text(token) for token in session.transcript.tokens)

        rnd = session.run_round(max_tokens=0)
        try:
            assert isinstance(next(event for event in rnd if isinstance(event, Synced)), Synced)
            with pytest.raises(EngineError, match="round"):
                session.set_think(ThinkMode.NONE)
        finally:
            rnd.close()
