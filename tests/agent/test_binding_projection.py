"""Exact projection must follow preparation without consuming any live state."""

from copy import deepcopy
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from test_binding import fake_engine, tool, user

from ds4py._native.fake import FakeSession
from ds4py.binding import MessageBinder
from ds4py.capabilities.system_notes import DS4SystemNotes
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.session import Session
from ds4py.errors import EngineError


def snapshot(binder: MessageBinder, messages: list[ModelMessage]) -> object:
    session = binder.session

    def native_snapshot(engine: object, native: object) -> object:
        assert isinstance(native, FakeSession)
        return native.tokens(), list(native.calls)

    native = session._worker.call(native_snapshot).result()
    notes = binder.note_provider
    return deepcopy(
        (
            messages,
            binder.records(),
            binder._bound_messages,
            binder._signature,
            binder._restored,
            binder._restored_notes,
            None
            if notes is None
            else (notes.hints_enabled, notes._date_emitted, notes._hints_full_emitted, notes._hints_applied),
            session.transcript.chat,
            session.transcript.tokens,
            [(s.start, s.end, s.kind, dict(s.meta)) for s in session.transcript.segments],
            native,
        )
    )


def assert_projection(binder: MessageBinder, messages: list[ModelMessage], expected: int, *, bound: bool) -> None:
    before = snapshot(binder, messages)
    count = binder.count_bound_request if bound else binder.count_rendered_request
    counts = [count(messages) for _ in range(2)]
    assert snapshot(binder, messages) == before
    binder.prepare(messages)
    actual = len(binder.session.transcript) + len(
        PromptBuilder(binder.session.engine).assistant_prefix(binder.session.transcript.tokens)
    )
    assert actual == expected
    assert counts == [expected, expected]


def test_pending_note_rewrites_unchanged_bound_request() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        notes = DS4SystemNotes(session, hints=True, clock=lambda: "fixed")
        binder.note_provider = notes
        messages: list[ModelMessage] = [user("go")]
        binder.prepare(messages)
        notes.hints_enabled = False
        # Historical projection: 1435; preparation: 1528.
        assert_projection(binder, messages, 1528, bound=True)


def test_fresh_configured_notes_match_prepare() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        binder.note_provider = DS4SystemNotes(session, hints=True, clock=lambda: "fixed")
        # Historical projection: 676; preparation: 1440.
        assert_projection(binder, [user("request")], 1440, bound=False)


def test_preview_consumes_hints_once_across_suffix_requests() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        notes = DS4SystemNotes(session, clock=lambda: "fixed")
        binder.note_provider = notes
        messages: list[ModelMessage] = [user("go")]
        binder.prepare(messages)
        notes.hints_enabled = True
        messages.extend([user("one"), user("two")])
        # Historical projection: 2113; preparation: 1443.
        assert_projection(binder, messages, 1443, bound=True)


def test_clock_failure_leaves_binding_messages_and_provider_untouched() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)

        def fail_clock() -> str:
            raise ValueError("clock failed")

        binder.note_provider = DS4SystemNotes(session, clock=fail_clock)
        messages: list[ModelMessage] = [user("go")]
        before = snapshot(binder, messages)
        with pytest.raises(ValueError, match="clock failed"):
            binder.prepare(messages)
        assert snapshot(binder, messages) == before


@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
def test_partial_append_rolls_back_to_last_completed_message(fail_at: int, monkeypatch: pytest.MonkeyPatch) -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        notes = DS4SystemNotes(session, hints=True, clock=lambda: "fixed")
        binder.note_provider = notes
        messages: list[ModelMessage] = [user("one"), user("two")]
        session.start_chat()
        preamble = session.transcript.tokens
        calls = 0

        def fail_append(self: Session, tokens: Any, kind: str, meta: Any = None) -> Any:
            nonlocal calls
            calls += 1
            result = Session.append_tokens(self, tokens, kind, meta)
            if calls == fail_at:
                raise EngineError("append failed")
            return result

        monkeypatch.setattr(session, "append_tokens", fail_append.__get__(session))
        with pytest.raises(EngineError, match="append failed"):
            binder.prepare(messages)
        if fail_at <= 3:
            assert session.transcript.tokens == preamble
            assert binder.records() == ()
            assert not notes._date_emitted
            assert not notes._hints_full_emitted
            assert all(message.metadata is None for message in messages)
        else:
            assert len(binder.records()) == 1
            assert len(session.transcript) == binder.records()[0].end
            assert [s.meta.get("subtype") for s in session.transcript.segments if s.kind == "note"] == [
                "date",
                "hints_full",
            ]
            assert notes._date_emitted and notes._hints_full_emitted
            assert messages[0].metadata is not None
            assert messages[1].metadata is None
        monkeypatch.undo()
        binder.prepare(messages)
        assert len(binder.records()) == 2
        assert [s.meta.get("subtype") for s in session.transcript.segments if s.kind == "note"] == [
            "date",
            "hints_full",
        ]


def test_bound_projection_requires_initialized_chat_but_accepts_zero_records() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        with pytest.raises(EngineError, match="initialized chat"):
            binder.count_bound_request([user("go")])
        session.start_chat()
        binder.note_provider = DS4SystemNotes(session, hints=True, clock=lambda: "fixed")
        assert binder.count_bound_request([user("request")]) == 1440
        session.new()
        binder.forget()
        with pytest.raises(EngineError, match="initialized chat"):
            binder.count_bound_request([user("go")])


@pytest.mark.parametrize(
    "transition",
    ["same-key", "unkeyed", "merged", "configuration", "native-restore", "rendered-restore", "compacted", "new"],
)
def test_projection_tracks_binding_transitions_without_live_mutation(transition: str) -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        notes = DS4SystemNotes(session, hints=True, clock=lambda: "fixed")
        binder.note_provider = notes
        messages: list[ModelMessage] = [user("one"), user("two")]
        binder.prepare(messages)
        original_tokens = session.transcript.tokens
        saved = binder.export_segments()
        tools = []
        instructions = None
        if transition == "same-key":
            first = messages[0].parts[0]
            assert isinstance(first, UserPromptPart)
            first.content = "edited"
        elif transition == "unkeyed":
            messages = deepcopy(messages)
            for message in messages:
                message.metadata = None
        elif transition == "merged":
            first, second = messages
            assert isinstance(first, ModelRequest) and isinstance(second, ModelRequest)
            messages = [ModelRequest([*first.parts, *second.parts])]
        elif transition == "configuration":
            tools = [tool()]
            instructions = "new rules"
        elif transition == "native-restore":
            binder = MessageBinder(session)
            binder.note_provider = notes
            binder.restore(messages, saved)
        elif transition == "rendered-restore":
            session.new()
            binder.forget()
            binder.restore_notes(saved)
            notes.restore_from_segments(saved)
        elif transition == "compacted":
            notes._on_compacted(SessionEvent("compacted", session.id, {}))
            notes.hints_enabled = False
        elif transition == "new":
            session.new()
            binder.forget()
            notes.sync_from_segments([])
        messages.extend([user("three"), user("four")])
        before = snapshot(binder, messages)
        counts = [binder.count_rendered_request(messages, tools=tools, instructions=instructions) for _ in range(2)]
        assert snapshot(binder, messages) == before
        binder.prepare(messages, tools=tools, instructions=instructions)
        actual = len(session.transcript) + len(PromptBuilder(engine).assistant_prefix(session.transcript.tokens))
        assert counts == [actual, actual]
        subtypes = [s.meta.get("subtype") for s in session.transcript.segments if s.kind == "note"]
        assert subtypes.count("date") == 1
        assert subtypes.count("hints_full") == 1
        assert subtypes.count("hints_off") == (1 if transition == "compacted" else 0)
        if transition in {"native-restore", "rendered-restore", "merged", "unkeyed"}:
            assert session.transcript.tokens[: len(original_tokens)] == original_tokens
        assert binder._restored_notes == {}


def test_native_start_failure_does_not_consume_planned_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        notes = DS4SystemNotes(session, hints=True, clock=lambda: "fixed")
        binder.note_provider = notes
        messages: list[ModelMessage] = [user("go")]

        def fail_sync(self: FakeSession, *args: Any, **kwargs: Any) -> Any:
            raise EngineError("native failed")

        monkeypatch.setattr(FakeSession, "sync", fail_sync)
        with pytest.raises(EngineError, match="native failed"):
            binder.prepare(messages)
        assert not session.transcript.chat
        assert session.transcript.tokens == ()
        assert binder.records() == ()
        assert binder._signature is None
        assert not notes._date_emitted and not notes._hints_full_emitted
        assert messages[0].metadata is None
        monkeypatch.undo()
        binder.prepare(messages)
        assert notes._date_emitted and notes._hints_full_emitted


def test_prepare_renders_note_content_once_without_re_reading_clock() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        calls: list[str] = []

        def clock() -> str:
            calls.append("read")
            assert len(calls) == 1
            return "chosen time"

        notes = DS4SystemNotes(session, clock=clock)
        binder.note_provider = notes
        binder.prepare([user("one"), user("two"), user("three")])
        assert calls == ["read"]
        note_texts = [s.meta.get("text") for s in session.transcript.segments if s.kind == "note"]
        assert note_texts == [
            "Current local date and time at session start: chosen time. Use this only when date or time matters."
        ]


def test_speculative_later_clock_failure_leaves_earlier_hints_unconsumed() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)

        def fail_clock() -> str:
            raise ValueError("later clock failed")

        binder.note_provider = DS4SystemNotes(session, hints=True, clock=fail_clock)
        messages: list[ModelMessage] = [ModelRequest([ToolReturnPart("read", "result")]), user("go")]
        before = snapshot(binder, messages)
        with pytest.raises(ValueError, match="later clock failed"):
            binder.prepare(messages)
        assert snapshot(binder, messages) == before


def test_unchanged_pending_date_renders_clock_once() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        messages: list[ModelMessage] = [user("go")]
        binder.prepare(messages)
        calls = 0

        def clock() -> str:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise ValueError("clock was read twice")
            return "chosen time"

        notes = DS4SystemNotes(session, clock=clock)
        binder.note_provider = notes
        binder.prepare(messages)
        assert calls == 1
        assert notes._date_emitted
        assert len([s for s in session.transcript.segments if s.kind == "note"]) == 1
