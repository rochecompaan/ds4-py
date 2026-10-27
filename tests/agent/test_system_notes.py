from datetime import datetime

from ds4py.capabilities.system_notes import DS4SystemNotes
from pydantic_ai.messages import ModelRequest, UserPromptPart


class MockSession:
    def __init__(self):
        self.notes = []
        self.listeners = []

    def append_system_note(self, text, subtype=None):
        self.notes.append((text, subtype))

    def on_event(self, listener, kinds=None):
        self.listeners.append((listener, kinds))


def user_req(text="hello"):
    return ModelRequest([UserPromptPart(text)])


def test_date_note_emitted_once():
    session = MockSession()
    notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    result = notes.notes([user_req()])
    assert len(result) == 1
    assert "Current local date and time at session start: 2024-01-01 12:00:00" in result[0][0]
    assert result[0][1] == "date"
    result2 = notes.notes([user_req()])
    assert len(result2) == 0


def test_hints_off_by_default():
    session = MockSession()
    notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    result = notes.notes([user_req()])
    assert len(result) == 1
    assert not any(k.startswith("hints") for _, k in result)


def test_hints_lifecycle():
    session = MockSession()
    notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0), hints=True)
    result = notes.notes([user_req()])
    assert len(result) == 2
    assert result[1][1] == "hints_full"
    assert result[1][0].startswith("The user enabled programming hints.")

    result = notes.notes([user_req()])
    assert len(result) == 0

    notes.hints_enabled = False
    result = notes.notes([user_req()])
    assert len(result) == 1
    assert result[0][1] == "hints_off"

    notes.hints_enabled = True
    result = notes.notes([user_req()])
    assert len(result) == 1
    assert result[0][1] == "hints_on"


def test_compaction_resets_hints():
    session = MockSession()
    notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0), hints=True)
    notes.notes([user_req()])
    assert notes._hints_full_emitted is True
    notes._on_compacted(None)
    assert notes._hints_full_emitted is False
    result = notes.notes([user_req()])
    assert len(result) == 1
    assert result[0][1] == "hints_full"


def test_restore_from_segments():
    session = MockSession()
    notes = DS4SystemNotes(session)

    class Seg:
        def __init__(self, kind):
            self.kind = kind

    notes.restore_from_segments([Seg("system_note:date"), Seg("system_note:hints_on")])
    assert notes._date_emitted is True
    assert notes._hints_applied is True


def test_note_segments_have_empty_keys_and_do_not_become_messages():
    import asyncio
    from dataclasses import replace

    import ds4py
    from ds4py import ThinkMode
    from ds4py._native.fake import FAKE_EOS, FakeNative
    from ds4py._native.fake_chat import tokenize_rendered
    from ds4py.engine.sampling import SamplingPolicy
    from ds4py.model import DS4Model
    from pydantic_ai.models import ModelRequestParameters

    info = replace(
        FakeNative()._info,
        name="fake-ds4",
        family="deepseek4",
        is_deepseek41=False,
        is_glm_dsa=False,
        is_qwen4=False,
    )
    tokens = tuple(tokenize_rendered("answer"))
    script = lambda prompt, generated: (
        tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS
    )
    with ds4py.open("f.gguf", ctx=4096, think=ThinkMode.NONE, native=FakeNative(info=info, script=script)) as engine:
        session = engine.session()
        notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
        model = DS4Model(session, sampling=SamplingPolicy(seed=1), note_provider=notes)
        asyncio.run(model.request([user_req()], None, ModelRequestParameters()))
        note_segments = [s for s in session.transcript.segments if s.kind == "note"]
        assert len(note_segments) == 1
        assert note_segments[0].meta.get("subtype") == "date"
        assert note_segments[0].meta.get("key", "") == ""


def _fake_engine(text="answer"):
    from dataclasses import replace

    import ds4py
    from ds4py import ThinkMode
    from ds4py._native.fake import FAKE_EOS, FakeNative
    from ds4py._native.fake_chat import tokenize_rendered

    info = replace(
        FakeNative()._info,
        name="fake-ds4",
        family="deepseek4",
        is_deepseek41=False,
        is_glm_dsa=False,
        is_qwen4=False,
    )
    tokens = tuple(tokenize_rendered(text))

    def script(prompt, generated):
        return tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS

    return ds4py.open(
        "f.gguf", ctx=4096, think=ThinkMode.NONE,
        native=FakeNative(info=info, script=script),
    )


def _model(session, notes):
    from ds4py.engine.sampling import SamplingPolicy
    from ds4py.model import DS4Model

    return DS4Model(session, sampling=SamplingPolicy(seed=1), note_provider=notes)


def test_date_note_renders_immediately_before_first_user_request():
    import asyncio

    from pydantic_ai.models import ModelRequestParameters

    with _fake_engine() as engine:
        session = engine.session()
        notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
        model = _model(session, notes)

        asyncio.run(model.request([user_req()], None, ModelRequestParameters()))

        kinds = [segment.kind for segment in session.transcript.segments]
        note_index = kinds.index("note")
        assert kinds[note_index + 1] == "user"
        assert all(kind == "system" for kind in kinds[:note_index])


def test_hint_note_renders_immediately_before_tool_continuation_request():
    import asyncio

    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
    from pydantic_ai.models import ModelRequestParameters

    with _fake_engine() as engine:
        session = engine.session()
        notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
        model = _model(session, notes)
        first = user_req()
        asyncio.run(model.request([first], None, ModelRequestParameters()))

        notes.hints_enabled = True
        response = ModelResponse([
            ToolCallPart("read_file", {"path": "a.txt"}, tool_call_id="call-1"),
        ])
        continuation = ModelRequest([
            ToolReturnPart("read_file", "data", tool_call_id="call-1"),
        ])
        asyncio.run(model.request(
            [first, response, continuation], None, ModelRequestParameters(),
        ))

        kinds = [segment.kind for segment in session.transcript.segments]
        note_index = max(index for index, kind in enumerate(kinds) if kind == "note")
        tool_index = max(index for index, kind in enumerate(kinds) if kind == "tool")
        assert note_index + 1 == tool_index


def test_exported_provenance_names_note_subtypes_and_default():
    from ds4py.binding import MessageBinder

    with _fake_engine() as engine:
        session = engine.session()
        notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0), hints=True)
        binder = MessageBinder(session)
        binder.note_provider = notes
        binder.prepare([user_req()])

        kinds = [segment.kind for segment in binder.export_segments()]
        assert kinds.count("system_note:date") == 1
        assert kinds.count("system_note:hints_full") == 1

        session.append_system_note("plain note")
        assert [segment.kind for segment in binder.export_segments()].count("system_note") == 1


def test_fresh_provider_restores_exported_note_provenance():
    from ds4py.binding import MessageBinder

    with _fake_engine() as engine:
        session = engine.session()
        notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0), hints=True)
        binder = MessageBinder(session)
        binder.note_provider = notes
        binder.prepare([user_req()])
        exported = binder.export_segments()

        fresh = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0), hints=False)
        fresh.restore_from_segments(exported)

        assert fresh._date_emitted is True
        assert fresh.hints_enabled is True
        assert fresh._hints_applied is True
        assert fresh.notes([user_req("next")]) == []


def test_public_append_system_note_passes_subtype_metadata():
    with _fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        segment = session.append_system_note("setting", subtype="hints_on")
        assert segment.kind == "note"
        assert segment.meta["subtype"] == "hints_on"


def test_restore_restores_hints_enabled_from_latest_note():
    class Seg:
        def __init__(self, kind):
            self.kind = kind

    session = MockSession()
    notes = DS4SystemNotes(session)
    notes.restore_from_segments([Seg("system_note:hints_full")])
    assert notes.hints_enabled is True
    assert notes._hints_applied is True
    assert notes._hints_full_emitted is True

    off = DS4SystemNotes(session)
    off.restore_from_segments([Seg("system_note:hints_on"), Seg("system_note:hints_off")])
    assert off.hints_enabled is False
    assert off._hints_applied is False


def test_never_applied_hints_stay_unset_after_compaction():
    session = MockSession()
    notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    notes._on_compacted(None)
    result = notes.notes([user_req()])
    assert [subtype for _, subtype in result] == ["date"]


def test_compaction_while_disabled_reasserts_hints_off():
    session = MockSession()
    notes = DS4SystemNotes(
        session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0), hints=True,
    )
    assert [subtype for _, subtype in notes.notes([user_req()])] == ["date", "hints_full"]
    notes.hints_enabled = False
    assert [subtype for _, subtype in notes.notes([user_req()])] == ["hints_off"]

    notes._on_compacted(None)

    assert [subtype for _, subtype in notes.notes([user_req()])] == ["hints_off"]
    assert notes.notes([user_req()]) == []


def test_hint_texts_match_pinned_trailing_newlines():
    from ds4py.capabilities.system_notes import HINTS_FULL, HINTS_OFF, HINTS_ON

    assert HINTS_FULL == (
        "The user enabled programming hints. Occasionally explain a useful concept "
        "behind the current work: a design choice, failure mechanism, trade-off, "
        "or way to verify correctness. Base each hint on code or results you have "
        "actually examined. Write one or two sentences as a normal Markdown "
        "blockquote starting with > **Hint:**. Hints are prose, not tool calls. "
        "Put them after the relevant discovery, including between tool calls. "
        "Prefer no hint to routine narration, repetition, or generic advice. "
        "Do not invent personal experience or a learner profile. Keep working "
        "without waiting for an answer. Later system notes may enable or disable "
        "hints; follow the latest setting.\n"
    )
    assert HINTS_ON == "The user enabled programming hints. Produce them from now on.\n"
    assert HINTS_OFF == (
        "Programming hints are disabled. Do not produce teaching asides; "
        "continue the task normally.\n"
    )


def test_default_clock_is_timezone_aware():
    notes = DS4SystemNotes(MockSession())
    when = notes.clock()
    assert when.tzinfo is not None
    result = notes.notes([user_req()])
    assert result[0][0].startswith("Current local date and time at session start: ")
    assert result[0][0].endswith(". Use this only when date or time matters.")


def test_unchanged_prepare_does_not_duplicate_or_move_notes():
    from ds4py.binding import MessageBinder

    with _fake_engine() as engine:
        session = engine.session()
        notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
        binder = MessageBinder(session)
        binder.note_provider = notes
        request = user_req()
        binder.prepare([request])
        before = session.transcript.segments
        provenance = binder.export_segments()

        change = binder.prepare([request])

        assert change.kind == "unchanged"
        assert session.transcript.segments == before
        assert binder.export_segments() == provenance


def test_same_key_user_rewrite_keeps_date_note_once():
    from copy import deepcopy

    from ds4py.binding import MessageBinder

    with _fake_engine() as engine:
        session = engine.session()
        notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
        binder = MessageBinder(session)
        binder.note_provider = notes
        request = user_req("original")
        binder.prepare([request])

        changed = deepcopy(request)
        changed.parts[0].content = "changed"
        assert changed.metadata["ds4.key"] == request.metadata["ds4.key"]

        change = binder.prepare([changed])

        assert change.kind == "rewrite"
        kinds = [segment.kind for segment in binder.export_segments()]
        assert kinds.count("system_note:date") == 1
        assert kinds[kinds.index("system_note:date") + 1] == "user"


def test_hints_setting_change_on_identical_messages_emits_note_once():
    from ds4py.binding import MessageBinder

    with _fake_engine() as engine:
        session = engine.session()
        notes = DS4SystemNotes(session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
        binder = MessageBinder(session)
        binder.note_provider = notes
        request = user_req()
        binder.prepare([request])

        notes.hints_enabled = True
        change = binder.prepare([request])

        assert change.kind == "rewrite"
        kinds = [segment.kind for segment in binder.export_segments()]
        assert kinds.count("system_note:date") == 1
        assert kinds.count("system_note:hints_full") == 1

        again = binder.prepare([request])
        assert again.kind == "unchanged"
        assert [segment.kind for segment in binder.export_segments()] == kinds

        notes.hints_enabled = False
        off = binder.prepare([request])

        assert off.kind == "rewrite"
        kinds = [segment.kind for segment in binder.export_segments()]
        assert kinds.count("system_note:date") == 1
        assert kinds.count("system_note:hints_full") == 1
        assert kinds.count("system_note:hints_off") == 1


def test_incremental_notes_match_fresh_full_rebuild_tokens():
    from ds4py.binding import MessageBinder
    from pydantic_ai.messages import ModelResponse, TextPart

    first = user_req("first")
    response = ModelResponse([TextPart("answer")])
    follow_up = user_req("follow-up")
    with _fake_engine() as incremental_engine, _fake_engine() as fresh_engine:
        incremental_session = incremental_engine.session()
        incremental_binder = MessageBinder(incremental_session)
        incremental_binder.note_provider = DS4SystemNotes(
            incremental_session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0), hints=True,
        )
        incremental_binder.prepare([first])
        incremental_binder.prepare([first, response])
        incremental_binder.prepare([first, response, follow_up])

        fresh_session = fresh_engine.session()
        fresh_binder = MessageBinder(fresh_session)
        fresh_binder.note_provider = DS4SystemNotes(
            fresh_session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0), hints=True,
        )
        fresh_binder.prepare([first, response, follow_up])

        assert fresh_session.transcript.tokens == incremental_session.transcript.tokens
        fresh_kinds = [segment.kind for segment in fresh_binder.export_segments()]
        assert fresh_kinds.index("system_note:date") < fresh_kinds.index("user")
        assert fresh_kinds.index("system_note:hints_full") < fresh_kinds.index("user")


def test_compaction_reasserts_full_note_across_rewrite_with_surviving_notes():
    from copy import deepcopy

    from ds4py.binding import MessageBinder
    from pydantic_ai.messages import ModelResponse, TextPart

    with _fake_engine() as engine:
        session = engine.session()
        notes = DS4SystemNotes(
            session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0), hints=True,
        )
        binder = MessageBinder(session)
        binder.note_provider = notes
        first = user_req("first")
        response = ModelResponse([TextPart("answer")])
        follow_up = user_req("follow-up")

        binder.prepare([first])
        notes.hints_enabled = False
        binder.prepare([first, response, follow_up])
        notes.hints_enabled = True
        binder.prepare([first, response, follow_up])
        notes._on_compacted(None)

        changed = deepcopy(response)
        changed.parts[0].content = "changed"
        binder.prepare([first, changed, follow_up])

        kinds = [segment.kind for segment in binder.export_segments()]
        assert kinds.count("system_note:hints_full") == 2
        assert kinds[-2:] == ["system_note:hints_full", "user"]


def test_compaction_reasserts_hints_off_across_rewrite_with_surviving_notes():
    from copy import deepcopy

    from ds4py.binding import MessageBinder
    from pydantic_ai.messages import ModelResponse, TextPart

    with _fake_engine() as engine:
        session = engine.session()
        notes = DS4SystemNotes(
            session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0), hints=True,
        )
        binder = MessageBinder(session)
        binder.note_provider = notes
        first = user_req("first")
        response = ModelResponse([TextPart("answer")])
        follow_up = user_req("follow-up")

        binder.prepare([first])
        notes.hints_enabled = False
        binder.prepare([first])
        notes.hints_enabled = True
        binder.prepare([first, response, follow_up])
        notes.hints_enabled = False
        binder.prepare([first, response, follow_up])
        notes._on_compacted(None)

        changed = deepcopy(response)
        changed.parts[0].content = "changed"
        binder.prepare([first, changed, follow_up])

        kinds = [segment.kind for segment in binder.export_segments()]
        assert kinds.count("system_note:hints_off") == 2
        assert kinds[-2:] == ["system_note:hints_off", "user"]


def test_compaction_reasserts_hints_off_across_full_rebuild():
    from ds4py.binding import MessageBinder

    with _fake_engine() as engine:
        session = engine.session()
        notes = DS4SystemNotes(
            session, clock=lambda: datetime(2024, 1, 1, 12, 0, 0), hints=True,
        )
        binder = MessageBinder(session)
        binder.note_provider = notes
        request = user_req()
        binder.prepare([request])
        notes.hints_enabled = False
        binder.prepare([request])
        notes._on_compacted(None)

        binder.prepare([request], instructions="changed")

        kinds = [segment.kind for segment in binder.export_segments()]
        assert kinds.count("system_note:hints_off") == 1
        assert kinds[kinds.index("system_note:hints_off") + 1] == "user"
