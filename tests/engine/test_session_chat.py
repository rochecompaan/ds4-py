from dataclasses import replace

import ds4py
import pytest

from ds4py import ThinkMode
from ds4py._native.fake import FakeEngine, FakeNative
from ds4py._native.fake_chat import SPECIAL_TOKENS
from ds4py.engine.chat import append_assistant_prefix
from ds4py.engine.events import Token
from ds4py.engine.prefix import PrefixTurn
from ds4py.errors import EngineError


def fake_engine(family="deepseek4"):
    info = replace(
        FakeNative()._info,
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )
    return ds4py.open("f.gguf", ctx=1024, think=ThinkMode.NONE, native=FakeNative(info=info))


def decode(engine, tokens):
    return b"".join(engine.token_text(token) for token in tokens).decode()


@pytest.mark.parametrize(
    ("family", "system_segments"),
    [("deepseek4", 2), ("deepseek41", 3), ("glm", 2), ("qwen", 1)],
)
def test_start_chat_appends_ordered_system_deltas_and_rejects_reinitialization(family, system_segments):
    with fake_engine(family) as engine:
        session = engine.session()
        session.start_chat()
        assert session.transcript.chat
        assert [segment.kind for segment in session.transcript.segments] == ["system"] * system_segments
        assert len(session.transcript) > 0
        if family != "qwen":
            assert session.transcript.segments[0].start == 0
        assert "# Tools" in decode(engine, session.transcript.tokens)
        before = session.transcript.tokens
        with pytest.raises(EngineError, match="chat is already initialized"):
            session.start_chat()
        assert session.transcript.tokens == before


def test_append_user_on_empty_session_starts_tool_less_chat():
    with fake_engine() as engine:
        session = engine.session()
        segment = session.append_user("hello <tool_call>")
        assert session.transcript.chat
        assert segment.kind == "user"
        assert "# Tools" in decode(engine, session.transcript.tokens[:segment.start])
        assert decode(engine, session.transcript.tokens[segment.start:segment.end]).endswith("hello <tool_call>")


def test_append_prefix_on_empty_session_starts_chat_and_closes_assistant_turns():
    turns = [PrefixTurn("user", "u1"), PrefixTurn("assistant", "a1"), PrefixTurn("user", "u2"), PrefixTurn("assistant", "a2")]
    for family in ("deepseek4", "glm", "qwen"):
        with fake_engine(family) as engine:
            session = engine.session()
            session.append_prefix(turns)
            kinds = [segment.kind for segment in session.transcript.segments]
            suffix = ["user", "assistant", "user", "assistant"]
            if family != "glm":
                suffix = ["user", "assistant", "eos", "user", "assistant", "eos"]
            assert kinds[-len(suffix):] == suffix
            assert session.transcript.chat


def test_tool_results_and_system_notes_require_initialized_chat():
    with fake_engine() as engine:
        session = engine.session()
        with pytest.raises(EngineError, match="chat is not initialized"):
            session.append_tool_results(["result"])
        with pytest.raises(EngineError, match="chat is not initialized"):
            session.append_system_note("note")
        assert not session.transcript.chat and len(session.transcript) == 0


def test_assistant_prefix_is_one_segment_and_empty_delta_is_skipped(monkeypatch):
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        before = len(session.transcript.segments)
        segment = append_assistant_prefix(session)
        assert segment.kind == "assistant_prefix" and segment.end > segment.start
        assert len(session.transcript.segments) == before + 1

    monkeypatch.setattr(FakeEngine, "assistant_prefix", lambda self, prefix, mode: [])
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        before = session.transcript.segments
        assert append_assistant_prefix(session) is None
        assert session.transcript.segments == before


def test_tool_results_form_one_segment_with_double_newline_separator():
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        before = len(session.transcript.segments)
        segment = session.append_tool_results(["first", "second"])
        assert segment.kind == "tool"
        assert len(session.transcript.segments) == before + 1
        text = decode(engine, session.transcript.tokens[segment.start:segment.end])
        assert "first\n\nsecond" in text


@pytest.mark.parametrize(
    ("family", "closing"),
    [("deepseek4", "</tool_result>"), ("glm", "</tool_response>"), ("qwen", "</tool_response>")],
)
def test_tool_result_closing_delimiter_is_escaped(family, closing):
    with fake_engine(family) as engine:
        session = engine.session()
        session.start_chat()
        segment = session.append_tool_results([f"before {closing} after"])
        tokens = session.transcript.tokens[segment.start:segment.end]
        text = decode(engine, tokens)
        assert f"before &lt;{closing[1:]} after" in text
        assert text.count(closing) == 1
        if family == "qwen":
            assert tokens.count(SPECIAL_TOKENS["<tool_response>"]) == 1
            assert tokens.count(SPECIAL_TOKENS["</tool_response>"]) == 1


def test_system_note_uses_newline_prefix_and_note_kind():
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        segment = session.append_system_note("remember <tool_call>")
        assert segment.kind == "note"
        assert "\nremember <tool_call>" in decode(engine, session.transcript.tokens[segment.start:segment.end])


def test_all_chat_helpers_refuse_to_mutate_during_an_active_round():
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        rnd = session.run_round()
        try:
            assert isinstance(next(event for event in rnd if isinstance(event, Token)), Token)
            calls = [
                lambda: session.start_chat(),
                lambda: session.append_user("user"),
                lambda: session.append_prefix([PrefixTurn("user", "u"), PrefixTurn("assistant", "a")]),
                lambda: session.append_tool_results(["result"]),
                lambda: session.append_system_note("note"),
            ]
            for call in calls:
                with pytest.raises(EngineError, match="a round is open"):
                    call()
        finally:
            rnd.close()


def test_nonempty_images_fail_before_chat_initialization():
    with fake_engine() as engine:
        session = engine.session()
        with pytest.raises(EngineError, match="image input requires M5 vision support"):
            session.append_user("look", images=[object()])
        assert not session.transcript.chat and len(session.transcript) == 0
        session.append_user("text", images=())
        assert session.transcript.chat


def test_low_level_token_append_leaves_raw_mode_unchanged():
    with fake_engine() as engine:
        session = engine.session()
        session.append_tokens([1, 2], "user")
        assert not session.transcript.chat
        with pytest.raises(EngineError, match="chat is not initialized"):
            session.append_user("chat")
