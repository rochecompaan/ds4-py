from typing import Any, cast
import threading
from dataclasses import replace

import ds4py
from ds4py.engine.session import Engine
import pytest

from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeEngine, FakeNative
from ds4py._native.fake_chat import SPECIAL_TOKENS, tokenize_rendered
from ds4py.engine.chat import append_assistant_prefix
from ds4py.engine.events import Prefill, Stopped, Synced, Token, ToolCalls
from ds4py.engine.prefix import PrefixTurn
from ds4py.errors import EngineError


def fake_engine(family: Any="deepseek4", script: Any=None, cancel: Any=None) -> Engine:
    info = replace(
        FakeNative()._info,
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )
    native = FakeNative(info=info) if script is None else FakeNative(info=info, script=script)
    return ds4py.open("f.gguf", ctx=1024, think=ThinkMode.NONE, native=native, cancel=cancel)


def decode(engine: Any, tokens: Any) -> str:
    return b"".join(engine.token_text(token) for token in tokens).decode()


def scripted_text(text: Any) -> Any:
    tokens = tuple(tokenize_rendered(text))
    return lambda prompt, generated: tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS


@pytest.mark.parametrize(
    ("family", "system_segments"),
    [("deepseek4", 2), ("deepseek41", 3), ("glm", 2), ("qwen", 1)],
)
def test_start_chat_appends_ordered_system_deltas_and_rejects_reinitialization(family: Any, system_segments: Any) -> None:
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


def test_append_user_on_empty_session_starts_tool_less_chat() -> None:
    with fake_engine() as engine:
        session = engine.session()
        segment = session.append_user("hello <tool_call>")
        assert session.transcript.chat
        assert segment.kind == "user"
        assert "# Tools" in decode(engine, session.transcript.tokens[:segment.start])
        assert decode(engine, session.transcript.tokens[segment.start:segment.end]).endswith("hello <tool_call>")


def test_append_prefix_on_empty_session_starts_chat_and_closes_assistant_turns() -> None:
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


def test_tool_results_and_system_notes_require_initialized_chat() -> None:
    with fake_engine() as engine:
        session = engine.session()
        with pytest.raises(EngineError, match="chat is not initialized"):
            session.append_tool_results(["result"])
        with pytest.raises(EngineError, match="chat is not initialized"):
            session.append_system_note("note")
        assert not session.transcript.chat and len(session.transcript) == 0


def test_assistant_prefix_is_one_segment_and_empty_delta_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        before = len(session.transcript.segments)
        segment = cast(Any, append_assistant_prefix(session))
        assert segment.kind == "assistant_prefix" and segment.end > segment.start
        assert len(session.transcript.segments) == before + 1

    monkeypatch.setattr(FakeEngine, "assistant_prefix", lambda self, prefix, mode: [])
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        before = session.transcript.segments
        assert append_assistant_prefix(session) is None
        assert session.transcript.segments == before


def test_tool_results_form_one_segment_with_double_newline_separator() -> None:
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
def test_tool_result_closing_delimiter_is_escaped(family: Any, closing: Any) -> None:
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


def test_system_note_uses_newline_prefix_and_note_kind() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        segment = session.append_system_note("remember <tool_call>")
        assert segment.kind == "note"
        assert "\nremember <tool_call>" in decode(engine, session.transcript.tokens[segment.start:segment.end])


def test_all_chat_helpers_refuse_to_mutate_during_an_active_round() -> None:
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


def test_nonempty_images_fail_before_chat_initialization() -> None:
    with fake_engine() as engine:
        session = engine.session()
        with pytest.raises(EngineError, match="image input requires M5 vision support"):
            session.append_user("look", images=[object()])
        assert not session.transcript.chat and len(session.transcript) == 0
        session.append_user("text", images=())
        assert session.transcript.chat


def test_low_level_token_append_leaves_raw_mode_unchanged() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.append_tokens([1, 2], "user")
        assert not session.transcript.chat
        with pytest.raises(EngineError, match="chat is not initialized"):
            session.append_user("chat")


@pytest.mark.parametrize("family", ["deepseek4", "deepseek41", "glm", "qwen"])
def test_chat_run_round_syncs_prefix_and_closes_assistant_turn(family: Any) -> None:
    with fake_engine(family) as engine:
        session = engine.session()
        session.start_chat()
        before = len(session.transcript)
        events = list(session.run_round(max_tokens=10))
        significant = [event for event in events if not isinstance(event, Prefill)]
        synced = next(event for event in significant if isinstance(event, Synced))
        segments = session.transcript.segments
        prefix = next(segment for segment in segments if segment.kind == "assistant_prefix")
        assistant = next(segment for segment in segments if segment.kind == "assistant")
        assert prefix.start == before and synced.pos == prefix.end
        assert synced.prefilled == synced.pos
        assert [event.id for event in significant if isinstance(event, Token)] == engine.tokenize("ok")
        assert decode(engine, session.transcript.tokens[assistant.start:assistant.end]) == "ok"
        assert significant[-1] == Stopped("stop_token")
        suffix = ["assistant_prefix", "assistant"] + ([] if family == "glm" else ["eos"])
        assert [segment.kind for segment in segments[-len(suffix):]] == suffix
        eos_tokens = 0 if family == "glm" else 1
        assert session.ctx_used == len(session.transcript) - eos_tokens


def test_closing_chat_round_drains_worker_before_eos_close() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        events = session.run_round()
        for event in events:
            if isinstance(event, Token):
                break
        events.close()
        kinds = [segment.kind for segment in session.transcript.segments]
        assert kinds[-2:] == ["assistant", "eos"] and kinds.count("eos") == 1
        session.append_user("next")
        assert session.transcript.segments[-1].kind == "user"


def test_public_chat_round_emits_calls_before_stopped_then_closes_turn() -> None:
    text = "<tool_call><function=read><parameter=path>/tmp/x</parameter></function></tool_call>"
    with fake_engine("qwen", script=scripted_text(text)) as engine:
        session = engine.session()
        session.start_chat()
        events = list(session.run_round())
        calls = next(event for event in events if isinstance(event, ToolCalls))
        assert calls.calls[0].name == "read" and calls.calls[0].args["path"] == "/tmp/x"
        assert events[-1] == Stopped("tool_call_complete")
        assert events.index(calls) == len(events) - 2
        assistant = next(segment for segment in session.transcript.segments if segment.kind == "assistant")
        assert list(session.transcript.tokens[assistant.start:assistant.end]) == tokenize_rendered(text)
        assert session.transcript.segments[-1].kind == "eos"


def test_chat_caller_limit_and_interruption_close_turn_once() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        events = list(session.run_round(max_tokens=0))
        assert events[-1] == Stopped("max_tokens")
        kinds = [segment.kind for segment in session.transcript.segments]
        assert kinds[-2:] == ["assistant_prefix", "eos"] and "assistant" not in kinds

    cancel = threading.Event()
    def interrupted(prompt: Any, generated: Any) -> Any:
        if len(generated) == 1:
            cancel.set()
        return ord("a") + 1
    with fake_engine(script=interrupted, cancel=cancel) as engine:
        session = engine.session()
        session.start_chat()
        events = list(session.run_round())
        assert events[-1] == Stopped("interrupted")
        assert [segment.kind for segment in session.transcript.segments][-2:] == ["assistant", "eos"]
        assert [segment.kind for segment in session.transcript.segments].count("eos") == 1
