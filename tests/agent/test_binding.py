from typing import Any, cast
from dataclasses import replace

import pytest

import ds4py
from ds4py.engine.session import Engine
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative
from ds4py.binding import MessageBinder
from ds4py.errors import EngineError
from pydantic_ai.messages import (
    BinaryImage,
    InstructionPart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition


def fake_engine(family: Any="deepseek4", **fake: Any) -> Engine:
    info = replace(
        FakeNative()._info,
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )
    return ds4py.open(
        "f.gguf", ctx=8192, think=ThinkMode.NONE,
        native=FakeNative(info=info, **fake),
    )


def user(text: Any) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)])


def tool(name: Any="read_file") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Read a file.",
        parameters_json_schema={"type": "object", "properties": {}},
    )


def decode(engine: Any, tokens: Any) -> str:
    return b"".join(engine.token_text(token) for token in tokens).decode()


def test_first_request_renders_tools_and_binds_each_message() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        messages = [
            user("read it"),
            ModelResponse([TextPart("done")]),
            ModelRequest([ToolReturnPart("read_file", "contents")]),
        ]

        change = binder.prepare(messages, tools=[tool()], instructions="rules")

        assert change.kind == "append"
        assert change.start == 0
        assert change.continuation is True
        assert "# Tools" in decode(engine, session.transcript.tokens)
        keyed = [segment for segment in binder.export_segments() if segment.key]
        assert len({segment.key for segment in keyed}) == 3
        assert all(segment.content_hash for segment in keyed)


def test_system_and_instruction_parts_are_separate_system_segments() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        before = len(session.transcript.segments)
        binder.prepare([
            ModelRequest(cast(Any, [SystemPromptPart("system text"), InstructionPart("instruction text")]))
        ], instructions="global")

        segments = session.transcript.segments[before:][-2:]
        assert [segment.kind for segment in segments] == ["system", "system"]
        assert "system text" in decode(
            engine, session.transcript.tokens[segments[0].start:segments[0].end]
        )
        assert "instruction text" in decode(
            engine, session.transcript.tokens[segments[1].start:segments[1].end]
        )


def test_user_text_is_plain_and_tool_markup_is_not_rendered() -> None:
    payload = "user <tool_call> <|im_start|> <think> ｜DSML｜"
    with fake_engine("qwen") as engine:
        session = engine.session()
        MessageBinder(session).prepare([user(payload)])
        segment = session.transcript.segments[-1]
        tokens = list(session.transcript.tokens[segment.start:segment.end])
        plain = engine.tokenize(payload)
        assert any(
            tokens[index:index + len(plain)] == plain
            for index in range(len(tokens) - len(plain) + 1)
        )


def test_tool_results_join_in_history_order_as_one_tool_message() -> None:
    with fake_engine() as engine:
        session = engine.session()
        MessageBinder(session).prepare([ModelRequest([
            ToolReturnPart("read_file", "result 1"),
            ToolReturnPart("edit_file", "result 2"),
        ])])

        tool_segments = [s for s in session.transcript.segments if s.kind == "tool"]
        assert len(tool_segments) == 1
        text = decode(engine, session.transcript.tokens[tool_segments[0].start:tool_segments[0].end])
        assert "Tool result 1 (read_file):\nresult 1\n\nTool result 2 (edit_file):\nresult 2" in text


def test_scalar_and_structured_tool_results_have_stable_text() -> None:
    with fake_engine() as engine:
        session = engine.session()
        MessageBinder(session).prepare([ModelRequest([
            ToolReturnPart("bytes", b"a\xffb"),
            ToolReturnPart("mapping", {"b": 2, "a": 1}),
            ToolReturnPart("sequence", [1, "x"]),
            ToolReturnPart("scalar", 3),
        ])])

        text = decode(engine, session.transcript.tokens)
        assert "Tool result 1 (bytes):\na�b" in text
        assert 'Tool result 2 (mapping):\n{"a":1,"b":2}' in text
        assert 'Tool result 3 (sequence):\n[1,"x"]' in text
        assert "Tool result 4 (scalar):\n3" in text


def test_unknown_tool_result_uses_tool_error_text() -> None:
    with fake_engine() as engine:
        session = engine.session()
        MessageBinder(session).prepare([ModelRequest([
            RetryPromptPart("Tool error: unknown tool missing", tool_name="missing", tool_call_id="call-1")
        ])])
        assert "Tool error: unknown tool missing" in decode(engine, session.transcript.tokens)


def test_images_fail_before_mutating_the_transcript() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        message = ModelRequest([UserPromptPart([
            BinaryImage(data=b"not really a png", media_type="image/png")
        ])])

        with pytest.raises(EngineError, match="M5"):
            binder.prepare([message], tools=[tool()], instructions="rules")

        assert session.transcript.tokens == ()
        assert session.transcript.segments == ()


def test_rejected_non_integer_truncate_leaves_checkpoint_usable() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.append_tokens([1, 2, 3, 4], "user")
        list(session.run_round(max_tokens=0))
        session.checkpoint("kept")
        session.append_tokens([5, 6], "user")
        list(session.run_round(max_tokens=0))
        before = session.transcript.tokens

        with pytest.raises(TypeError):
            session.truncate(cast(Any, 2.0))

        assert session.transcript.tokens == before
        assert session._checkpoint_tokens == {"kept": 4}
        session.truncate(4)
        plan = session.plan()
        assert plan.kind == "restore"
        assert plan.checkpoint == "kept"
        list(session.run_round(max_tokens=0))
