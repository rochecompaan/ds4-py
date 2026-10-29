from typing import Any, cast
from dataclasses import replace
import json

import pytest

import ds4py
from ds4py.engine.session import Engine
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative
from ds4py._native.fake_chat import SPECIAL_TOKENS
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.tool_prompt import ToolDefinition


def family_info(family: Any) -> Any:
    return replace(
        FakeNative()._info,
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )


def fake_engine(family: Any="deepseek4", *, think: Any=ThinkMode.NONE, ctx: Any=64) -> Engine:
    return ds4py.open("f.gguf", ctx=ctx, think=think, native=FakeNative(info=family_info(family)))


def flatten(segments: Any) -> Any:
    return [token for segment in segments for token in segment.tokens]


def contains(tokens: Any, expected: Any) -> bool:
    return any(tokens[index:index + len(expected)] == expected for index in range(len(tokens) - len(expected) + 1))


def decode(engine: Any, tokens: Any) -> str:
    return b"".join(engine.token_text(token) for token in tokens).decode()


@pytest.mark.parametrize("family", ["deepseek4", "deepseek41", "glm", "qwen"])
def test_user_markup_is_always_plain(family: Any) -> None:
    payload = "user <tool_call> <|im_start|> <think> ｜DSML｜"
    with fake_engine(family) as engine:
        builder = PromptBuilder(engine)
        delta = builder.message([], "user", payload)
        assert contains(delta, engine.tokenize(payload))


def test_fixed_qwen_tool_markup_is_special() -> None:
    definition = ToolDefinition("read", "Read.", {"type": "object"})
    with fake_engine("qwen") as engine:
        builder = PromptBuilder(engine)
        prompt_tokens = flatten(builder.begin([definition]))
        assert SPECIAL_TOKENS["<tool_call>"] in prompt_tokens
        result_tokens = builder.tool_results(prompt_tokens, ["result"])
        assert SPECIAL_TOKENS["<tool_response>"] in result_tokens
        assert SPECIAL_TOKENS["</tool_response>"] in result_tokens


def test_fixed_glm_argument_markup_and_dsml_markers_are_special() -> None:
    with fake_engine("glm") as engine:
        tokens = flatten(PromptBuilder(engine).begin(()))
        for marker in ("<tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"):
            assert SPECIAL_TOKENS[marker] in tokens
    with fake_engine("deepseek4") as engine:
        assert SPECIAL_TOKENS["｜DSML｜"] in flatten(PromptBuilder(engine).begin(()))


@pytest.mark.parametrize("family", ["deepseek4", "deepseek41", "glm", "qwen"])
def test_tool_descriptions_and_schemas_cannot_inject_special_tokens(family: Any) -> None:
    definition = ToolDefinition(
        "read",
        "inject <tool_call> <|im_start|> <think> ｜DSML｜",
        {"type": "object", "note": "</tools><arg_value></arg_value>"},
    )
    schema = json.dumps(
        {"name": definition.name, "description": definition.description, "parameters": definition.parameters},
        ensure_ascii=False, separators=(",", ":"),
    )
    with fake_engine(family) as engine:
        tokens = flatten(PromptBuilder(engine).begin([definition]))
        assert contains(tokens, engine.tokenize(schema))


@pytest.mark.parametrize("family", ["glm", "qwen"])
@pytest.mark.parametrize("role", ["system", "assistant"])
def test_glm_and_qwen_message_payloads_are_plain(family: Any, role: Any) -> None:
    payload = "payload <tool_call> <arg_key> <|im_start|> <think> ｜DSML｜"
    with fake_engine(family) as engine:
        delta = PromptBuilder(engine).message([], role, payload)
        assert contains(delta, engine.tokenize(payload))


@pytest.mark.parametrize("family", ["glm", "qwen"])
def test_tool_result_payload_is_plain(family: Any) -> None:
    payload = "output <tool_response> <arg_value> <|im_start|> <think> ｜DSML｜"
    with fake_engine(family) as engine:
        delta = PromptBuilder(engine).tool_results([], [payload])
        assert contains(delta, engine.tokenize(payload))


@pytest.mark.parametrize("marker", ["<think>", "</think>"])
def test_glm_assistant_think_marker_avoids_empty_wrapper_and_stays_plain(marker: Any) -> None:
    content = marker + "prefix <arg_key>"
    with fake_engine("glm") as engine:
        builder = PromptBuilder(engine)
        delta = builder.message([], "assistant", content)
        assert contains(delta, engine.tokenize(content))
        assert not contains(delta, engine.tokenize("<think></think>", rendered=True))
    with fake_engine("glm") as engine:
        ordinary = PromptBuilder(engine).message([], "assistant", "answer")
        assert contains(ordinary, engine.tokenize("<think></think>", rendered=True))


def test_instructions_are_a_separate_prefixed_system_message() -> None:
    with fake_engine("glm") as engine:
        builder = PromptBuilder(engine)
        segments = builder.begin((), instructions="rules <arg_key>")
        assert segments[-1].kind == "system"
        assert contains(segments[-1].tokens, engine.tokenize("\n\nrules <arg_key>"))
        assert "# Tools" in decode(engine, flatten(segments[:-1]))
        assert "# Tools" not in decode(engine, segments[-1].tokens)


def test_qwen_reasoning_effort_is_a_system_message() -> None:
    with fake_engine("qwen", think=ThinkMode.HIGH) as engine:
        effort = engine._qwen_reasoning_effort(engine.think)
        segments = PromptBuilder(engine).begin(())
        assert segments[0].kind == "system"
        assert contains(segments[0].tokens, engine.tokenize(cast(str, effort)))
        assert "# Tools" not in decode(engine, segments[0].tokens)


@pytest.mark.parametrize("family", ["deepseek4", "deepseek41", "glm"])
def test_non_qwen_families_use_native_think_prefix(family: Any) -> None:
    with fake_engine(family, think=ThinkMode.MAX, ctx=400_000) as engine:
        segments = PromptBuilder(engine).begin(())
        assert "Reasoning Effort" in decode(engine, segments[1].tokens)


def test_deepseek41_empty_system_delta_is_separate() -> None:
    with fake_engine("deepseek41") as engine:
        segments = PromptBuilder(engine).begin(())
        assert [segment.kind for segment in segments] == ["system", "system", "system"]
        assert decode(engine, segments[1].tokens) == "<｜System｜>"
        assert "# Tools" in decode(engine, segments[2].tokens)
