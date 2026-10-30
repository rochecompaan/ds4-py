from dataclasses import replace
from typing import Any, cast

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import SPECIAL_TOKENS, tokenize_rendered
from ds4py.binding import MessageBinder, _arguments
from ds4py.engine.session import Engine
from ds4py.engine.tool_syntax import make_parser
from ds4py.engine.tool_syntax.render import render_tool_call

FAMILIES = ("deepseek4", "deepseek41", "glm", "qwen")


def fake_engine(family: Any, *, think: Any = ThinkMode.NONE, script: Any = None) -> Engine:
    info = replace(
        FakeNative()._info,
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )
    return ds4py.open(
        "f.gguf",
        ctx=8192,
        think=think,
        native=FakeNative(info=info, script=script),
    )


def decode(engine: Any, tokens: Any) -> bytes:
    return b"".join(engine.token_text(token) for token in tokens)


def parse(data: Any, engine: Any, *, bytewise: Any = False) -> Any:
    parser = make_parser(engine.info)
    if bytewise:
        for byte in data:
            parser.feed(bytes([byte]))
    else:
        parser.feed(data)
    return parser.finish()


def contains(tokens: Any, expected: Any) -> Any:
    tokens = list(tokens)
    return any(tokens[index : index + len(expected)] == expected for index in range(len(tokens) - len(expected) + 1))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("bytewise", [False, True])
def test_restored_assistant_tool_call_round_trips_through_family_parser(family: Any, bytewise: Any) -> None:
    closing = {
        "deepseek4": "</｜DSML｜parameter>",
        "deepseek41": "</｜DSML｜ parameter>",
        "glm": "</arg_value>",
        "qwen": "</parameter>",
    }[family]
    args = {"path": f"before{closing}after"}
    response = ModelResponse(
        [
            TextPart("I will read it. "),
            ToolCallPart("read_file", args, tool_call_id="call-1"),
        ]
    )

    with fake_engine(family) as engine:
        session = engine.session()
        MessageBinder(session).prepare([response])
        assistant = next(s for s in session.transcript.segments if s.kind == "assistant")
        tokens = session.transcript.tokens[assistant.start : assistant.end]
        snapshot = parse(decode(engine, tokens), engine, bytewise=bytewise)

        assert snapshot.state == "DONE", snapshot.error
        assert len(snapshot.calls) == 1
        assert snapshot.calls[0].name == "read_file"
        assert dict(snapshot.calls[0].args) == args
        marker = "｜DSML｜" if family.startswith("deepseek") else "<tool_call>"
        assert SPECIAL_TOKENS[marker] in tokens


def restored_snapshot(response: Any, family: Any, bytewise: Any) -> Any:
    with fake_engine(family) as engine:
        session = engine.session()
        MessageBinder(session).prepare([response])
        assistant = next(s for s in session.transcript.segments if s.kind == "assistant")
        return parse(
            decode(engine, session.transcript.tokens[assistant.start : assistant.end]),
            engine,
            bytewise=bytewise,
        )


@pytest.mark.parametrize("family", ["deepseek4", "deepseek41"])
@pytest.mark.parametrize("bytewise", [False, True])
def test_restored_dsml_multi_call_canonical_uses_one_outer_block(family: Any, bytewise: Any) -> None:
    response = ModelResponse(
        [
            ToolCallPart("first", {"value": "one"}),
            ToolCallPart("second", {"value": "two"}),
        ]
    )

    snapshot = restored_snapshot(response, family, bytewise)

    assert snapshot.state == "DONE", snapshot.error
    assert [call.name for call in snapshot.calls] == ["first", "second"]
    assert [dict(call.args) for call in snapshot.calls] == [
        {"value": "one"},
        {"value": "two"},
    ]


@pytest.mark.parametrize("family", ["deepseek4", "deepseek41"])
@pytest.mark.parametrize("bytewise", [False, True])
def test_restored_dsml_multi_call_sampled_raw_uses_one_outer_block(family: Any, bytewise: Any) -> None:
    invoke = " invoke" if family == "deepseek41" else "invoke"
    parameter = " parameter" if family == "deepseek41" else "parameter"

    def raw(name: Any, value: Any) -> str:
        return (
            f'<｜DSML｜{invoke} name="{name}">\n'
            f'<｜DSML｜{parameter} name="value" string="true">{value}'
            f"</｜DSML｜{parameter}>\n"
            f"</｜DSML｜{invoke}>"
        )

    response = ModelResponse(
        [
            ToolCallPart("ignored", {}, provider_details={"ds4py.raw": raw("first", "one")}),
            ToolCallPart("ignored", {}, provider_details={"ds4py.raw": raw("second", "two")}),
        ],
        metadata={"ds4.source": "sampled"},
    )

    snapshot = restored_snapshot(response, family, bytewise)

    assert snapshot.state == "DONE", snapshot.error
    assert [call.name for call in snapshot.calls] == ["first", "second"]
    assert [dict(call.args) for call in snapshot.calls] == [
        {"value": "one"},
        {"value": "two"},
    ]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("bytewise", [False, True])
def test_contiguous_calls_remain_supported(family: Any, bytewise: Any) -> None:
    response = ModelResponse(
        [
            ToolCallPart("first", {"value": "one"}),
            ToolCallPart("second", {"value": "two"}),
        ]
    )

    snapshot = restored_snapshot(response, family, bytewise)

    assert snapshot.state == "DONE", snapshot.error
    assert [call.name for call in snapshot.calls] == ["first", "second"]


@pytest.mark.parametrize("family", ["deepseek4", "deepseek41"])
@pytest.mark.parametrize("bytewise", [False, True])
def test_empty_text_between_dsml_calls_is_ignored(family: Any, bytewise: Any) -> None:
    response = ModelResponse(
        [
            ToolCallPart("first", {"value": "one"}),
            TextPart(""),
            ToolCallPart("second", {"value": "two"}),
        ]
    )

    snapshot = restored_snapshot(response, family, bytewise)

    assert snapshot.state == "DONE", snapshot.error
    assert [call.name for call in snapshot.calls] == ["first", "second"]
    assert [dict(call.args) for call in snapshot.calls] == [
        {"value": "one"},
        {"value": "two"},
    ]


@pytest.mark.parametrize("family", FAMILIES)
def test_tool_call_after_ordinary_text_following_call_is_rejected(family: Any) -> None:
    response = ModelResponse(
        [
            ToolCallPart("first", {"value": "one"}),
            TextPart("between"),
            ToolCallPart("second", {"value": "two"}),
        ]
    )

    with fake_engine(family) as engine:
        session = engine.session()
        with pytest.raises(ValueError, match="after ordinary text"):
            MessageBinder(session).prepare([response])
        assert session.transcript.tokens == ()


@pytest.mark.parametrize("family", ["deepseek4", "deepseek41", "qwen"])
@pytest.mark.parametrize("bytewise", [False, True])
def test_structured_arguments_safely_preserve_nested_closing_delimiters(family: Any, bytewise: Any) -> None:
    closing = {
        "deepseek4": "</｜DSML｜parameter>",
        "deepseek41": "</｜DSML｜ parameter>",
        "qwen": "</parameter>",
    }[family]
    args = {
        "payload": {
            "nested": [f"before{closing}after", {"count": 2}],
            "enabled": True,
        }
    }
    response = ModelResponse([ToolCallPart("structured", args)])

    snapshot = restored_snapshot(response, family, bytewise)

    assert snapshot.state == "DONE", snapshot.error
    assert len(snapshot.calls) == 1
    assert snapshot.calls[0].name == "structured"
    assert dict(snapshot.calls[0].args) == args


@pytest.mark.parametrize(
    ("family", "args"),
    [
        (
            "glm",
            {
                "number_string": "1",
                "bool_string": "true",
                "object_string": '{"x":1}',
                "ordinary": "hello",
            },
        ),
        ("qwen", {"integer": 1, "object": {"nested": True}, "ordinary": "hello"}),
    ],
)
@pytest.mark.parametrize("bytewise", [False, True])
def test_canonical_arguments_preserve_types_and_json_looking_strings(family: Any, args: Any, bytewise: Any) -> None:
    response = ModelResponse([ToolCallPart("typed", args)])

    snapshot = restored_snapshot(response, family, bytewise)

    assert snapshot.state == "DONE", snapshot.error
    assert len(snapshot.calls) == 1
    assert snapshot.calls[0].name == "typed"
    assert dict(snapshot.calls[0].args) == args


@pytest.mark.parametrize(
    ("family", "value"),
    [
        ("glm", 1),
        ("glm", {"nested": True}),
        ("qwen", "1"),
        ("qwen", "true"),
        ("qwen", '{"x":1}'),
    ],
)
def test_render_tool_call_rejects_unrepresentable_argument_values(family: Any, value: Any) -> None:
    with pytest.raises(ValueError, match="cannot preserve"):
        render_tool_call(family, "typed", {"value": value})


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "value",
    [
        {"nested": ("not", "JSON")},
        {"nested": {1: "non-string key"}},
    ],
)
def test_recursive_non_json_arguments_are_rejected(family: Any, value: Any) -> None:
    with pytest.raises(ValueError, match=r"JSON|mapping|preserve"):
        render_tool_call(family, "typed", {"value": value})


@pytest.mark.parametrize("args", ["[]", "false", "0", '""'])
def test_falsy_non_object_json_arguments_are_rejected(args: str) -> None:
    with pytest.raises(ValueError, match="mapping"):
        _arguments(args)


@pytest.mark.parametrize("family", ["deepseek4", "deepseek41", "qwen"])
@pytest.mark.parametrize("bytewise", [False, True])
def test_valid_recursive_json_still_round_trips(family: Any, bytewise: Any) -> None:
    args = {
        "value": {
            "none": None,
            "boolean": True,
            "integer": 3,
            "float": 1.5,
            "string": "text",
            "list": [None, False, {"nested": "value"}],
        }
    }
    response = ModelResponse([ToolCallPart("typed", args)])

    snapshot = restored_snapshot(response, family, bytewise)

    assert snapshot.state == "DONE", snapshot.error
    assert dict(snapshot.calls[0].args) == args


@pytest.mark.parametrize("bytewise", [False, True])
def test_qwen_63_character_integer_round_trips(bytewise: Any) -> None:
    args = {"value": int("9" * 63)}
    response = ModelResponse([ToolCallPart("typed", args)])

    snapshot = restored_snapshot(response, "qwen", bytewise)

    assert snapshot.state == "DONE", snapshot.error
    assert dict(snapshot.calls[0].args) == args


@pytest.mark.parametrize("value", [int("9" * 64), 10**100])
def test_qwen_rejects_integers_that_parser_would_restore_as_strings(value: Any) -> None:
    with pytest.raises(ValueError, match="cannot preserve"):
        render_tool_call("qwen", "typed", {"value": value})


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("bytewise", [False, True])
def test_safe_structural_names_round_trip_exactly(family: Any, bytewise: Any) -> None:
    response = ModelResponse([ToolCallPart("safe tool", {"safe key": "value"})])

    snapshot = restored_snapshot(response, family, bytewise)

    assert snapshot.state == "DONE", snapshot.error
    assert snapshot.calls[0].name == "safe tool"
    assert dict(snapshot.calls[0].args) == {"safe key": "value"}


@pytest.mark.parametrize("bytewise", [False, True])
def test_glm_argument_key_uses_existing_closing_tag_escaping(bytewise: Any) -> None:
    args = {"bad</arg_key>key": "value"}
    response = ModelResponse([ToolCallPart("safe", args)])

    snapshot = restored_snapshot(response, "glm", bytewise)

    assert snapshot.state == "DONE", snapshot.error
    assert dict(snapshot.calls[0].args) == args


@pytest.mark.parametrize(
    ("family", "name", "key"),
    [
        *[
            case
            for family in ("deepseek4", "deepseek41")
            for case in [
                (family, 'bad"name', "safe"),
                (family, "safe", 'bad"key'),
                (family, "bad>name", "safe"),
                (family, "safe", "bad>key"),
                (family, "", "safe"),
                (family, "safe", ""),
            ]
        ],
        ("glm", " padded ", "safe"),
        ("glm", "safe", " padded "),
        ("glm", "bad<name", "safe"),
        ("glm", "", "safe"),
        ("glm", "safe", ""),
        ("qwen", "bad>name", "safe"),
        ("qwen", "safe", "bad>key"),
        ("qwen", " padded ", "safe"),
        ("qwen", "safe", " padded "),
        ("qwen", "", "safe"),
        ("qwen", "safe", ""),
    ],
)
def test_render_rejects_nonrepresentable_structural_names(family: Any, name: Any, key: Any) -> None:
    with pytest.raises(ValueError, match="cannot preserve"):
        render_tool_call(family, name, {key: "value"})


@pytest.mark.parametrize("family", FAMILIES)
def test_render_rejects_non_string_argument_keys(family: Any) -> None:
    with pytest.raises(ValueError, match="cannot preserve"):
        render_tool_call(family, "safe", cast("Any", {1: "value"}))


@pytest.mark.parametrize("family", FAMILIES)
def test_user_and_tool_result_payload_markup_stays_plain(family: Any) -> None:
    payload = "payload <tool_call> ｜DSML｜ <|im_start|>"
    with fake_engine(family) as engine:
        session = engine.session()
        MessageBinder(session).prepare(
            [
                ModelRequest(
                    [
                        UserPromptPart(payload),
                        ToolReturnPart("read_file", payload),
                    ]
                )
            ]
        )

        for kind in ("user", "tool"):
            segment = next(s for s in session.transcript.segments if s.kind == kind)
            tokens = session.transcript.tokens[segment.start : segment.end]
            assert contains(tokens, engine.tokenize(payload))


@pytest.mark.parametrize("sampled", [False, True])
def test_raw_tool_syntax_is_trusted_only_for_sampled_responses(sampled: Any) -> None:
    raw = "<tool_call>\n<function=raw_tool>\n<parameter=path>\nraw.txt\n</parameter>\n</function>\n</tool_call>"
    response = ModelResponse(
        [
            ToolCallPart(
                "canonical_tool",
                {"path": "canonical.txt"},
                provider_details={"ds4py.raw": raw},
            )
        ],
        metadata={"ds4.source": "sampled"} if sampled else None,
    )

    with fake_engine("qwen") as engine:
        session = engine.session()
        MessageBinder(session).prepare([response])
        assistant = next(s for s in session.transcript.segments if s.kind == "assistant")
        snapshot = parse(
            decode(engine, session.transcript.tokens[assistant.start : assistant.end]),
            engine,
        )

        assert snapshot.state == "DONE", snapshot.error
        assert snapshot.calls[0].name == ("raw_tool" if sampled else "canonical_tool")
        assert dict(snapshot.calls[0].args) == {"path": "raw.txt" if sampled else "canonical.txt"}


def test_sampled_thinking_response_round_trips_native_tokens() -> None:
    request = ModelRequest([UserPromptPart("think")])
    response = ModelResponse([ThinkingPart("reason"), TextPart("answer")])
    sequence = tuple(tokenize_rendered("reason</think>answer"))

    def script(prompt: Any, generated: Any) -> Any:
        return sequence[len(generated)] if len(generated) < len(sequence) else FAKE_EOS

    with fake_engine("qwen", think=ThinkMode.LOW, script=script) as engine:
        session = engine.session()
        binder = MessageBinder(session)
        binder.prepare([request])
        list(session.run_round(max_tokens=32))
        binder.bind_response(response)
        original = list(session.transcript.tokens)

    with fake_engine("qwen", think=ThinkMode.LOW) as engine:
        session = engine.session()
        MessageBinder(session).prepare([request, response])
        assert list(session.transcript.tokens) == original


def test_render_tool_call_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="family"):
        render_tool_call("unknown", "read", {})
