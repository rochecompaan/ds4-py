from dataclasses import replace
import json
from pathlib import Path

import pytest

from ds4py._native.fake import FakeNative
from ds4py.engine.tool_syntax import make_parser
from ds4py.engine.tool_syntax.qwen import Qwen


CASES = json.loads((Path(__file__).parents[2] / "fixtures" / "tool_syntax" / "qwen.json").read_text())


def observed_calls(snapshot):
    return [{"name": call.name, "args": dict(call.args)} for call in snapshot.calls]


def parse(text):
    parser = Qwen()
    parser.feed(text.encode())
    return parser.finish()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["source"])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096], ids=lambda size: f"chunk-{size}")
def test_qwen_fixtures(case, chunk_size):
    parser = Qwen()
    data = case["input"].encode()
    for offset in range(0, len(data), chunk_size):
        parser.feed(data[offset:offset + chunk_size])
    snapshot = parser.finish()
    assert snapshot.state == case["state"], snapshot.error
    assert observed_calls(snapshot) == case["calls"]
    if case["state"] == "ERROR":
        assert snapshot.error


def test_numeric_classification_uses_the_c_64_byte_guard():
    def numeric(length):
        digits = "1" * length
        text = f"<tool_call><function=n><parameter=value>{digits}</parameter></function></tool_call>"
        return parse(text).calls[0].args["value"]

    assert numeric(63) == int("1" * 63)
    assert numeric(64) == "1" * 64


@pytest.mark.parametrize(("text", "expected", "value_type"), [
    ("+1", 1, int),
    ("01", 1, int),
    ("1.", 1.0, float),
    (".5", 0.5, float),
])
def test_c_numeric_forms_preserve_typed_arguments(text, expected, value_type):
    snapshot = parse(
        f"<tool_call><function=echo><parameter=value>{text}</parameter>"
        "</function></tool_call>"
    )
    assert snapshot.state == "DONE", snapshot.error
    value = snapshot.calls[0].args["value"]
    assert value == expected
    assert type(value) is value_type


def test_ordinary_text_is_not_json_decoded_by_experiment():
    snapshot = parse("<tool_call><function=echo><parameter=a>001x</parameter><parameter=b>True</parameter><parameter=c>\"hello\"</parameter></function></tool_call>")
    assert snapshot.calls[0].args == {"a": "001x", "b": "True", "c": "\"hello\""}


def test_qwen_closing_delimiter_requests_greedy_sampling():
    parser = Qwen()
    parser.feed(b"<tool_call><function=bash><parameter=command>printf hi")
    assert not parser.wants_greedy_sampling()
    snapshot = parser.feed(b"</param")
    assert snapshot.param_close_prefix and parser.wants_greedy_sampling()
    parser.feed(b"eter></function></tool_call>")
    assert parser.finish().state == "DONE"
    assert not parser.wants_greedy_sampling()


def test_literal_think_markup_does_not_change_think_state():
    case = next(case for case in CASES if "literal_markup" in case["source"])
    snapshot = parse(case["input"])
    assert snapshot.state == "DONE"
    assert not snapshot.in_think and not snapshot.dsml_in_think


def test_make_parser_selects_qwen():
    info = replace(FakeNative()._info, family="qwen", is_qwen4=True)
    assert isinstance(make_parser(info), Qwen)
