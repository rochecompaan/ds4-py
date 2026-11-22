import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ds4py._native.fake import FakeNative
from ds4py.engine.tool_syntax import make_parser
from ds4py.engine.tool_syntax.hermes import Hermes

CASES = json.loads((Path(__file__).parents[2] / "fixtures" / "tool_syntax" / "hermes.json").read_text())


def observed_calls(snapshot: Any) -> list[dict[str, object]]:
    return [{"name": call.name, "args": dict(call.args)} for call in snapshot.calls]


def parse_chunks(chunks: list[bytes]) -> Any:
    parser = Hermes()
    for chunk in chunks:
        parser.feed(chunk)
    return parser.finish()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["source"])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096], ids=lambda size: f"chunk-{size}")
def test_hermes_fixtures(case: Any, chunk_size: int) -> None:
    data = case["input"].encode()
    snapshot = parse_chunks([data[offset : offset + chunk_size] for offset in range(0, len(data), chunk_size)])
    assert snapshot.state == case["state"], snapshot.error
    assert observed_calls(snapshot) == case["calls"]
    assert [call.raw for call in snapshot.calls] == case.get("raw", [])
    if case["state"] == "ERROR":
        assert snapshot.error is not None
        assert case["error"] in snapshot.error


@pytest.mark.parametrize(
    "value",
    [
        "1x",
        "0x10",
        "nan",
        "01",
        "1.",
        "-",
        "+1",
        "inf",
        "1e",
        "[0x1]",
        "[1,]",
        "tru",
        '{"a" 1}',
        "NaN",
        "Infinity",
        "-Infinity",
    ],
)
def test_invalid_json_values_are_errors(value: str) -> None:
    text = f'<tool_call>{{"name":"read","arguments":{{"offset":{value}}}}}</tool_call>'
    snapshot = parse_chunks([text.encode()])
    assert snapshot.state == "ERROR"
    assert snapshot.calls == ()


@pytest.mark.parametrize(
    "value", ["0", "-0", "7", "-1.5e3", "2E+2", "true", "false", "null", '[1, 2.5, {"a": null}]', "{}"]
)
def test_valid_json_values_keep_their_types(value: str) -> None:
    text = f'<tool_call>{{"name":"read","arguments":{{"offset":{value}}}}}</tool_call>'
    snapshot = parse_chunks([text.encode()])
    assert snapshot.state == "DONE", snapshot.error
    assert dict(snapshot.calls[0].args)["offset"] == json.loads(value)


@pytest.mark.parametrize("depth", [256, 257, 5000])
def test_json_depth_is_bounded_without_recursion_escaping(depth: int) -> None:
    value = "[" * depth + "]" * depth
    text = f'<tool_call>{{"name":"read","arguments":{{"value":{value}}}}}</tool_call>'
    snapshot = parse_chunks([text.encode()])
    if depth == 256:
        assert snapshot.state == "DONE", snapshot.error
    else:
        assert snapshot.state == "ERROR"
        assert snapshot.calls == ()


def test_capture_restore_and_fork_resume_mid_call() -> None:
    first = b'<tool_call>{"name":"write","arguments":{"path":"a.py","content":"print('
    second = b'"hi")"}}}</tool_call><tool_call>{"name":"read","arguments":{}}</tool_call>'
    parser = Hermes()
    parser.feed(first)
    state = parser.capture()
    fork = parser.fork()
    restored = Hermes()
    restored.restore(state)
    expected = parse_chunks([first + second])
    for candidate in (parser, fork, restored):
        candidate.feed(second)
        snapshot = candidate.finish()
        assert snapshot == expected
        assert state.raw == first


def test_greedy_sampling_is_structural_outside_argument_values() -> None:
    parser = Hermes()
    parser.feed(b"<tool")
    assert parser.wants_greedy_sampling()
    parser.feed(b'_call>{"name":"wri')
    assert parser.wants_greedy_sampling()
    parser.feed(b'te","arguments":{"path":')
    assert parser.wants_greedy_sampling()
    parser.feed(b'"a.py')
    assert not parser.wants_greedy_sampling()
    parser.feed(b'","content')
    assert parser.wants_greedy_sampling()
    parser.feed(b'":"print(\\"x\\")')
    assert not parser.wants_greedy_sampling()
    parser.feed(b'"}}</tool_call>')
    assert not parser.wants_greedy_sampling()
    assert parser.finish().state == "DONE"
    assert not parser.wants_greedy_sampling()


def test_prose_after_call_waits_for_finish_and_allows_another_call() -> None:
    parser = Hermes()
    parser.feed(b'<tool_call>{"name":"read","arguments":{}}</tool_call>prose')
    assert parser.state != "DONE"
    assert len(parser.capture().calls) == 1
    parser.feed(b'<tool_call>{"name":"bash","arguments":{}}</tool_call>')
    snapshot = parser.finish()
    assert snapshot.state == "DONE"
    assert [call.name for call in snapshot.calls] == ["read", "bash"]


def test_partial_opener_after_call_is_prose_at_finish() -> None:
    parser = Hermes()
    parser.feed(b'<tool_call>{"name":"read","arguments":{}}</tool_call>see <tool_ca')
    snapshot = parser.finish()
    assert snapshot.state == "DONE"
    assert len(snapshot.calls) == 1


def test_make_parser_selects_hermes_before_qwen() -> None:
    info = replace(FakeNative()._info, family="hermes", is_qwen4=True, is_mellum=True)
    assert isinstance(make_parser(info), Hermes)
