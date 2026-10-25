from dataclasses import replace
import json
from pathlib import Path

import pytest

from ds4py._native.fake import FakeNative
from ds4py.engine.tool_syntax import make_parser
from ds4py.engine.tool_syntax.glm import Glm


CASES = json.loads((Path(__file__).parents[2] / "fixtures" / "tool_syntax" / "glm.json").read_text())


def observed_calls(snapshot):
    return [{"name": call.name, "args": dict(call.args)} for call in snapshot.calls]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["source"])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096], ids=lambda size: f"chunk-{size}")
def test_glm_fixtures(case, chunk_size):
    parser = Glm()
    data = case["input"].encode()
    for offset in range(0, len(data), chunk_size):
        parser.feed(data[offset:offset + chunk_size])
    snapshot = parser.finish()
    assert snapshot.state == case["state"], snapshot.error
    assert observed_calls(snapshot) == case["calls"]
    if case["state"] == "ERROR":
        assert snapshot.error


def test_raw_ignores_prose_before_and_after_the_canonical_block():
    stanza = b"<tool_call>read<arg_key>path</arg_key><arg_value>x</arg_value></tool_call>"
    snapshot = Glm().feed(b"before" + stanza + b"after")
    assert snapshot.state == "DONE"
    assert snapshot.raw == stanza


def test_partial_next_call_does_not_finish_as_trailing_whitespace():
    parser = Glm()
    parser.feed(b"<tool_call>one</tool_call><tool")
    snapshot = parser.finish()
    assert snapshot.state == "ERROR" and snapshot.calls == ()


def test_greedy_sampling_boundaries():
    opening = Glm()
    opening.feed(b"<tool")
    assert opening.wants_greedy_sampling()

    parser = Glm()
    parser.feed(b"<tool_call>bash<arg_key>command</arg_key><arg_value>printf hi")
    assert not parser.wants_greedy_sampling()
    snapshot = parser.feed(b"</arg")
    assert snapshot.param_close_prefix and parser.wants_greedy_sampling()
    snapshot = parser.feed(b"_value></tool_call>")
    assert parser.finish().state == "DONE"
    assert not parser.wants_greedy_sampling()


def test_tool_opener_inside_split_think_marker_is_an_error():
    parser = Glm()
    parser.feed(b"<th")
    parser.feed(b"ink>")
    snapshot = parser.feed(b"<tool_call>bash</tool_call>")
    assert snapshot.state == "ERROR" and snapshot.calls == ()
    assert snapshot.in_think and snapshot.dsml_in_think
    assert parser.finish() == snapshot


def test_literal_think_markup_inside_value_does_not_change_think_state():
    case = next(case for case in CASES if "literal_markup" in case["source"])
    parser = Glm()
    parser.feed(case["input"].encode())
    snapshot = parser.finish()
    assert snapshot.state == "DONE"
    assert not snapshot.in_think and not snapshot.dsml_in_think


def test_make_parser_selects_glm():
    info = replace(FakeNative()._info, family="glm", is_glm_dsa=True)
    assert isinstance(make_parser(info), Glm)
