from dataclasses import replace
import json
from pathlib import Path

import pytest

from ds4py._native.fake import FakeNative
from ds4py.engine.tool_syntax import make_parser
from ds4py.engine.tool_syntax.dsml import Dsml


CASES = json.loads((Path(__file__).parents[2] / "fixtures" / "tool_syntax" / "dsml.json").read_text())


def observed_calls(snapshot):
    return [{"name": call.name, "args": dict(call.args)} for call in snapshot.calls]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["source"])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096], ids=lambda size: f"chunk-{size}")
def test_dsml_fixtures(case, chunk_size):
    parser = Dsml(v41=case.get("v41", False))
    data = case["input"].encode()
    for offset in range(0, len(data), chunk_size):
        snapshot = parser.feed(data[offset:offset + chunk_size])
    snapshot = parser.finish()
    assert snapshot.state == case["state"], snapshot.error
    assert observed_calls(snapshot) == case["calls"]
    if case["state"] == "ERROR":
        assert snapshot.error


def test_raw_contains_only_the_canonical_tool_stanza():
    parser = Dsml()
    ordinary = parser.feed("ordinary prose ".encode())
    assert ordinary.raw == b""
    started = parser.feed("<DSML｜tool_calls>".encode())
    assert started.raw == "<｜DSML｜tool_calls>".encode()
    assert b"ordinary" not in started.raw

    implicit = Dsml()
    snapshot = implicit.feed("<DSML｜invoke".encode())
    assert snapshot.raw == "<｜DSML｜tool_calls><｜DSML｜invoke".encode()


def test_snapshots_are_immutable_and_do_not_change_with_later_feeds():
    parser = Dsml()
    first = parser.feed("<｜DSML｜tool_calls><｜DSML｜invoke name=\"read\"><｜DSML｜parameter name=\"path\" string=\"true\">a</｜DSML｜parameter></｜DSML｜invoke>".encode())
    assert len(first.calls) == 1
    with pytest.raises(TypeError):
        first.calls[0].args["path"] = "changed"
    second = parser.feed("</｜DSML｜tool_calls>".encode())
    assert second.state == "DONE"
    assert first.state == "STRUCTURAL"
    assert first.raw != second.raw
    assert dict(first.calls[0].args) == {"path": "a"}


def test_feed_after_done_or_error_is_a_no_op():
    text = CASES[0]["input"].encode()
    parser = Dsml()
    done = parser.feed(text + b"ignored trailing prose")
    assert done.state == "DONE"
    assert done.raw == text
    assert parser.feed(b"more") == done

    error_parser = Dsml()
    error = error_parser.feed("｜DSML｜".encode())
    assert error.state == "ERROR"
    assert error_parser.feed(text) == error


@pytest.mark.parametrize("partial", ["<｜DS", "<DSML｜tool_calls>", "<｜DSML｜tool_calls><｜DSML｜invoke name=\"read\">"])
def test_finish_rejects_partial_openers_and_active_stanzas(partial):
    parser = Dsml()
    parser.feed(partial.encode())
    snapshot = parser.finish()
    assert snapshot.state == "ERROR"
    assert snapshot.calls == ()
    assert "incomplete" in snapshot.error


def test_finish_allows_ordinary_text_and_a_single_angle_bracket():
    for text in ("ordinary prose", "1 < 2"):
        parser = Dsml()
        parser.feed(text.encode())
        snapshot = parser.finish()
        assert snapshot.state == "SEARCH" and snapshot.calls == ()


def test_greedy_sampling_follows_structure_and_partial_closing_delimiter():
    opening = Dsml()
    opening.feed(b"<")
    assert not opening.wants_greedy_sampling()
    opening.feed("｜".encode())
    assert opening.wants_greedy_sampling()

    parser = Dsml()
    structural = parser.feed("<｜DSML｜tool_calls>".encode())
    assert structural.state == "STRUCTURAL" and parser.wants_greedy_sampling()
    value = parser.feed("<｜DSML｜invoke name=\"read\"><｜DSML｜parameter name=\"path\" string=\"true\">abc".encode())
    assert value.state == "PARAM_VALUE" and not parser.wants_greedy_sampling()
    closing = parser.feed("</｜DSML｜param".encode())
    assert closing.param_close_prefix and parser.wants_greedy_sampling()
    done = parser.feed("eter></｜DSML｜invoke></｜DSML｜tool_calls>".encode())
    assert done.state == "DONE" and not parser.wants_greedy_sampling()

    malformed = Dsml()
    assert malformed.feed("｜DSML｜".encode()).state == "ERROR"
    assert not malformed.wants_greedy_sampling()


@pytest.mark.parametrize("marker", ["｜DSML｜", "|DSML|", "<DSML｜", "</DSML｜"])
def test_malformed_dsml_markers_outside_blocks_are_errors(marker):
    parser = Dsml()
    snapshot = parser.feed(("prose " + marker + " trailing").encode())
    assert snapshot.state == "ERROR" and snapshot.calls == ()


def test_split_think_markers_reject_dsml_inside_thinking():
    parser = Dsml()
    parser.feed(b"<th")
    assert parser.feed(b"ink>").in_think
    snapshot = parser.feed("｜DSML｜".encode())
    assert snapshot.state == "ERROR"
    assert snapshot.in_think and snapshot.dsml_in_think
    assert snapshot.calls == ()


def test_think_tags_inside_parameter_values_are_literal():
    snapshot = Dsml().feed(next(case for case in CASES if case["source"].startswith("C test_agent_tool_argument"))["input"].encode())
    assert snapshot.state == "DONE"
    assert not snapshot.in_think and not snapshot.dsml_in_think


def test_make_parser_selects_deepseek_versions():
    base = FakeNative()._info
    assert isinstance(make_parser(base), Dsml)
    v41 = make_parser(replace(base, family="deepseek41", is_deepseek41=True))
    assert isinstance(v41, Dsml) and v41.v41
