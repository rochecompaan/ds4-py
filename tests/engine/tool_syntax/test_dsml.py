import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from ds4py._native.fake import FakeNative
from ds4py.engine.tool_syntax import make_parser
from ds4py.engine.tool_syntax.dsml import Dsml

CASES = json.loads((Path(__file__).parents[2] / "fixtures" / "tool_syntax" / "dsml.json").read_text())


def observed_calls(snapshot: Any) -> Any:
    return [{"name": call.name, "args": dict(call.args)} for call in snapshot.calls]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["source"])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096], ids=lambda size: f"chunk-{size}")
def test_dsml_fixtures(case: Any, chunk_size: Any) -> None:
    parser = Dsml(v41=case.get("v41", False))
    data = case["input"].encode()
    for offset in range(0, len(data), chunk_size):
        snapshot = parser.feed(data[offset : offset + chunk_size])
    snapshot = parser.finish()
    assert snapshot.state == case["state"], snapshot.error
    assert observed_calls(snapshot) == case["calls"]
    if case["state"] == "ERROR":
        assert snapshot.error


def test_raw_contains_only_the_canonical_tool_stanza() -> None:
    parser = Dsml()
    ordinary = parser.feed(b"ordinary prose ")
    assert ordinary.raw == b""
    started = parser.feed("<DSML｜tool_calls>".encode())
    assert started.raw == "<｜DSML｜tool_calls>".encode()
    assert b"ordinary" not in started.raw

    implicit = Dsml()
    snapshot = implicit.feed("<DSML｜invoke".encode())
    assert snapshot.raw == "<｜DSML｜tool_calls><｜DSML｜invoke".encode()


def test_snapshots_are_immutable_and_do_not_change_with_later_feeds() -> None:
    parser = Dsml()
    first = parser.feed(
        '<｜DSML｜tool_calls><｜DSML｜invoke name="read"><｜DSML｜parameter name="path" string="true">a</｜DSML｜parameter></｜DSML｜invoke>'.encode()
    )
    assert len(first.calls) == 1
    with pytest.raises(TypeError):
        cast("Any", first.calls[0].args)["path"] = "changed"
    second = parser.feed("</｜DSML｜tool_calls>".encode())
    assert second.state == "DONE"
    assert first.state == "STRUCTURAL"
    assert first.raw != second.raw
    assert dict(first.calls[0].args) == {"path": "a"}


def test_feed_after_done_or_error_is_a_no_op() -> None:
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


@pytest.mark.parametrize("partial", ["<｜DS", "<DSML｜tool_calls>", '<｜DSML｜tool_calls><｜DSML｜invoke name="read">'])
def test_finish_rejects_partial_openers_and_active_stanzas(partial: Any) -> None:
    parser = Dsml()
    parser.feed(partial.encode())
    snapshot = parser.finish()
    assert snapshot.state == "ERROR"
    assert snapshot.calls == ()
    assert "incomplete" in cast("Any", snapshot.error)


def test_finish_allows_ordinary_text_and_a_single_angle_bracket() -> None:
    for text in ("ordinary prose", "1 < 2"):
        parser = Dsml()
        parser.feed(text.encode())
        snapshot = parser.finish()
        assert snapshot.state == "SEARCH" and snapshot.calls == ()


def test_greedy_sampling_follows_structure_and_partial_closing_delimiter() -> None:
    opening = Dsml()
    opening.feed(b"<")
    assert not opening.wants_greedy_sampling()
    opening.feed("｜".encode())
    assert opening.wants_greedy_sampling()

    parser = Dsml()
    structural = parser.feed("<｜DSML｜tool_calls>".encode())
    assert structural.state == "STRUCTURAL" and parser.wants_greedy_sampling()
    value = parser.feed('<｜DSML｜invoke name="read"><｜DSML｜parameter name="path" string="true">abc'.encode())
    assert value.state == "PARAM_VALUE" and not parser.wants_greedy_sampling()
    closing = parser.feed("</｜DSML｜param".encode())
    assert closing.param_close_prefix and parser.wants_greedy_sampling()
    done = parser.feed("eter></｜DSML｜invoke></｜DSML｜tool_calls>".encode())
    assert done.state == "DONE" and not parser.wants_greedy_sampling()

    malformed = Dsml()
    assert malformed.feed("｜DSML｜".encode()).state == "ERROR"
    assert not malformed.wants_greedy_sampling()


@pytest.mark.parametrize("marker", ["｜DSML｜", "|DSML|", "<DSML｜", "</DSML｜"])
def test_malformed_dsml_markers_outside_blocks_are_errors(marker: Any) -> None:
    parser = Dsml()
    snapshot = parser.feed(("prose " + marker + " trailing").encode())
    assert snapshot.state == "ERROR" and snapshot.calls == ()


@pytest.mark.parametrize("v41", [False, True])
@pytest.mark.parametrize("explicit_invoke_close", [True, False])
def test_invalid_utf8_in_ignored_attribute_is_parser_error_for_all_chunks(
    v41: Any,
    explicit_invoke_close: Any,
) -> None:
    invoke, parameter, calls = (
        (" invoke", " parameter", " calls")
        if v41
        else (
            "invoke",
            "parameter",
            "tool_calls",
        )
    )
    invoke_close = f"</｜DSML｜{invoke}>" if explicit_invoke_close else ""
    data = (
        f'<｜DSML｜{calls}><｜DSML｜{invoke} name="read" extra="'.encode()
        + b"\xff"
        + (
            f'"><｜DSML｜{parameter} name="path" string="true">x</｜DSML｜{parameter}>{invoke_close}</｜DSML｜{calls}>'
        ).encode()
    )
    partitions = [[data], [bytes([byte]) for byte in data]]
    partitions += [[data[:split], data[split:]] for split in range(len(data) + 1)]

    for chunks in partitions:
        parser = Dsml(v41=v41)
        snapshot: Any = None
        for chunk in chunks:
            snapshot = parser.feed(chunk)
        assert snapshot.state == "ERROR"
        assert "raw" in cast("Any", snapshot.error) and "UTF-8" in cast("Any", snapshot.error)
        assert snapshot.calls == ()


def test_split_think_markers_reject_dsml_inside_thinking() -> None:
    parser = Dsml()
    parser.feed(b"<th")
    assert parser.feed(b"ink>").in_think
    snapshot = parser.feed("｜DSML｜".encode())
    assert snapshot.state == "ERROR"
    assert snapshot.in_think and snapshot.dsml_in_think
    assert snapshot.calls == ()


def test_think_tags_inside_parameter_values_are_literal() -> None:
    snapshot = Dsml().feed(
        next(case for case in CASES if case["source"].startswith("C test_agent_tool_argument"))["input"].encode()
    )
    assert snapshot.state == "DONE"
    assert not snapshot.in_think and not snapshot.dsml_in_think


def test_make_parser_selects_deepseek_versions() -> None:
    base = FakeNative()._info
    assert isinstance(make_parser(base), Dsml)
    v41 = make_parser(replace(base, family="deepseek41", is_deepseek41=True))
    assert isinstance(v41, Dsml) and v41.v41
