import random
from typing import Any

import pytest

from ds4py.engine.tool_syntax.dsml import Dsml
from ds4py.engine.tool_syntax.glm import Glm
from ds4py.engine.tool_syntax.qwen import Qwen
from ds4py.engine.tool_syntax.text import escape_tool_text

FAMILIES = [("dsml", False), ("dsml", True), ("glm", False), ("qwen", False)]


def parser_for(syntax: Any, v41: Any) -> Any:
    if syntax == "dsml":
        return Dsml(v41=v41)
    if syntax == "glm":
        return Glm()
    return Qwen()


def closing_tag(syntax: Any, v41: Any) -> str:
    if syntax == "glm":
        return "</arg_value>"
    if syntax == "qwen":
        return "</parameter>"
    return "</｜DSML｜ parameter>" if v41 else "</｜DSML｜parameter>"


def make_stanza(syntax: Any, v41: Any, value: Any) -> str:
    close = closing_tag(syntax, v41)
    escaped = escape_tool_text(value, close)
    if syntax == "glm":
        return f"<tool_call>echo<arg_key>value</arg_key><arg_value>{escaped}</arg_value></tool_call>"
    if syntax == "qwen":
        return f"<tool_call><function=echo><parameter=value>{escaped}</parameter></function></tool_call>"
    calls, invoke, parameter = (" calls", " invoke", " parameter") if v41 else ("tool_calls", "invoke", "parameter")
    return (
        f'<｜DSML｜{calls}><｜DSML｜{invoke} name="echo">'
        f'<｜DSML｜{parameter} name="value" string="true">{escaped}</｜DSML｜{parameter}>'
        f"</｜DSML｜{invoke}></｜DSML｜{calls}>"
    )


def partitions(data: Any, rng: Any) -> Any:
    cuts = sorted(rng.randint(0, len(data)) for _ in range(20))
    bounds = [0, *cuts, len(data)]
    return [data[start:end] for start, end in zip(bounds, bounds[1:])]


@pytest.mark.parametrize("syntax,v41", FAMILIES)
@pytest.mark.parametrize("iteration", range(25))
def test_seeded_valid_calls_roundtrip_and_match_random_chunking(
    parse_chunks: Any, syntax: Any, v41: Any, iteration: Any
) -> None:
    rng = random.Random(f"0:valid:{syntax}:{v41}:{iteration}")
    tokens = (
        "a",
        "Z",
        "0",
        "α",
        "雪",
        "🙂",
        "<",
        ">",
        "&",
        "\n",
        "\t",
        " ",
        "&lt;",
        "</tool_call>",
        "</think>",
        closing_tag(syntax, v41),
    )
    body = "".join(rng.choice(tokens) for _ in range(rng.randint(0, 30)))
    expected_value = "α" + body + "ω"
    case = {"syntax": syntax, "v41": v41}
    data = make_stanza(syntax, v41, expected_value).encode()
    expected = parse_chunks(case, [data])
    observed = parse_chunks(case, partitions(data, rng))
    assert observed == expected
    assert observed["state"] == "DONE"
    assert observed["calls"] == (("echo", (("value", expected_value),)),)


@pytest.mark.parametrize("syntax,v41", FAMILIES)
def test_seeded_plain_prose_never_becomes_executable(parse_chunks: Any, syntax: Any, v41: Any) -> None:
    rng = random.Random(f"0:prose:{syntax}:{v41}")
    tokens = ("word", "α", "1 < 2", "</", "<thinking>", "&amp;", "DSML", "|", "｜", "\n")
    markers = ("<tool_call>", "<｜DSML｜", "<DSML｜", "｜DSML｜", "|DSML|", "</DSML｜", "<think>", "</think>")
    for _ in range(40):
        text = "".join(rng.choice(tokens) for _ in range(30))
        for marker in markers:
            text = text.replace(marker, " ")
        data = text.encode()
        case = {"syntax": syntax, "v41": v41}
        observed = parse_chunks(case, partitions(data, rng))
        assert observed["state"] == "SEARCH"
        assert observed["error"] is None
        assert observed["calls"] == () and observed["raw"] == b""


@pytest.mark.parametrize("syntax,v41", FAMILIES)
def test_empty_chunks_and_repeated_terminal_operations_are_stable(syntax: Any, v41: Any) -> None:
    parser = parser_for(syntax, v41)
    initial = parser.feed(b"")
    assert parser.feed(b"") == initial
    data = make_stanza(syntax, v41, "abc").encode()
    for byte in data:
        assert parser.feed(b"") == parser.feed(b"")
        parser.feed(bytes([byte]))
    done = parser.finish()
    assert done.state == "DONE"
    assert parser.finish() == done
    assert parser.feed(b"") == done
    assert parser.feed(b"ignored") == done
    assert not parser.wants_greedy_sampling()

    broken = parser_for(syntax, v41)
    broken.feed(data[:-3])
    error = broken.finish()
    assert error.state == "ERROR"
    assert broken.finish() == error
    assert broken.feed(data) == error
    assert not broken.wants_greedy_sampling()


@pytest.mark.parametrize("syntax,v41", FAMILIES)
def test_alternating_empty_chunks_matches_one_shot(parse_chunks: Any, syntax: Any, v41: Any) -> None:
    case = {"syntax": syntax, "v41": v41}
    data = make_stanza(syntax, v41, "x<>&\ny").encode()
    chunks = [piece for byte in data for piece in (b"", bytes([byte]), b"")]
    assert parse_chunks(case, chunks) == parse_chunks(case, [data])
