from typing import Any

import pytest

from ds4py.engine.tool_syntax.dsml import Dsml
from ds4py.engine.tool_syntax.glm import Glm
from ds4py.engine.tool_syntax.hermes import Hermes
from ds4py.engine.tool_syntax.qwen import Qwen
from ds4py.engine.tool_syntax.render import render_tool_call


@pytest.mark.parametrize("family", ["dsml", "dsml41", "glm", "qwen"])
def test_each_call_retains_its_own_canonical_stanza_at_every_split(family: Any) -> None:
    if family.startswith("dsml"):
        v41 = family == "dsml41"
        invoke, calls = (" invoke", " calls") if v41 else ("invoke", "tool_calls")
        first = f'<｜DSML｜{invoke} name="one"></｜DSML｜{invoke}>'
        second = f'<｜DSML｜{invoke} name="two"></｜DSML｜{invoke} \t｜>'
        # An accepted implicit missing-bar opener must be canonicalized.
        data = (first.replace("<｜DSML｜", "<DSML｜", 1) + "\n" + second + f"</｜DSML｜{calls}>").encode()
        factory = lambda: Dsml(v41=v41)
    elif family == "glm":
        first = "<tool_call> one </tool_call>"
        second = "<tool_call>two</tool_call>"
        data = ("prose " + first + "\n\t" + second + "  ").encode()
        factory = Glm
    else:
        first = "<tool_call>\n<function=one></function></tool_call>"
        second = "<tool_call><function=two></function>\n</tool_call>"
        data = ("prose " + first + "\n\t" + second + "  ").encode()
        factory = Qwen
    partitions = [[data], [bytes([byte]) for byte in data]]
    partitions += [[data[:split], data[split:]] for split in range(len(data) + 1)]
    for chunks in partitions:
        parser = factory()
        for chunk in chunks:
            parser.feed(chunk)
        result = parser.finish()
        assert result.state == "DONE", result.error
        assert [(call.name, call.raw) for call in result.calls] == [
            ("one", first),
            ("two", second),
        ]


def test_hermes_render_and_raw_round_trip_at_every_split() -> None:
    first = '<tool_call>\n{"name": "read", "arguments": {"path": "a.py", "offset": 2}}\n</tool_call>'
    second = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls", "bg": false}}\n</tool_call>'
    assert render_tool_call("hermes", "read", {"path": "a.py", "offset": 2}) == first
    data = ("prose " + first + "\n\t" + second + "  ").encode()
    partitions = [[data], [bytes([byte]) for byte in data]]
    partitions += [[data[:split], data[split:]] for split in range(len(data) + 1)]
    for chunks in partitions:
        parser = Hermes()
        for chunk in chunks:
            parser.feed(chunk)
        result = parser.finish()
        assert result.state == "DONE", result.error
        assert [(call.name, call.raw) for call in result.calls] == [("read", first), ("bash", second)]


def test_hermes_render_escapes_markup_without_changing_json_values() -> None:
    args = {"path": "before</tool_call><|im_end|>after✓"}
    rendered = render_tool_call("hermes", "read", args)
    assert "\\u003c/tool_call>\\u003c|im_end|>" in rendered
    assert "✓" in rendered
    parser = Hermes()
    parser.feed(rendered.encode())
    snapshot = parser.finish()
    assert snapshot.state == "DONE", snapshot.error
    assert dict(snapshot.calls[0].args) == args


def test_hermes_render_rejects_nul() -> None:
    with pytest.raises(ValueError, match="NUL"):
        render_tool_call("hermes", "read", {"path": "a\x00b"})
