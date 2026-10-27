import pytest

from ds4py.engine.tool_syntax.text import escape_tool_text, unescape_tool_text


CLOSING_TAGS = [
    "</｜DSML｜parameter>",
    "</｜DSML｜ parameter>",
    "</arg_value>",
    "</parameter>",
    "</tool_result>",
    "</tool_response>",
]


@pytest.mark.parametrize("closing", CLOSING_TAGS)
def test_literal_and_escaped_closing_delimiters_round_trip(closing):
    escaped = "&lt;" + closing[1:]
    doubly_escaped = "&amp;lt;" + closing[1:]

    assert escape_tool_text(f"before {closing} after", closing) == f"before {escaped} after"
    assert escape_tool_text(escaped, closing) == doubly_escaped
    assert unescape_tool_text(escaped, closing) == closing
    assert unescape_tool_text(doubly_escaped, closing) == escaped


@pytest.mark.parametrize("closing", CLOSING_TAGS)
@pytest.mark.parametrize("prefixes", [0, 1, 2, 3])
def test_repeated_amp_prefixes_follow_the_c_loop(closing, prefixes):
    source = "&" + "amp;" * prefixes + "lt;" + closing[1:]
    escaped = "&" + "amp;" * (prefixes + 1) + "lt;" + closing[1:]

    assert escape_tool_text(source, closing) == escaped
    assert unescape_tool_text(escaped, closing) == source


@pytest.mark.parametrize("closing", CLOSING_TAGS)
def test_ordinary_entities_html_and_unrelated_text_are_unchanged(closing):
    text = "ordinary &amp; &lt; <p>&amp;</p> &lt;/other> &amp;lt;/other> &am &l ✓"
    assert unescape_tool_text(text, closing) == text
    assert escape_tool_text(text, closing) == text


@pytest.mark.parametrize("closing", CLOSING_TAGS)
def test_escape_then_unescape_returns_the_original_payload(closing):
    escaped = "&lt;" + closing[1:]
    payload = f"<p>&amp;</p>\n{closing}\n{escaped}\n&amp;{escaped[1:]}\n✓"
    assert unescape_tool_text(escape_tool_text(payload, closing), closing) == payload
