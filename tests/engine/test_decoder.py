from ds4py.engine.decoder import Decoder
from ds4py.engine.events import TextDelta
from ds4py.engine.tool_syntax.glm import Glm
from ds4py.engine.transcript import Transcript


def collapse(deltas):
    result = []
    for delta in deltas:
        if result and result[-1][1] == delta.thinking:
            result[-1] = (result[-1][0] + delta.text, delta.thinking)
        else:
            result.append((delta.text, delta.thinking))
    return result


def test_ascii_passes_through_and_empty_feeds_emit_nothing():
    decoder = Decoder()
    assert decoder.feed(b"") == []
    assert decoder.feed(b"hello ASCII") == [TextDelta("hello ASCII", False)]
    assert decoder.feed(b"") == []


def test_multibyte_utf8_is_identical_at_every_split_and_bytewise():
    text = "Aα雪🙂Z"
    data = text.encode()
    expected = "".join(delta.text for delta in Decoder().feed(data))
    assert expected == text
    for split in range(len(data) + 1):
        decoder = Decoder()
        deltas = decoder.feed(data[:split]) + decoder.feed(data[split:])
        assert "".join(delta.text for delta in deltas) == text
        assert all("�" not in delta.text for delta in deltas)
    decoder = Decoder()
    deltas = [delta for byte in data for delta in decoder.feed(bytes([byte]))]
    assert "".join(delta.text for delta in deltas) == text


def test_think_markers_work_at_every_split_and_change_delta_labels():
    data = b"before<think>inside</think>after"
    expected = [("before", False), ("inside", True), ("after", False)]
    assert collapse(Decoder().feed(data)) == expected
    for split in range(len(data) + 1):
        decoder = Decoder()
        deltas = decoder.feed(data[:split]) + decoder.feed(data[split:])
        assert collapse(deltas) == expected, split


def test_pending_scanner_flushes_ordinary_and_marker_like_text():
    for text in ("<", "</", "<thinking>", "<thinkx>", "<<", "a < b"):
        decoder = Decoder()
        deltas = decoder.feed(text.encode()) + decoder.finish()
        assert "".join(delta.text for delta in deltas) == text
        assert len(decoder._pending) <= 16

    decoder = Decoder()
    assert collapse(decoder.feed(b"<think>x")) == [("x", True)]


def test_finish_flushes_safe_pending_text_and_incomplete_utf8():
    decoder = Decoder()
    assert decoder.feed(b"<th") == []
    assert decoder.finish() == [TextDelta("<th", False)]
    assert decoder.finish() == []

    incomplete = Decoder()
    assert incomplete.feed(b"\xf0\x9f") == []
    tail = incomplete.finish()
    assert "".join(delta.text for delta in tail) == "�"
    assert incomplete.finish() == []
    assert Decoder().finish() == []


def test_initial_thinking_labels_pre_marker_output():
    decoder = Decoder(thinking=True)
    assert collapse(decoder.feed(b"reason</think>answer")) == [
        ("reason", True), ("answer", False),
    ]


def test_display_decoding_does_not_mutate_transcript_or_tool_parser():
    transcript = Transcript()
    parser = Glm()
    before = parser.feed(b"")
    decoder = Decoder()
    decoder.feed(b"<tool_call>read<arg_key>path</arg_key><arg_value>x</arg_value></tool_call>")
    decoder.finish()
    assert transcript.tokens == () and transcript.segments == ()
    assert parser.feed(b"") == before
