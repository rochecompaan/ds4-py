import pytest

from ds4py.engine.pause import ControlSpan, StopSequenceMatcher


def test_utf8_stop_sequence_matches_across_bytes_and_token_chunks() -> None:
    sequence = "🛑stop".encode()
    matcher = StopSequenceMatcher([sequence])
    data = b"prefix" + sequence + b"suffix"
    for byte in data:
        match = matcher.feed(bytes([byte]))
        if match is not None:
            assert match == sequence
            break
    else:
        pytest.fail("UTF-8 stop sequence did not match")


def test_kmp_state_stays_bounded() -> None:
    matcher = StopSequenceMatcher([b"abab"])
    assert matcher.feed(b"ab") is None and matcher.pending == 2
    assert matcher.feed(b"x" * 20_000) is None
    assert matcher.pending == 0


def test_overlapping_matches_retain_fallback_state_for_resume() -> None:
    matcher = StopSequenceMatcher([b"abab"])
    assert matcher.feed(b"abab") == b"abab"
    assert matcher.pending == 2
    assert matcher.feed(b"ab") == b"abab"


def test_control_span_stops_and_resumes_with_matcher_state() -> None:
    span = ControlSpan([b"stop"])
    assert span.feed(b"st") is None
    assert span.active_stop_sequences == [b"stop"]
    assert span.feed(b"op") == b"stop" and span.stop_sequence == b"stop"
    span.resume()
    assert span.stop_sequence is None
    assert span.feed(b"st") is None
    assert span.feed(b"op") == b"stop"


def test_empty_sequences_are_rejected_and_empty_region_has_no_active_list() -> None:
    with pytest.raises(ValueError):
        StopSequenceMatcher([b""])
    with pytest.raises(UnicodeDecodeError):
        StopSequenceMatcher([b"\xff"])
    span = ControlSpan([b"stop"])
    assert span.active_stop_sequences == []
    assert span.feed(b"ordinary") is None
    assert span.active_stop_sequences == []
