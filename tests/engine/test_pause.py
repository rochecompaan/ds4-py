import pytest

from ds4py.engine.pause import (
    DECODE_RESULT, ENCODE_STEP_TOKENS, ControlSpan, StopSequenceMatcher,
)


def test_utf8_stop_sequence_matches_across_bytes_and_token_chunks():
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


def test_kmp_state_clones_inherit_without_sharing_and_stays_bounded():
    matcher = StopSequenceMatcher([b"abab"])
    assert matcher.feed(b"ab") is None and matcher.pending == 2
    clone = matcher.clone()
    assert clone is not matcher and clone.pending == 2
    assert clone.feed(b"ab") == b"abab" and matcher.pending == 2
    inherited = StopSequenceMatcher([b"zz"])
    inherited.inherit(matcher)
    assert inherited.sequences == matcher.sequences and inherited.pending == 2
    long_input = b"x" * 20_000
    assert matcher.feed(long_input) is None
    assert matcher.pending == 0 and matcher.buffered == 0


def test_overlapping_matches_retain_fallback_state_for_resume():
    matcher = StopSequenceMatcher([b"abab"])
    assert matcher.feed(b"abab") == b"abab"
    assert matcher.pending == 2
    assert matcher.feed(b"ab") == b"abab"


def test_control_span_gates_regions_and_preserves_state_across_feed_append():
    span = ControlSpan([b"stop"], state="SEARCH")
    assert span.feed(b"st", state="SEARCH") is None
    assert span.matcher.pending == 0 and span.active_stop_sequences == []
    assert span.feed(b"st", state=ENCODE_STEP_TOKENS) is None
    assert span.active_stop_sequences == [b"stop"]
    assert span.Append(b"op") == b"stop" and span.stop_sequence == b"stop"
    span.finish()
    assert span.stop_sequence is None
    span.Feed(b"st", state=DECODE_RESULT)
    assert span.Append(b"op") == b"stop"
    span.done()
    assert span.stop_sequence is None


def test_control_span_record_keeps_override_ids_only_for_explicit_record():
    span = ControlSpan([b"stop"])
    record = span.Record(b"result", (4, 7), done=True)
    assert record.text == b"result" and record.override_ids == (4, 7)
    assert span.text_record == record and span.stop_sequence is None


def test_empty_sequences_are_rejected_and_empty_region_has_no_active_list():
    with pytest.raises(ValueError):
        StopSequenceMatcher([b""])
    with pytest.raises(UnicodeDecodeError):
        StopSequenceMatcher([b"\xff"])
    span = ControlSpan([b"stop"])
    assert span.active_stop_sequences == []
    assert span.Feed(b"ordinary") is None
    assert span.active_stop_sequences == []
