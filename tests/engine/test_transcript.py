from typing import Any, cast

import pytest

from ds4py.engine.transcript import KINDS, Transcript


def test_transcript_segments() -> None:
    t = Transcript()
    t.append([1, 2, 3], "user")
    t.append([4], "assistant")
    t.extend([5, 6])
    assert t.tokens == (1, 2, 3, 4, 5, 6)
    assert len(t) == 6
    assert [(s.start, s.end, s.kind) for s in t.segments] == [(0, 3, "user"), (3, 6, "assistant")]
    t.truncate(4)
    assert [(s.start, s.end) for s in t.segments] == [(0, 3), (3, 4)]
    t.truncate(3)
    assert len(t.segments) == 1
    assert t.tokens == (1, 2, 3)
    with pytest.raises(ValueError, match="kind"):
        t.append([7], "chat")
    with pytest.raises(ValueError):
        t.append([], "user")


def test_snapshots_and_metadata_do_not_change_when_transcript_changes() -> None:
    t = Transcript()
    tokens, meta = [1], {"turn": 1}
    segment = t.append(tokens, "user", meta)
    old_tokens, old_segments = t.tokens, t.segments
    tokens.append(99)
    meta["turn"] = 99
    t.extend([2])
    assert old_tokens == (1,)
    assert old_segments == (segment,)
    assert segment.end == 1 and segment.meta == {"turn": 1}
    with pytest.raises(TypeError):
        cast(Any, t.segments[0].meta)["turn"] = 2
    assert t.tokens == (1, 2)


def test_chat_flag_is_explicit_and_snapshot_safe() -> None:
    transcript = Transcript()
    assert not transcript.chat
    transcript.append([1], "user")
    assert not transcript.chat
    transcript.set_chat()
    assert transcript.chat
    transcript.extend([2])
    assert transcript.chat


def test_truncate_bounds_and_extend_without_segment() -> None:
    t = Transcript()
    with pytest.raises(ValueError):
        t.extend([1])
    t.extend([])
    for kind in KINDS:
        t.append([1], kind)
    before = t.tokens
    for n in (-1, len(t) + 1):
        with pytest.raises(ValueError):
            t.truncate(n)
        assert t.tokens == before
    t.truncate(len(t))
    assert t.tokens == before
    t.truncate(0)
    assert t.tokens == () and t.segments == ()
    t.append([2], "user")
    assert t.segments[0].start == 0
