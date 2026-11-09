"""Transcript replacement validates first, then commits through the engine worker."""

from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

import ds4py
from ds4py._native.fake import FakeNative
from ds4py.engine.session import Engine
from ds4py.engine.transcript import Segment, TranscriptReplacement
from ds4py.engine.worker import SYSTEM

if TYPE_CHECKING:
    from collections.abc import Sequence

EMPTY = MappingProxyType({})


def fake_engine(*, ctx: int = 1024) -> Engine:
    return ds4py.open("f.gguf", ctx=ctx, native=FakeNative())


def compacted(system: Sequence[int], tail: Sequence[int]) -> TranscriptReplacement:
    tokens = (*system, *tail)
    return TranscriptReplacement(
        tuple(tokens),
        (
            Segment(0, len(system), "system", EMPTY),
            Segment(len(system), len(tokens), "user", EMPTY),
        ),
    )


def test_replace_transcript_uses_longest_valid_checkpoint() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        system = session.transcript.tokens
        session.append_user("old question")
        list(session.run_round(max_tokens=0))

        compacted_replacement = compacted(system, (401, 402, 403))
        result = session.replace_transcript(compacted_replacement, reason="compaction")

        assert result.kind == "restore"
        assert result.checkpoint == SYSTEM
        assert session.transcript.tokens == compacted_replacement.tokens
        assert session.transcript.segments == compacted_replacement.segments


def test_replace_transcript_rebuilds_from_zero_without_a_matching_checkpoint() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        system = session.transcript.tokens
        session.append_user("old question")
        list(session.run_round(max_tokens=0))
        session.drop_checkpoint(SYSTEM)

        replacement = compacted(system, (501, 502))
        result = session.replace_transcript(replacement, reason="compaction")

        assert result.kind == "rebuild"
        assert result.checkpoint is None
        assert session.ctx_used == len(replacement.tokens)
        assert session.transcript.tokens == replacement.tokens
        assert session.transcript.segments == replacement.segments


@pytest.mark.parametrize(
    ("tokens", "segments"),
    [
        ((1, 2, 3, 4), (Segment(0, 2, "system", EMPTY), Segment(3, 4, "user", EMPTY))),
        ((1, 2, 3, 4), (Segment(0, 3, "system", EMPTY), Segment(2, 4, "user", EMPTY))),
        ((1, 2, 3, 4), (Segment(0, 2, "system", EMPTY), Segment(2, 3, "user", EMPTY))),
        ((1, 2, 3, 4), (Segment(1, 4, "user", EMPTY),)),
    ],
    ids=["gap", "overlap", "stale-token-count", "nonzero-start"],
)
def test_invalid_replacement_leaves_native_and_transcript_state_unchanged(
    tokens: tuple[int, ...],
    segments: tuple[Segment, ...],
) -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.append_tokens([7, 8], "user")
        list(session.run_round(max_tokens=0))
        before_tokens = session.transcript.tokens
        before_segments = session.transcript.segments

        with pytest.raises(ValueError, match="replacement segments"):
            session.replace_transcript(TranscriptReplacement(tokens, segments), reason="compaction")

        assert session.transcript.tokens == before_tokens
        assert session.transcript.segments == before_segments
        assert session.ctx_used == 2
