"""Transcript replacement validates first, then commits through the engine worker."""

from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

import ds4py
from ds4py._native.fake import FakeNative
from ds4py.engine.session import Engine
from ds4py.engine.transcript import Segment, Transcript, TranscriptReplacement
from ds4py.engine.worker import SYSTEM
from ds4py.errors import CheckpointError, EngineError

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
        ((1, 2, 3, 4), (Segment(0, 0, "user", EMPTY),)),
        ((1, 2, 3, 4), (Segment(0, -1, "user", EMPTY),)),
    ],
    ids=["gap", "overlap", "stale-token-count", "nonzero-start", "zero-length", "reversed"],
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


def test_replace_transcript_drops_checkpoints_past_the_new_end() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        system = session.transcript.tokens
        session.append_user("old question")
        list(session.run_round(max_tokens=0))
        turn = session.checkpoint("turn")
        assert len(system) < turn.tokens

        replacement = compacted(system, (601,))
        assert len(replacement.tokens) < turn.tokens
        session.replace_transcript(replacement, reason="compaction")

        with pytest.raises(CheckpointError):
            session.drop_checkpoint("turn")
        session.drop_checkpoint(SYSTEM)


def test_interrupted_replacement_leaves_native_and_transcript_state_unchanged() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        session.append_user("old question")
        list(session.run_round(max_tokens=0))
        tail = (701, 702)
        tokens = (*session.transcript.tokens, *tail)
        segments = (
            *session.transcript.segments,
            Segment(len(tokens) - len(tail), len(tokens), "user", EMPTY),
        )
        before_tokens = session.transcript.tokens
        before_segments = session.transcript.segments
        before_ctx = session.ctx_used
        session.cancel()

        with pytest.raises(EngineError, match=r"transcript replacement \(compaction\) did not complete"):
            session.replace_transcript(TranscriptReplacement(tokens, segments), reason="compaction")

        assert session.transcript.tokens == before_tokens
        assert session.transcript.segments == before_segments
        assert session.ctx_used == before_ctx


@pytest.mark.parametrize(
    "segments",
    [
        (Segment(1, 3, "user", EMPTY),),
        (Segment(0, 1, "user", EMPTY), Segment(2, 3, "user", EMPTY)),
        (Segment(0, 2, "user", EMPTY), Segment(1, 3, "user", EMPTY)),
        (Segment(0, 0, "user", EMPTY),),
        (Segment(0, -1, "user", EMPTY),),
        (Segment(0, 2, "user", EMPTY),),
    ],
)
@pytest.mark.parametrize("operation", ["replace", "install"])
def test_direct_transcript_validation_is_atomic(segments: tuple[Segment, ...], operation: str) -> None:
    transcript = Transcript()
    transcript.append([7, 8, 9], "user")
    before = (transcript.tokens, transcript.segments, transcript.chat)

    def invalid_operation() -> None:
        if operation == "replace":
            transcript.replace([1, 2, 3], segments)
        else:
            transcript.install(segments)

    with pytest.raises(ValueError, match="replacement segments"):
        invalid_operation()
    assert (transcript.tokens, transcript.segments, transcript.chat) == before


def test_slice_concat_preserves_exact_tokens_and_metadata() -> None:
    original = TranscriptReplacement(
        (10, 20, 30, 40, 50),
        (
            Segment(0, 1, "system", EMPTY),
            Segment(1, 3, "note", {"text": "remember"}),
            Segment(3, 5, "user", {"key": "u"}),
        ),
    )
    tail = original.slice(1)
    assert tail.tokens == (20, 30, 40, 50)
    assert tail.segments == (Segment(0, 2, "note", {"text": "remember"}), Segment(2, 4, "user", {"key": "u"}))
    assert TranscriptReplacement.concat(original.slice(0, 1), tail) == original
    assert original.slice(5) == TranscriptReplacement((), ())
    assert original.slice(2, 4).segments == (
        Segment(0, 1, "note", {"text": "remember"}),
        Segment(1, 2, "user", {"key": "u"}),
    )
    for start, end in [(-1, 2), (2, 1), (0, 6)]:
        with pytest.raises(ValueError, match="slice positions"):
            original.slice(start, end)
    with pytest.raises(ValueError, match="replacement segments"):
        TranscriptReplacement.concat(TranscriptReplacement((1,), ()))


def test_empty_replacement_is_valid_everywhere() -> None:
    transcript = Transcript()
    transcript.replace((), ())
    transcript.install(())
    assert transcript.chat and transcript.tokens == transcript.segments == ()
    with fake_engine() as engine:
        session = engine.session()
        session.append_tokens([7, 8], "user")
        list(session.run_round(max_tokens=0))
        session.checkpoint("history")
        assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == (7, 8)
        assert session.history.checkpoint_tokens == {"history": 2}

        session.replace_transcript(TranscriptReplacement((), ()), reason="empty")

        assert session.ctx_used == 0 and session.transcript.chat
        assert session.transcript.tokens == session.transcript.segments == ()
        assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == ()
        assert session.history.checkpoint_tokens == {}
        with pytest.raises(CheckpointError):
            session.drop_checkpoint("history")


@pytest.mark.parametrize("loaded", [None, 3])
def test_provenance_translates_structural_failures(loaded: int | None) -> None:
    from ds4py.binding_provenance import prepare_provenance

    with pytest.raises(EngineError, match="save record") as error:
        prepare_provenance([], [Segment(1, 3, "system", EMPTY)], loaded=loaded)
    assert isinstance(error.value.__cause__, ValueError)
