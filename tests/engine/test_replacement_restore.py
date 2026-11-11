"""A failed rollback must poison the worker, never reuse mismatched native KV."""

from collections.abc import Sequence

import pytest

import ds4py
from ds4py._native.fake import FakeNative, FakeSession
from ds4py._native.protocol import CancelFn, ProgressFn, SyncStatus
from ds4py.engine.transcript import Segment, TranscriptReplacement
from ds4py.errors import EngineError, WorkerPoisonedError


@pytest.mark.parametrize("failure", ["error", "interrupted"])
@pytest.mark.parametrize("full", [False, True])
def test_replacement_restoration_fails_closed(monkeypatch: pytest.MonkeyPatch, failure: str, full: bool) -> None:
    with ds4py.open("f.gguf", ctx=3 if full else 1024, native=FakeNative()) as engine:
        session = engine.session()
        session.append_tokens((1, 2, 3), "user")
        session._worker.call(lambda e, native: native.sync((1, 2) if full else (1, 2, 3))).result()
        if full:
            session._worker.call(lambda e, native: native.eval(3)).result()
        sync = FakeSession.sync
        mismatched: list[tuple[int, ...]] = []

        def fail_restore(
            native: FakeSession,
            tokens: Sequence[int],
            *,
            progress: ProgressFn | None = None,
            cancel: CancelFn | None = None,
        ) -> SyncStatus:
            if tuple(tokens) == (4, 5):
                sync(native, tokens, progress=progress, cancel=cancel)
                return SyncStatus.INTERRUPTED
            assert tuple(tokens) == ((1, 2) if full else (1, 2, 3))
            mismatched.append(tuple(native.tokens()))
            if failure == "error":
                raise EngineError("native restoration failed")
            return SyncStatus.INTERRUPTED

        monkeypatch.setattr(FakeSession, "sync", fail_restore)
        replacement = TranscriptReplacement((4, 5), (Segment(0, 2, "user", {}),))
        with pytest.raises(EngineError) as caught:
            session.replace_transcript(replacement, reason="review")
        assert mismatched == [(4, 5)]
        assert session.transcript.tokens == (1, 2, 3)
        assert session._worker._poisoned_by is caught.value
        assert isinstance(caught.value.__context__, EngineError)
        assert "transcript replacement (review) did not complete" in str(caught.value.__context__)
        with pytest.raises(WorkerPoisonedError):
            session.plan()
        with pytest.raises(WorkerPoisonedError):
            session.replace_transcript(replacement, reason="retry")
