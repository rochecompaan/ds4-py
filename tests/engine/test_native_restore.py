"""Recovery must restore actual native prefixes or make the worker unusable."""

from collections.abc import Sequence

import pytest

import ds4py
from ds4py._native.fake import FakeSession
from ds4py._native.protocol import CancelFn, ProgressFn, SessionHandle, SyncStatus
from ds4py.engine.events import SummaryToken, Synced, TextDelta, Token
from ds4py.errors import EngineError, WorkerPoisonedError


def test_invalid_summary_restore_interrupted_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from ds4py._native.fake import FakeNative

    with ds4py.open("f.gguf", ctx=4096, native=FakeNative()) as engine:
        session = engine.session()
        session.append_tokens((11, 12), "user")
        session._worker.call(lambda e, native: native.sync((31, 32))).result()
        session._worker.call(lambda e, native: native.invalidate()).result()
        before = (session.transcript.tokens, session.transcript.segments)
        observed: list[object] = []
        session.on_event(lambda event: observed.append(event.payload.get("event")))
        sync = FakeSession.sync
        private_lengths: list[int] = []

        def interrupt_restore(
            native: FakeSession,
            tokens: Sequence[int],
            *,
            progress: ProgressFn | None = None,
            cancel: CancelFn | None = None,
        ) -> SyncStatus:
            if tuple(tokens) == (31, 32):
                private_lengths.append(len(native.tokens()))
                return SyncStatus.INTERRUPTED
            return sync(native, tokens, progress=progress, cancel=cancel)

        monkeypatch.setattr(FakeSession, "sync", interrupt_restore)
        with pytest.raises(EngineError):
            session.prepare_summary(0)
        assert private_lengths == [926]
        assert (session.transcript.tokens, session.transcript.segments) == before
        assert not any(isinstance(event, (Token, TextDelta)) for event in observed)
        with pytest.raises(WorkerPoisonedError):
            session.plan()


@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("tokens", [(), (31, 32), (31,) * 2047 + (32,)])
def test_restore_tokens_preserves_prefix_and_validity(tokens: tuple[int, ...], valid: bool) -> None:
    from ds4py._native.fake import FakeNative
    from ds4py.engine.native_restore import restore_native_tokens

    with ds4py.open("f.gguf", ctx=2048, native=FakeNative()) as engine:
        session = engine.session()

        def restore(native: SessionHandle) -> None:
            assert isinstance(native, FakeSession)
            native.sync((91, 92))
            native.calls.clear()
            poisoned: list[BaseException] = []
            restore_native_tokens(native, tokens, valid=valid, poison=poisoned.append)
            assert tuple(native.tokens()) == tokens
            assert native.checkpoint_valid() is (valid and bool(tokens))
            expected = [("rewind", 0)] if not tokens else [("sync", min(len(tokens), 2047))]
            if len(tokens) == 2048:
                expected.append(("eval", 32))
            assert native.calls == expected
            assert not poisoned

        session._worker.call(lambda e, native: restore(native)).result()


@pytest.mark.parametrize("failure", ["interrupted", "sync", "eval", "rewind"])
def test_restore_tokens_poison_on_incomplete_or_failed_restore(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from ds4py._native.fake import FakeNative
    from ds4py.engine.native_restore import restore_native_tokens

    with ds4py.open("f.gguf", ctx=3, native=FakeNative()) as engine:
        session = engine.session()
        original = EngineError("original operation failed")
        injected = EngineError("restoration failed")
        evaluations: list[int] = []

        def fail(native: FakeSession, *args: object, **kwargs: object) -> SyncStatus:
            if failure == "interrupted":
                return SyncStatus.INTERRUPTED
            raise injected

        def evaluate(native: FakeSession, token: int) -> None:
            evaluations.append(token)

        monkeypatch.setattr(FakeSession, "eval", evaluate)
        monkeypatch.setattr(FakeSession, "sync" if failure == "interrupted" else failure, fail)
        tokens = () if failure == "rewind" else (31, 32, 33)

        def restore(native: SessionHandle) -> None:
            try:
                raise original
            except EngineError:
                restore_native_tokens(native, tokens, valid=False, poison=session._worker._poison)

        with pytest.raises(EngineError) as caught:
            session._worker.call(lambda e, native: restore(native)).result()
        assert caught.value.__context__ is original
        if failure != "interrupted":
            assert caught.value is injected
        assert session._worker._poisoned_by is caught.value
        assert not evaluations
        with pytest.raises(WorkerPoisonedError):
            session.plan()


@pytest.mark.parametrize("operation", ["summary", "replacement"])
@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("tokens", [(), (31, 32), (31,) * 2047 + (32,)])
def test_recovery_restores_actual_prefix(
    operation: str, valid: bool, tokens: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ds4py._native.fake import FakeNative
    from ds4py.engine.transcript import Segment, TranscriptReplacement

    with ds4py.open("f.gguf", ctx=2048, native=FakeNative()) as engine:
        session = engine.session()
        # Python includes an unevaluated EOS; it is not the native restore source.
        session.append_tokens((11, 12, 0), "user")

        def seed(native: SessionHandle) -> None:
            if tokens:
                native.sync(tokens[:-1] if len(tokens) == 2048 else tokens)
                if len(tokens) == 2048:
                    native.eval(tokens[-1])
            if not valid:
                native.invalidate()

        session._worker.call(lambda e, native: seed(native)).result()
        before = (session.transcript.tokens, session.transcript.segments)
        sync = FakeSession.sync

        def interrupt_replacement(
            native: FakeSession,
            target: Sequence[int],
            *,
            progress: ProgressFn | None = None,
            cancel: CancelFn | None = None,
        ) -> SyncStatus:
            status = sync(native, target, progress=progress, cancel=cancel)
            return SyncStatus.INTERRUPTED if tuple(target) == (41, 42) else status

        monkeypatch.setattr(FakeSession, "sync", interrupt_replacement)
        observed: list[object] = []
        session.on_event(lambda event: observed.append(event.payload.get("event")))
        if operation == "summary":
            assert session.prepare_summary(0).summary == "ok"
        else:
            replacement = TranscriptReplacement((41, 42), (Segment(0, 2, "user", {}),))
            with pytest.raises(EngineError, match="did not complete"):
                session.replace_transcript(replacement, reason="test")
            assert not any(isinstance(event, (Synced, SummaryToken, Token, TextDelta)) for event in observed)
        state = session._worker.call(lambda e, native: (tuple(native.tokens()), native.checkpoint_valid())).result()
        assert state == (tokens, valid and bool(tokens))
        assert (session.transcript.tokens, session.transcript.segments) == before


@pytest.mark.parametrize("failure", ["none", "load", "close", "load_and_close"])
def test_summary_snapshot_preference_and_disposal(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from ds4py._native.fake import FakeNative, FakeSnapshot

    with ds4py.open("f.gguf", ctx=2048, native=FakeNative()) as engine:
        session = engine.session()
        session._worker.call(lambda e, native: native.sync((31, 32))).result()
        load, close = FakeSession.load_snapshot, FakeSnapshot.close
        actions: list[str] = []

        def load_snapshot(native: FakeSession, snapshot: FakeSnapshot) -> None:
            actions.append("load")
            if "load" in failure:
                raise EngineError("load failed")
            load(native, snapshot)

        def close_snapshot(snapshot: FakeSnapshot) -> None:
            actions.append("close")
            close(snapshot)
            if "close" in failure:
                raise EngineError("close failed")

        def no_fallback(native: FakeSession) -> None:
            pytest.fail("valid snapshot should be preferred to token rebuild")

        monkeypatch.setattr(FakeSession, "load_snapshot", load_snapshot)
        monkeypatch.setattr(FakeSnapshot, "close", close_snapshot)
        monkeypatch.setattr(FakeSession, "invalidate", no_fallback)
        if failure == "none":
            assert session.prepare_summary(0).summary == "ok"
            assert session._worker.call(lambda e, native: tuple(native.tokens())).result() == (31, 32)
        else:
            with pytest.raises(EngineError):
                session.prepare_summary(0)
            with pytest.raises(WorkerPoisonedError):
                session.plan()
        assert actions == ["load", "close"]


@pytest.mark.parametrize("failure", ["interrupted", "sync", "eval", "rewind"])
@pytest.mark.parametrize("operation", ["summary", "replacement"])
def test_failed_operation_and_failed_recovery_keep_error_context(
    operation: str, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ds4py._native.fake import FakeNative
    from ds4py.engine.transcript import Segment, TranscriptReplacement

    original = EngineError("original operation failed")
    recovery = EngineError("recovery failed")

    def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
        raise original

    with ds4py.open("f.gguf", ctx=2048, native=FakeNative(script=script)) as engine:
        session = engine.session()
        session.append_tokens((11, 12, 0), "user")
        tokens = () if failure == "rewind" else ((31,) * 2047 + (32,) if failure == "eval" else (31, 32))

        def seed(native: SessionHandle) -> None:
            if tokens:
                native.sync(tokens[:-1] if failure == "eval" else tokens)
                if failure == "eval":
                    native.eval(32)
            native.invalidate()

        session._worker.call(lambda e, native: seed(native)).result()
        sync = FakeSession.sync

        def fail_sync(
            native: FakeSession,
            target: Sequence[int],
            *,
            progress: ProgressFn | None = None,
            cancel: CancelFn | None = None,
        ) -> SyncStatus:
            if tuple(target) == (41, 42):
                sync(native, target, progress=progress, cancel=cancel)
                raise original
            if tuple(target) == tokens and failure in ("interrupted", "sync"):
                if failure == "interrupted":
                    return SyncStatus.INTERRUPTED
                raise recovery
            return sync(native, target, progress=progress, cancel=cancel)

        def fail(native: FakeSession, *args: object) -> None:
            raise recovery

        monkeypatch.setattr(FakeSession, "sync", fail_sync)
        if failure in ("eval", "rewind"):
            monkeypatch.setattr(FakeSession, failure, fail)

        def perform_operation() -> None:
            if operation == "summary":
                session.prepare_summary(0)
            else:
                replacement = TranscriptReplacement((41, 42), (Segment(0, 2, "user", {}),))
                session.replace_transcript(replacement, reason="test")

        with pytest.raises(EngineError) as caught:
            perform_operation()
        assert caught.value is not original
        assert caught.value.__context__ is original
        assert session.transcript.tokens == (11, 12, 0)
        with pytest.raises(WorkerPoisonedError):
            session.plan()
