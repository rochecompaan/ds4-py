"""Checkpoint disposal is part of the pre-publication replacement transaction."""

import asyncio
from collections.abc import Sequence

import pytest
from test_compaction_commit import fake_engine, message_history, replacement_for

from ds4py._native.fake import FakeSession, FakeSnapshot
from ds4py._native.protocol import CancelFn, ProgressFn, SyncStatus
from ds4py.binding import MessageBinder
from ds4py.compaction_commit import commit_compaction
from ds4py.engine.transcript import TranscriptReplacement
from ds4py.errors import EngineError, WorkerPoisonedError


@pytest.mark.parametrize("restore_fails", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_post_sync_snapshot_release_failure_cannot_publish_or_reuse(
    monkeypatch: pytest.MonkeyPatch,
    restore_fails: bool,
    empty: bool,
) -> None:
    # Without the cleanup transaction, successful native sync escapes while
    # binder/messages/Python tokens remain old and the worker remains usable.
    async def scenario() -> None:
        with fake_engine() as engine:
            session = engine.session()
            binder = MessageBinder(session)
            messages = message_history()
            binder.prepare(messages)
            before = session.transcript.tokens
            session._worker.call(lambda e, native: native.sync(before)).result()
            session.checkpoint("tail")
            original = list(messages), session.transcript.segments, binder.records(), binder.export_segments()
            kept = [] if empty else list(messages[2:])
            replacement = TranscriptReplacement((), ()) if empty else replacement_for(session, kept)
            sync = FakeSession.sync
            close = FakeSnapshot.close
            native_states: list[tuple[int, ...]] = []
            failed_native_states: list[tuple[int, ...]] = []
            failed_restores: list[tuple[int, ...]] = []
            current: FakeSession | None = None

            def observe_sync(
                native: FakeSession,
                tokens: Sequence[int],
                *,
                progress: ProgressFn | None = None,
                cancel: CancelFn | None = None,
            ) -> SyncStatus:
                nonlocal current
                current = native
                if restore_fails and failed_native_states and tuple(tokens) == before:
                    failed_restores.append(tuple(native.tokens()))
                    raise EngineError("restoration failed after release")
                status = sync(native, tokens, progress=progress, cancel=cancel)
                native_states.append(tuple(native.tokens()))
                return status

            def release(snapshot: FakeSnapshot) -> None:
                if tuple(snapshot.tokens) == before:
                    assert current is not None
                    failed_native_states.append(tuple(current.tokens()))
                    raise EngineError("snapshot release failed")
                close(snapshot)

            monkeypatch.setattr(FakeSession, "sync", observe_sync)
            monkeypatch.setattr(FakeSnapshot, "close", release)
            with pytest.raises(EngineError) as caught:
                await commit_compaction(binder, messages, kept, replacement)
            assert failed_native_states == [replacement.tokens]
            assert session.transcript.tokens == before
            assert (messages, session.transcript.segments, binder.records(), binder.export_segments()) == original
            assert session._worker._poisoned_by is caught.value
            if restore_fails:
                assert str(caught.value) == "restoration failed after release"
                assert isinstance(caught.value.__context__, EngineError)
                assert str(caught.value.__context__) == "snapshot release failed"
                assert failed_restores == [replacement.tokens]
            else:
                assert str(caught.value) == "snapshot release failed"
                assert native_states[-1] == before
                assert not failed_restores
            with pytest.raises(WorkerPoisonedError):
                session.plan()
            stable = list(messages), binder.records(), session.transcript.tokens, list(native_states)
            await asyncio.sleep(0)
            assert (messages, binder.records(), session.transcript.tokens, native_states) == stable

    asyncio.run(scenario())
