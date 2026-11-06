"""Fatal decode failures permanently disable ordinary native work."""

import io
import threading
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast, override

import pytest

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeEngine, FakeNative, FakeSession
from ds4py._native.protocol import EngineConfig, EngineHandle, SessionHandle, Snapshot
from ds4py.engine.decode import DecodeContext, GenerateJob
from ds4py.engine.events import Event
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.transcript import Transcript
from ds4py.engine.worker import (
    CheckpointJob,
    EngineWorker,
    LoadPayloadJob,
    ResetJob,
    SavePayloadJob,
    SyncJob,
    _JobEvents,
)
from ds4py.errors import EngineError, WorkerPoisonedError


class FaultNative(FakeNative):
    """Fail one native eval and optionally fail the attempted rewind cleanup."""

    def __init__(
        self,
        original: EngineError,
        *,
        cleanup: EngineError | None = None,
        partial: bool = False,
    ) -> None:
        super().__init__()
        self.original = original
        self.cleanup = cleanup
        self.partial = partial
        self.fault_calls: list[tuple[str, int]] = []
        self.session: FakeSession | None = None

    @override
    def open_engine(self, config: EngineConfig) -> FakeEngine:
        engine = super().open_engine(config)
        create_session = engine.create_session

        def create(ctx: int) -> FakeSession:
            session = create_session(ctx)
            self.session = session
            evaluate = session.eval
            rewind = session.rewind

            def eval_with_fault(token: int) -> None:
                self.fault_calls.append(("eval", token))
                if token == 99:
                    if self.partial:
                        evaluate(token)
                    raise self.original
                evaluate(token)

            def rewind_with_fault(position: int) -> None:
                self.fault_calls.append(("rewind", position))
                if self.cleanup is not None:
                    raise self.cleanup
                rewind(position)

            dynamic = cast("Any", session)
            dynamic.eval = eval_with_fault
            dynamic.rewind = rewind_with_fault
            return session

        cast("Any", engine).create_session = create
        return engine


class UncertainMultiTokenStep:
    hooks_every_sample = True

    def __init__(self, *, attempt_cleanup: bool = False) -> None:
        self.attempt_cleanup = attempt_cleanup
        self.run_calls = 0
        self.finally_calls = 0

    def run(self, session: SessionHandle, ctx: DecodeContext) -> None:
        self.run_calls += 1
        block_start = session.pos()
        ctx.begin_staging()
        try:
            for token in (98, 99):
                try:
                    verdict = ctx.accept(token)
                    session.eval(token)
                    ctx.stage(token, verdict)
                except EngineError as original:
                    if self.attempt_cleanup:
                        try:
                            session.rewind(block_start)
                        except EngineError as cleanup:
                            ctx.poison(cleanup)
                    ctx.poison(original)
        finally:
            # Poisoning is terminal: only Python-local bookkeeping is allowed here.
            self.finally_calls += 1


def run_uncertain_block(
    native: FaultNative,
    step: UncertainMultiTokenStep,
    *,
    snapshots: list[Snapshot] | None = None,
) -> tuple[Transcript, EngineWorker, EngineError]:
    worker = EngineWorker(native, EngineConfig(Path("f.gguf"), ctx=64))
    transcript = Transcript()
    transcript.append([5, 6], "user")
    list(worker.submit(SyncJob((5, 6))))
    if snapshots is not None:
        snapshots.append(worker.call(lambda engine, session: session.save_snapshot()).result())
    job = GenerateJob(SamplingPolicy(), Rng(1), 10, transcript, decode_step=step)
    events = worker.submit(job)
    with pytest.raises(EngineError) as caught:
        next(events)
    return transcript, worker, caught.value


@pytest.mark.parametrize("partial", [False, True], ids=["before-side-effect", "after-side-effect"])
def test_multi_token_native_failure_poisoning_discards_staged_tokens(partial: bool) -> None:
    original = EngineError("native eval failed")
    native = FaultNative(original, partial=partial)
    step = UncertainMultiTokenStep()
    transcript, worker, error = run_uncertain_block(native, step)
    try:
        assert error is original
        assert step.run_calls == 1
        assert step.finally_calls == 1
        assert transcript.tokens == (5, 6)
        assert native.fault_calls == [("eval", 98), ("eval", 99)]
    finally:
        worker.close()


def test_poison_preserves_original_and_cleanup_exception_chaining() -> None:
    original = EngineError("native eval failed")
    cleanup = EngineError("native rewind cleanup failed")
    native = FaultNative(original, cleanup=cleanup)
    step = UncertainMultiTokenStep(attempt_cleanup=True)
    transcript, worker, error = run_uncertain_block(native, step)
    try:
        assert error is cleanup
        assert cleanup.__context__ is original
        assert step.run_calls == 1
        assert step.finally_calls == 1
        assert transcript.tokens == (5, 6)
        assert native.fault_calls == [("eval", 98), ("eval", 99), ("rewind", 2)]
    finally:
        worker.close()


def assert_rejected(error: BaseException, fatal: BaseException) -> None:
    assert type(error) is WorkerPoisonedError
    assert str(error) == "engine worker is poisoned"
    assert error.__cause__ is fatal


def assert_event_wait_completes(events: _JobEvents) -> None:
    waiter = threading.Thread(target=events.wait, daemon=True)
    waiter.start()
    waiter.join(1)
    assert not waiter.is_alive()


def next_bounded(events: _JobEvents) -> Event:
    future: Future[Event] = Future()

    def consume() -> None:
        try:
            future.set_result(next(events))
        except EngineError as error:
            future.set_exception(error)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    try:
        return future.result(timeout=1)
    except TimeoutError as error:
        state = (events.finished.is_set(), events.queue.empty())
        raise AssertionError(f"event read timed out; finished/empty={state}") from error


def test_worker_rejects_all_later_calls_and_submitted_jobs_before_native() -> None:
    original = EngineError("native eval failed")
    native = FaultNative(original)
    step = UncertainMultiTokenStep()
    snapshots: list[Snapshot] = []
    transcript, worker, error = run_uncertain_block(native, step, snapshots=snapshots)
    snapshot = snapshots[0]
    try:
        assert error is original
        fake_session = native.session
        assert fake_session is not None
        calls_before = list(fake_session.calls)
        faults_before = list(native.fault_calls)
        entered: list[str] = []

        def callback(
            name: str,
            operation: Callable[[EngineHandle, SessionHandle], object],
        ) -> Callable[[EngineHandle, SessionHandle], object]:
            def run(engine: EngineHandle, session: SessionHandle) -> object:
                entered.append(name)
                return operation(engine, session)

            return run

        callbacks = [
            callback("eval", lambda engine, session: session.eval(100)),
            callback("rewind", lambda engine, session: session.rewind(2)),
            callback("save_snapshot", lambda engine, session: session.save_snapshot()),
            callback("load_snapshot", lambda engine, session: session.load_snapshot(snapshot)),
            callback("extension", handles_for_extension),
        ]
        rejections = []
        for operation in callbacks:
            future = worker.call(operation)
            with pytest.raises(WorkerPoisonedError) as caught:
                future.result(timeout=1)
            assert future.exception(timeout=1) is caught.value
            assert_rejected(caught.value, original)
            rejections.append(caught.value)

        assert rejections[0] is not rejections[1]
        with pytest.raises(WorkerPoisonedError) as caught:
            worker.plan((5, 6))
        assert_rejected(caught.value, original)

        jobs = [
            SyncJob((5, 6)),
            ResetJob(),
            CheckpointJob("after"),
            CheckpointJob("after", drop=True),
            GenerateJob(SamplingPolicy(), Rng(1), 2, transcript),
            SavePayloadJob((5, 6), b"header", io.BytesIO()),
            LoadPayloadJob(io.BytesIO(), 0, Transcript()),
        ]
        for job in jobs:
            events = worker.submit(job)
            with pytest.raises(WorkerPoisonedError) as caught:
                next(events)
            assert_rejected(caught.value, original)
            assert_event_wait_completes(events)

        assert entered == []
        assert fake_session.calls == calls_before
        assert native.fault_calls == faults_before
    finally:
        worker.close()
        snapshot.close()


def handles_for_extension(engine: EngineHandle, session: SessionHandle) -> tuple[int, int]:
    return engine.raw, session.raw


def test_rejection_chains_through_cleanup_to_the_original_native_error() -> None:
    original = EngineError("native eval failed")
    cleanup = EngineError("native cleanup failed")
    native = FaultNative(original, cleanup=cleanup)
    step = UncertainMultiTokenStep(attempt_cleanup=True)
    _transcript, worker, error = run_uncertain_block(native, step)
    try:
        assert error is cleanup
        with pytest.raises(WorkerPoisonedError) as caught:
            worker.call(lambda engine, session: session.pos()).result(timeout=1)
        assert_rejected(caught.value, cleanup)
        assert caught.value.__cause__ is cleanup
        assert cleanup.__context__ is original
    finally:
        worker.close()


def test_work_queued_before_poisoning_is_rejected_and_close_still_finishes() -> None:
    original = EngineError("native eval failed")
    native = FaultNative(original)
    worker = EngineWorker(native, EngineConfig(Path("f.gguf"), ctx=64))
    stopped = threading.Event()
    join = worker._thread.join

    def signal_stop_queued(timeout: float | None = None) -> None:
        # close() enqueues _STOP before joining the owner thread.
        stopped.set()
        join(timeout)

    cast("Any", worker._thread).join = signal_stop_queued
    transcript = Transcript()
    transcript.append([5, 6], "user")
    list(worker.submit(SyncJob((5, 6))))
    entered = threading.Event()
    release = threading.Event()

    class BlockingStep(UncertainMultiTokenStep):
        @override
        def run(self, session: SessionHandle, ctx: DecodeContext) -> None:
            entered.set()
            assert release.wait(2)
            super().run(session, ctx)

    step = BlockingStep()
    failing = worker.submit(GenerateJob(SamplingPolicy(), Rng(1), 10, transcript, decode_step=step))
    assert entered.wait(1)
    callback_entered = False

    def forbidden_callback(engine: EngineHandle, session: SessionHandle) -> None:
        nonlocal callback_entered
        callback_entered = True

    queued_call = worker.call(forbidden_callback)
    queued_job = worker.submit(SyncJob((5, 6)))
    closer = threading.Thread(target=worker.close, daemon=True)
    closer.start()
    assert stopped.wait(1)
    release.set()
    try:
        with pytest.raises(EngineError) as initial:
            next_bounded(failing)
        assert initial.value is original
        with pytest.raises(WorkerPoisonedError) as call_error:
            queued_call.result(timeout=1)
        assert_rejected(call_error.value, original)
        with pytest.raises(WorkerPoisonedError) as job_error:
            next_bounded(queued_job)
        assert_rejected(job_error.value, original)
        assert_event_wait_completes(queued_job)
        closer.join(1)
        assert not closer.is_alive()
        assert not callback_entered
        assert native.fault_calls == [("eval", 98), ("eval", 99)]
    finally:
        release.set()
        worker.close()
        closer.join(1)


def poison_public_session() -> tuple[Any, Any, FaultNative, EngineError, UncertainMultiTokenStep]:
    original = EngineError("native eval failed")
    native = FaultNative(original)
    engine = ds4py.open("f.gguf", ctx=64, think=ThinkMode.NONE, native=native)
    session = engine.session()
    session.append_tokens([5, 6], "user")
    step = UncertainMultiTokenStep()
    session.set_decode_step(step)
    with pytest.raises(EngineError) as caught:
        list(session.run_round(max_tokens=10))
    assert caught.value is original
    return engine, session, native, original, step


def test_public_facades_and_extension_callbacks_are_rejected_after_poisoning() -> None:
    engine, session, native, original, step = poison_public_session()
    try:
        fake_session = native.session
        assert fake_session is not None
        calls_before = list(fake_session.calls)
        faults_before = list(native.fault_calls)
        extension_calls = 0

        def extension(handles: Any) -> None:
            nonlocal extension_calls
            extension_calls += 1

        operations = [
            lambda: engine.tokenize("after"),
            lambda: engine.token_text(98),
            lambda: session.ctx_used,
            session.plan,
            lambda: session.set_think(ThinkMode.HIGH),
            lambda: session.reconcile_round(2),
            lambda: list(session.run_round(max_tokens=1)),
            lambda: session.checkpoint("after"),
            lambda: session.drop_checkpoint("after"),
            lambda: session.save_payload(io.BytesIO(), header=b"header"),
            lambda: session.load_payload(io.BytesIO(), payload_bytes=0),
            lambda: session.on_worker(extension).result(),
        ]
        for operation in operations:
            with pytest.raises(WorkerPoisonedError) as caught:
                operation()
            assert_rejected(caught.value, original)

        assert extension_calls == 0
        assert session._decode_step is step
        assert fake_session.calls == calls_before
        assert native.fault_calls == faults_before
        assert ds4py.WorkerPoisonedError is WorkerPoisonedError
    finally:
        engine.close()


def test_session_lifecycle_operations_do_not_clear_poisoning() -> None:
    engine, session, _native, original, _step = poison_public_session()
    try:

        class ScopedState:
            closes = 0

            def close(self) -> None:
                self.closes += 1

        state = ScopedState()
        assert session.state("scoped", lambda: state) is state
        lifecycle_events = []
        session.on_event(lambda event: lifecycle_events.append(event.kind))
        old = session.id

        with pytest.raises(WorkerPoisonedError) as new_error:
            session.new()
        assert_rejected(new_error.value, original)
        assert session.id == old
        assert state.closes == 1
        assert lifecycle_events == ["session_end"]
        with pytest.raises(WorkerPoisonedError):
            session.plan()

        target = "a" * 32
        with pytest.raises(WorkerPoisonedError) as switch_error:
            session.switch(target, session.plan)
        assert_rejected(switch_error.value, original)
        assert session.id == target
        with pytest.raises(WorkerPoisonedError):
            _ = session.ctx_used

        forked = session.fork()
        assert forked == session.id != target
        switched = session.switch("b" * 32)
        assert switched == session.id
        with pytest.raises(WorkerPoisonedError) as later:
            list(session.run_round(max_tokens=1))
        assert_rejected(later.value, original)
    finally:
        engine.close()
