"""Serial job execution and native resource ownership on one daemon thread."""

import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from contextlib import ExitStack
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Any, BinaryIO, Generic, TypeVar

from ds4py._native.protocol import (
    EngineConfig,
    EngineHandle,
    EngineInfo,
    Native,
    SessionHandle,
    Snapshot,
    SyncStatus,
)
from ds4py.errors import CheckpointError, EmptyPromptError, EngineError

from .decode import GenerateJob, generate
from .events import Checkpointed, Dropped, Event, Interrupted, Loaded, Prefill, Saved, Stopped, Synced
from .plan import Plan, plan_sync
from .transcript import Transcript

SYSTEM = "system"
T = TypeVar("T")
_END = object()
_STOP = object()


@dataclass(frozen=True)
class SyncJob:
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class CheckpointJob:
    key: str
    drop: bool = False


@dataclass(frozen=True)
class ResetJob:
    """Return the native session to an empty, unsynchronized transcript."""


@dataclass(frozen=True)
class SavePayloadJob:
    """Sync the transcript, then write the caller header and native payload."""

    tokens: tuple[int, ...]
    header: bytes
    file: BinaryIO

    def __post_init__(self) -> None:
        if not isinstance(self.header, bytes):
            raise ValueError("payload header must be bytes")


@dataclass(frozen=True)
class LoadPayloadJob:
    """Read one payload after the caller's header and replace the transcript."""

    file: BinaryIO
    payload_bytes: int
    transcript: Transcript


Job = SyncJob | CheckpointJob | ResetJob | GenerateJob | SavePayloadJob | LoadPayloadJob


@dataclass(frozen=True)
class Handles:
    engine: int
    session: int


class _JobEvents(Iterator[Event]):
    """A resumable event channel with an independent native-completion barrier."""

    def __init__(self) -> None:
        self.queue: SimpleQueue[Any] = SimpleQueue()
        self.finished = threading.Event()
        self._exhausted = False

    def __next__(self) -> Event:
        if self._exhausted:
            raise StopIteration
        while True:
            try:
                event = self.queue.get(timeout=0.1)
                break
            except Empty:
                # A caller interrupt can consume _END before recording EOF.
                # Recheck durable completion after every bounded receive, not
                # just once before a wait that could outlive the lost marker.
                if self.finished.is_set():
                    # Completion may have raced the timeout. All publications
                    # now precede finished, so drain any newly queued events.
                    try:
                        event = self.queue.get_nowait()
                    except Empty:
                        self._exhausted = True
                        raise StopIteration
                    break
        if event is _END:
            self._exhausted = True
            raise StopIteration
        if isinstance(event, BaseException):
            raise event
        return event

    def wait(self) -> None:
        # Repeated Ctrl+C must not detach a round from still-running native work.
        interrupted: KeyboardInterrupt | None = None
        while not self.finished.is_set():
            try:
                self.finished.wait()
            except KeyboardInterrupt as error:
                interrupted = error
        if interrupted is not None:
            raise interrupted


@dataclass(frozen=True)
class _Submission:
    job: Job
    events: _JobEvents


@dataclass(frozen=True)
class _Call(Generic[T]):
    fn: Callable[[EngineHandle, SessionHandle], T]
    future: Future[T]


class EngineWorker:
    def __init__(
        self,
        native: Native,
        config: EngineConfig,
        *,
        cancel: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
        checkpoint_budget_bytes: int = 1 << 30,
        throttle: float = 0.1,
    ) -> None:
        message = "checkpoint_budget_bytes must be a non-negative integer"
        if isinstance(checkpoint_budget_bytes, bool) or not isinstance(checkpoint_budget_bytes, int):
            raise TypeError(message)
        if checkpoint_budget_bytes < 0:
            raise ValueError(message)
        self._native = native
        self.cancel_event = cancel if cancel is not None else threading.Event()
        self._clock, self._throttle = clock, throttle
        self._checkpoint_budget_bytes = checkpoint_budget_bytes
        self._checkpoints: dict[str, tuple[tuple[int, ...], Snapshot]] = {}
        self._queue: SimpleQueue[Any] = SimpleQueue()
        self._lock = threading.Lock()
        self._closed = False
        ready: Future[EngineInfo] = Future()
        self._thread = threading.Thread(target=self._run, args=(config, ready), daemon=True)
        self._thread.start()
        try:
            self.info: EngineInfo = ready.result()
        except BaseException:
            # Startup may still succeed after the caller is interrupted. Queue
            # shutdown before joining so it cannot wait forever for another job.
            self.close()
            raise

    def _enqueue(self, item: object) -> None:
        # Serialize admission with the stop marker, including concurrent close().
        with self._lock:
            if self._closed:
                raise EngineError("engine worker is closed")
            self._queue.put(item)

    def submit(self, job: Job) -> _JobEvents:
        events = _JobEvents()
        self._enqueue(_Submission(job, events))
        return events

    def call(self, fn: Callable[[EngineHandle, SessionHandle], T]) -> Future[T]:
        future = Future()
        self._enqueue(_Call(fn, future))
        return future

    def plan(self, tokens: Sequence[int]) -> Plan:
        target = tuple(tokens)
        return self.call(lambda e, s: self._plan(s, target)).result()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(_STOP)
        self._thread.join()

    def _close_checkpoints(self) -> None:
        with ExitStack() as cleanup:
            for _, snapshot in self._checkpoints.values():
                cleanup.callback(snapshot.close)
            self._checkpoints.clear()

    def _run(self, config: EngineConfig, ready: Future[EngineInfo]) -> None:
        with ExitStack() as cleanup:
            try:
                engine = self._native.open_engine(config)
                cleanup.callback(engine.close)
                session = engine.create_session(config.ctx)
                cleanup.callback(session.close)
                cleanup.callback(self._close_checkpoints)
                ready.set_result(engine.info())
            except BaseException as error:
                ready.set_exception(error)
                return
            while True:
                item = self._queue.get()
                if item is _STOP:
                    return
                if isinstance(item, _Call):
                    if not item.future.set_running_or_notify_cancel():
                        continue
                    try:
                        result = item.fn(engine, session)
                    except BaseException as error:
                        item.future.set_exception(error)
                    else:
                        item.future.set_result(result)
                else:
                    try:
                        self._execute(engine, session, item.job, item.events.queue.put)
                    except BaseException as error:
                        item.events.queue.put(error)
                    finally:
                        item.events.queue.put(_END)
                        item.events.finished.set()

    def _plan(self, session: SessionHandle, tokens: Sequence[int]) -> Plan:
        return plan_sync(
            tokens,
            live=session.tokens(),
            live_valid=session.checkpoint_valid(),
            checkpoints={key: tokens for key, (tokens, _) in self._checkpoints.items()},
            ctx=session.ctx(),
            restore_enabled=self._native.restore_enabled,
        )

    def _execute(
        self,
        engine: EngineHandle,
        session: SessionHandle,
        job: Job,
        emit: Callable[[Event], None],
    ) -> None:
        if isinstance(job, SyncJob):
            self._sync(session, job.tokens, emit)
        elif isinstance(job, CheckpointJob):
            self._checkpoint(session, job, emit)
        elif isinstance(job, ResetJob):
            self._reset(session)
        elif isinstance(job, GenerateJob):
            reason = generate(engine, session, job, cancelled=self.cancel_event.is_set, emit=emit)
            if reason == "interrupted":
                self.cancel_event.clear()
            emit(Stopped(reason))
        elif isinstance(job, SavePayloadJob):
            self._save_payload(session, job, emit)
        elif isinstance(job, LoadPayloadJob):
            self._load_payload(session, job, emit)
        else:
            raise EngineError(f"unsupported engine job: {type(job).__name__}")

    def _sync(
        self,
        session: SessionHandle,
        tokens: Sequence[int],
        emit: Callable[[Event], None],
    ) -> SyncStatus:
        plan = self._plan(session, tokens)
        if not plan.fits:
            raise EngineError(f"prompt of {len(tokens)} tokens does not fit a context of {session.ctx()}")
        if plan.kind == "restore":
            assert plan.checkpoint is not None
            session.load_snapshot(self._checkpoints[plan.checkpoint][1])
        common = session.common_prefix(tokens)
        cached = common if common == session.pos() else 0
        prefilled = len(tokens) - cached
        start = self._clock()
        last_emit: float | None = None

        def progress(event: str, current: int, total: int) -> None:
            nonlocal last_emit
            if event == "prefill_work":
                done = current
            elif event in ("prefill_chunk", "prefill_display"):
                done = current - cached
            else:
                return
            now = self._clock()
            if last_emit is not None and now - last_emit < self._throttle:
                return
            done = max(0, min(done, prefilled))
            elapsed = now - start
            emit(Prefill(done, prefilled, cached, done / elapsed if elapsed > 0 else 0.0))
            last_emit = now

        status = session.sync(tokens, cancel=self.cancel_event.is_set, progress=progress)
        if status == SyncStatus.INTERRUPTED:
            self.cancel_event.clear()
            emit(Interrupted(session.pos()))
        else:
            emit(Synced(session.pos(), plan, prefilled))
        return status

    def _save_payload(
        self,
        session: SessionHandle,
        job: SavePayloadJob,
        emit: Callable[[Event], None],
    ) -> None:
        status = self._sync(session, job.tokens, emit)
        if status is not SyncStatus.DONE:
            raise EngineError("save was interrupted before staging the payload")
        staged = session.stage_payload()
        try:
            payload_bytes = staged.bytes
            start = job.file.tell()
            job.file.write(job.header)
            staged.write(job.file)
            written = job.file.tell() - start - len(job.header)
            if written != payload_bytes:
                raise EngineError(f"staged payload wrote {written} bytes, declared {payload_bytes}")
        finally:
            staged.close()
        emit(Saved(len(job.tokens), payload_bytes))

    def _load_payload(
        self,
        session: SessionHandle,
        job: LoadPayloadJob,
        emit: Callable[[Event], None],
    ) -> None:
        session.load_payload(job.file, job.payload_bytes)
        tokens = session.tokens()
        job.transcript.reset()
        if tokens:
            job.transcript.append(tokens, "system")
        job.transcript.set_chat()
        self._close_checkpoints()
        emit(Loaded(len(tokens), job.payload_bytes))

    def _reset(self, session: SessionHandle) -> None:
        try:
            status = session.sync(())
        except EmptyPromptError:
            # The native contract rejects an empty prompt. Rewind to the same
            # empty prefix; the next sync then rebuilds from position zero.
            session.invalidate()
            session.rewind(0)
        else:
            if status == SyncStatus.INTERRUPTED:
                self.cancel_event.clear()
                raise EngineError("session reset was interrupted")
        self._close_checkpoints()

    def _checkpoint(
        self,
        session: SessionHandle,
        job: CheckpointJob,
        emit: Callable[[Event], None],
    ) -> None:
        if job.drop:
            if job.key not in self._checkpoints:
                raise CheckpointError(f"unknown checkpoint {job.key!r}")
            _, snapshot = self._checkpoints.pop(job.key)
            snapshot.close()
            emit(Dropped(job.key))
            return
        budget = self._checkpoint_budget_bytes
        if budget == 0:
            raise CheckpointError("checkpoint creation is disabled (checkpoint_budget_bytes=0)")
        if not session.checkpoint_valid() or session.pos() <= 0:
            raise CheckpointError("no valid state to checkpoint")
        retained = sum(snapshot.nbytes for key, (_, snapshot) in self._checkpoints.items() if key != job.key)
        error = f"checkpoint_budget_bytes ({budget}) exceeded; drop a checkpoint first"
        if retained + session.payload_bytes() > budget:
            raise CheckpointError(error)
        tokens = tuple(session.tokens())
        snapshot = session.save_snapshot()
        if retained + snapshot.nbytes > budget:
            snapshot.close()
            raise CheckpointError(error)
        old = self._checkpoints.get(job.key)
        self._checkpoints[job.key] = (tokens, snapshot)
        if old is not None:
            old[1].close()
        emit(Checkpointed(job.key, len(tokens), snapshot.nbytes))
