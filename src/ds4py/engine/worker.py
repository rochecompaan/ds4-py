"""Serial job execution and native resource ownership on one daemon thread."""
import threading
import time
from concurrent.futures import Future
from contextlib import ExitStack
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Callable, Iterator, Sequence, TypeVar

from ds4py._native.protocol import (
    EngineConfig, EngineHandle, EngineInfo, Native, SessionHandle, Snapshot, SyncStatus,
)
from ds4py.errors import CheckpointError, EngineError
from .decode import GenerateJob, generate
from .events import Checkpointed, Dropped, Event, Interrupted, Prefill, Stopped, Synced
from .plan import Plan, plan_sync

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
class Handles:
    engine: int
    session: int


class _JobEvents(Iterator[Event]):
    """A resumable event channel with an independent native-completion barrier."""

    def __init__(self):
        self.queue = SimpleQueue()
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
        interrupted = None
        while not self.finished.is_set():
            try:
                self.finished.wait()
            except KeyboardInterrupt as error:
                interrupted = error
        if interrupted is not None:
            raise interrupted


@dataclass(frozen=True)
class _Submission:
    job: SyncJob | CheckpointJob | GenerateJob
    events: _JobEvents


@dataclass(frozen=True)
class _Call:
    fn: Callable
    future: Future


class EngineWorker:
    def __init__(self, native: Native, config: EngineConfig, *,
                 cancel: threading.Event | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 max_checkpoints: int = 4, throttle: float = 0.1):
        self._native = native
        self.cancel_event = cancel if cancel is not None else threading.Event()
        self._clock, self._throttle = clock, throttle
        self._max_checkpoints = max_checkpoints
        self._checkpoints: dict[str, tuple[tuple[int, ...], Snapshot]] = {}
        self._queue = SimpleQueue()
        self._lock = threading.Lock()
        self._closed = False
        ready = Future()
        self._thread = threading.Thread(target=self._run, args=(config, ready), daemon=True)
        self._thread.start()
        try:
            self.info: EngineInfo = ready.result()
        except BaseException:
            self._thread.join()
            raise

    def _enqueue(self, item):
        # Serialize admission with the stop marker, including concurrent close().
        with self._lock:
            if self._closed:
                raise EngineError("engine worker is closed")
            self._queue.put(item)

    def submit(self, job: SyncJob | CheckpointJob | GenerateJob) -> _JobEvents:
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

    def _close_checkpoints(self):
        with ExitStack() as cleanup:
            for _, snapshot in self._checkpoints.values():
                cleanup.callback(snapshot.close)
            self._checkpoints.clear()

    def _run(self, config, ready):
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

    def _plan(self, session, tokens) -> Plan:
        return plan_sync(
            tokens, live=session.tokens(), live_valid=session.checkpoint_valid(),
            checkpoints={key: tokens for key, (tokens, _) in self._checkpoints.items()},
            ctx=session.ctx(), restore_enabled=self._native.restore_enabled,
        )

    def _execute(self, engine, session, job, emit):
        if isinstance(job, SyncJob):
            self._sync(session, job.tokens, emit)
        elif isinstance(job, CheckpointJob):
            self._checkpoint(session, job, emit)
        elif isinstance(job, GenerateJob):
            reason = generate(engine, session, job,
                              cancelled=self.cancel_event.is_set, emit=emit)
            if reason == "interrupted":
                self.cancel_event.clear()
            emit(Stopped(reason))
        else:
            raise EngineError(f"unsupported engine job: {type(job).__name__}")

    def _sync(self, session, tokens, emit):
        plan = self._plan(session, tokens)
        if not plan.fits:
            raise EngineError(
                f"prompt of {len(tokens)} tokens does not fit a context of {session.ctx()}"
            )
        if plan.kind == "restore":
            session.load_snapshot(self._checkpoints[plan.checkpoint][1])
        common = session.common_prefix(tokens)
        cached = common if common == session.pos() else 0
        prefilled = len(tokens) - cached
        start = self._clock()
        last_emit = None

        def progress(event, current, total):
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

    def _checkpoint(self, session, job, emit):
        if job.drop:
            if job.key not in self._checkpoints:
                raise CheckpointError(f"unknown checkpoint {job.key!r}")
            _, snapshot = self._checkpoints.pop(job.key)
            snapshot.close()
            emit(Dropped(job.key))
            return
        if not session.checkpoint_valid() or session.pos() <= 0:
            raise CheckpointError("no valid state to checkpoint")
        count = sum(key != SYSTEM for key in self._checkpoints)
        if job.key != SYSTEM and job.key not in self._checkpoints and count >= self._max_checkpoints:
            raise CheckpointError(f"max_checkpoints ({self._max_checkpoints}) reached; drop one first")
        tokens = tuple(session.tokens())
        snapshot = session.save_snapshot()
        old = self._checkpoints.get(job.key)
        self._checkpoints[job.key] = (tokens, snapshot)
        if old is not None:
            old[1].close()
        emit(Checkpointed(job.key, len(tokens), snapshot.nbytes))
