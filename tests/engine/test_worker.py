import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Never, cast, override

import pytest

from ds4py._native.fake import FakeClock, FakeNative
from ds4py._native.protocol import EngineConfig
from ds4py.engine.events import Checkpointed, Dropped, Interrupted, Prefill, Synced
from ds4py.engine.plan import Plan
from ds4py.engine.worker import SYSTEM, CheckpointJob, EngineWorker, SyncJob
from ds4py.errors import CheckpointError, EngineError, ModelFileError

_workers = []


@pytest.fixture(autouse=True)
def close_workers() -> Iterator[Any]:
    yield
    for w in _workers:
        w.close()
    _workers.clear()


P = list(range(1, 41))


def worker(
    *, ctx: Any = 64, clock: Any = None, cancel: Any = None, checkpoint_budget_bytes: Any = None, **fake: Any
) -> Any:
    clock = clock or FakeClock()
    limits = {} if checkpoint_budget_bytes is None else {"checkpoint_budget_bytes": checkpoint_budget_bytes}
    w = EngineWorker(
        FakeNative(clock=clock, **fake), EngineConfig(Path("f.gguf"), ctx=ctx), clock=clock, cancel=cancel, **limits
    )
    _workers.append(w)
    return w


def run(w: Any, job: Any) -> list[Any]:
    return list(w.submit(job))


def sync_calls(w: Any) -> list[Any]:
    return w.call(lambda e, s: [n for name, n in s.calls if name == "sync"]).result()


def test_each_sync_reports_the_plan_it_carried_out() -> None:
    w = worker()
    run(w, SyncJob(tuple(P[:8])))
    assert run(w, CheckpointJob("turn")) == [Checkpointed("turn", 8, 8 * 1024)]
    plans = []
    for tokens, kind in [(P[:20], "extend"), ([*P[:8], 99, 98], "restore"), ([50, 51], "rebuild")]:
        predicted = w.plan(tokens)
        *_, synced = run(w, SyncJob(tuple(tokens)))
        assert predicted.kind == kind and synced == Synced(len(tokens), predicted, predicted.to_prefill)
        plans.append(predicted)
    assert sync_calls(w)[1:] == [p.to_prefill for p in plans] == [12, 2, 2]


def test_prefill_events_come_at_most_every_100ms() -> None:
    clock = FakeClock()
    w = worker(clock=clock, chunk=4, chunk_seconds=0.03)
    events = run(w, SyncJob(tuple(P)))
    prefills = [e for e in events if isinstance(e, Prefill)]
    assert [(e.done, e.total, e.cached_prefix) for e in prefills] == [(0, 40, 0), (16, 40, 0), (32, 40, 0)]
    assert prefills[1].tps == pytest.approx(16 / 0.12)
    assert events[-1] == Synced(40, Plan("rebuild", 0, 40, True, None), 40)


def test_prefill_counts_only_the_new_tokens() -> None:
    w = worker(chunk=4)
    run(w, SyncJob(tuple(P[:6])))
    assert run(w, SyncJob(tuple(P)))[0] == Prefill(0, 34, 6, 0.0)


def test_cancel_interrupts_sync_and_clears_the_flag() -> None:
    cancel = threading.Event()
    w = worker(cancel=cancel)
    run(w, SyncJob(tuple(P[:6])))
    cancel.set()  # Ctrl+C
    assert run(w, SyncJob(tuple(P)))[-1] == Interrupted(6)
    assert w.cancel_event is cancel and not cancel.is_set()
    assert w.plan(P) == Plan("extend", 6, 34, True, None)  # the prefix done so far is kept


# Checkpoint payload admission and ownership.
def track_snapshots(w: Any) -> Any:
    saved, closed = [], []

    def install(e: Any, s: Any) -> None:
        original_save = s.save_snapshot

        def save() -> Any:
            snapshot = original_save()
            original_close = snapshot.close

            def close() -> None:
                closed.append((snapshot, threading.get_ident()))
                original_close()

            snapshot.close = close
            saved.append(snapshot)
            return snapshot

        s.save_snapshot = save

    w.call(install).result()
    return saved, closed


def test_checkpoint_budget_counts_system_and_admits_exact_fit() -> None:
    w = worker(checkpoint_budget_bytes=8 * 1024)
    saved, closed = track_snapshots(w)
    owner = w.call(lambda e, s: threading.get_ident()).result()
    with pytest.raises(CheckpointError, match="no valid state"):
        run(w, CheckpointJob("a"))
    run(w, SyncJob(tuple(P[:4])))
    for key in (SYSTEM, "a", "a"):  # replacing a full-budget snapshot must work
        assert run(w, CheckpointJob(key)) == [Checkpointed(key, 4, 4096)]
    assert [s.closed for s in saved] == [False, True, False]
    with pytest.raises(CheckpointError, match="checkpoint_budget_bytes"):
        run(w, CheckpointJob("b"))
    assert len(saved) == 3  # refused before allocating, without evicting a key
    assert run(w, CheckpointJob("a", drop=True)) == [Dropped("a")]
    run(w, CheckpointJob("b"))
    run(w, CheckpointJob(SYSTEM, drop=True))
    w.close()
    assert all(s.closed for s in saved)
    assert len(closed) == 4 and all(thread == owner for _, thread in closed)


def test_checkpoint_drop_rejects_an_unknown_key() -> None:
    with pytest.raises(CheckpointError, match="unknown checkpoint"):
        run(worker(), CheckpointJob("zz", drop=True))


def test_small_snapshots_are_not_limited_by_count() -> None:
    w = worker()
    run(w, SyncJob((1,)))
    for key in (SYSTEM, "a", "b", "c", "d", "e"):
        run(w, CheckpointJob(key))
    for key in (SYSTEM, "a", "b", "c", "d", "e"):
        assert run(w, CheckpointJob(key, drop=True)) == [Dropped(key)]


def test_over_budget_replacement_keeps_old_snapshot_and_other_keys() -> None:
    w = worker(checkpoint_budget_bytes=6 * 1024)
    saved, _ = track_snapshots(w)
    run(w, SyncJob(tuple(P[:2])))
    run(w, CheckpointJob(SYSTEM))
    run(w, SyncJob(tuple(P[:4])))
    run(w, CheckpointJob("a"))
    run(w, SyncJob(tuple(P[:5])))
    with pytest.raises(CheckpointError, match="checkpoint_budget_bytes"):
        run(w, CheckpointJob("a"))
    assert len(saved) == 2 and not any(s.closed for s in saved)
    *_, synced = run(w, SyncJob((*P[:4], 99)))
    assert synced.plan == Plan("restore", 4, 1, True, "a")
    run(w, CheckpointJob(SYSTEM, drop=True))
    run(w, CheckpointJob("a"))  # enough room now; the old a is freed only on success
    assert [s.closed for s in saved] == [True, True, False]


@pytest.mark.parametrize("replacement", [False, True])
def test_oversized_returned_snapshot_is_freed_without_losing_retained_state(replacement: Any) -> None:
    w = worker(checkpoint_budget_bytes=6 * 1024)
    saved, closed = track_snapshots(w)
    owner = w.call(lambda e, s: threading.get_ident()).result()
    run(w, SyncJob(tuple(P[:2])))
    run(w, CheckpointJob(SYSTEM))
    if replacement:
        run(w, SyncJob(tuple(P[:3])))
        run(w, CheckpointJob("a"))
    run(w, SyncJob(tuple(P[:4])))

    def underestimate(e: Any, s: Any) -> None:
        original_save = s.save_snapshot

        def save() -> Any:
            snap = original_save()
            snap.nbytes = 4097  # payload_bytes preflight was 4096
            return snap

        s.save_snapshot = save

    w.call(underestimate).result()
    with pytest.raises(CheckpointError, match="checkpoint_budget_bytes"):
        run(w, CheckpointJob("a"))
    assert saved[-1].closed and not any(s.closed for s in saved[:-1])
    assert closed == [(saved[-1], owner)]
    prefix = 3 if replacement else 2
    *_, synced = run(w, SyncJob((*P[:prefix], 99)))
    assert synced.plan == Plan("restore", prefix, 1, True, "a" if replacement else SYSTEM)
    w.close()
    assert all(s.closed for s in saved) and len(closed) == len(saved)


def test_zero_budget_disables_even_zero_byte_checkpoints() -> None:
    w = worker(checkpoint_budget_bytes=0)
    with pytest.raises(CheckpointError, match="checkpoint_budget_bytes"):
        run(w, CheckpointJob(SYSTEM))
    run(w, SyncJob((1,)))

    def empty_payload(e: Any, s: Any) -> None:
        s.payload_bytes = lambda: 0
        s.save_snapshot = lambda: pytest.fail("zero budget allocated a snapshot")

    w.call(empty_payload).result()
    with pytest.raises(CheckpointError, match="checkpoint_budget_bytes"):
        run(w, CheckpointJob(SYSTEM))


@pytest.mark.parametrize(
    ("budget", "error"),
    [
        (-1, ValueError),
        (1.5, TypeError),
        ("1024", TypeError),
        (True, TypeError),
        (False, TypeError),
        (None, TypeError),
    ],
)
def test_bad_checkpoint_budget_is_refused_before_native_open(budget: Any, error: Any) -> None:
    class NeverOpened(FakeNative):
        @override
        def open_engine(self, config: Any) -> Any:
            pytest.fail("invalid budget opened a native engine")

    with pytest.raises(error, match="checkpoint_budget_bytes must be a non-negative integer"):
        EngineWorker(cast("Any", NeverOpened()), EngineConfig(Path("f")), checkpoint_budget_bytes=budget)


def test_default_budget_admits_one_gib_and_rejects_one_more_byte() -> None:
    w = worker()  # use the production default, not a test helper's copy of it
    run(w, SyncJob((1,)))

    def large_payload(e: Any, s: Any) -> None:
        original_save = s.save_snapshot

        def save() -> Any:
            snap = original_save()
            snap.nbytes = 1 << 30  # report a large payload without allocating a GiB
            return snap

        s.payload_bytes = lambda: 1 << 30
        s.save_snapshot = save

    w.call(large_payload).result()
    assert run(w, CheckpointJob(SYSTEM)) == [Checkpointed(SYSTEM, 1, 1 << 30)]
    run(w, CheckpointJob(SYSTEM, drop=True))
    w.call(lambda e, s: setattr(s, "payload_bytes", lambda: (1 << 30) + 1)).result()
    with pytest.raises(CheckpointError, match="checkpoint_budget_bytes"):
        run(w, CheckpointJob("too_large"))


def test_a_checkpoint_is_kept_until_its_replacement_is_saved() -> None:
    w = worker(checkpoint_budget_bytes=16 * 1024)
    saved = []

    def wrap_save(e: Any, s: Any) -> None:  # the third save runs out of memory
        real = s.save_snapshot

        def save() -> Any:
            if len(saved) == 2:
                raise EngineError("out of memory")
            saved.append(real())
            return saved[-1]

        s.save_snapshot = save

    w.call(wrap_save).result()
    for n in (8, 12):
        run(w, SyncJob(tuple(P[:n])))
        run(w, CheckpointJob("a"))
    assert [snap.closed for snap in saved] == [True, False]  # the replaced snapshot is closed
    run(w, SyncJob(tuple(P[:16])))
    with pytest.raises(EngineError, match="out of memory"):
        run(w, CheckpointJob("a"))
    *_, synced = run(w, SyncJob((*P[:12], 99)))
    assert synced.plan == Plan("restore", 12, 1, True, "a")  # closing first would fail this restore


def test_sync_refuses_a_prompt_that_does_not_fit() -> None:
    w = worker(ctx=8)
    with pytest.raises(EngineError, match="8 tokens does not fit a context of 8"):
        run(w, SyncJob(tuple(P[:8])))
    assert sync_calls(w) == []


def test_engine_calls_run_on_one_worker_thread() -> None:
    w = worker()
    ids = {w.call(lambda e, s: threading.get_ident()).result() for _ in range(3)}
    assert len(ids) == 1 and threading.get_ident() not in ids
    assert isinstance(w.call(lambda e, s: 1 / 0).exception(), ZeroDivisionError)
    assert w.call(lambda e, s: s.pos()).result() == 0  # the worker survives a failed call
    assert w.info.name == "fake"


def test_open_errors_reach_the_caller() -> None:
    class Broken(FakeNative):
        @override
        def open_engine(self, config: Any) -> Never:
            raise ModelFileError("no model")

    with pytest.raises(ModelFileError):
        EngineWorker(Broken(), EngineConfig(Path("f")))


def test_completion_racing_empty_receive_preserves_queued_events_and_errors() -> None:
    from queue import Empty, SimpleQueue

    from ds4py.engine.events import Event
    from ds4py.engine.worker import _END, _JobEvents

    events = _JobEvents()
    failure = EngineError("queued native error")
    synced = Synced(2, Plan("extend", 2, 0, True, None), 0)

    class CompletingQueue(SimpleQueue[Event | BaseException | _END]):
        @override
        def get(self, *args: Any, **kwargs: Any) -> Any:
            try:
                return super().get_nowait()
            except Empty:
                # The producer publishes after a receive times out, but before
                # the consumer observes completion. Those items still matter.
                self.put(synced)
                self.put(failure)
                self.put(_END)
                events.finished.set()
                raise

    events.queue = CompletingQueue()
    assert next(events) == synced
    with pytest.raises(EngineError, match="queued native error"):
        next(events)
    with pytest.raises(StopIteration):
        next(events)


def test_close() -> None:
    w = worker()
    w.close()
    w.close()
    with pytest.raises(EngineError, match="closed"):
        w.submit(SyncJob((1,)))


def test_submit_enqueues_before_iteration_and_plan_waits_for_it() -> None:
    w = worker()
    events = w.submit(SyncJob((1, 2, 3)))
    assert w.plan([1, 2, 3, 4]) == Plan("extend", 3, 1, True, None)
    assert list(events)[-1] == Synced(3, Plan("rebuild", 0, 3, True, None), 3)


def test_progress_work_is_relative_and_chunk_and_display_are_absolute() -> None:
    class TickClock(FakeClock):
        @override
        def __call__(self) -> Any:
            self.advance(0.2)
            return super().__call__()

    w = worker(clock=TickClock())
    run(w, SyncJob(tuple(P[:6])))

    def install(e: Any, s: Any) -> None:
        real = s.sync

        def sync(tokens: Any, *, progress: Any, cancel: Any) -> Any:
            progress("ignored", 999, 40)
            progress("prefill_work", -5, 34)
            progress("prefill_work", 4, 34)
            progress("prefill_chunk", 14, 40)
            progress("prefill_display", 999, 40)
            return real(tokens, cancel=cancel)

        s.sync = sync

    w.call(install).result()
    events = run(w, SyncJob(tuple(P)))
    assert [(e.done, e.total, e.cached_prefix) for e in events[:-1]] == [
        (0, 34, 6),
        (4, 34, 6),
        (8, 34, 6),
        (34, 34, 6),
    ]


def test_close_drains_jobs_then_frees_snapshots_session_and_engine_on_owner() -> None:
    log = []

    class TrackingNative(FakeNative):
        @override
        def open_engine(self, config: Any) -> Any:
            e = super().open_engine(config)
            create, engine_close = e.create_session, e.close

            def create_session(ctx: Any) -> Any:
                s = create(ctx)
                save, session_close = s.save_snapshot, s.close

                def save_snapshot() -> Any:
                    snap = save()
                    close = snap.close

                    def close_snapshot() -> None:
                        log.append(("snapshot", threading.get_ident()))
                        close()

                    snap.close = close_snapshot  # ty: ignore[invalid-assignment]  # Instance-level test double replaces a bound method.
                    return snap

                def close_session() -> None:
                    log.append(("session", threading.get_ident()))
                    session_close()

                s.save_snapshot, s.close = save_snapshot, close_session  # ty: ignore[invalid-assignment]  # Instance-level test double replaces bound methods.
                return s

            def close_engine() -> None:
                log.append(("engine", threading.get_ident()))
                engine_close()

            e.create_session, e.close = create_session, close_engine  # ty: ignore[invalid-assignment]  # Instance-level test double replaces bound methods.
            return e

    w = EngineWorker(TrackingNative(), EngineConfig(Path("f")))
    _workers.append(w)
    run(w, SyncJob((1, 2)))
    owner = w.call(lambda e, s: threading.get_ident()).result()
    pending = w.submit(CheckpointJob("a"))
    w.close()
    assert list(pending) == [Checkpointed("a", 2, 2048)]
    assert log == [(name, owner) for name in ("snapshot", "session", "engine")]
    with pytest.raises(EngineError, match="closed"):
        w.call(lambda e, s: None)


def test_session_open_failure_closes_engine_on_opening_thread() -> None:
    log = []

    class Broken(FakeNative):
        @override
        def open_engine(self, config: Any) -> Any:
            e = super().open_engine(config)

            def create(ctx: Any) -> Never:
                log.append(("open", threading.get_ident()))
                raise EngineError("session failed")

            real_close = e.close

            def close() -> None:
                log.append(("close", threading.get_ident()))
                real_close()

            e.create_session, e.close = create, close  # ty: ignore[invalid-assignment]  # Instance-level test double replaces bound methods.
            return e

    with pytest.raises(EngineError, match="session failed"):
        EngineWorker(Broken(), EngineConfig(Path("f")))
    assert len(log) == 2 and log[0][0] == "open" and log[1] == ("close", log[0][1])
    assert log[0][1] != threading.get_ident()
