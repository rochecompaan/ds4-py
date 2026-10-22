import threading
from pathlib import Path

import pytest

from ds4py._native.fake import FakeClock, FakeNative
from ds4py._native.protocol import EngineConfig
from ds4py.engine.events import Checkpointed, Dropped, Interrupted, Prefill, Synced
from ds4py.engine.plan import Plan
from ds4py.engine.worker import SYSTEM, CheckpointJob, EngineWorker, SyncJob
from ds4py.errors import CheckpointError, EngineError, ModelFileError


_workers = []


@pytest.fixture(autouse=True)
def close_workers():
    yield
    for w in _workers:
        w.close()
    _workers.clear()


P = list(range(1, 41))

def worker(*, ctx=64, clock=None, cancel=None, max_checkpoints=4, **fake):
    clock = clock or FakeClock()
    w = EngineWorker(FakeNative(clock=clock, **fake), EngineConfig(Path("f.gguf"), ctx=ctx),
                     clock=clock, cancel=cancel, max_checkpoints=max_checkpoints)
    _workers.append(w)
    return w

def run(w, job):
    return list(w.submit(job))

def sync_calls(w):
    return w.call(lambda e, s: [n for name, n in s.calls if name == "sync"]).result()

def test_each_sync_reports_the_plan_it_carried_out():
    w = worker()
    run(w, SyncJob(tuple(P[:8])))
    assert run(w, CheckpointJob("turn")) == [Checkpointed("turn", 8, 8 * 1024)]
    plans = []
    for tokens, kind in [(P[:20], "extend"), (P[:8] + [99, 98], "restore"), ([50, 51], "rebuild")]:
        predicted = w.plan(tokens)
        *_, synced = run(w, SyncJob(tuple(tokens)))
        assert predicted.kind == kind and synced == Synced(len(tokens), predicted, predicted.to_prefill)
        plans.append(predicted)
    assert sync_calls(w)[1:] == [p.to_prefill for p in plans] == [12, 2, 2]

def test_prefill_events_come_at_most_every_100ms():
    clock = FakeClock()
    w = worker(clock=clock, chunk=4, chunk_seconds=0.03)
    events = run(w, SyncJob(tuple(P)))
    prefills = [e for e in events if isinstance(e, Prefill)]
    assert [(e.done, e.total, e.cached_prefix) for e in prefills] == [(0, 40, 0), (16, 40, 0), (32, 40, 0)]
    assert prefills[1].tps == pytest.approx(16 / 0.12)
    assert events[-1] == Synced(40, Plan("rebuild", 0, 40, True, None), 40)

def test_prefill_counts_only_the_new_tokens():
    w = worker(chunk=4)
    run(w, SyncJob(tuple(P[:6])))
    assert run(w, SyncJob(tuple(P)))[0] == Prefill(0, 34, 6, 0.0)

def test_cancel_interrupts_sync_and_clears_the_flag():
    cancel = threading.Event()
    w = worker(cancel=cancel)
    run(w, SyncJob(tuple(P[:6])))
    cancel.set()                                                    # Ctrl+C
    assert run(w, SyncJob(tuple(P)))[-1] == Interrupted(6)
    assert w.cancel_event is cancel and not cancel.is_set()
    assert w.plan(P) == Plan("extend", 6, 34, True, None)          # the prefix done so far is kept

def test_checkpoint_limits():
    w = worker(max_checkpoints=2)
    with pytest.raises(CheckpointError, match="no valid state"): run(w, CheckpointJob("a"))
    run(w, SyncJob(tuple(P[:4])))
    for key in (SYSTEM, "a", "b", "a"):                             # system is not counted; "a" is replaced
        run(w, CheckpointJob(key))
    with pytest.raises(CheckpointError, match="max_checkpoints"): run(w, CheckpointJob("c"))
    assert run(w, CheckpointJob("a", drop=True)) == [Dropped("a")]
    run(w, CheckpointJob("c"))
    with pytest.raises(CheckpointError, match="unknown checkpoint"): run(w, CheckpointJob("zz", drop=True))

def test_a_checkpoint_is_kept_until_its_replacement_is_saved():
    w = worker()
    saved = []
    def wrap_save(e, s):                                            # the third save runs out of memory
        real = s.save_snapshot
        def save():
            if len(saved) == 2: raise EngineError("out of memory")
            saved.append(real())
            return saved[-1]
        s.save_snapshot = save
    w.call(wrap_save).result()
    for n in (8, 12):
        run(w, SyncJob(tuple(P[:n]))); run(w, CheckpointJob("a"))
    assert [snap.closed for snap in saved] == [True, False]         # the replaced snapshot is closed
    run(w, SyncJob(tuple(P[:16])))
    with pytest.raises(EngineError, match="out of memory"): run(w, CheckpointJob("a"))
    *_, synced = run(w, SyncJob(tuple(P[:12] + [99])))
    assert synced.plan == Plan("restore", 12, 1, True, "a")         # closing first would fail this restore

def test_sync_refuses_a_prompt_that_does_not_fit():
    w = worker(ctx=8)
    with pytest.raises(EngineError, match="8 tokens does not fit a context of 8"):
        run(w, SyncJob(tuple(P[:8])))
    assert sync_calls(w) == []

def test_engine_calls_run_on_one_worker_thread():
    w = worker()
    ids = {w.call(lambda e, s: threading.get_ident()).result() for _ in range(3)}
    assert len(ids) == 1 and threading.get_ident() not in ids
    assert isinstance(w.call(lambda e, s: 1 / 0).exception(), ZeroDivisionError)
    assert w.call(lambda e, s: s.pos()).result() == 0               # the worker survives a failed call
    assert w.info.name == "fake"

def test_open_errors_reach_the_caller():
    class Broken(FakeNative):
        def open_engine(self, config): raise ModelFileError("no model")
    with pytest.raises(ModelFileError): EngineWorker(Broken(), EngineConfig(Path("f")))

def test_close():
    w = worker(); w.close(); w.close()
    with pytest.raises(EngineError, match="closed"): w.submit(SyncJob((1,)))


def test_submit_enqueues_before_iteration_and_plan_waits_for_it():
    w = worker()
    events = w.submit(SyncJob((1, 2, 3)))
    assert w.plan([1, 2, 3, 4]) == Plan("extend", 3, 1, True, None)
    assert list(events)[-1] == Synced(3, Plan("rebuild", 0, 3, True, None), 3)


def test_progress_work_is_relative_and_chunk_and_display_are_absolute():
    class TickClock(FakeClock):
        def __call__(self):
            self.advance(0.2)
            return super().__call__()
    w = worker(clock=TickClock())
    run(w, SyncJob(tuple(P[:6])))

    def install(e, s):
        real = s.sync
        def sync(tokens, *, progress, cancel):
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
        (0, 34, 6), (4, 34, 6), (8, 34, 6), (34, 34, 6),
    ]


def test_close_drains_jobs_then_frees_snapshots_session_and_engine_on_owner():
    log = []
    class TrackingNative(FakeNative):
        def open_engine(self, config):
            e = super().open_engine(config)
            create, engine_close = e.create_session, e.close
            def create_session(ctx):
                s = create(ctx)
                save, session_close = s.save_snapshot, s.close
                def save_snapshot():
                    snap = save()
                    close = snap.close
                    def close_snapshot():
                        log.append(("snapshot", threading.get_ident()))
                        close()
                    snap.close = close_snapshot
                    return snap
                def close_session():
                    log.append(("session", threading.get_ident()))
                    session_close()
                s.save_snapshot, s.close = save_snapshot, close_session
                return s
            def close_engine():
                log.append(("engine", threading.get_ident()))
                engine_close()
            e.create_session, e.close = create_session, close_engine
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


def test_session_open_failure_closes_engine_on_opening_thread():
    log = []
    class Broken(FakeNative):
        def open_engine(self, config):
            e = super().open_engine(config)
            def create(ctx):
                log.append(("open", threading.get_ident()))
                raise EngineError("session failed")
            real_close = e.close
            def close():
                log.append(("close", threading.get_ident()))
                real_close()
            e.create_session, e.close = create, close
            return e
    with pytest.raises(EngineError, match="session failed"):
        EngineWorker(Broken(), EngineConfig(Path("f")))
    assert len(log) == 2 and log[0][0] == "open" and log[1] == ("close", log[0][1])
    assert log[0][1] != threading.get_ident()
