from typing import Any
from pathlib import Path

import pytest

from ds4py._native.fake import FakeNative, FakeClock, FakeSession
from ds4py._native.protocol import EngineConfig
from ds4py.engine.worker import EngineWorker
from ds4py.errors import EngineError
from ds4py.abi.check import run_live_checks


def live(script: Any=lambda prompt, generated: 98, clock: Any=None, **fake: Any) -> Any:    # 98 is "a"; the run never stops early
    clock = clock or FakeClock()
    w = EngineWorker(FakeNative(script=script, clock=clock, chunk=4, chunk_seconds=0.01, **fake),
                     EngineConfig(Path("f.gguf"), ctx=512), clock=clock)
    try:
        return run_live_checks(w, clock=clock, prefix=128, tail=16, generate=32)
    finally:
        w.close()

def test_the_checks_pass_on_an_engine_that_reuses_its_kv() -> None:
    r = live()
    assert r["sync_plans"]["ok"] and r["restore"] == {
        "ok": True, "message": "32 greedy tokens after a restore match a rebuild", "tokens": 32}
    assert r["sync_plans"]["prefilled"] == {"extend": 16, "restore": 16, "rebuild": 128}
    assert r["sync_plans"]["seconds"] == pytest.approx({"extend": 0.04, "restore": 0.04, "rebuild": 0.32})
    assert r["sync_plans"]["message"] == ("extend of 16 tokens took 0.040 s, restore of 16 took 0.040 s, "
                                          "rebuild of 128 took 0.320 s")

def test_a_restore_that_changes_the_output_fails_the_restore_check() -> None:
    starts = []
    def drifting(prompt: Any, generated: Any) -> Any:                    # each run answers with a different token
        if not generated:
            starts.append(None)
        return 2 + len(starts)
    r = live(script=drifting)
    assert r["sync_plans"]["ok"] and not r["restore"]["ok"]
    assert r["restore"]["message"] == "the runs differ at token 0: 3 after a restore, 4 after a rebuild"

def test_a_snapshot_load_that_does_not_restore_fails_the_sync_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeSession, "load_snapshot", lambda self, snap: None)
    r = live()
    assert not r["sync_plans"]["ok"]
    assert r["sync_plans"]["message"] == "the restore sync prefilled 128 tokens; its plan prefills 16"

def test_a_slow_restore_is_reported_but_not_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    load = FakeSession.load_snapshot
    def slow_load(self: Any, snap: Any) -> None:                          # loading the snapshot takes a second
        clock.advance(1.0)
        load(self, snap)
    monkeypatch.setattr(FakeSession, "load_snapshot", slow_load)
    r = live(clock=clock)
    assert r["sync_plans"]["ok"] and r["sync_plans"]["prefilled"]["restore"] == 16
    assert r["sync_plans"]["seconds"]["restore"] == pytest.approx(1.04)

def test_the_context_must_hold_the_check() -> None:
    clock = FakeClock()
    w = EngineWorker(FakeNative(clock=clock), EngineConfig(Path("f.gguf"), ctx=64), clock=clock)
    try:
        with pytest.raises(EngineError, match="needs a context of at least 161"):
            run_live_checks(w, clock=clock, prefix=128, tail=16, generate=32)
    finally:
        w.close()


def test_disabled_restore_is_a_plan_failure() -> None:
    r = live(restore_enabled=False)
    assert not r['sync_plans']['ok']
    assert r['sync_plans']['message'] == (
        'the restore sync planned rebuild of 128 tokens, expected restore of 16')


def test_different_stop_points_report_the_end_reason() -> None:
    starts = []
    def early_stop(prompt: Any, generated: Any) -> int:
        if not generated:
            starts.append(None)
        return 0 if len(starts) == 1 else 98
    r = live(script=early_stop)
    assert not r['restore']['ok']
    assert r['restore']['message'] == (
        'the runs differ at token 0: end (stop_token) after a restore, 98 after a rebuild')
