import ctypes
from dataclasses import replace
from pathlib import Path

from ds4py._native.fake import FakeNative
from ds4py._native.protocol import EngineConfig
from ds4py.engine.sampling import Rng, SamplingPolicy

INFO = FakeNative().open_engine(EngineConfig(Path("f"))).info()


def test_glm_defaults_unless_set():
    glm = replace(INFO, family="glm", is_glm_dsa=True)
    assert SamplingPolicy.for_engine(glm) == SamplingPolicy(top_p=0.95, min_p=0.0)
    assert SamplingPolicy.for_engine(glm, min_p=0.1).min_p == 0.1
    assert SamplingPolicy.for_engine(INFO, temperature=0.0) == SamplingPolicy(temperature=0.0)
    assert SamplingPolicy.for_engine(glm, temperature=0.5, top_p=0.8, min_p=0.2, top_k=4, seed=0) == SamplingPolicy(0.5, 0.8, 0.2, 4, 0)


def test_rng_seed():
    assert Rng(7).state.value == 7
    assert Rng(None).state.value != 0 and Rng(0).state.value != 0
    rng = Rng(7)
    assert isinstance(rng.state, ctypes.c_uint64)
    ctypes.pointer(rng.state).contents.value = 8
    assert rng.state.value == 8


def test_rng_matches_native_seed_formula(monkeypatch):
    monkeypatch.setattr("time.time", lambda: 100.75)
    monkeypatch.setattr("os.getpid", lambda: 2)
    monkeypatch.setattr("time.process_time_ns", lambda: 3)
    assert Rng(None).state.value == 8589934695
    monkeypatch.setattr("time.time", lambda: 0)
    monkeypatch.setattr("os.getpid", lambda: 0)
    monkeypatch.setattr("time.process_time_ns", lambda: 0)
    assert Rng(0).state.value == 1
    assert Rng(-1).state.value == 18446744073709551615
