import ctypes as C
import gc
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

from ds4py._native.api import LibNative, NativeSession
from ds4py._native.loader import load_library
from ds4py._native.protocol import EngineConfig
from ds4py._native.structs import (
    ArgmaxContext, EffLogits, Logits, TokenScore,
)
from ds4py._native.typemap import ctypes_signature, parse_prototype
from ds4py.engine.hooks import DisjointTokenRanges, LogitHookSet, TokenOverrides
from ds4py.engine.sampling import SamplingPolicy


def test_hook_struct_layouts_and_by_value_eff_signature():
    pointer = C.sizeof(C.c_void_p)
    assert C.sizeof(Logits) == pointer + 8
    assert (Logits.p.offset, Logits.cap.offset, Logits.len.offset) == (0, pointer, pointer + 4)
    assert C.sizeof(EffLogits) == 2 * pointer
    assert (EffLogits.p.offset, EffLogits.bytes.offset) == (0, pointer)
    assert C.sizeof(ArgmaxContext) == pointer + 8
    assert (ArgmaxContext.records.offset, ArgmaxContext.cap.offset,
            ArgmaxContext.collect.offset) == (0, pointer, pointer + 4)
    signature = ctypes_signature(parse_prototype(
        "ds4_eff_logits ds4_session_chain_consts(ds4_session *s, float t, int k, float m, float p)"
    ))
    assert signature[0] is EffLogits and signature[1][1:] == [C.c_float, C.c_int, C.c_float, C.c_float]


class FakeHookLibrary:
    def __init__(self):
        self.data = (C.c_float * 8)(*range(8))
        self.logits = Logits(C.cast(self.data, C.POINTER(C.c_float)), 8, 8)
        self.eff = EffLogits(C.pointer(self.logits), C.cast(self.data, C.POINTER(C.c_uint8)))
        self.candidates = (C.c_int * 3)(1, 2, 4)
        self.logit_fn = self.sampling_fn = self.argmax_fn = None
        self.context = None
        self.consts = []
        self.functions = {
            "ds4_session_chain_consts": self.chain_consts,
            "ds4_session_set_sampling_chain": self.set_sampling_chain,
            "ds4_session_set_logit_hook": self.set_logit_hook,
            "ds4_session_set_sampling_hook": self.set_sampling_hook,
            "ds4_session_set_argmax_fn": self.set_argmax_fn,
            "ds4_session_set_argmax_context": self.set_argmax_context,
            "ds4_session_sample": self.sample,
            "ds4_session_argmax": self.argmax,
            "ds4_session_free": lambda raw: None,
        }

    def fn(self, name):
        return self.functions[name]

    def chain_consts(self, raw, *values):
        self.consts.append(values)
        return self.eff

    def set_sampling_chain(self, raw, eff, ids, count, fn):
        self.chain = (eff, list(ids[:count]) if ids else [], fn)

    def set_logit_hook(self, raw, fn, ud): self.logit_fn = fn
    def set_sampling_hook(self, raw, fn, ud): self.sampling_fn = fn
    def set_argmax_fn(self, raw, fn, ud): self.argmax_fn = fn
    def set_argmax_context(self, raw, ctx): self.context = ctx

    def sample(self, raw, *args):
        if self.sampling_fn:
            self.sampling_fn(C.pointer(self.eff), self.candidates, len(self.candidates))
        return 7

    def argmax(self, raw):
        if self.argmax_fn and self.context:
            self.argmax_fn(self.context)
        return 7


def native_session():
    lib = FakeHookLibrary()
    engine = SimpleNamespace(_lib=lib, _sessions=set(), _session_allocations={1})
    return NativeSession(engine, 1), lib


def test_native_session_registers_clears_and_applies_sampling_dispatch():
    session, lib = native_session()
    hooks = LogitHookSet(
        allow=DisjointTokenRanges(((2, 3),)), overrides=TokenOverrides(((4, 8.0),)),
    )
    session.set_logit_hooks(hooks)
    assert lib.logit_fn and lib.sampling_fn and lib.argmax_fn
    hooks.before_sample(session, SamplingPolicy(), thinking=False)
    assert lib.consts == [(1.0, 0, 0.05, 1.0)]
    assert session.sample(1.0, 0, 1.0, 0.05, C.c_uint64(1)) == 7
    assert lib.data[4] == 8.0 and lib.data[0] == float("-inf")
    assert list(lib.candidates) == [-1, 2, 4]
    session.clear_logit_hooks()
    assert not lib.logit_fn and not lib.sampling_fn and not lib.argmax_fn
    assert not lib.context


def test_native_argmax_callback_copies_records():
    session, lib = native_session()
    hooks = LogitHookSet()
    session.set_logit_hooks(hooks)
    records = (TokenScore * 1)((7, 2.0, -0.5))
    session.set_argmax_context(ArgmaxContext(records, 1, 1))
    assert session.argmax() == 7 and hooks.argmax_records[0].id == 7
    records[0].id = 3
    assert hooks.argmax_records[0].id == 7
    session.clear_logit_hooks()


def test_native_callback_exception_reraises_after_native_return_and_recovers():
    class BadHooks:
        def logit(self, rctx, token): return 0
        def sampling(self, eff, candidates, count): raise RuntimeError("hook failed")
        def argmax(self, ctx): pass

    session, lib = native_session()
    bad = BadHooks()
    session.set_logit_hooks(bad)
    with pytest.raises(RuntimeError, match="hook failed"):
        session.sample(1.0, 0, 1.0, 0.05, C.c_uint64(1))
    session.set_logit_hooks(LogitHookSet())
    assert session.sample(1.0, 0, 1.0, 0.05, C.c_uint64(1)) == 7
    session.clear_logit_hooks()


def test_native_callbacks_hold_only_weak_hook_references():
    session, lib = native_session()
    hooks = LogitHookSet()
    session.set_logit_hooks(hooks)
    reference = weakref.ref(hooks)
    del hooks
    gc.collect()
    assert reference() is None
    assert session.sample(1.0, 0, 1.0, 0.05, C.c_uint64(1)) == 7
    session.clear_logit_hooks()


M2 = Path(__file__).parents[2] / "src/ds4py/_native/libds4_m2.so"
MODEL = os.environ.get("DS4_TEST_MODEL")
HOOK_FUNCTIONS = (
    "ds4_session_chain_consts", "ds4_session_set_sampling_chain",
    "ds4_session_set_logit_hook", "ds4_session_set_sampling_hook",
    "ds4_session_set_argmax_fn", "ds4_session_set_argmax_context",
)


@pytest.mark.skipif(not M2.exists() or not MODEL, reason="M2 library and DS4_TEST_MODEL are required")
def test_strict_m2_hook_seam_loads_and_runs_argmax():
    library = load_library(M2)
    assert all(library.has(name) for name in HOOK_FUNCTIONS)
    native = LibNative(library)
    engine = native.open_engine(EngineConfig(Path(MODEL), ctx=256))
    try:
        session = engine.create_session(256)
        hooks = LogitHookSet()
        session.set_logit_hooks(hooks)
        session.sync(engine.tokenize("Hello"))
        hooks.before_sample(session, SamplingPolicy(), thinking=False)
        assert session.argmax() >= 0
        session.clear_logit_hooks()
    finally:
        engine.close()
