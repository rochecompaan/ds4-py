import fcntl
import hashlib
import os
import re
from pathlib import Path
import pytest
from ds4py.errors import AbiMismatch, EngineError, EngineLockedError, ModelFileError
from ds4py._native.api import InstanceLock, LibNative, lock_path, engine_lock_path, engine_options
from ds4py._native.protocol import EngineConfig, ThinkMode
from ds4py._native.modelcheck import exit_error, NATIVE_REFUSED, PYTHON_ERROR
from ds4py._native.loader import load_library
from contract import CONTRACT, check_snapshot_restore_matches_rebuild

def test_instance_lock(tmp_path, monkeypatch):
    monkeypatch.delenv("DS4_LOCK_FILE", raising=False)
    assert lock_path() == Path("/tmp/ds4.lock")
    path = tmp_path / "ds4.lock"
    monkeypatch.setenv("DS4_LOCK_FILE", str(path))
    assert (lock_path(), engine_lock_path()) == (path, tmp_path / "ds4.lock.engine")
    with open(path, "w") as held:
        held.write("4242"); held.flush()
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(EngineLockedError, match=rf"{re.escape(str(path))}.*pid 4242"):
            InstanceLock.acquire(path)
    with InstanceLock.acquire(path):
        assert path.read_text() == f"{os.getpid()}\n"      # a C tool's refusal names this process
        with pytest.raises(EngineLockedError):              # a second descriptor conflicts, as ds4's would
            InstanceLock.acquire(path)
    InstanceLock.acquire(path).close()                      # closing released it

def test_exit_error():
    m, lk = Path("/m.gguf"), Path("/tmp/ds4.lock.engine")
    assert exit_error(0, "", m, lk) is None
    e = exit_error(1, "ds4: loading\nds4: only GGUF v3 is supported\n", m, lk)
    assert type(e) is ModelFileError and "/m.gguf" in str(e) and str(e).endswith("only GGUF v3 is supported")
    e = exit_error(2, "ds4: another ds4 process is already running (pid 7); refusing to start\n", m, lk)
    assert type(e) is EngineLockedError and str(lk) in str(e) and "pid 7" in str(e)
    assert type(exit_error(NATIVE_REFUSED, "ds4: V4.1 vision requires Metal\n", m, lk)) is EngineError
    tb = "Traceback (most recent call last):\n  ...\nImportError: no module named ds4py\n"
    for status in (1, 2, NATIVE_REFUSED, PYTHON_ERROR, -9):      # no ds4 line: not a ds4 refusal
        e = exit_error(status, tb, m, lk)
        assert type(e) is EngineError and "ImportError" in str(e)

def test_engine_options(tmp_path, monkeypatch):
    o = engine_options(EngineConfig(Path("/m.gguf"), ctx=4096, backend="rocm", threads=3))
    assert (o.model_path, o.backend, o.context_size, o.placement_ctx_hint, o.n_threads) == (b"/m.gguf", 1, 4096, 4096, 3)
    assert (o.mtp_draft_tokens, o.mtp_margin, o.inspect_only) == (1, 3.0, False)
    assert engine_options(EngineConfig(Path("/m.gguf")), inspect_only=True).inspect_only
    monkeypatch.chdir(tmp_path)
    assert EngineConfig(Path("m.gguf")).model == Path.cwd() / "m.gguf"   # absolute on the caller's thread
    with pytest.raises(ValueError, match="vulkan"): engine_options(EngineConfig(Path("/m"), backend="vulkan"))

@pytest.mark.build
def test_open_failures_become_exceptions(built_upstream, fresh_loader, tmp_path, monkeypatch):
    lock = tmp_path / "ds4.lock"
    monkeypatch.setenv("DS4_LOCK_FILE", str(lock))
    native = LibNative(load_library(built_upstream[0]))
    assert native.parse_think_level("42") == 1042 and native.parse_think_level("high") is None
    assert native.think_mode_for_context(ThinkMode.MAX, 4096) == ThinkMode.HIGH
    (tmp_path / "v0.gguf").write_bytes(b"GGUF" + bytes(60))        # GGUF version 0
    for model, text in [("missing.gguf", "cannot open model"), ("v0.gguf", "only GGUF v3")]:
        with pytest.raises(ModelFileError, match=text):           # ds4 exit()s; only the child dies
            native.open_engine(EngineConfig(tmp_path / model))
    with InstanceLock.acquire(lock):                               # another holder of the real lock
        with pytest.raises(EngineLockedError):
            native.open_engine(EngineConfig(tmp_path / "v0.gguf"))
    with InstanceLock.acquire(engine_lock_path()):                 # ds4's own lock: the child exits 2
        with pytest.raises(EngineLockedError, match="already running"):
            native.open_engine(EngineConfig(tmp_path / "v0.gguf"))
    InstanceLock.acquire(lock).close()                             # the failed opens released it

@pytest.mark.live
@pytest.mark.parametrize("check", CONTRACT, ids=lambda f: f.__name__)
def test_contract_live(live_engine, check):
    if check is check_snapshot_restore_matches_rebuild and not live_engine._test_native.restore_enabled:
        pytest.skip("snapshot restore disabled by live checks")
    check(live_engine, live_engine.tokenize("The quick brown fox jumps over the lazy dog. " * 2))


@pytest.fixture(scope="module")
def live_engine():
    model = os.environ.get("DS4_TEST_MODEL")
    if not model:
        pytest.skip("DS4_TEST_MODEL is not set")
    native = LibNative(load_library())
    previous = os.environ.get("DS4_LOCK_FILE")
    engine = native.open_engine(EngineConfig(Path(model), ctx=4096, backend=native.backend))
    engine._test_native = native
    engine._test_lock_env = previous
    yield engine
    engine.close()


@pytest.mark.live
def test_token_text_live(live_engine):
    assert b"".join(live_engine.token_text(t) for t in live_engine.tokenize("Hello world")) == b"Hello world"


@pytest.mark.live
def test_lock_lifetime_live(live_engine):
    assert os.environ.get("DS4_LOCK_FILE") == live_engine._test_lock_env
    with pytest.raises(EngineLockedError):
        InstanceLock.acquire(lock_path())


def test_session_abi_and_ownership(stub_engine):
    import ctypes
    from ds4py._native.protocol import SyncStatus
    engine, c = stub_engine
    session = engine.create_session(64)
    events = []
    assert session.sync([1, 2, 3], progress=lambda *e: events.append(e)) is SyncStatus.DONE
    assert events == [("prefill_display", 0, 3), ("prefill_chunk", 3, 3)]
    assert c.stub_clears() == 3
    tokens = session.tokens()
    session.sync([9])
    assert tokens == [1, 2, 3] and session.tokens() == [9]
    rng = ctypes.c_uint64(10)
    assert session.sample(0.5, 7, 0.75, 0.25, rng) == 42 and rng.value == 18
    from array import array
    values = array("f", [0.0]) * engine.info().vocab_size
    session.copy_logits(values)
    assert list(values[:4]) == [0.0, 1.0, 2.0, 3.0]
    values[7] = 9.5
    session.set_logits(values)
    copied = array("f", [0.0]) * len(values)
    session.copy_logits(copied)
    assert copied[7] == pytest.approx(9.5)
    c.stub_copy_logits_result(0)
    with pytest.raises(EngineError, match="copy_logits failed"):
        session.copy_logits(values)
    c.stub_copy_logits_result(256)
    doubles = array("d", [1.0]) * len(values)
    session.set_logits(doubles)
    session.copy_logits(values)
    assert list(values) == pytest.approx([1.0] * len(values))
    with pytest.raises(TypeError, match="contiguous float32"):
        session.copy_logits(doubles)
    with pytest.raises(TypeError, match="contiguous float32"):
        session.copy_logits(memoryview(array("f", [0.0]) * 512)[::2])
    strided_source = array("f", [0.0]) * 512
    strided_source[::2] = array("f", [2.0]) * 256
    session.set_logits(memoryview(strided_source)[::2])
    session.copy_logits(values)
    assert list(values) == pytest.approx([2.0] * len(values))
    for bad in (array("f", [0.0]), array("f", [0.0]) * (len(values) + 1)):
        with pytest.raises(ValueError, match="vocabulary size"):
            session.copy_logits(bad)
        with pytest.raises(ValueError, match="vocabulary size"):
            session.set_logits(bad)
    assert engine.tokenize("text") == [7, 9]
    assert engine.token_text(7) == b"a\0b"
    snap = session.save_snapshot()
    assert snap.nbytes == 4
    session.load_snapshot(snap)
    snap.close(); snap.close()
    assert c.stub_snapshots() == 1
    with pytest.raises(EngineError, match="closed"):
        session.load_snapshot(snap)
    session.close(); session.close()
    assert c.stub_frees() == 1
    with pytest.raises(EngineError, match="closed"):
        session.tokens()


def test_callbacks_errors_and_thread_confinement(stub_engine):
    from concurrent.futures import ThreadPoolExecutor
    from ds4py._native.protocol import SyncStatus
    engine, c = stub_engine
    session = engine.create_session(64)
    for kind in ("progress", "cancel"):
        error = RuntimeError(kind)
        def fail(*args):
            raise error
        with pytest.raises(RuntimeError) as caught:
            session.sync([1], **{kind: fail})
        assert caught.value is error
    assert c.stub_clears() == 6
    assert session.sync([1], cancel=lambda: True) is SyncStatus.INTERRUPTED
    with pytest.raises(EngineError, match="empty prompt"):
        session.sync([])
    with pytest.raises(EngineError, match="eval rejected"):
        session.eval(8)
    assert c.stub_clears() == 12
    with ThreadPoolExecutor(1) as pool:
        for method in (session.tokens, session.close, session.save_snapshot, engine.eos, engine.close):
            with pytest.raises(AssertionError, match="another thread"):
                pool.submit(method).result()
    assert c.stub_snapshots() == 0
    engine.close()
    assert c.stub_frees() == 1


def test_session_finalizer_frees(stub_engine):
    import gc
    engine, c = stub_engine
    session = engine.create_session(64)
    del session
    gc.collect()
    assert c.stub_frees() == 1


@pytest.mark.parametrize("backend", ["cpu", "metal"])
def test_open_environment_and_failure_cleanup(stub_engine, tmp_path, monkeypatch, backend):
    import ctypes
    from types import SimpleNamespace
    import ds4py._native.api as api
    engine, c = stub_engine
    lib = engine._lib
    (tmp_path / "metal").mkdir()
    (tmp_path / "metal" / "moe.metal").write_bytes(b"moe")
    (tmp_path / "metal" / "flash_attn.metal").write_bytes(b"flash")
    lib.manifest = SimpleNamespace(
        backend=backend, live_checks={"restore": {"ok": False}},
        source_path=str(tmp_path / "deleted-checkout"),
        metal_sources={"metal/moe.metal": hashlib.sha256(b"moe").hexdigest(),
                       "metal/flash_attn.metal": hashlib.sha256(b"flash").hexdigest()},
    )
    seen = []
    original_open = lib.fn("ds4_engine_open")
    def observe(*args):
        seen.append({k: os.environ.get(k) for k in ("DS4_METAL_MOE_SOURCE", "DS4_METAL_FLASH_ATTN_SOURCE")})
        return original_open(*args)
    monkeypatch.setitem(lib._functions, "ds4_engine_open", observe)
    native = LibNative(lib)
    assert not native.restore_enabled
    monkeypatch.setattr(api, "inspect_model", lambda *args: None)  # actual child refusal is covered by CPU build test
    lock = tmp_path / "open.lock"
    monkeypatch.setenv("DS4_LOCK_FILE", str(lock))
    override = tmp_path / "user-override.metal"
    override.write_bytes(b"moe")
    monkeypatch.setenv("DS4_METAL_MOE_SOURCE", str(override))
    monkeypatch.delenv("DS4_METAL_FLASH_ATTN_SOURCE", raising=False)
    environment = dict(os.environ)
    before = Path.cwd()
    opened = native.open_engine(EngineConfig(Path("model"), backend=backend))
    c.stub_opened_lock.restype = ctypes.c_char_p
    assert c.stub_opened_lock() == os.fsencode(tmp_path / "open.lock.engine")
    assert Path.cwd() == before
    assert seen[-1] == {
        "DS4_METAL_MOE_SOURCE": str(override),
        "DS4_METAL_FLASH_ATTN_SOURCE": str(tmp_path.resolve() / "metal/flash_attn.metal") if backend == "metal" else None,
    }
    assert dict(os.environ) == environment
    with pytest.raises(EngineLockedError):
        InstanceLock.acquire(lock)
    opened.close()
    with pytest.raises(EngineError, match="ds4_engine_open failed"):
        native.open_engine(EngineConfig(Path("model"), ctx=13, backend=backend))
    assert dict(os.environ) == environment
    InstanceLock.acquire(lock).close()
    monkeypatch.delenv("DS4_LOCK_FILE")
    monkeypatch.setattr(api, "lock_path", lambda: lock)
    native.open_engine(EngineConfig(Path("model"), backend=backend)).close()
    assert "DS4_LOCK_FILE" not in os.environ


@pytest.fixture
def metal_native(stub_engine, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import ds4py._native.api as api
    engine, _ = stub_engine
    lib = engine._lib
    (tmp_path / "metal").mkdir()
    shader = tmp_path / "metal/dense.metal"
    shader.write_bytes(b"first bundle")
    lib.manifest = SimpleNamespace(
        backend="metal", live_checks={}, source_path="/deleted/checkout",
        metal_sources={"metal/dense.metal": hashlib.sha256(b"first bundle").hexdigest()},
    )
    monkeypatch.setattr(api, "inspect_model", lambda *args: None)
    monkeypatch.setenv("DS4_LOCK_FILE", str(tmp_path / "native.lock"))
    monkeypatch.delenv("DS4_METAL_DENSE_SOURCE", raising=False)
    return LibNative(lib), shader


def test_metal_environment_restored_on_python_exception(metal_native, tmp_path, monkeypatch):
    native, _ = metal_native
    before = dict(os.environ)
    def fail(*args):
        assert os.environ["DS4_METAL_DENSE_SOURCE"] == str(tmp_path.resolve() / "metal/dense.metal")
        raise RuntimeError("open raised")
    monkeypatch.setitem(native.lib._functions, "ds4_engine_open", fail)
    with pytest.raises(RuntimeError, match="open raised"):
        native.open_engine(EngineConfig(tmp_path / "model", backend="metal"))
    assert dict(os.environ) == before
    InstanceLock.acquire(tmp_path / "native.lock").close()


def test_later_metal_open_uses_another_bundle(metal_native, tmp_path, monkeypatch):
    native, shader = metal_native
    original_open = native.lib.fn("ds4_engine_open")
    seen = []
    def observe(*args):
        seen.append(os.environ["DS4_METAL_DENSE_SOURCE"])
        return original_open(*args)
    monkeypatch.setitem(native.lib._functions, "ds4_engine_open", observe)
    native.open_engine(EngineConfig(tmp_path / "model", backend="metal")).close()
    second = tmp_path / "second"
    (second / "metal").mkdir(parents=True)
    (second / "metal/dense.metal").write_bytes(b"second bundle")
    native.lib.path = second / "stub.so"
    native.lib.manifest.metal_sources = {"metal/dense.metal": hashlib.sha256(b"second bundle").hexdigest()}
    native.open_engine(EngineConfig(tmp_path / "model", backend="metal")).close()
    assert seen == [str(shader.resolve()), str(second.resolve() / "metal/dense.metal")]
    assert "DS4_METAL_DENSE_SOURCE" not in os.environ


@pytest.mark.parametrize("change", ["bundle", "override"])
def test_open_rechecks_shader_files_after_library_load(metal_native, tmp_path, monkeypatch, change):
    native, shader = metal_native
    target = shader
    if change == "override":
        target = tmp_path / "override.metal"
        target.write_bytes(shader.read_bytes())
        monkeypatch.setenv("DS4_METAL_DENSE_SOURCE", str(target))
    target.write_bytes(b"wrong after library load")
    before = dict(os.environ)
    monkeypatch.setitem(native.lib._functions, "ds4_engine_open", lambda *a: pytest.fail("native open called"))
    with pytest.raises(AbiMismatch, match="ds4py build"):
        native.open_engine(EngineConfig(tmp_path / "model", backend="metal"))
    assert dict(os.environ) == before
    InstanceLock.acquire(tmp_path / "native.lock").close()


def test_info_captured_once_on_open(stub_engine):
    engine, c = stub_engine
    assert c.stub_info_reads() == 1
    first, second = engine.info(), engine.info()
    assert first is second
    assert (first.name, first.model_id, first.family, first.vocab_size) == ("stub-model", 17, "glm", 256)
    assert c.stub_info_reads() == 1


@pytest.mark.parametrize("already_closed", [False, True])
def test_off_owner_gc_preserves_session_cleanup(stub_engine, monkeypatch, already_closed):
    import gc
    import sys
    import threading
    import weakref

    engine, c = stub_engine
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        session = engine.create_session(64)
        if already_closed:
            session.close()
        # Guarantee collection on the other thread even if callback cycles change.
        session._test_cycle = session
        ref = weakref.ref(session)
        del session
        collector = threading.Thread(target=gc.collect)
        collector.start()
        collector.join()
        assert ref() is None
        assert c.stub_frees() == int(already_closed)
        assert c.stub_wrong_thread_frees() == 0
        engine.close()
        engine.close()
        assert c.stub_frees() == 1
        assert c.stub_wrong_thread_frees() == 0
        assert unraisable == []
    finally:
        if was_enabled:
            gc.enable()


def test_failed_snapshot_save_frees_partial_allocation(stub_engine):
    engine, c = stub_engine
    session = engine.create_session(64)
    c.stub_fail_snapshot(1)
    with pytest.raises(EngineError, match="^snapshot write rejected$"):
        session.save_snapshot()
    assert c.stub_snapshots() == 1
    c.stub_fail_snapshot(0)
    snap = session.save_snapshot()
    assert snap.nbytes == 4
    assert c.stub_snapshots() == 1  # successful save transfers ownership
    session.load_snapshot(snap)
    snap.close()
    snap.close()
    assert c.stub_snapshots() == 2


def test_weakref_engine_close_before_session_finalizer_frees_once(stub_engine, monkeypatch):
    import gc
    import sys
    import weakref

    engine, c = stub_engine
    calls, unraisable = [], []
    native_free = engine._lib.fn("ds4_session_free")
    native_close = engine._lib.fn("ds4_engine_close")
    freed = set()

    def guarded_free(raw):
        calls.append(("session_free", raw))
        # Still perform the real C free, but intercept any duplicate attempt so
        # a regression reports an assertion instead of crashing pytest.
        if raw not in freed:
            freed.add(raw)
            native_free(raw)

    def recorded_close(raw):
        calls.append(("engine_close", raw))
        native_close(raw)

    monkeypatch.setitem(engine._lib._functions, "ds4_session_free", guarded_free)
    monkeypatch.setitem(engine._lib._functions, "ds4_engine_close", recorded_close)
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        session = engine.create_session(64)
        raw, engine_raw = session.raw, engine.raw
        ref = weakref.ref(session, lambda _: engine.close())
        del session
        gc.collect()
        assert ref() is None
        assert calls == [("session_free", raw), ("engine_close", engine_raw)]
        assert c.stub_frees() == 1 and c.stub_wrong_thread_frees() == 0
        assert unraisable == []
        engine.close()
        assert calls == [("session_free", raw), ("engine_close", engine_raw)]
    finally:
        if was_enabled:
            gc.enable()
