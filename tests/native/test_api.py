import fcntl
import os
import re
from pathlib import Path
import pytest
from ds4py.errors import EngineError, EngineLockedError, ModelFileError
from ds4py._native.api import InstanceLock, LibNative, lock_path, engine_lock_path, metal_source_env, engine_options
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

def test_metal_source_env(tmp_path, monkeypatch):
    (tmp_path / "metal").mkdir()
    for stem in ("flash_attn", "moe"):
        (tmp_path / "metal" / f"{stem}.metal").write_text("")
    monkeypatch.chdir(tmp_path)
    here = Path.cwd()                  # tmp_path may be a symlink, as on macOS
    assert metal_source_env(Path(".")) == {
        "DS4_METAL_FLASH_ATTN_SOURCE": str(here / "metal" / "flash_attn.metal"),
        "DS4_METAL_MOE_SOURCE": str(here / "metal" / "moe.metal"),
    }
    with pytest.raises(EngineError, match="no Metal sources"):
        metal_source_env(tmp_path / "metal")

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


@pytest.fixture
def stub_engine(cc, upstream_headers, tmp_path):
    # A compiled boundary double is needed because real sessions require a model.
    import ctypes
    import subprocess
    from types import SimpleNamespace
    from ds4py._native.api import NativeEngine
    from ds4py._native.bindings import CORE_BINDINGS
    from ds4py._native.loader import Library
    from ds4py._native.structs import EngineOptions

    path = tmp_path / "stub.so"
    subprocess.run([cc, "-shared", "-fPIC", "-I", str(upstream_headers),
                    str(Path(__file__).with_name("api_stub.c")), "-o", str(path)], check=True)
    handle = ctypes.CDLL(str(path))
    bindings = [b for b in CORE_BINDINGS if hasattr(handle, b.name)]
    lib = Library(path, None, SimpleNamespace(available={b.name for b in bindings}, absent=set()), handle, bindings)
    engine = NativeEngine(lib, 0x1234, EngineOptions(), InstanceLock.acquire(tmp_path / "lock"))
    yield engine, handle
    engine.close()


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
    (tmp_path / "metal" / "moe.metal").write_text("")
    lib.manifest = SimpleNamespace(backend=backend, live_checks={"restore": {"ok": False}}, source_path=str(tmp_path))
    native = LibNative(lib)
    assert not native.restore_enabled
    monkeypatch.setattr(api, "inspect_model", lambda *args: None)  # actual child refusal is covered by CPU build test
    lock = tmp_path / "open.lock"
    monkeypatch.setenv("DS4_LOCK_FILE", str(lock))
    monkeypatch.setenv("DS4_METAL_MOE_SOURCE", "user-override")
    before = Path.cwd()
    opened = native.open_engine(EngineConfig(Path("model"), backend=backend))
    c.stub_opened_lock.restype = ctypes.c_char_p
    assert c.stub_opened_lock() == os.fsencode(tmp_path / "open.lock.engine")
    assert Path.cwd() == before
    assert os.environ["DS4_METAL_MOE_SOURCE"] == "user-override"
    assert os.environ["DS4_LOCK_FILE"] == str(lock)
    with pytest.raises(EngineLockedError):
        InstanceLock.acquire(lock)
    opened.close()
    with pytest.raises(EngineError, match="ds4_engine_open failed"):
        native.open_engine(EngineConfig(Path("model"), ctx=13, backend=backend))
    assert os.environ["DS4_LOCK_FILE"] == str(lock)
    InstanceLock.acquire(lock).close()
    monkeypatch.delenv("DS4_LOCK_FILE")
    monkeypatch.setattr(api, "lock_path", lambda: lock)
    native.open_engine(EngineConfig(Path("model"), backend=backend)).close()
    assert "DS4_LOCK_FILE" not in os.environ


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
