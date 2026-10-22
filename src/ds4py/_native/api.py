"""Thread-confined ctypes lifecycle and ownership adapters for ds4."""
import ctypes as C
import fcntl
import os
from pathlib import Path
import threading
import weakref

from ds4py.errors import EngineError, EngineLockedError
from . import structs
from .loader import Library
from .modelcheck import inspect_model
from .protocol import BACKENDS, EngineConfig, EngineInfo, SyncStatus, family_of

_open_lock = threading.Lock()


# Instance locking and open-time configuration.
def lock_path() -> Path:
    return Path(os.environ.get("DS4_LOCK_FILE") or "/tmp/ds4.lock")


def engine_lock_path() -> Path:
    path = lock_path()
    return path.with_name(path.name + ".engine")


class InstanceLock:
    def __init__(self, fd):
        self._fd = fd

    @classmethod
    def acquire(cls, path: Path) -> "InstanceLock":
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pid = os.read(fd, 64).decode(errors="replace").strip()
                owner = f" (pid {pid})" if pid else ""
                raise EngineLockedError(f"another ds4 process holds {path}{owner}; stop it or set DS4_LOCK_FILE") from None
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
        except BaseException:
            os.close(fd)
            raise
        return cls(fd)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def metal_source_env(src: Path) -> dict[str, str]:
    directory = src.absolute() / "metal"
    sources = {f"DS4_METAL_{p.stem.upper()}_SOURCE": str(p) for p in directory.glob("*.metal")}
    if not sources:
        raise EngineError(f"no Metal sources in {directory}")
    return sources


def engine_options(config: EngineConfig, *, inspect_only: bool = False) -> structs.EngineOptions:
    if config.backend not in BACKENDS:
        raise ValueError(f"unknown backend: {config.backend}")
    return structs.EngineOptions(
        model_path=os.fsencode(config.model), backend=BACKENDS[config.backend],
        n_threads=config.threads, context_size=config.ctx, placement_ctx_hint=config.ctx,
        mtp_draft_tokens=1, mtp_margin=3.0, inspect_only=inspect_only,
    )


class LibNative:
    def __init__(self, lib: Library):
        self.lib = lib
        self.backend = lib.manifest.backend
        self.restore_enabled = lib.manifest.live_checks.get("restore", {}).get("ok") is not False

    def bind(self, name):
        return self.lib.fn(name)

    def parse_think_level(self, text):
        mode = C.c_int()
        return mode.value if self.bind("ds4_think_mode_parse_level")(text.encode(), C.byref(mode)) else None

    def think_mode_for_context(self, mode, ctx):
        return self.bind("ds4_think_mode_for_context")(mode, ctx)

    def open_engine(self, config: EngineConfig):
        """Inspect then open, holding the original lock until engine close.

        Inspection is diagnostic, not protection against every native exit:
        it skips tokenizer loading. Invalid tokenizer tables, missing required
        tokens, model changes between inspection and open, and GPU setup errors
        can terminate Python. Keep the model unchanged for the engine lifetime.
        Open temporarily changes the process-wide DS4_LOCK_FILE. Do not start
        ds4 subprocesses during open: another thread's child can inherit that
        path, take the second lock first, and make native open exit(2). These
        restrictions reduce risk but do not guarantee against process termination.
        """
        with _open_lock:
            lock = InstanceLock.acquire(lock_path())
            try:
                second = engine_lock_path()
                inspect_model(self.lib.path, config, second)
                if self.backend == "metal":
                    for key, value in metal_source_env(Path(self.lib.manifest.source_path)).items():
                        os.environ.setdefault(key, value)
                options = engine_options(config)
                raw = C.c_void_p()
                previous = os.environ.get("DS4_LOCK_FILE")
                try:
                    os.environ["DS4_LOCK_FILE"] = str(second)
                    status = self.bind("ds4_engine_open")(C.byref(raw), C.byref(options))
                finally:
                    if previous is None:
                        os.environ.pop("DS4_LOCK_FILE", None)
                    else:
                        os.environ["DS4_LOCK_FILE"] = previous
                if status:
                    raise EngineError(f"ds4_engine_open failed ({status})")
                try:
                    return NativeEngine(self.lib, raw.value, options, lock)
                except BaseException:
                    self.bind("ds4_engine_close")(raw)
                    raise
            except BaseException:
                lock.close()
                raise


# Engine and session calls share thread and closed-handle checks.
class _Handle:
    def __init__(self, lib, raw):
        self._lib, self.raw = lib, raw
        self._thread = threading.get_ident()

    def _assert_thread(self):
        assert threading.get_ident() == self._thread, "native handle used on another thread"

    def _call(self, name, *args):
        self._assert_thread()
        if not self.raw:
            raise EngineError("native handle is closed")
        return self._lib.fn(name)(self.raw, *args)


class NativeEngine(_Handle):
    def __init__(self, lib, raw, options, lock):
        super().__init__(lib, raw)
        self._options, self._lock = options, lock
        self._model_path = options.model_path
        self._sessions = weakref.WeakSet()
        # Retain cleanup ownership even if a wrapper is collected off-thread.
        self._session_allocations = set()
        self._info = self._read_info()

    def info(self):
        self._assert_thread()
        if not self.raw:
            raise EngineError("native handle is closed")
        return self._info

    def _read_info(self):
        def get(name):
            return self._call("ds4_engine_" + name)
        deep, glm, qwen = get("is_deepseek41"), get("is_glm_dsa"), get("is_qwen4")
        return EngineInfo(get("model_name").decode(), get("model_id"),
                          family_of(deepseek41=deep, glm=glm, qwen=qwen),
                          deep, glm, qwen, get("has_vision"), get("routed_quant_bits"),
                          get("mtp_draft_tokens"), get("vocab_size"))

    def tokenize(self, text, *, rendered=False):
        self._assert_thread()
        tokens = structs.Tokens()
        try:
            self._call("ds4_tokenize_rendered_chat" if rendered else "ds4_tokenize_text", text.encode(), C.byref(tokens))
            return list(tokens.v[:tokens.len])
        finally:
            self._lib.fn("ds4_tokens_free")(C.byref(tokens))

    def token_text(self, token):
        length = C.c_size_t()
        ptr = self._call("ds4_token_text", token, C.byref(length))
        try:
            return C.string_at(ptr, length.value)
        finally:
            self._lib.free(ptr)

    def eos(self):
        return self._call("ds4_token_eos")

    def is_stop(self, token):
        return self._call("ds4_token_is_stop", token)

    def create_session(self, ctx):
        self._assert_thread()
        if not self.raw:
            raise EngineError("native handle is closed")
        raw = C.c_void_p()
        if self._lib.fn("ds4_session_create")(C.byref(raw), self.raw, ctx):
            raise EngineError("ds4_session_create failed")
        self._session_allocations.add(raw.value)
        session = NativeSession(self, raw.value)
        self._sessions.add(session)
        return session

    def close(self):
        self._assert_thread()
        if self.raw:
            for session in list(self._sessions):
                session.close()
            while self._session_allocations:
                raw = self._session_allocations.pop()
                self._lib.fn("ds4_session_free")(raw)
            self._lib.fn("ds4_engine_close")(self.raw)
            self.raw = 0
            self._lock.close()


class NativeSnapshot:
    def __init__(self, lib, data):
        self._lib, self._data = lib, data
        self.nbytes = data.len
        self._closed = False

    def close(self):
        if not self._closed:
            self._lib.fn("ds4_session_snapshot_free")(C.byref(self._data))
            self._closed = True


def _tokens(tokens):
    array = (C.c_int * len(tokens))(*tokens)
    return structs.Tokens(array, len(tokens), len(tokens))


class NativeSession(_Handle):
    def __init__(self, engine, raw):
        super().__init__(engine._lib, raw)
        self._engine = engine
        self._progress = self._cancel = self._callback_error = None
        self._progress_fn = structs.PROGRESS_FN(self._on_progress)
        self._cancel_fn = structs.CANCEL_FN(self._on_cancel)

    def _on_progress(self, ud, event, current, total):
        try:
            if self._progress is not None and self._callback_error is None:
                self._progress(event.decode(), current, total)
        except BaseException as exc:
            self._callback_error = exc

    def _on_cancel(self, ud):
        if self._callback_error is not None:
            return True
        try:
            return bool(self._cancel and self._cancel())
        except BaseException as exc:
            self._callback_error = exc
            return True

    def _error_call(self, name, *args):
        error = C.create_string_buffer(512)
        status = self._call(name, *args, error, len(error))
        if status:
            raise EngineError(error.value.decode(errors="replace"))

    def sync(self, tokens, *, progress=None, cancel=None):
        self._assert_thread()
        self._progress, self._cancel, self._callback_error = progress, cancel, None
        try:
            self._call("ds4_session_set_progress", self._progress_fn, None)
            self._call("ds4_session_set_display_progress", self._progress_fn, None)
            self._call("ds4_session_set_cancel", self._cancel_fn, None)
            error = C.create_string_buffer(512)
            status = self._call("ds4_session_sync", C.byref(_tokens(tokens)), error, len(error))
            if self._callback_error is not None:
                raise self._callback_error
            if status == 0:
                return SyncStatus.DONE
            if status == 2:
                return SyncStatus.INTERRUPTED
            raise EngineError(error.value.decode(errors="replace"))
        finally:
            self._call("ds4_session_set_progress", structs.PROGRESS_FN(), None)
            self._call("ds4_session_set_display_progress", structs.PROGRESS_FN(), None)
            self._call("ds4_session_set_cancel", structs.CANCEL_FN(), None)
            self._progress = self._cancel = self._callback_error = None

    def sample(self, temperature, top_k, top_p, min_p, rng):
        return self._call("ds4_session_sample", temperature, top_k, top_p, min_p, C.byref(rng))

    def argmax(self):
        return self._call("ds4_session_argmax")

    def eval(self, token):
        self._error_call("ds4_session_eval", token)

    def rewind(self, pos):
        self._call("ds4_session_rewind", pos)

    def pos(self):
        return self._call("ds4_session_pos")

    def ctx(self):
        return self._call("ds4_session_ctx")

    def tokens(self):
        data = self._call("ds4_session_tokens").contents
        return list(data.v[:data.len])

    def common_prefix(self, tokens):
        return self._call("ds4_session_common_prefix", C.byref(_tokens(tokens)))

    def checkpoint_valid(self):
        return self._call("ds4_session_checkpoint_valid")

    def invalidate(self):
        self._call("ds4_session_invalidate")

    def payload_bytes(self):
        return self._call("ds4_session_payload_bytes")

    def save_snapshot(self):
        self._assert_thread()
        data = structs.SessionSnapshot()
        try:
            self._error_call("ds4_session_save_snapshot", C.byref(data))
            return NativeSnapshot(self._lib, data)
        except BaseException:
            self._lib.fn("ds4_session_snapshot_free")(C.byref(data))
            raise

    def load_snapshot(self, snap):
        self._assert_thread()
        if snap._closed:
            raise EngineError("snapshot is closed")
        self._error_call("ds4_session_load_snapshot", C.byref(snap._data))

    def close(self):
        self._assert_thread()
        if self.raw:
            raw, self.raw = self.raw, 0
            self._engine._sessions.discard(self)
            # A weakref callback may have closed the engine before __del__.
            # The registry, not a stale wrapper pointer, owns the right to free.
            if raw in self._engine._session_allocations:
                self._engine._session_allocations.remove(raw)
                self._lib.fn("ds4_session_free")(raw)

    def __del__(self):
        # Cyclic GC may run on any thread. Leave off-thread allocations to the
        # engine's owner-thread close; never call C or assert from that finalizer.
        if self.raw and threading.get_ident() == self._thread:
            self.close()
