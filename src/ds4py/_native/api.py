"""Thread-confined ctypes lifecycle and ownership adapters for ds4."""

import ctypes as C
import fcntl
import os
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Self, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from .loader import Library

from ds4py.abi.shaders import metal_source_env
from ds4py.errors import EmptyPromptError, EngineError, EngineLockedError

from . import chat, payload, structs
from ._results import bool_result, int_result
from .modelcheck import inspect_model
from .protocol import (
    BACKENDS,
    CancelFn,
    EngineConfig,
    EngineInfo,
    ProgressFn,
    Snapshot,
    SyncStatus,
    family_of,
)

_open_lock = threading.Lock()


# Instance locking and open-time configuration.
def lock_path() -> Path:
    return Path(os.environ.get("DS4_LOCK_FILE") or "/tmp/ds4.lock")  # noqa: S108 - Preserve the documented default lock location.


def engine_lock_path() -> Path:
    path = lock_path()
    return path.with_name(path.name + ".engine")


class InstanceLock:
    def __init__(self, fd: int) -> None:
        self._fd = fd

    @classmethod
    def acquire(cls, path: Path) -> InstanceLock:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pid = os.read(fd, 64).decode(errors="replace").strip()
                owner = f" (pid {pid})" if pid else ""
                raise EngineLockedError(
                    f"another ds4 process holds {path}{owner}; stop it or set DS4_LOCK_FILE"
                ) from None
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
        except BaseException:
            os.close(fd)
            raise
        return cls(fd)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@contextmanager
def _open_environment(second: Path, sources: dict[str, str]) -> Generator[None]:
    changes = {key: value for key, value in sources.items() if key not in os.environ}
    changes["DS4_LOCK_FILE"] = str(second)
    previous = {key: os.environ.get(key) for key in changes}
    try:
        os.environ.update(changes)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def engine_options(config: EngineConfig, *, inspect_only: bool = False) -> structs.EngineOptions:
    if config.backend not in BACKENDS:
        raise ValueError(f"unknown backend: {config.backend}")
    return structs.EngineOptions(
        model_path=os.fsencode(config.model),
        backend=BACKENDS[config.backend],
        n_threads=config.threads,
        context_size=config.ctx,
        placement_ctx_hint=config.ctx,
        mtp_draft_tokens=1,
        mtp_margin=3.0,
        inspect_only=inspect_only,
    )


class LibNative:
    def __init__(self, lib: Library) -> None:
        self.lib = lib
        self.backend = lib.manifest.backend
        self.restore_enabled = lib.manifest.live_checks.get("restore", {}).get("ok") is not False

    def bind(self, name: str) -> Callable[..., Any]:
        return self.lib.fn(name)

    def parse_think_level(self, text: str) -> int | None:
        mode = C.c_int()
        return mode.value if self.bind("ds4_think_mode_parse_level")(text.encode(), C.byref(mode)) else None

    def think_mode_for_context(self, mode: int, ctx: int) -> int:
        return int_result(self.bind("ds4_think_mode_for_context")(mode, ctx))

    def open_engine(self, config: EngineConfig) -> NativeEngine:
        """Inspect then open, holding the original lock until engine close.

        Inspection is diagnostic, not protection against every native exit:
        it skips tokenizer loading. Invalid tokenizer tables, missing required
        tokens, model changes between inspection and open, and GPU setup errors
        can terminate Python. Keep the model unchanged for the engine lifetime.
        Open temporarily changes process-wide lock and Metal source variables.
        The open lock serializes ds4py opens, not other threads or subprocesses.
        Do not start
        ds4 subprocesses during open: another thread's child can inherit that
        path, take the second lock first, and make native open exit(2). These
        restrictions reduce risk but do not guarantee against process termination.
        """
        with _open_lock:
            lock = InstanceLock.acquire(lock_path())
            try:
                second = engine_lock_path()
                sources = (
                    metal_source_env(
                        self.lib.path.resolve().parent,
                        self.lib.manifest.metal_sources,
                        os.environ,
                    )
                    if self.backend == "metal"
                    else {}
                )
                inspect_model(self.lib.path, config, second)
                options = engine_options(config)
                raw = C.c_void_p()
                with _open_environment(second, sources):
                    status = self.bind("ds4_engine_open")(C.byref(raw), C.byref(options))
                if status:
                    raise EngineError(  # noqa: TRY301 - Keep native open cleanup in the same guarded scope.
                        f"ds4_engine_open failed ({status})"
                    )
                try:
                    return NativeEngine(self.lib, cast("int", raw.value), options, lock)
                except BaseException:
                    self.bind("ds4_engine_close")(raw)
                    raise
            except BaseException:
                lock.close()
                raise


# Engine and session calls share thread and closed-handle checks.
class _Handle:
    def __init__(self, lib: Library, raw: int) -> None:
        self._lib, self.raw = lib, raw
        self._thread = threading.get_ident()

    def _assert_thread(self) -> None:
        if threading.get_ident() != self._thread:
            raise AssertionError("native handle used on another thread")

    def _call(self, name: str, *args: Any) -> Any:
        self._assert_thread()
        if not self.raw:
            raise EngineError("native handle is closed")
        return self._lib.fn(name)(self.raw, *args)


class NativeEngine(_Handle):
    def __init__(
        self,
        lib: Library,
        raw: int,
        options: structs.EngineOptions,
        lock: InstanceLock,
    ) -> None:
        super().__init__(lib, raw)
        self._options, self._lock = options, lock
        self._model_path = options.model_path
        self._sessions: weakref.WeakSet[NativeSession] = weakref.WeakSet()
        # Retain cleanup ownership even if a wrapper is collected off-thread.
        self._session_allocations: set[int] = set()
        self._info = self._read_info()

    def info(self) -> EngineInfo:
        self._assert_thread()
        if not self.raw:
            raise EngineError("native handle is closed")
        return self._info

    def _read_info(self) -> EngineInfo:
        def get(name: str) -> Any:
            return self._call("ds4_engine_" + name)

        deep, glm, qwen = get("is_deepseek41"), get("is_glm_dsa"), get("is_qwen4")
        return EngineInfo(
            get("model_name").decode(),
            get("model_id"),
            family_of(deepseek41=deep, glm=glm, qwen=qwen),
            deep,
            glm,
            qwen,
            get("has_vision"),
            get("routed_quant_bits"),
            get("mtp_draft_tokens"),
            get("vocab_size"),
        )

    def tokenize(self, text: str, *, rendered: bool = False) -> list[int]:
        self._assert_thread()
        tokens = structs.Tokens()
        try:
            self._call(
                "ds4_tokenize_rendered_chat" if rendered else "ds4_tokenize_text", text.encode(), C.byref(tokens)
            )
            return list(tokens.v[: tokens.len])
        finally:
            self._lib.fn("ds4_tokens_free")(C.byref(tokens))

    def chat_begin(self) -> list[int]:
        """Return the chat-begin delta on the owning worker thread."""
        return chat.append_delta(self, (), "ds4_chat_begin")

    def think_prefix(self, prefix: Sequence[int], mode: int) -> list[int]:
        """Return only the appended think delta; worker-thread use only."""
        return chat.append_delta(self, prefix, "ds4_chat_append_think_prefix", mode)

    def render_message(self, prefix: Sequence[int], role: str, content: str) -> list[int]:
        """Return the native role/message delta without changing prefix; worker only."""
        return chat.append_delta(
            self,
            prefix,
            "ds4_chat_append_message",
            role.encode(),
            content.encode(),
        )

    def assistant_prefix(self, prefix: Sequence[int], mode: int) -> list[int]:
        """Return only the appended assistant opener; worker-thread use only."""
        return chat.append_delta(self, prefix, "ds4_chat_append_assistant_prefix", mode)

    def qwen_reasoning_effort(self, mode: int) -> str | None:
        """Copy borrowed effort text to Python, or return None; worker only."""
        return chat.qwen_reasoning_effort(self, mode)

    def token_text(self, token: int) -> bytes:
        length = C.c_size_t()
        ptr = self._call("ds4_token_text", token, C.byref(length))
        try:
            return C.string_at(ptr, length.value)
        finally:
            self._lib.free(ptr)

    def eos(self) -> int:
        return int_result(self._call("ds4_token_eos"))

    def is_stop(self, token: int) -> bool:
        return bool_result(self._call("ds4_token_is_stop", token))

    def is_stop_for_think_mode(self, token: int, mode: int) -> bool:
        """Apply the native mode-aware stop policy on the owning worker."""
        return bool_result(self._call("ds4_token_is_stop_for_think_mode", token, mode))

    def create_session(self, ctx: int) -> NativeSession:
        self._assert_thread()
        if not self.raw:
            raise EngineError("native handle is closed")
        raw = C.c_void_p()
        if self._lib.fn("ds4_session_create")(C.byref(raw), self.raw, ctx):
            raise EngineError("ds4_session_create failed")
        self._session_allocations.add(cast("int", raw.value))
        session = NativeSession(self, cast("int", raw.value))
        self._sessions.add(session)
        return session

    def close(self) -> None:
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
    def __init__(self, lib: Library, data: structs.SessionSnapshot) -> None:
        self._lib, self._data = lib, data
        self.nbytes = data.len
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._lib.fn("ds4_session_snapshot_free")(C.byref(self._data))
            self._closed = True


class NativeStagedPayload:
    def __init__(self, lib: Library, data: structs.SessionPayloadFile) -> None:
        self._lib, self._data = lib, data
        self.bytes = data.bytes
        self._closed = False

    def write(self, file: BinaryIO) -> None:
        if self._closed:
            raise EngineError("staged payload is closed")
        error = C.create_string_buffer(512)
        with payload.native_stream(file, "wb") as stream:
            status = self._lib.fn("ds4_session_write_staged_payload")(C.byref(self._data), stream, error, len(error))
        if status:
            raise EngineError(error.value.decode(errors="replace"))

    def close(self) -> None:
        if not self._closed:
            self._lib.fn("ds4_session_payload_file_free")(C.byref(self._data))
            self._closed = True


def _tokens(tokens: Sequence[int]) -> structs.Tokens:
    array = (C.c_int * len(tokens))(*tokens)
    return structs.Tokens(array, len(tokens), len(tokens))


class NativeSession(_Handle):
    def __init__(self, engine: NativeEngine, raw: int) -> None:
        super().__init__(engine._lib, raw)
        self._engine = engine
        self._progress = self._cancel = self._callback_error = None
        self._progress_fn = structs.PROGRESS_FN(self._on_progress)
        self._cancel_fn = structs.CANCEL_FN(self._on_cancel)

    def _on_progress(self, ud: Any, event: bytes, current: int, total: int) -> None:
        try:
            if self._progress is not None and self._callback_error is None:
                self._progress(event.decode(), current, total)
        except BaseException as exc:  # noqa: BLE001 - Propagate every callback failure to the worker.
            self._callback_error = exc

    def _on_cancel(self, ud: Any) -> bool:
        if self._callback_error is not None:
            return True
        try:
            return bool(self._cancel and self._cancel())
        except BaseException as exc:  # noqa: BLE001 - Cancellation must stop native work on every failure.
            self._callback_error = exc
            return True

    def _error_call(self, name: str, *args: Any) -> None:
        error = C.create_string_buffer(512)
        status = self._call(name, *args, error, len(error))
        if status:
            raise EngineError(error.value.decode(errors="replace"))

    def sync(
        self,
        tokens: Sequence[int],
        *,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> SyncStatus:
        self._assert_thread()
        self._progress, self._cancel, self._callback_error = progress, cancel, None
        try:
            self._call("ds4_session_set_progress", self._progress_fn, None)
            self._call("ds4_session_set_display_progress", self._progress_fn, None)
            self._call("ds4_session_set_cancel", self._cancel_fn, None)
            if not tokens:
                raise EmptyPromptError("empty prompt")
            error = C.create_string_buffer(512)
            status = self._call("ds4_session_sync", C.byref(_tokens(tokens)), error, len(error))
            if self._callback_error is not None:
                raise self._callback_error
            if status == 0:
                return SyncStatus.DONE
            if status == SyncStatus.INTERRUPTED.value:
                return SyncStatus.INTERRUPTED
            raise EngineError(error.value.decode(errors="replace"))
        finally:
            self._call("ds4_session_set_progress", structs.PROGRESS_FN(), None)
            self._call("ds4_session_set_display_progress", structs.PROGRESS_FN(), None)
            self._call("ds4_session_set_cancel", structs.CANCEL_FN(), None)
            self._progress = self._cancel = self._callback_error = None

    def sample(
        self,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        rng: C.c_uint64,
    ) -> int:
        return int_result(self._call("ds4_session_sample", temperature, top_k, top_p, min_p, C.byref(rng)))

    def argmax(self) -> int:
        return int_result(self._call("ds4_session_argmax"))

    def copy_logits(self, out: Any) -> None:
        """Copy into a vocabulary-sized writable float32 buffer; worker only."""
        from .logits import copy_logits  # noqa: PLC0415 - Keep native module initialization acyclic.

        copy_logits(self, out)

    def set_logits(self, values: Sequence[float]) -> None:
        """Replace logits from a vocabulary-sized sequence; worker only."""
        from .logits import set_logits  # noqa: PLC0415 - Keep native module initialization acyclic.

        set_logits(self, values)

    def eval(self, token: int) -> None:
        self._error_call("ds4_session_eval", token)

    def rewind(self, pos: int) -> None:
        self._call("ds4_session_rewind", pos)

    def pos(self) -> int:
        return int_result(self._call("ds4_session_pos"))

    def ctx(self) -> int:
        return int_result(self._call("ds4_session_ctx"))

    def tokens(self) -> list[int]:
        data = self._call("ds4_session_tokens").contents
        return list(data.v[: data.len])

    def common_prefix(self, tokens: Sequence[int]) -> int:
        return int_result(self._call("ds4_session_common_prefix", C.byref(_tokens(tokens))))

    def checkpoint_valid(self) -> bool:
        return bool_result(self._call("ds4_session_checkpoint_valid"))

    def invalidate(self) -> None:
        self._call("ds4_session_invalidate")

    def payload_bytes(self) -> int:
        return int_result(self._call("ds4_session_payload_bytes"))

    def stage_payload(self) -> NativeStagedPayload:
        self._assert_thread()
        data = structs.SessionPayloadFile()
        try:
            self._error_call("ds4_session_stage_payload", C.byref(data))
            return NativeStagedPayload(self._lib, data)
        except BaseException:
            self._lib.fn("ds4_session_payload_file_free")(C.byref(data))
            raise

    def load_payload(self, file: BinaryIO, payload_bytes: int) -> None:
        self._assert_thread()
        with payload.native_stream(file, "rb") as stream:
            self._error_call("ds4_session_load_payload", stream, payload_bytes)

    def save_snapshot(self) -> NativeSnapshot:
        self._assert_thread()
        data = structs.SessionSnapshot()
        try:
            self._error_call("ds4_session_save_snapshot", C.byref(data))
            return NativeSnapshot(self._lib, data)
        except BaseException:
            self._lib.fn("ds4_session_snapshot_free")(C.byref(data))
            raise

    def load_snapshot(self, snap: Snapshot) -> None:
        self._assert_thread()
        native_snap = cast("NativeSnapshot", snap)
        if native_snap._closed:
            raise EngineError("snapshot is closed")
        self._error_call("ds4_session_load_snapshot", C.byref(native_snap._data))

    def close(self) -> None:
        self._assert_thread()
        if self.raw:
            raw, self.raw = self.raw, 0
            self._engine._sessions.discard(self)
            # A weakref callback may have closed the engine before __del__.
            # The registry, not a stale wrapper pointer, owns the right to free.
            if raw in self._engine._session_allocations:
                self._engine._session_allocations.remove(raw)
                self._lib.fn("ds4_session_free")(raw)

    def __del__(self) -> None:
        # Cyclic GC may run on any thread. Leave off-thread allocations to the
        # engine's owner-thread close; never call C or assert from that finalizer.
        if self.raw and threading.get_ident() == self._thread:
            self.close()
