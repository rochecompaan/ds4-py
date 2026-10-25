"""Public token-oriented engine facade and lazy, exclusively owned rounds."""
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence, TypeVar

from ds4py._native.api import LibNative
from ds4py._native.loader import load_library
from ds4py._native.protocol import EngineConfig, Native, ThinkMode
from ds4py.errors import EngineError
from . import chat as chat_ops
from .decode import GenerateJob
from .events import Checkpointed, Event, Interrupted, Stopped, Synced
from .hooks import LogitHookSet
from .plan import Plan
from .sampling import Rng, SamplingPolicy
from .transcript import Segment, Transcript, TranscriptView
from .worker import CheckpointJob, EngineWorker, Handles, SyncJob

T = TypeVar("T")


def open(model: str | Path = "ds4flash.gguf", *, ctx: int = 100_000,
         backend: str | None = None, think: int = ThinkMode.HIGH,
         seed: int | None = None, threads: int = 0, lib: str | Path | None = None,
         native: Native | None = None, cancel: threading.Event | None = None) -> "Engine":
    """Open an in-process engine with one worker owning all native handles.

    Child inspection provides diagnostics, not containment: inspect_only skips
    tokenizer validation. Invalid tokenizer tables or missing required tokens
    can still terminate Python. Model replacement/change races between inspection
    and open can also be fatal; keep the model unchanged for the engine lifetime.

    Python holds the original instance lock for the engine lifetime. Native open
    temporarily changes process-wide DS4_LOCK_FILE for a second lock.
    A ds4 subprocess in another thread can inherit that path and take the lock
    first, causing native exit(2). Do not start ds4 subprocesses during open;
    this reduces but does not eliminate the race. Other native exits, including
    GPU setup failures, also terminate Python. M1 does not guarantee process
    survival or universal exception safety for native failures.
    """
    native = native or LibNative(load_library(lib))
    if backend is not None and backend != native.backend:
        raise EngineError(
            f"this libds4 was built for {native.backend}, not {backend}; "
            f"run ds4py build --backend {backend}"
        )
    config = EngineConfig(Path(model), ctx, native.backend, threads)
    worker = EngineWorker(native, config, cancel=cancel)
    try:
        think = worker.call(lambda e, s: native.think_mode_for_context(think, ctx)).result()
        return Engine(native, worker, config, think, SamplingPolicy.for_engine(worker.info, seed=seed))
    except BaseException:
        worker.close()
        raise


class Engine:
    def __init__(self, native: Native, worker: EngineWorker, config: EngineConfig,
                 think: int, sampling_defaults: SamplingPolicy):
        self.native, self._worker = native, worker
        self.info, self.config, self.think = worker.info, config, think
        self.sampling_defaults = sampling_defaults
        self._session: Session | None = None
        self._session_lock = threading.Lock()

    def tokenize(self, text: str, *, rendered: bool = False) -> list[int]:
        return self._worker.call(lambda e, s: e.tokenize(text, rendered=rendered)).result()

    def _chat_begin(self) -> list[int]:
        return self._worker.call(lambda e, s: e.chat_begin()).result()

    def _think_prefix(self, prefix: Sequence[int], mode: int) -> list[int]:
        return self._worker.call(lambda e, s: e.think_prefix(prefix, mode)).result()

    def _render_message(self, prefix: Sequence[int], role: str, content: str) -> list[int]:
        return self._worker.call(lambda e, s: e.render_message(prefix, role, content)).result()

    def _assistant_prefix(self, prefix: Sequence[int], mode: int) -> list[int]:
        return self._worker.call(lambda e, s: e.assistant_prefix(prefix, mode)).result()

    def _qwen_reasoning_effort(self, mode: int) -> str | None:
        return self._worker.call(lambda e, s: e.qwen_reasoning_effort(mode)).result()

    def token_text(self, token: int) -> bytes:
        return self._worker.call(lambda e, s: e.token_text(token)).result()

    def _eos(self) -> int:
        return self._worker.call(lambda e, s: e.eos()).result()

    def session(self) -> "Session":
        with self._session_lock:
            if self._session is not None:
                raise EngineError("one session per engine")
            self._session = Session(self)
            return self._session

    def close(self) -> None:
        self._worker.close()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Session:
    def __init__(self, engine: Engine):
        self._engine, self._worker = engine, engine._worker
        self._transcript = Transcript()
        self._transcript_view = TranscriptView(self._transcript)
        self._rng = Rng(engine.sampling_defaults.seed)
        self._lock = threading.Lock()
        self._active_round = False
        self._paused_stream = None
        self._paused_control = None

    @property
    def transcript(self) -> TranscriptView:
        """Read-only history; append_tokens is the guarded caller mutation path."""
        return self._transcript_view

    @property
    def ctx_used(self) -> int:
        return self._worker.call(lambda e, s: s.pos()).result()

    def append_tokens(self, tokens: Sequence[int], kind: str,
                      meta: Mapping | None = None) -> Segment:
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            return self._transcript.append(tokens, kind, meta)

    def start_chat(self, tools=(), instructions=None) -> None:
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            chat_ops.start_chat(self, tools, instructions)

    def append_user(self, text: str, images=()) -> Segment:
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            return chat_ops.append_user(self, text, images)

    def append_prefix(self, turns) -> None:
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            chat_ops.append_prefix(self, turns)

    def append_tool_results(self, results) -> Segment:
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            return chat_ops.append_tool_results(self, results)

    def append_system_note(self, text: str) -> Segment:
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            return chat_ops.append_system_note(self, text)

    def _job_events(self, job) -> Iterator[Event]:
        # submit may fail before there is an iterator to drain.
        events = self._worker.submit(job)
        exhausted = interrupted = terminal = False
        try:
            for event in events:
                terminal = isinstance(event, (Synced, Interrupted, Stopped))
                interrupted = isinstance(event, Interrupted) or (
                    isinstance(event, Stopped) and event.reason == "interrupted"
                )
                yield event
            exhausted = True
        finally:
            if not exhausted:
                # A published terminal event means native work has already ended.
                if not terminal:
                    self.cancel()
                try:
                    # The barrier survives interruption of __next__, including
                    # another Ctrl+C during cleanup. Never release a live job.
                    events.wait()
                    for event in events:
                        if isinstance(event, Interrupted) or (
                            isinstance(event, Stopped) and event.reason == "interrupted"
                        ):
                            interrupted = True
                finally:
                    if not interrupted:
                        self._worker.cancel_event.clear()
        return interrupted

    def _raw_round(self, max_tokens, sampling, hooks=None, stop_sequences=()):
        resuming = self._paused_control is not None and self._paused_stream is None
        control = self._paused_control if resuming else None
        try:
            tokens = self._transcript.tokens
            if (yield from self._job_events(SyncJob(tokens))):
                return
            job = GenerateJob(
                sampling or self._engine.sampling_defaults, self._rng,
                max_tokens, self._transcript, hooks=hooks,
                stop_sequences=list(stop_sequences), control=control, resume=resuming,
            )
            control = job.control
            yield from self._job_events(job)
        finally:
            if control is not None and control.stop_sequence is not None:
                self._paused_control = control
            else:
                self._paused_control = None

    def run_round(self, *, max_tokens: int = 50_000,
                  sampling: SamplingPolicy | None = None,
                  hooks: LogitHookSet | None = None,
                  stop_sequences: Sequence[bytes] = ()) -> Iterator[Event]:
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            self._active_round = True
        try:
            if self._transcript.chat:
                yield from chat_ops.run_round(
                    self, max_tokens, sampling, hooks, stop_sequences,
                )
            else:
                yield from self._raw_round(max_tokens, sampling, hooks, stop_sequences)
        finally:
            with self._lock:
                self._active_round = False

    def plan(self, tokens: Sequence[int] | None = None) -> Plan:
        return self._worker.plan(self._transcript.tokens if tokens is None else tokens)

    def checkpoint(self, key: str) -> Checkpointed:
        return list(self._worker.submit(CheckpointJob(key)))[0]

    def drop_checkpoint(self, key: str) -> None:
        list(self._worker.submit(CheckpointJob(key, drop=True)))

    def on_worker(self, fn: Callable[[Handles], T]) -> Future[T]:
        return self._worker.call(lambda e, s: fn(Handles(e.raw, s.raw)))

    def cancel(self) -> None:
        self._worker.cancel_event.set()
