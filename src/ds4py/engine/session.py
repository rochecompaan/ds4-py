"""Public token-oriented engine facade and lazy, exclusively owned rounds."""
import operator
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
from .events import Checkpointed, Event, Interrupted, ReserveReached, Stopped, Synced
from .pause import resolve_control
from .plan import Plan
from .prompt import PromptBuilder
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
    """One exclusively owned transcript; caller mutations occur between rounds.

    Chat helpers dispatch rendering/tokenization to the engine worker. Do not
    call these blocking methods from a logits hook or an on_worker callback.
    """
    def __init__(self, engine: Engine):
        self._engine, self._worker = engine, engine._worker
        self._transcript = Transcript()
        self._transcript_view = TranscriptView(self._transcript)
        self._rng = Rng(engine.sampling_defaults.seed)
        self._lock = threading.Lock()
        self._active_round = False
        self._logits_hooks = []
        self._paused_stream = None
        self._paused_control = None
        self._paused_sampled_tokens = None
        self._checkpoint_tokens: dict[str, int] = {}

    @property
    def engine(self) -> Engine:
        """The facade-owned engine; ownership remains with the caller."""
        return self._engine

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
        """Initialize chat once, between rounds, with tools and optional instructions.

        Rendering runs on the worker; caller-provided payloads stay plain text.
        """
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            chat_ops.start_chat(self, tools, instructions)

    def append_message(self, role: str, text: str, *, images=(),
                       meta: Mapping | None = None) -> Segment:
        """Append one plain-payload chat message with optional provenance metadata."""
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            if images:
                raise EngineError("image input requires M5 vision support")
            chat_ops._start_if_empty(self)
            chat_ops.retire_paused(self)
            tokens = PromptBuilder(self._engine).message(self._transcript.tokens, role, text)
            return self._transcript.append(tokens, role, meta)

    def truncate(self, token_count: int) -> None:
        """Truncate between rounds and invalidate checkpoints beyond the new end."""
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            token_count = operator.index(token_count)
            if not 0 <= token_count <= len(self._transcript):
                raise ValueError("truncate position is outside the transcript")
            for key, length in tuple(self._checkpoint_tokens.items()):
                if length > token_count:
                    self.drop_checkpoint(key)
            self._paused_stream = self._paused_control = self._paused_sampled_tokens = None
            self._transcript.truncate(token_count)

    def append_user(self, text: str, images=()) -> Segment:
        """Append a user turn between rounds, initializing an empty chat if needed.

        A stop-sequence-paused assistant is closed first; the next run_round
        starts a fresh assistant prefix/parser. Nonempty images require M5.
        Rendering/tokenization is dispatched to the worker.
        """
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            return chat_ops.append_user(self, text, images)

    def append_prefix(self, turns) -> None:
        """Append PrefixTurn history between rounds, starting an empty chat if needed.

        Retires a paused assistant and closes each supplied assistant turn using
        the family's EOS policy. Rendering is dispatched to the worker.
        """
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            chat_ops.append_prefix(self, turns)

    def append_tool_results(self, results) -> Segment:
        """Append results joined by two newlines as one tool message in initialized chat.

        Between rounds only; closes a paused assistant before appending. Worker
        rendering escapes the matching tool-result closing delimiter.
        """
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            return chat_ops.append_tool_results(self, results)

    def append_system_note(self, text: str) -> Segment:
        """Append a newline-prefixed note between rounds in initialized chat.

        Closes a paused assistant first and dispatches plain-payload rendering
        to the worker.
        """
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            return chat_ops.append_system_note(self, text)

    def add_logits_hook(self, hook: Callable) -> None:
        """Register a lazy pre-sample hook on the worker-thread decode path.

        Registration is allowed only between rounds. Each invocation receives
        a reusable LogitsView; only committed tokens enter its sampled history.
        The hook must not call engine, session, harness, cancellation, or
        blocking-I/O APIs; those calls can deadlock the native worker.
        """
        with self._lock:
            if self._active_round:
                raise EngineError("cannot add a logits hook during an active round")
            self._logits_hooks.append(hook)

    def _reserve_decision(self, job) -> Iterator[Event]:
        decision = {}
        pause = ReserveReached(
            job.reserve,
            on_resume=lambda reserve: decision.update(
                action="resume", reserve=reserve,
            ),
            on_fail=lambda: decision.update(action="fail"),
        )
        yield pause
        if not decision:
            raise EngineError("reserve pause requires a decision before advancing")
        if decision["action"] == "fail":
            yield Stopped("context_exhausted")
            return False
        return "resume", decision["reserve"]

    def _job_events(self, job, *, pause_on_reserve=False) -> Iterator[Event]:
        # submit may fail before there is an iterator to drain.
        events = self._worker.submit(job)
        exhausted = interrupted = terminal = False
        try:
            for event in events:
                terminal = isinstance(event, (Synced, Interrupted, Stopped))
                interrupted = isinstance(event, Interrupted) or (
                    isinstance(event, Stopped) and event.reason == "interrupted"
                )
                if (pause_on_reserve and isinstance(event, Stopped)
                        and event.reason == "context_reserve"):
                    return (yield from self._reserve_decision(job))
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

    def _raw_round(self, max_tokens, sampling, stop_sequences=()):
        resuming = self._paused_control is not None and self._paused_stream is None
        control = resolve_control(self._paused_control if resuming else None, stop_sequences)
        sampled_tokens = self._paused_sampled_tokens if resuming else []
        assert sampled_tokens is not None
        try:
            tokens = self._transcript.tokens
            if (yield from self._job_events(SyncJob(tokens))):
                return
            job = GenerateJob(
                sampling or self._engine.sampling_defaults, self._rng,
                max_tokens, self._transcript,
                logits_hooks=tuple(self._logits_hooks), control=control, resume=resuming,
                sampled_tokens=sampled_tokens,
            )
            yield from self._job_events(job)
        finally:
            if control is not None and control.stop_sequence is not None:
                self._paused_control = control
                self._paused_sampled_tokens = sampled_tokens
            else:
                self._paused_control = self._paused_sampled_tokens = None

    def run_round(self, *, max_tokens: int = 50_000,
                  sampling: SamplingPolicy | None = None,
                  stop_sequences: Sequence[bytes] = ()) -> Iterator[Event]:
        """Synchronize and generate on the worker; ownership begins at iteration.

        While open, other rounds and caller mutations are refused. A stop
        sequence leaves the raw or chat round paused: another run resumes it
        with unchanged stop_sequences; changed sequences raise ValueError before
        sampling. Appending a new chat message closes the pause first.
        ReserveReached requires resume(reserve) or fail() before advancing.
        Closing the iterator drains native work.
        """
        with self._lock:
            if self._active_round:
                raise EngineError("a round is open")
            self._active_round = True
        try:
            if self._transcript.chat:
                yield from chat_ops.run_round(
                    self, max_tokens, sampling, stop_sequences,
                )
            else:
                yield from self._raw_round(max_tokens, sampling, stop_sequences)
        finally:
            with self._lock:
                self._active_round = False

    def plan(self, tokens: Sequence[int] | None = None) -> Plan:
        return self._worker.plan(self._transcript.tokens if tokens is None else tokens)

    def checkpoint(self, key: str) -> Checkpointed:
        event = list(self._worker.submit(CheckpointJob(key)))[0]
        self._checkpoint_tokens[key] = event.tokens
        return event

    def drop_checkpoint(self, key: str) -> None:
        list(self._worker.submit(CheckpointJob(key, drop=True)))
        self._checkpoint_tokens.pop(key, None)

    def on_worker(self, fn: Callable[[Handles], T]) -> Future[T]:
        return self._worker.call(lambda e, s: fn(Handles(e.raw, s.raw)))

    def cancel(self) -> None:
        self._worker.cancel_event.set()
