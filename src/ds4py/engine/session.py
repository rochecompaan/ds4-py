"""Public token-oriented engine facade and lazy, exclusively owned rounds."""

import operator
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Self, cast

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
    from concurrent.futures import Future
    from contextlib import AbstractContextManager
from uuid import uuid4

from ds4py._native.api import LibNative
from ds4py._native.loader import load_library
from ds4py._native.protocol import EngineConfig, Native, ThinkMode
from ds4py.errors import EngineError

from . import chat as chat_ops
from . import rebind, session_operations
from .compaction import SummaryJob, SummaryResult, summary_suffix
from .decode import DecodeStep, GenerateJob
from .events import (
    Checkpointed,
    Event,
    Interrupted,
    Loaded,
    ReserveReached,
    Saved,
    Stopped,
    Synced,
)
from .lifecycle import SessionEvent, SessionLifecycle, in_listener
from .pause import ControlSpan, resolve_control
from .reserve import ReserveAuthority, ReserveOperation, ReserveOperations, legacy_context

if TYPE_CHECKING:
    from .logits import LogitsView
    from .plan import Plan
    from .prefix import PrefixTurn
    from .stream import RoundStream
    from .tool_prompt import ToolDefinition
from .prompt import PromptBuilder
from .sampling import Rng, SamplingPolicy
from .transcript import Segment, Transcript, TranscriptReplacement, TranscriptView, validate_segment_coverage
from .worker import (
    CheckpointJob,
    EngineWorker,
    Handles,
    Job,
    LoadPayloadJob,
    ResetJob,
    SavePayloadJob,
    SyncJob,
)


def open(  # noqa: PLR0913 - Engine construction mirrors the public CLI options.
    model: str | Path = "ds4flash.gguf",
    *,
    ctx: int = 100_000,
    backend: str | None = None,
    think: int = ThinkMode.HIGH,
    seed: int | None = None,
    threads: int = 0,
    mtp_path: str | Path | None = None,
    mtp_draft_tokens: int = 1,
    lib: str | Path | None = None,
    native: Native | None = None,
    cancel: threading.Event | None = None,
) -> Engine:
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
            f"this libds4 was built for {native.backend}, not {backend}; run ds4py build --backend {backend}"
        )
    config = EngineConfig(
        Path(model),
        ctx=ctx,
        backend=native.backend,
        threads=threads,
        mtp_path=Path(mtp_path) if mtp_path is not None else None,
        mtp_draft_tokens=mtp_draft_tokens,
    )
    worker = EngineWorker(native, config, cancel=cancel)
    try:
        think = worker.call(lambda e, s: native.think_mode_for_context(think, ctx)).result()
        return Engine(
            native,
            worker,
            config,
            think,
            SamplingPolicy.for_engine(worker.info, seed=seed),
        )
    except BaseException:
        worker.close()
        raise


class Engine:
    def __init__(
        self,
        native: Native,
        worker: EngineWorker,
        config: EngineConfig,
        think: int,
        sampling_defaults: SamplingPolicy,
    ) -> None:
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

    def session(self) -> Session:
        with self._session_lock:
            if self._session is not None:
                raise EngineError("one session per engine")
            self._session = Session(self)
            return self._session

    def close(self) -> None:
        with self._session_lock:
            session = self._session
        try:
            if session is not None:
                session.close()
        finally:
            self._worker.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Session:
    """One exclusively owned transcript; caller mutations occur between rounds.

    Chat helpers dispatch rendering/tokenization to the engine worker. Do not
    call these blocking methods from a logits hook or an on_worker callback.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine, self._worker = engine, engine._worker
        self._transcript = Transcript()
        self._transcript_view = TranscriptView(self._transcript)
        self._rng = Rng(engine.sampling_defaults.seed)
        self._lock = threading.RLock()
        self._active_round = False
        self._operations = session_operations.OperationState()
        self._reserve_event: ReserveReached | None = None
        self._reserve_operations: ReserveOperations | None = None
        self._decode_step: DecodeStep | None = None
        self._logits_hooks: list[Callable[[LogitsView], None]] = []
        self._paused_stream: RoundStream | None = None
        self._paused_control: ControlSpan | None = None
        self._paused_sampled_tokens: list[int] | None = None
        self._checkpoint_tokens: dict[str, int] = {}
        self._lifecycle = SessionLifecycle()
        self._closed = False

    @property
    def id(self) -> str:
        """The live session identity; a fork or switch replaces it."""
        return self._lifecycle.id

    @property
    def engine(self) -> Engine:
        """The facade-owned engine; ownership remains with the caller."""
        return self._engine

    @property
    def transcript(self) -> TranscriptView:
        """Read-only history; append_tokens is the guarded caller mutation path."""
        return self._transcript_view

    @property
    def paused(self) -> bool:
        """True while a stop-sequence-paused assistant may be retired."""
        return self._paused_stream is not None

    @property
    def ctx_used(self) -> int:
        return self._worker.call(lambda e, s: s.pos()).result()

    def on_event(
        self,
        listener: Callable[[SessionEvent], None],
        *,
        kinds: Iterable[str] | None = None,
    ) -> Callable[[], None]:
        """Subscribe on the consuming thread; the returned call unsubscribes.

        Listeners must not call blocking session methods. ``kinds`` filters by
        event kind, including engine class names such as ``"Prefill"``.
        """
        return self._lifecycle.on_event(listener, kinds=kinds)

    def emit(self, kind: str, **payload: object) -> None:
        """Publish an extension/status event on this session's bus."""
        self._lifecycle.emit(kind, **payload)

    def events_on_loop(self, loop: AbstractEventLoop) -> AbstractContextManager[None]:
        """Scope all session-bus delivery to one asyncio loop's owning thread."""
        return self._lifecycle.deliver_on(loop)

    def state[T](self, key: str, factory: Callable[[], T]) -> T:
        """Return the session-owned state for ``key``, creating it once."""
        self._require_open()
        return self._lifecycle.state(key, factory)

    def new(self) -> str:
        """End the old session, drop scoped state, and reset history/KV."""
        with self._lock:
            self._require_idle()
            old = self.id
            self._lifecycle.emit("session_end", old=old)
            self._lifecycle.close_states()
            self._reset()
            new = self._lifecycle.set_id()
            self._lifecycle.emit("session_start", old=old, new=new)
            return new

    def switch(
        self,
        new_id: str,
        restore: Callable[[], Mapping[str, object] | None] | None = None,
        **payload: object,
    ) -> str:
        """Change identity across a restore boundary, closing old scoped state.

        ``restore`` runs between ``session_end`` and ``session_start`` so a
        runtime can load history/KV without a second identity or state set.
        Its returned mapping can refine the final ``switched`` payload.
        """
        with self._lock:
            self._require_idle()
            old = self.id
            self._lifecycle.emit("switching", old=old, new=new_id)
            self._lifecycle.emit("session_end", old=old)
            self._lifecycle.close_states()
            try:
                restored = restore() if restore is not None else None
            except BaseException as error:
                # Restore can reject a snapshot after old scoped state is closed.
                # Balance the lifecycle and adopt the target identity, then
                # preserve the original failure for the caller.
                self._lifecycle.set_id(new_id)
                self._lifecycle.emit(
                    "session_start",
                    old=old,
                    new=new_id,
                    status="error",
                    error=str(error),
                )
                raise
            self._lifecycle.set_id(new_id)
            self._lifecycle.emit("session_start", old=old, new=new_id)
            self._lifecycle.emit(
                "switched",
                **{"old": old, "new": new_id, **payload, **(restored or {})},
            )
            return new_id

    def fork(self) -> str:
        """Change only identity; transcript, KV, checkpoints, and state survive."""
        with self._lock:
            self._require_idle()
            old = self.id
            new = uuid4().hex
            self._lifecycle.emit("forked", old=old, new=new)
            self._lifecycle.set_id(new)
            return new

    def set_think(self, mode: int) -> None:
        """Set the think mode between rounds; bound prefixes rebuild next request."""
        with self._lock:
            self._require_idle()
            normalized = self._worker.call(
                lambda e, s: self._engine.native.think_mode_for_context(
                    mode,
                    self._engine.config.ctx,
                )
            ).result()
            self._engine.think = normalized

    def set_decode_step(self, step: DecodeStep | None) -> None:
        """Install the decode step between rounds; ``None`` restores the default."""
        with self._lock:
            self._require_idle()
            if self._paused_stream is not None or self._paused_control is not None:
                raise EngineError("cannot change decode step during a paused round")
            self._decode_step = step

    def close(self) -> None:
        """End the live session and close all state; safe to call repeatedly."""
        with self._lock:
            if self._closed:
                return
            reason = self._blocked_reason("close")
            if reason is not None:
                raise EngineError(reason)
            self._revoke_reserve()
            self._reserve_event = None
            self._closed = True
            self._decode_step = None
            self._lifecycle.emit("session_end", old=self.id)
            self._lifecycle.close_states()

    def _blocked_reason(
        self,
        policy: session_operations.GuardPolicy,
        *,
        authority: ReserveAuthority | None = None,
        binding: session_operations.BindingTransaction | None = None,
    ) -> str | None:
        return session_operations.blocked_reason(self, policy, authority=authority, binding=binding)

    def _exclusive(
        self,
        kind: session_operations.OperationKind,
        *,
        authority: ReserveAuthority | None = None,
    ) -> AbstractContextManager[session_operations.Claim]:
        return session_operations.exclusive(self, kind, authority=authority)

    def _require_open(self) -> None:
        reason = self._blocked_reason("open")
        if reason is not None:
            raise EngineError(reason)

    def _require_idle(self) -> None:
        reason = self._blocked_reason("idle")
        if reason is not None:
            raise EngineError(reason)

    def _reset(self) -> None:
        for _ in self._job_events(ResetJob()):
            pass
        self._checkpoint_tokens.clear()
        self._paused_stream = self._paused_control = self._paused_sampled_tokens = None
        self._transcript.reset()

    def append_tokens(self, tokens: Sequence[int], kind: str, meta: Mapping[str, object] | None = None) -> Segment:
        with self._lock:
            self._require_idle()
            return self._transcript.append(tokens, kind, meta)

    def start_chat(
        self,
        tools: Iterable[ToolDefinition] = (),
        instructions: str | None = None,
    ) -> None:
        """Initialize chat once, between rounds, with tools and optional instructions.

        Rendering runs on the worker; caller-provided payloads stay plain text.
        """
        with self._lock:
            self._require_idle()
            chat_ops.start_chat(self, tools, instructions)

    def append_message(
        self,
        role: str,
        text: str,
        *,
        images: Sequence[object] = (),
        meta: Mapping[str, object] | None = None,
    ) -> Segment:
        """Append one plain-payload chat message with optional provenance metadata."""
        with self._lock:
            self._require_idle()
            if images:
                raise EngineError("image input requires M5 vision support")
            chat_ops._start_if_empty(self)
            chat_ops.retire_paused(self)
            tokens = PromptBuilder(self._engine).message(self._transcript.tokens, role, text)
            return self._transcript.append(tokens, role, meta)

    def truncate(self, token_count: int) -> None:
        """Truncate through the same claimed pipeline as binding preparation."""
        with self.binding_transaction() as transaction:
            transaction.truncate(token_count)

    def _truncate_position(self, token_count: int) -> int:
        position = operator.index(token_count)
        if not 0 <= position <= len(self._transcript):
            raise ValueError("truncate position is outside the transcript")
        return position

    def _stale_checkpoints(self, position: int) -> tuple[str, ...]:
        return tuple(key for key, length in self._checkpoint_tokens.items() if length > position)

    def _truncate_python(self, position: int) -> None:
        """Commit one truncation of history, pause state and stale checkpoint keys.

        Native checkpoint disposal is not part of this step: the binding
        transaction dispatches those drops with only its claim held, so event
        delivery never waits on the mutation lock.
        """
        for key in self._stale_checkpoints(position):
            del self._checkpoint_tokens[key]
        self._paused_stream = self._paused_control = self._paused_sampled_tokens = None
        self._transcript.truncate(position)

    def apply_appends(self, replacement: TranscriptReplacement, *, retire_paused: bool = False) -> None:
        """Append one validated group of segments atomically between rounds.

        Python-only: appends never synchronize native state. Kinds, coverage and
        resulting offsets are checked before one publication, so invalid input
        changes neither history, chat flag nor Session-owned pause state.
        """
        with self._lock:
            self._commit_appends(replacement, retire_paused=retire_paused)

    @contextmanager
    def binding_transaction(self) -> Generator[session_operations.BindingTransaction]:
        """Claim all binding boundaries of one real preparation plan.

        The claim begins before planning and excludes every other round,
        mutation or close until it is released in ``finally``. Native preamble
        work and native checkpoint disposal release the mutation lock but never
        the claim.
        """
        with self._exclusive("binding") as claim:
            transaction = session_operations.BindingTransaction(self, claim)
            try:
                yield transaction
            finally:
                transaction.close()

    def _commit_appends(
        self,
        replacement: TranscriptReplacement,
        *,
        retire_paused: bool = False,
        publish: Callable[[], None] | None = None,
        binding: session_operations.BindingTransaction | None = None,
    ) -> None:
        # The lock is held; admission is granted here so every caller shares it.
        reason = self._blocked_reason("idle", binding=binding)
        if reason is not None:
            raise EngineError(reason)
        self._transcript.append_group(replacement)
        paused = (self._paused_stream, self._paused_control, self._paused_sampled_tokens)
        if retire_paused:
            self._paused_stream = self._paused_control = self._paused_sampled_tokens = None
        try:
            if publish is not None:
                publish()
        except BaseException:
            self._transcript.rollback(len(self._transcript) - len(replacement.tokens))
            self._paused_stream, self._paused_control, self._paused_sampled_tokens = paused
            raise

    def install_segments(self, segments: Sequence[Segment]) -> None:
        """Install saved provenance over tokens an already-loaded payload set.

        Between rounds only. The saved ranges must describe the loaded tokens
        without gaps or overlap, so invalid input leaves the transcript intact.
        """
        with self._lock:
            self._require_idle()
            self._transcript.install(segments)

    def replace_transcript(
        self, replacement: TranscriptReplacement, *, reason: str, authority: ReserveAuthority | None = None
    ) -> Plan:
        """Replace history between rounds or at the explicit reserve decision.

        Segments must cover the tokens without gaps or overlap, so invalid input
        changes neither native state nor history. The worker restores the longest
        valid checkpoint or rebuilds from token zero; only a completed sync
        commits the new tokens and provenance. Checkpoints beyond the new end and
        a paused round are discarded.

        ``authority`` is an opaque one-shot replacement ticket issued on the
        selected reserve owner's loop by ``ReserveOperations.issue_authority``.
        Strict paused callers require it; idle callers leave it as None. A
        supplied invalid, stale or listener-delivered ticket always rejects.
        """
        with self._exclusive("replacement", authority=authority):
            # Event delivery can synchronously wait for the loop. Never retain
            # the mutation lock across worker emissions; the claim rejects writes.
            plan = rebind.replace_transcript(self, replacement, reason=reason)
            with self._lock:
                self._require_open()
                self._transcript.replace(replacement.tokens, replacement.segments)
                self._paused_stream = self._paused_control = self._paused_sampled_tokens = None
                return plan

    def _reserve_authorized(self, authority: ReserveAuthority | None, operation: ReserveOperation) -> bool:
        operations = self._reserve_operations
        if self._reserve_event is None or operations is None:
            return False
        if authority is not None:
            return operations.consume(authority, operation)
        return operations.legacy_allows()

    def _revoke_reserve(self) -> None:
        if self._reserve_operations is not None:
            self._reserve_operations.revoke()

    @contextmanager
    def reserve_window(self, event: ReserveReached, *, strict: bool = False) -> Generator[ReserveOperations]:
        """Select explicit operations, or the legacy naked-to_thread adapter.

        Legacy child executors remain ambiguous; this is not security isolation.
        """
        with self._lock:
            self._require_open()
            if (
                event is not self._reserve_event
                or in_listener.get()
                or self._reserve_operations is not None
                or self._operations.kind is not None
            ):
                raise EngineError("reserve window requires an unclaimed live pause")
            operations = ReserveOperations(self, strict=strict)
            self._reserve_operations = operations
        try:
            with legacy_context(operations, enabled=not strict):
                yield operations
        finally:
            with self._lock:
                operations.revoke()
                if self._reserve_operations is operations:
                    self._reserve_operations = None

    def open_assistant_start(self) -> int | None:
        """Observe the sampled start of the trailing unkeyed assistant structure."""
        with self._lock:
            tokens, segments = self._transcript.tokens, self._transcript.segments
            try:
                validate_segment_coverage(segments, len(tokens))
            except ValueError as error:
                raise EngineError(f"invalid open assistant structure: {error}") from error
            tail = list(segments)
            # A supplied EOS and the round's own close can both be present.
            while tail and tail[-1].kind == "eos" and not tail[-1].meta.get("key"):
                tail.pop()
            if tail and tail[-1].kind == "assistant" and not tail[-1].meta.get("key"):
                tail.pop()
            if tail and tail[-1].kind == "assistant_prefix" and not tail[-1].meta.get("key"):
                return tail[-1].end
            if segments and segments[-1].kind in {"assistant", "eos"} and not segments[-1].meta.get("key"):
                raise EngineError("invalid open assistant structure: missing unkeyed prefix")
            return None

    def prepare_summary(self, bottom: int, *, authority: ReserveAuthority | None = None) -> SummaryResult:
        """Summarize only the caller's token prefix without changing live history.

        Runs between rounds (including reserve-paused continuations). Private
        output emits SummaryToken, never response Token/TextDelta events.

        ``authority`` is an opaque one-shot summary ticket issued on the
        selected reserve owner's loop by ``ReserveOperations.issue_authority``.
        Strict paused callers require it; idle callers leave it as None. A
        supplied invalid, stale or listener-delivered ticket always rejects.
        """
        with self._exclusive("summary", authority=authority):
            bottom = operator.index(bottom)
            if not 0 <= bottom <= len(self._transcript):
                raise ValueError("summary position is outside the transcript")
            prefix = self._transcript.tokens[:bottom]
            job = SummaryJob(prefix + summary_suffix(self._engine, prefix))
            for _ in self._job_events(job):
                pass
            return job.result.result()

    def reconcile_round(self, token_count: int) -> None:
        """Rewind native state and transcript to one closed round's sampled end.

        The response owner calls this after closing and draining its facade
        iterator. Cancellation cannot still be observed by a native job here.
        """
        with self.binding_transaction() as transaction:
            transaction.truncate(token_count)
            self._worker.call(lambda engine, native: native.rewind(token_count)).result()
            self.clear_cancellation()

    def append_user(self, text: str, images: Sequence[object] = ()) -> Segment:
        """Append a user turn between rounds, initializing an empty chat if needed.

        A stop-sequence-paused assistant is closed first; the next run_round
        starts a fresh assistant prefix/parser. Nonempty images require M5.
        Rendering/tokenization is dispatched to the worker.
        """
        with self._lock:
            self._require_idle()
            return chat_ops.append_user(self, text, images)

    def append_prefix(self, turns: Iterable[PrefixTurn]) -> None:
        """Append PrefixTurn history between rounds, starting an empty chat if needed.

        Retires a paused assistant and closes each supplied assistant turn using
        the family's EOS policy. Rendering is dispatched to the worker.
        """
        with self._lock:
            self._require_idle()
            chat_ops.append_prefix(self, turns)

    def append_tool_results(self, results: Iterable[str]) -> Segment:
        """Append results joined by two newlines as one tool message in initialized chat.

        Between rounds only; closes a paused assistant before appending. Worker
        rendering escapes the matching tool-result closing delimiter.
        """
        with self._lock:
            self._require_idle()
            return chat_ops.append_tool_results(self, results)

    def append_system_note(self, text: str, *, subtype: str | None = None) -> Segment:
        """Append a newline-prefixed note between rounds in initialized chat.

        Closes a paused assistant first and dispatches plain-payload rendering
        to the worker. ``subtype`` becomes save provenance metadata.
        """
        with self._lock:
            self._require_idle()
            return chat_ops.append_system_note(self, text, subtype)

    def add_logits_hook(self, hook: Callable[[LogitsView], None]) -> None:
        """Register a lazy pre-sample hook on the worker-thread decode path.

        Registration is allowed only between rounds. Each invocation receives
        a reusable LogitsView whose sampled history includes any accepted
        staging prefix, before that prefix is published to the caller.
        The hook must not call engine, session, harness, cancellation, or
        blocking-I/O APIs; those calls can deadlock the native worker.
        """
        with self._lock:
            self._require_open()
            if self._blocked_reason("idle") is not None:
                raise EngineError("cannot add a logits hook during an active round")
            self._logits_hooks.append(hook)

    def _publish(self, event: Event) -> None:
        self._lifecycle.emit(type(event).__name__, event=event)

    def _reserve_decision(
        self,
        job: GenerateJob,
    ) -> Generator[Event, None, bool | tuple[str, int]]:
        decision: dict[str, object] = {}

        def decide(action: str, reserve: int = 0) -> None:
            with self._lock:
                if self._reserve_event is not pause:
                    raise EngineError("reserve pause has expired")
                self._revoke_reserve()
                self._reserve_event = None
                decision.update(action=action, reserve=reserve)

        pause = ReserveReached(
            job.reserve,
            on_resume=lambda reserve: decide("resume", reserve),
            on_fail=lambda: decide("fail"),
        )
        self._reserve_event = pause
        try:
            self._publish(pause)
            yield pause
        finally:
            with self._lock:
                self._revoke_reserve()
                self._reserve_event = None
        if not decision:
            raise EngineError("reserve pause requires a decision before advancing")
        if decision["action"] == "fail":
            stopped = Stopped("context_exhausted")
            self._publish(stopped)
            yield stopped
            return False
        return "resume", cast("int", decision["reserve"])

    def _job_events(
        self,
        job: Job,
        *,
        pause_on_reserve: bool = False,
    ) -> Generator[Event, None, bool | tuple[str, int] | None]:
        # submit may fail before there is an iterator to drain.
        events = self._worker.submit(job)
        exhausted = interrupted = terminal = False
        published: Event | None = None
        try:
            for event in events:
                if isinstance(job, (SavePayloadJob, LoadPayloadJob)):
                    terminal = isinstance(event, (Saved, Loaded, Interrupted, Stopped))
                else:
                    terminal = isinstance(event, (Synced, Interrupted, Stopped))
                interrupted = isinstance(event, Interrupted) or (
                    isinstance(event, Stopped) and event.reason == "interrupted"
                )
                if pause_on_reserve and isinstance(event, Stopped) and event.reason == "context_reserve":
                    return (yield from self._reserve_decision(cast("GenerateJob", job)))
                self._publish(event)
                published = event
                yield event
            exhausted = True
        finally:
            if not exhausted:  # ty: ignore[redundant-condition-strict]  # Early exits bypass the normal assignment.
                # A published terminal event means native work has already ended.
                if not terminal:
                    self.cancel()
                try:
                    # The barrier survives interruption of __next__, including
                    # another Ctrl+C during cleanup. Never release a live job.
                    events.wait()
                    for event in events:
                        if event is not published:
                            self._publish(event)
                        if isinstance(event, Interrupted) or (
                            isinstance(event, Stopped) and event.reason == "interrupted"
                        ):
                            interrupted = True
                finally:
                    if not interrupted:
                        self.clear_cancellation()
        return interrupted

    def _raw_round(
        self,
        max_tokens: int,
        sampling: SamplingPolicy | None,
        stop_sequences: Sequence[bytes] = (),
    ) -> Generator[Event]:
        resuming = self._paused_control is not None and self._paused_stream is None
        control = resolve_control(self._paused_control if resuming else None, stop_sequences)
        sampled_tokens = self._paused_sampled_tokens if resuming else []
        if sampled_tokens is None:
            raise AssertionError("paused raw round requires sampled tokens")
        try:
            tokens = self._transcript.tokens
            if bool((yield from self._job_events(SyncJob(tokens)))):
                return
            job = GenerateJob(
                sampling or self._engine.sampling_defaults,
                self._rng,
                max_tokens,
                self._transcript,
                logits_hooks=tuple(self._logits_hooks),
                control=control,
                resume=resuming,
                sampled_tokens=sampled_tokens,
                decode_step=self._decode_step,
            )
            yield from self._job_events(job)
        finally:
            if control.stop_sequence is not None:
                self._paused_control = control
                self._paused_sampled_tokens = sampled_tokens
            else:
                self._paused_control = self._paused_sampled_tokens = None

    def run_round(
        self,
        *,
        max_tokens: int = 50_000,
        sampling: SamplingPolicy | None = None,
        stop_sequences: Sequence[bytes] = (),
    ) -> Generator[Event]:
        """Synchronize and generate on the worker; ownership begins at iteration.

        While open, other rounds and caller mutations are refused. A stop
        sequence leaves the raw or chat round paused: another run resumes it
        with unchanged stop_sequences; changed sequences raise ValueError before
        sampling. Appending a new chat message closes the pause first.
        ReserveReached requires resume(reserve) or fail() before advancing.
        Closing the iterator drains native work.
        """
        with self._lock:
            self._require_idle()
            self._active_round = True
        try:
            if self._transcript.chat:
                yield from chat_ops.run_round(
                    self,
                    max_tokens,
                    sampling,
                    stop_sequences,
                )
            else:
                yield from self._raw_round(max_tokens, sampling, stop_sequences)
        finally:
            with self._lock:
                self._active_round = False

    def plan(self, tokens: Sequence[int] | None = None) -> Plan:
        return self._worker.plan(self._transcript.tokens if tokens is None else tokens)

    def checkpoint(self, key: str) -> Checkpointed:
        with self._lock:
            self._require_idle()
            return self._checkpoint_native(key)

    def _checkpoint_native(self, key: str) -> Checkpointed:
        # Admission is already granted. The binding transaction runs this with
        # only its claim held, so delivery never waits on this thread's lock.
        event = next(iter(self._job_events(CheckpointJob(key))))
        if not isinstance(event, Checkpointed):
            raise AssertionError(  # noqa: TRY004 - Preserve the impossible-event assertion contract.
                "checkpoint job returned a non-checkpoint event"
            )
        self._checkpoint_tokens[key] = event.tokens
        return event

    def drop_checkpoint(self, key: str) -> None:
        with self._lock:
            self._require_idle()
            self._drop_checkpoint_locked(key)

    def _drop_checkpoint_locked(self, key: str) -> None:
        self._drop_checkpoint(key)
        self._checkpoint_tokens.pop(key, None)

    def _drop_checkpoint(self, key: str) -> None:
        # Admission is already granted; the caller may hold the mutation lock or
        # only its operation claim, so a delivered event never waits on it.
        list(self._job_events(CheckpointJob(key, drop=True)))

    def save_payload(self, file: BinaryIO, *, header: bytes) -> Saved:
        """Sync the transcript and write the header plus staged native payload.

        Call on the consuming thread between rounds. The caller owns ``file``;
        only a duplicated descriptor reaches native code. A failed sync or write
        leaves the Python file usable.
        """
        with self._lock:
            self._require_idle()
            saved = None
            for event in self._job_events(
                SavePayloadJob(self._transcript.tokens, header, file),
            ):
                if isinstance(event, Saved):
                    saved = event
            if saved is None:
                raise EngineError("save did not complete")
            return saved

    def load_payload(self, file: BinaryIO, *, payload_bytes: int) -> list[int]:
        """Load a payload after the caller's header and adopt its token list.

        Call on the consuming thread between rounds. The loaded KV becomes the
        live checkpoint; stale snapshots and paused streams from the previous
        transcript are discarded. No tokenization or prefill occurs during load.
        """
        payload_bytes = operator.index(payload_bytes)
        if payload_bytes < 0:
            raise ValueError("payload byte count cannot be negative")
        with self._lock:
            self._require_idle()
            loaded = None
            for event in self._job_events(
                LoadPayloadJob(file, payload_bytes, self._transcript),
            ):
                if isinstance(event, Loaded):
                    loaded = event
            if loaded is None:
                raise EngineError("load did not complete")
            self._checkpoint_tokens.clear()
            self._paused_stream = self._paused_control = self._paused_sampled_tokens = None
            return list(self._transcript.tokens)

    def on_worker[T](self, fn: Callable[[Handles], T]) -> Future[T]:
        """Run ``fn`` on the engine worker and return a future for its result.

        The handles are borrowed and confined to the callback: do not retain
        them or use them on another thread. Blocking session or engine calls
        from the callback can deadlock the worker.
        """
        self._require_open()
        return self._worker.call(lambda e, s: fn(Handles(e.raw, s.raw)))

    def cancel(self) -> None:
        """Request worker cancellation from any thread without closing the round."""
        self._worker.cancel_event.set()

    def clear_cancellation(self) -> None:
        """Release the worker cancellation signal once its work has drained.

        Call only after retained native/summary/replacement work has finished;
        no operation may still be waiting to observe the signal.
        """
        self._worker.cancel_event.clear()

    def poison(self, error: BaseException) -> None:
        """Fail the worker closed after unrecoverable native cleanup.

        Dispatches once through the worker queue from the consuming thread. A
        worker callback must never call this, because it would wait on itself.
        """
        self._worker.call(lambda engine, native: self._worker._poison(error)).result()
