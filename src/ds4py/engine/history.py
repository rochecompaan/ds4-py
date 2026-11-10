"""Session-owned transcript history, pause/checkpoint state, and claimed mutations.

``Session`` publishes events and submits native jobs; this module owns everything
history mutations touch. Callers either take the mutation lock for one atomic
edit or an explicit claim that excludes every other writer while native work
runs without the lock.

The facade reach-back is deliberately narrow: only ``Session._job_events`` and
``Session.engine`` are used, for native jobs and rendering. Those calls are the
sanctioned job/event boundary; history state stays in this owner.
"""

from __future__ import annotations

import operator
from contextlib import contextmanager
from threading import RLock
from typing import TYPE_CHECKING

from ds4py.errors import EngineError

from . import rebind, session_operations
from .events import Checkpointed, Interrupted
from .prompt import PromptBuilder
from .transcript import Transcript, TranscriptReplacement, TranscriptView
from .worker import SYSTEM, CheckpointJob, SyncJob

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

    from .pause import ControlSpan
    from .plan import Plan
    from .reserve import ReserveAuthority
    from .session import Session
    from .stream import RoundStream
    from .tool_prompt import ToolDefinition
    from .transcript import Segment


class SessionHistory:
    """The transcript, pause/checkpoint state, and exclusive claim of one Session."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._transcript = Transcript()
        self._view = TranscriptView(self._transcript)
        self._lock = RLock()
        self._operations = session_operations.OperationState()
        self._paused_stream: RoundStream | None = None
        self._paused_control: ControlSpan | None = None
        self._paused_sampled_tokens: list[int] | None = None
        self._checkpoints: dict[str, int] = {}

    @property
    def transcript(self) -> TranscriptView:
        """Read-only history; mutation methods on this owner write it."""
        return self._view

    @property
    def mutable_transcript(self) -> Transcript:
        """The storage native jobs mutate directly; only those jobs need this."""
        return self._transcript

    @property
    def paused(self) -> bool:
        return self._paused_stream is not None

    @property
    def paused_stream(self) -> RoundStream | None:
        return self._paused_stream

    @property
    def paused_control(self) -> ControlSpan | None:
        return self._paused_control

    @property
    def paused_sampled_tokens(self) -> list[int] | None:
        return self._paused_sampled_tokens

    @property
    def checkpoint_tokens(self) -> dict[str, int]:
        """A snapshot of the checkpoint keys Python has observed."""
        with self._lock:
            return dict(self._checkpoints)

    @property
    def operations(self) -> session_operations.OperationState:
        """The owner's admission state; mutations go through its transitions."""
        return self._operations

    @property
    def mutation_lock(self) -> RLock:
        """The lock that excludes every history writer."""
        return self._lock

    # -- admission ---------------------------------------------------------

    def _blocked_reason(
        self,
        policy: session_operations.GuardPolicy,
        *,
        authority: ReserveAuthority | None = None,
        binding: session_operations.BindingLiveness | None = None,
    ) -> str | None:
        return session_operations.blocked_reason(self._operations, policy, authority=authority, binding=binding)

    def require_open(self) -> None:
        reason = self._blocked_reason("open")
        if reason is not None:
            raise EngineError(reason)

    def require_idle(self) -> None:
        reason = self._blocked_reason("idle")
        if reason is not None:
            raise EngineError(reason)

    @contextmanager
    def edit(self) -> Generator[None]:
        """Hold the mutation lock for one caller edit between rounds."""
        with self._lock:
            self.require_idle()
            yield

    @contextmanager
    def claim(
        self,
        kind: session_operations.OperationKind,
        *,
        authority: ReserveAuthority | None = None,
    ) -> Generator[session_operations.Claim]:
        """Admit and claim one exclusive operation; the lock only guards the claim."""
        with session_operations.exclusive(self._operations, self._lock, kind, authority=authority) as claim:
            yield claim

    def begin_round(self) -> None:
        with self._lock:
            self.require_idle()
            self._operations.begin_round()

    def end_round(self) -> None:
        with self._lock:
            self._operations.end_round()

    def close(self) -> bool:
        """Close admission once; return False when the session was already closed."""
        with self._lock:
            if self._operations.closed:
                return False
            reason = self._blocked_reason("close")
            if reason is not None:
                raise EngineError(reason)
            self._operations.close()
            return True

    # -- pause state -------------------------------------------------------

    def pause_stream(self, stream: RoundStream) -> None:
        with self._lock:
            self._paused_stream = stream
            self._paused_control = stream.control

    def clear_stream_pause(self) -> None:
        with self._lock:
            self._paused_stream = self._paused_control = None

    def pause_raw(self, control: ControlSpan | None, sampled_tokens: list[int] | None) -> None:
        with self._lock:
            self._paused_control = control
            self._paused_sampled_tokens = sampled_tokens

    def clear_pause(self) -> None:
        with self._lock:
            self._clear_pause()

    def _clear_pause(self) -> None:
        self._paused_stream = self._paused_control = self._paused_sampled_tokens = None

    # -- checkpoints -------------------------------------------------------

    def checkpoint(self, key: str) -> Checkpointed:
        """Snapshot native state under a claim, never the mutation lock."""
        with self.claim("checkpoint"):
            return self._checkpoint_native(key)

    def _checkpoint_native(self, key: str) -> Checkpointed:
        event = next(iter(self._session._job_events(CheckpointJob(key))))
        if not isinstance(event, Checkpointed):
            raise AssertionError(  # noqa: TRY004 - Preserve the impossible-event assertion contract.
                "checkpoint job returned a non-checkpoint event"
            )
        with self._lock:
            self._checkpoints[key] = event.tokens
        return event

    def drop_checkpoint(self, key: str) -> None:
        """Dispose one native checkpoint under a claim, then forget its key."""
        with self.claim("checkpoint"):
            self.dispose_checkpoint(key)
            self.forget_checkpoint(key)

    def dispose_checkpoint(self, key: str) -> None:
        """Dispatch native disposal; callers already hold the exclusive claim."""
        list(self._session._job_events(CheckpointJob(key, drop=True)))

    def forget_checkpoint(self, key: str) -> None:
        with self._lock:
            self._checkpoints.pop(key, None)

    def clear_checkpoints(self) -> None:
        with self._lock:
            self._checkpoints.clear()

    def stale_checkpoints(self, position: int) -> tuple[str, ...]:
        with self._lock:
            return tuple(key for key, length in self._checkpoints.items() if length > position)

    # -- transcript mutations ---------------------------------------------

    def annotate(self, start: int, *, kinds: tuple[str, ...], meta: Mapping[str, object]) -> None:
        """Merge provenance into existing segments under the mutation lock."""
        with self._lock:
            self._transcript.annotate(start, kinds=kinds, meta=meta)

    def append_segment(self, tokens: Sequence[int], kind: str, meta: Mapping[str, object] | None = None) -> Segment:
        """Append during an owned round or an already admitted edit."""
        with self._lock:
            return self._transcript.append(tokens, kind, meta)

    def append_tokens(self, tokens: Sequence[int], kind: str, meta: Mapping[str, object] | None = None) -> Segment:
        """Append one caller segment between rounds."""
        with self.edit():
            return self._transcript.append(tokens, kind, meta)

    def apply_appends(self, replacement: TranscriptReplacement, *, retire_paused: bool = False) -> None:
        """Append one validated group atomically between rounds."""
        with self._lock:
            self._commit_appends(replacement, retire_paused=retire_paused)

    def _commit_appends(
        self,
        replacement: TranscriptReplacement,
        *,
        retire_paused: bool = False,
        publish: Callable[[], None] | None = None,
        binding: session_operations.BindingLiveness | None = None,
    ) -> None:
        # The lock is held; admission is granted here so every caller shares it.
        reason = self._blocked_reason("idle", binding=binding)
        if reason is not None:
            raise EngineError(reason)
        self._transcript.append_group(replacement)
        paused = (self._paused_stream, self._paused_control, self._paused_sampled_tokens)
        if retire_paused:
            self._clear_pause()
        try:
            if publish is not None:
                publish()
        except BaseException:
            self._transcript.rollback(len(self._transcript) - len(replacement.tokens))
            self._paused_stream, self._paused_control, self._paused_sampled_tokens = paused
            raise

    def install(self, segments: Sequence[Segment]) -> None:
        """Install saved provenance over an already-loaded payload between rounds."""
        with self.edit():
            self._transcript.install(segments)

    def reset(self) -> None:
        """Drop history, pause state, and checkpoint keys after a native reset."""
        with self._lock:
            self._checkpoints.clear()
            self._clear_pause()
            self._transcript.reset()

    def rollback(self, length: int) -> None:
        with self._lock:
            self._transcript.rollback(length)

    def truncate(self, token_count: int) -> None:
        """Truncate through the same claimed pipeline as binding preparation."""
        with self.binding_transaction() as transaction:
            transaction.truncate(token_count)

    def truncate_position(self, token_count: int) -> int:
        position = operator.index(token_count)
        if not 0 <= position <= len(self._transcript):
            raise ValueError("truncate position is outside the transcript")
        return position

    def commit_truncation(self, position: int) -> None:
        """Commit one locked truncation; native disposal is the transaction's job."""
        with self._lock:
            for key in self.stale_checkpoints(position):
                del self._checkpoints[key]
            self._clear_pause()
            self._transcript.truncate(position)

    def replace_transcript(
        self,
        replacement: TranscriptReplacement,
        *,
        reason: str,
        authority: ReserveAuthority | None = None,
    ) -> Plan:
        """Replace history through the worker and commit only a completed sync."""
        with self.claim("replacement", authority=authority):
            plan = rebind.replace_transcript(self._session, self, replacement, reason=reason)
            with self._lock:
                self.require_open()
                self._transcript.replace(replacement.tokens, replacement.segments)
                self._clear_pause()
                return plan

    # -- preamble ----------------------------------------------------------

    def start_chat(
        self,
        tools: Iterable[ToolDefinition] = (),
        instructions: str | None = None,
    ) -> None:
        with self.binding_transaction() as transaction:
            transaction.start_chat(tools, instructions)

    def _stage_chat(
        self,
        tools: Iterable[ToolDefinition] = (),
        instructions: str | None = None,
    ) -> None:
        """Append rendered preamble tokens; the transaction holds the lock."""
        if self._transcript.chat:
            raise EngineError("chat is already initialized")
        for segment in PromptBuilder(self._session.engine).begin(tools, instructions):
            if segment.tokens:
                self._transcript.append(segment.tokens, segment.kind)

    def _sync_chat(self) -> None:
        """Sync the staged preamble and hold its system checkpoint without the lock."""
        tokens = self._transcript.tokens
        if not tokens:
            return
        if any(isinstance(event, Interrupted) for event in self._session._job_events(SyncJob(tokens))):
            raise EngineError("chat start was interrupted")
        self._checkpoint_native(SYSTEM)

    def _set_chat(self) -> None:
        with self._lock:
            self._transcript.set_chat()

    @contextmanager
    def binding_transaction(self) -> Generator[BindingTransaction]:
        """Claim all binding boundaries of one real preparation plan."""
        with self.claim("binding") as claim:
            transaction = BindingTransaction(self, claim)
            try:
                yield transaction
            finally:
                transaction.close()


class BindingTransaction:
    """One claimed binding boundary sequence over Session-owned history.

    The handle admits only the planned truncate/start/append sequence. Native
    preamble work and native checkpoint disposal release the mutation lock but
    never the claim, and every operation re-checks the prefix revision its plan
    observed.
    """

    def __init__(self, history: SessionHistory, claim: session_operations.Claim) -> None:
        self._history = history
        self._claim = claim
        self._active = True
        self._revision, self._length = history.transcript.snapshot()

    def close(self) -> None:
        self._active = False

    def live(self, state: session_operations.OperationState) -> bool:
        """Return True only while this transaction still owns its claim."""
        return self._active and state.owns(self._claim)

    def _observed(self) -> tuple[int, int]:
        return self._history.transcript.snapshot()

    def _check_current(self) -> None:
        revision, length = self._observed()
        if not self.live(self._history.operations):
            raise EngineError("the binding transaction is no longer active")
        if (revision, length) != (self._revision, self._length):
            raise EngineError("the transcript changed outside its binding transaction")

    def _committed(self) -> None:
        self._revision, self._length = self._observed()

    def truncate(self, token_count: int) -> None:
        """Truncate history and invalidate later checkpoints inside the claim.

        The Python truncation commits under the mutation lock, then each stale
        native checkpoint is disposed with only the claim held: a dispatched
        drop can wait on cross-thread event delivery, and the claim already
        excludes every writer and every idle checkpoint reader from the gap.
        """
        history = self._history
        with history.mutation_lock:
            self._check_current()
            position = history.truncate_position(token_count)
            stale = history.stale_checkpoints(position)
            history.commit_truncation(position)
            self._committed()
        for key in stale:
            history.dispose_checkpoint(key)
        with history.mutation_lock:
            self._check_current()

    def start_chat(
        self,
        tools: Iterable[ToolDefinition] = (),
        instructions: str | None = None,
    ) -> None:
        """Render, sync and hold the system preamble; roll Python state back on failure."""
        history = self._history
        with history.mutation_lock:
            self._check_current()
            start = len(history.transcript)
            try:
                history._stage_chat(tools, instructions)
            except BaseException:
                history.rollback(start)
                self._committed()
                raise
        try:
            history._sync_chat()
        except BaseException:
            with history.mutation_lock:
                history.rollback(start)
                self._committed()
            raise
        with history.mutation_lock:
            history._set_chat()
            self._committed()

    def apply_appends(
        self,
        replacement: TranscriptReplacement,
        *,
        retire_paused: bool = False,
        publish: Callable[[], None] | None = None,
    ) -> None:
        """Install one staged group and publish its state under the same lock."""
        history = self._history
        with history.mutation_lock:
            self._check_current()
            try:
                history._commit_appends(
                    replacement,
                    retire_paused=retire_paused,
                    publish=publish,
                    binding=self,
                )
            except BaseException:
                # A rejected publication restored the whole group; the claim is
                # still coherent, so keep the observed prefix as its expectation.
                self._committed()
                raise
            self._committed()
