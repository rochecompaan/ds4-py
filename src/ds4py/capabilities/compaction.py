"""The selected context owner: summarize old history and commit a verbatim tail."""

from __future__ import annotations

import asyncio
from asyncio import CancelledError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override
from uuid import uuid4

from pydantic_ai.messages import InstructionPart, ModelRequest, SystemPromptPart, UserPromptPart

from ds4py.async_session import run_session_operation, session_operation_owner
from ds4py.capabilities.compaction_tail import plan_tail
from ds4py.capabilities.context import ContextRequest, ContextStrategy, RoomResult, usage_charge
from ds4py.compaction_commit import commit_compaction
from ds4py.engine.compaction import SUMMARY_PROMPT, summary_budget, summary_suffix
from ds4py.engine.transcript import TranscriptReplacement
from ds4py.errors import EngineError
from ds4py.summary_usage import summary_usage

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext

    from ds4py.binding import MessageBinder
    from ds4py.engine.reserve import ReserveOperations


type CompactionReason = Literal["threshold", "tool_result", "reserve", "manual"]
type CompactionOutcome = Literal["compacted", "noop", "failed", "cancelled"]

# C's conservative trigger reserve is deliberately not the generation request.
TRIGGER_REASON = "context pressure before continuing the current task"
TRIGGER_SUMMARY_MARGIN = 64
TRIGGER_THRESHOLD_MARGIN = 128
TRIGGER_SOFT_THRESHOLD = 0.85


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    dropped: int
    kept: int
    generated_tokens: int


class LLMCompaction(ContextStrategy):
    """Automatic and manual compaction share one ordered, in-place commit."""

    reserve_operations: bool = True

    def __init__(self, binder: MessageBinder, *, soft: float = TRIGGER_SOFT_THRESHOLD) -> None:
        super().__init__(binder)
        self.soft = soft
        self._attempt: asyncio.Task[object] | None = None

    @override
    def reserve_messages(
        self, request_messages: list[ModelMessage], run_context: RunContext[object] | None
    ) -> list[ModelMessage]:
        """Persistent compaction commits the harness history, not its request copy."""
        return run_context.messages if run_context is not None else request_messages

    def summary_reserve(self) -> int:
        """Reason-bearing trigger suffix, C's +64, and the shared summary budget."""
        engine = self.binder.session.engine
        prompt = SUMMARY_PROMPT + f"\nCompaction reason: {TRIGGER_REASON}\n"
        return (
            len(summary_suffix(engine, (), prompt=prompt)) + TRIGGER_SUMMARY_MARGIN + summary_budget(engine.config.ctx)
        )

    def should_compact(self) -> bool:
        session = self.binder.session
        used, size = len(session.transcript), session.engine.config.ctx
        if not used:
            return False
        return used >= int(size * self.soft) or size - used <= max(
            min(8192, size // 8), self.summary_reserve() + TRIGGER_THRESHOLD_MARGIN
        )

    async def compact_messages(
        self, messages: list[ModelMessage], *, reason: CompactionReason, operations: ReserveOperations | None = None
    ) -> CompactionResult:
        if self._attempt is not None:
            raise EngineError("a compaction is already active")
        self._attempt = asyncio.current_task()
        session = self.binder.session
        try:
            with session.events_on_loop(asyncio.get_running_loop()), session_operation_owner(session):
                return await self._compact_messages(messages, reason=reason, operations=operations)
        finally:
            self._attempt = None

    async def _compact_messages(
        self, messages: list[ModelMessage], *, reason: CompactionReason, operations: ReserveOperations | None = None
    ) -> CompactionResult:
        session = self.binder.session
        old = len(session.transcript)
        outcome: CompactionOutcome = "failed"
        try:
            session.emit("before_compact", reason=reason, old_tokens=old)
            # Let queued consumers paint the status before starting native work.
            await asyncio.sleep(0)
            tail = plan_tail(self.binder, messages)
            if not tail.cut_messages or tail.bottom_tokens <= tail.system_end_tokens:
                outcome = "noop"
                return CompactionResult("", 0, len(messages), 0)
            summary = (
                await operations.prepare_summary_async(tail.bottom_tokens)
                if operations is not None
                else await run_session_operation(session, lambda: session.prepare_summary(tail.bottom_tokens))
            )
            parts = [
                part
                for message in tail.snapshot[: tail.cut_messages]
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, (SystemPromptPart, InstructionPart))
            ]
            key = uuid4().hex
            request = ModelRequest(
                [*parts, UserPromptPart(summary.summary)],
                instructions=next(
                    (
                        message.instructions
                        for message in reversed(tail.snapshot)
                        if isinstance(message, ModelRequest) and message.instructions is not None
                    ),
                    None,
                ),
                metadata={"ds4.key": key},
            )
            prefix = TranscriptReplacement(session.transcript.tokens, session.transcript.segments).slice(
                0, tail.system_end_tokens
            )
            rendered = self.binder.render_request(request, prefix=prefix)
            replacement = TranscriptReplacement.concat(rendered, tail.retained)
            await commit_compaction(
                self.binder, messages, [request, *tail.messages], replacement, operations=operations
            )
            generated = len(summary.tokens)
            session.emit(
                "compacted",
                reason=reason,
                old_tokens=old,
                new_tokens=len(replacement.tokens),
                generated_tokens=generated,
            )
            outcome = "compacted"
            return CompactionResult(summary.summary, tail.cut_messages, len(tail.messages), generated)
        except CancelledError:
            # Cancellation can arrive at the paint checkpoint before a native
            # operation exists to consume it. All submitted work is drained by
            # this point, so clear that signal before releasing attempt ownership.
            session.clear_cancellation()
            outcome = "cancelled"
            raise
        finally:
            session.emit("compaction_finished", reason=reason, outcome=outcome)

    @override
    async def make_room(self, messages: list[ModelMessage], request: ContextRequest) -> RoomResult:
        old = len(self.binder.session.transcript)
        result = await self.compact_messages(messages, reason=request.reason, operations=request.operations)
        return RoomResult(old - len(self.binder.session.transcript) >= request.needed_tokens, result.generated_tokens)

    @override
    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if self.should_compact():
            with summary_usage(self.binder.session, usage_charge(ctx)) as attempt:
                result = await self.compact_messages(ctx.messages, reason="threshold")
                attempt.report_legacy(result.generated_tokens)
        return await super().before_model_request(ctx, request_context)
