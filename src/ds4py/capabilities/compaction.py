"""The selected context owner: summarize old history and commit a verbatim tail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override
from uuid import uuid4

from pydantic_ai.messages import InstructionPart, ModelRequest, SystemPromptPart, UserPromptPart

from ds4py._native.protocol import ThinkMode
from ds4py.binding import _ScratchSession
from ds4py.binding_provenance import _content_hash
from ds4py.capabilities.compaction_tail import plan_tail
from ds4py.capabilities.context import ContextRequest, ContextStrategy, RoomResult
from ds4py.compaction_commit import commit_compaction
from ds4py.engine.compaction import SUMMARY_PROMPT
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.transcript import Segment, TranscriptReplacement
from ds4py.message_rendering import MessageRenderer

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext

    from ds4py.binding import MessageBinder


type CompactionReason = Literal["threshold", "tool_result", "reserve", "manual"]


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    dropped: int
    kept: int
    generated_tokens: int


class LLMCompaction(ContextStrategy):
    """Automatic and manual compaction share one ordered, in-place commit."""

    def __init__(self, binder: MessageBinder, *, soft: float = 0.85) -> None:
        super().__init__(binder)
        self.soft = soft

    def summary_reserve(self) -> int:
        """Exact rendered private suffix, C's +64, and C's summary budget."""
        engine = self.binder.session.engine
        prompt = SUMMARY_PROMPT + "\nCompaction reason: context pressure before continuing the current task\n"
        tokens = PromptBuilder(engine).message((), "user", prompt)
        tokens.extend(engine._assistant_prefix(tokens, ThinkMode.NONE))
        return len(tokens) + 64 + max(256, min(engine.config.ctx // 8, 4096))

    def should_compact(self) -> bool:
        session = self.binder.session
        used, size = len(session.transcript), session.engine.config.ctx
        if not used:
            return False
        return used >= int(size * self.soft) or size - used <= max(min(8192, size // 8), self.summary_reserve() + 128)

    async def compact_messages(self, messages: list[ModelMessage], *, reason: CompactionReason) -> CompactionResult:
        session = self.binder.session
        old = len(session.transcript)
        session.emit("before_compact", reason=reason, old_tokens=old)
        tail = plan_tail(self.binder, messages)
        if not tail.dropped or tail.bottom <= tail.system_end:
            return CompactionResult("", 0, len(messages), 0)
        summary = session.prepare_summary(tail.bottom)
        parts = [
            part
            for message in tail.snapshot[: tail.dropped]
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
        scratch = _ScratchSession(session.engine)
        for segment in session.transcript.segments:
            if segment.end > tail.system_end:
                break
            scratch.append_tokens(session.transcript.tokens[segment.start : segment.end], segment.kind, segment.meta)
        MessageRenderer(session.engine).append(scratch, request, key, _content_hash(request))
        offset = len(scratch.transcript)
        replacement = TranscriptReplacement(
            (*scratch.transcript.tokens, *tail.tokens),
            (
                *scratch.transcript.segments,
                *(
                    Segment(segment.start + offset, segment.end + offset, segment.kind, segment.meta)
                    for segment in tail.segments
                ),
            ),
        )
        commit_compaction(self.binder, messages, [request, *tail.messages], replacement)
        generated = len(summary.tokens)
        session.emit(
            "compacted", reason=reason, old_tokens=old, new_tokens=len(replacement.tokens), generated_tokens=generated
        )
        return CompactionResult(summary.summary, tail.dropped, len(tail.messages), generated)

    @override
    async def make_room(self, messages: list[ModelMessage], request: ContextRequest) -> RoomResult:
        old = len(self.binder.session.transcript)
        result = await self.compact_messages(messages, reason=request.reason)
        return RoomResult(old - len(self.binder.session.transcript) >= request.needed_tokens, result.generated_tokens)

    @override
    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if self.should_compact():
            result = await self.compact_messages(ctx.messages, reason="threshold")
            ctx.usage.output_tokens += result.generated_tokens
        return await super().before_model_request(ctx, request_context)
