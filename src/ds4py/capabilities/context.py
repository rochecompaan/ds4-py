"""The one registered context owner for tool overflow and reserve recovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, ToolReturn, ToolReturnPart

from ds4py.binding_plan import request_instructions
from ds4py.errors import ContextExhausted, EngineError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage, ToolCallPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext, ToolDefinition

    from ds4py.binding import MessageBinder

_RESERVE = 1
TOOL_RESULT_DOES_NOT_FIT = "Tool error: tool result still does not fit"


@dataclass(frozen=True)
class ContextRequest:
    """The exact room a selected owner must make for one trigger."""

    needed_tokens: int
    reason: Literal["tool_result", "reserve"]


@dataclass(frozen=True)
class RoomResult:
    """Whether the owner made enough room and how many summary tokens it generated."""

    made_room: bool
    generated_tokens: int = 0


@dataclass(frozen=True)
class ContextComposition:
    """Select one automatic owner; manual compaction remains available."""

    owner: ContextStrategy | None = None


class ContextStrategy(AbstractCapability, ABC):
    """One capability owns the shared tool fit check and the reserve response hook.

    ``make_room`` mutates the supplied request message list in place and returns
    only the shared fit result, never a second message view. ``reserve_messages``
    selects an existing authority (the request view by default); persistent
    owners may instead select canonical harness history. Neither creates a
    second canonical history or changes the other owner's selection.
    """

    def __init__(self, binder: MessageBinder) -> None:
        self.binder = binder

    @abstractmethod
    async def make_room(self, messages: list[ModelMessage], request: ContextRequest) -> RoomResult:
        """Make room inside the live request messages and report the post-move fit."""

    @override
    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Keep keyed canonical boundaries instead of the harness's merged view."""
        request_context.messages = ctx.messages
        await self._fit_pending_results(ctx, request_context)
        return request_context

    async def _fit_pending_results(self, ctx: RunContext[object], request_context: ModelRequestContext) -> None:
        """Check the complete grouped request before it can enter the transcript."""
        if not ctx.messages or not isinstance(pending := ctx.messages[-1], ModelRequest):
            return
        results: list[ToolReturnPart] = [part for part in pending.parts if isinstance(part, ToolReturnPart)]
        if not results:
            return

        def needed() -> int:
            return self._needed_tokens(ctx.messages, request_context=request_context)

        overflow = needed()
        if overflow <= 0:
            return
        room = await self.make_room(ctx.messages, ContextRequest(overflow, "tool_result"))
        ctx.usage.output_tokens += room.generated_tokens
        # Compaction may replace the message objects, so select from live history.
        pending = ctx.messages[-1]
        results = [part for part in pending.parts if isinstance(part, ToolReturnPart)]
        for part in reversed(results):
            if needed() <= 0:
                return
            part.content = TOOL_RESULT_DOES_NOT_FIT
        if needed() > 0:
            raise ContextExhausted("pending tool errors do not fit the context")

    def reserve_messages(
        self, request_messages: list[ModelMessage], run_context: RunContext[object] | None
    ) -> list[ModelMessage]:
        """Custom/view-only owners recover within the actual model request view."""
        return request_messages

    def _tool_return_message(self, call: ToolCallPart, result: object) -> ModelRequest:
        content = result.return_value if isinstance(result, ToolReturn) else result
        return ModelRequest([ToolReturnPart(call.tool_name, content, tool_call_id=call.tool_call_id)])

    def _needed_tokens(
        self, messages: Sequence[ModelMessage], *, request_context: ModelRequestContext | None = None
    ) -> int:
        if request_context is not None:
            params = request_context.model_request_parameters
            count = self.binder.count_rendered_request(
                messages, tools=params.declared_tool_defs.values(), instructions=request_instructions(messages, params)
            )
        elif self.binder.session.transcript.chat:
            count = self.binder.count_bound_request(messages)
        else:
            raise EngineError("after-tool fit requires an initialized chat")
        return count + _RESERVE - self.binder.session.engine.config.ctx

    @override
    async def after_tool_execute(
        self,
        ctx: RunContext[object],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, object],
        result: object,
    ) -> object:
        """Ask this owner for room before an oversized tool result enters history."""
        candidate = self._tool_return_message(call, result)
        needed = self._needed_tokens([*ctx.messages, candidate])
        if needed <= 0:
            return result
        room = await self.make_room(ctx.messages, ContextRequest(needed, "tool_result"))
        ctx.usage.output_tokens += room.generated_tokens
        if not room.made_room or self._needed_tokens([*ctx.messages, candidate]) > 0:
            return TOOL_RESULT_DOES_NOT_FIT
        return result
