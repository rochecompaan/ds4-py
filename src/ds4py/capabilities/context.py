"""The one registered context owner for tool overflow and reserve recovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, ToolReturn, ToolReturnPart

if TYPE_CHECKING:
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
    only the shared fit result. The owner never returns a second message view.
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
        return request_context

    def _tool_return_message(self, call: ToolCallPart, result: object) -> ModelRequest:
        content = result.return_value if isinstance(result, ToolReturn) else result
        return ModelRequest([ToolReturnPart(call.tool_name, content, tool_call_id=call.tool_call_id)])

    def _needed_tool_tokens(self, messages: list[ModelMessage], candidate: ModelRequest) -> int:
        current = self.binder.count_tokens(messages)
        added = self.binder.count_tokens([*messages, candidate]) - current
        session = self.binder.session
        return len(session.transcript) + added + _RESERVE - session.engine.config.ctx

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
        needed = self._needed_tool_tokens(ctx.messages, self._tool_return_message(call, result))
        if needed <= 0:
            return result
        room = await self.make_room(ctx.messages, ContextRequest(needed, "tool_result"))
        ctx.usage.output_tokens += room.generated_tokens
        if not room.made_room:
            return TOOL_RESULT_DOES_NOT_FIT
        return result
