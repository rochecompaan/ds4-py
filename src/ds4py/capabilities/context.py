"""The one registered context owner for tool overflow and reserve recovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, override

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, ToolReturn, ToolReturnPart

from ds4py.binding_plan import request_instructions
from ds4py.errors import ContextExhausted, EngineError
from ds4py.summary_usage import summary_usage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic_ai.messages import ModelMessage, ToolCallPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext, ToolDefinition

    from ds4py.binding import MessageBinder
    from ds4py.engine.reserve import ReserveOperations

_RESERVE = 1
TOOL_RESULT_DOES_NOT_FIT = "Tool error: tool result still does not fit"


@dataclass(frozen=True)
class ContextRequest:
    """The exact room a selected owner must make for one trigger."""

    needed_tokens: int
    reason: Literal["tool_result", "reserve"]
    operations: ReserveOperations | None = None


@dataclass(frozen=True)
class RoomResult:
    """Whether the owner made enough room and how many summary tokens it generated."""

    made_room: bool
    generated_tokens: int = 0


@dataclass(frozen=True)
class ContextComposition:
    """Select one automatic owner; manual compaction remains available."""

    owner: ContextStrategy | None = None


def usage_charge(ctx: RunContext[object]) -> Callable[[int], None]:
    """Charge private summary output to the harness request usage."""

    def charge(tokens: int) -> None:
        ctx.usage.output_tokens += tokens

    return charge


class ContextStrategy(AbstractCapability, ABC):
    """One capability owns the shared tool fit check and the reserve response hook.

    ``make_room`` mutates the supplied request message list in place and returns
    only the shared fit result, never a second message view. ``reserve_messages``
    selects an existing authority (the request view by default); persistent
    owners may instead select canonical harness history. Neither creates a
    second canonical history or changes the other owner's selection.

    Set ``reserve_operations = True`` to receive strict operations through
    ``ContextRequest.operations`` during reserve recovery. Raw Session calls
    then reject mid-round; use the handle's sync/retained async methods instead.
    False preserves the legacy direct/naked-to_thread adapter and its documented
    child-executor ambiguity. LLMCompaction subclasses inherit True and must
    explicitly opt out with False if they still use raw Session calls.
    """

    reserve_operations: bool = False

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
            return self._needed_for_request(ctx.messages, request_context)

        overflow = needed()
        if overflow <= 0:
            return
        room = await self._recover_room(ctx, overflow)
        # Compaction may replace the message objects, so select from live history.
        pending = ctx.messages[-1]
        results = [part for part in pending.parts if isinstance(part, ToolReturnPart)]
        self._reject_until_fit(results, room, needed)

    def reserve_messages(
        self, request_messages: list[ModelMessage], run_context: RunContext[object] | None
    ) -> list[ModelMessage]:
        """Custom/view-only owners recover within the actual model request view."""
        return request_messages

    def _tool_return_message(self, call: ToolCallPart, result: object) -> ModelRequest:
        content = result.return_value if isinstance(result, ToolReturn) else result
        return ModelRequest([ToolReturnPart(call.tool_name, content, tool_call_id=call.tool_call_id)])

    def _needed_for_request(self, messages: Sequence[ModelMessage], request_context: ModelRequestContext) -> int:
        """Measure the effective configured projection, not the old bound chat."""
        params = request_context.model_request_parameters
        count = self.binder.count_rendered_request(
            messages, tools=params.declared_tool_defs.values(), instructions=request_instructions(messages, params)
        )
        return count + _RESERVE - self.binder.session.engine.config.ctx

    def _needed_for_chat(self, messages: Sequence[ModelMessage]) -> int:
        """Measure the strict bound projection; after-tool calls require a prepared chat."""
        if not self.binder.session.transcript.chat:
            raise EngineError("after-tool fit requires an initialized chat")
        count = self.binder.count_bound_request(messages)
        return count + _RESERVE - self.binder.session.engine.config.ctx

    async def _recover_room(self, ctx: RunContext[object], needed_tokens: int) -> RoomResult:
        """Run the one owner attempt and account for its summary output once."""
        with summary_usage(self.binder.session, usage_charge(ctx)) as attempt:
            room = await self.make_room(ctx.messages, ContextRequest(needed_tokens, "tool_result"))
            attempt.report_legacy(room.generated_tokens)
        return room

    def _reject_until_fit(self, results: Sequence[ToolReturnPart], room: RoomResult, needed: Callable[[], int]) -> None:
        """Reject trailing live results until the complete request fits exactly.

        An explicit owner failure forces the first rejection even when the
        complete request currently fits; subsequent rejections are minimal.
        """
        for rejected, part in enumerate(reversed(results)):
            if (room.made_room or rejected > 0) and needed() <= 0:
                return
            part.content = TOOL_RESULT_DOES_NOT_FIT
        if needed() > 0:
            raise ContextExhausted("pending tool errors do not fit the context")

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
        needed = self._needed_for_chat([*ctx.messages, candidate])
        if needed <= 0:
            return result
        room = await self._recover_room(ctx, needed)
        part = cast("ToolReturnPart", candidate.parts[0])

        def still_needed() -> int:
            return self._needed_for_chat([*ctx.messages, candidate])

        original = part.content
        self._reject_until_fit([part], room, still_needed)
        return TOOL_RESULT_DOES_NOT_FIT if part.content is not original else result
