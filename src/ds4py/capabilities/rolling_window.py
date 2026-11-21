"""The view-only RollingWindow context owner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, override

from pydantic_ai.messages import ModelRequest, ToolReturnPart

from ds4py.binding_plan import request_instructions
from ds4py.binding_provenance import ensure_message_keys
from ds4py.capabilities.context import TOOL_RESULT_DOES_NOT_FIT, ContextRequest, ContextStrategy, RoomResult
from ds4py.context_paging.history import PAGING_TOOL_NAMES, project_history
from ds4py.context_paging.meter import RequestMeter
from ds4py.context_paging.selection import OverflowMode, SelectionResult, SelectionState, select_view
from ds4py.context_paging.settings import DEFAULT_TOKEN_BUDGET, RollingWindowSettings, default_trim_to
from ds4py.context_paging.tools import create_context_paging_toolset
from ds4py.errors import ContextExhausted, ContextSelectionError
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY, DS4_VIEW_OF

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage, ToolCallPart
    from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
    from pydantic_ai.tools import RunContext, ToolDefinition
    from pydantic_ai.toolsets import FunctionToolset

    from ds4py.binding import MessageBinder
    from ds4py.capabilities.compaction import LLMCompaction


@dataclass(slots=True)
class _RollingState:
    settings: RollingWindowSettings
    selection: SelectionState | None = None
    request_view: tuple[ModelMessage, ...] | None = None
    canonical_messages: tuple[ModelMessage, ...] = ()
    request_params: ModelRequestParameters | None = None


class RollingWindow(ContextStrategy):
    """Select transient request views while canonical harness history stays intact."""

    reserve_operations = True

    def __init__(
        self,
        binder: MessageBinder,
        *,
        disabled_owner: LLMCompaction,
        enabled: bool | None = None,
        token_budget: int | None = None,
        trim_to: int | None = None,
    ) -> None:
        super().__init__(binder)
        budget = DEFAULT_TOKEN_BUDGET if token_budget is None else token_budget
        settings = RollingWindowSettings(
            enabled=True if enabled is None else enabled,
            token_budget=budget,
            trim_to=default_trim_to(budget) if trim_to is None else trim_to,
            trim_to_explicit=trim_to is not None,
        )
        self.disabled_owner = disabled_owner
        self._tools: FunctionToolset[object] = create_context_paging_toolset(
            is_enabled=lambda: self._state.settings.enabled,
            history_source=lambda: self._state.canonical_messages,
        )
        self._initial_settings = settings

    @property
    def _state(self) -> _RollingState:
        return self.binder.session.state("rolling_window", lambda: _RollingState(self._initial_settings))

    @property
    def enabled(self) -> bool:
        return self._state.settings.enabled

    @override
    def get_toolset(self) -> FunctionToolset[object]:
        return self._tools

    @override
    def request_messages(
        self, request_messages: list[ModelMessage], run_context: RunContext[object] | None
    ) -> list[ModelMessage]:
        """Materialize the exact stored ordinary-request view after normalization."""
        del run_context
        state = self._state
        if not state.settings.enabled or state.request_view is None:
            return request_messages
        return list(state.request_view)

    @override
    def reserve_messages(
        self, request_messages: list[ModelMessage], run_context: RunContext[object] | None
    ) -> list[ModelMessage]:
        """Recover in the same actual view, or delegate while paging is disabled."""
        if not self._state.settings.enabled:
            return self.disabled_owner.reserve_messages(request_messages, run_context)
        del run_context
        view = self._state.request_view
        return list(view) if view is not None else request_messages

    @override
    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        ensure_message_keys(ctx.messages)
        state = self._state
        request_context.messages = ctx.messages
        if not state.settings.enabled:
            result = await self.disabled_owner.before_model_request(ctx, request_context)
            state.canonical_messages = tuple(ctx.messages)
            state.request_view = None
            return result
        params = request_context.model_request_parameters
        selected = self._select_with_tool_fallback(ctx.messages, params, state.selection)
        projected = self.binder.project_request(
            selected.candidate_messages,
            tools=params.declared_tool_defs.values(),
            instructions=self._instructions(selected.candidate_messages, request_context),
        )
        if self.binder.reserve_overflow(projected) > 0:
            raise ContextExhausted("selected rolling-window request does not fit the model context")
        state.canonical_messages = tuple(ctx.messages)
        state.request_params = params
        state.selection = selected.state
        state.request_view = selected.candidate_messages
        return request_context

    def _select_with_tool_fallback(
        self,
        messages: Sequence[ModelMessage],
        params: ModelRequestParameters,
        remembered: SelectionState | None,
    ) -> SelectionResult:
        try:
            return self._select(messages, params, remembered)
        except ContextSelectionError as selection_error:
            pending = messages[-1] if messages and isinstance(messages[-1], ModelRequest) else None
            results = (
                [
                    part
                    for part in pending.parts
                    if isinstance(part, ToolReturnPart) and part.tool_name not in PAGING_TOOL_NAMES
                ]
                if pending is not None
                else []
            )
            if not results:
                raise
            rejected: set[str] = set()
            for part in reversed(results):
                rejected.add(part.tool_call_id)
                working = deepcopy(tuple(messages))
                clone = working[-1]
                if not isinstance(clone, ModelRequest):
                    raise TypeError("pending tool fallback requires a request")
                metadata = clone.metadata or {}
                original_key = metadata.get(DS4_KEY)
                if not isinstance(original_key, str) or not original_key:
                    raise ContextExhausted(
                        "pending tool fallback requires a canonical request key"
                    ) from selection_error
                clone.metadata = {
                    **metadata,
                    DS4_KEY: f"{original_key}:rejected:{len(rejected)}",
                    DS4_CONTEXT_PAGING: "recovery",
                    DS4_VIEW_OF: original_key,
                }
                for candidate in clone.parts:
                    if isinstance(candidate, ToolReturnPart) and candidate.tool_call_id in rejected:
                        tool_part = cast("ToolReturnPart", candidate)
                        tool_part.content = TOOL_RESULT_DOES_NOT_FIT
                try:
                    return self._select(working, params, remembered)
                except ContextSelectionError:
                    continue
            raise ContextExhausted("pending tool errors do not fit the rolling-window context") from selection_error

    @staticmethod
    def _instructions(messages: Sequence[ModelMessage], request_context: ModelRequestContext) -> str | None:
        return request_instructions(messages, request_context.model_request_parameters)

    def _select(
        self,
        messages: Sequence[ModelMessage],
        params: ModelRequestParameters,
        remembered: SelectionState | None,
        *,
        overflow: OverflowMode = "normal",
    ) -> SelectionResult:
        return select_view(
            project_history(messages),
            settings=self._state.settings,
            model_context=self.binder.session.engine.config.ctx,
            meter=RequestMeter(self.binder, params),
            remembered=remembered,
            overflow=overflow,
        )

    @override
    async def make_room(self, messages: list[ModelMessage], request: ContextRequest) -> RoomResult:
        if not self._state.settings.enabled:
            return await self.disabled_owner.make_room(messages, request)
        state = self._state
        canonical = state.canonical_messages or tuple(messages)
        params = state.request_params
        if params is None:
            raise ContextSelectionError("REQUEST_PARAMETERS_UNAVAILABLE", "Reserve selection lacks request parameters.")
        try:
            selected = self._select(
                canonical,
                params,
                state.selection,
                overflow="recovery" if request.reason == "reserve" else "normal",
            )
        except ContextSelectionError as error:
            if request.reason == "reserve":
                raise ContextExhausted(str(error)) from error
            raise
        if request.reason == "reserve":
            projected = self.binder.project_request(selected.candidate_messages, bound=True)
            if self.binder.reserve_overflow(projected) > 0:
                raise ContextExhausted("selected rolling-window reserve view does not fit the model context")
            if request.operations is None:
                raise ContextExhausted("rolling-window reserve recovery requires retained operations")
            replacement = self.binder.render_request_view(selected.candidate_messages, params)
            await request.operations.replace_transcript_async(replacement, reason="context_paging")
        state.selection = selected.state
        state.request_view = selected.candidate_messages
        messages[:] = selected.candidate_messages
        return RoomResult(made_room=True)

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
        if not self._state.settings.enabled:
            return await self.disabled_owner.after_tool_execute(
                ctx, call=call, tool_def=tool_def, args=args, result=result
            )
        part = self._tool_return_part(call, result)
        response_key = next(
            (
                (message.metadata or {}).get(DS4_KEY)
                for message in reversed(ctx.messages)
                if not isinstance(message, ModelRequest)
            ),
            "pending",
        )
        pending = ModelRequest([part], metadata={DS4_KEY: f"{response_key}:{call.tool_call_id}:pending-result"})
        self._state.canonical_messages = (*ctx.messages, pending)
        return result
