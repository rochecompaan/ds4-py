"""The view-only RollingWindow context owner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast, override

from pydantic_ai.messages import ModelRequest, ModelResponse, ToolReturnPart

from ds4py.binding_plan import request_instructions
from ds4py.binding_provenance import ensure_message_keys
from ds4py.capabilities.context import TOOL_RESULT_DOES_NOT_FIT, ContextRequest, ContextStrategy, RoomResult
from ds4py.capabilities.rolling_window_session import RollingWindowSession
from ds4py.context_paging.history import PAGING_TOOL_NAMES, ModelTurnHistoryItem, project_history
from ds4py.context_paging.meter import RequestMeter
from ds4py.context_paging.selection import (
    ContextCutFrontier,
    OverflowMode,
    SelectionResult,
    SelectionState,
    select_view,
)
from ds4py.context_paging.settings import DEFAULT_TOKEN_BUDGET, RollingWindowSettings, default_trim_to
from ds4py.context_paging.stats import RollingWindowStatisticsState
from ds4py.context_paging.tools import create_context_paging_toolset
from ds4py.errors import ContextExhausted, ContextSelectionError
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY, DS4_PAGING_INPUT_TOKENS, DS4_VIEW_OF

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pydantic_ai.messages import ModelMessage, ToolCallPart
    from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
    from pydantic_ai.tools import RunContext, ToolDefinition
    from pydantic_ai.toolsets import FunctionToolset

    from ds4py.binding import MessageBinder
    from ds4py.capabilities.compaction import LLMCompaction
    from ds4py.context_paging.history import RestoreBoundary


@dataclass(slots=True)
class _RollingState:
    settings: RollingWindowSettings
    selection: SelectionState | None = None
    request_view: tuple[ModelMessage, ...] | None = None
    canonical_messages: tuple[ModelMessage, ...] = ()
    request_params: ModelRequestParameters | None = None
    request_input_tokens: int | None = None
    restore_ids: set[str] = field(default_factory=set)


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
            history_source=lambda: project_history(
                self._state.canonical_messages,
                restore_ids=self._state.restore_ids,
            ),
        )
        self._initial_settings = settings
        self._base_settings = settings
        self._enabled_override = enabled
        self._budget_override = token_budget
        self._trim_override = trim_to
        self._session_controller = RollingWindowSession(self)

    @property
    def _state(self) -> _RollingState:
        return self.binder.session.state("rolling_window", lambda: _RollingState(self._base_settings))

    @property
    def enabled(self) -> bool:
        return self._state.settings.enabled

    def initialize(self, *, project_dir: Path, store_root: Path, project_trusted: bool) -> None:
        """Resolve file and constructor settings after the session event bus exists."""
        self._session_controller.initialize(
            project_dir=project_dir,
            store_root=store_root,
            project_trusted=project_trusted,
        )

    def command(self, action: str | None) -> str:
        return self._session_controller.command(action)

    def reset_cut(self) -> None:
        self._state.selection = None

    def statistics_state(self) -> RollingWindowStatisticsState:
        """Copy only the live values needed by the read-only report."""
        state = self._state
        return RollingWindowStatisticsState(
            enabled=state.settings.enabled,
            token_budget=state.settings.token_budget,
            model_window=self.binder.session.engine.config.ctx,
            input_tokens=state.request_input_tokens,
        )

    def adopt_history(self, canonical_messages: Sequence[ModelMessage]) -> None:
        """Track the runtime's authoritative history after terminal run adoption."""
        self._state.canonical_messages = tuple(canonical_messages)

    def reset_after_fork(self) -> None:
        """Retain canonical identity and KV while resetting per-session policy state."""
        from ds4py.binding_provenance import prepare_request_view_provenance  # noqa: PLC0415 - Restore-only validation path.
        from ds4py.context_paging.history import derive_frontier  # noqa: PLC0415 - Avoid the history/selection cycle.

        state = self._state
        state.settings = self._base_settings
        state.request_input_tokens = None
        self._classify_restore_interruptions("forked")
        state.request_view = self.binder.request_view()
        if not state.canonical_messages or not state.request_view:
            state.selection = None
            return
        prepared = prepare_request_view_provenance(
            canonical_messages=state.canonical_messages,
            request_view=state.request_view,
            segments=self.binder.export_segments(),
            loaded=None,
        )
        frontier = derive_frontier(state.canonical_messages, prepared)
        state.selection = None if frontier is None else SelectionState(frontier, None)

    def restore_state(
        self,
        canonical_messages: Sequence[ModelMessage],
        frontier: ContextCutFrontier | None = None,
        *,
        restore_boundary: RestoreBoundary | None = None,
    ) -> None:
        """Adopt validated canonical identity and the binder's freshly restored view."""
        state = self._state
        state.settings = self._base_settings
        state.canonical_messages = tuple(canonical_messages)
        state.selection = None if frontier is None else SelectionState(frontier, None)
        state.request_params = None
        state.request_input_tokens = None
        state.request_view = self.binder.request_view()
        self._classify_restore_interruptions(restore_boundary)

    def _classify_restore_interruptions(self, boundary: RestoreBoundary | None) -> None:
        state = self._state
        projected = project_history(state.canonical_messages, restore_boundary=boundary)
        state.restore_ids = {
            item.history_id
            for item in projected.items
            if isinstance(item, ModelTurnHistoryItem) and item.interrupted_by_restore
        }

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
    def record_response(self, response: ModelResponse) -> None:
        """Stamp the exact selected input on the final canonical response only."""
        tokens = self._state.request_input_tokens
        if not self._state.settings.enabled or tokens is None:
            return
        if response.metadata is None:
            response.metadata = {}
        # Mutate the shared metadata mapping so later canonical copies retain it.
        response.metadata[DS4_PAGING_INPUT_TOKENS] = tokens

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
            state.request_input_tokens = None
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
        tokens = selected.measurement.total_tokens
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 <= tokens <= 2**53 - 1:
            raise ValueError("selected rolling-window input must be a non-negative safe integer")
        state.canonical_messages = tuple(ctx.messages)
        state.request_params = params
        state.selection = selected.state
        state.request_view = selected.candidate_messages
        state.request_input_tokens = tokens
        return request_context

    def _select_with_tool_fallback(
        self,
        messages: Sequence[ModelMessage],
        params: ModelRequestParameters,
        remembered: SelectionState | None,
        *,
        overflow: OverflowMode = "normal",
    ) -> SelectionResult:
        model_context: int | None = None
        if overflow == "recovery":
            session = self.binder.session
            model_context = session.engine.config.ctx
            sampled_start = session.open_assistant_start()
            if sampled_start is not None:
                sampled = len(session.transcript.tokens) - sampled_start
                # Selection applies the response reserve once to this remaining window.
                model_context = max(1, model_context - sampled)
        try:
            return self._select(messages, params, remembered, overflow=overflow, model_context=model_context)
        except ContextSelectionError as selection_error:
            if selection_error.code != "ACTIVE_REQUEST_TOO_LARGE":
                raise
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
                    # Recovery already failed; each retry must keep its rejected results instead of
                    # replacing them with the same recovery references and rejecting their view-only key.
                    return self._select(
                        working,
                        params,
                        remembered,
                        overflow="protected-overflow",
                        provenance_messages=messages,
                        model_context=model_context,
                    )
                except ContextSelectionError:
                    continue
            raise ContextExhausted("pending tool errors do not fit the rolling-window context") from selection_error

    @staticmethod
    def _instructions(messages: Sequence[ModelMessage], request_context: ModelRequestContext) -> str | None:
        return request_instructions(messages, request_context.model_request_parameters)

    def _select(  # noqa: PLR0913 - Tool fallback must share the exact recovery-window measurement.
        self,
        messages: Sequence[ModelMessage],
        params: ModelRequestParameters,
        remembered: SelectionState | None,
        *,
        overflow: OverflowMode = "normal",
        provenance_messages: Sequence[ModelMessage] | None = None,
        model_context: int | None = None,
    ) -> SelectionResult:
        if model_context is None:
            model_context = self.binder.session.engine.config.ctx
        return select_view(
            project_history(messages, restore_ids=self._state.restore_ids),
            settings=self._state.settings,
            model_context=model_context,
            meter=RequestMeter(self.binder, params, provenance_messages=provenance_messages),
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
            selected = self._select_with_tool_fallback(
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
            if request.operations is None:
                raise ContextExhausted("rolling-window reserve recovery requires retained operations")
            rendered = self.binder.render_request_view(selected.candidate_messages, params)
            if self.binder.reserve_overflow(len(rendered.replacement.tokens)) > 0:
                raise ContextExhausted("selected rolling-window reserve view does not fit the model context")
            await request.operations.replace_transcript_async(rendered.replacement, reason="context_paging")
            self.binder.adopt_rendered_view(rendered)
        state.selection = selected.state
        state.request_view = selected.candidate_messages
        state.request_input_tokens = selected.measurement.total_tokens
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
