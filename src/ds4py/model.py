"""Adapt the ds4 engine facade to Pydantic AI requests and streamed responses."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import math
import operator
import sys
from uuid import uuid4

import anyio

from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.usage import RequestUsage

from ds4py.binding import MessageBinder
from ds4py.engine.events import (
    Interrupted, ReserveReached, Stopped, Synced, TextDelta, Token, ToolCalls,
    ToolSyntax,
)
from ds4py.engine.sampling import SamplingPolicy
from ds4py.errors import EngineError


_END = object()


class ContextReserveError(EngineError):
    """The response hit the context reserve and no compaction strategy is installed."""


def _context_reserve_message(needed):
    return (
        f"context reserve reached with {needed} token(s) needed; "
        "M4 compaction is not available in this build"
    )


def _next(iterator):
    """Keep StopIteration out of the Future used by asyncio.to_thread."""
    try:
        return next(iterator)
    except StopIteration:
        return _END


def _integer(value, name):
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a concrete integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a concrete integer") from error


def _finite(value, name):
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a concrete finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a concrete finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a concrete finite number")
    return value


def _instructions(parts):
    if not parts:
        return None
    return "\n\n".join(part.content for part in parts) or None


class DS4Model(Model):
    """A harness model backed by one facade-owned native/sampling session.

    Calls are confined to the session's consuming event-loop thread. Do not
    overlap requests or streams, and do not call the facade from a native worker
    callback. A response must be drained or closed before the next round.
    """

    supported_tool_addition_modes = frozenset()
    supported_tool_deferral_modes = frozenset()

    def __init__(self, session, *, binder=None, sampling: SamplingPolicy,
                 max_tokens=50_000, stop_sequences=(), note_provider=None,
                 reserve_handler=None):
        super().__init__()
        self.session = session
        self.binder = binder or MessageBinder(session)
        self.sampling = sampling
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences
        self.note_provider = note_provider
        # No compaction strategy is installed. A handler may free space and return the new
        # reserve budget; otherwise the default policy fails the open response.
        self.reserve_handler = reserve_handler

    @property
    def model_name(self):
        return self.session.engine.info.name

    @property
    def system(self):
        return "ds4"

    @classmethod
    def supported_native_tools(cls):
        return frozenset()

    def _instance_profile_defaults(self, profile):
        return ModelProfile(**{
            **profile,
            "context_window": self.session.engine.config.ctx,
            "supports_thinking": True,
            "supports_tools": True,
            "supports_inline_system_prompts": True,
            "supports_json_schema_output": False,
            "supports_json_object_output": False,
            "default_structured_output_mode": "prompted",
            "json_schema_transformer": OpenAIJsonSchemaTransformer,
            "thinking_tags": ("<think>", "</think>"),
        })

    def _request_config(self, settings):
        settings = settings or {}
        policy = SamplingPolicy(
            temperature=_finite(settings.get("temperature", self.sampling.temperature), "temperature"),
            top_p=_finite(settings.get("top_p", self.sampling.top_p), "top_p"),
            min_p=_finite(settings.get("min_p", self.sampling.min_p), "min_p"),
            top_k=_integer(settings.get("top_k", self.sampling.top_k), "top_k"),
            seed=settings.get("seed", self.sampling.seed),
        )
        max_tokens = _integer(settings.get("max_tokens", self.max_tokens), "max_tokens")
        stops = settings.get("stop_sequences", self.stop_sequences)
        if stops is None:
            raise ValueError("stop_sequences must be concrete strings or bytes")
        stop_sequences = tuple(
            stop.encode() if isinstance(stop, str) else bytes(stop) for stop in stops
        )
        return policy, max_tokens, stop_sequences

    @asynccontextmanager
    async def _open_stream(self, messages, settings, params):
        policy, max_tokens, stop_sequences = self._request_config(settings)
        instruction_parts = self._get_instruction_parts(messages, params)
        self.binder.prepare(
            messages,
            tools=params.declared_tool_defs.values(),
            instructions=_instructions(instruction_parts),
            note_provider=self.note_provider,
        )
        response = DS4StreamedResponse(
            self, params, policy, max_tokens, stop_sequences,
        )
        yield response

    async def request(self, messages, model_settings, model_request_parameters) -> ModelResponse:
        settings, params = self.prepare_request(model_settings, model_request_parameters)
        async with self._open_stream(messages, settings, params) as stream:
            async for _ in stream:
                pass
            return stream.get()

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters,
                             run_context=None):
        settings, params = self.prepare_request(model_settings, model_request_parameters)
        async with self._open_stream(messages, settings, params) as stream:
            yield stream


class DS4StreamedResponse(StreamedResponse):
    """Translate facade events while finalizing sampled provenance exactly once."""

    def __init__(self, model: DS4Model, params: ModelRequestParameters,
                 sampling: SamplingPolicy, max_tokens: int, stop_sequences):
        super().__init__(model_request_parameters=params)
        self._model = model
        self._session = model.session
        self._binder = model.binder
        self._timestamp = datetime.now(timezone.utc)
        self._response_key = uuid4().hex
        self.metadata = {"ds4.key": self._response_key, "ds4.source": "sampled"}
        self._round = self._session.run_round(
            max_tokens=max_tokens,
            sampling=sampling,
            stop_sequences=stop_sequences,
        )
        self._round_exhausted = False
        self._reconciled = False
        self._round_started = False
        self._bound = False
        self._round_start = len(self._session.transcript)
        self._sampled_start = None
        self._prefilled = 0
        self._output_tokens = 0
        self._tool_index = 0
        self._in_tool_syntax = False
        self._pull_lock = asyncio.Lock()
        self._pending_pull = None
        self._cleanup_task = None
        self._reserve_needed = None
        self._context_exhausted = False

    @property
    def model_name(self):
        return self._model.model_name

    @property
    def provider_name(self):
        return "ds4"

    @property
    def provider_url(self):
        return None

    @property
    def timestamp(self):
        return self._timestamp

    def _interrupted(self):
        # StreamedResponse.get() gives cancellation precedence only when the
        # response is explicitly incomplete. Do not call session.cancel() after
        # the facade has already consumed and cleared its interruption.
        self.state = "incomplete"
        self._cancelled = True

    def _stopped(self, reason):
        self.finish_reason = {
            "stop_token": "stop",
            "max_tokens": "length",
            "tool_call_complete": "tool_call",
            "parser_error": "error",
            "context_exhausted": "error",
        }.get(reason)
        if reason == "interrupted":
            self._interrupted()

    def _finalize(self):
        if self._bound:
            return
        self._usage = RequestUsage(
            input_tokens=self._prefilled,
            output_tokens=self._output_tokens,
            details={"prefill_tokens": self._prefilled},
        )
        # A round that never reached its first sync never opened a response: its
        # assistant prefix (if any) was rolled back, so the only assistant/eos
        # segments left belong to an earlier response and must keep their
        # provenance. Only a started round owns the trailing response boundary.
        if self._sampled_start is not None:
            self._binder.bind_response(self.get())
        self._bound = True

    def __aiter__(self):
        # The harness wrappers do not propagate aclose() through every nested
        # async generator. This outer boundary owns early-close finalization.
        parent = super().__aiter__()

        async def iterate():
            try:
                async for event in parent:
                    yield event
            except asyncio.CancelledError:
                # Cleanup cancels the facade before closing the round; do not
                # re-arm cancellation after it has drained and cleared the event.
                self._interrupted()
                raise
            finally:
                await self._finish()

        return iterate()

    def _dispatch(self, event):
        """Apply one facade event; return harness stream events to publish."""
        outputs = []
        if isinstance(event, ReserveReached):
            self._reserve_needed = event.needed
            budget = (
                self._model.reserve_handler(event.needed)
                if self._model.reserve_handler is not None else None
            )
            if budget is None:
                event.fail()
            else:
                event.resume(budget)
        elif isinstance(event, ToolSyntax):
            self._in_tool_syntax = event.state != "DONE"
        elif isinstance(event, TextDelta):
            if not self._in_tool_syntax:
                if event.thinking:
                    outputs.extend(self._parts_manager.handle_thinking_delta(
                        vendor_part_id="thinking", content=event.text,
                    ))
                else:
                    outputs.extend(self._parts_manager.handle_text_delta(
                        vendor_part_id="content", content=event.text,
                    ))
        elif isinstance(event, ToolCalls):
            for call in event.calls:
                index = self._tool_index
                self._tool_index += 1
                outputs.append(self._parts_manager.handle_tool_call_part(
                    vendor_part_id=f"tool:{index}",
                    tool_name=call.name,
                    args=dict(call.args),
                    tool_call_id=f"{self._response_key}:tool:{index}",
                    provider_details={"ds4py.raw": call.raw},
                ))
        elif isinstance(event, Token):
            self._output_tokens += 1
        elif isinstance(event, Synced):
            if self._sampled_start is None:
                self._sampled_start = event.pos
            # A reserve-resumed round may sync again; that later sync reports
            # only the work it did, so total prefill is the sum of all syncs.
            self._prefilled += event.prefilled
        elif isinstance(event, Stopped):
            self._stopped(event.reason)
            if event.reason == "context_exhausted":
                self._context_exhausted = True
        elif isinstance(event, Interrupted):
            self._interrupted()
        return outputs

    async def _pull(self):
        async with self._pull_lock:
            if self._round_exhausted:
                return _END
            self._round_started = True
            pull = asyncio.get_running_loop().run_in_executor(
                None, _next, self._round,
            )
            try:
                return await asyncio.shield(pull)
            except asyncio.CancelledError:
                # The executor thread cannot be cancelled; keep the future so a
                # cancellation drain can finish the already-running pull.
                self._pending_pull = pull
                raise

    async def _drain_locked(self, pull):
        """Consume facade events until interruption or facade completion."""
        self._round_started = True
        while not self._round_exhausted:
            if pull is None:
                pull = asyncio.get_running_loop().run_in_executor(
                    None, _next, self._round,
                )
            try:
                event = await asyncio.shield(pull)
            except asyncio.CancelledError:
                self._pending_pull = pull
                raise
            pull = None
            if event is _END:
                self._round_exhausted = True
                return
            self._dispatch(event)
            if isinstance(event, (Stopped, Interrupted)):
                return

    async def _close_round_locked(self):
        if self._reconciled:
            return
        pending, self._pending_pull = self._pending_pull, None
        if pending is not None:
            try:
                event = await asyncio.shield(pending)
            except asyncio.CancelledError:
                self._pending_pull = pending
                raise
            if event is not _END:
                self._dispatch(event)
        await asyncio.to_thread(self._round.close)
        sampled_start = (
            self._sampled_start if self._sampled_start is not None else self._round_start
        )
        cutoff = sampled_start + self._output_tokens

        def reconcile():
            self._session.truncate(cutoff)
            self._session._worker.call(
                lambda engine, native: native.rewind(cutoff)
            ).result()
            # The facade round is closed, so no job remains to observe and
            # clear a cancellation that arrived after the last sample.
            self._session._worker.cancel_event.clear()

        await asyncio.to_thread(reconcile)
        self._round_exhausted = True
        self._reconciled = True

    async def _cleanup(self):
        """Terminal drain, close, reconcile, and finalization in one owner."""
        try:
            async with self._pull_lock:
                pending, self._pending_pull = self._pending_pull, None
                if pending is not None:
                    # A cancelled pull already has a sample in flight. Finish and
                    # publish it, then drain to the facade's terminal event.
                    # Closing without draining would leave that pull executing.
                    await self._drain_locked(pending)
                await self._close_round_locked()
        finally:
            self._finalize()

    async def _finish(self):
        # Every terminal path awaits this same task, so drain/close/reconcile
        # runs once even when the consumer and the harness race to close.
        task = self._cleanup_task
        if task is None:
            task = self._cleanup_task = asyncio.create_task(self._cleanup())
        # AnyIO propagates request-deadline/level cancellation into every
        # checkpoint. Shield the whole teardown, not just the executor future,
        # so cleanup cannot be interrupted halfway through reconciliation.
        primary = sys.exc_info()[0] is not None
        try:
            with anyio.CancelScope(shield=True):
                await asyncio.shield(task)
        except BaseException:
            # A timeout/cancellation already unwinding keeps its own surface;
            # a teardown failure must not replace the original exception.
            if not primary:
                raise

    async def _get_event_iterator(self):
        try:
            while True:
                try:
                    event = await self._pull()
                except asyncio.CancelledError:
                    self._session.cancel()
                    self._interrupted()
                    raise
                if event is _END:
                    # A naturally finished facade round already closed its
                    # assistant with EOS; only early/interrupted cleanup rolls
                    # that uncommitted tail back.
                    self._round_exhausted = True
                    self._reconciled = True
                    break
                for part_event in self._dispatch(event):
                    yield part_event
                if self._context_exhausted:
                    raise ContextReserveError(
                        _context_reserve_message(self._reserve_needed)
                    )
        finally:
            await self._finish()

    async def close_stream(self):
        self._cancelled = True
        self._interrupted()
        if self._round_started and not self._round_exhausted:
            self._session.cancel()
        await self._finish()
