"""Bind harness message history to the engine's single token transcript."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
)

from ds4py._json import is_string_key_mapping
from ds4py.binding_plan import (
    BindingChange,
    BindingConfig,
    BindingPlan,
    BindingState,
    PreparedMessages,
    ScratchSession,
    plan_transition,
    request_instructions,
)
from ds4py.binding_provenance import (
    BoundMessage,
    _content_hash,
    _digest,
    _ensure_key,
    _key_recovery_hash,
    _validate,
    keyed_messages,
    prepare_provenance,
)
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.tool_prompt import ToolDefinition
from ds4py.engine.transcript import TranscriptReplacement
from ds4py.errors import EngineError
from ds4py.message_rendering import MessageRenderer
from ds4py.store import SegmentProvenance

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pydantic_ai.models import ModelRequestParameters

    from ds4py.binding_plan import BindingBoundary
    from ds4py.binding_provenance import PreparedProvenance
    from ds4py.capabilities.system_notes import DS4SystemNotes
    from ds4py.engine.session import Session
    from ds4py.engine.session_operations import BindingTransaction


def _definitions(tools: Iterable[object]) -> tuple[ToolDefinition, ...]:
    definitions = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if not isinstance(name, str):
            raise TypeError("tool name must be a string")
        description = getattr(tool, "description", None)
        if description is not None and not isinstance(description, str):
            raise TypeError("tool description must be a string or None")
        if hasattr(tool, "parameters_json_schema"):
            parameters = tool.parameters_json_schema
        else:
            parameters = getattr(tool, "parameters", None)
        if not is_string_key_mapping(parameters):
            raise TypeError("tool parameters must be a string-keyed mapping")
        definitions.append(ToolDefinition(name, description or "", parameters))
    return tuple(definitions)


def _tool_digest(tools: Iterable[ToolDefinition]) -> str:
    return _digest(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]
    )


class MessageBinder:
    """Longest-valid-prefix binding keyed by persistent message key and content hash.

    The binder belongs to one session's consuming thread. ``prepare`` requires
    an idle native round. ``bind_response`` runs during response finalization,
    before that round closes.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self._renderer = MessageRenderer(session.engine)
        self.note_provider: DS4SystemNotes | None = None
        self._records: list[BoundMessage] = []
        self._bound_messages: list[ModelMessage] = []
        self._signature = None
        self._restored = False
        self._restored_notes: dict[str, list[tuple[str, str | None]]] = {}

    def render_request(self, message: ModelRequest, *, prefix: TranscriptReplacement) -> TranscriptReplacement:
        """Render a keyed request after an exact prefix, without an assistant prefix."""
        scratch = ScratchSession(self.session.engine)
        scratch._transcript.replace(prefix.tokens, prefix.segments)
        request = deepcopy(message)
        key = _ensure_key(request)
        self._renderer.append(scratch, request, key, _content_hash(request))
        return TranscriptReplacement(scratch.transcript.tokens, scratch.transcript.segments)

    def forget(self) -> None:
        """Drop binding state so the next request renders the message list fresh."""
        self._records = []
        self._bound_messages = []
        self._signature = None
        self._restored = False
        self._restored_notes = {}

    def restore_notes(self, segments: Sequence[SegmentProvenance]) -> None:
        """Schedule exact saved notes before the message keys that followed them."""
        restored: dict[str, list[tuple[str, str | None]]] = {}
        pending: list[tuple[str, str | None]] = []
        for segment in segments:
            if segment.kind == "system_note" or segment.kind.startswith("system_note:"):
                text = segment.text
                if not isinstance(text, str):
                    raise ValueError("saved note text is required for rendered restore")
                subtype = segment.kind.partition(":")[2] or None
                pending.append((text, subtype))
            elif segment.key and pending:
                restored.setdefault(segment.key, []).extend(pending)
                pending = []
        if pending:
            raise ValueError("saved note has no following message boundary")
        self._restored_notes = restored

    def adopt(self, messages: Sequence[ModelMessage]) -> list[ModelMessage]:
        """Reconcile bound content by message key without truncating harness history.

        Recover exact merged request boundaries first, just as for model input.
        The incoming sequence is authoritative: it may contain history omitted
        from the model's request view. A candidate with the same stable key can
        inherit its bound counterpart's content (for example, a partial response
        that the harness later completed). An unkeyed processed copy can recover
        its key only from a unique match that differs at most by request instructions.
        """
        candidates = self.canonicalize(list(messages))
        _validate(candidates)
        bound = {record.key: index for index, record in enumerate(self._records)}
        recoverable: dict[str, list[int]] = {}
        for index, message in enumerate(self._bound_messages):
            recoverable.setdefault(_key_recovery_hash(message), []).append(index)
        adopted = []
        for candidate in candidates:
            adopted_candidate = candidate
            key = (candidate.metadata or {}).get("ds4.key")
            index = bound.get(key) if key is not None else None
            if index is None and not key:
                matches = recoverable.get(_key_recovery_hash(candidate), [])
                if len(matches) == 1:
                    index = matches[0]
            if index is not None and _content_hash(candidate) != self._records[index].content_hash:
                adopted_candidate = deepcopy(self._bound_messages[index])
            elif index is not None:
                sampled = (self._bound_messages[index].metadata or {}).get("ds4.sampled")
                if sampled is not None:
                    metadata = dict(adopted_candidate.metadata or {})
                    metadata["ds4.sampled"] = deepcopy(sampled)
                    adopted_candidate.metadata = metadata
            adopted.append(adopted_candidate)
        keyed_messages(adopted, self._records)
        return adopted

    def adopt_failed(self, messages: Sequence[ModelMessage]) -> list[ModelMessage]:
        """Drop an unexecuted suffix while preserving bound trailing requests."""
        bound_keys = {record.key for record in self._records}
        adopted = self.adopt(messages)
        bound_indices = [
            index for index, message in enumerate(adopted) if (message.metadata or {}).get("ds4.key") in bound_keys
        ]
        if not bound_indices:
            return []
        return adopted[: bound_indices[-1] + 1]

    def canonicalize(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Undo only a unique exact harness merge of adjacent bound requests.

        The harness normalizes after capability hooks, discarding application
        keys when merging requests. Recover known part boundaries, never an
        unrelated or deliberately changed request-only view.
        """
        canonical: list[ModelMessage] = []
        changed = False
        for message in messages:
            matches: list[list[ModelMessage]] = []
            if isinstance(message, ModelRequest) and not (message.metadata or {}).get("ds4.key"):
                for start, first in enumerate(self._bound_messages):
                    if not isinstance(first, ModelRequest):
                        continue
                    parts = list(first.parts)
                    instructions = first.instructions
                    for end in range(start + 1, len(self._bound_messages)):
                        following = self._bound_messages[end]
                        if not isinstance(following, ModelRequest):
                            break
                        if (
                            bool(instructions)
                            and bool(following.instructions)
                            and instructions != following.instructions
                        ):
                            break
                        instructions = instructions or following.instructions
                        parts.extend(following.parts)
                        if parts == message.parts and instructions == message.instructions:
                            matches.append(self._bound_messages[start : end + 1])
            if len(matches) == 1:
                canonical.extend(deepcopy(matches[0]))
                changed = True
            else:
                canonical.append(message)
        return canonical if changed else messages

    def records(self) -> tuple[BoundMessage, ...]:
        """The keyed ranges currently bound to the transcript."""
        return tuple(self._records)

    def apply_restore(self, prepared: PreparedProvenance) -> None:
        """Assign restored state after the session installs a loaded payload."""
        self._records = list(prepared.records)
        self._bound_messages = list(prepared.bound_messages)
        self._signature = None
        self._restored = True

    def apply_compaction(self, prepared: PreparedProvenance) -> None:
        """Assign prepared records while keeping the current request signature."""
        self._records = list(prepared.records)
        self._bound_messages = list(prepared.bound_messages)

    def _restored_preamble(
        self,
        definitions: Sequence[ToolDefinition],
        instructions: str | None,
    ) -> bool:
        """True when the loaded transcript still starts with the current preamble."""
        if not self.session.transcript.chat:
            return False
        segments = PromptBuilder(self.session.engine).begin(definitions, instructions)
        preamble = tuple(token for segment in segments for token in segment.tokens)
        loaded = self.session.transcript.tokens
        return bool(preamble) and len(preamble) <= len(loaded) and loaded[: len(preamble)] == preamble

    def validate_restore(
        self,
        messages: Sequence[ModelMessage],
        segments: Sequence[SegmentProvenance],
    ) -> None:
        """Validate saved history without installing a native transcript."""
        prepare_provenance(messages, segments, loaded=None)

    def restore(
        self,
        messages: Sequence[ModelMessage],
        segments: Sequence[SegmentProvenance],
    ) -> None:
        """Bind a loaded native payload to a saved message history.

        ``segments`` is a save record's provenance. The transcript already holds
        the loaded tokens, so preparation rebuilds the prefix records without
        rendering anything, the session installs the saved ranges, and the
        binder then assigns the prepared records. Snapshot messages are matched
        to keyed ranges in order and by content hash: a record and its snapshot
        are written by one save, so the order is fixed, and the hash proves the
        snapshot still describes the tokens.
        """
        prepared = prepare_provenance(
            messages,
            segments,
            loaded=len(self.session.transcript),
        )
        self.session.install_segments(prepared.installed)
        self.apply_restore(prepared)

    def prepare_messages(self, messages: Sequence[ModelMessage]) -> PreparedMessages:
        """Canonicalize one request and isolate the copies used for planning."""
        targets = tuple(self.canonicalize(list(messages)))
        return PreparedMessages(targets, tuple(deepcopy(message) for message in targets))

    def _binding_state(self) -> BindingState:
        """Snapshot the persisted planner state without exposing the binder."""
        return BindingState(
            tuple(self._records),
            tuple(self._bound_messages),
            self._signature,
            self._restored,
            MappingProxyType({key: tuple(notes) for key, notes in self._restored_notes.items()}),
            self.session,
            self._renderer,
        )

    def _system_changed(self, config: BindingConfig) -> bool:
        changed = self._signature != config.signature or not self.session.transcript.chat
        if self._restored and self._restored_preamble(config.definitions, config.instructions):
            return False
        return changed

    def _plan(
        self,
        prepared: PreparedMessages,
        config: BindingConfig,
        provider: DS4SystemNotes | None,
        *,
        system_changed: bool,
    ) -> BindingPlan:
        return plan_transition(self._binding_state(), prepared, config, provider, system_changed=system_changed)

    def _configured_config(
        self,
        tools: Iterable[object],
        instructions: str | None,
    ) -> BindingConfig:
        definitions = _definitions(tools)
        signature = (_tool_digest(definitions), instructions, int(self.session.engine.think))
        return BindingConfig(definitions, instructions, signature)

    def _publish_boundary(
        self,
        boundary: BindingBoundary,
        provider: DS4SystemNotes | None,
        targets: Sequence[ModelMessage],
        plan: BindingPlan,
    ) -> None:
        """Install one boundary's records, provider notes and keys, or restore it.

        Publication runs inside the Session-owned transaction, so a failure here
        restores this boundary's binder/provider/harness state and the Session
        restores the group tokens together. Only this boundary's newly committed
        message range is captured, published and restored; earlier boundaries
        stay committed and are never rewritten.
        """
        previous_state = (self._records, self._bound_messages, self._restored_notes)
        previous_notes = provider.checkpoint() if provider is not None else None
        start, end = boundary.start_messages, boundary.end_messages
        published = [(message, message.metadata) for message in targets[start:end]]
        try:
            self._records = list(boundary.records)
            self._bound_messages = list(boundary.bound_messages)
            self._restored_notes = {key: list(notes) for key, notes in boundary.restored_notes}
            if provider is not None and boundary.notes is not None:
                provider.apply_preview(boundary.notes)
            self._publish_keys(targets, plan, start_messages=start, end_messages=end)
        except BaseException as error:
            self._records, self._bound_messages, self._restored_notes = previous_state
            for message, metadata in published:
                message.metadata = metadata
            if provider is not None and previous_notes is not None:
                try:
                    provider.apply_preview(previous_notes)
                except BaseException as nested:  # noqa: BLE001 - Keep the publication error primary.
                    error.add_note(f"system notes could not be restored: {nested}")
            raise

    def prepare(
        self,
        messages: Sequence[ModelMessage],
        *,
        tools: Iterable[object] = (),
        instructions: str | None = None,
        note_provider: DS4SystemNotes | None = None,
    ) -> BindingChange:
        """Apply one planned transition inside a claimed Session transaction.

        The claim is taken before planning, so preparation is admitted only
        between rounds: it rejects while a round is open, while another
        exclusive operation is active, and while a binding transaction already
        holds the claim. Observation-only callers use the unclaimed counting and
        planning entries instead.
        """
        _, change = self._prepare_claimed(
            messages,
            tools=tools,
            instructions=instructions,
            note_provider=note_provider,
        )
        return change

    def prepare_model_request(
        self,
        messages: Sequence[ModelMessage],
        params: ModelRequestParameters,
        *,
        note_provider: DS4SystemNotes | None = None,
    ) -> list[ModelMessage]:
        """Prepare one model request and return its canonical publication targets.

        The Session binding transaction is claimed before canonicalization, so
        the plan is built and published from exactly one preprocessing pass and
        its targets become the response view without a second canonicalization.
        """
        prepared, _ = self._prepare_claimed(
            messages,
            tools=params.declared_tool_defs.values(),
            params=params,
            note_provider=note_provider,
        )
        canonical = prepared.targets
        if (
            isinstance(messages, list)
            and len(canonical) == len(messages)
            and all(target is message for target, message in zip(canonical, messages, strict=True))
        ):
            # Without a recovered merge the harness list is already the response
            # view; keep its identity so the reserve owner mutates what the
            # stream reads back.
            return cast("list[ModelMessage]", messages)
        return list(canonical)

    def _prepare_claimed(
        self,
        messages: Sequence[ModelMessage],
        *,
        tools: Iterable[object] = (),
        instructions: str | None = None,
        params: ModelRequestParameters | None = None,
        note_provider: DS4SystemNotes | None = None,
    ) -> tuple[PreparedMessages, BindingChange]:
        """Canonicalize, plan and apply one request inside the Session claim.

        ``params`` resolves the effective instructions from the canonical
        targets; otherwise the explicit ``instructions`` argument is planned.
        """
        provider = note_provider if note_provider is not None else self.note_provider
        with self.session.binding_transaction() as transaction:
            prepared = self.prepare_messages(messages)
            effective = request_instructions(prepared.targets, params) if params is not None else instructions
            config = self._configured_config(tools, effective)
            plan = self._plan(prepared, config, provider, system_changed=self._system_changed(config))
            return prepared, self._apply_plan(transaction, plan, provider, prepared.targets)

    def _apply_plan(
        self,
        transaction: BindingTransaction,
        plan: BindingPlan,
        provider: DS4SystemNotes | None,
        targets: Sequence[ModelMessage],
    ) -> BindingChange:
        """Apply one prepared plan through the claimed Session transaction."""
        if plan.truncate_to is not None:
            transaction.truncate(plan.truncate_to)
        if plan.reset_chat:
            self._signature = None
            self._restored = False
        transaction.apply_appends(
            plan.initial.appends,
            publish=partial(self._publish_boundary, plan.initial, provider, targets, plan),
        )
        if plan.reset_chat:
            transaction.start_chat(plan.definitions, plan.instructions)
        self._signature = plan.signature
        self._restored = False
        for boundary in plan.boundaries:
            transaction.apply_appends(
                boundary.appends,
                retire_paused=boundary.retire_paused,
                publish=partial(self._publish_boundary, boundary, provider, targets, plan),
            )
        return plan.change

    def _publish_keys(
        self,
        targets: Sequence[ModelMessage],
        plan: BindingPlan,
        *,
        start_messages: int,
        end_messages: int,
    ) -> None:
        """Publish planned metadata for only the newly committed message range."""
        if not 0 <= start_messages <= end_messages <= len(targets) or end_messages > len(plan.messages):
            raise AssertionError("published key range must align with planned targets")
        for index in range(start_messages, end_messages):
            planned = plan.messages[index].metadata
            target = targets[index]
            if target.metadata == planned:
                continue
            target.metadata = deepcopy(planned)

    def project_request(
        self,
        messages: Sequence[ModelMessage],
        *,
        tools: Iterable[object] = (),
        instructions: str | None = None,
        bound: bool = False,
    ) -> int:
        """Project the exact request preparation would apply, without live mutation.

        ``bound`` projects against the initialized native preamble and sampled
        binding, rejecting before preprocessing when no chat exists. The
        configured form projects the request that ``prepare`` would apply.
        """
        if bound and not self.session.transcript.chat:
            raise EngineError("bound request requires an initialized chat")
        prepared = self.prepare_messages(messages)
        if bound:
            config = BindingConfig((), None, self._signature)
            plan = self._plan(prepared, config, self.note_provider, system_changed=False)
            return plan.projected_tokens
        config = self._configured_config(tools, instructions)
        plan = self._plan(prepared, config, self.note_provider, system_changed=self._system_changed(config))
        return plan.projected_tokens

    def bind_response(self, response: ModelResponse) -> None:
        key = _ensure_key(response)
        if response.metadata is None:
            raise AssertionError("_ensure_key must initialize metadata")
        response.metadata["ds4.source"] = "sampled"
        content_hash = _content_hash(response)
        segments = self.session.transcript.segments
        prefix_index = max(
            (index for index, segment in enumerate(segments) if segment.kind == "assistant_prefix"),
            default=None,
        )
        trailing = [
            segment
            for segment in segments[(prefix_index + 1) if prefix_index is not None else 0 :]
            if segment.kind in ("assistant", "eos")
        ]
        sampled_start = trailing[0].start if trailing else len(self.session.transcript)
        response.metadata["ds4.sampled"] = {
            "content_hash": content_hash,
            "segments": [
                {
                    "kind": segment.kind,
                    "tokens": list(self.session.transcript.tokens[segment.start : segment.end]),
                }
                for segment in trailing
            ],
        }
        self.session._transcript.annotate(
            sampled_start,
            kinds=("assistant", "eos"),
            meta={"key": key, "content_hash": content_hash, "source": "sampled"},
        )
        start = segments[prefix_index].start if prefix_index is not None else sampled_start
        record = BoundMessage(key, content_hash, start, len(self.session.transcript))
        if self._records and self._records[-1].key == key and self._records[-1].start == start:
            self._records[-1] = record
            self._bound_messages[-1] = deepcopy(response)
        else:
            self._records.append(record)
            self._bound_messages.append(deepcopy(response))

    def export_segments(self) -> tuple[SegmentProvenance, ...]:
        exported = []
        for segment in self.session.transcript.segments:
            kind = segment.kind
            text = segment.meta.get("text") if kind == "note" else None
            if kind == "note":
                subtype = segment.meta.get("subtype")
                kind = f"system_note:{subtype}" if subtype else "system_note"
            exported.append(
                SegmentProvenance(
                    segment.start,
                    segment.end,
                    kind,
                    str(segment.meta.get("key", "")),
                    str(segment.meta.get("content_hash", "")),
                    str(segment.meta.get("source", "rendered")),
                    text=str(text) if text is not None else None,
                )
            )
        return tuple(exported)
