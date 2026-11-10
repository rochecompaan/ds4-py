"""Count a pending request using the exact bound native prefix, without syncing."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelRequest

from ds4py.binding_provenance import _content_hash
from ds4py.engine.prompt import PromptBuilder
from ds4py.message_rendering import MessageRenderer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage

    from ds4py.binding import MessageBinder, _ScratchSession


def count_bound_request(binder: MessageBinder, scratch: _ScratchSession, messages: Sequence[ModelMessage]) -> int:
    """Reuse sampled tokens and notes through the common bound message prefix.

    The scratch target has the current request's rendered system preamble.
    Everything after the first changed message uses the ordinary renderer.
    """
    candidates = binder.canonicalize(deepcopy(list(messages)))
    binder._keyed(candidates)
    records = binder.records()
    common = 0
    for message, record in zip(candidates, records):
        if (message.metadata or {}).get("ds4.key") != record.key or _content_hash(message) != record.content_hash:
            break
        common += 1
    transcript = binder.session.transcript
    preamble_end = records[0].start
    if common:
        end = records[common - 1].end
        for segment in transcript.segments:
            if segment.start >= preamble_end and segment.end <= end:
                scratch.append_tokens(transcript.tokens[segment.start : segment.end], segment.kind, segment.meta)
    if common == len(records) and common < len(candidates) and binder.session._paused_stream is not None:
        # prepare() retires a paused response only when appending a suffix to
        # the complete binding. Unchanged/resliced histories do not take that path.
        ending = PromptBuilder(binder.session.engine).append_assistant_end()
        if ending:
            scratch.append_tokens(ending, "eos")
    renderer = MessageRenderer(binder.session.engine)
    for index, message in enumerate(candidates[common:], common):
        if isinstance(message, ModelRequest) and binder.note_provider is not None:
            for text, subtype in binder.note_provider.pending(candidates[: index + 1]):
                tokens = PromptBuilder(binder.session.engine).message(scratch.transcript.tokens, "system", "\n" + text)
                scratch.append_tokens(tokens, "note", {"subtype": subtype})
        renderer.append(scratch, message, str((message.metadata or {}).get("ds4.key", "")), _content_hash(message))
    prefix = PromptBuilder(binder.session.engine).assistant_prefix(scratch.transcript.tokens)
    return len(scratch.transcript) + len(prefix)
