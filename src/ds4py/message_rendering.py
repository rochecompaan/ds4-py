"""Render canonical harness messages into engine token segments."""

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, TypeIs, cast

from pydantic_ai.messages import (
    InstructionBaselinePart,
    InstructionDeltaPart,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ds4py.engine.prompt import PromptBuilder, RenderedSegment
from ds4py.engine.tool_syntax.render import render_tool_call

if TYPE_CHECKING:
    from ds4py.engine.session import Engine
    from ds4py.engine.transcript import Segment, TranscriptView

DSML_FAMILIES = ("deepseek4", "deepseek41")


class RenderTarget(Protocol):
    @property
    def transcript(self) -> TranscriptView: ...

    def append_tokens(
        self,
        tokens: Sequence[int],
        kind: str,
        meta: Mapping[str, object] | None = None,
    ) -> Segment: ...


def stable_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping) or (isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _is_sampled(message: ModelResponse) -> bool:
    return bool((message.metadata or {}).get("ds4.source") == "sampled")


def _content_parts(message: ModelResponse) -> list[object]:
    return [part for part in message.parts if not isinstance(part, (TextPart, ThinkingPart)) or part.content]


def _is_item_sequence(value: object) -> TypeIs[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_token_id(value: object) -> TypeIs[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _last_subsequence_index(haystack: Sequence[int], needle: Sequence[int]) -> int:
    for index in range(len(haystack) - len(needle), -1, -1):
        if haystack[index : index + len(needle)] == needle:
            return index
    return -1


def _dsml_calls_tag(family: str) -> str:
    return " calls" if family == "deepseek41" else "tool_calls"


def _recorded_sampled_segments(
    message: ModelResponse,
    content_hash: str,
) -> list[RenderedSegment] | None:
    """Return exact sampled IDs when they still belong to this response content."""
    record = (message.metadata or {}).get("ds4.sampled")
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise ValueError("sampled response record must be a mapping")
    if record.get("content_hash") != content_hash:
        return None
    raw_segments = record.get("segments")
    if not _is_item_sequence(raw_segments):
        raise ValueError("sampled response segments must be a sequence")
    segments = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            raise ValueError("sampled response segment must be a mapping")
        kind = raw_segment.get("kind")
        if kind not in {"assistant", "eos"}:
            raise ValueError("sampled response segment kind must be assistant or eos")
        raw_tokens = raw_segment.get("tokens")
        if not _is_item_sequence(raw_tokens):
            raise ValueError("sampled response segment tokens must be a sequence")
        if not all(_is_token_id(token) for token in raw_tokens):
            raise ValueError("sampled response token IDs must be non-negative integers")
        tokens = list(cast("Sequence[int]", raw_tokens))
        segments.append(RenderedSegment(kind, tokens))
    return segments


def _retry_text(part: RetryPromptPart) -> str:
    text = stable_text(part.content)
    return text if text.startswith("Tool error:") else "Tool error: " + text


class MessageRenderer:
    """Render one message history without owning prefix-binding state."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def request_segments(
        self,
        builder: PromptBuilder,
        prefix: Sequence[int],
        message: ModelRequest,
    ) -> list[RenderedSegment]:
        segments: list[RenderedSegment] = []

        def append(role: str, text: str) -> None:
            nonlocal prefix
            tokens = builder.message(prefix, role, text)
            if tokens:
                kind = "tool" if role in ("tool", "function") else role
                segments.append(RenderedSegment(kind, tokens))
                prefix = (*prefix, *tokens)

        parts = list(message.parts)
        index = 0
        while index < len(parts):
            part = parts[index]
            if isinstance(part, (InstructionBaselinePart, InstructionDeltaPart)):
                # Framework bookkeeping: the request's ``instructions`` carry the
                # prefix these parts describe, and no model is sent them.
                index += 1
            elif isinstance(part, (SystemPromptPart, InstructionPart)):
                append("system", part.content)
                index += 1
            elif isinstance(part, UserPromptPart):
                append("user", cast("str", part.content))
                index += 1
            else:
                results = []
                while index < len(parts) and isinstance(parts[index], (ToolReturnPart, RetryPromptPart)):
                    current = parts[index]
                    if isinstance(current, ToolReturnPart):
                        results.append(
                            f"Tool result {len(results) + 1} ({current.tool_name}):\n{stable_text(current.content)}"
                        )
                    else:
                        results.append(_retry_text(cast("RetryPromptPart", current)))
                    index += 1
                append("tool", "\n\n".join(results))
        return segments

    def _dsml_calls(self, parts: Sequence[ToolCallPart], sampled: bool) -> str:
        family = self.engine.info.family
        calls = _dsml_calls_tag(family)
        stanzas = []
        for part in parts:
            details = part.provider_details or {}
            raw = details.get("ds4py.raw")
            if sampled and isinstance(raw, str):
                stanzas.append(raw)
                continue
            markup = render_tool_call(family, part.tool_name, part.args)
            stanzas.append("\n".join(markup.split("\n")[1:-1]))
        return f"<｜DSML｜{calls}>\n{'\n'.join(stanzas)}\n</｜DSML｜{calls}>"

    def _response_text(self, parts: Sequence[object], index: int, sampled: bool) -> str:
        """Keep a sampled DSML opener only in the tool-call group that owns it."""
        text = cast("TextPart", parts[index]).content
        family = self.engine.info.family
        follows_tool_call = (
            sampled
            and family in DSML_FAMILIES
            and index + 1 < len(parts)
            and isinstance(parts[index + 1], ToolCallPart)
        )
        if not follows_tool_call:
            return text
        calls = _dsml_calls_tag(family)
        return text.removesuffix(f"<｜DSML｜{calls}>")

    @staticmethod
    def _assistant_prefix(
        builder: PromptBuilder,
        prefix: Sequence[int],
        parts: Sequence[object],
        sampled: bool,
    ) -> tuple[list[int], bool]:
        """Return the native assistant opener and whether it leaves thinking open."""
        if not sampled:
            return [], False
        assistant_prefix = builder.assistant_prefix(prefix)
        think_open = builder.engine.tokenize("<think>", rendered=True)
        think_close = builder.engine.tokenize("</think>", rendered=True)
        opens_thinking = _last_subsequence_index(assistant_prefix, think_open) > _last_subsequence_index(
            assistant_prefix, think_close
        )
        in_thinking = opens_thinking and bool(parts) and not isinstance(parts[0], ThinkingPart)
        return assistant_prefix, in_thinking

    @staticmethod
    def _assistant_end(builder: PromptBuilder, message: ModelResponse, sampled: bool) -> list[int]:
        """Reproduce the absent native EOS on a length-capped sampled response."""
        if sampled and message.finish_reason == "length":
            return []
        return builder.append_assistant_end()

    def _recorded_response_segments(
        self,
        builder: PromptBuilder,
        prefix: Sequence[int],
        message: ModelResponse,
        content_hash: str,
    ) -> list[RenderedSegment] | None:
        if not _is_sampled(message):
            return None
        recorded = _recorded_sampled_segments(message, content_hash)
        if recorded is None:
            return None
        parts = _content_parts(message)
        assistant_prefix, _ = self._assistant_prefix(builder, prefix, parts, sampled=True)
        return [RenderedSegment("assistant_prefix", assistant_prefix), *recorded]

    def response_segments(  # noqa: PLR0915 - Response rendering handles every message-part family.
        self,
        builder: PromptBuilder,
        prefix: Sequence[int],
        message: ModelResponse,
    ) -> list[RenderedSegment]:
        spans: list[tuple[str, bool]] = []
        sampled = _is_sampled(message)
        family = self.engine.info.family
        parts = _content_parts(message)
        assistant_prefix, in_thinking = self._assistant_prefix(builder, prefix, parts, sampled)

        def close_thinking() -> None:
            nonlocal in_thinking
            if sampled and in_thinking:
                spans.append(("</think>", True))
                in_thinking = False

        index = 0
        while index < len(parts):
            part = parts[index]
            if isinstance(part, TextPart):
                close_thinking()
                spans.append((self._response_text(parts, index, sampled), sampled))
                index += 1
            elif isinstance(part, ThinkingPart):
                if sampled:
                    # The native assistant prefix already opened this span.
                    spans.append((part.content, True))
                    in_thinking = True
                else:
                    spans.extend((("<think>", True), (part.content, False), ("</think>", True)))
                index += 1
            elif family in DSML_FAMILIES:
                close_thinking()
                end = index
                while end < len(parts) and isinstance(parts[end], ToolCallPart):
                    end += 1
                tool_parts = [part for part in parts[index:end] if isinstance(part, ToolCallPart)]
                if len(tool_parts) != end - index:
                    raise AssertionError("DSML tool-call group contains a non-tool part")
                spans.append((self._dsml_calls(tool_parts, sampled), True))
                index = end
            else:
                close_thinking()
                if not isinstance(part, ToolCallPart):
                    raise AssertionError("tool-only message part expected")
                details = part.provider_details or {}
                raw = details.get("ds4py.raw")
                markup = (
                    raw
                    if sampled and isinstance(raw, str)
                    else render_tool_call(
                        family,
                        part.tool_name,
                        part.args,
                    )
                )
                spans.append((markup, True))
                index += 1
        if sampled:
            body = []
            for text, rendered in spans:
                if text:
                    body.extend(builder.engine.tokenize(text, rendered=rendered))
            segments = [
                RenderedSegment("assistant_prefix", assistant_prefix),
                RenderedSegment("assistant", body),
            ]
        else:
            segments = [
                RenderedSegment(
                    "assistant",
                    builder.restored_assistant(prefix, spans),
                )
            ]
        end = self._assistant_end(builder, message, sampled)
        if end:
            segments.append(RenderedSegment("eos", end))
        return segments

    def append(
        self,
        target: RenderTarget,
        message: ModelMessage,
        key: str,
        content_hash: str,
    ) -> tuple[int, int]:
        """Append one message and return its start and end token positions."""
        start = len(target.transcript)
        builder = PromptBuilder(self.engine)
        if isinstance(message, ModelRequest):
            segments = self.request_segments(builder, target.transcript.tokens, message)
        else:
            segments = self._recorded_response_segments(builder, target.transcript.tokens, message, content_hash)
            if segments is None:
                segments = self.response_segments(builder, target.transcript.tokens, message)
        meta = {"key": key, "content_hash": content_hash, "source": "rendered"}
        sampled = isinstance(message, ModelResponse) and _is_sampled(message)
        for segment in segments:
            if segment.tokens:
                segment_meta = None if sampled and segment.kind == "assistant_prefix" else meta
                target.append_tokens(segment.tokens, segment.kind, segment_meta)
        return start, len(target.transcript)

    def preflight(self, messages: Iterable[ModelMessage]) -> None:
        family = self.engine.info.family
        for message in messages:
            if not isinstance(message, ModelResponse):
                continue
            sampled = _is_sampled(message)
            saw_call = separated = False
            for part in message.parts:
                if isinstance(part, TextPart):
                    separated = separated or (saw_call and bool(part.content))
                    continue
                if not isinstance(part, ToolCallPart):
                    continue
                if separated:
                    raise ValueError("cannot restore a tool call after ordinary text following a call")
                saw_call = True
                details = part.provider_details or {}
                if sampled and isinstance(details.get("ds4py.raw"), str):
                    continue
                render_tool_call(family, part.tool_name, part.args)
