"""Render canonical harness messages into engine token segments."""
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, cast

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
from ds4py.engine.session import Engine
from ds4py.engine.transcript import Segment, TranscriptView
from ds4py.engine.tool_syntax.render import render_tool_call


class RenderTarget(Protocol):
    @property
    def transcript(self) -> TranscriptView: ...

    def append_tokens(
        self, tokens: Sequence[int], kind: str, meta: Mapping[str, object] | None = None,
    ) -> Segment: ...


def stable_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _retry_text(part: RetryPromptPart) -> str:
    text = stable_text(part.content)
    return text if text.startswith("Tool error:") else "Tool error: " + text


class MessageRenderer:
    """Render one message history without owning prefix-binding state."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def request_segments(
        self, builder: PromptBuilder, prefix: Sequence[int], message: ModelRequest,
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
                append("user", cast(str, part.content))
                index += 1
            else:
                results = []
                while index < len(parts) and isinstance(
                    parts[index], (ToolReturnPart, RetryPromptPart)
                ):
                    current = parts[index]
                    if isinstance(current, ToolReturnPart):
                        results.append(
                            f"Tool result {len(results) + 1} ({current.tool_name}):\n"
                            f"{stable_text(current.content)}"
                        )
                    else:
                        results.append(_retry_text(cast(RetryPromptPart, current)))
                    index += 1
                append("tool", "\n\n".join(results))
        return segments

    def _dsml_calls(self, parts: Sequence[ToolCallPart], sampled: bool) -> str:
        family = self.engine.info.family
        calls = " calls" if family == "deepseek41" else "tool_calls"
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

    def response_segments(
        self, builder: PromptBuilder, prefix: Sequence[int], message: ModelResponse,
    ) -> list[RenderedSegment]:
        spans: list[tuple[str, bool]] = []
        sampled = (message.metadata or {}).get("ds4.source") == "sampled"
        family = self.engine.info.family
        parts = [
            part for part in message.parts
            if not isinstance(part, (TextPart, ThinkingPart)) or part.content
        ]
        in_thinking = False

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
                spans.append((part.content, sampled))
                index += 1
            elif isinstance(part, ThinkingPart):
                if sampled:
                    # The native assistant prefix already opened this span.
                    spans.append((part.content, True))
                    in_thinking = True
                else:
                    spans.extend((("<think>", True), (part.content, False), ("</think>", True)))
                index += 1
            elif family in ("deepseek4", "deepseek41"):
                close_thinking()
                end = index
                while end < len(parts) and isinstance(parts[end], ToolCallPart):
                    end += 1
                tool_parts = cast(list[ToolCallPart], parts[index:end])
                spans.append((self._dsml_calls(tool_parts, sampled), True))
                index = end
            else:
                close_thinking()
                part = cast(ToolCallPart, part)
                details = part.provider_details or {}
                raw = details.get("ds4py.raw")
                markup = raw if sampled and isinstance(raw, str) else render_tool_call(
                    family, part.tool_name, part.args,
                )
                spans.append((markup, True))
                index += 1
        if sampled:
            body = []
            for text, rendered in spans:
                if text:
                    body.extend(builder.engine.tokenize(text, rendered=rendered))
            segments = [
                RenderedSegment("assistant_prefix", builder.assistant_prefix(prefix)),
                RenderedSegment("assistant", body),
            ]
        else:
            segments = [RenderedSegment(
                "assistant", builder.restored_assistant(prefix, spans),
            )]
        end = builder.append_assistant_end()
        if end:
            segments.append(RenderedSegment("eos", end))
        return segments

    def append(
        self, target: RenderTarget, message: ModelMessage, key: str, content_hash: str,
    ) -> tuple[int, int]:
        """Append one message and return its start and end token positions."""
        start = len(target.transcript)
        builder = PromptBuilder(self.engine)
        if isinstance(message, ModelRequest):
            segments = self.request_segments(builder, target.transcript.tokens, message)
        else:
            segments = self.response_segments(builder, target.transcript.tokens, message)
        meta = {"key": key, "content_hash": content_hash, "source": "rendered"}
        sampled = isinstance(message, ModelResponse) and (
            (message.metadata or {}).get("ds4.source") == "sampled"
        )
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
            sampled = (message.metadata or {}).get("ds4.source") == "sampled"
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
