"""Bind harness message history to the engine's single token transcript."""
from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from uuid import uuid4

from pydantic_ai.messages import (
    InstructionPart,
    ModelMessagesTypeAdapter,
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

from ds4py.engine import chat as chat_ops
from ds4py.engine.prompt import PromptBuilder, RenderedSegment
from ds4py.engine.tool_prompt import ToolDefinition
from ds4py.engine.tool_syntax.render import render_tool_call
from ds4py.engine.transcript import Transcript, TranscriptView
from ds4py.errors import EngineError


@dataclass(frozen=True)
class BindingChange:
    kind: str
    start: int
    continuation: bool


@dataclass(frozen=True)
class SegmentProvenance:
    start: int
    end: int
    kind: str
    key: str
    content_hash: str
    source: str


@dataclass(frozen=True)
class _BoundMessage:
    key: str
    content_hash: str
    start: int
    end: int


class _ScratchSession:
    """Rendering target with no native session/KV state."""

    def __init__(self, engine):
        self._engine = engine
        self._transcript = Transcript()
        self.transcript = TranscriptView(self._transcript)

    def start_chat(self, tools=(), instructions=None):
        for segment in PromptBuilder(self._engine).begin(tools, instructions):
            self._transcript.append(segment.tokens, segment.kind)
        self._transcript.set_chat()

    def append_tokens(self, tokens, kind, meta=None):
        return self._transcript.append(tokens, kind, meta)


def _stable_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _definitions(tools) -> tuple[ToolDefinition, ...]:
    return tuple(
        ToolDefinition(
            tool.name,
            tool.description or "",
            tool.parameters_json_schema if hasattr(tool, "parameters_json_schema")
            else tool.parameters,
        )
        for tool in tools
    )


def _digest(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tool_digest(tools) -> str:
    return _digest([
        {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
        for tool in tools
    ])


def _ensure_key(message) -> str:
    metadata = dict(message.metadata or {})
    key = metadata.get("ds4.key")
    if not key:
        key = uuid4().hex
        metadata["ds4.key"] = key
    message.metadata = metadata
    return key


def _arguments(args):
    if isinstance(args, str):
        args = json.loads(args)
    return dict(args or {})


def _projection(message) -> dict:
    dumped = ModelMessagesTypeAdapter.dump_python([message], warnings=False)[0]
    projected = {"kind": dumped["kind"], "parts": []}
    if isinstance(message, ModelRequest) and message.instructions is not None:
        projected["instructions"] = message.instructions
    for part in dumped["parts"]:
        kind = part["part_kind"]
        if kind in ("system-prompt", "instruction", "user-prompt", "text", "thinking"):
            projected["parts"].append({"kind": kind, "content": part["content"]})
        elif kind == "tool-call":
            item = {
                "kind": kind,
                "name": part["tool_name"],
                "args": _arguments(part["args"]),
                "id": part["tool_call_id"],
            }
            details = part.get("provider_details") or {}
            raw = details.get("ds4py.raw")
            if (message.metadata or {}).get("ds4.source") == "sampled" and isinstance(raw, str):
                item["raw"] = raw
            projected["parts"].append(item)
        elif kind == "tool-return":
            projected["parts"].append({
                "kind": kind,
                "name": part["tool_name"],
                "content": _stable_text(part["content"]),
                "id": part["tool_call_id"],
            })
        elif kind == "retry-prompt":
            projected["parts"].append({
                "kind": kind,
                "name": part["tool_name"],
                "content": _stable_text(part["content"]),
                "id": part["tool_call_id"],
            })
    return projected


def _content_hash(message) -> str:
    return _digest(_projection(message))


def _validate(messages) -> None:
    for message in messages:
        parts = message.parts
        for part in parts:
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                raise EngineError("image input requires M5 vision support")
            if isinstance(message, ModelRequest) and not isinstance(part, (
                SystemPromptPart, InstructionPart, UserPromptPart, ToolReturnPart, RetryPromptPart,
            )):
                raise EngineError(f"unsupported harness request part: {type(part).__name__}")
            if isinstance(message, ModelResponse) and not isinstance(
                part, (TextPart, ThinkingPart, ToolCallPart)
            ):
                raise EngineError(f"unsupported harness response part: {type(part).__name__}")


def _retry_text(part) -> str:
    text = _stable_text(part.content)
    return text if text.startswith("Tool error:") else "Tool error: " + text


class MessageBinder:
    """Longest-valid-prefix binding keyed by persistent message key and content hash."""

    def __init__(self, session):
        self.session = session
        self._records: list[_BoundMessage] = []
        self._signature = None

    def _request_segments(self, builder, prefix, message):
        segments = []

        def append(role, text):
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
            if isinstance(part, (SystemPromptPart, InstructionPart)):
                append("system", part.content)
                index += 1
            elif isinstance(part, UserPromptPart):
                append("user", part.content)
                index += 1
            else:
                results = []
                while index < len(parts) and isinstance(parts[index], (ToolReturnPart, RetryPromptPart)):
                    current = parts[index]
                    if isinstance(current, ToolReturnPart):
                        results.append(
                            f"Tool result {len(results) + 1} ({current.tool_name}):\n"
                            f"{_stable_text(current.content)}"
                        )
                    else:
                        results.append(_retry_text(current))
                    index += 1
                append("tool", "\n\n".join(results))
        return segments

    def _dsml_calls(self, parts, sampled):
        family = self.session.engine.info.family
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

    def _response_segments(self, builder, prefix, message):
        spans = []
        sampled = (message.metadata or {}).get("ds4.source") == "sampled"
        family = self.session.engine.info.family
        parts = [
            part for part in message.parts
            if not isinstance(part, (TextPart, ThinkingPart)) or part.content
        ]
        index = 0
        while index < len(parts):
            part = parts[index]
            if isinstance(part, TextPart):
                spans.append((part.content, False))
                index += 1
            elif isinstance(part, ThinkingPart):
                spans.extend((("<think>", True), (part.content, False), ("</think>", True)))
                index += 1
            elif family in ("deepseek4", "deepseek41"):
                end = index
                while end < len(parts) and isinstance(parts[end], ToolCallPart):
                    end += 1
                spans.append((self._dsml_calls(parts[index:end], sampled), True))
                index = end
            else:
                details = part.provider_details or {}
                raw = details.get("ds4py.raw")
                markup = raw if sampled and isinstance(raw, str) else render_tool_call(
                    family, part.tool_name, part.args,
                )
                spans.append((markup, True))
                index += 1
        segments = [RenderedSegment(
            "assistant", builder.restored_assistant(prefix, spans),
        )]
        end = builder.append_assistant_end()
        if end:
            segments.append(RenderedSegment("eos", end))
        return segments

    def _append_message(self, target, message, key, content_hash):
        start = len(target.transcript)
        builder = PromptBuilder(target._engine)
        if isinstance(message, ModelRequest):
            segments = self._request_segments(builder, target.transcript.tokens, message)
        else:
            segments = self._response_segments(builder, target.transcript.tokens, message)
        meta = {"key": key, "content_hash": content_hash, "source": "rendered"}
        for segment in segments:
            if segment.tokens:
                target.append_tokens(segment.tokens, segment.kind, meta)
        return _BoundMessage(key, content_hash, start, len(target.transcript))

    @staticmethod
    def _continuation(messages):
        return bool(
            messages and isinstance(messages[-1], ModelRequest)
            and messages[-1].parts
            and isinstance(messages[-1].parts[-1], (ToolReturnPart, RetryPromptPart))
        )

    def _preflight(self, messages):
        family = self.session.engine.info.family
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

    def _retire_paused(self):
        if self.session._paused_stream is None:
            return
        previous = self._records[-1]
        start = len(self.session.transcript)
        chat_ops.retire_paused(self.session)
        end = len(self.session.transcript)
        self.session._transcript.annotate(
            start,
            kinds=("eos",),
            meta={
                "key": previous.key,
                "content_hash": previous.content_hash,
                "source": "sampled",
            },
        )
        self._records[-1] = _BoundMessage(
            previous.key, previous.content_hash, previous.start, end,
        )

    def prepare(self, messages, *, tools=(), instructions=None) -> BindingChange:
        messages = list(messages)
        _validate(messages)
        keyed = [(_ensure_key(message), _content_hash(message)) for message in messages]
        definitions = _definitions(tools)
        signature = (_tool_digest(definitions), instructions, int(self.session.engine.think))
        had_binding = self._signature is not None
        system_changed = self._signature != signature or not self.session.transcript.chat

        if system_changed:
            self._preflight(messages)
            start = 0
            if len(self.session.transcript):
                self.session.truncate(0)
            self._records = []
            self.session.start_chat(definitions, instructions)
            self._signature = signature
            common = 0
            kind = "rewrite" if had_binding else "append"
        else:
            common = 0
            while (
                common < len(messages) and common < len(self._records)
                and keyed[common] == (self._records[common].key, self._records[common].content_hash)
            ):
                common += 1
            if common == len(messages) == len(self._records):
                return BindingChange("unchanged", len(self.session.transcript), self._continuation(messages))
            self._preflight(messages[common:])
            if common < len(self._records):
                start = self._records[common].start
                self.session.truncate(start)
                self._records = self._records[:common]
                kind = "rewrite"
            else:
                self._retire_paused()
                start = len(self.session.transcript)
                kind = "append"

        for message, (key, content_hash) in zip(messages[common:], keyed[common:]):
            self._records.append(self._append_message(
                self.session, message, key, content_hash,
            ))
        return BindingChange(kind, start, self._continuation(messages))

    def count_tokens(self, messages, *, tools=(), instructions=None) -> int:
        messages = list(messages)
        _validate(messages)
        scratch = _ScratchSession(self.session.engine)
        scratch.start_chat(_definitions(tools), instructions)
        for message in messages:
            key = (message.metadata or {}).get("ds4.key", "")
            self._append_message(scratch, message, key, _content_hash(message))
        prefix = PromptBuilder(self.session.engine).assistant_prefix(scratch.transcript.tokens)
        if prefix:
            scratch.append_tokens(prefix, "assistant_prefix")
        return len(scratch.transcript)

    def bind_response(self, response) -> None:
        key = _ensure_key(response)
        response.metadata["ds4.source"] = "sampled"
        content_hash = _content_hash(response)
        segments = self.session.transcript.segments
        prefix_index = max(
            (index for index, segment in enumerate(segments) if segment.kind == "assistant_prefix"),
            default=None,
        )
        trailing = [
            segment for segment in segments[(prefix_index + 1) if prefix_index is not None else 0:]
            if segment.kind in ("assistant", "eos")
        ]
        sampled_start = trailing[0].start if trailing else len(self.session.transcript)
        self.session._transcript.annotate(
            sampled_start, kinds=("assistant", "eos"),
            meta={"key": key, "content_hash": content_hash, "source": "sampled"},
        )
        start = segments[prefix_index].start if prefix_index is not None else sampled_start
        record = _BoundMessage(key, content_hash, start, len(self.session.transcript))
        if self._records and self._records[-1].key == key and self._records[-1].start == start:
            self._records[-1] = record
        else:
            self._records.append(record)

    def export_segments(self) -> tuple[SegmentProvenance, ...]:
        return tuple(
            SegmentProvenance(
                segment.start,
                segment.end,
                segment.kind,
                str(segment.meta.get("key", "")),
                str(segment.meta.get("content_hash", "")),
                str(segment.meta.get("source", "rendered")),
            )
            for segment in self.session.transcript.segments
        )
