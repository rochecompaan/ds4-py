"""Render trusted chat structure while keeping caller payloads plain."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ds4py.errors import EngineError

from .tool_prompt import ToolDefinition, ToolPromptGenerator
from .tool_syntax.text import escape_tool_text

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .session import Engine


@dataclass(frozen=True)
class RenderedSegment:
    """Internal rendered delta and transcript kind; not a second history."""

    kind: str
    tokens: list[int]


class PromptBuilder:
    """Build deltas without mutating history; native work dispatches to the engine worker.

    Call from outside the worker, between rounds. Trusted markup is rendered;
    caller payloads are tokenized as plain text.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.family = engine.info.family

    def begin(
        self,
        tools: Iterable[ToolDefinition] = (),
        instructions: str | None = None,
    ) -> list[RenderedSegment]:
        """Build initial chat segments for definitions and optional plain instructions."""
        segments: list[RenderedSegment] = []
        prefix: list[int] = []

        def append(kind: str, tokens: list[int]) -> None:
            if tokens:
                segments.append(RenderedSegment(kind, tokens))
                prefix.extend(tokens)

        append("system", self.engine._chat_begin())
        if self.family == "qwen":
            effort = self.engine._qwen_reasoning_effort(self.engine.think)
            if bool(effort):
                append("system", self.message(prefix, "system", effort))
        else:
            append("system", self.engine._think_prefix(prefix, self.engine.think))

        if self.family == "deepseek41":
            append("system", self.engine._render_message(prefix, "system", ""))
        tool_tokens = self._tool_tokens(tools)
        if self.family == "glm":
            role = self.engine._render_message(prefix, "system", "")
            append("system", role + tool_tokens)
        elif self.family == "qwen":
            append("system", self._qwen_open("system") + tool_tokens + self._qwen_close())
        else:
            append("system", tool_tokens)

        if bool(instructions):
            append("system", self.message(prefix, "system", "\n\n" + instructions))
        return segments

    def _tool_tokens(self, tools: Iterable[ToolDefinition]) -> list[int]:
        tokens: list[int] = []
        for span in ToolPromptGenerator(self.family).spans(tools):
            tokens.extend(self.engine.tokenize(span.text, rendered=span.rendered))
        return tokens

    def _qwen_open(self, role: str) -> list[int]:
        return self.engine.tokenize("<|im_start|>", rendered=True) + self.engine.tokenize(role + "\n")

    def _qwen_close(self) -> list[int]:
        return self.engine.tokenize("<|im_end|>", rendered=True) + self.engine.tokenize("\n")

    def message(self, prefix: Sequence[int], role: str, text: str) -> list[int]:
        """Render one role delta against prefix, keeping text untrusted."""
        if role in ("tool", "function"):
            return self._tool_message(prefix, text)
        if self.family == "glm":
            if role in ("system", "developer"):
                return self.engine._render_message(prefix, "system", "") + self.engine.tokenize(text)
            if role == "assistant":
                return self._glm_assistant(prefix, text)
        elif self.family == "qwen" and role == "assistant":
            return self._qwen_open("assistant") + self.engine.tokenize(text) + self._qwen_close()
        return self.engine._render_message(prefix, role, text)

    def _glm_assistant(self, prefix: Sequence[int], text: str) -> list[int]:
        if text.startswith(("<think>", "</think>")):
            marker = "<think>" if text.startswith("<think>") else "</think>"
            rendered = self.engine._render_message(prefix, "assistant", marker)
            suffix = self.engine.tokenize(marker, rendered=True)
            if not suffix or rendered[-len(suffix) :] != suffix:
                raise EngineError("native GLM assistant template has an unexpected think marker")
            return rendered[: -len(suffix)] + self.engine.tokenize(text)
        return self.engine._render_message(prefix, "assistant", "") + self.engine.tokenize(text)

    def _tool_message(self, prefix: Sequence[int], text: str) -> list[int]:
        if self.family == "glm":
            closing = self.engine.tokenize("</tool_response>", rendered=True)
            rendered = self.engine._render_message(prefix, "tool", "")
            if not closing or rendered[-len(closing) :] != closing:
                raise EngineError("native GLM tool template has an unexpected closing delimiter")
            escaped = escape_tool_text(text, "</tool_response>")
            return rendered[: -len(closing)] + self.engine.tokenize(escaped) + closing
        if self.family == "qwen":
            escaped = escape_tool_text(text, "</tool_response>")
            return (
                self._qwen_open("user")
                + self.engine.tokenize("<tool_response>", rendered=True)
                + self.engine.tokenize("\n")
                + self.engine.tokenize(escaped)
                + self.engine.tokenize("\n")
                + self.engine.tokenize("</tool_response>", rendered=True)
                + self._qwen_close()
            )
        return self.engine._render_message(prefix, "tool", text)

    def restored_assistant(
        self,
        prefix: Sequence[int],
        spans: Iterable[tuple[str, bool]],
    ) -> list[int]:
        """Render an assistant body from plain text and trusted-markup spans.

        This path is only for genuinely reconstructed history. Ordinary external
        assistant text uses message(), which never renders payload markup.
        """
        tokens = self.message(prefix, "assistant", "")
        closing = self._qwen_close() if self.family == "qwen" else []
        if closing:
            if tokens[-len(closing) :] != closing:
                raise EngineError("native Qwen assistant template has an unexpected close")
            tokens = tokens[: -len(closing)]
        for text, rendered in spans:
            if text:
                tokens.extend(self.engine.tokenize(text, rendered=rendered))
        return tokens + closing

    def assistant_prefix(self, prefix: Sequence[int]) -> list[int]:
        """Build the family/mode assistant opener through the worker."""
        return self.engine._assistant_prefix(prefix, self.engine.think)

    def tool_results(self, prefix: Sequence[int], results: Iterable[str]) -> list[int]:
        """Render all results as one message joined by two newlines."""
        return self._tool_message(prefix, "\n\n".join(results))

    def append_assistant_end(self) -> list[int]:
        """Return the family EOS delta (empty for GLM) through the worker."""
        return [] if self.family == "glm" else [self.engine._eos()]
