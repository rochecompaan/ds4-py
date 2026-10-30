"""Prompt-toolkit fragments, stream previews, and live footer state."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from prompt_toolkit.formatted_text import FormattedText, StyleAndTextTuples
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.utils import get_cwidth
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)

from ds4py.engine.events import Prefill, Synced, TextDelta

if TYPE_CHECKING:
    from prompt_toolkit.output import Output

    from ds4py.engine.lifecycle import SessionEvent
    from ds4py.runtime import AgentRuntime

_INLINE = re.compile(r"(\*\*[^*\n]+\*\*|(?<!\*)\*[^*\n]+\*(?!\*)|`[^`\n]+`)")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def _flatten(value: object) -> str:
    return " ".join(str(value).split())


def _fit(value: object, width: int) -> str:
    """Return one display line whose terminal column width is at most width."""
    text = _flatten(value)
    if width <= 0:
        return ""
    if get_cwidth(text) <= width:
        return text
    if width == 1:
        return "…"
    result: list[str] = []
    used = 0
    for char in text:
        char_width = get_cwidth(char)
        if used + char_width > width - 1:
            break
        result.append(char)
        used += char_width
    return "".join(result) + "…"


def markdown_fragments(text: str) -> StyleAndTextTuples:
    """Convert headings and simple inline Markdown to styled fragments."""
    fragments: StyleAndTextTuples = []
    for line in text.splitlines(keepends=True):
        ending = "\n" if line.endswith("\n") else ""
        content = line[:-1] if ending else line
        heading = _HEADING.match(content)
        if heading:
            fragments.append(("class:markdown.heading bold", heading.group(1)))
            if ending:
                fragments.append(("", ending))
            continue
        offset = 0
        for match in _INLINE.finditer(content):
            if match.start() > offset:
                fragments.append(("class:text", content[offset : match.start()]))
            token = match.group(0)
            if token.startswith("**"):
                fragments.append(("class:markdown.bold bold", token[2:-2]))
            elif token.startswith("*"):
                fragments.append(("class:markdown.italic italic", token[1:-1]))
            else:
                fragments.append(("class:markdown.code", token[1:-1]))
            offset = match.end()
        if offset < len(content):
            fragments.append(("class:text", content[offset:]))
        if ending:
            fragments.append(("", ending))
    return fragments


class StreamRenderer:
    """Render harness deltas and bounded tool previews on the UI thread."""

    def __init__(self, output: Output | None = None) -> None:
        self.output = output

    def handle_event(self, event: object) -> StyleAndTextTuples:
        fragments = self._fragments(event)
        if fragments:
            newline = isinstance(event, (FunctionToolCallEvent, FunctionToolResultEvent))
            print_formatted_text(
                FormattedText(fragments),
                output=self.output,
                end="\n" if newline else "",
                flush=True,
            )
        return fragments

    def _fragments(  # noqa: PLR0911 - Each harness event type has its own renderer.
        self, event: object
    ) -> StyleAndTextTuples:
        if isinstance(event, TextDelta):
            if event.thinking:
                return [("dim italic class:thinking", event.text)]
            return markdown_fragments(event.text)
        if isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta):
                return markdown_fragments(event.delta.content_delta)
            if isinstance(event.delta, ThinkingPartDelta):
                return [("dim italic class:thinking", event.delta.content_delta or "")]
            return []
        if isinstance(event, PartStartEvent):
            if isinstance(event.part, TextPart):
                return markdown_fragments(event.part.content)
            if isinstance(event.part, ThinkingPart):
                return [("dim italic class:thinking", event.part.content)]
            return []
        if isinstance(event, FunctionToolCallEvent):
            part = event.part
            return [("class:tool.call", _fit(f"→ {part.tool_name} {part.args}", 110))]
        if isinstance(event, FunctionToolResultEvent):
            part = event.part
            content = event.content if event.content is not None else part.content
            return [("class:tool.result", _fit(f"← {part.tool_name}: {content}", 110))]
        return []


class FooterModel:
    """Session-event-backed status rendered outside the conversation."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self.status = "idle"
        self.plan: str | None = None
        self.ctx_used = 0
        self.ctx_total = runtime.context_size
        self.cwd = str(runtime.project_dir)
        self.prefill: tuple[int, int, float] | None = None
        self.service_pid: int | None = None
        self.service_status: str | None = None
        self.custom: dict[str, str] = {}
        self._unsubscribe = runtime.on_event(self.handle_event)

    def handle_event(self, event: SessionEvent) -> None:
        if event.kind == "run_start":
            self.status = "running"
            self.prefill = None
        elif event.kind == "run_end":
            self.status = "idle"
            self.prefill = None
        elif event.kind == "Prefill":
            inner = event.payload["event"]
            if not isinstance(inner, Prefill):
                raise TypeError("Prefill event payload has the wrong type")
            self.prefill = (inner.done, inner.total, inner.tps)
        elif event.kind == "Synced":
            inner = event.payload["event"]
            if not isinstance(inner, Synced):
                raise TypeError("Synced event payload has the wrong type")
            self.plan = inner.plan.kind
            self.ctx_used = inner.pos
        elif event.kind in {"service_started", "service_status", "service_stopped"}:
            pid = event.payload.get("pid", self.service_pid)
            if pid is not None and not isinstance(pid, int):
                raise TypeError("service pid must be an integer or None")
            status = event.payload.get("status")
            if status is not None and not isinstance(status, str):
                raise TypeError("service status must be a string or None")
            self.service_pid = pid
            self.service_status = status
        elif event.kind == "status":
            key = str(event.payload["key"])
            value = event.payload.get("text")
            if bool(value):
                self.custom[key] = str(value)
            else:
                self.custom.pop(key, None)

    def _parts(self) -> list[str]:
        parts = [self.status]
        if bool(self.plan):
            parts.append(self.plan)
        parts.append(f"ctx {self.ctx_used}/{self.ctx_total}")
        if self.prefill is not None:
            done, total, tps = self.prefill
            filled = min(32, round(32 * done / total)) if total else 0
            parts.append(f"prefill {'▶' * filled}{'·' * (32 - filled)} {done}/{total} {tps:.1f} tok/s")
        if self.service_status is not None:
            parts.append(f"shell {self.service_pid or '-'} {self.service_status}")
        parts.extend(f"{key}: {value}" for key, value in sorted(self.custom.items()))
        if self.cwd:
            parts.append(self.cwd)
        return parts

    def render(self, width: int) -> StyleAndTextTuples:
        return [("class:footer", _fit(" | ".join(self._parts()), width))]

    def close(self) -> None:
        self._unsubscribe()
