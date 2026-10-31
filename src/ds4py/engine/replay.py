"""Read token-level ds4-agent traces for deterministic live replay."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Never

from ds4py._native.protocol import THINK_LEVEL_BASE

from .ds4py_replay import read_ds4py
from .replay_types import _THINK_NAMES, ReplayError, ReplayRound, ReplayTrace

_MAX_THINK_LEVEL = 100

_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:\d{2})?\s*"
)
_TOKENS = re.compile(r"^tokens label=(\S+) start=(-?\d+) len=(-?\d+)$")
_TOKEN = re.compile(r"^token index=(-?\d+) id=(-?\d+)(?:\s|$)")
_PREFILL = re.compile(
    r"^prefill tool_round=(-?\d+) transcript=(-?\d+) prompt=(-?\d+) "
    r"cached=(-?\d+) suffix=(-?\d+) think=(\S+)$"
)
_GENERATION = re.compile(
    r"^generation finished tool_round=(-?\d+) generated=(-?\d+) "
    r"carried=(-?\d+) context_limited=(-?\d+)$"
)
_WORKER = re.compile(r"^agent worker start ctx=(\S+) backend=(\S+) model=(.*?) trace=(\S+)$")


@dataclass
class _Block:
    label: str
    start: int
    length: int
    line: int
    ids: list[int] = field(default_factory=list)


@dataclass
class _Round:
    number: int
    transcript: int
    prompt_length: int
    cached: int
    suffix_length: int
    think: int
    prompt: tuple[int, ...] | None = None
    sampled: list[int] = field(default_factory=list)
    context_limited: bool = False
    prefix_only: bool = False


def _think_mode(value: str, reader: _Reader, line: int) -> int:
    if value in _THINK_NAMES:
        return _THINK_NAMES[value]
    if value.isdecimal():
        level = int(value)
        if 1 <= level <= _MAX_THINK_LEVEL:
            return THINK_LEVEL_BASE + level
    return reader.error("unsupported think name", line)


def read_replay(path: str | Path) -> ReplayTrace:
    """Parse a C text trace or ds4-py JSONL prefill records."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        return read_ds4py(path, text)
    return _Reader(path).read(text)


class _Reader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.ctx: int | None = None
        self.model: str | None = None
        self.base: tuple[int, ...] | None = None
        self.rounds: list[ReplayRound] = []
        self.current: _Round | None = None
        self.block: _Block | None = None
        self.can_compact = False
        self.line = 0

    def error(self, message: str, line: int | None = None) -> Never:
        raise ReplayError(f"{self.path}:{line or self.line or 1}: {message}")

    def read(self, text: str | None = None) -> ReplayTrace:
        if text is None:
            text = self.path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for number, raw in enumerate(lines, 1):
            self.line = number
            text = _TIMESTAMP.sub("", raw, count=1)
            if text:
                self.dispatch(text)
        self.line = len(lines)
        if self.block is not None:
            self.error("wrong token count in truncated token block")
        if self.current is not None:
            self.error("truncated final round: missing generation finished")
        if self.ctx is None:
            self.error("missing context in agent worker start")
        if self.base is None:
            self.error("missing initial system prompt")
        if self.model is None:
            self.error("missing model in agent worker start")
        if not self.rounds:
            self.error("no replay rounds")
        return ReplayTrace(self.ctx, self.model, tuple(self.rounds))

    def dispatch(  # noqa: PLR0912 - Trace dispatch handles each distinct record family.
        self, text: str
    ) -> None:
        if "raw one-shot prompt mode: skipping agent system prompt" in text:
            self.error("raw traces are not replayable")
        if self.block is not None:
            if _TOKEN.match(text):
                self.token_line(text, sampled=False)
            else:
                self.error("wrong token count before the next trace line")
            return
        if text.startswith("tokens label="):
            self.start_block(text)
        elif text == "prefill sync done" or text.startswith("prefill sync done "):
            pass
        elif text.startswith("prefill "):
            self.prefill(text)
        elif text.startswith("token index="):
            if self.current is None:
                self.error("sampled token outside a round")
            self.token_line(text, sampled=True)
        elif text.startswith("generation finished "):
            self.generation(text)
        elif text.startswith("think="):
            if self.current is None:
                self.error("think mode outside a round")
            self.current.think = _think_mode(text[6:], self, self.line)
        elif text.startswith("edit old preflight failed:"):
            if self.current is not None:
                self.current.prefix_only = True
        elif text.startswith("agent worker start "):
            self.worker(text)

    def worker(self, text: str) -> None:
        if self.ctx is not None:
            self.error("duplicate worker start")
        match = _WORKER.fullmatch(text)
        if match is None:
            self.error("invalid context or worker start")
        try:
            ctx = int(match.group(1))
        except ValueError:
            self.error("invalid context")
        if ctx <= 0:
            self.error("invalid context")
        if not match.group(3):
            self.error("missing model in agent worker start")
        self.ctx = ctx
        self.model = match.group(3)

    def start_block(  # noqa: C901, PLR0912 - Trace block validation mirrors each label contract.
        self, text: str
    ) -> None:
        match = _TOKENS.fullmatch(text)
        if match is None:
            self.error("malformed token block")
        label, start_text, length_text = match.groups()
        if label == "raw_prompt_tokens":
            self.error("raw traces are not replayable")
        start, length = int(start_text), int(length_text)
        if label == "initial_system_prompt":
            if self.base is not None:
                self.error("duplicate initial system prompt")
            if self.current is not None or start != 0:
                self.error("invalid initial system prompt block")
        elif label == "prefill_suffix":
            if self.current is None:
                self.error("prefill suffix without prefill")
            if self.current.prompt is not None:
                self.error("duplicate prefill suffix")
            if start != self.current.cached:
                self.error("suffix start does not equal cached prefix length")
            if length != self.current.prompt_length:
                self.error("suffix block length does not equal prompt length")
        elif label == "compacted_transcript":
            if not self.can_compact or self.current is not None:
                self.error("unexpected compacted transcript")
            if start != 0:
                self.error("compacted transcript must start at zero")
        else:
            self.error("unexpected token block")
        if length < start or start < 0:
            self.error("invalid token block length")
        self.block = _Block(label, start, length, self.line)
        if length == start:
            self.finish_block()

    def token_line(self, text: str, *, sampled: bool) -> None:
        match = _TOKEN.match(text)
        if match is None:
            self.error("malformed token line")
        index, token_id = (int(value) for value in match.groups())
        if token_id < 0:
            self.error("negative token id")
        if sampled:
            if self.current is None:
                self.error("sampled token outside a round")
            if self.current.prompt is None:
                self.error("sampled token before prefill suffix")
            expected = len(self.current.sampled) + 1
            if index != expected:
                self.error("non-contiguous token index in sampled tokens")
            self.current.sampled.append(token_id)
            return
        block = self.block
        if block is None:
            raise AssertionError("token line requires an open block")
        expected = block.start + len(block.ids)
        if index != expected:
            self.error("non-contiguous token index in token block")
        block.ids.append(token_id)
        if len(block.ids) == block.length - block.start:
            self.finish_block()
        elif len(block.ids) > block.length - block.start:
            self.error("wrong token count in token block")

    def finish_block(self) -> None:
        block = self.block
        if block is None:
            raise AssertionError("finish requires an open block")
        self.block = None
        ids = tuple(block.ids)
        if block.label == "initial_system_prompt":
            self.base = ids
        elif block.label == "compacted_transcript":
            self.base = ids
            self.can_compact = False
        else:
            current = self.current
            if current is None or self.base is None:
                raise AssertionError("suffix requires a round and base history")
            expected = current.prompt_length - current.cached
            if len(ids) != expected:
                self.error("suffix length does not match prefill", block.line)
            current.prompt = self.base[: current.cached] + ids

    def prefill(self, text: str) -> None:
        match = _PREFILL.fullmatch(text)
        if match is None:
            self.error("malformed prefill")
        if self.current is not None:
            self.error("second prefill before generation finished")
        if self.base is None:
            self.error("missing initial system prompt")
        number, transcript, prompt_length, cached, suffix_length = (int(value) for value in match.groups()[:5])
        think = _think_mode(match.group(6), self, self.line)
        if self.rounds:
            next_number = self.rounds[-1].number + 1
            valid_number = number in (0, next_number)
        else:
            valid_number = number in (0, 1)
        if not valid_number:
            self.error("non-contiguous tool round")
        if cached < 0:
            self.error("negative cached prefix length")
        if cached > len(self.base):
            self.error("cached prefix exceeds rebuilt history")
        if transcript < 0 or prompt_length < 0 or cached > transcript:
            self.error("cached prefix beyond traced transcript")
        if transcript != prompt_length:
            self.error("transcript/prompt length mismatch")
        if suffix_length != prompt_length - cached:
            self.error("suffix length does not match prefill")
        self.current = _Round(
            number,
            transcript,
            prompt_length,
            cached,
            suffix_length,
            think,
        )
        self.can_compact = False

    def generation(self, text: str) -> None:
        match = _GENERATION.fullmatch(text)
        if match is None:
            self.error("malformed generation finished line")
        number, generated, carried, limited = (int(value) for value in match.groups())
        current = self.current
        if current is None or current.prompt is None:
            self.error("generation finished without a complete prefill")
        if number != current.number:
            self.error("generation round does not match prefill")
        if generated != len(current.sampled):
            self.error("generated count does not match sampled tokens")
        if carried < 0 or limited not in (0, 1):
            self.error("invalid generation flags")
        current.context_limited = bool(limited)
        self.rounds.append(
            ReplayRound(
                number=current.number,
                prompt=current.prompt,
                sampled=tuple(current.sampled),
                think=current.think,
                cached=current.cached,
                context_limited=current.context_limited,
                prefix_only=current.prefix_only,
            )
        )
        self.base = current.prompt + tuple(current.sampled)
        self.current = None
        self.can_compact = True
