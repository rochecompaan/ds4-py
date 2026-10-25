"""Per-assistant-round parser, control matching, and display stream state."""
from dataclasses import replace

from .decoder import Decoder
from .events import Thinking, ToolCalls, ToolSyntax
from .pause import ControlSpan, PausedAfterEos, Resume, StreamCheckpoint
from .tool_syntax import make_parser


class RoundStream:
    def __init__(self, info, think_mode, *, control: ControlSpan | None = None):
        thinking = int(think_mode) in (1, 2, 3, 4) or 1001 <= int(think_mode) <= 1100
        self.parser = make_parser(info)
        self.parser.in_think = thinking
        self.decoder = Decoder(thinking=thinking)
        self.control = control or ControlSpan()
        self.checkpoint = StreamCheckpoint(thinking=thinking)
        self.snapshot = self.parser.feed(b"")
        self.sampled_tokens = []
        self.think_tokens = 0
        self.finalized = False
        self.stop_was_eos = False
        self.paused: PausedAfterEos | None = None
        self._eos_token: int | None = None
        self._eos_text = b""
        self._text_parts: list[str] = []
        self._last_state = self.snapshot.state
        self._last_thinking = thinking
        self._calls_emitted = False

    @property
    def active_stop_sequences(self) -> list[bytes]:
        return self.control.active_stop_sequences

    @property
    def stop_sequence(self) -> bytes | None:
        return self.control.stop_sequence

    def _publish(self, deltas, emit):
        if self.decoder.thinking != self._last_thinking:
            self._last_thinking = self.decoder.thinking
            emit(Thinking(self._last_thinking))
        if self.snapshot.state != self._last_state:
            self._last_state = self.snapshot.state
            emit(ToolSyntax(self._last_state))
        for delta in deltas:
            emit(delta)

    def _match_deltas(self, deltas) -> None:
        if not self.control.sequences:
            return
        for delta in deltas:
            if delta.thinking:
                continue
            self._text_parts.append(delta.text)
            if self.control.Feed(delta.text.encode("utf-8")) is not None:
                break

    def observe(self, token, data, emit):
        self.sampled_tokens.append(token)
        self.snapshot = self.parser.feed(data)
        was_thinking = self.decoder.thinking
        deltas = self.decoder.feed(data)
        self._match_deltas(deltas)
        if self.decoder.thinking:
            self.think_tokens = self.think_tokens + 1 if was_thinking else 1
        self._publish(deltas, emit)

    def mark_eos(self, token: int, text: bytes = b"", *,
                 paused: PausedAfterEos | None = None) -> None:
        self.stop_was_eos = True
        self.paused = paused
        self._eos_token, self._eos_text = token, bytes(text)
        self.checkpoint = replace(
            self.checkpoint, thinking=self.decoder.thinking, text="".join(self._text_parts),
        )

    def resume(self, emit, resume: Resume | None = None) -> bytes | None:
        """Retain KMP state and replay a paused EOS before more sampling."""
        resume = resume or Resume()
        self.control.resume()
        if self.stop_was_eos or resume.force_end_thinking:
            self.checkpoint = replace(self.checkpoint, force=True)
            if self._eos_token is not None:
                self.observe(self._eos_token, self._eos_text, emit)
            if resume.force_end_thinking and self.decoder.thinking:
                self.snapshot = self.parser.feed(b"</think>")
                deltas = self.decoder.feed(b"</think>")
                self._publish(deltas, emit)
            self.checkpoint = replace(self.checkpoint, force=False)
        self.stop_was_eos = False
        self.paused = None
        self._eos_token = None
        self._eos_text = b""
        self.finalized = False
        return self.stop_sequence

    def finalize(self, *, parser, emit):
        if self.finalized:
            return
        if parser:
            self.snapshot = self.parser.finish()
        deltas = self.decoder.finish()
        self.control.finish()
        self.finalized = True
        self._publish(deltas, emit)

    @property
    def malformed(self):
        return self.snapshot.state == "ERROR" or self.snapshot.dsml_in_think

    @property
    def complete(self):
        return self.snapshot.state == "DONE" and not self.snapshot.dsml_in_think

    def emit_calls(self, emit):
        if self._calls_emitted or not self.complete:
            return
        self._calls_emitted = True
        emit(ToolCalls(tuple(self.snapshot.calls)))
