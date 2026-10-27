"""Per-assistant-round parser, control matching, and display stream state."""

from .decoder import Decoder
from .events import Thinking, ToolCalls, ToolSyntax
from .pause import ControlSpan
from .tool_syntax import make_parser


class RoundStream:
    def __init__(self, info, think_mode, *, control: ControlSpan | None = None):
        thinking = int(think_mode) in (1, 2, 3, 4) or 1001 <= int(think_mode) <= 1100
        self.parser = make_parser(info)
        self.parser.in_think = thinking
        self.decoder = Decoder(thinking=thinking)
        self.control = control or ControlSpan()
        self.snapshot = self.parser.feed(b"")
        self.sampled_tokens = []
        self.think_tokens = 0
        self.finalized = False
        self._control_pending = bytearray()
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
        if self.snapshot.in_think != self._last_thinking:
            self._last_thinking = self.snapshot.in_think
            emit(Thinking(self._last_thinking))
        if self.snapshot.state != self._last_state:
            self._last_state = self.snapshot.state
            emit(ToolSyntax(self._last_state))
        for delta in deltas:
            emit(delta)

    def _match_control(self, data):
        if data and self.control.sequences:
            self.control.feed(data)

    def _observe_control(self, data):
        # Control scans sampled bytes, not display deltas. Only actual think
        # markers outside argument values are withheld from the matcher.
        markers = (b"<think>", b"</think>")
        for byte in data:
            literal = self.parser.state == "PARAM_VALUE"
            thinking = self.parser.in_think
            self.parser._feed_byte(byte)
            if literal:
                if not thinking:
                    self._match_control(bytes([byte]))
                continue
            self._control_pending.append(byte)
            while self._control_pending:
                pending = bytes(self._control_pending)
                if pending in markers:
                    self._control_pending.clear()
                    break
                if any(marker.startswith(pending) for marker in markers):
                    break
                if not thinking:
                    self._match_control(bytes(self._control_pending[:1]))
                del self._control_pending[:1]
        self.snapshot = self.parser._snapshot()

    def observe(self, token, data, emit):
        self.sampled_tokens.append(token)
        was_thinking = self.snapshot.in_think
        self._observe_control(data)
        deltas = self.decoder.feed(data)
        if self.snapshot.in_think:
            self.think_tokens = self.think_tokens + 1 if was_thinking else 1
        else:
            self.think_tokens = 0
        self._publish(deltas, emit)

    def resume(self) -> None:
        """Retain matcher state and clear the matched stop before continuing."""
        self.control.resume()
        self.finalized = False

    def finalize(self, *, parser, emit):
        if self.finalized:
            return
        if parser:
            self.snapshot = self.parser.finish()
        deltas = self.decoder.finish()
        if not self.snapshot.in_think:
            self._match_control(bytes(self._control_pending))
        self._control_pending.clear()
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
