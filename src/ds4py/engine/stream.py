"""Per-assistant-round parser and display stream state."""
from .decoder import Decoder
from .events import Thinking, ToolCalls, ToolSyntax
from .tool_syntax import make_parser


class RoundStream:
    def __init__(self, info, think_mode):
        thinking = int(think_mode) in (1, 2, 3, 4) or 1001 <= int(think_mode) <= 1100
        self.parser = make_parser(info)
        self.parser.in_think = thinking
        self.decoder = Decoder(thinking=thinking)
        self.snapshot = self.parser.feed(b"")
        self.sampled_tokens = []
        self.think_tokens = 0
        self.finalized = False
        self._last_state = self.snapshot.state
        self._last_thinking = thinking
        self._calls_emitted = False

    def _publish(self, deltas, emit):
        if self.decoder.thinking != self._last_thinking:
            self._last_thinking = self.decoder.thinking
            emit(Thinking(self._last_thinking))
        if self.snapshot.state != self._last_state:
            self._last_state = self.snapshot.state
            emit(ToolSyntax(self._last_state))
        for delta in deltas:
            emit(delta)

    def observe(self, token, data, emit):
        self.sampled_tokens.append(token)
        self.snapshot = self.parser.feed(data)
        deltas = self.decoder.feed(data)
        if self.decoder.thinking:
            self.think_tokens += 1
        self._publish(deltas, emit)

    def finalize(self, *, parser, emit):
        if self.finalized:
            return
        if parser:
            self.snapshot = self.parser.finish()
        deltas = self.decoder.finish()
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
