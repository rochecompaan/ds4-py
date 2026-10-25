"""Display-only assistant text decoder."""
import codecs

from .events import TextDelta

_THINK_OPEN = b"<think>"
_THINK_CLOSE = b"</think>"
_MARKERS = (_THINK_OPEN, _THINK_CLOSE)
_PENDING_LIMIT = 16


class Decoder:
    def __init__(self, *, thinking=False):
        self.thinking = thinking
        self._pending = bytearray()
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

    @staticmethod
    def _emit(deltas, text, thinking):
        if not text:
            return
        if deltas and deltas[-1].thinking == thinking:
            deltas[-1] = TextDelta(deltas[-1].text + text, thinking)
        else:
            deltas.append(TextDelta(text, thinking))

    def _decode(self, data, deltas, *, final=False):
        text = self._utf8.decode(data, final=final)
        self._emit(deltas, text, self.thinking)

    def _marker_boundary(self, deltas):
        # A valid codepoint cannot span an ASCII marker. If invalid bytes are
        # pending, attribute their deterministic replacement to the old side.
        self._decode(b"", deltas, final=True)
        self._utf8.reset()

    def feed(self, data: bytes) -> list[TextDelta]:
        deltas = []
        for byte in bytes(data):
            self._pending.append(byte)
            while self._pending:
                pending = bytes(self._pending)
                if pending in _MARKERS:
                    self._pending.clear()
                    self._marker_boundary(deltas)
                    self.thinking = pending == _THINK_OPEN
                    break
                if any(marker.startswith(pending) for marker in _MARKERS):
                    break
                first = bytes(self._pending[:1])
                del self._pending[0]
                self._decode(first, deltas)
            if len(self._pending) > _PENDING_LIMIT:
                raise RuntimeError("display marker buffer exceeded its bound")
        return deltas

    def finish(self) -> list[TextDelta]:
        deltas = []
        if self._pending:
            pending = bytes(self._pending)
            self._pending.clear()
            self._decode(pending, deltas)
        self._decode(b"", deltas, final=True)
        self._utf8.reset()
        return deltas
