"""Shared byte-stream state for family-specific tool parsers."""
from .model import Snapshot

_THINK_OPEN = b"<think>"
_THINK_CLOSE = b"</think>"
_DSML_MARKERS = tuple(
    marker.encode() for marker in ("｜DSML｜", "|DSML|", "<DSML｜", "</DSML｜")
)


class ToolParser:
    def __init__(self):
        self.state = "SEARCH"
        self.raw = bytearray()
        self.parse_pos = 0
        self.error = None
        self.param_close_prefix = False
        self.in_think = False
        self.dsml_in_think = False
        self._calls = []
        self._active = False
        self._pending = bytearray()
        self._marker_tail = bytearray()
        self._forms = tuple(self._opening_forms())

    def _opening_forms(self):
        raise NotImplementedError

    def _start(self, implicit_invoke):
        raise NotImplementedError

    def _parse(self):
        raise NotImplementedError

    def _update_param_close_prefix(self):
        self.param_close_prefix = False

    def _snapshot(self):
        return Snapshot(
            self.state, bytes(self.raw), tuple(self._calls), self.error,
            self.param_close_prefix, self.in_think, self.dsml_in_think,
        )

    def _fail(self, message):
        self.state = "ERROR"
        self.error = message
        self._calls.clear()
        self.param_close_prefix = False

    def feed(self, data: bytes):
        if self.state in ("DONE", "ERROR"):
            return self._snapshot()
        for byte in bytes(data):
            if self.state in ("DONE", "ERROR"):
                break
            if self._active:
                self.raw.append(byte)
                self._parse()
                if self.state == "PARAM_VALUE":
                    self._update_param_close_prefix()
                else:
                    self.param_close_prefix = False
            else:
                self._feed_inactive(byte)
        return self._snapshot()

    def _feed_inactive(self, byte):
        self._pending.append(byte)
        while self._pending and self.state not in ("DONE", "ERROR"):
            pending = bytes(self._pending)
            if pending in (_THINK_OPEN, _THINK_CLOSE):
                self.in_think = pending == _THINK_OPEN
                self._pending.clear()
                return
            form = next((form for form in self._forms if pending == form[0]), None)
            if form is not None:
                self._pending.clear()
                if self.in_think:
                    self.dsml_in_think = True
                    self._fail("tool calling is not allowed inside <think></think>")
                else:
                    self._start(form[1])
                return
            markers = (_THINK_OPEN, _THINK_CLOSE) + tuple(form[0] for form in self._forms)
            if any(marker.startswith(pending) for marker in markers):
                return
            ordinary = self._pending[0]
            del self._pending[0]
            self._note_ordinary_byte(ordinary)

    def _note_ordinary_byte(self, byte):
        self._marker_tail.append(byte)
        maximum = max(map(len, _DSML_MARKERS))
        if len(self._marker_tail) > maximum:
            del self._marker_tail[0]
        tail = bytes(self._marker_tail)
        if not any(tail.endswith(marker) for marker in _DSML_MARKERS):
            return
        if self.in_think:
            self.dsml_in_think = True
            self._fail("tool calling is not allowed inside <think></think>")
        else:
            self._fail("DSML markup outside a valid tool_calls block")
        self._pending.clear()

    def finish(self):
        if self.state in ("DONE", "ERROR"):
            return self._snapshot()
        if self._active:
            self._fail("incomplete tool-call stanza")
        elif len(self._pending) > 1 and any(
            form[0].startswith(self._pending) for form in self._forms
        ):
            self._fail("incomplete tool-call opening marker")
        return self._snapshot()

    def wants_greedy_sampling(self) -> bool:
        if self.state in ("DONE", "ERROR"):
            return False
        pending = bytes(self._pending)
        if len(pending) > 1 and any(form[0].startswith(pending) for form in self._forms):
            return True
        if not self._active:
            return False
        if self.state == "STRUCTURAL":
            return True
        if self.state == "PARAM_VALUE":
            return self.param_close_prefix
        return False


def make_parser(info):
    from .dsml import Dsml
    if info.is_qwen4:
        from .qwen import Qwen
        return Qwen()
    if info.is_glm_dsa:
        from .glm import Glm
        return Glm()
    return Dsml(v41=info.is_deepseek41)
