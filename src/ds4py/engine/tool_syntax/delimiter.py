"""Shared lifecycle for GLM and Qwen delimiter-based tool calls."""

from copy import deepcopy
from types import MappingProxyType

from .base import ToolParser
from .model import ToolArgument, ToolCall

_ASCII_SPACE = b" \t\r\n"
_START = b"<tool_call>"


class DelimiterParser(ToolParser):
    """Own common delimiter buffering, call publication, and finalization."""

    _call_close: bytes
    _parameter_close: bytes
    _family: str

    def __init__(self) -> None:
        self._after_call = False
        self._call_start = 0
        self._current_name: str | None = None
        self._args: dict[str, ToolArgument] = {}
        self._param_name: str | None = None
        self._param_value_start = 0
        self._param_scan_pos = 0
        super().__init__()

    def _opening_forms(self) -> tuple[tuple[bytes, bool], ...]:
        return ((_START, False),)

    def _start(self, implicit_invoke: bool) -> None:
        self.state = "STRUCTURAL"
        self._active = True
        self._call_start = len(self.raw)
        self.raw.extend(_START)
        self.parse_pos = len(self.raw)

    @staticmethod
    def _skip_space(data: bytes | bytearray, position: int) -> int:
        while position < len(data) and data[position] in _ASCII_SPACE:
            position += 1
        return position

    def _partial(self, prefix: bytes, position: int) -> bool:
        remaining = bytes(self.raw[position:])
        return len(remaining) < len(prefix) and prefix.startswith(remaining)

    def _decode(self, value: bytes | bytearray, description: str) -> str | None:
        try:
            return bytes(value).decode()
        except UnicodeDecodeError:
            self._fail(f"{description} is not valid UTF-8")
            return None

    def _push_current(self) -> None:
        if self._current_name is not None:
            args = MappingProxyType({name: deepcopy(argument.value) for name, argument in self._args.items()})
            raw = bytes(self.raw[self._call_start : self.parse_pos]).decode()
            self._calls.append(ToolCall(self._current_name, args, raw))
        self._current_name = None
        self._args = {}
        self._param_name = None

    def _parse(self) -> None:
        while self.state in ("STRUCTURAL", "PARAM_VALUE"):
            if self.state == "PARAM_VALUE":
                if not self._parse_parameter_value():
                    return
                continue
            self.parse_pos = self._skip_space(self.raw, self.parse_pos)
            if self.parse_pos >= len(self.raw):
                return
            if self._after_call:
                self._parse_after_call()
                if self.state != "STRUCTURAL" or self._after_call:
                    return
                continue
            if self._current_name is None:
                if not self._parse_call() or self.state == "ERROR":
                    return
                continue
            if self.raw.startswith(self._call_close, self.parse_pos):
                if not self._parse_call_close():
                    return
                continue
            if self._partial(self._call_close, self.parse_pos):
                return
            self._parse_parameter()
            return

    def _parse_call(self) -> bool:
        raise NotImplementedError

    def _parse_parameter(self) -> None:
        raise NotImplementedError

    def _parse_parameter_value(self) -> bool:
        raise NotImplementedError

    def _parse_call_close(self) -> bool:
        self.parse_pos += len(self._call_close)
        self._push_current()
        self._after_call = True
        return True

    def _parse_after_call(self) -> None:
        if self.raw.startswith(_START, self.parse_pos):
            self._call_start = self.parse_pos
            self.parse_pos += len(_START)
            self._after_call = False
            return
        if self._partial(_START, self.parse_pos):
            return
        # The first non-tool byte belongs to ordinary output, not the stanza.
        del self.raw[self.parse_pos :]
        self.state = "DONE"

    def _update_param_close_prefix(self) -> None:
        tail = bytes(self.raw[-len(self._parameter_close) :])
        start = tail.rfind(b"<")
        self.param_close_prefix = start >= 0 and self._parameter_close.startswith(tail[start:])

    def _finish_active(self) -> None:
        trailing = self.raw[self.parse_pos :]
        if self._after_call and all(byte in _ASCII_SPACE for byte in trailing):
            self.parse_pos = len(self.raw)
            self._after_call = False
            self.state = "DONE"
            return
        self._fail(f"incomplete {self._family} tool-call stanza")
