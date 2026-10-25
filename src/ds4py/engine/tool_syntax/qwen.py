"""Incremental Qwen function/parameter tool-call parser."""
from copy import deepcopy
import json
import re
from types import MappingProxyType

from .base import ToolParser
from .model import ToolArgument, ToolCall
from .text import unescape_tool_text

_ASCII_SPACE = b" \t\r\n"
_START = b"<tool_call>"
_CLOSE = b"</tool_call>"
_FUNCTION = b"<function="
_FUNCTION_CLOSE = b"</function>"
_PARAMETER = b"<parameter="
_PARAMETER_CLOSE = b"</parameter>"
_NUMBER = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def value_is_json(value):
    start = 0
    end = len(value)
    while start < end and value[start] in b" \t":
        start += 1
    while end > start and value[end - 1] in b" \t\n":
        end -= 1
    trimmed = value[start:end]
    if not trimmed:
        return False
    if ((trimmed.startswith(b"{") and trimmed.endswith(b"}"))
            or (trimmed.startswith(b"[") and trimmed.endswith(b"]"))):
        return True
    if trimmed in (b"true", b"false", b"null"):
        return True
    return len(trimmed) < 64 and _NUMBER.fullmatch(trimmed) is not None


class Qwen(ToolParser):
    def __init__(self):
        self._after_call = False
        self._current_name = None
        self._args = {}
        self._param_name = None
        self._param_value_start = 0
        self._param_scan_pos = 0
        super().__init__()

    def _opening_forms(self):
        return ((_START, False),)

    def _start(self, implicit_invoke):
        self.state = "STRUCTURAL"
        self._active = True
        self.raw.extend(_START)
        self.parse_pos = len(self.raw)

    @staticmethod
    def _skip_space(data, position):
        while position < len(data) and data[position] in _ASCII_SPACE:
            position += 1
        return position

    def _partial(self, prefix, position):
        remaining = bytes(self.raw[position:])
        return len(remaining) < len(prefix) and prefix.startswith(remaining)

    def _decode(self, value, description):
        try:
            return bytes(value).decode()
        except UnicodeDecodeError:
            self._fail(f"{description} is not valid UTF-8")
            return None

    def _push_current(self):
        if self._current_name is not None:
            args = MappingProxyType({
                name: deepcopy(argument.value) for name, argument in self._args.items()
            })
            self._calls.append(ToolCall(self._current_name, args))
        self._current_name = None
        self._args = {}
        self._param_name = None

    def _parse(self):
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
                if not self._parse_function():
                    return
                continue
            if self.raw.startswith(_FUNCTION_CLOSE, self.parse_pos):
                if not self._parse_function_close():
                    return
                continue
            if self._partial(_FUNCTION_CLOSE, self.parse_pos):
                return
            self._parse_parameter()
            return

    def _parse_after_call(self):
        if self.raw.startswith(_START, self.parse_pos):
            self.parse_pos += len(_START)
            self._after_call = False
            return
        if self._partial(_START, self.parse_pos):
            return
        del self.raw[self.parse_pos:]
        self.state = "DONE"

    def _parse_function(self):
        if not self.raw.startswith(_FUNCTION, self.parse_pos):
            if self._partial(_FUNCTION, self.parse_pos):
                return False
            self._fail("expected <function=...> in Qwen tool call")
            return False
        start = self.parse_pos + len(_FUNCTION)
        end = self.raw.find(b">", start)
        if end < 0:
            return False
        name = self._decode(
            self.raw[start:end].strip(_ASCII_SPACE), "Qwen function name"
        )
        if not name:
            if self.state != "ERROR":
                self._fail("Qwen tool call without function name")
            return False
        self._current_name = name
        self.parse_pos = end + 1
        return True

    def _parse_function_close(self):
        after = self._skip_space(self.raw, self.parse_pos + len(_FUNCTION_CLOSE))
        if after >= len(self.raw) or self._partial(_CLOSE, after):
            return False
        if not self.raw.startswith(_CLOSE, after):
            self._fail("expected </tool_call> after </function>")
            return False
        self.parse_pos = after + len(_CLOSE)
        self._push_current()
        self._after_call = True
        return True

    def _parse_parameter(self):
        if not self.raw.startswith(_PARAMETER, self.parse_pos):
            if self._partial(_PARAMETER, self.parse_pos):
                return
            self._fail("expected <parameter=...> or </function> in Qwen tool call")
            return
        start = self.parse_pos + len(_PARAMETER)
        end = self.raw.find(b">", start)
        if end < 0:
            return
        name = self._decode(
            self.raw[start:end].strip(_ASCII_SPACE), "Qwen parameter name"
        )
        if not name:
            if self.state != "ERROR":
                self._fail("empty <parameter=> name in Qwen tool call")
            return
        self._param_name = name
        self.parse_pos = end + 1
        self._param_value_start = self.parse_pos
        self._param_scan_pos = self.parse_pos
        self.state = "PARAM_VALUE"

    def _parse_parameter_value(self):
        end = self.raw.find(_PARAMETER_CLOSE, self._param_scan_pos)
        if end < 0:
            self._param_scan_pos = max(
                self._param_value_start, len(self.raw) - len(_PARAMETER_CLOSE) + 1
            )
            return False
        close_start = end
        start = self._param_value_start
        if start < end and self.raw[start] == 10:
            start += 1
        if end > start and self.raw[end - 1] == 10:
            end -= 1
        value_bytes = bytes(self.raw[start:end])
        text = self._decode(value_bytes, f"Qwen parameter {self._param_name!r}")
        if text is None:
            return True
        is_json = value_is_json(value_bytes)
        if is_json:
            try:
                value = json.loads(text, parse_constant=self._reject_json_constant)
            except (json.JSONDecodeError, ValueError):
                self._fail(f"Qwen parameter {self._param_name!r} contains malformed JSON")
                return True
        else:
            value = unescape_tool_text(text, _PARAMETER_CLOSE.decode())
        self._args[self._param_name] = ToolArgument(self._param_name, value, not is_json)
        self._param_name = None
        self.parse_pos = close_start + len(_PARAMETER_CLOSE)
        self.state = "STRUCTURAL"
        return True

    @staticmethod
    def _reject_json_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    def _update_param_close_prefix(self):
        tail = bytes(self.raw[-len(_PARAMETER_CLOSE):])
        start = tail.rfind(b"<")
        self.param_close_prefix = start >= 0 and _PARAMETER_CLOSE.startswith(tail[start:])

    def _finish_active(self):
        trailing = self.raw[self.parse_pos:]
        if self._after_call and all(byte in _ASCII_SPACE for byte in trailing):
            self.parse_pos = len(self.raw)
            self._after_call = False
            self.state = "DONE"
            return
        self._fail("incomplete Qwen tool-call stanza")
