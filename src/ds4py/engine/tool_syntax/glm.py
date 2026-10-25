"""Incremental GLM name-first tool-call parser."""
from copy import deepcopy
from types import MappingProxyType

from .base import ToolParser
from .model import ToolArgument, ToolCall
from .text import unescape_tool_text

_ASCII_SPACE = b" \t\r\n"
_START = b"<tool_call>"
_CLOSE = b"</tool_call>"
_ARG_KEY = b"<arg_key>"
_ARG_KEY_CLOSE = b"</arg_key>"
_ARG_VALUE = b"<arg_value>"
_ARG_VALUE_CLOSE = b"</arg_value>"


class Glm(ToolParser):
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
                if not self._parse_function_name():
                    return
                if self.state == "ERROR":
                    return
                continue
            if self.raw.startswith(_CLOSE, self.parse_pos):
                self.parse_pos += len(_CLOSE)
                self._push_current()
                self._after_call = True
                continue
            if self._partial(_CLOSE, self.parse_pos):
                return
            self._parse_argument()
            return

    def _parse_after_call(self):
        if self.raw.startswith(_START, self.parse_pos):
            self.parse_pos += len(_START)
            self._after_call = False
            return
        if self._partial(_START, self.parse_pos):
            return
        # The first non-tool byte belongs to ordinary output, not the stanza.
        del self.raw[self.parse_pos:]
        self.state = "DONE"

    def _parse_function_name(self):
        end = self.raw.find(b"<", self.parse_pos)
        if end < 0:
            return False
        at_key = self.raw.startswith(_ARG_KEY, end)
        at_close = self.raw.startswith(_CLOSE, end)
        if not at_key and not at_close:
            if self._partial(_ARG_KEY, end) or self._partial(_CLOSE, end):
                return False
            self._fail("expected <arg_key> or </tool_call> in GLM tool call")
            return True
        name = bytes(self.raw[self.parse_pos:end]).strip(_ASCII_SPACE)
        name = self._decode(name, "GLM function name")
        if not name:
            if self.state != "ERROR":
                self._fail("GLM tool call without function name")
            return True
        self._current_name = name
        self.parse_pos = end
        return True

    def _parse_argument(self):
        if not self.raw.startswith(_ARG_KEY, self.parse_pos):
            if self._partial(_ARG_KEY, self.parse_pos):
                return
            self._fail("expected <arg_key> in GLM tool call")
            return
        key_start = self.parse_pos + len(_ARG_KEY)
        key_end = self.raw.find(_ARG_KEY_CLOSE, key_start)
        if key_end < 0:
            return
        key = bytes(self.raw[key_start:key_end]).strip(_ASCII_SPACE)
        key = self._decode(key, "GLM argument name")
        if not key:
            if self.state != "ERROR":
                self._fail("empty <arg_key> in GLM tool call")
            return
        key = unescape_tool_text(key, _ARG_KEY_CLOSE.decode())
        cursor = self._skip_space(self.raw, key_end + len(_ARG_KEY_CLOSE))
        if not self.raw.startswith(_ARG_VALUE, cursor):
            if self._partial(_ARG_VALUE, cursor):
                return
            self._fail("expected <arg_value> in GLM tool call")
            return
        self._param_name = key
        self.parse_pos = cursor + len(_ARG_VALUE)
        self._param_value_start = self.parse_pos
        self._param_scan_pos = self.parse_pos
        self.state = "PARAM_VALUE"

    def _parse_parameter_value(self):
        end = self.raw.find(_ARG_VALUE_CLOSE, self._param_scan_pos)
        if end < 0:
            self._param_scan_pos = max(
                self._param_value_start, len(self.raw) - len(_ARG_VALUE_CLOSE) + 1
            )
            return False
        text = self._decode(
            self.raw[self._param_value_start:end], f"GLM argument {self._param_name!r}"
        )
        if text is None:
            return True
        value = unescape_tool_text(text, _ARG_VALUE_CLOSE.decode())
        self._args[self._param_name] = ToolArgument(self._param_name, value, True)
        self._param_name = None
        self.parse_pos = end + len(_ARG_VALUE_CLOSE)
        self.state = "STRUCTURAL"
        return True

    def _update_param_close_prefix(self):
        tail = bytes(self.raw[-len(_ARG_VALUE_CLOSE):])
        start = tail.rfind(b"<")
        self.param_close_prefix = start >= 0 and _ARG_VALUE_CLOSE.startswith(tail[start:])

    def _finish_active(self):
        trailing = self.raw[self.parse_pos:]
        if self._after_call and all(byte in _ASCII_SPACE for byte in trailing):
            self.parse_pos = len(self.raw)
            self._after_call = False
            self.state = "DONE"
            return
        self._fail("incomplete GLM tool-call stanza")
