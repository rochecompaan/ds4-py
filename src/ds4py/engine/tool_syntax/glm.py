"""Incremental GLM name-first tool-call parser."""

from .delimiter import DelimiterParser
from .model import ToolArgument
from .text import unescape_tool_text

_ASCII_SPACE = b" \t\r\n"
_CLOSE = b"</tool_call>"
_ARG_KEY = b"<arg_key>"
_ARG_KEY_CLOSE = b"</arg_key>"
_ARG_VALUE = b"<arg_value>"
_ARG_VALUE_CLOSE = b"</arg_value>"


class Glm(DelimiterParser):
    """Streaming GLM name-first parser with string-valued arguments."""

    _call_close = _CLOSE
    _parameter_close = _ARG_VALUE_CLOSE
    _family = "GLM"

    def _parse_call(self):
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

    def _parse_parameter(self):
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
