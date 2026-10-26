"""Incremental DeepSeek DSML tool-call parser."""
from copy import deepcopy
import json
from types import MappingProxyType

from .base import ToolParser
from .model import ToolArgument, ToolCall
from .text import unescape_tool_text

_ASCII_SPACE = b" \t\r\n"
_CLOSE_PREFIX = "</｜DSML｜".encode()
_BAR = "｜".encode()


class Dsml(ToolParser):
    """Streaming DeepSeek DSML parser; v41 selects the spaced tag spellings."""
    def __init__(self, *, v41: bool = False):
        self.v41 = v41
        self._calls_name = " calls" if v41 else "tool_calls"
        self._invoke_name = " invoke" if v41 else "invoke"
        self._parameter_name = " parameter" if v41 else "parameter"
        self._canonical_start = f"<｜DSML｜{self._calls_name}>".encode()
        self._canonical_invoke = f"<｜DSML｜{self._invoke_name}".encode()
        self._param_close_text = f"</｜DSML｜{self._parameter_name}>"
        self._param_close = self._param_close_text.encode()
        self._call_start = 0
        self._current_name = None
        self._args = {}
        self._param_name = None
        self._param_is_string = False
        self._param_value_start = 0
        self._param_scan_pos = 0
        super().__init__()

    def _opening_forms(self):
        missing_bar = f"<DSML｜{self._calls_name}>".encode()
        invoke_missing_bar = f"<DSML｜{self._invoke_name}".encode()
        return (
            (self._canonical_start, False),
            (missing_bar, False),
            (self._canonical_invoke, True),
            (invoke_missing_bar, True),
        )

    def _start(self, implicit_invoke):
        self.state = "STRUCTURAL"
        self._active = True
        self.raw.extend(self._canonical_start)
        self.parse_pos = len(self.raw)
        if implicit_invoke:
            self.raw.extend(self._canonical_invoke)

    @staticmethod
    def _skip_space(data, position):
        while position < len(data) and data[position] in _ASCII_SPACE:
            position += 1
        return position

    def _close_prefix(self, name):
        return f"</｜DSML｜{name}".encode()

    def _close_tag_at(self, position, name):
        prefix = self._close_prefix(name)
        if not self.raw.startswith(prefix, position):
            return None
        cursor = self._skip_space(self.raw, position + len(prefix))
        if self.raw.startswith(_BAR, cursor):
            cursor = self._skip_space(self.raw, cursor + len(_BAR))
        if cursor < len(self.raw) and self.raw[cursor:cursor + 1] == b">":
            return cursor + 1
        return None

    @staticmethod
    def _attr(tag, name):
        pattern = name.encode() + b'=\"'
        start = tag.find(pattern)
        if start < 0:
            return None
        start += len(pattern)
        end = tag.find(b'"', start)
        if end < 0:
            return None
        return tag[start:end]

    def _open_tag_is(self, tag, name):
        prefix = f"<｜DSML｜{name}".encode()
        if not tag.startswith(prefix):
            return False
        return tag[len(prefix):len(prefix) + 1] in (b">", b" ", b"\t", b"\r", b"\n")

    def _decode(self, value, description):
        try:
            return bytes(value).decode()
        except UnicodeDecodeError:
            self._fail(f"{description} is not valid UTF-8")
            return None

    def _push_current(self, end):
        if self._current_name is not None:
            args = MappingProxyType({
                name: deepcopy(argument.value) for name, argument in self._args.items()
            })
            raw = bytes(self.raw[self._call_start:end]).decode()
            self._calls.append(ToolCall(self._current_name, args, raw))
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
            if self._parse_close(self._calls_name, done=True):
                return
            if self._parse_close(self._invoke_name, done=False):
                continue
            end = self.raw.find(b">", self.parse_pos)
            if end < 0:
                return
            tag = bytes(self.raw[self.parse_pos:end + 1])
            if self._open_tag_is(tag, self._invoke_name):
                self._parse_invoke(tag, end + 1)
            elif self._open_tag_is(tag, self._parameter_name):
                self._parse_parameter(tag, end + 1)
            else:
                shown = tag[:80].decode(errors="replace")
                self._fail(f"unexpected DSML tag: {shown}")
                return
            if self.state == "ERROR":
                return

    def _parse_close(self, name, *, done):
        end = self._close_tag_at(self.parse_pos, name)
        if end is None:
            return False
        self._push_current(self.parse_pos if done else end)
        self.parse_pos = end
        if done:
            self.state = "DONE"
        return True

    def _parse_invoke(self, tag, end):
        name = self._attr(tag, "name")
        name = self._decode(name, "tool invoke name") if name is not None else None
        if not name:
            self._fail("tool invoke without name")
            return
        self._call_start = self.parse_pos
        self._current_name = name
        self._args = {}
        self.parse_pos = end

    def _parse_parameter(self, tag, end):
        name = self._attr(tag, "name")
        name = self._decode(name, "tool parameter name") if name is not None else None
        if not name:
            self._fail("tool parameter without name")
            return
        self._param_name = name
        self._param_is_string = self._attr(tag, "string") == b"true"
        self.parse_pos = end
        self._param_value_start = end
        self._param_scan_pos = end
        self.state = "PARAM_VALUE"

    def _find_parameter_close(self):
        cursor = self._param_scan_pos
        while True:
            candidate = self.raw.find(_CLOSE_PREFIX, cursor)
            if candidate < 0:
                self._param_scan_pos = max(
                    self._param_value_start, len(self.raw) - len(_CLOSE_PREFIX) + 1
                )
                return None
            end = self._close_tag_at(candidate, self._parameter_name)
            if end is not None:
                return candidate, end
            if self.raw.find(b">", candidate) < 0:
                self._param_scan_pos = candidate
                return None
            cursor = candidate + 1

    def _parse_parameter_value(self):
        found = self._find_parameter_close()
        if found is None:
            return False
        end, after = found
        value_bytes = bytes(self.raw[self._param_value_start:end])
        text = self._decode(value_bytes, f"tool parameter {self._param_name!r}")
        if text is None:
            return True
        if self._param_is_string:
            value = unescape_tool_text(text, self._param_close_text)
        else:
            try:
                value = json.loads(text, parse_constant=self._reject_json_constant)
            except (json.JSONDecodeError, ValueError):
                self._fail(f"tool parameter {self._param_name!r} requires valid JSON")
                return True
        self._args[self._param_name] = ToolArgument(
            self._param_name, value, self._param_is_string
        )
        self._param_name = None
        self.param_close_prefix = False
        self.parse_pos = after
        self.state = "STRUCTURAL"
        return True

    @staticmethod
    def _reject_json_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    def _update_param_close_prefix(self):
        tail = bytes(self.raw[-64:])
        start = tail.rfind(b"<")
        if start < 0:
            self.param_close_prefix = False
            return
        matches, complete = self._parameter_close_tail(tail[start:])
        self.param_close_prefix = matches and not complete

    def _parameter_close_tail(self, tail):
        prefix = self._param_close[:-1]
        if len(tail) <= len(prefix):
            return prefix.startswith(tail), False
        if not tail.startswith(prefix):
            return False, False
        cursor = len(prefix)
        while cursor < len(tail) and tail[cursor] in _ASCII_SPACE:
            cursor += 1
        remaining = tail[cursor:]
        if remaining and len(remaining) < len(_BAR) and _BAR.startswith(remaining):
            return True, False
        if remaining.startswith(_BAR):
            cursor += len(_BAR)
        while cursor < len(tail) and tail[cursor] in _ASCII_SPACE:
            cursor += 1
        if cursor == len(tail):
            return True, False
        if tail[cursor:] == b">":
            return True, True
        return False, False
