"""Incremental Hermes JSON tool-call parser."""

import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Never, override

from .base import ToolParser, freeze_calls, thaw_calls
from .model import ToolCall

_START = b"<tool_call>"
_CLOSE = b"</tool_call>"
_ASCII_SPACE = b" \t\r\n"
_MAX_DEPTH = 256
_TOP_LEVEL_DEPTH = 1


def _reject_constant(value: str) -> Never:
    raise ValueError(f"invalid JSON constant: {value}")


def _loads(text: str) -> object:
    try:
        # The reference accepts raw control bytes inside strings during the
        # bounded repair path; NUL remains explicitly rejected below.
        return json.loads(text, parse_constant=_reject_constant, strict=False)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"invalid JSON in Hermes tool call: {error}") from error


def _validate_strings(value: object) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            if "\x00" in item:
                raise ValueError("NUL is not allowed in a Hermes tool call")
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())


def _validate_depth(value: object, depth: int = 0) -> None:
    pending = [(value, depth)]
    while pending:
        item, level = pending.pop()
        if not isinstance(item, (list, dict)):
            continue
        if level >= _MAX_DEPTH:
            raise ValueError("tool call argument nests too deeply")
        children = item if isinstance(item, list) else item.values()
        pending.extend((child, level + 1) for child in children)


def _decode_call(body: str) -> ToolCall:
    value = _loads(body)
    _validate_strings(value)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object after <tool_call>")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Hermes tool call requires a nonempty string name")
    arguments = value.get("arguments", {})
    if isinstance(arguments, str):
        arguments = _loads(arguments)
        _validate_strings(arguments)
    if not isinstance(arguments, dict):
        raise TypeError("Hermes tool call arguments must be an object")
    for key, item in value.items():
        if key not in ("name", "arguments"):
            _validate_depth(item)
    copied: dict[str, object] = {}
    for key, item in arguments.items():
        if not isinstance(key, str):
            raise TypeError("Hermes argument keys must be strings")
        _validate_depth(item)
        copied[key] = item
    return ToolCall(name, MappingProxyType(copied), "")


def _eat(text: str, position: int, pattern: str, end: int) -> int | None:
    for char in pattern:
        if char == " ":
            while position < end and text[position] in " \t\r\n":
                position += 1
        elif position < end and text[position] == char:
            position += 1
        else:
            return None
    return position


def _bare_quote(text: str, position: int, start: int) -> bool:
    backslashes = 0
    while position - backslashes > start and text[position - backslashes - 1] == "\\":
        backslashes += 1
    return text[position] == '"' and backslashes % 2 == 0


def _folds_member(text: str, start: int, end: int) -> bool:
    for position in range(start, end):
        if not _bare_quote(text, position, start):
            continue
        for key in ("path", "content", "edits", "oldText", "newText"):
            if _eat(text, position, f'" , "{key}" :', end) is not None:
                return True
    return False


def _edit_boundary(text: str, position: int, end: int, *, to_new: bool) -> int | None:
    pattern = '" , "newText" : "' if to_new else '" } , { "oldText" : "'
    return _eat(text, position, pattern, end)


@dataclass(frozen=True)
class _RepairShape:
    value_start: int
    value_end: int
    edit: bool
    legacy: bool


def _repair_shape(text: str) -> _RepairShape | None:
    end = len(text)
    position = _eat(text, 0, ' { "name" : "write" , "arguments" : { "path" : ', end)
    edit = position is None
    if edit:
        position = _eat(text, 0, ' { "name" : "edit" , "arguments" : { "path" : ', end)
    if position is None:
        return None
    try:
        path, after_path = json.JSONDecoder().raw_decode(text, position)
    except ValueError, RecursionError:
        return None
    if not isinstance(path, str) or "\x00" in path:
        return None
    legacy = False
    if edit:
        value_start = _eat(text, after_path, ' , "edits" : [ { "oldText" : "', end)
        legacy = value_start is None
        if legacy:
            value_start = _eat(text, after_path, ' , "oldText" : "', end)
    else:
        value_start = _eat(text, after_path, ' , "content" : "', end)
    if value_start is None:
        return None
    tail_pattern = r'"\s*\}\s*\]\s*\}\s*\}\s*\Z' if edit and not legacy else r'"\s*\}\s*\}\s*\Z'
    tail = re.search(tail_pattern, text)
    return None if tail is None else _RepairShape(value_start, tail.start(), edit, legacy)


def _repair_cuts(text: str, shape: _RepairShape) -> list[int] | None:
    cuts = [shape.value_start]
    to_new = True
    if shape.edit:
        position = shape.value_start
        while position < shape.value_end:
            if _bare_quote(text, position, shape.value_start):
                boundary = _edit_boundary(text, position, shape.value_end, to_new=to_new)
                if boundary is not None and not (shape.legacy and not to_new):
                    cuts.extend((position, boundary))
                    position = boundary
                    to_new = not to_new
                    continue
                opposite = _edit_boundary(text, position, shape.value_end, to_new=not to_new)
                if boundary is not None or opposite is not None:
                    return None
            position += 1
        if to_new:
            return None
    cuts.append(shape.value_end)
    if any(_folds_member(text, cuts[index], cuts[index + 1]) for index in range(0, len(cuts), 2)):
        return None
    return cuts


def _escape_repair_cuts(text: str, cuts: list[int]) -> str | None:
    output = []
    copied = 0
    changed = False
    for index in range(0, len(cuts), 2):
        start, stop = cuts[index], cuts[index + 1]
        output.append(text[copied:start])
        for position in range(start, stop):
            if _bare_quote(text, position, start):
                output.append("\\")
                changed = True
            output.append(text[position])
        copied = stop
    output.append(text[copied:])
    return "".join(output) if changed else None


def repair_hermes_quotes(text: str) -> str | None:
    """Escape bare quotes only in the reference's unambiguous write/edit forms."""
    shape = _repair_shape(text)
    if shape is None:
        return None
    cuts = _repair_cuts(text, shape)
    return None if cuts is None else _escape_repair_cuts(text, cuts)


def _key_before(data: bytes | bytearray, start: int, position: int) -> str:
    cursor = position
    while cursor > start and data[cursor - 1] in _ASCII_SPACE:
        cursor -= 1
    if cursor <= start or data[cursor - 1] != ord(":"):
        return ""
    cursor -= 1
    while cursor > start and data[cursor - 1] in _ASCII_SPACE:
        cursor -= 1
    if cursor <= start or data[cursor - 1] != ord('"'):
        return ""
    close = cursor - 1
    opening = data.rfind(b'"', start, close)
    if opening < 0:
        return ""
    return bytes(data[opening + 1 : close]).decode(errors="replace")


@dataclass(frozen=True)
class HermesState:
    call_start: int
    body_start: int
    scan_pos: int
    body_end: int | None
    after_call: bool
    in_string: bool
    escape: bool
    containers: tuple[str, ...]
    string_value: bool
    pending: tuple[ToolCall, ...]
    deferred_error: str | None


class Hermes(ToolParser):
    """Parse one strict JSON object inside each Hermes tool-call stanza."""

    def __init__(self) -> None:
        self._call_start = 0
        self._body_start = 0
        self._scan_pos = 0
        self._body_end: int | None = None
        self._after_call = False
        self._in_string = False
        self._escape = False
        self._containers: list[str] = []
        self._string_value = False
        self._pending_call: ToolCall | None = None
        self._deferred_error: str | None = None
        super().__init__()

    @override
    def _opening_forms(self) -> tuple[tuple[bytes, bool], ...]:
        return ((_START, False),)

    @override
    def _capture_family(self) -> HermesState:
        return HermesState(
            self._call_start,
            self._body_start,
            self._scan_pos,
            self._body_end,
            self._after_call,
            self._in_string,
            self._escape,
            tuple(self._containers),
            self._string_value,
            freeze_calls([self._pending_call] if self._pending_call is not None else []),
            self._deferred_error,
        )

    @override
    def _restore_family(self, state: object) -> None:
        if not isinstance(state, HermesState):
            raise TypeError("Hermes requires HermesState")
        self._call_start = state.call_start
        self._body_start = state.body_start
        self._scan_pos = state.scan_pos
        self._body_end = state.body_end
        self._after_call = state.after_call
        self._in_string = state.in_string
        self._escape = state.escape
        self._containers = list(state.containers)
        self._string_value = state.string_value
        pending = thaw_calls(state.pending)
        self._pending_call = pending[0] if pending else None
        self._deferred_error = state.deferred_error

    @override
    def _new_fork(self) -> Hermes:
        return Hermes()

    def _begin_call(self, start: int, *, present: bool) -> None:
        self._call_start = start
        self._body_start = start + len(_START)
        self._scan_pos = self._body_start
        self._body_end = None
        self._after_call = False
        self._in_string = False
        self._escape = False
        self._containers = []
        self._string_value = False
        self._pending_call = None
        self._deferred_error = None
        if not present:
            self.raw.extend(_START)
        self.parse_pos = self._body_start
        self.state = "STRUCTURAL"
        self._active = True

    @override
    def _start(self, implicit_invoke: bool) -> None:
        self._begin_call(len(self.raw), present=False)

    def _previous_significant(self, position: int) -> int | None:
        for index in range(position - 1, self._body_start - 1, -1):
            if self.raw[index] not in _ASCII_SPACE:
                return self.raw[index]
        return None

    def _scan_string(self, byte: int) -> None:
        if byte == 0:
            self._fail("NUL is not allowed in a Hermes tool call")
        elif self._escape:
            self._escape = False
        elif byte == ord("\\"):
            self._escape = True
        elif byte == ord('"'):
            self._in_string = False
            self.state = "STRUCTURAL"

    def _start_string(self, index: int) -> None:
        previous = self._previous_significant(index)
        self._string_value = len(self._containers) > _TOP_LEVEL_DEPTH and (
            previous == ord(":") or (self._containers[-1] == "[" and previous in (ord("["), ord(",")))
        )
        self._in_string = True
        self.state = "PARAM_VALUE" if self._string_value else "STRUCTURAL"

    def _scan_body(self) -> None:
        while self._scan_pos < len(self.raw):
            index = self._scan_pos
            byte = self.raw[index]
            self._scan_pos = index + 1
            if self._in_string:
                self._scan_string(byte)
                if self.state == "ERROR":
                    return
                continue
            if byte == 0:
                self._fail("NUL is not allowed in a Hermes tool call")
                return
            if byte == ord("`"):
                key = _key_before(self.raw, self._body_start, index)
                self._fail(f'argument "{key}" is delimited with backticks; JSON strings need double quotes')
                return
            if not self._containers and byte in _ASCII_SPACE:
                self._body_start = index + 1
                continue
            if not self._containers and byte != ord("{"):
                self._fail("expected a JSON object after <tool_call>")
                return
            if byte == ord('"'):
                self._start_string(index)
                continue
            if byte in (ord("{"), ord("[")):
                self._containers.append(chr(byte))
                continue
            if byte in (ord("}"), ord("]")):
                if self._containers and self._containers[-1] == chr(ord("{") if byte == ord("}") else ord("[")):
                    self._containers.pop()
                if not self._containers:
                    self._body_end = index + 1
                    self.parse_pos = self._body_end
                    self.state = "STRUCTURAL"
                    return

    def _decode_body(self) -> None:
        if self._body_end is None:
            raise AssertionError("Hermes body decoding requires a complete object")
        try:
            body = bytes(self.raw[self._body_start : self._body_end]).decode()
            self._pending_call = _decode_call(body)
        except (UnicodeDecodeError, TypeError, ValueError) as error:
            self._pending_call = None
            self._deferred_error = str(error)

    def _repair_pending(self, close_end: int) -> bool:
        body_end = self._body_end
        if body_end is None:
            body_end = close_end - len(_CLOSE)
            while body_end > self._body_start and self.raw[body_end - 1] in _ASCII_SPACE:
                body_end -= 1
        try:
            body = bytes(self.raw[self._body_start : body_end]).decode()
            repaired = repair_hermes_quotes(body)
            if repaired is None:
                return False
            self._pending_call = _decode_call(repaired)
        except UnicodeDecodeError, TypeError, ValueError:
            return False
        self._deferred_error = None
        return True

    def _publish_or_fail(self, close_end: int) -> None:
        if self._deferred_error is not None and not self._repair_pending(close_end):
            self._fail(self._deferred_error or "invalid JSON in Hermes tool call")
            return
        if self._pending_call is None:
            raise AssertionError("completed Hermes call requires a pending value")
        pending = self._pending_call
        raw = bytes(self.raw[self._call_start : close_end]).decode()
        self._calls.append(ToolCall(pending.name, pending.args, raw))
        self._pending_call = None
        self._after_call = True
        self.parse_pos = close_end

    def _parse_close(self) -> bool:
        if self._body_end is None:
            raise AssertionError("Hermes close parsing requires a body")
        position = self._body_end
        while position < len(self.raw) and self.raw[position] in _ASCII_SPACE:
            position += 1
        remaining = bytes(self.raw[position:])
        if not remaining:
            return False
        if len(remaining) < len(_CLOSE) and _CLOSE.startswith(remaining):
            return False
        if not remaining.startswith(_CLOSE):
            self._fail("expected </tool_call> after the Hermes call object")
            return False
        self._publish_or_fail(position + len(_CLOSE))
        return self.state != "ERROR"

    def _parse_after_call(self) -> bool:
        position = self.raw.find(_START, self.parse_pos)
        if position < 0:
            self.parse_pos = max(self.parse_pos, len(self.raw) - len(_START) + 1)
            return False
        self._begin_call(position, present=True)
        return True

    @override
    def _parse(self) -> None:
        while self.state in ("STRUCTURAL", "PARAM_VALUE"):
            if self._after_call:
                if not self._parse_after_call():
                    return
                continue
            if self._body_end is None:
                self._scan_body()
                if self.state == "ERROR" or self._body_end is None:
                    return
                self._decode_body()
            if not self._parse_close():
                return
            if self.state == "ERROR":
                return

    @override
    def _finish_active(self) -> None:
        if self._after_call:
            self.state = "DONE"
            return
        close_end = len(self.raw)
        while close_end > 0 and self.raw[close_end - 1] in _ASCII_SPACE:
            close_end -= 1
        if self._body_end is None and self.raw.startswith(_CLOSE, close_end - len(_CLOSE)):
            self._deferred_error = self._deferred_error or "invalid JSON in Hermes tool call"
            self._publish_or_fail(close_end)
            if self.state != "ERROR":
                self.state = "DONE"
            return
        self._fail(self._deferred_error or "incomplete Hermes tool-call stanza")

    @override
    def wants_greedy_sampling(self) -> bool:
        if self.state in ("DONE", "ERROR"):
            return False
        if not self._active:
            return super().wants_greedy_sampling()
        if self._after_call:
            tail = bytes(self.raw)
            return any(_START.startswith(tail[-size:]) for size in range(2, min(len(_START), len(tail) + 1)))
        return self.state == "STRUCTURAL"
