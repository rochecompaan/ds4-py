"""Incremental Hermes JSON tool-call parser."""

import json
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
        return json.loads(text, parse_constant=_reject_constant)
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
                self._fail("argument is delimited with backticks; JSON strings need double quotes")
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

    def _publish_or_fail(self, close_end: int) -> None:
        if self._deferred_error is not None:
            self._fail(self._deferred_error)
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
        self._fail(self._deferred_error or "incomplete Hermes tool-call stanza")

    @override
    def wants_greedy_sampling(self) -> bool:
        if self.state in ("DONE", "ERROR"):
            return False
        if not self._active:
            return super().wants_greedy_sampling()
        if self._after_call:
            return False
        return self.state == "STRUCTURAL"
