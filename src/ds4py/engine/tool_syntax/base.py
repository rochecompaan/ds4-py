"""Shared byte-stream state for family-specific tool parsers."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from .model import ToolArgument

if TYPE_CHECKING:
    from ds4py._native.protocol import EngineInfo

from .model import Snapshot, ToolCall

type FrozenValue = bool | int | float | str | tuple[FrozenValue, ...] | MappingProxyType[str, FrozenValue] | None


@dataclass(frozen=True)
class ToolParserState:
    state: str
    raw: bytes
    parse_pos: int
    error: str | None
    param_close_prefix: bool
    in_think: bool
    dsml_in_think: bool
    calls: tuple[ToolCall, ...]
    active: bool
    pending: bytes
    marker_tail: bytes
    family: object


def freeze_value(value: object) -> FrozenValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("tool argument keys must be strings")
            frozen[key] = freeze_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    raise TypeError(f"unsupported tool argument value: {type(value).__name__}")


def thaw_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported frozen tool argument value: {type(value).__name__}")


def freeze_argument(argument: ToolArgument) -> ToolArgument:
    return ToolArgument(argument.name, freeze_value(argument.value), argument.is_string)


def thaw_arguments(arguments: tuple[tuple[str, ToolArgument], ...]) -> dict[str, ToolArgument]:
    return {
        name: ToolArgument(argument.name, thaw_value(argument.value), argument.is_string)
        for name, argument in arguments
    }


def freeze_calls(calls: list[ToolCall]) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(
            call.name, MappingProxyType({name: freeze_value(value) for name, value in call.args.items()}), call.raw
        )
        for call in calls
    )


def thaw_calls(calls: tuple[ToolCall, ...]) -> list[ToolCall]:
    return [
        ToolCall(
            call.name,
            MappingProxyType({name: thaw_value(value) for name, value in call.args.items()}),
            call.raw,
        )
        for call in calls
    ]


_THINK_OPEN = b"<think>"
_THINK_CLOSE = b"</think>"
_DSML_MARKERS = tuple(marker.encode() for marker in ("｜DSML｜", "|DSML|", "<DSML｜", "</DSML｜"))


class ToolParser:
    """Per-round byte parser with immutable snapshots and per-call canonical raw text.

    Own it on the decode worker; do not feed it concurrently.
    """

    def __init__(self) -> None:
        self.state = "SEARCH"
        self.raw = bytearray()
        self.parse_pos = 0
        self.error: str | None = None
        self.param_close_prefix = False
        self.in_think = False
        self.dsml_in_think = False
        self._calls: list[ToolCall] = []
        self._active = False
        self._pending = bytearray()
        self._marker_tail = bytearray()
        self._forms = tuple(self._opening_forms())

    def _opening_forms(self) -> tuple[tuple[bytes, bool], ...]:
        raise NotImplementedError

    def _capture_family(self) -> object:
        raise NotImplementedError

    def _restore_family(self, state: object) -> None:
        raise NotImplementedError

    def _new_fork(self) -> ToolParser:
        raise NotImplementedError

    def capture(self) -> ToolParserState:
        return ToolParserState(
            self.state,
            bytes(self.raw),
            self.parse_pos,
            self.error,
            self.param_close_prefix,
            self.in_think,
            self.dsml_in_think,
            freeze_calls(self._calls),
            self._active,
            bytes(self._pending),
            bytes(self._marker_tail),
            self._capture_family(),
        )

    def restore(self, state: ToolParserState) -> None:
        self.state = state.state
        self.raw = bytearray(state.raw)
        self.parse_pos = state.parse_pos
        self.error = state.error
        self.param_close_prefix = state.param_close_prefix
        self.in_think = state.in_think
        self.dsml_in_think = state.dsml_in_think
        self._calls = thaw_calls(state.calls)
        self._active = state.active
        self._pending = bytearray(state.pending)
        self._marker_tail = bytearray(state.marker_tail)
        self._restore_family(state.family)

    def fork(self) -> ToolParser:
        parser = self._new_fork()
        parser.restore(self.capture())
        return parser

    def _start(self, implicit_invoke: bool) -> None:
        raise NotImplementedError

    def _parse(self) -> None:
        raise NotImplementedError

    def _update_param_close_prefix(self) -> None:
        self.param_close_prefix = False

    def _snapshot(self) -> Snapshot:
        return Snapshot(
            self.state,
            bytes(self.raw),
            tuple(self._calls),
            self.error,
            self.param_close_prefix,
            self.in_think,
            self.dsml_in_think,
        )

    def _fail(self, message: str) -> None:
        self.state = "ERROR"
        self.error = message
        self._calls.clear()
        self.param_close_prefix = False

    def feed(self, data: bytes) -> Snapshot:
        """Consume bytes incrementally; terminal DONE/ERROR snapshots remain stable."""
        if self.state in ("DONE", "ERROR"):
            return self._snapshot()
        for byte in bytes(data):
            if self.state in ("DONE", "ERROR"):
                break
            self._feed_byte(byte)
        return self._snapshot()

    def _feed_byte(self, byte: int) -> None:
        """Apply one byte without copying the parser's accumulated state."""
        if self._active:
            self.raw.append(byte)
            self._parse()
            if self.state == "PARAM_VALUE":
                self._update_param_close_prefix()
            else:
                self.param_close_prefix = False
        else:
            self._feed_inactive(byte)

    def _feed_inactive(self, byte: int) -> None:
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
            markers = (_THINK_OPEN, _THINK_CLOSE, *(form[0] for form in self._forms))
            if any(marker.startswith(pending) for marker in markers):
                return
            ordinary = self._pending[0]
            del self._pending[0]
            self._note_ordinary_byte(ordinary)

    def _note_ordinary_byte(self, byte: int) -> None:
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

    def finish(self) -> Snapshot:
        """Finalize end-of-generation syntax, rejecting unfinished active calls."""
        if self.state in ("DONE", "ERROR"):
            return self._snapshot()
        if self._active:
            self._finish_active()
        elif len(self._pending) > 1 and any(form[0].startswith(self._pending) for form in self._forms):
            self._fail("incomplete tool-call opening marker")
        return self._snapshot()

    def _finish_active(self) -> None:
        self._fail("incomplete tool-call stanza")

    def wants_greedy_sampling(self) -> bool:
        """Request structural greedy sampling, never in DONE/ERROR."""
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


def make_parser(info: EngineInfo) -> ToolParser:
    """Select a fresh parser from engine family flags; performs no native calls."""
    from .dsml import Dsml  # noqa: PLC0415 - Lazy parser registration avoids family import cycles.

    if info.is_qwen4:
        from .qwen import Qwen  # noqa: PLC0415 - Lazy parser registration avoids family import cycles.

        return Qwen()
    if info.is_glm_dsa:
        from .glm import Glm  # noqa: PLC0415 - Lazy parser registration avoids family import cycles.

        return Glm()
    return Dsml(v41=info.is_deepseek41)
