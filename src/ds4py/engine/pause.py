"""Bounded UTF-8 stop matching and resumable control-span state."""

from dataclasses import dataclass
from typing import Iterable

ENCODE_STEP_TOKENS = "ENCODE_STEP_TOKENS"
DECODE_RESULT = "DECODE_RESULT"
_FEED_STATES = frozenset((ENCODE_STEP_TOKENS, DECODE_RESULT))


def _failure_table(sequence: bytes) -> tuple[int, ...]:
    table = [0] * len(sequence)
    prefix = 0
    for index in range(1, len(sequence)):
        while prefix and sequence[index] != sequence[prefix]:
            prefix = table[prefix - 1]
        if sequence[index] == sequence[prefix]:
            prefix += 1
        table[index] = prefix
    return tuple(table)


class StopSequenceMatcher:
    """KMP matchers with fixed memory and independent, cloneable state."""

    def __init__(self, sequences: Iterable[bytes] = ()):
        normalized = []
        for sequence in sequences:
            sequence = bytes(sequence)
            if not sequence:
                raise ValueError("stop sequences must not be empty")
            sequence.decode("utf-8")
            normalized.append(sequence)
        self.sequences = tuple(normalized)
        self._tables = tuple(_failure_table(sequence) for sequence in self.sequences)
        self._states = [0] * len(self.sequences)

    @property
    def pending(self) -> int:
        return max(self._states, default=0)

    @property
    def buffered(self) -> int:
        return self.pending

    @property
    def active_sequences(self) -> list[bytes]:
        return [sequence for sequence, state in zip(self.sequences, self._states) if state]

    def feed(self, data: bytes) -> bytes | None:
        """Consume arbitrary byte chunks and return the first completed sequence."""
        matched = None
        for byte in bytes(data):
            for index, sequence in enumerate(self.sequences):
                state = self._states[index]
                table = self._tables[index]
                while state and byte != sequence[state]:
                    state = table[state - 1]
                if byte == sequence[state]:
                    state += 1
                if state == len(sequence):
                    if matched is None:
                        matched = sequence
                    state = table[-1]
                self._states[index] = state
        return matched

    def clone(self) -> "StopSequenceMatcher":
        clone = StopSequenceMatcher(self.sequences)
        clone._states = self._states.copy()
        return clone

    def inherit(self, parent: "StopSequenceMatcher") -> "StopSequenceMatcher":
        self.sequences = parent.sequences
        self._tables = parent._tables
        self._states = parent._states.copy()
        return self


@dataclass(frozen=True, slots=True)
class TextRecord:
    text: bytes
    override_ids: tuple[int, ...] = ()


class ControlSpan:
    """Structural/text region with resumable stop matching.

    Thinking regions do not call this object. Parser states other than
    ENCODE_STEP_TOKENS and DECODE_RESULT cannot advance its matcher.
    """

    def __init__(self, sequences: Iterable[bytes] = (), *, state: str = DECODE_RESULT):
        self.matcher = StopSequenceMatcher(sequences)
        self.state = state
        self.stop_sequence: bytes | None = None
        self.active_stop_sequences: list[bytes] = []
        self.text_record: TextRecord | None = None

    @property
    def sequences(self) -> tuple[bytes, ...]:
        return self.matcher.sequences

    def _feeds(self, state: str | None) -> bool:
        if state is not None and state not in _FEED_STATES:
            return False
        if state is not None:
            self.state = state
        return self.state in _FEED_STATES

    def feed(self, data: bytes, *, state: str | None = None) -> bytes | None:
        if not self._feeds(state) or self.stop_sequence is not None:
            return None
        matched = self.matcher.feed(data)
        if matched is not None:
            self.stop(matched)
        else:
            self.active_stop_sequences = self.matcher.active_sequences
        return matched

    def append(self, data: bytes, *, state: str | None = None) -> bytes | None:
        """Append existing text without replacing or resetting matcher state."""
        return self.feed(data, state=state)

    def stop(self, sequence: bytes) -> None:
        self.stop_sequence = sequence
        self.active_stop_sequences = []

    def finish(self) -> None:
        self.stop_sequence = None

    def done(self) -> None:
        self.finish()
        self.active_stop_sequences = []

    def resume(self) -> None:
        self.stop_sequence = None
        self.active_stop_sequences = self.matcher.active_sequences

    def clone(self) -> "ControlSpan":
        clone = ControlSpan(self.sequences, state=self.state)
        clone.matcher = self.matcher.clone()
        clone.stop_sequence = self.stop_sequence
        clone.active_stop_sequences = self.active_stop_sequences.copy()
        clone.text_record = self.text_record
        return clone

    def record(self, text: bytes, override_ids=(), *, done: bool = False) -> TextRecord:
        self.text_record = TextRecord(bytes(text), tuple(override_ids))
        if done:
            self.done()
        return self.text_record

    def Feed(self, data: bytes, *, state: str | None = None) -> bytes | None:
        return self.feed(data, state=state)

    def Append(self, data: bytes, *, state: str | None = None) -> bytes | None:
        return self.append(data, state=state)

    def Record(self, text: bytes, override_ids=(), *, done: bool = False) -> TextRecord:
        return self.record(text, override_ids, done=done)


def resolve_control(current: ControlSpan | None, sequences: Iterable[bytes]) -> ControlSpan:
    """Create fresh control or validate that a paused round keeps its sequences."""
    if current is None:
        return ControlSpan(sequences)
    if current.sequences != ControlSpan(sequences).sequences:
        raise ValueError("stop_sequences cannot change while resuming a round")
    return current


@dataclass(frozen=True, slots=True)
class Pause:
    reason: str = "pause"


class PausedAfterEos(Exception):
    """Remember an EOS sampled at a pause boundary for forced stream replay."""

    def __init__(self, token: int, text: bytes = b""):
        super().__init__("paused after EOS")
        self.token, self.text = token, bytes(text)


@dataclass(frozen=True, slots=True)
class Resume:
    force_end_thinking: bool = False


@dataclass(frozen=True, slots=True)
class StreamCheckpoint:
    force: bool = False
    thinking: bool = False
    text: str = ""
