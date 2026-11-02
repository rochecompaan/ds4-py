"""Bounded UTF-8 stop matching and resumable control-span state."""

from collections.abc import Iterable


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
    """KMP matchers with fixed memory and independent state."""

    def __init__(self, sequences: Iterable[bytes] = ()) -> None:
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


class ControlSpan:
    """Resumable stop matching over sampled control-region bytes."""

    def __init__(self, sequences: Iterable[bytes] = ()) -> None:
        self.matcher = StopSequenceMatcher(sequences)
        self.stop_sequence: bytes | None = None
        self.active_stop_sequences: list[bytes] = []

    @property
    def sequences(self) -> tuple[bytes, ...]:
        return self.matcher.sequences

    def feed(self, data: bytes) -> bytes | None:
        if self.stop_sequence is not None:
            return None
        matched = self.matcher.feed(data)
        if matched is not None:
            self.stop(matched)
        else:
            self.active_stop_sequences = self.matcher.active_sequences
        return matched

    def stop(self, sequence: bytes) -> None:
        self.stop_sequence = sequence
        self.active_stop_sequences = []

    def resume(self) -> None:
        self.stop_sequence = None
        self.active_stop_sequences = self.matcher.active_sequences


def resolve_control(current: ControlSpan | None, sequences: Iterable[bytes]) -> ControlSpan:
    """Create fresh control or validate that a paused round keeps its sequences."""
    if current is None:
        return ControlSpan(sequences)
    if current.sequences != ControlSpan(sequences).sequences:
        raise ValueError("stop_sequences cannot change while resuming a round")
    return current
