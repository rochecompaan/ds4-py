"""Thread-safe token history with segment boundaries."""
from dataclasses import dataclass, replace
from threading import Lock
from types import MappingProxyType
from typing import Mapping, Sequence

KINDS = ("system", "user", "assistant_prefix", "assistant", "tool", "eos", "note", "reminder", "compaction")


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    kind: str
    meta: Mapping[str, object]


class Transcript:
    def __init__(self):
        self._lock = Lock()
        self._tokens: list[int] = []
        self._segments: list[Segment] = []

    @property
    def tokens(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._tokens)

    @property
    def segments(self) -> tuple[Segment, ...]:
        with self._lock:
            return tuple(self._segments)

    def __len__(self) -> int:
        with self._lock:
            return len(self._tokens)

    def append(self, tokens: Sequence[int], kind: str,
               meta: Mapping | None = None) -> Segment:
        with self._lock:
            if kind not in KINDS:
                raise ValueError(f"unknown segment kind: {kind}")
            added = list(tokens)
            if not added:
                raise ValueError("a segment must contain tokens")
            start = len(self._tokens)
            segment = Segment(start, start + len(added), kind,
                              MappingProxyType(dict(meta) if meta is not None else {}))
            self._tokens.extend(added)
            self._segments.append(segment)
            return segment

    def extend(self, tokens: Sequence[int]) -> None:
        with self._lock:
            added = list(tokens)
            if not added:
                return
            if not self._segments:
                raise ValueError("cannot extend without a segment")
            self._tokens.extend(added)
            self._segments[-1] = replace(self._segments[-1], end=len(self._tokens))

    def truncate(self, n: int) -> None:
        with self._lock:
            if not 0 <= n <= len(self._tokens):
                raise ValueError("truncate position is outside the transcript")
            del self._tokens[n:]
            while self._segments and self._segments[-1].start >= n:
                self._segments.pop()
            if self._segments and self._segments[-1].end > n:
                self._segments[-1] = replace(self._segments[-1], end=n)
