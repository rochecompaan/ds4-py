"""Thread-safe token history with segment boundaries."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
from threading import Lock
from types import MappingProxyType

KINDS = (
    "system",
    "user",
    "assistant_prefix",
    "assistant",
    "tool",
    "eos",
    "note",
    "reminder",
    "compaction",
)


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    kind: str
    meta: Mapping[str, object]

    def shifted(self, delta: int) -> Segment:
        """Return the same provenance in another token coordinate space."""
        return replace(self, start=self.start + delta, end=self.end + delta)


def validate_segment_coverage(segments: Sequence[Segment], token_count: int) -> None:
    """Require positive contiguous ranges covering exactly the supplied tokens."""
    expected = 0
    for segment in segments:
        if segment.start != expected or segment.end <= segment.start:
            raise ValueError("replacement segments must cover the tokens without gaps from token zero")
        expected = segment.end
    if expected != token_count:
        raise ValueError(f"replacement segments cover {expected} tokens, not the {token_count} tokens present")


@dataclass(frozen=True)
class TranscriptReplacement:
    """Validated tokens and provenance that replace a chat transcript."""

    tokens: tuple[int, ...]
    segments: tuple[Segment, ...]

    def slice(self, start_tokens: int, end_tokens: int | None = None) -> TranscriptReplacement:
        """Return a relative token slice, clipping provenance at its boundaries."""
        validate_segment_coverage(self.segments, len(self.tokens))
        end = len(self.tokens) if end_tokens is None else end_tokens
        if not 0 <= start_tokens <= end <= len(self.tokens):
            raise ValueError("slice positions are outside the transcript")
        return TranscriptReplacement(
            self.tokens[start_tokens:end],
            tuple(
                replace(segment, start=max(segment.start, start_tokens), end=min(segment.end, end)).shifted(
                    -start_tokens
                )
                for segment in self.segments
                if segment.end > start_tokens and segment.start < end and start_tokens < end
            ),
        )

    @staticmethod
    def concat(*parts: TranscriptReplacement) -> TranscriptReplacement:
        """Join relative, fully covered values without touching live history."""
        tokens: list[int] = []
        segments: list[Segment] = []
        for part in parts:
            validate_segment_coverage(part.segments, len(part.tokens))
            segments.extend(segment.shifted(len(tokens)) for segment in part.segments)
            tokens.extend(part.tokens)
        return TranscriptReplacement(tuple(tokens), tuple(segments))


class TranscriptView:
    """Live, read-only observations of session-owned history."""

    def __init__(self, transcript: Transcript) -> None:
        self._transcript = transcript

    @property
    def tokens(self) -> tuple[int, ...]:
        return self._transcript.tokens

    @property
    def segments(self) -> tuple[Segment, ...]:
        return self._transcript.segments

    @property
    def chat(self) -> bool:
        return self._transcript.chat

    def __len__(self) -> int:
        return len(self._transcript)


class Transcript:
    def __init__(self) -> None:
        self._lock = Lock()
        self._tokens: list[int] = []
        self._segments: list[Segment] = []
        self._chat = False

    @property
    def tokens(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._tokens)

    @property
    def segments(self) -> tuple[Segment, ...]:
        with self._lock:
            return tuple(self._segments)

    @property
    def chat(self) -> bool:
        with self._lock:
            return self._chat

    def set_chat(self) -> None:
        with self._lock:
            self._chat = True

    def __len__(self) -> int:
        with self._lock:
            return len(self._tokens)

    def append(self, tokens: Sequence[int], kind: str, meta: Mapping[str, object] | None = None) -> Segment:
        with self._lock:
            if kind not in KINDS:
                raise ValueError(f"unknown segment kind: {kind}")
            added = list(tokens)
            if not added:
                raise ValueError("a segment must contain tokens")
            start = len(self._tokens)
            segment = Segment(
                start,
                start + len(added),
                kind,
                MappingProxyType(dict(meta) if meta is not None else {}),
            )
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

    def annotate(self, start: int, *, kinds: tuple[str, ...], meta: Mapping[str, object]) -> None:
        """Merge provenance into existing ranges without changing token history."""
        with self._lock:
            self._segments = [
                replace(
                    segment,
                    meta=MappingProxyType({**segment.meta, **meta}),
                )
                if segment.start >= start and segment.kind in kinds
                else segment
                for segment in self._segments
            ]

    def replace(self, tokens: Sequence[int], segments: Sequence[Segment]) -> None:
        """Atomically replace all tokens and provenance with a validated chat history."""
        with self._lock:
            replacement_tokens, replacement_segments = list(tokens), list(segments)
            validate_segment_coverage(replacement_segments, len(replacement_tokens))
            self._tokens = replacement_tokens
            self._segments = replacement_segments
            self._chat = True

    def install(self, segments: Sequence[Segment]) -> None:
        """Replace provenance over tokens a native payload already established.

        The loaded KV is authoritative for the tokens; the saved ranges must
        describe them without gaps or overlap, and the transcript is a chat
        transcript from then on.
        """
        with self._lock:
            installed = list(segments)
            validate_segment_coverage(installed, len(self._tokens))
            self._segments = installed
            self._chat = True

    def reset(self) -> None:
        """Drop all tokens, segments, and chat initialization."""
        with self._lock:
            self._tokens.clear()
            self._segments.clear()
            self._chat = False

    def truncate(self, n: int) -> None:
        with self._lock:
            if not 0 <= n <= len(self._tokens):
                raise ValueError("truncate position is outside the transcript")
            del self._tokens[n:]
            while self._segments and self._segments[-1].start >= n:
                self._segments.pop()
            if self._segments and self._segments[-1].end > n:
                self._segments[-1] = replace(self._segments[-1], end=n)
            if n == 0:
                self._chat = False
