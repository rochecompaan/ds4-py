"""Lazy, reusable access to the live logits and committed decode metadata."""

from array import array
from collections.abc import Sequence


class LogitsView:
    """A worker-thread view passed to each registered pre-sample hook.

    The float buffer is allocated once per generation invocation. Native logits
    are copied only when logits() is called; the returned memory view remains
    backed by that same buffer on later calls.

    Hooks must not call engine, session, harness, cancellation, or blocking-I/O
    APIs. They run synchronously on the native worker thread, so those calls can
    deadlock the worker. Use only this view and hook-owned state.
    """

    def __init__(self, session, vocab_size: int):
        self._session = session
        self._vocab_size = vocab_size
        self._buffer = array("f", [0.0]) * vocab_size
        self._position = 0
        self._in_think = False
        self._think_tokens = 0
        self._parser_state = None
        self._sampled_tokens: tuple[int, ...] = ()

    @property
    def position(self) -> int:
        """Native position before this sample."""
        return self._position

    @property
    def in_think(self) -> bool:
        """Parser/control thinking state, not the display decoder state."""
        return self._in_think

    @property
    def think_tokens(self) -> int:
        """Committed tokens counted in the current think block."""
        return self._think_tokens

    @property
    def parser_state(self):
        """Current parser phase, or None for raw rounds."""
        return self._parser_state

    @property
    def sampled_tokens(self) -> tuple[int, ...]:
        """Immutable history of committed sampled IDs for this assistant round."""
        return self._sampled_tokens

    def _update(
        self, *, position, in_think, think_tokens, parser_state, sampled_tokens
    ):
        self._position = position
        self._in_think = in_think
        self._think_tokens = think_tokens
        self._parser_state = parser_state
        self._sampled_tokens = tuple(sampled_tokens)

    def logits(self):
        """Copy and return a view of this invocation's reusable float buffer."""
        self._session.copy_logits(self._buffer)
        return memoryview(self._buffer)

    def set_logits(self, values: Sequence[float]) -> None:
        """Replace native logits after checking the exact vocabulary length."""
        if len(values) != self._vocab_size:
            raise ValueError(
                f"logits buffer must equal the vocabulary size ({self._vocab_size})"
            )
        self._session.set_logits(values)

    def bias(self, token_id: int, delta: float) -> None:
        """Add delta to one token's logit through the copy/set facade methods."""
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError("token ID must be an integer")
        if not 0 <= token_id < self._vocab_size:
            raise ValueError(f"token ID must be in [0, {self._vocab_size})")
        self.logits()
        self._buffer[token_id] += delta
        self.set_logits(self._buffer)
