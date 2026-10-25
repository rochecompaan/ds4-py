"""Worker-thread sampling constraints and immutable argmax observations.

Hook callbacks run on the engine worker thread. They must not call back into the
engine, session, harness, blocking I/O, or cancellation APIs.
"""

from collections.abc import Iterable, Mapping
import ctypes as C
from dataclasses import dataclass, field
import threading

from ds4py._native.structs import ArgmaxContext, TokenScore
from .sampling import SamplingPolicy

_INT32_MAX = 2**31 - 1


def _token_id(value) -> int:
    if type(value) is not int:
        raise TypeError("token IDs must be integers")
    if not 0 <= value <= _INT32_MAX:
        raise ValueError(f"token ID outside native range: {value}")
    return value


def _range(value) -> tuple[int, int]:
    try:
        pair = tuple(value)
    except TypeError as exc:
        raise TypeError("a token range must contain two integer IDs") from exc
    if len(pair) != 2:
        raise ValueError("a token range must contain exactly two IDs")
    lo, hi = (_token_id(item) for item in pair)
    if lo > hi:
        raise ValueError(f"descending token range: {(lo, hi)}")
    return lo, hi


def _normalized(ranges) -> tuple[tuple[int, int], ...]:
    ordered = sorted(_range(item) for item in ranges)
    merged = []
    for lo, hi in ordered:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return tuple(merged)


@dataclass
class DisjointTokenRanges:
    """Ordered, normalized inclusive token-ID ranges."""

    ranges: tuple[tuple[int, int], ...] = ()

    def __init__(self, ranges: Iterable[tuple[int, int]] = ()):
        self.ranges = _normalized(ranges)

    def merge(self, *ranges) -> "DisjointTokenRanges":
        if len(ranges) == 1:
            try:
                possible_pair = tuple(ranges[0])
            except TypeError:
                possible_pair = ()
            if len(possible_pair) == 2 and all(type(item) is int for item in possible_pair):
                additions = (possible_pair,)
            else:
                additions = tuple(ranges[0])
        else:
            additions = ranges
        self.ranges = _normalized(self.ranges + additions)
        return self

    def contains(self, token: int) -> bool:
        return any(lo <= token <= hi for lo, hi in self.ranges)

    def ids(self) -> list[int]:
        return [token for lo, hi in self.ranges for token in range(lo, hi + 1)]

    def clear(self) -> None:
        self.ranges = ()


@dataclass
class TokenOverrides:
    """Mutable token-ID to logit overrides with deterministic dispatch order."""

    values: tuple[tuple[int, float], ...] = ()

    def __init__(self, values: Mapping[int, float] | Iterable[tuple[int, float]] = ()):
        items = values.items() if isinstance(values, Mapping) else values
        merged = {}
        for item in items:
            try:
                token, value = item
            except (TypeError, ValueError) as exc:
                raise ValueError("an override must be a (token ID, value) pair") from exc
            merged[_token_id(token)] = float(value)
        self.values = tuple(merged.items())

    def items(self) -> tuple[tuple[int, float], ...]:
        return self.values

    def ids(self) -> list[int]:
        return [token for token, _ in self.values]

    def contains(self, token: int) -> bool:
        return any(item == token for item, _ in self.values)

    def clear(self) -> None:
        self.values = ()


@dataclass(frozen=True, slots=True)
class ArgmaxRecord:
    """Python-owned copy of one native argmax record."""

    id: int
    logit: float
    logprob: float


@dataclass
class LogitHookSet:
    """Per-invocation constraints shared with native sampling callbacks.

    If ``enabled`` is omitted, hooks are active. An event can gate them without
    discarding ranges, overrides, or the last effective sampling policy.
    """

    enabled: threading.Event | None = None
    allow: DisjointTokenRanges = field(default_factory=DisjointTokenRanges)
    overrides: TokenOverrides = field(default_factory=TokenOverrides)
    repr_only_thinking: bool = True
    effective_policy: SamplingPolicy | None = None
    _argmax_records: tuple[ArgmaxRecord, ...] = field(init=False, default=(), repr=False)
    _records_buffer: object = field(init=False, default=None, repr=False)

    @property
    def active(self) -> bool:
        return self.enabled is None or self.enabled.is_set()

    @property
    def ids(self) -> list[int]:
        return self.allow.ids() + self.overrides.ids()

    @property
    def argmax_records(self) -> tuple[ArgmaxRecord, ...]:
        return self._argmax_records

    def before_sample(self, session, policy: SamplingPolicy, *, thinking: bool) -> None:
        """Install this sample's constants, candidate IDs, and argmax context."""
        self.effective_policy = policy
        eff = session.chain_consts(
            policy.temperature, policy.top_k, policy.min_p, policy.top_p,
        )
        ids = self.ids
        fn = getattr(session, "sampling_hook", None) if self.active else None
        session.set_sampling_chain(eff, ids, fn)
        count = len(ids)
        records = (TokenScore * count)() if count else C.POINTER(TokenScore)()
        self._records_buffer = records
        collect = int(
            self.active and (not self.repr_only_thinking or thinking)
        )
        session.set_argmax_context(ArgmaxContext(records, count, collect))

    def logit(self, rctx: int, token: int) -> int:
        """Continue native dispatch; aggregate edits happen in ``sampling``."""
        return 0

    def sampling(self, eff_pointer, candidates, count: int) -> None:
        """Mask ranges, apply overrides, then mask rejected candidate outputs."""
        if not eff_pointer:
            raise ValueError("native sampling supplied a null EffLogits pointer")
        logits = eff_pointer.contents.p.contents
        values = logits.p
        if not values and logits.len:
            raise ValueError("native sampling supplied a null logits buffer")
        if self.allow.ranges:
            for token in range(logits.len):
                if not self.allow.contains(token):
                    values[token] = float("-inf")
        override_ids = set()
        for token, value in self.overrides.items():
            if token >= logits.len:
                raise ValueError(f"override token {token} exceeds logits length {logits.len}")
            values[token] = value
            override_ids.add(token)
        if self.allow.ranges:
            for index in range(count):
                token = candidates[index]
                if not self.allow.contains(token) and token not in override_ids:
                    candidates[index] = -1

    def argmax(self, context_pointer) -> None:
        """Copy enabled native records; never retain native payload pointers."""
        if not context_pointer:
            raise ValueError("native argmax supplied a null context pointer")
        context = context_pointer.contents
        if not context.collect:
            self._argmax_records = ()
            return
        if context.cap < 0 or (context.cap and not context.records):
            raise ValueError("invalid native argmax records")
        self._argmax_records = tuple(
            ArgmaxRecord(record.id, float(record.logit), float(record.logprob))
            for record in context.records[:context.cap]
        )
