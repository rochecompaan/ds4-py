"""Pure planning of whole-prefix KV reuse."""
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence


@dataclass(frozen=True)
class Plan:
    kind: Literal["extend", "restore", "rebuild"]
    reused: int
    to_prefill: int
    fits: bool
    checkpoint: str | None


def plan_sync(tokens: Sequence[int], *, live: Sequence[int], live_valid: bool,
              checkpoints: Mapping[str, Sequence[int]], ctx: int, reserve: int = 1,
              restore_enabled: bool = True) -> Plan:
    target = tuple(tokens)
    current = tuple(live)
    fits = len(target) + max(1, reserve) <= ctx
    if live_valid and current and target[:len(current)] == current:
        return Plan("extend", len(current), len(target) - len(current), fits, None)
    reused, checkpoint = 0, None
    if restore_enabled:
        for key, values in checkpoints.items():
            prefix = tuple(values)
            if len(prefix) > reused and target[:len(prefix)] == prefix:
                reused, checkpoint = len(prefix), key
    kind = "restore" if checkpoint is not None else "rebuild"
    return Plan(kind, reused, len(target) - reused, fits, checkpoint)
