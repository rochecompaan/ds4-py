"""Strict reader for append-only ds4-py prefill trace records."""

import json
from pathlib import Path

from .replay_types import _THINK_NAMES, ReplayError, ReplayRound, ReplayTrace
from typing import Never


_FIELDS = (
    "session_id", "run_id", "step_index", "round_index", "prompt", "think",
    "cached", "prefill_suffix", "sampled",
)


def read_ds4py(path: Path, text: str) -> ReplayTrace:
    def error(line: int, message: str) -> Never:
        raise ReplayError(f"{path}:{line}: {message}")

    rounds = []
    last_steps = {}
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            error(line_number, f"invalid JSON: {exc.msg}")
        if not isinstance(value, dict):
            error(line_number, "record must be a JSON object")
        missing = [name for name in _FIELDS if name not in value]
        if missing:
            error(line_number, f"missing field {missing[0]}")
        for name in ("session_id", "run_id"):
            if not isinstance(value[name], str) or not value[name]:
                error(line_number, f"{name} must be a nonempty string")
        for name in ("step_index", "round_index", "cached"):
            if isinstance(value[name], bool) or not isinstance(value[name], int) or value[name] < 0:
                error(line_number, f"{name} must be a nonnegative integer")
        for name in ("prompt", "prefill_suffix", "sampled"):
            tokens = value[name]
            if not isinstance(tokens, list) or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in tokens
            ):
                error(line_number, f"{name} must be a list of nonnegative token IDs")
        think = value["think"]
        if isinstance(think, bool):
            error(line_number, "think must be a mode integer or name")
        if isinstance(think, str):
            if think in _THINK_NAMES:
                think = _THINK_NAMES[think]
            elif think.isdecimal() and 1 <= int(think) <= 100:
                think = 1000 + int(think)
            else:
                error(line_number, "unsupported think name")
        if not isinstance(think, int) or think < 0:
            error(line_number, "think must be a nonnegative mode integer or name")
        cached = value["cached"]
        prompt = tuple(value["prompt"])
        suffix = tuple(value["prefill_suffix"])
        if cached + len(suffix) != len(prompt):
            error(line_number, "cached + prefill_suffix length must equal prompt length")
        if prompt[cached:] != suffix:
            error(line_number, "suffix does not match prompt")
        if value["round_index"] != len(rounds):
            error(line_number, "non-contiguous round_index")
        run_id = value["run_id"]
        step = value["step_index"]
        if run_id in last_steps and step <= last_steps[run_id]:
            error(line_number, "step_index must increase within a run")
        last_steps[run_id] = step
        rounds.append(ReplayRound(
            number=value["round_index"], prompt=prompt, sampled=tuple(value["sampled"]),
            think=think, cached=cached, context_limited=False, prefix_only=False,
        ))
    if not rounds:
        error(1, "no replay rounds")
    return ReplayTrace(None, None, tuple(rounds))
