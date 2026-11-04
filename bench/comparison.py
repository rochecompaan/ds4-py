"""Pure comparisons for direct bake-off gate evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

TOP_K = 12
MAX_LOGIT_STEPS = 16
LOGIT_ABS_TOLERANCE = 1e-5
RATE_RELATIVE_TOLERANCE = 0.25
CACHED_FRACTION_ABS_TOLERANCE = 0.15


@dataclass(frozen=True)
class TopLogit:
    token: int
    logit: float


@dataclass(frozen=True)
class StepCapture:
    position: int
    top: tuple[TopLogit, ...]
    selected_token: int


@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    detail: str
    valid: bool = True
    max_abs_delta: float | None = None


@dataclass(frozen=True)
class RateEvidence:
    task_id: str
    prefill_tokens: int
    prefill_seconds: float
    decode_steps: int
    decode_seconds: float


def _invalid(detail: str) -> ComparisonResult:
    return ComparisonResult(passed=False, detail=detail, valid=False)


def compare_step(c: StepCapture, ds4py: StepCapture) -> ComparisonResult:
    """Compare one raw-logit ranking without accepting missing evidence."""
    if c.position != ds4py.position:
        return _invalid(f"position differs: C={c.position}, ds4-py={ds4py.position}")
    if len(c.top) != TOP_K or len(ds4py.top) != TOP_K:
        return _invalid(f"top-{TOP_K} capture is missing or short")
    for c_entry, ds4py_entry in zip(c.top, ds4py.top, strict=True):
        if not math.isfinite(c_entry.logit) or not math.isfinite(ds4py_entry.logit):
            return _invalid("logit capture contains a non-finite value")

    for rank, (c_entry, ds4py_entry) in enumerate(zip(c.top, ds4py.top, strict=True)):
        if c_entry.token != ds4py_entry.token:
            return ComparisonResult(
                passed=False,
                detail=f"token differs at rank {rank}: C={c_entry.token}, ds4-py={ds4py_entry.token}",
            )

    max_delta = max(abs(c_entry.logit - ds4py_entry.logit) for c_entry, ds4py_entry in zip(c.top, ds4py.top, strict=True))
    passed = max_delta < LOGIT_ABS_TOLERANCE
    return ComparisonResult(
        passed=passed,
        detail=f"max raw-logit delta {max_delta:.9g}; strict limit {LOGIT_ABS_TOLERANCE:.9g}",
        max_abs_delta=max_delta,
    )


def _rate_pair(c: RateEvidence, ds4py: RateEvidence) -> tuple[float, float, float, float] | None:
    values = (
        c.prefill_tokens,
        c.prefill_seconds,
        c.decode_steps,
        c.decode_seconds,
        ds4py.prefill_tokens,
        ds4py.prefill_seconds,
        ds4py.decode_steps,
        ds4py.decode_seconds,
    )
    if any(not math.isfinite(value) or value < 0 for value in values):
        return None
    if c.prefill_tokens == 0 or c.decode_steps == 0:
        return None
    if min(c.prefill_seconds, c.decode_seconds, ds4py.prefill_seconds, ds4py.decode_seconds) == 0:
        return None
    return (
        c.prefill_tokens / c.prefill_seconds,
        ds4py.prefill_tokens / ds4py.prefill_seconds,
        c.decode_steps / c.decode_seconds,
        ds4py.decode_steps / ds4py.decode_seconds,
    )


def compare_rate(c: RateEvidence, ds4py: RateEvidence) -> ComparisonResult:
    """Compare aggregate prefill and decode throughput for one task."""
    if c.task_id != ds4py.task_id:
        return _invalid(f"task differs: C={c.task_id}, ds4-py={ds4py.task_id}")
    rates = _rate_pair(c, ds4py)
    if rates is None:
        return _invalid("rate evidence has a non-finite, negative, or zero denominator")
    c_prefill, ds4py_prefill, c_decode, ds4py_decode = rates
    prefill_delta = abs(ds4py_prefill - c_prefill) / c_prefill
    decode_delta = abs(ds4py_decode - c_decode) / c_decode
    passed = max(prefill_delta, decode_delta) <= RATE_RELATIVE_TOLERANCE
    return ComparisonResult(
        passed=passed,
        detail=(
            f"prefill tokens/s C={c_prefill:.6g}, ds4-py={ds4py_prefill:.6g}, relative delta={prefill_delta:.6g}; "
            f"decode steps/s C={c_decode:.6g}, ds4-py={ds4py_decode:.6g}, relative delta={decode_delta:.6g}; "
            f"tolerance={RATE_RELATIVE_TOLERANCE:.6g}"
        ),
    )


def compare_cached_fraction(c: float, ds4py: float) -> ComparisonResult:
    """Compare aggregate cached-prefill fractions using an absolute tolerance."""
    if not math.isfinite(c) or not math.isfinite(ds4py):
        return _invalid("cached-prefill fraction is non-finite")
    delta = abs(ds4py - c)
    passed = delta <= CACHED_FRACTION_ABS_TOLERANCE
    return ComparisonResult(
        passed=passed,
        detail=f"cached fraction C={c:.6g}, ds4-py={ds4py:.6g}, absolute delta={delta:.6g}",
        max_abs_delta=delta,
    )
