from __future__ import annotations

import math
from dataclasses import replace

from bench.comparison import RateEvidence, StepCapture, TopLogit, compare_cached_fraction, compare_rate, compare_step


def _capture() -> StepCapture:
    return StepCapture(
        position=42,
        top=tuple(TopLogit(token=index, logit=-float(index)) for index in range(12)),
        selected_token=0,
    )


def _rate() -> RateEvidence:
    return RateEvidence(
        task_id="scale-generator", prefill_tokens=100, prefill_seconds=2.0, decode_steps=40, decode_seconds=4.0
    )


def test_topk_requires_identical_ordered_ids_and_positions() -> None:
    capture = _capture()
    reordered = replace(capture, top=(capture.top[1], capture.top[0], *capture.top[2:]))

    assert compare_step(capture, capture).passed
    ranking = compare_step(capture, reordered)
    assert not ranking.passed
    assert ranking.valid
    position = compare_step(capture, replace(capture, position=43))
    assert not position.passed
    assert not position.valid


def test_topk_accepts_small_raw_logit_delta_and_rejects_tolerance_boundary() -> None:
    capture = _capture()
    close = replace(capture, top=(replace(capture.top[0], logit=5e-8), *capture.top[1:]))
    boundary = replace(capture, top=(replace(capture.top[0], logit=1e-5), *capture.top[1:]))

    accepted = compare_step(capture, close)
    assert accepted.passed
    assert accepted.max_abs_delta == 5e-8
    rejected = compare_step(capture, boundary)
    assert not rejected.passed
    assert rejected.valid
    assert rejected.max_abs_delta == 1e-5


def test_topk_rejects_missing_or_short_capture() -> None:
    capture = _capture()

    for top in ((), capture.top[:11]):
        result = compare_step(capture, replace(capture, top=top))
        assert not result.passed
        assert not result.valid


def test_topk_rejects_non_finite_logits() -> None:
    capture = _capture()
    non_finite = replace(capture, top=(replace(capture.top[0], logit=math.nan), *capture.top[1:]))

    result = compare_step(capture, non_finite)
    assert not result.passed
    assert not result.valid


def test_rate_comparison_uses_aggregated_tokens_and_seconds() -> None:
    c = _rate()
    ds4py = replace(c, prefill_tokens=110, decode_steps=44)

    result = compare_rate(c, ds4py)
    assert result.passed
    assert result.valid


def test_rate_comparison_rejects_zero_denominator_and_relative_difference_above_tolerance() -> None:
    c = _rate()
    too_slow = replace(c, prefill_seconds=2.70)
    zero = replace(c, prefill_seconds=0.0)
    wrong_task = replace(c, task_id="zipper")

    slow_result = compare_rate(c, too_slow)
    assert not slow_result.passed
    assert slow_result.valid
    assert not compare_rate(c, zero).valid
    assert not compare_rate(c, wrong_task).valid


def test_rate_comparison_rejects_non_finite_values() -> None:
    c = _rate()

    assert not compare_rate(c, replace(c, decode_seconds=math.inf)).valid


def test_cached_fraction_uses_absolute_tolerance() -> None:
    assert compare_cached_fraction(0.25, 0.10).passed
    rejected = compare_cached_fraction(0.80, 0.64)
    assert not rejected.passed
    assert rejected.valid
    assert not compare_cached_fraction(0.75, math.nan).valid


def test_invalid_provenance_is_distinct_from_numeric_mismatch() -> None:
    capture = _capture()
    numeric = compare_step(capture, replace(capture, top=(replace(capture.top[0], logit=0.001), *capture.top[1:])))
    invalid = compare_step(capture, replace(capture, position=100))

    assert not numeric.passed and numeric.valid
    assert not invalid.passed and not invalid.valid
