"""One per-attempt summary collector and its legacy shortfall adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative
from ds4py.engine.events import SummaryToken
from ds4py.summary_usage import summary_usage

if TYPE_CHECKING:
    from ds4py.engine.session import Session


class Charges:
    """Record every private summary token charged to one attempt sink."""

    def __init__(self) -> None:
        self.total = 0

    def __call__(self, tokens: int) -> None:
        self.total += tokens


def emit_summary(session: Session, count: int, *, start: int = 0) -> None:
    for index in range(count):
        session.emit("SummaryToken", event=SummaryToken(start + index, b"a"))


def test_summary_usage_keeps_observed_output_without_result() -> None:
    # Break caught: an owner that cancels/errors before returning RoomResult
    # loses emitted usage, or the collector keeps charging after the attempt.
    with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=FakeNative()) as engine:
        session = engine.session()
        charged = Charges()

        def cancels_before_result() -> None:
            with summary_usage(session, charged):
                emit_summary(session, 13)
                assert charged.total == 13
                raise RuntimeError("owner cancelled before result")

        with pytest.raises(RuntimeError, match="owner cancelled before result"):
            cancels_before_result()
        emit_summary(session, 4, start=13)
        assert charged.total == 13


@pytest.mark.parametrize(
    ("emitted", "generated", "expected"),
    [(13, 13, 13), (13, 17, 17), (0, 7, 7)],
    ids=["equal", "shortfall", "report-only"],
)
def test_legacy_report_shortfall_is_charged_once(emitted: int, generated: int, expected: int) -> None:
    # Break caught: report_legacy double-charges observed events or keeps
    # charging when a custom owner reports more than once.
    with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=FakeNative()) as engine:
        session = engine.session()
        charged = Charges()
        with summary_usage(session, charged) as attempt:
            emit_summary(session, emitted)
            attempt.report_legacy(generated)
            attempt.report_legacy(generated + 20)
        assert charged.total == expected
