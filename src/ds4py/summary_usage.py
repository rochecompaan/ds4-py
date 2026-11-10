"""One per-attempt collector for private summary output usage."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from ds4py.engine.events import SummaryToken

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from ds4py.engine.lifecycle import SessionEvent
    from ds4py.engine.session import Session


class SummaryUsageAttempt:
    """Charge observed summary events now and one legacy result remainder once."""

    def __init__(self, charge: Callable[[int], None]) -> None:
        self._charge = charge
        self._observed = 0
        self._reported = False

    def observe(self, notification: SessionEvent) -> None:
        if isinstance(notification.payload.get("event"), SummaryToken):
            self._observed += 1
            self._charge(1)

    def report_legacy(self, generated_tokens: int) -> None:
        """Charge an event-less custom owner's positive unobserved remainder once."""
        if self._reported:
            return
        self._reported = True
        remaining = generated_tokens - self._observed
        if remaining > 0:
            self._charge(remaining)


@contextmanager
def summary_usage(session: Session, charge: Callable[[int], None]) -> Generator[SummaryUsageAttempt]:
    """Observe one owner attempt until its awaited work and drain have finished."""
    attempt = SummaryUsageAttempt(charge)
    unsubscribe = session.on_event(attempt.observe, kinds={"SummaryToken"})
    try:
        yield attempt
    finally:
        unsubscribe()
