"""One controller for a response's reserve decision and recovery accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ds4py.binding import RESPONSE_RESERVE
from ds4py.capabilities.context import ContextRequest
from ds4py.engine.events import ReserveReached, Synced
from ds4py.errors import ContextExhausted, EngineError
from ds4py.summary_usage import summary_usage

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.tools import RunContext

    from ds4py.binding import MessageBinder
    from ds4py.capabilities.context import ContextStrategy
    from ds4py.engine.lifecycle import SessionEvent
    from ds4py.engine.session import Session


class ReserveSettlement:
    """Own reserve decisions, owner recovery, and recovery usage for one response.

    The response keeps event translation, round draining, reconciliation, and
    finalization. This controller owns the reserve window, request/history
    selection, owner recovery, the exact post-recovery fit check, and the
    resume/fail decision, plus the summary and replacement-sync usage that
    finalization adds to the round's own counters.

    ``reconciled`` and ``observe_tail`` are the only response callbacks: the
    response owns reconciliation state and the trailing open-tail structure.
    """

    def __init__(  # noqa: PLR0913 - The collaborators are the response's reserve surface.
        self,
        *,
        session: Session,
        binder: MessageBinder,
        owner: ContextStrategy | None,
        messages: list[ModelMessage],
        run_context: RunContext[object] | None,
        reconciled: Callable[[], bool],
        observe_tail: Callable[[], None],
    ) -> None:
        self._session = session
        self._binder = binder
        self._owner = owner
        self._messages = messages
        self._run_context = run_context
        self._reconciled = reconciled
        self._tail_observer = observe_tail
        self._prefilled = 0
        self._cached = 0
        self._summary_tokens = 0

    @property
    def prefilled(self) -> int:
        """Tokens this response's replacement syncs prefilled."""
        return self._prefilled

    @property
    def cached(self) -> int:
        """Tokens this response's replacement syncs reused from cache."""
        return self._cached

    @property
    def summary_tokens(self) -> int:
        """Private summary output this response's owner generated."""
        return self._summary_tokens

    async def recover(self, event: ReserveReached) -> None:
        """Resume only when the recovered request fits the configured context."""
        try:
            messages = await self._recover(event)
            if self._reconciled():
                return
            projected = self._binder.project_request(messages, bound=True)
            fits = self._binder.reserve_overflow(projected) <= 0
        except BaseException:
            self.settle(event)
            raise
        if not fits:
            self.settle(event)
            raise ContextExhausted(self._exhausted_message(event.needed))
        event.resume(RESPONSE_RESERVE)

    def settle(self, event: ReserveReached) -> None:
        """Fail an undecided pause unless terminal cleanup already settled it."""
        if not self._reconciled():
            event.fail()

    async def _recover(self, event: ReserveReached) -> list[ModelMessage]:
        """Run the selected owner and account for its work in one unit."""
        owner = self._owner
        if owner is None:
            raise ContextExhausted(self._exhausted_message(event.needed))
        # Replacement syncs are ordered native work, but are not yielded by the
        # suspended round. Observe only this owner's work to avoid double counts.
        unsubscribe = self._session.on_event(self._observe_sync, kinds={"Synced"})
        try:
            with summary_usage(self._session, self._record_summary) as attempt:
                try:
                    strict = owner.reserve_operations
                    with self._session.reserve_window(event, strict=strict) as operations:
                        messages = owner.reserve_messages(self._messages, self._run_context)
                        room = await owner.make_room(
                            messages,
                            ContextRequest(event.needed, "reserve", operations if strict else None),
                        )
                except BaseException as error:
                    # Commit-then-raise owners can still move the open tail.
                    self._observe_moved_tail(error)
                    raise
                if self._observe_moved_tail():
                    attempt.report_legacy(room.generated_tokens)
                return messages
        finally:
            unsubscribe()

    def _observe_sync(self, notification: SessionEvent) -> None:
        sync = notification.payload.get("event")
        if isinstance(sync, Synced):
            self._prefilled += sync.prefilled
            self._cached += sync.plan.reused

    def _record_summary(self, tokens: int) -> None:
        self._summary_tokens += tokens

    def _observe_moved_tail(self, primary: BaseException | None = None) -> bool:
        """Observe a moved open tail without replacing an owner failure."""
        if self._reconciled():
            return False
        try:
            self._tail_observer()
        except EngineError:
            if primary is None:
                raise
        return True

    @staticmethod
    def _exhausted_message(needed: int | None) -> str:
        return f"context reserve reached with {needed} token(s) needed and no room was made"
