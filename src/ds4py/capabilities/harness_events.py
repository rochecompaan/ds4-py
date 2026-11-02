"""Publish harness events on the session bus, independently of telemetry."""

from typing import TYPE_CHECKING, override

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandStartedEvent

if TYPE_CHECKING:
    from pydantic_ai.messages import AgentStreamEvent
    from pydantic_ai.tools import RunContext

    from ds4py.engine.session import Session


class HarnessEventBridge(AbstractCapability):
    """Publish harness and shell-service events without owning their recording."""

    def __init__(self, session: Session) -> None:
        self._session = session

    @override
    async def on_event(self, ctx: RunContext[object], *, event: AgentStreamEvent) -> None:
        emit = getattr(self._session, "emit", None)
        if not callable(emit):
            return
        run_id = getattr(ctx, "run_id", None)
        emit(
            "agent_event",
            event=event,
            conversation_id=getattr(ctx, "conversation_id", None),
            run_id=run_id,
        )
        if isinstance(event, CommandStartedEvent):
            emit(
                "service_started",
                service="shell",
                pid=event.pid,
                status="running",
                run_id=run_id,
            )
        elif isinstance(event, CommandFinishedEvent):
            running = event.exit_code is None
            emit(
                "service_status" if running else "service_stopped",
                service="shell",
                pid=event.pid,
                status="running" if running else "stopped",
                exit_code=event.exit_code,
                run_id=run_id,
            )
