"""Session settings, status, commands, and lifecycle for RollingWindow."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ds4py.context_paging.settings import resolve_settings

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic_ai_harness.step_persistence import FileStepStore

    from ds4py.capabilities.rolling_window import RollingWindow
    from ds4py.context_paging.settings import RollingWindowSettings
    from ds4py.engine.lifecycle import SessionEvent


class RollingWindowSession:
    """Keep process-lifetime warnings separate from session-scoped policy state."""

    def __init__(self, owner: RollingWindow) -> None:
        self.owner = owner
        self.session = owner.binder.session
        self.warning_pairs: set[tuple[int, int]] = set()
        subscribe = getattr(self.session, "on_event", None)
        if callable(subscribe):
            subscribe(
                self._receive,
                kinds={
                    "session_end",
                    "session_start",
                    "forked",
                    "before_compact",
                    "compacted",
                    "model_changed",
                },
            )

    def initialize(self, *, project_dir: Path, store_root: Path, project_trusted: bool) -> None:
        """Resolve all settings before applying any enabled state."""
        owner = self.owner
        try:
            settings = resolve_settings(
                project_dir=project_dir,
                store_root=store_root,
                project_trusted=project_trusted,
                enabled=owner._enabled_override,
                token_budget=owner._budget_override,
                trim_to=owner._trim_override,
            )
        except (OSError, ValueError, TypeError) as error:
            self.session.emit("status", key="context-paging", text="paging off")
            self.session.emit("context_paging_error", message=str(error))
            raise
        owner._base_settings = settings
        owner._state.settings = settings
        self.refresh()

    def refresh(self) -> None:
        """Replay current diagnostics after a frontend subscribes."""
        self.publish_status()
        self.warn_for_trim(self.owner._base_settings, replay=True)

    def publish_status(self) -> None:
        self.session.emit("status", key="context-paging", text="paging on" if self.owner.enabled else "paging off")

    def warn_for_trim(self, settings: RollingWindowSettings, *, replay: bool = False) -> None:
        pair = (settings.token_budget, settings.trim_to)
        if not settings.trim_to_explicit or settings.trim_to < settings.token_budget:
            return
        if pair in self.warning_pairs and not replay:
            return
        self.warning_pairs.add(pair)
        self.session.emit(
            "context_paging_warning",
            message=(
                f"The explicit trim target ({settings.trim_to}) is at or above the token budget "
                f"({settings.token_budget}); stable cuts will trim to the effective budget."
            ),
        )

    def command(self, action: str | None) -> str:
        """Apply an idle on/off choice without resizing native state."""
        if action not in (None, "status", "on", "off"):
            raise ValueError("usage: /context-paging [on|off|status]")
        if action in ("on", "off"):
            enabled = action == "on"
            if self.owner.enabled != enabled:
                state = self.owner._state
                state.settings = replace(state.settings, enabled=enabled)
                self.owner.reset_cut()
                self.owner.reset_accounting()
                self.publish_status()
        return "paging on" if self.owner.enabled else "paging off"

    async def statistics(self, *, store: FileStepStore, session_dir: Path) -> str:
        """Provide the storage seam used by the read-only statistics report."""
        from ds4py.context_paging.stats import collect_statistics  # noqa: PLC0415 - Statistics are an optional command path.

        report = await collect_statistics(store=store, session_dir=session_dir, current=self.owner.statistics_state())
        return report.render()

    def _receive(self, event: SessionEvent) -> None:
        if event.kind == "session_end":
            self.session.emit("status", key="context-paging", text=None)
            return
        if event.kind == "session_start":
            if event.payload.get("status") != "error":
                self.publish_status()
                self.warn_for_trim(self.owner._base_settings)
            return
        if event.kind == "before_compact":
            if event.payload.get("reason") == "manual":
                self.owner.reset_accounting()
            return
        if event.kind == "compacted":
            if event.payload.get("reason") == "manual":
                self.owner.reset_cut()
            return
        if event.kind == "model_changed":
            self.owner.reset_cut()
            self.owner.reset_accounting()
            return
        if event.kind == "forked":
            self.owner.reset_after_fork()
