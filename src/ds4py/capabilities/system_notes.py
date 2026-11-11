"""System-note provider for the ds4-py default agent."""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, override

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    RetryPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from typing_extensions import sentinel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext

    from ds4py.engine.lifecycle import SessionEvent
    from ds4py.engine.session import Session

DATE_PREFIX = "Current local date and time at session start: "
DATE_SUFFIX = ". Use this only when date or time matters."

HINTS_FULL = (
    "The user enabled programming hints. Occasionally explain a useful concept "
    "behind the current work: a design choice, failure mechanism, trade-off, "
    "or way to verify correctness. Base each hint on code or results you have "
    "actually examined. Write one or two sentences as a normal Markdown "
    "blockquote starting with > **Hint:**. Hints are prose, not tool calls. "
    "Put them after the relevant discovery, including between tool calls. "
    "Prefer no hint to routine narration, repetition, or generic advice. "
    "Do not invent personal experience or a learner profile. Keep working "
    "without waiting for an answer. Later system notes may enable or disable "
    "hints; follow the latest setting.\n"
)
HINTS_ON = "The user enabled programming hints. Produce them from now on.\n"
HINTS_OFF = "Programming hints are disabled. Do not produce teaching asides; continue the task normally.\n"

_UNSET = sentinel("_UNSET")
_UNKNOWN = sentinel("_UNKNOWN")


@dataclass(frozen=True)
class NoteState:
    """Prepared note state at one successfully applied binding boundary."""

    hints_enabled: bool
    date_emitted: bool
    hints_full_emitted: bool
    hints_applied: bool | _UNSET | _UNKNOWN


class NotePreview:
    """Isolated date/hints cursor; owns no Session or event subscriptions."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime | str] | None = None,
        hints: bool = False,
    ) -> None:
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.hints_enabled = hints
        self._date_emitted = False
        self._date_text: str | None = None
        self._hints_full_emitted = False
        self._hints_applied: bool | _UNSET | _UNKNOWN = _UNSET

    def pending(self, messages: Sequence[ModelMessage]) -> list[tuple[str, str]]:
        """Return notes that would be emitted without mutating provider state."""
        if not messages:
            return []
        last = messages[-1]
        if not isinstance(last, ModelRequest):
            return []
        parts = last.parts
        is_user = any(isinstance(part, UserPromptPart) for part in parts)
        is_continuation = any(isinstance(part, (ToolReturnPart, RetryPromptPart)) for part in parts)
        if not (is_user or is_continuation):
            return []

        result: list[tuple[str, str]] = []
        if is_user and not self._date_emitted:
            if self._date_text is None:
                now = self.clock()
                when = now.strftime("%Y-%m-%d %H:%M:%S %Z") if isinstance(now, datetime) else str(now)
                self._date_text = f"{DATE_PREFIX}{when}{DATE_SUFFIX}"
            result.append((self._date_text, "date"))

        if self.hints_enabled and not self._hints_full_emitted:
            result.append((HINTS_FULL, "hints_full"))
        elif self._hints_applied is not _UNSET and (
            self._hints_applied is _UNKNOWN or self._hints_applied != self.hints_enabled
        ):
            result.append(
                (
                    HINTS_ON if self.hints_enabled else HINTS_OFF,
                    "hints_on" if self.hints_enabled else "hints_off",
                )
            )
        return result

    def notes(self, messages: Sequence[ModelMessage]) -> list[tuple[str, str]]:
        """Return ``(text, subtype)`` pairs to append before the request."""
        result = self.pending(messages)
        for _, subtype in result:
            if subtype == "date":
                self._date_emitted = True
            elif subtype == "hints_full":
                self._hints_full_emitted = True
                self._hints_applied = True
            elif subtype == "hints_on":
                self._hints_applied = True
            elif subtype == "hints_off":
                self._hints_applied = False
        return result

    def checkpoint(self) -> NoteState:
        return NoteState(self.hints_enabled, self._date_emitted, self._hints_full_emitted, self._hints_applied)

    def apply_preview(self, state: NoteState) -> None:
        """Assign a checkpoint without callbacks or clock reads."""
        self.hints_enabled = state.hints_enabled
        self._date_emitted = state.date_emitted
        self._hints_full_emitted = state.hints_full_emitted
        self._hints_applied = state.hints_applied

    def _segment_subtype(self, segment: object) -> object:
        kind = getattr(segment, "kind", segment)
        meta = getattr(segment, "meta", {}) or {}
        subtype = meta.get("subtype") if hasattr(meta, "get") else None
        if subtype:
            return subtype
        if isinstance(kind, str) and kind.startswith("system_note:"):
            return kind.split(":", 1)[1]
        return kind

    def sync_from_segments(self, segments: Sequence[object]) -> None:
        """Reset emitted-note state from the transcript without changing settings."""
        self._date_emitted = False
        self._hints_full_emitted = False
        self._hints_applied = _UNSET
        for segment in segments:
            subtype = self._segment_subtype(segment)
            if subtype == "date":
                self._date_emitted = True
            elif subtype == "hints_full":
                self._hints_full_emitted = True
                self._hints_applied = True
            elif subtype == "hints_on":
                self._hints_applied = True
            elif subtype == "hints_off":
                self._hints_applied = False

    def resync_from_segments(self, segments: Sequence[object]) -> None:
        """Resync transcript state without erasing a pending compaction reset."""
        compaction_invalidated = self._hints_applied is _UNKNOWN
        self.sync_from_segments(segments)
        if compaction_invalidated:
            self._hints_applied = _UNKNOWN
            self._hints_full_emitted = False

    def restore_from_segments(self, segments: Sequence[object]) -> None:
        """Restore date/hints state from saved transcript segments."""
        self.sync_from_segments(segments)
        # A saved history has already started, even if compaction removed its
        # original date note. Loading it must not invent a new session start.
        self._date_emitted = bool(segments)
        for segment in segments:
            subtype = self._segment_subtype(segment)
            if subtype in ("hints_full", "hints_on"):
                self.hints_enabled = True
            elif subtype == "hints_off":
                self.hints_enabled = False


class DS4SystemNotes(NotePreview, AbstractCapability):
    """Live note provider with explicit preview publication and subscriptions."""

    def __init__(
        self,
        session: Session,
        *,
        clock: Callable[[], datetime | str] | None = None,
        hints: bool = False,
    ) -> None:
        super().__init__(clock=clock, hints=hints)
        self.session = session
        self._compaction_subscribed = False

    @override
    def pending(self, messages: Sequence[ModelMessage]) -> list[tuple[str, str]]:
        """Observe through a fresh cursor, without caching on the live provider."""
        return self.preview().pending(messages)

    def preview(self) -> NotePreview:
        cursor = NotePreview(clock=self.clock)
        cursor.apply_preview(self.checkpoint())
        return cursor

    @override
    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if not self._compaction_subscribed:
            if hasattr(self.session, "on_event"):
                self.session.on_event(self._on_compacted, kinds="compacted")
            self._compaction_subscribed = True
        return request_context

    def _on_compacted(self, event: SessionEvent) -> None:
        if self._hints_applied is not _UNSET:
            self._hints_applied = _UNKNOWN
        self._hints_full_emitted = False
