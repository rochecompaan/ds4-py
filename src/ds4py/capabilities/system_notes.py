"""System-note provider for the ds4-py default agent."""

from datetime import datetime

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelRequest,
    RetryPromptPart,
    ToolReturnPart,
    UserPromptPart,
)

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
HINTS_OFF = (
    "Programming hints are disabled. Do not produce teaching asides; "
    "continue the task normally.\n"
)

_UNSET = object()
_UNKNOWN = object()


class DS4SystemNotes(AbstractCapability):
    """Tracks date and hints notes and asks the binder to append them."""

    def __init__(self, session, *, clock=None, hints=False):
        self.session = session
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.hints_enabled = hints
        self._date_emitted = False
        self._hints_full_emitted = False
        self._hints_applied = _UNSET
        self._compaction_subscribed = False
        self._compaction_subject = None

    def pending(self, messages):
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

        result = []
        if is_user and not self._date_emitted:
            now = self.clock()
            when = now.strftime("%Y-%m-%d %H:%M:%S %Z") if isinstance(now, datetime) else str(now)
            result.append((f"{DATE_PREFIX}{when}{DATE_SUFFIX}", "date"))

        if self.hints_enabled and not self._hints_full_emitted:
            result.append((HINTS_FULL, "hints_full"))
        elif self._hints_applied is not _UNSET and (
            self._hints_applied is _UNKNOWN
            or self._hints_applied != self.hints_enabled
        ):
            result.append((HINTS_ON if self.hints_enabled else HINTS_OFF,
                           "hints_on" if self.hints_enabled else "hints_off"))
        return result

    def notes(self, messages):
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

    async def before_model_request(self, ctx, request_context):
        if not self._compaction_subscribed:
            subject = getattr(ctx, "compaction_subject", None)
            if subject is None:
                subject = getattr(request_context, "compaction_subject", None)
            if subject is not None:
                self._compaction_subject = subject
            if hasattr(self.session, "on_event"):
                self.session.on_event(self._on_compacted, kinds="compacted")
            self._compaction_subscribed = True
        return request_context

    def _on_compacted(self, event):
        if self._hints_applied is not _UNSET:
            self._hints_applied = _UNKNOWN
        self._hints_full_emitted = False

    def _segment_subtype(self, segment):
        kind = getattr(segment, "kind", segment)
        meta = getattr(segment, "meta", {}) or {}
        subtype = meta.get("subtype") if hasattr(meta, "get") else None
        if subtype:
            return subtype
        if isinstance(kind, str) and kind.startswith("system_note:"):
            return kind.split(":", 1)[1]
        return kind

    def sync_from_segments(self, segments):
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

    def resync_from_segments(self, segments):
        """Resync transcript state without erasing a pending compaction reset."""
        compaction_invalidated = self._hints_applied is _UNKNOWN
        self.sync_from_segments(segments)
        if compaction_invalidated:
            self._hints_applied = _UNKNOWN
            self._hints_full_emitted = False

    def restore_from_segments(self, segments):
        """Restore date/hints state from saved transcript segments."""
        self.sync_from_segments(segments)
        for segment in segments:
            subtype = self._segment_subtype(segment)
            if subtype in ("hints_full", "hints_on"):
                self.hints_enabled = True
            elif subtype == "hints_off":
                self.hints_enabled = False
