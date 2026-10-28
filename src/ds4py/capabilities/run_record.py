"""Append harness and engine events to a JSONL telemetry file."""

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability


def _json_safe(value):
    """Return ``value`` as JSON types, encoding byte payloads as explicit hex."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _event_payload(event) -> dict:
    """Flatten one harness event without deep-copying it."""
    attributes = getattr(event, "__dict__", None)
    if not attributes:
        return {"repr": str(event)}
    return {str(key): _json_safe(item) for key, item in attributes.items()}


class DS4RunRecord(AbstractCapability):
    """One JSON object per line for harness events and session-bus events.

    Every line carries the timestamp, kind, session id, run id and a JSON-safe
    payload, and each line is flushed by its own open/append/close. A failed
    write is reported through ``failure`` so a caller can surface it: the
    session-bus path swallows it (a listener must not break the engine) while
    the harness hook raises it (a run must not report success without a record).
    """

    def __init__(self, session, path):
        self.session = session
        self.path = Path(path)
        self.failure = None
        self._run_id = None
        self._subscribed = False
        if hasattr(session, "on_event"):
            session.on_event(self._on_engine_event)
            self._subscribed = True

    def relocate(self, path):
        """Point later lines at another session directory."""
        self.path = Path(path)

    async def for_run(self, ctx):
        if not self._subscribed and hasattr(self.session, "on_event"):
            self.session.on_event(self._on_engine_event)
            self._subscribed = True
        self._run_id = getattr(ctx, "run_id", None)
        return self

    def _write(self, record: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as error:
            self.failure = error
            raise

    def _line(self, kind: str, payload: dict) -> dict:
        values = _json_safe(dict(payload))
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "session_id": getattr(self.session, "id", None),
            "run_id": values.pop("run_id", self._run_id),
            "payload": values,
        }

    def _on_engine_event(self, event):
        kind = getattr(event, "kind", type(event).__name__)
        try:
            self._write(self._line(kind, dict(getattr(event, "payload", {}) or {})))
        except OSError:
            # Telemetry must never break the engine that is publishing.
            pass

    async def on_event(self, ctx, *, event):
        line = self._line(type(event).__name__, _event_payload(event))
        line["conversation_id"] = getattr(ctx, "conversation_id", None)
        run_id = getattr(ctx, "run_id", None)
        if run_id is not None:
            line["run_id"] = run_id
        self._write(line)
