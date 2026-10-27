"""Append harness and engine events to a JSONL telemetry file."""

from datetime import datetime, timezone
import json
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability


def _json_default(value):
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


class DS4RunRecord(AbstractCapability):
    """One JSON object per line for harness events and session-bus events."""

    def __init__(self, session, path):
        self.session = session
        self.path = Path(path)
        self._subscribed = False

    async def for_run(self, ctx):
        if not self._subscribed and hasattr(self.session, "on_event"):
            self.session.on_event(self._on_engine_event)
            self._subscribed = True
        return self

    def _write(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")

    def _on_engine_event(self, event):
        self._write({
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": getattr(event, "kind", type(event).__name__),
            "session_id": getattr(event, "session_id", None),
            "payload": dict(getattr(event, "payload", {})),
        })

    async def on_event(self, ctx, *, event):
        self._write({
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": type(event).__name__,
            "conversation_id": getattr(ctx, "conversation_id", None),
            "run_id": getattr(ctx, "run_id", None),
            "payload": getattr(event, "__dict__", str(event)),
        })
