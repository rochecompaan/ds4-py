"""Append-only token traces captured from committed facade events."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, TypedDict, cast

from ds4py.engine.ds4py_replay import read_ds4py
from ds4py.engine.events import Stopped, Synced, Token
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.session import Session


@dataclass(frozen=True)
class PrefillTraceRecord:
    session_id: str
    run_id: str
    step_index: int
    round_index: int
    prompt: tuple[int, ...]
    think: int
    cached: int
    prefill_suffix: tuple[int, ...]
    sampled: tuple[int, ...]


class _CurrentRound(TypedDict):
    prompt: tuple[int, ...]
    cached: int
    prefill_suffix: tuple[int, ...]
    sampled: list[int]


class PrefillTraceWriter:
    """Subscribe before a run and append one record at each round terminal.

    ``step_index`` counts model rounds within one harness run. ``round_index``
    counts all rounds written by this subscriber across appended runs.
    """

    def __init__(self, session: Session, path: str | Path, *, think: int) -> None:
        self.session = session
        self.path = Path(path)
        self.think = int(think)
        self._needs_separator = False
        self.failure: OSError | None = None
        if self.path.exists() and self.path.stat().st_size:
            text = self.path.read_text(encoding="utf-8")
            self.round_index = len(read_ds4py(self.path, text).rounds)
            self._needs_separator = not text.endswith(("\n", "\r"))
        else:
            self.round_index = 0
        self._step_index = 0
        self._run_id: str | None = None
        self._active_run = False
        self._current: _CurrentRound | None = None
        self._unsubscribe: Callable[[], None] | None = session.on_event(self._on_event)

    def __enter__(self) -> "PrefillTraceWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def append(self, record: PrefillTraceRecord) -> None:
        line = json.dumps(asdict(record), separators=(",", ":"), ensure_ascii=False)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                if self._needs_separator:
                    stream.write("\n")
                    self._needs_separator = False
                stream.write(line + "\n")
        except OSError as error:
            self.failure = error
            self.session.emit("trace_error", path=str(self.path), error=error)
            raise

    def _finish_round(self, session_id: str) -> None:
        if self._current is None:
            return
        current = self._current
        self._current = None
        self.append(
            PrefillTraceRecord(
                session_id=session_id,
                run_id=cast(str, self._run_id),
                step_index=self._step_index,
                round_index=self.round_index,
                prompt=current["prompt"],
                think=self.think,
                cached=current["cached"],
                prefill_suffix=current["prefill_suffix"],
                sampled=tuple(current["sampled"]),
            )
        )
        self._step_index += 1
        self.round_index += 1

    def _on_event(self, event: SessionEvent) -> None:
        if event.kind == "run_start":
            self._finish_round(event.session_id)
            self._run_id = event.payload["run_id"]
            self._active_run = True
            self._step_index = 0
            return
        if event.kind == "run_end":
            self._finish_round(event.session_id)
            self._active_run = False
            return
        if event.kind == "model_step":
            if self._active_run and event.payload["run_id"] == self._run_id:
                self._step_index = event.payload["step_index"]
            return
        if not self._active_run:
            return
        engine_event = event.payload.get("event")
        if isinstance(engine_event, Synced):
            if self._current is not None:
                # A reserve resume syncs the same model round again. Its prompt
                # contains tokens already committed to this record's sample.
                return
            prompt = tuple(self.session.transcript.tokens[: engine_event.pos])
            cached = engine_event.plan.reused
            self._current = {
                "prompt": prompt,
                "cached": cached,
                "prefill_suffix": prompt[cached:],
                "sampled": [],
            }
        elif isinstance(engine_event, Token) and self._current is not None:
            self._current["sampled"].append(engine_event.id)
        elif isinstance(engine_event, Stopped):
            self._finish_round(event.session_id)
