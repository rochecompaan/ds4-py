"""Append-only token traces captured from committed facade events."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypedDict, cast

from ds4py.engine.ds4py_replay import read_ds4py
from ds4py.engine.events import Stopped, Synced, Token

if TYPE_CHECKING:
    from collections.abc import Callable

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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def append(self, record: PrefillTraceRecord) -> None:
        line = json.dumps(asdict(record), separators=(",", ":"), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as stream:
            if self._needs_separator:
                stream.write("\n")
                self._needs_separator = False
            stream.write(line + "\n")

    def _finish_round(self, session_id: str) -> None:
        if self._current is None:
            return
        current = self._current
        self.append(
            PrefillTraceRecord(
                session_id=session_id,
                run_id=cast("str", self._run_id),
                step_index=self._step_index,
                round_index=self.round_index,
                prompt=current["prompt"],
                think=self.think,
                cached=current["cached"],
                prefill_suffix=current["prefill_suffix"],
                sampled=tuple(current["sampled"]),
            )
        )
        self._current = None
        self._step_index += 1
        self.round_index += 1

    def _on_event(self, event: SessionEvent) -> None:
        if event.kind == "run_start":
            self._finish_round(event.session_id)
            run_id = event.payload["run_id"]
            if not isinstance(run_id, str):
                raise TypeError("run_start run_id must be a string")
            self._run_id = run_id
            self._active_run = True
            self._step_index = 0
            return
        if event.kind == "run_end":
            self._finish_round(event.session_id)
            self._active_run = False
            return
        if event.kind == "model_step":
            if self._active_run and event.payload["run_id"] == self._run_id:
                step_index = event.payload["step_index"]
                if not isinstance(step_index, int):
                    raise TypeError("model_step step_index must be an integer")
                self._step_index = step_index
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
