"""Own one agent, its session lifecycle, saves, and telemetry.

The TUI talks to ``AgentRuntime`` and to session events, never to native
handles: one prompt is one harness run, saves and restores go through
``SessionStore`` and the harness step store, and every run leaves a record.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord as StepRunRecord,
    StepPersistence,
)

from ds4py.agent import build_agent_components
from ds4py.capabilities.run_record import DS4RunRecord
from ds4py.capabilities.system_notes import DS4SystemNotes
from ds4py.errors import Ds4pyError, EngineError, StoreError
from ds4py.store import (
    KV_HEADER,
    KV_MAGIC,
    KV_VERSION,
    SaveRecord,
    SegmentProvenance,
    SessionStore,
    token_digest,
)

_PLACEHOLDER_HEADER = KV_HEADER.pack(KV_MAGIC, KV_VERSION, 0)
_SAVE_PREFIX = ".save-"


def _bindable(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop a trailing empty response an interrupted stream left behind."""
    while messages and not messages[-1].parts:
        messages.pop()
    return messages


class SessionStepStore:
    """Session-scoped locator for the step store of the live session.

    The harness sees one state object for the whole session, so a fork keeps
    this object (and its identity) while only the delegate moves to the new
    session directory.
    """

    def __init__(self, store):
        self._store = store
        self._closed = False

    @property
    def current(self):
        return self._store

    @property
    def closed(self) -> bool:
        return self._closed

    def relocate(self, store) -> None:
        self._store = store

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._store, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class _RestorePlan:
    path: str
    payload_bytes: int | None
    mismatch: str | None


class AgentRuntime:
    """One project session driven by one installed engine.

    ``messages``, ``session_id``, ``title``, and ``status`` are read-only
    views. ``run`` executes one prompt, ``save`` / ``switch`` publish and adopt
    session directories, ``new`` / ``fork`` change identity, and ``close``
    stops the session and everything scoped to it.
    """

    def __init__(
        self,
        model,
        project_dir,
        *,
        store=None,
        clock=None,
        read_only=False,
        interactive=False,
        answerer=None,
    ):
        self.model = model
        self.project_dir = Path(project_dir)
        self.store = store if store is not None else SessionStore()
        self._clock = clock if clock is not None else (lambda: datetime.now().astimezone())
        self._session = model.session
        self._messages: list[ModelMessage] = []
        self._title = ""
        self._status = "idle"
        self._closed = False
        self._running = False
        self._run_task = None
        self._interrupted = False
        self._created_at = self._timestamp()
        self._session_dir = self.store.session_dir(self._session.id)
        self.components = build_agent_components(
            model,
            self.project_dir,
            read_only=read_only,
            interactive=interactive,
            answerer=answerer,
            step_store=FileStepStore(self._session_dir / "steps"),
            session_dir=self._session_dir,
        )
        self._agent = self.components.agent
        self._notes = next(
            (item for item in self.components.components if isinstance(item, DS4SystemNotes)),
            None,
        )
        self._record = next(
            (item for item in self.components.components if isinstance(item, DS4RunRecord)),
            None,
        )
        if self._notes is not None:
            # build_agent_components owns the wall clock; a runtime clock keeps a
            # fresh render byte-identical to the transcript it was saved from.
            self._notes.clock = self._clock
        self._session.on_event(self._on_engine_event, kinds={"Interrupted", "Stopped"})

    @property
    def messages(self) -> list[ModelMessage]:
        return list(self._messages)

    @property
    def session_id(self) -> str:
        return self._session.id

    @property
    def title(self) -> str:
        return self._title

    @property
    def status(self) -> str:
        return self._status

    @property
    def created_at(self) -> str:
        return self._created_at

    def on_event(self, listener, *, kinds=None):
        """Subscribe to session events; returns the unsubscribe call."""
        return self._session.on_event(listener, kinds=kinds)

    def _timestamp(self) -> str:
        now = self._clock()
        return now.isoformat() if isinstance(now, datetime) else str(now)

    def _require_open(self) -> None:
        if self._closed:
            raise EngineError("the runtime is closed")

    def _step_store(self, session_id: str) -> FileStepStore:
        return FileStepStore(self.store.session_dir(session_id) / "steps")

    def _holder(self) -> SessionStepStore:
        session_id = self._session.id
        return self._session.state(
            "step_store", lambda: SessionStepStore(self._step_store(session_id)),
        )

    def _on_engine_event(self, event) -> None:
        if event.kind == "Interrupted":
            self._interrupted = True
            return
        inner = (event.payload or {}).get("event")
        if getattr(inner, "reason", None) == "interrupted":
            self._interrupted = True

    def _note_title(self) -> None:
        if self._title:
            return
        for message in self._messages:
            for part in message.parts:
                if not isinstance(part, UserPromptPart) or not isinstance(part.content, str):
                    continue
                text = " ".join(part.content.split())
                if text:
                    self._title = text[:80]
                    return

    async def _stop_active(self) -> None:
        if not self._running:
            return
        task = self._run_task
        self.cancel()
        if task is not None and task is not asyncio.current_task():
            await asyncio.wait([task])

    async def _latest(self, run_id: str):
        try:
            holder = self._holder()
            return await holder.current.latest_snapshot(
                run_id=run_id, include_interrupted=True,
            )
        except (Ds4pyError, OSError, ValueError):
            return None

    async def _adopt_history(self, run_id, messages, snapshot=None) -> None:
        """Adopt canonical keys and keep the terminal harness snapshot exact."""
        self._messages = self.model.binder.adopt(messages)
        if snapshot is None:
            snapshot = await self._latest(run_id)
        if snapshot is not None:
            snapshot.messages = deepcopy(self._messages)
            snapshot.idempotency_key = None
            await self._holder().current.save_snapshot(snapshot)

    async def _fail(self, run_id: str, status: str, error, *, persist=True) -> None:
        self._status = status
        if persist:
            try:
                snapshot = await self._latest(run_id)
                if snapshot is not None:
                    await self._adopt_history(
                        run_id, _bindable(list(snapshot.messages)), snapshot=snapshot,
                    )
                    self._note_title()
            except Exception as persistence_error:  # noqa: BLE001 - preserve the run error
                if error is None:
                    error = persistence_error
        self._session.emit(
            "run_end",
            run_id=run_id,
            status=status,
            error=None if error is None else str(error),
        )

    async def run(self, prompt: str):
        """Run one prompt as one harness run and record its terminal status."""
        self._require_open()
        if self._running:
            raise EngineError("a run is already active")
        self._running = True
        self._run_task = asyncio.current_task()
        self._interrupted = False
        self._status = "running"
        run_id = uuid4().hex
        try:
            if self._record is not None:
                self._record.failure = None
            self._session.emit("run_start", run_id=run_id)
            holder = self._holder()
            try:
                result = await self._agent.run(
                    prompt,
                    message_history=list(self._messages),
                    conversation_id=self._session.id,
                    run_id=run_id,
                    capabilities=[StepPersistence(store=holder.current)],
                )
            except asyncio.CancelledError:
                await self._fail(run_id, "interrupted", None)
                raise
            except BaseException as error:
                await self._fail(
                    run_id, "interrupted" if self._interrupted else "error", error,
                )
                raise
            try:
                await self._adopt_history(run_id, list(result.all_messages()))
            except asyncio.CancelledError:
                await self._fail(run_id, "interrupted", None, persist=False)
                raise
            except BaseException as error:
                await self._fail(run_id, "error", error, persist=False)
                raise
            self._note_title()
            if self._record is not None and self._record.failure is not None:
                failure = self._record.failure
                await self._fail(run_id, "error", failure)
                raise EngineError(f"run record failed: {failure}") from failure
            if self._record is not None:
                try:
                    self._record.write_run_end(run_id=run_id, status="success")
                except OSError as failure:
                    await self._fail(run_id, "error", failure)
                    raise EngineError(f"run record failed: {failure}") from failure
            self._status = "success"
            self._session.emit("run_end", run_id=run_id, status="success")
            return result
        finally:
            self._running = False

    def cancel(self) -> None:
        """Cancel the active run's engine work; a no-op while idle."""
        if self._status == "running":
            self._session.cancel()

    def load_prefix(self, turns) -> list[ModelMessage]:
        """Seed the history from prefix-file turns so the next run renders them.

        A stripped restore keeps messages but no tokens; prefix turns are the
        same situation before the first run. Nothing renders here: the next
        ``prepare`` renders these messages through the binder like any history.
        """
        self._require_open()
        messages: list[ModelMessage] = []
        for turn in turns:
            if turn.role == "assistant":
                messages.append(ModelResponse([TextPart(turn.content)]))
            elif turn.role == "user":
                messages.append(ModelRequest([UserPromptPart(turn.content)]))
            else:
                raise EngineError(f"unsupported prefix turn role: {turn.role!r}")
        self._messages = messages
        self.model.binder.forget()
        self._title = ""
        self._note_title()
        return list(messages)

    async def save(self) -> SaveRecord:
        """Write the KV file, then a complete snapshot, then the save record."""
        self._require_open()
        await self._stop_active()
        session_id = self._session.id
        session_dir = self.store.session_dir(session_id)
        tokens = list(self._session.transcript.tokens)
        if not tokens:
            raise StoreError("cannot save a session with an empty transcript")
        holder = self._holder()
        run_id = await self._next_save_run_id(holder.current)

        written = {}

        def write_payload(file):
            saved = self._session.save_payload(file, header=_PLACEHOLDER_HEADER)
            written["tokens"] = saved.tokens
            written["payload_bytes"] = saved.payload_bytes
            return saved.payload_bytes

        payload_bytes = self.store.write_kv_stream_atomic(
            session_dir / "session.kv", write_payload,
        )
        if written.get("tokens") != len(tokens) or written.get("payload_bytes") != payload_bytes:
            raise EngineError(
                f"the native payload covers {written.get('tokens')} tokens, "
                f"the transcript has {len(tokens)}"
            )

        snapshot = ContinuableSnapshot(
            run_id=run_id,
            step_index=0,
            messages=deepcopy(list(self._messages)),
            conversation_id=session_id,
            agent_name="ds4py",
        )
        await holder.current.register_run(StepRunRecord(
            run_id=run_id, conversation_id=session_id, agent_name="ds4py",
        ))
        await holder.current.save_snapshot(snapshot)

        info = self._session.engine.info
        record = SaveRecord(
            session_id=session_id,
            conversation_id=session_id,
            run_id=run_id,
            step_index=snapshot.step_index,
            model_id=info.model_id,
            quant_bits=info.routed_quant_bits,
            ctx=self._session.engine.config.ctx,
            title=self._title,
            created_at=self._created_at,
            last_used=self._timestamp(),
            payload_bytes=payload_bytes,
            tokens=len(tokens),
            tokens_sha256=token_digest(tokens),
            segments=tuple(
                SegmentProvenance(
                    segment.start, segment.end, segment.kind, segment.key,
                    segment.content_hash, segment.source,
                )
                for segment in self.model.binder.export_segments()
            ),
        )
        self.store.write_record_atomic(session_dir / "save.json", record)
        self._session.emit("saved", record=record.to_dict(), payload_bytes=payload_bytes)
        return record

    async def _next_save_run_id(self, store) -> str:
        prefix = f"{self._session.id}{_SAVE_PREFIX}"
        runs = await store.list_runs(conversation_id=self._session.id)
        numbers = [
            int(run.run_id[len(prefix):])
            for run in runs
            if run.run_id.startswith(prefix) and run.run_id[len(prefix):].isdigit()
        ]
        return f"{prefix}{max(numbers, default=0) + 1}"

    def _restore_plan(self, record: SaveRecord, session_dir: Path) -> _RestorePlan:
        kv_path = session_dir / "session.kv"
        if record.payload_bytes == 0:
            return _RestorePlan("render", None, "the save record declares no KV payload")
        if not kv_path.is_file():
            return _RestorePlan("render", None, f"{kv_path}: the KV file is missing")
        declared = self.store.read_kv_header(kv_path)
        if declared != record.payload_bytes:
            return _RestorePlan(
                "render", None,
                f"{kv_path}: the KV payload is {declared} bytes, "
                f"the save record declares {record.payload_bytes}",
            )
        return _RestorePlan("kv", declared, None)

    async def switch(self, session_prefix: str) -> SaveRecord:
        """Adopt a saved session: load its KV, or render its snapshot fresh."""
        self._require_open()
        await self._stop_active()
        record, _record_path = self.store.find(session_prefix)
        self.store.validate_for_engine(record, self._session.engine.info)
        session_dir = self.store.session_dir(record.session_id)
        snapshots = await self._step_store(record.session_id).list_snapshots(
            run_id=record.run_id, include_interrupted=True,
        )
        snapshot = next(
            (item for item in snapshots if item.step_index == record.step_index), None,
        )
        if snapshot is None:
            raise StoreError(
                f"{session_dir}: the save record names snapshot {record.run_id} "
                f"step {record.step_index}, which is missing"
            )
        plan = self._restore_plan(record, session_dir)
        if self._record is not None:
            self._record.relocate(session_dir / "events.jsonl")
        outcome = {}

        def restore():
            try:
                self.model.binder.validate_restore(snapshot.messages, record.segments)
            except EngineError as error:
                raise StoreError(
                    f"{session_dir / 'save.json'}: cannot restore: {error}"
                ) from error
            loaded = None
            if plan.payload_bytes is not None:
                with (session_dir / "session.kv").open("rb") as file:
                    file.seek(KV_HEADER.size)
                    loaded = self._session.load_payload(
                        file, payload_bytes=plan.payload_bytes,
                    )
                if len(loaded) != record.tokens:
                    outcome["rejected"] = (
                        f"the loaded payload has {len(loaded)} tokens, "
                        f"the save record declares {record.tokens}"
                    )
                elif token_digest(loaded) != record.tokens_sha256:
                    outcome["rejected"] = (
                        "the loaded payload token hash does not match the save record"
                    )
                if "rejected" in outcome:
                    loaded = None
            if loaded is not None:
                try:
                    self.model.binder.restore(snapshot.messages, record.segments)
                except EngineError as error:
                    raise StoreError(
                        f"{session_dir / 'save.json'}: cannot restore: {error}"
                    ) from error
                if self._notes is not None:
                    self._notes.restore_from_segments(self._session.transcript.segments)
            else:
                self._session.truncate(0)
                self.model.binder.forget()
                if self._notes is not None:
                    self._notes.restore_from_segments(record.segments)
                    self._notes.sync_from_segments(())
            self._messages = list(snapshot.messages)
            self._title = record.title
            self._created_at = record.created_at
            return {
                "restore_path": "kv" if loaded is not None else "render",
                "mismatch": (
                    None if loaded is not None
                    else outcome.get("rejected", plan.mismatch)
                ),
            }

        self._session.switch(
            record.session_id,
            restore=restore,
            record=record.to_dict(),
            restore_path=plan.path,
            mismatch=plan.mismatch,
        )
        self._status = "idle"
        if "rejected" in outcome:
            self._session.emit(
                "restore_rejected",
                record=record.to_dict(),
                mismatch=outcome["rejected"],
            )
        return record

    async def new(self) -> str:
        """Start a clean session; the old directory and its files stay intact."""
        self._require_open()
        await self._stop_active()
        session_id = self._session.new()
        self._messages = []
        self._title = ""
        self._status = "idle"
        self._created_at = self._timestamp()
        self.model.binder.forget()
        if self._record is not None:
            self._record.relocate(self.store.session_dir(session_id) / "events.jsonl")
        if self._notes is not None:
            self._notes.sync_from_segments(())
        return session_id

    async def fork(self) -> str:
        """Copy identity for a new session directory; live state stays bound."""
        self._require_open()
        await self._stop_active()
        session_id = self._session.fork()
        self._created_at = self._timestamp()
        self._holder().relocate(self._step_store(session_id))
        if self._record is not None:
            self._record.relocate(self.store.session_dir(session_id) / "events.jsonl")
        return session_id

    async def close(self) -> None:
        """Stop active work, then close the session and everything scoped to it."""
        if self._closed:
            return
        await self._stop_active()
        self._closed = True
        self._status = "closed"
        self._session.close()
