"""Session-owned binding transactions publish whole append groups atomically."""

import asyncio
import threading
from collections.abc import Sequence
from types import MappingProxyType
from typing import override

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative, FakeSession
from ds4py._native.protocol import CancelFn, ProgressFn, SyncStatus
from ds4py.binding import MessageBinder
from ds4py.capabilities.system_notes import DS4SystemNotes, NotePreview, NoteState
from ds4py.engine.events import Stopped
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.session import Engine, Session
from ds4py.engine.transcript import Segment, TranscriptReplacement
from ds4py.errors import CheckpointError, EngineError


def fake_engine(*, cancel: threading.Event | None = None) -> Engine:
    return ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=FakeNative(), cancel=cancel)


def user(text: str) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)])


def replacement(tokens: Sequence[int], *segments: tuple[int, int, str]) -> TranscriptReplacement:
    return TranscriptReplacement(
        tuple(tokens),
        tuple(Segment(start, end, kind, MappingProxyType({})) for start, end, kind in segments),
    )


def decode(engine: Engine, tokens: Sequence[int]) -> str:
    return b"".join(engine.token_text(token) for token in tokens).decode()


class PlanningGate(DS4SystemNotes):
    """Hold planning open so a second thread's mutation races the claimed plan."""

    def __init__(self, session: Session) -> None:
        super().__init__(session, clock=lambda: "fixed")
        self.entered = threading.Event()
        self.settled = threading.Event()
        self._gated = False

    @override
    def preview(self) -> NotePreview:
        if not self._gated:
            self._gated = True
            self.entered.set()
            # The publisher holds only its claim here, never the mutation lock,
            # so waiting for the other thread is a bounded deadlock guard.
            assert self.settled.wait(10), "the external mutation attempt never settled"
        return super().preview()


class FailingNotes(DS4SystemNotes):
    """Fail the second publication that installs an already emitted start date."""

    def __init__(self, session: Session, *, after: bool) -> None:
        super().__init__(session, clock=lambda: "fixed")
        self.seen = 0
        self._after = after

    @override
    def apply_preview(self, state: NoteState) -> None:
        if state.date_emitted:
            self.seen += 1
            if self.seen == 2:
                if self._after:
                    super().apply_preview(state)
                raise ValueError("preview failed")
        super().apply_preview(state)


def test_binding_boundary_cannot_interleave_external_append() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        notes = PlanningGate(session)
        binder.note_provider = notes
        rejected: list[str] = []

        def intruder() -> None:
            assert notes.entered.wait(10), "binding planning never started"
            try:
                session.append_system_note("intruder")
            except EngineError as error:
                rejected.append(str(error))
            notes.settled.set()

        messages: list[ModelMessage] = [user("one"), user("two")]
        thread = threading.Thread(target=intruder, daemon=True)
        thread.start()
        try:
            binder.prepare(messages)
        finally:
            notes.settled.set()
            thread.join(10)
        # The claim covers planning through publication, so the concurrent
        # writer is refused instead of landing between planned boundaries.
        written = [segment for segment in session.transcript.segments if segment.meta.get("text") == "intruder"]
        assert rejected == ["a round is open"]
        assert not written
        assert binder.records()[-1].end == len(session.transcript)


def test_native_preamble_work_keeps_the_binding_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        in_preamble = threading.Event()
        attempted_done = threading.Event()
        rejected: list[str] = []
        append = FakeSession.sync

        def gated_sync(
            self: FakeSession,
            tokens: Sequence[int],
            *,
            progress: ProgressFn | None = None,
            cancel: CancelFn | None = None,
        ) -> SyncStatus:
            in_preamble.set()
            assert attempted_done.wait(3), "the external append attempt never completed"
            return append(self, tokens, progress=progress, cancel=cancel)

        def intruder() -> None:
            assert in_preamble.wait(3), "binding preparation never reached native work"
            try:
                session.append_system_note("intruder")
            except EngineError as error:
                rejected.append(str(error))
            attempted_done.set()

        monkeypatch.setattr(FakeSession, "sync", gated_sync)
        thread = threading.Thread(target=intruder, daemon=True)
        thread.start()
        try:
            binder.prepare([user("one"), user("two")])
        finally:
            attempted_done.set()
            thread.join(10)
        assert rejected == ["a round is open"]
        assert "intruder" not in decode(engine, session.transcript.tokens)
        assert binder.records()[-1].end == len(session.transcript)


def test_preamble_event_delivery_does_not_deadlock_a_mutation_attempt() -> None:
    # The preparing thread holds only its claim during native work, so a
    # loop-delivered listener can never wait on that thread's mutation lock.
    async def scenario() -> None:
        with fake_engine() as engine:
            session = engine.session()
            binder = MessageBinder(session)
            outcomes: list[str] = []

            def attempt_mutation() -> None:
                try:
                    session.append_tokens([9], "user")
                except EngineError:
                    outcomes.append("rejected")
                else:
                    outcomes.append("appended")

            def listener(event: object) -> None:
                done = threading.Event()

                def attempt() -> None:
                    attempt_mutation()
                    done.set()

                threading.Thread(target=attempt, daemon=True).start()
                if not done.wait(5):
                    outcomes.append("blocked")

            session.on_event(listener, kinds={"Synced"})
            with session.events_on_loop(asyncio.get_running_loop()):
                await asyncio.wait_for(asyncio.to_thread(binder.prepare, [user("go")]), 30)
            assert outcomes == ["rejected"]

    asyncio.run(scenario())


def test_checkpoint_drop_does_not_deadlock_a_listener_mutation_attempt() -> None:
    # Checkpoint disposal during the transaction runs with only the claim held,
    # so a loop-delivered listener can never wait on the producer's lock.
    async def scenario() -> None:
        with fake_engine() as engine:
            session = engine.session()
            session.start_chat()
            session.checkpoint("turn")
            stale = len(session._checkpoint_tokens)
            assert stale == 2
            dropped: list[str] = []
            observed: list[int] = []
            outcomes: list[str] = []

            def attempt_mutation() -> None:
                try:
                    session.append_tokens([9], "user")
                except EngineError:
                    outcomes.append("rejected")
                else:
                    outcomes.append("appended")

            def listener(event: SessionEvent) -> None:
                dropped.append(event.kind)
                observed.append(len(session.transcript))
                done = threading.Event()

                def attempt() -> None:
                    attempt_mutation()
                    done.set()

                threading.Thread(target=attempt, daemon=True).start()
                if not done.wait(5):
                    outcomes.append("blocked")

            session.on_event(listener, kinds={"Dropped"})

            def truncate() -> None:
                with session.binding_transaction() as transaction:
                    transaction.truncate(0)

            with session.events_on_loop(asyncio.get_running_loop()):
                await asyncio.wait_for(asyncio.to_thread(truncate), 30)
            assert dropped == ["Dropped"] * stale
            assert outcomes == ["rejected"] * stale
            # The Python truncation committed before disposal was dispatched, and
            # the refused writer never interleaved with the transaction.
            assert observed == [0] * stale
            assert session.transcript.tokens == ()
            assert session._checkpoint_tokens == {}

    asyncio.run(scenario())


def test_session_truncate_drops_checkpoints_outside_the_mutation_lock() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        session.checkpoint("stale")
        locked: list[bool] = []
        session.on_event(lambda _event: locked.append(session._lock.locked()), kinds={"Dropped"})

        session.truncate(0)

        assert locked == [False, False]
        assert session.transcript.tokens == ()
        assert session._checkpoint_tokens == {}


def test_reconcile_drops_checkpoints_outside_the_mutation_lock() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        session.checkpoint("stale")
        locked: list[bool] = []
        session.on_event(lambda _event: locked.append(session._lock.locked()), kinds={"Dropped"})

        session.reconcile_round(0)

        assert locked == [False, False]
        assert session.transcript.tokens == ()
        assert session._checkpoint_tokens == {}


def test_failed_checkpoint_drop_propagates_and_keeps_the_session_usable() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        # A key Python lists but the worker never snapshotted: disposal rejects
        # from the native job, after the locked Python truncation committed.
        session._checkpoint_tokens["ghost"] = len(session.transcript)
        with session.binding_transaction() as transaction:
            with pytest.raises(CheckpointError, match="ghost"):
                transaction.truncate(0)
            assert session.transcript.tokens == ()
            assert session._checkpoint_tokens == {}
        # The rejected disposal releases the claim and later admission still works.
        assert session._operations.kind is None
        session.append_tokens([9], "user")
        assert session.transcript.tokens == (9,)


def test_binding_plan_cannot_apply_against_changed_prefix() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        group = replacement((101, 102), (0, 2, "user"))
        with session.binding_transaction() as transaction:
            # A stale writer commits history the planned offsets never observed.
            session._transcript.append([7], "user")
            stale = session.transcript.segments
            with pytest.raises(EngineError, match="binding transaction"):
                transaction.apply_appends(group)
            assert session.transcript.segments == stale
            assert session.transcript.tokens[-1] == 7
        session.apply_appends(group)
        assert session.transcript.tokens[-2:] == (101, 102)


def test_apply_appends_installs_one_relative_group_and_copies_metadata() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        start = len(session.transcript)
        meta = {"text": "staged"}
        group = TranscriptReplacement(
            (101, 102, 103),
            (Segment(0, 1, "note", meta), Segment(1, 3, "user", MappingProxyType({}))),
        )
        session.apply_appends(group)
        meta["text"] = "changed"
        installed = session.transcript.segments[-2:]
        assert [(segment.start, segment.end, segment.kind) for segment in installed] == [
            (start, start + 1, "note"),
            (start + 1, start + 3, "user"),
        ]
        assert dict(installed[0].meta) == {"text": "staged"}
        assert session.transcript.tokens[start:] == (101, 102, 103)
        assert session.transcript.chat
        # Appends are pure Python: they never synchronize native state.
        native = session._worker.call(lambda e, n: tuple(n.tokens())).result()
        assert native == session.transcript.tokens[:start]


@pytest.mark.parametrize(
    ("tokens", "segments", "message"),
    [
        ((1, 2), ((0, 2, "bogus"),), "unknown segment kind"),
        ((1, 2), ((0, 1, "user"),), "replacement segments cover"),
        ((1,), ((0, 0, "user"),), "must cover the tokens"),
    ],
    ids=["unknown-kind", "uncovered", "empty-segment"],
)
def test_apply_appends_rejects_an_invalid_group_without_mutation(
    tokens: Sequence[int], segments: Sequence[tuple[int, int, str]], message: str
) -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        before = (
            session.transcript.tokens,
            session.transcript.segments,
            session.transcript.chat,
            session._paused_stream,
            session._paused_control,
            session._paused_sampled_tokens,
        )
        native = session._worker.call(lambda e, n: tuple(n.tokens())).result()
        with pytest.raises(ValueError, match=message):
            session.apply_appends(replacement(tokens, *segments))
        # A rejected group never reaches native state.
        assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == native
        assert (
            session.transcript.tokens,
            session.transcript.segments,
            session.transcript.chat,
            session._paused_stream,
            session._paused_control,
            session._paused_sampled_tokens,
        ) == before


def test_apply_appends_accepts_an_empty_group_as_a_no_op() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        before = session.transcript.tokens, session.transcript.segments
        session.apply_appends(TranscriptReplacement((), ()))
        assert (session.transcript.tokens, session.transcript.segments) == before


@pytest.mark.parametrize("after", [False, True])
def test_later_publication_failure_retains_the_completed_prefix(after: bool) -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        notes = FailingNotes(session, after=after)
        binder.note_provider = notes
        messages: list[ModelMessage] = [user("one"), user("two")]
        with pytest.raises(ValueError, match="preview failed"):
            binder.prepare(messages)
        # The first group published; the failing group left no partial tokens.
        assert len(binder.records()) == 1
        assert binder.records()[0].end == len(session.transcript)
        assert messages[0].metadata is not None
        assert messages[1].metadata is None
        assert notes._date_emitted and not notes._hints_full_emitted
        # The failed publication released its claim, so an idle operation is
        # admitted again before the retry.
        assert session._operations.kind is None
        session.checkpoint("after-failure")
        binder.prepare(messages)
        assert len(binder.records()) == 2
        assert binder.records()[-1].end == len(session.transcript)
        assert [segment.meta.get("subtype") for segment in session.transcript.segments if segment.kind == "note"] == [
            "date"
        ]
        assert messages[1].metadata is not None


def test_failed_retirement_restores_the_whole_group_and_paused_state() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.append_user("first")
        assert list(session.run_round(stop_sequences=[b"ok"]))[-1] == Stopped("stop_sequence")
        paused = (session._paused_stream, session._paused_control, session._paused_sampled_tokens)
        assert paused[0] is not None and paused[1] is not None
        tokens = PromptBuilder(engine).append_assistant_end()
        group = replacement(tokens, (0, len(tokens), "eos"))
        before = session.transcript.tokens
        native = session._worker.call(lambda e, n: tuple(n.tokens())).result()

        def fail() -> None:
            raise EngineError("publication failed")

        with session.binding_transaction() as transaction:
            with pytest.raises(EngineError, match="publication failed"):
                transaction.apply_appends(group, retire_paused=True, publish=fail)
            assert session.transcript.tokens == before
            assert (
                session._paused_stream,
                session._paused_control,
                session._paused_sampled_tokens,
            ) == paused
            # Python-only appends never roll native state back or forward.
            assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == native
            transaction.apply_appends(group, retire_paused=True)
        assert session.transcript.tokens[len(before) :] == tuple(tokens)
        assert (
            session._paused_stream,
            session._paused_control,
            session._paused_sampled_tokens,
        ) == (None, None, None)


def test_preamble_listener_mutation_is_rejected_instead_of_reentering() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        rejected: list[str] = []

        def listener(event: object) -> None:
            try:
                session.append_tokens([9], "user")
            except EngineError as error:
                rejected.append(str(error))

        session.on_event(listener, kinds={"Synced"})
        binder.prepare([user("go")])
        assert rejected == ["a round is open"]
        assert 9 not in session.transcript.tokens
        assert [segment.kind for segment in session.transcript.segments].count("user") == 1


def test_clear_cancellation_clears_the_worker_signal_after_drain() -> None:
    cancel = threading.Event()
    with fake_engine(cancel=cancel) as engine:
        session = engine.session()
        session.append_user("first")
        assert isinstance(list(session.run_round())[-1], Stopped)
        session.cancel()
        assert cancel.is_set()
        session.clear_cancellation()
        assert not cancel.is_set()
        assert isinstance(list(session.run_round())[-1], Stopped)


def test_binding_claim_preserves_each_distinct_guard_policy() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.append_user("hi")
        with session.binding_transaction():
            # Idle mutations and identity changes are refused while planning.
            with pytest.raises(EngineError, match="a round is open"):
                session.append_tokens([7], "user")
            with pytest.raises(EngineError, match="a round is open"):
                session.new()
            with pytest.raises(EngineError, match="a round is open"):
                session.checkpoint("turn")
            # Close refuses a live native operation instead of racing it.
            with pytest.raises(EngineError, match="native operation is active"):
                session.close()
            # Observation and pure planning carry no idle restriction.
            assert session.ctx_used > 0
            assert session.plan() is not None
            assert session.transcript.tokens
        session.append_tokens([7], "user")
        assert session.transcript.tokens[-1] == 7
