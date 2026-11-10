"""A live reserve authorizes only the scoped owner, never inherited tasks/listeners.

Admission policy is also exercised directly against ``OperationState`` so it
provably reads that state instead of reaching back into ``Session``.
"""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import copy_context

import pytest

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative
from ds4py.engine.events import Event, ReserveReached, Stopped
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.reserve import ReserveOperations, legacy_context
from ds4py.engine.session import Session
from ds4py.engine.session_operations import OperationState, blocked_reason
from ds4py.engine.transcript import TranscriptReplacement
from ds4py.errors import EngineError


def pause_event() -> ReserveReached:
    return ReserveReached(1, on_resume=lambda reserve: None, on_fail=lambda: None)


@contextmanager
def windowed_state() -> Generator[tuple[OperationState, ReserveOperations]]:
    with ds4py.open("f.gguf", native=FakeNative()) as engine:
        operations = ReserveOperations(engine.session(), strict=False)
        state = OperationState()
        state.begin_reserve(pause_event())
        state.install_reserve(operations)
        yield state, operations


@contextmanager
def paused() -> Generator[tuple[Session, ReserveReached, Generator[Event]]]:
    with ds4py.open("f.gguf", ctx=673, think=ThinkMode.NONE, native=FakeNative()) as engine:
        session = engine.session()
        session.append_user("hi")
        round_ = session.run_round()
        try:
            pause = next(event for event in round_ if isinstance(event, ReserveReached))
            yield session, pause, round_
        finally:
            round_.close()


def replace(session: Session) -> None:
    session.replace_transcript(
        TranscriptReplacement(session.transcript.tokens, session.transcript.segments), reason="owner"
    )


def test_pause_alone_does_not_authorize_replacement_or_summary() -> None:
    with paused() as (session, _, _):
        with pytest.raises(EngineError, match="round is open"):
            replace(session)
        with pytest.raises(EngineError, match="round is open"):
            session.prepare_summary(0)


def test_owner_permission_propagates_only_to_executor_not_tasks_or_listeners() -> None:
    async def scenario() -> None:
        with paused() as (session, pause, _):
            denied = []

            def listener(event: object) -> None:
                with pytest.raises(EngineError, match="round is open"):
                    replace(session)
                denied.append(True)

            session.on_event(listener, kinds={"probe"})
            with session.events_on_loop(asyncio.get_running_loop()):
                with session.reserve_window(pause):
                    replace(session)
                    await asyncio.to_thread(replace, session)
                    session.emit("probe")
                    await asyncio.to_thread(session.emit, "probe")

                    async def child() -> None:
                        with pytest.raises(EngineError, match="round is open"):
                            replace(session)

                    await asyncio.create_task(child())
                    stale = copy_context()
                await asyncio.to_thread(stale.run, listener, None)
            assert denied == [True, True, True]

    asyncio.run(scenario())


@pytest.mark.parametrize("decision", ["resume", "fail", "close"])
def test_decisions_and_close_revoke_even_captured_permission(decision: str) -> None:
    with paused() as (session, pause, round_):
        with session.reserve_window(pause):
            stale = copy_context()
            if decision == "resume":
                pause.resume(1)
            elif decision == "fail":
                pause.fail()
            else:
                session.close()
            with pytest.raises(EngineError):
                stale.run(replace, session)
        with pytest.raises(EngineError), session.reserve_window(pause):
            pass
        round_.close()


def test_expired_pause_and_nested_windows_reject() -> None:
    with paused() as (session, pause, round_):
        with session.reserve_window(pause), pytest.raises(EngineError), session.reserve_window(pause):
            pass
        round_.close()
        with pytest.raises(EngineError), session.reserve_window(pause):
            pass


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("stale_action", ["fail", "resume"])
def test_expired_round_cannot_revoke_another_rounds_reserve_window(stale_action: str, strict: bool) -> None:
    with paused() as (session, expired_pause, expired_round):
        expired_round.close()
        session.append_user("next")
        prefix = PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens)
        capacity = len(session.transcript) + len(prefix) + 1
        session._worker.call(lambda e, n: setattr(n, "_ctx", capacity)).result()
        current_round = session.run_round()
        try:
            current_pause = next(event for event in current_round if isinstance(event, ReserveReached))
            with session.reserve_window(current_pause, strict=strict) as operations:
                replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
                operations.replace_transcript(replacement, reason="current owner")
                ticket = operations.issue_authority("replacement")
                if stale_action == "fail":
                    with pytest.raises(EngineError, match="expired"):
                        expired_pause.fail()
                else:
                    with pytest.raises(EngineError, match="expired"):
                        expired_pause.resume(1)
                # A stale handle must neither revoke permission nor consume the
                # live decision: native replacement and that decision still work.
                session.replace_transcript(replacement, reason="still current owner", authority=ticket)
                current_pause.fail()
            assert list(current_round) == [Stopped("context_exhausted")]
        finally:
            current_round.close()


@pytest.mark.parametrize("strict", [False, True])
def test_closed_round_retains_window_claim_until_first_exit(strict: bool) -> None:
    with paused() as (session, expired_pause, expired_round):
        expired_window = session.reserve_window(expired_pause, strict=strict)
        expired_window.__enter__()
        try:
            expired_round.close()
            session.append_user("next")
            prefix = PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens)
            capacity = len(session.transcript) + len(prefix) + 1
            session._worker.call(lambda e, n: setattr(n, "_ctx", capacity)).result()
            current_round = session.run_round()
            try:
                current_pause = next(event for event in current_round if isinstance(event, ReserveReached))
                with (
                    pytest.raises(EngineError, match="unclaimed live pause"),
                    session.reserve_window(current_pause, strict=strict),
                ):
                    pass
                # This is the old window's FIRST exit: round close revokes its
                # authority but does not release its retained Session claim.
                expired_window.__exit__(None, None, None)
                with session.reserve_window(current_pause, strict=strict) as operations:
                    replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
                    operations.replace_transcript(replacement, reason="released old claim")
                    current_pause.fail()
                assert list(current_round) == [Stopped("context_exhausted")]
            finally:
                current_round.close()
        finally:
            expired_window.__exit__(None, None, None)


@pytest.mark.parametrize("strict", [False, True])
def test_synthetic_stale_first_exit_preserves_newer_window_identity(strict: bool) -> None:
    """Defensive identity branch, not a publicly reachable window interleaving."""
    with paused() as (session, expired_pause, expired_round):
        expired_window = session.reserve_window(expired_pause, strict=strict)
        expired_window.__enter__()
        try:
            expired_round.close()
            session.append_user("next")
            prefix = PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens)
            capacity = len(session.transcript) + len(prefix) + 1
            session._worker.call(lambda e, n: setattr(n, "_ctx", capacity)).result()
            current_round = session.run_round()
            try:
                current_pause = next(event for event in current_round if isinstance(event, ReserveReached))
                # Public admission forbids this replacement while the old
                # context remains entered. Install a newer handle explicitly to
                # cover the defensive identity guard in that context's finally.
                operations = ReserveOperations(session, strict=strict)
                with session.history.mutation_lock:
                    session.history.operations.install_reserve(operations)
                replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
                ticket = operations.issue_authority("replacement")
                expired_window.__exit__(None, None, None)
                session.replace_transcript(replacement, reason="newer ticket survived", authority=ticket)
                operations.replace_transcript(replacement, reason="newer handle survived")
                current_pause.fail()
                assert list(current_round) == [Stopped("context_exhausted")]
            finally:
                current_round.close()
        finally:
            expired_window.__exit__(None, None, None)


@pytest.mark.parametrize("sampled", [0, 3])
def test_open_assistant_observation_is_exact_and_unkeyed(sampled: int) -> None:
    with ds4py.open("f.gguf", native=FakeNative()) as engine:
        session = engine.session()
        session.append_tokens([10, 11], "user", {"key": "u"})
        assert session.open_assistant_start() is None
        session.append_tokens([20, 21], "assistant_prefix")
        if sampled:
            session.append_tokens([30] * sampled, "assistant")
        assert session.open_assistant_start() == 4
        session.append_tokens([40], "eos")
        assert session.open_assistant_start() == 4
        replacement = TranscriptReplacement(session.transcript.tokens, session.transcript.segments).slice(2)
        session.replace_transcript(replacement, reason="move")
        assert session.open_assistant_start() == 2


def test_unkeyed_assistant_without_prefix_is_invalid() -> None:
    with ds4py.open("f.gguf", native=FakeNative()) as engine:
        session = engine.session()
        session.append_tokens([10], "assistant")
        with pytest.raises(EngineError, match="assistant"):
            session.open_assistant_start()


def test_scoped_summary_supports_executor_and_denies_nested_summary() -> None:
    async def scenario() -> None:
        with paused() as (session, pause, _):
            session._worker.call(lambda e, n: setattr(n, "_ctx", 4096)).result()
            denied = []

            def nested(event: object) -> None:
                with pytest.raises(EngineError, match="round is open"):
                    session.prepare_summary(0)
                with pytest.raises(EngineError), session.reserve_window(pause):
                    pass
                denied.append(True)

            session.on_event(nested, kinds={"SummaryToken"})
            with session.reserve_window(pause):
                first = session.prepare_summary(0)
                second = await asyncio.to_thread(session.prepare_summary, 0)
                assert first.summary == second.summary
                assert denied

    asyncio.run(scenario())


def test_event_loop_callback_cannot_borrow_owner_permission() -> None:
    async def scenario() -> None:
        with paused() as (session, pause, _):
            completed = asyncio.get_running_loop().create_future()

            def callback() -> None:
                try:
                    replace(session)
                except EngineError:
                    completed.set_result(True)
                else:
                    completed.set_result(False)

            with session.reserve_window(pause):
                asyncio.get_running_loop().call_soon(callback)
                assert await completed

    asyncio.run(scenario())


def test_blocked_reason_admits_a_fresh_operation_state() -> None:
    state = OperationState()
    assert blocked_reason(state, "open") is None
    assert blocked_reason(state, "idle") is None
    assert blocked_reason(state, "native") is None
    assert blocked_reason(state, "close") is None


def test_closed_state_rejects_every_policy() -> None:
    state = OperationState()
    state.close()
    assert blocked_reason(state, "open") == "session is closed"
    assert blocked_reason(state, "idle") == "session is closed"
    assert blocked_reason(state, "native") == "session is closed"
    assert blocked_reason(state, "close") == "session is closed"


def test_open_round_blocks_idle_work_and_unwindowed_native_work() -> None:
    state = OperationState()
    state.begin_round()
    assert blocked_reason(state, "idle") == "a round is open"
    # A pause alone installs no window, so it authorizes no native work.
    state.begin_reserve(pause_event())
    assert blocked_reason(state, "native") == "a round is open"
    state.end_round()
    assert blocked_reason(state, "idle") is None


def test_claimed_operation_blocks_close_and_idle_work() -> None:
    state = OperationState()
    claim = state.claim("replacement")
    assert blocked_reason(state, "close") == "a native operation is active; drain before closing"
    assert blocked_reason(state, "idle") == "a round is open"
    state.release(claim)
    assert blocked_reason(state, "close") is None


def test_live_window_admits_paused_native_work_and_consumes_its_ticket_once() -> None:
    with windowed_state() as (state, operations):
        state.begin_round()
        assert blocked_reason(state, "native") is None
        ticket = operations.issue_authority("summary")
        assert state.consume_reserve_authority(ticket, "summary")
        assert not state.consume_reserve_authority(ticket, "summary")
        state.end_reserve()
        assert blocked_reason(state, "native") == "a round is open"


def test_legacy_admission_requires_an_active_window_context() -> None:
    with windowed_state() as (state, operations):
        assert not state.consume_reserve_authority(None, "replacement")
        with legacy_context(operations, enabled=True):
            assert state.consume_reserve_authority(None, "replacement")
        state.end_reserve()
        with legacy_context(operations, enabled=True):
            assert not state.consume_reserve_authority(None, "replacement")
