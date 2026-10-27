"""Session-scoped state identity and lifecycle across prompts, new, switch, fork, exit."""
import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeEngine, FakeNative, FakeSession, reply
from ds4py._native.protocol import SyncStatus


class EmptySyncSession(FakeSession):
    def __init__(self, engine, ctx):
        super().__init__(engine, ctx)

    def sync(self, tokens, *, progress=None, cancel=None):
        if not tokens:
            self._tokens = []
            self._prompt_length = 0
            self._valid = True
            self._logits = None
            return SyncStatus.DONE
        return super().sync(tokens, progress=progress, cancel=cancel)


class ResetNative(FakeNative):
    def open_engine(self, config):
        engine = FakeEngine(self)

        def create_session(ctx):
            session = EmptySyncSession(engine, ctx)
            engine._sessions.add(session)
            return session

        engine.create_session = create_session
        return engine


def reset_engine(*, ctx=1024, script=None):
    native = ResetNative(script=script or reply(b"ok"))
    return ds4py.open("f.gguf", ctx=ctx, think=ThinkMode.NONE, native=native)


class State:
    def __init__(self):
        self.closes = 0

    def close(self):
        self.closes += 1


def test_state_is_created_once_and_survives_two_prompts():
    calls = []
    state = State()

    def factory():
        calls.append("created")
        return state

    with reset_engine() as engine:
        session = engine.session()
        assert session.state("jobs", factory) is state
        session.append_user("one")
        assert list(session.run_round(max_tokens=2))[-1].reason == "max_tokens"
        assert session.state("jobs", lambda: (_ for _ in ()).throw(AssertionError("recreated"))) is state
        session.append_user("two")
        assert list(session.run_round(max_tokens=2))[-1].reason == "max_tokens"
        assert session.state("jobs", lambda: State()) is state
        assert calls == ["created"]


def test_new_closes_all_state_and_a_new_factory_creates_a_new_object():
    with reset_engine() as engine:
        session = engine.session()
        first, second = State(), State()
        session.state("a", lambda: first)
        session.state("b", lambda: second)

        session.new()

        assert (first.closes, second.closes) == (1, 1)
        assert session.state("a", lambda: State()) is not first
        assert session.state("b", lambda: State()) is not second


def test_switch_closes_state_before_restore_and_fork_keeps_it():
    with reset_engine() as engine:
        session = engine.session()
        state = State()
        session.state("jobs", lambda: state)
        order = []

        def restore():
            order.append(("restore", state.closes))

        session.on_event(lambda event: order.append(event.kind))
        session.switch("b" * 32, restore)

        assert order == [
            "switching", "session_end", ("restore", 1), "session_start", "switched",
        ]
        assert state.closes == 1

        kept = State()
        session.state("jobs", lambda: kept)
        new_id = session.fork()
        assert new_id == session.id
        assert kept.closes == 0
        assert session.state("jobs", lambda: State()) is kept


def test_exit_closes_state_exactly_once():
    engine = reset_engine()
    session = engine.session()
    state = State()
    session.state("jobs", lambda: state)

    engine.close()
    engine.close()

    assert state.closes == 1


def test_state_objects_are_independent_per_key_and_close_in_creation_order():
    with reset_engine() as engine:
        session = engine.session()
        closed = []
        first = State()
        second = State()
        first.close = lambda: closed.append("first")
        second.close = lambda: closed.append("second")
        session.state("first", lambda: first)
        session.state("second", lambda: second)

        session.new()

        assert closed == ["first", "second"]
