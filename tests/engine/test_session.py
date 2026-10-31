import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Never, override

import pytest

import ds4py
from ds4py import ThinkMode, open
from ds4py._native.fake import FakeNative, FakeSession, reply
from ds4py._native.protocol import EngineConfig
from ds4py.engine.events import (
    Checkpointed,
    Interrupted,
    Prefill,
    Stopped,
    Synced,
    Token,
    ToolCalls,
)
from ds4py.engine.plan import Plan
from ds4py.engine.sampling import SamplingPolicy
from ds4py.engine.session import Engine, Session
from ds4py.engine.transcript import Transcript
from ds4py.engine.worker import SYSTEM, EngineWorker
from ds4py.errors import CheckpointError, EngineError


def fake_engine(*, ctx: Any = 64, cancel: Any = None, **fake: Any) -> Engine:
    return ds4py.open("f.gguf", ctx=ctx, cancel=cancel, native=FakeNative(**fake))


def test_open_checks_backend_and_think_mode() -> None:
    with fake_engine() as e:
        assert e.info.name == "fake" and e.config.backend == "cpu" and e.think == ThinkMode.HIGH
    with pytest.raises(EngineError, match="built for cpu"):
        ds4py.open("f.gguf", backend="metal", native=FakeNative())
    with ds4py.open("f.gguf", ctx=64, think=ThinkMode.MAX, native=FakeNative()) as e:
        assert e.think == ThinkMode.HIGH


@pytest.mark.parametrize("fail", [False, True])
def test_think_helper_runs_on_owner_and_failure_closes_worker(fail: Any) -> None:
    handles, calls = [], []

    class TrackingNative(FakeNative):
        @override
        def open_engine(self, config: Any) -> Any:
            engine = super().open_engine(config)
            handles.append(engine)
            create = engine.create_session

            def create_session(ctx: Any) -> Any:
                session = create(ctx)
                handles.append(session)
                return session

            engine.create_session = create_session  # ty: ignore[invalid-assignment]  # Instance-level test double replaces a bound method.
            return engine

        @override
        def think_mode_for_context(self, mode: Any, ctx: Any) -> Any:
            calls.append(threading.current_thread())
            if fail:
                raise EngineError("think helper failed")
            return super().think_mode_for_context(mode, ctx)

    native = TrackingNative()
    if fail:
        with pytest.raises(EngineError, match="think helper failed"):
            ds4py.open("f.gguf", ctx=64, think=ThinkMode.MAX, native=native)
    else:
        with ds4py.open("f.gguf", ctx=64, think=ThinkMode.MAX, native=native) as e:
            assert e.think == ThinkMode.HIGH
            assert e.session().on_worker(lambda h: threading.current_thread()).result() is calls[0]
    assert len(handles) == 2 and all(handle.raw == 0 for handle in handles)
    assert len(calls) == 1 and calls[0] is not threading.current_thread()
    assert not calls[0].is_alive()


def test_sampling_defaults_follow_the_engine() -> None:
    native = FakeNative()
    handle = native.open_engine(EngineConfig(Path("f")))
    glm = replace(handle.info(), family="glm", is_glm_dsa=True)
    handle.close()
    with ds4py.open("f.gguf", seed=5, native=FakeNative(info=glm)) as e:
        assert e.sampling_defaults == SamplingPolicy(top_p=0.95, min_p=0.0, seed=5)


def test_a_round_syncs_generates_and_extends_the_transcript() -> None:
    with fake_engine(script=reply(b"ok")) as e:
        s = e.session()
        assert isinstance(e, Engine) and isinstance(s, Session)
        s.append_tokens(e.tokenize("hi"), "user")
        events = list(s.run_round(max_tokens=10))
        assert [type(ev).__name__ for ev in events if not isinstance(ev, Prefill)] == [
            "Synced",
            "Token",
            "Token",
            "Stopped",
        ]
        assert s.transcript.tokens == tuple(e.tokenize("hiok"))
        assert [seg.kind for seg in s.transcript.segments] == ["user", "assistant"]
        assert s.ctx_used == 4 and s.plan() == Plan("extend", 4, 0, True, None)
        with pytest.raises(EngineError, match="one session"):
            e.session()


def test_malformed_dsml_raw_utf8_finishes_as_parser_error() -> None:
    payload = (
        '<｜DSML｜tool_calls><｜DSML｜invoke name="read" extra="'.encode()
        + b"\xff"
        + (
            '"><｜DSML｜parameter name="path" string="true">x</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>'
        ).encode()
    )
    with ds4py.open(
        "f.gguf",
        think=ThinkMode.NONE,
        native=FakeNative(script=reply(payload)),
    ) as engine:
        session = engine.session()
        session.append_user("hi")
        events = list(session.run_round())

    assert events[-1] == Stopped("parser_error")
    assert not any(isinstance(event, ToolCalls) for event in events)


def test_public_round_forwards_stop_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    from ds4py.engine import session as session_module

    original = session_module.GenerateJob
    captured = []

    def capture_job(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(session_module, "GenerateJob", capture_job)
    with fake_engine(script=reply(b"ok")) as e:
        session = e.session()
        session.append_tokens([5, 6], "user")
        assert isinstance(
            list(
                session.run_round(
                    max_tokens=10,
                    stop_sequences=[b"stop"],
                )
            )[-1],
            Stopped,
        )
    assert captured[0]["control"].sequences == (b"stop",)


@pytest.mark.parametrize("pause_at", [Synced, Token])
def test_overlapping_round_is_refused_without_releasing_the_first(
    pause_at: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    with fake_engine(script=reply(b"ok")) as e:
        s = e.session()
        s.append_tokens([5, 6], "user")
        first, unused = s.run_round(), s.run_round()
        second = None
        try:
            next(ev for ev in first if isinstance(ev, pause_at))
            unused.close()
            second = s.run_round()
            with monkeypatch.context() as mp:
                mp.setattr(EngineWorker, "submit", lambda *a, **k: pytest.fail("overlapping round submitted a job"))
                with pytest.raises(EngineError, match="a round is open"):
                    next(second)
                with pytest.raises(EngineError, match="a round is open"):
                    s.append_tokens([7], "user")
                with pytest.raises(EngineError, match="a round is open"):
                    next(s.run_round())
            assert isinstance(list(first)[-1], Stopped)
            assert s.ctx_used == len(s.transcript)
            s.append_tokens([7], "user")
            assert isinstance(list(s.run_round(max_tokens=0))[-1], Stopped)
        finally:
            first.close()
            unused.close()
            if second is not None:  # ty: ignore[redundant-condition-strict]  # second is assigned only on the alternate path.
                second.close()


@pytest.mark.parametrize("method", ["sync", "eval"])
def test_worker_error_releases_the_round(method: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    original = getattr(FakeSession, method)
    failed = False

    def fail_once(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        if not failed:
            failed = True
            raise EngineError("native job failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FakeSession, method, fail_once)
    with fake_engine() as e:
        s = e.session()
        s.append_tokens([5, 6], "user")
        with pytest.raises(EngineError, match="native job failed"):
            list(s.run_round())
        s.append_tokens([7], "user")
        assert isinstance(list(s.run_round())[-1], Stopped)
        assert s.ctx_used == len(s.transcript)


def test_ctrl_c_during_sync_keeps_the_transcript() -> None:
    with fake_engine() as e:
        s = e.session()
        s.append_tokens(e.tokenize("hello"), "user")
        s.cancel()
        assert list(s.run_round())[-1] == Interrupted(0)
        assert s.transcript.tokens == tuple(e.tokenize("hello"))
        assert isinstance(list(s.run_round())[-1], Stopped)


def test_closing_a_round_early_keeps_what_was_evaluated() -> None:
    cancel = threading.Event()
    with fake_engine(script=reply(b"abcdefgh"), cancel=cancel) as e:
        s = e.session()
        s.append_tokens([5, 6], "user")
        rnd = s.run_round()
        first = next(ev for ev in rnd if isinstance(ev, Token))
        assert s.transcript.tokens[2] == first.id
        with pytest.raises(EngineError, match="a round is open"):
            s.append_tokens([7], "user")
        rnd.close()
        assert s.ctx_used == len(s.transcript)
        assert s.plan().kind == "extend" and not cancel.is_set()
        s.append_tokens([7], "user")
        assert isinstance(list(s.run_round(max_tokens=0))[-1], Stopped)


def test_checkpoints_through_the_session() -> None:
    with fake_engine() as e:
        s = e.session()
        s.append_tokens(list(range(1, 9)), "system")
        list(s.run_round(max_tokens=0))
        assert s.checkpoint(SYSTEM) == Checkpointed(SYSTEM, 8, 8 * 1024)
        s.drop_checkpoint(SYSTEM)
        with pytest.raises(CheckpointError):
            s.drop_checkpoint(SYSTEM)


def test_extension_binding_runs_on_the_worker(ext_site: Any) -> None:
    native = FakeNative(bindings={"ds4_engine_layer_count": lambda engine_ptr: 43})
    with ds4py.open("f.gguf", ctx=64, native=native) as e:
        layer_count = e.native.bind("ds4_engine_layer_count")
        count, session_ptr, thread = (
            e.session().on_worker(lambda h: (layer_count(h.engine), h.session, threading.get_ident())).result()
        )
        assert count == 43 and isinstance(session_ptr, int) and thread != threading.get_ident()


# Exceptional admission and deterministic in-flight round cleanup.


@pytest.mark.parametrize("failure", ["capture", "submit"])
def test_failure_before_submission_releases_ownership(failure: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    with open("f.gguf", ctx=64, native=FakeNative()) as e:
        s = e.session()
        s.append_tokens([5, 6], "user")
        with monkeypatch.context() as mp:

            def fail(*args: Any) -> Never:
                raise EngineError("admission failed")

            if failure == "capture":
                mp.setattr(Transcript, "tokens", property(fail))
            else:
                mp.setattr(EngineWorker, "submit", fail)
            with pytest.raises(EngineError, match="admission failed"):
                next(s.run_round())
        s.append_tokens([7], "user")
        assert list(s.run_round(max_tokens=0))[-1] == Stopped("max_tokens")


@pytest.mark.parametrize("fail_eval", [False, True])
def test_close_drains_inflight_commits_before_releasing_ownership(
    fail_eval: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancel, evaluating, release = threading.Event(), threading.Event(), threading.Event()
    original = FakeSession.eval

    def blocked_eval(self: Any, token: Any) -> Any:
        if token == ord("b") + 1:
            evaluating.set()
            assert release.wait(5), "test failed to release eval"
            if fail_eval:
                raise EngineError("drained eval failed")
        return original(self, token)

    monkeypatch.setattr(FakeSession, "eval", blocked_eval)
    with open("f.gguf", ctx=64, native=FakeNative(script=reply(b"abc")), cancel=cancel) as e:
        s = e.session()
        s.append_tokens([5, 6], "user")
        rnd = s.run_round()
        assert next(ev for ev in rnd if isinstance(ev, Token)) == Token(98, b"a")
        assert evaluating.wait(5)
        errors = []

        def close() -> None:
            try:
                rnd.close()
            except EngineError as error:
                errors.append(str(error))

        closer = threading.Thread(target=close)
        closer.start()
        try:
            assert cancel.wait(5)
            with pytest.raises(EngineError, match="a round is open"):
                s.append_tokens([7], "user")
            with pytest.raises(EngineError, match="a round is open"):
                next(s.run_round())
        finally:
            release.set()
            closer.join(5)
        assert not closer.is_alive()
        assert errors == (["drained eval failed"] if fail_eval else [])
        assert s.transcript.tokens == ((5, 6, 98) if fail_eval else (5, 6, 98, 99))
        assert s.ctx_used == len(s.transcript) and not cancel.is_set()
        s.append_tokens([7], "user")
        assert list(s.run_round(max_tokens=0))[-1] == Stopped("max_tokens")


def test_round_uses_current_sampling_defaults_but_keeps_one_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = FakeSession.sample

    def record(self: Any, temperature: Any, top_k: Any, top_p: Any, min_p: Any, rng: Any) -> Any:
        calls.append((temperature, top_k, top_p, min_p, rng, rng.value))
        return original(self, temperature, top_k, top_p, min_p, rng)

    monkeypatch.setattr(FakeSession, "sample", record)
    with open("f.gguf", ctx=64, seed=9, native=FakeNative()) as e:
        s = e.session()
        s.append_tokens([5, 6], "user")
        e.sampling_defaults = SamplingPolicy(temperature=0.5, top_k=3, seed=42)
        list(s.run_round(max_tokens=1))
        list(s.run_round(max_tokens=1, sampling=SamplingPolicy(top_p=0.7)))
        assert calls[0][:4] == (0.5, 3, 1.0, 0.05)
        assert calls[1][:4] == (1.0, 0, 0.7, 0.05)
        assert calls[0][4] is calls[1][4]
        assert calls[0][5] == 9 and calls[1][5] != 9
        assert e.token_text(98) == b"a"


def test_cancel_after_sync_interrupts_generation_and_clears_flag() -> None:
    with open("f.gguf", ctx=64, native=FakeNative()) as e:
        s = e.session()
        s.append_tokens([5, 6], "user")
        rnd = s.run_round()
        next(ev for ev in rnd if isinstance(ev, Synced))
        s.cancel()
        assert list(rnd) == [Stopped("interrupted")]
        assert s.transcript.tokens == (5, 6)
        assert list(s.run_round(max_tokens=1))[-1] == Stopped("max_tokens")


def test_negative_round_budget_does_not_generate() -> None:
    with fake_engine(script=reply(b"ok")) as engine:
        session = engine.session()
        session.append_tokens([5, 6], "user")
        events = list(session.run_round(max_tokens=-1))
        assert not any(isinstance(event, Token) for event in events)
        assert events[-1] == Stopped("max_tokens")
        assert session.transcript.tokens == (5, 6) and session.ctx_used == 2
        assert list(session.run_round(max_tokens=1))[-1] == Stopped("max_tokens")
        assert session.transcript.tokens == (5, 6, 112)


def test_context_reserve_passes_through() -> None:
    with open("f.gguf", ctx=4, native=FakeNative()) as e:
        s = e.session()
        s.append_tokens([5, 6], "user")
        assert list(s.run_round())[-1] == Stopped("context_reserve")
        assert s.ctx_used == len(s.transcript) == 3
