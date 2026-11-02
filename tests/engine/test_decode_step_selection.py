import gc
import weakref
from typing import Any

import pytest

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative, reply
from ds4py.engine.decode import PlainStep
from ds4py.engine.events import ReserveReached, Stopped, Token
from ds4py.errors import EngineError


class RecordingStep:
    def __init__(self, *, hooks_every_sample: bool = True) -> None:
        self.calls = 0
        self.hooks_every_sample = hooks_every_sample

    def run(self, session: Any, ctx: Any) -> None:
        self.calls += 1
        PlainStep().run(session, ctx)


def stop_then_next(prompt: Any, generated: Any) -> Any:
    stop = tuple(byte + 1 for byte in b"stop")
    source = (
        b"next"
        if any(tuple(prompt[index : index + len(stop)]) == stop for index in range(len(prompt) - len(stop) + 1))
        else b"stop"
    )
    return source[len(generated)] + 1 if len(generated) < len(source) else 0


@pytest.mark.parametrize("chat", [False, True], ids=["raw", "chat"])
def test_session_forwards_the_installed_step(chat: bool) -> None:
    with ds4py.open("f.gguf", think=ThinkMode.NONE, native=FakeNative(script=reply(b"ok"))) as engine:
        session = engine.session()
        if chat:
            session.append_user("hi")
        else:
            session.append_tokens([5, 6], "user")
        step = RecordingStep()
        session.set_decode_step(step)

        events = list(session.run_round())

        assert step.calls == 1
        assert b"".join(event.text_bytes for event in events if isinstance(event, Token)) == b"ok"
        assert events[-1] == Stopped("stop_token")


def test_default_step_matches_explicit_plain_step() -> None:
    def run(*, install_plain_step: bool) -> tuple[list[Any], tuple[int, ...]]:
        with ds4py.open("f.gguf", native=FakeNative(script=reply(b"ok"))) as engine:
            session = engine.session()
            session.append_tokens([5, 6], "user")
            if install_plain_step:
                session.set_decode_step(PlainStep())
            events = list(session.run_round())
            return events, session.transcript.tokens

    assert run(install_plain_step=False) == run(install_plain_step=True)


def test_active_round_rejects_step_changes() -> None:
    with ds4py.open("f.gguf", native=FakeNative(script=reply(b"ok"))) as engine:
        session = engine.session()
        session.append_tokens([5, 6], "user")
        events = session.run_round()
        next(event for event in events if isinstance(event, Token))
        try:
            with pytest.raises(EngineError, match="a round is open"):
                session.set_decode_step(RecordingStep())
        finally:
            events.close()


@pytest.mark.parametrize("chat", [False, True], ids=["raw", "chat"])
def test_paused_round_rejects_step_changes(chat: bool) -> None:
    with ds4py.open("f.gguf", think=ThinkMode.NONE, native=FakeNative(script=stop_then_next)) as engine:
        session = engine.session()
        if chat:
            session.append_user("hi")
        else:
            session.append_tokens([5, 6], "user")
        assert list(session.run_round(stop_sequences=[b"stop"]))[-1] == Stopped("stop_sequence")
        with pytest.raises(EngineError, match="paused round"):
            session.set_decode_step(None)


def test_raw_stop_sequence_resume_uses_the_same_step() -> None:
    with ds4py.open("f.gguf", native=FakeNative(script=stop_then_next)) as engine:
        session = engine.session()
        session.append_tokens([5, 6], "user")
        step = RecordingStep()
        session.set_decode_step(step)

        assert list(session.run_round(stop_sequences=[b"stop"]))[-1] == Stopped("stop_sequence")
        assert step.calls == 1
        assert list(session.run_round(stop_sequences=[b"stop"]))[-1] == Stopped("stop_token")
        assert step.calls == 2


def test_reserve_resume_uses_the_same_step() -> None:
    with ds4py.open("f.gguf", ctx=673, think=ThinkMode.NONE, native=FakeNative(script=reply(b"ok"))) as engine:
        session = engine.session()
        session.append_user("hi")
        step = RecordingStep()
        session.set_decode_step(step)
        events = session.run_round()
        pause = next(event for event in events if isinstance(event, ReserveReached))
        assert step.calls == 1

        session._worker.call(lambda engine_handle, native: setattr(native, "_ctx", 1024)).result()
        pause.resume(0)
        assert list(events)[-1] == Stopped("stop_token")
        assert step.calls == 2


@pytest.mark.parametrize(("hooks_every_sample", "expected_calls"), [(False, 0), (True, 1)])
def test_hooks_control_step_fallback(hooks_every_sample: bool, expected_calls: int) -> None:
    with ds4py.open("f.gguf", native=FakeNative(script=reply(b"ok"))) as engine:
        session = engine.session()
        session.append_tokens([5, 6], "user")
        step = RecordingStep(hooks_every_sample=hooks_every_sample)
        session.set_decode_step(step)
        session.add_logits_hook(lambda view: None)

        assert list(session.run_round())[-1] == Stopped("stop_token")
        assert step.calls == expected_calls


def test_closing_the_session_releases_the_step() -> None:
    with ds4py.open("f.gguf", native=FakeNative()) as engine:
        session = engine.session()
        step = RecordingStep()
        step_ref = weakref.ref(step)
        session.set_decode_step(step)
        session.append_tokens([5, 6], "user")
        assert list(session.run_round())[-1] == Stopped("stop_token")
        session.close()
        del step
        gc.collect()
        assert step_ref() is None
