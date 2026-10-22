"""Public history observation must not bypass exclusive round ownership."""
import threading

import pytest

import ds4py
from ds4py._native.fake import FakeNative, FakeSession, reply
from ds4py.engine.events import Stopped, Synced, Token


@pytest.mark.parametrize("round_open", [False, True])
@pytest.mark.parametrize("method,args", [
    ("append", ([7], "user")),
    ("extend", ([7],)),
    ("truncate", (0,)),
])
def test_public_transcript_has_no_mutators(round_open, method, args):
    with ds4py.open("f.gguf", ctx=64, native=FakeNative()) as engine:
        session = engine.session()
        session.append_tokens([5, 6], "user")
        view = session.transcript
        rnd = session.run_round(max_tokens=0)
        try:
            if round_open:
                next(event for event in rnd if isinstance(event, Synced))
            with pytest.raises(AttributeError):
                getattr(view, method)(*args)
            assert view.tokens == (5, 6)
            assert len(view) == 2
            if round_open:
                with pytest.raises(ds4py.EngineError, match="a round is open"):
                    session.append_tokens([7], "user")
            assert list(rnd)[-1] == Stopped("max_tokens")
            assert session.ctx_used == 2
        finally:
            rnd.close()
        session.append_tokens([7], "user")
        assert view.tokens == (5, 6, 7)  # A retained view observes later guarded writes.


def test_retained_view_observes_worker_commits_without_exposing_writes(monkeypatch):
    evaluating, release = threading.Event(), threading.Event()
    original_eval = FakeSession.eval

    def blocked_eval(self, token):
        if token == 99:
            evaluating.set()
            assert release.wait(5), "eval release timed out"
        return original_eval(self, token)

    monkeypatch.setattr(FakeSession, "eval", blocked_eval)
    with ds4py.open("f.gguf", ctx=64, native=FakeNative(script=reply(b"ab"))) as engine:
        session = engine.session()
        view = session.transcript
        session.append_tokens([5, 6], "user", {"label": "prompt"})
        tokens, segments = view.tokens, view.segments
        rnd = session.run_round(max_tokens=2)
        try:
            assert next(event for event in rnd if isinstance(event, Token)) == Token(98, b"a")
            assert evaluating.wait(5)
            for method, args in [("append", ([7], "user")), ("extend", ([7],)), ("truncate", (0,))]:
                with pytest.raises(AttributeError):
                    getattr(view, method)(*args)
            assert view.tokens == (5, 6, 98)
            assert len(view) == 3
            assert [(s.start, s.end, s.kind) for s in view.segments] == [
                (0, 2, "user"), (2, 3, "assistant"),
            ]
            assert tokens == (5, 6) and len(segments) == 1
            with pytest.raises(TypeError):
                view.segments[0].meta["label"] = "changed"
        finally:
            release.set()
            rnd.close()
        assert view.tokens == (5, 6, 98, 99)
        assert session.ctx_used == len(view) == 4
