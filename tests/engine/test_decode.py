from typing import Any, cast
from collections.abc import Iterator
import threading
from pathlib import Path

import pytest

from ds4py._native.fake import FAKE_EOS, FakeNative, reply
from ds4py._native.protocol import EngineConfig
from ds4py.engine.decode import PlainStep, generate
from ds4py.engine.events import Stopped, Token
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.transcript import Transcript
from ds4py.engine.worker import EngineWorker, GenerateJob, SyncJob
from ds4py.errors import EngineError

_workers = []


@pytest.fixture(autouse=True)
def close_workers() -> Iterator[Any]:
    yield
    for w in _workers:
        w.close()
    _workers.clear()


def worker(script: Any, *, ctx: Any=64, cancel: Any=None) -> Any:
    w = EngineWorker(FakeNative(script=script), EngineConfig(Path("f"), ctx=ctx), cancel=cancel)
    _workers.append(w)
    return w

def gen(w: Any, prompt: Any, max_tokens: Any=50, transcript: Any=None) -> Any:
    transcript = Transcript() if transcript is None else transcript
    list(w.submit(SyncJob(tuple(prompt))))
    return list(w.submit(GenerateJob(SamplingPolicy(), Rng(1), max_tokens, transcript)))

def tok(c: Any) -> Token:
    return Token(ord(c) + 1, c.encode())

def test_stop_token_is_neither_kept_nor_evaluated() -> None:
    w, t = worker(reply(b"hi")), Transcript()
    assert gen(w, [5, 6], transcript=t) == [tok("h"), tok("i"), Stopped("stop_token")]
    assert t.tokens == (tok("h").id, tok("i").id) and t.segments[0].kind == "assistant"
    calls, pos = w.call(lambda e, s: (s.calls, s.pos())).result()
    assert ("eval", FAKE_EOS) not in calls and pos == 4

def test_max_tokens_keeps_the_last_token() -> None:
    w = worker(reply(b"hello"))
    assert gen(w, [5, 6], max_tokens=2) == [tok("h"), tok("e"), Stopped("max_tokens")]
    assert w.call(lambda e, s: s.pos()).result() == 4
    assert gen(w, [5, 6], max_tokens=0) == [Stopped("max_tokens")]

def test_context_reserve_matches_the_ds4_agent_clamp() -> None:
    w = worker(reply(b"abcdef"), ctx=8)                    # room 3, less a reserve of 1
    assert gen(w, [1, 2, 3, 4, 5]) == [tok("a"), tok("b"), Stopped("context_reserve")]

def test_ctrl_c_keeps_the_partial_tokens() -> None:
    cancel = threading.Event()
    def script(prompt: Any, generated: Any) -> Any:
        if len(generated) == 2: cancel.set()               # Ctrl+C while the third token is sampled
        return ord("a") + 1
    w, t = worker(script, cancel=cancel), Transcript()
    assert gen(w, [5, 6], transcript=t) == [tok("a")] * 3 + [Stopped("interrupted")]
    assert len(t) == 3 and not cancel.is_set() and w.call(lambda e, s: s.pos()).result() == 5

def test_sampling_needs_a_valid_state() -> None:
    w = worker(reply(b"x"))
    list(w.submit(SyncJob((5, 6, 7))))
    w.call(lambda e, s: s.rewind(1)).result()               # invalidates the checkpoint, as in ds4
    with pytest.raises(EngineError, match="sync first"):
        list(w.submit(GenerateJob(SamplingPolicy(), Rng(1), 5, Transcript())))

class StubEngine:
    def is_stop(self, token: Any) -> Any: return token == 0
    def token_text(self, token: Any) -> bytes: return b"x"

class StubSession:                                          # just what PlainStep and generate use
    def __init__(self, tokens: Any, fail_eval: Any=None) -> None:
        self.tokens, self.fail_eval, self.seen = list(tokens), fail_eval, []
    def ctx(self) -> int: return 100
    def pos(self) -> int: return 10
    def sample(self, temperature: Any, top_k: Any, top_p: Any, min_p: Any, rng: Any) -> Any:
        self.seen.append(("sample", temperature, top_k, top_p, min_p, rng.value))
        return self.tokens.pop(0)
    def eval(self, token: Any) -> None:
        if token == self.fail_eval: raise EngineError("eval failed")
        self.seen.append(("eval", token))

def test_plain_step_follows_the_contract() -> None:
    s, t, out = StubSession([7, 0]), Transcript(), []
    policy = SamplingPolicy(temperature=0.5, top_p=0.9, min_p=0.0, top_k=3)
    reason = generate(cast(Any, StubEngine()), cast(Any, s), GenerateJob(policy, Rng(9), 10, t), cancelled=lambda: False, emit=out.append)
    assert reason == "stop_token" and out == [Token(7, b"x")] and PlainStep.hooks_every_sample
    assert s.seen == [("sample", 0.5, 3, 0.9, 0.0, 9), ("eval", 7), ("sample", 0.5, 3, 0.9, 0.0, 9)]
    assert t.tokens == (7,)

@pytest.mark.parametrize("budget", [0, -1])
def test_nonpositive_budget_never_samples_or_commits(budget: Any) -> None:
    session, transcript, events = StubSession([7]), Transcript(), []
    transcript.append([1, 2], "user")
    rng = Rng(9)
    reason = generate(cast(Any, StubEngine()), cast(Any, session),
                      GenerateJob(SamplingPolicy(), rng, budget, transcript),
                      cancelled=lambda: False, emit=events.append)
    assert reason == "max_tokens" and events == []
    assert session.seen == [] and rng.state.value == 9
    assert transcript.tokens == (1, 2)
    assert [segment.kind for segment in transcript.segments] == ["user"]


def test_a_token_is_committed_after_its_eval_and_before_its_event() -> None:
    s, t, out = StubSession([7, 8], fail_eval=8), Transcript(), []
    t.append([1, 2], "user")
    job = GenerateJob(SamplingPolicy(), Rng(9), 10, t)
    with pytest.raises(EngineError, match="eval failed"):
        generate(cast(Any, StubEngine()), cast(Any, s), job, cancelled=lambda: False, emit=lambda ev: out.append((ev, t.tokens)))
    assert out == [(Token(7, b"x"), (1, 2, 7))]             # 8 failed its eval: not kept, not published
    assert [(seg.kind, seg.end) for seg in t.segments] == [("user", 2), ("assistant", 3)]
