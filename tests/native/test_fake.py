from typing import Any
import ctypes
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from ds4py._native.fake import FAKE_EOS, FakeClock, FakeNative, reply
from ds4py._native.protocol import EngineConfig, SyncStatus, ThinkMode
from ds4py.errors import EngineError, MissingBinding
from contract import CONTRACT


def engine(**kwargs: Any) -> Any:
    return FakeNative(**kwargs).open_engine(EngineConfig(Path("fake.gguf")))


@pytest.mark.parametrize("check", CONTRACT, ids=lambda f: f.__name__)
def test_contract(check: Any) -> None:
    e = engine()
    check(e, e.tokenize("The quick brown fox"))


def test_tokenizer_round_trips_bytes() -> None:
    e = engine()
    assert b"".join(e.token_text(t) for t in e.tokenize("héllo")) == "héllo".encode()
    assert e.tokenize("héllo", rendered=True) == e.tokenize("héllo")
    assert e.eos() == FAKE_EOS and e.is_stop(FAKE_EOS) and not e.is_stop(e.tokenize("a")[0])
    assert e.token_text(FAKE_EOS) == b"</s>"
    assert (e.info().name, e.info().family, e.info().mtp_draft_tokens, e.info().vocab_size) == ("fake", "deepseek4", 1, 279)
    custom = replace(e.info(), family="glm", is_glm_dsa=True)
    assert engine(info=custom).info() == custom


def test_progress_events_follow_ds4() -> None:
    clock = FakeClock()
    s = engine(chunk=4, clock=clock, chunk_seconds=0.05).create_session(64)
    s.sync(list(range(1, 7)))
    events = []
    s.sync(list(range(1, 17)), progress=lambda *e: events.append(e))
    assert events == [("prefill_chunk", 6, 16), ("prefill_chunk", 10, 16),
                      ("prefill_chunk", 14, 16), ("prefill_chunk", 16, 16)]
    assert clock() == pytest.approx(0.05 * 2 + 0.05 * 3)
    assert s.calls == [("sync", 6), ("sync", 10)]


def test_cancel_keeps_the_partial_prefix_valid() -> None:
    s = engine(chunk=4).create_session(64)
    seen = []
    status = s.sync(list(range(1, 17)), progress=lambda *e: seen.append(e), cancel=lambda: len(seen) >= 2)
    assert status is SyncStatus.INTERRUPTED and s.pos() == 4 and s.checkpoint_valid()
    assert s.tokens() == [1, 2, 3, 4] and s.calls == [("sync", 4)]
    s.sync(list(range(1, 17)))
    assert s.calls[-1] == ("sync", 12)


def test_script_drives_sampling_and_calls_are_logged() -> None:
    s = engine(script=reply(b"hi")).create_session(64)
    s.sync([5, 6])
    out = []
    while (t := s.sample(1.0, 0, 1.0, 0.05, ctypes.c_uint64(3))) != FAKE_EOS:
        s.eval(t)
        out.append(t)
    assert bytes(t - 1 for t in out) == b"hi"
    assert ("eval", FAKE_EOS) not in s.calls and s.calls[0] == ("sync", 2)


def test_script_inputs_rng_and_snapshot_generated_state() -> None:
    inputs = []
    def script(prompt: Any, generated: Any) -> Any:
        inputs.append((list(prompt), list(generated)))
        return len(generated) + 20
    s = engine(script=script).create_session(64)
    s.sync([5, 6])
    rng = ctypes.c_uint64(3)
    assert s.argmax() == 20 and rng.value == 3
    assert s.sample(1.0, 0, 1.0, 0.05, rng) == 20
    assert rng.value == 2092789425003139053
    s.eval(20)
    snap = s.save_snapshot()
    assert snap.nbytes == s.payload_bytes() == 3072
    s.sync([8])
    s.load_snapshot(snap)
    assert s.argmax() == 21 and inputs[-1] == ([5, 6], [20])
    s.eval(21)
    s.load_snapshot(snap)
    assert s.tokens() == [5, 6, 20]
    snap.close()
    snap.close()
    assert snap.closed
    with pytest.raises(EngineError, match="snapshot is closed"):
        s.load_snapshot(snap)


def test_sync_rebuilds_instead_of_reusing_a_partial_match() -> None:
    s = engine().create_session(64)
    s.sync([1, 2, 3, 4])
    s.sync([1, 2, 9])
    assert s.calls[-1] == ("sync", 3)
    s.rewind(2)
    assert s.calls[-1] == ("rewind", 2) and not s.checkpoint_valid()
    assert s.argmax() == s.sample(1, 0, 1, 0, ctypes.c_uint64(1)) == -1
    s.sync([1, 2, 3])
    assert s.calls[-1] == ("sync", 3)
    s.rewind(3)
    s.rewind(4)
    assert s.checkpoint_valid() and s.calls[-1] == ("sync", 3)
    s.invalidate()
    s.sync([1, 2, 3])
    assert s.calls[-1] == ("sync", 3)
    s.sync([1, 2, 3])
    assert s.calls[-1] == ("sync", 0)


def test_eval_context_full_and_token_copy() -> None:
    s = engine().create_session(3)
    s.sync([1, 2])
    s.eval(3)
    with pytest.raises(EngineError):
        s.eval(4)
    tokens = s.tokens()
    tokens.clear()
    assert s.ctx() == s.pos() == 3 and s.tokens() == [1, 2, 3]
    assert s.calls[-1] == ("eval", 3)


@pytest.mark.parametrize("state", ["fresh", "invalidated", "rewound"])
@pytest.mark.parametrize("operation", ["eval", "save_snapshot"])
def test_invalid_checkpoint_operations_leave_state_unchanged(state: Any, operation: Any) -> None:
    s = engine().create_session(64)
    if state != "fresh":
        s.sync([1, 2, 3])
        if state == "invalidated":
            s.invalidate()
        else:
            s.rewind(2)
    tokens, calls = s.tokens(), list(s.calls)
    assert not s.checkpoint_valid()
    with pytest.raises(EngineError):
        if operation == "eval":
            s.eval(4)
        else:
            s.save_snapshot()
    assert s.tokens() == tokens and s.calls == calls
    assert not s.checkpoint_valid()


def test_snapshot_cannot_resurrect_rewound_checkpoint() -> None:
    s = engine().create_session(64)
    s.sync([1, 2, 3])
    s.rewind(2)
    try:
        snap = s.save_snapshot()
    except EngineError:
        pass
    else:
        s.load_snapshot(snap)
        snap.close()
    s.sync([1, 2, 4])
    assert s.calls[-1] == ("sync", 3)
    assert s.tokens() == [1, 2, 4] and s.checkpoint_valid()


def test_bind(ext_site: Any) -> None:
    native = FakeNative(bindings={"ds4_engine_layer_count": lambda e: 43},
                        absent={"ds4_engine_sampling_defaults"})
    assert native.bind("ds4_engine_layer_count")(None) == 43
    with pytest.raises(MissingBinding, match="absent"):
        native.bind("ds4_engine_sampling_defaults")
    with pytest.raises(MissingBinding, match="does not declare"):
        native.bind("ds4_session_tp_spec_cycle")
    assert native.backend == "cpu" and native.restore_enabled
    assert not FakeNative(restore_enabled=False).restore_enabled


@pytest.mark.parametrize("text, expected", [("0", 1000), ("100", 1100), ("050", 1050),
    ("101", None), ("", None), ("-1", None), ("+1", None), (" 1", None), ("１", None), ("high", None)])
def test_think_level(text: Any, expected: Any) -> None:
    assert FakeNative().parse_think_level(text) == expected


def test_think_context() -> None:
    n = FakeNative()
    assert n.think_mode_for_context(ThinkMode.MAX, 393215) == ThinkMode.HIGH
    assert n.think_mode_for_context(ThinkMode.MAX, 393216) == ThinkMode.MAX
    assert n.think_mode_for_context(ThinkMode.LOW, 64) == ThinkMode.LOW


def test_thread_confinement_and_close() -> None:
    e = engine()
    s = e.create_session(64)
    assert e.raw == id(e) and s.raw == id(s)
    with ThreadPoolExecutor(1) as pool:
        for call in (e.info, lambda: e.tokenize("x"), lambda: e.create_session(64), e.close,
                     s.pos, lambda: s.sync([1]), s.argmax, s.close):
            with pytest.raises(AssertionError, match="thread"):
                pool.submit(call).result()
    s.close()
    s.close()
    with pytest.raises(EngineError, match="closed"):
        s.pos()
    other = e.create_session(64)
    e.close()
    e.close()
    with pytest.raises(EngineError, match="closed"):
        other.pos()
    with pytest.raises(EngineError, match="closed"):
        e.info()


def test_fake_logits_can_be_copied_set_and_are_invalidated_by_eval() -> None:
    from array import array

    # The fake normally prefers 7, then 9, then EOS.
    sequence = iter((7, 9, FAKE_EOS))
    s = engine(script=lambda prompt, generated: next(sequence)).create_session(64)
    s.sync([5, 6])
    values = array("f", [0.0]) * s._engine.info().vocab_size
    s.copy_logits(values)
    assert (values[7], values[8]) == (1.0, 0.0)
    values[8] = 3.0
    s.set_logits(values)
    assert s.sample(0.0, 0, 1.0, 0.0, ctypes.c_uint64(1)) == 8
    s.eval(8)
    s.copy_logits(values)
    assert (values[8], values[9]) == (0.0, 1.0)
    for bad in (array("f", [0.0]), array("f", [0.0]) * (len(values) + 1)):
        with pytest.raises(ValueError, match="vocabulary size"):
            s.copy_logits(bad)
        with pytest.raises(ValueError, match="vocabulary size"):
            s.set_logits(bad)


@pytest.mark.parametrize("operation", ["sync", "rewind", "snapshot"])
def test_fake_logits_cache_is_invalidated_by_state_changes(operation: Any) -> None:
    from array import array

    s = engine(script=lambda prompt, generated: 7).create_session(64)
    s.sync([5, 6])
    values = array("f", [0.0]) * s._engine.info().vocab_size
    s.copy_logits(values)
    values[8] = 3.0
    s.set_logits(values)
    if operation == "sync":
        s.sync([5, 6])
    elif operation == "rewind":
        s.rewind(s.pos())
    else:
        snapshot = s.save_snapshot()
        s.load_snapshot(snapshot)
        snapshot.close()
    assert s.argmax() == 7


def test_think_mode_none_stops_at_think_opener() -> None:
    from ds4py import ThinkMode
    from ds4py._native.fake_chat import tokenize_rendered
    native = FakeNative()
    engine = native.open_engine(EngineConfig(Path('f')))
    try:
        token = tokenize_rendered('<think>')[0]
        assert engine.is_stop_for_think_mode(token, ThinkMode.NONE)
        assert not engine.is_stop_for_think_mode(token, ThinkMode.HIGH)
        assert not engine.is_stop(token)
    finally:
        engine.close()
