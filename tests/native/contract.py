import ctypes
import pytest
from ds4py._native.protocol import SyncStatus
from ds4py.errors import EngineError

def rebuilt_argmax(s, tokens):
    s.invalidate(); s.sync(tokens)
    return s.argmax()

def check_extend_matches_rebuild(engine, prompt):
    s = engine.create_session(len(prompt) + 64)
    assert s.sync(prompt[:-2]) is SyncStatus.DONE and s.pos() == len(prompt) - 2
    events = []
    assert s.sync(prompt, progress=lambda *e: events.append(e)) is SyncStatus.DONE
    assert s.tokens() == list(prompt) and s.checkpoint_valid()
    chunks = [e for e in events if e[0] == "prefill_chunk"]    # Metal interleaves prefill_display
    assert chunks[-1] == ("prefill_chunk", len(prompt), len(prompt))
    assert s.argmax() == rebuilt_argmax(s, prompt)

def check_common_prefix(engine, prompt):
    assert prompt[-1] != prompt[0]
    s = engine.create_session(len(prompt) + 64)
    s.sync(prompt)
    diverged = prompt[:-1] + [prompt[0]]
    assert s.common_prefix(diverged) == len(prompt) - 1
    s.sync(diverged); assert s.tokens() == diverged
    s.invalidate(); assert s.common_prefix(diverged) == 0

def check_eval_rewind_and_greedy(engine, prompt):
    s = engine.create_session(len(prompt) + 64)
    s.sync(prompt)
    tok = s.argmax()
    assert s.sample(0.0, 0, 1.0, 0.0, ctypes.c_uint64(1)) == tok      # temperature 0 is greedy
    s.eval(tok); assert s.pos() == len(prompt) + 1
    s.rewind(len(prompt)); assert s.tokens() == list(prompt)
    s.sync(prompt); assert s.argmax() == tok      # rewind usually invalidates; sync rebuilds

def check_snapshot_restore_matches_rebuild(engine, prompt):
    s = engine.create_session(len(prompt) + 64)
    s.sync(prompt[:4]); snap = s.save_snapshot()
    assert snap.nbytes > 0 and s.payload_bytes() > 0
    s.sync(prompt[:2] + prompt[:2])
    s.load_snapshot(snap); snap.close()
    with pytest.raises(EngineError, match="closed"): s.load_snapshot(snap)   # and the session is unchanged
    assert s.tokens() == list(prompt[:4]) and s.checkpoint_valid()
    s.sync(prompt)
    assert s.argmax() == rebuilt_argmax(s, prompt)

def check_cancel_interrupts(engine, prompt):
    s = engine.create_session(len(prompt) + 64)
    assert s.sync(prompt, cancel=lambda: True) is SyncStatus.INTERRUPTED
    assert s.pos() < len(prompt)

def check_context_errors(engine, prompt):
    s = engine.create_session(len(prompt))
    with pytest.raises(EngineError): s.sync(prompt)           # needs one free slot
    with pytest.raises(EngineError): s.sync([])

CONTRACT = [check_extend_matches_rebuild, check_common_prefix, check_eval_rewind_and_greedy,
            check_snapshot_restore_matches_rebuild, check_cancel_interrupts, check_context_errors]
