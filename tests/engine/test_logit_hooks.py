import math
import threading
from dataclasses import replace

import pytest

from ds4py._native.structs import ArgmaxContext, EffLogits, TokenScore
from ds4py.engine.decode import GenerateJob, generate
from ds4py.engine.hooks import (
    ArgmaxRecord, DisjointTokenRanges, LogitHookSet, TokenOverrides,
)
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.transcript import Transcript
from fake_steps import FakeMultiTokenStep


def test_disjoint_ranges_validate_normalize_merge_and_clear():
    ranges = DisjointTokenRanges(((8, 9), (2, 3)))
    ranges.merge((4, 5), (7, 7))
    assert ranges.ranges == ((2, 5), (7, 9))
    assert ranges.ids() == [2, 3, 4, 5, 7, 8, 9]
    assert ranges.contains(4) and not ranges.contains(6)
    ranges.clear()
    assert ranges.ranges == () and ranges.ids() == []
    for invalid in ((-1, 2), (3, 2), (1.5, 2), (True, 2), (1, 2, 3)):
        with pytest.raises((TypeError, ValueError)):
            DisjointTokenRanges((invalid,))


def test_token_overrides_validate_copy_and_clear():
    overrides = TokenOverrides(((2, 1.5), (4, -2)))
    assert overrides.items() == ((2, 1.5), (4, -2.0))
    assert overrides.ids() == [2, 4]
    overrides.clear()
    assert overrides.items() == () and overrides.ids() == []
    with pytest.raises(ValueError):
        TokenOverrides(((-1, 2),))
    with pytest.raises(TypeError):
        TokenOverrides(((True, 2),))


class RecordingSession:
    def __init__(self, tokens=(1, 0)):
        self.tokens_to_sample = list(tokens)
        self.tokens = []
        self.registrations = []
        self.clears = 0
        self.consts = []
        self.chains = []
        self.contexts = []
        self.sampling_hook = object()

    def set_logit_hooks(self, hooks): self.registrations.append(hooks)
    def clear_logit_hooks(self): self.clears += 1
    def chain_consts(self, *values):
        self.consts.append(values)
        return EffLogits()
    def set_sampling_chain(self, eff, ids, fn): self.chains.append((eff, list(ids), fn))
    def set_argmax_context(self, ctx): self.contexts.append(ctx)
    def sample(self, *args): return self.tokens_to_sample.pop(0)
    def eval(self, token): self.tokens.append(token)
    def pos(self): return len(self.tokens)
    def ctx(self): return 100


class HookEngine:
    def is_stop(self, token): return token == 0
    def token_text(self, token): return bytes([token])


def run_with_hooks(session, hooks, step=None):
    transcript = Transcript()
    job = GenerateJob(
        SamplingPolicy(), Rng(1), 10, transcript, hooks=hooks, stop_sequences=[b"x"],
    )
    reason = generate(
        HookEngine(), session, job, cancelled=lambda: False, emit=lambda event: None,
        step=step,
    )
    return reason, transcript


def test_plain_step_prepares_hooks_for_every_sample_and_always_clears():
    enabled = threading.Event()
    enabled.set()
    hooks = LogitHookSet(
        enabled=enabled,
        allow=DisjointTokenRanges(((2, 3),)),
        overrides=TokenOverrides(((4, 2.0),)),
    )
    session = RecordingSession()
    reason, transcript = run_with_hooks(session, hooks)
    assert reason == "stop_token" and transcript.tokens == (1,)
    assert session.registrations == [hooks] and session.clears == 1
    assert len(session.consts) == len(session.chains) == len(session.contexts) == 2
    assert session.consts[0] == (1.0, 0, 0.05, 1.0)
    assert session.chains[0][1:] == ([2, 3, 4], session.sampling_hook)
    assert session.contexts[0].cap == 3 and session.contexts[0].collect == 0


def test_multi_token_step_does_not_run_per_sample_hook_setup():
    hooks = LogitHookSet()
    session = RecordingSession()
    reason, _ = run_with_hooks(session, hooks, FakeMultiTokenStep([[]]))
    assert reason == "stop_token"
    assert session.registrations == [hooks] and session.clears == 1
    assert session.consts == session.chains == session.contexts == []


def test_disabling_event_preserves_constraints_and_effective_policy():
    enabled = threading.Event()
    enabled.set()
    hooks = LogitHookSet(
        enabled=enabled, allow=DisjointTokenRanges(((1, 2),)),
        overrides=TokenOverrides(((3, 4.0),)), repr_only_thinking=False,
    )
    session = RecordingSession()
    policy = SamplingPolicy(temperature=0.7, top_k=5, top_p=0.8, min_p=0.02)
    hooks.before_sample(session, policy, thinking=False)
    enabled.clear()
    hooks.before_sample(session, policy, thinking=False)
    assert not hooks.active and hooks.effective_policy == policy
    assert hooks.allow.ranges == ((1, 2),) and hooks.overrides.ids() == [3]
    assert session.chains[-1][2] is None and session.contexts[-1].collect == 0
    ungated = replace(hooks, enabled=None)
    assert ungated.active and ungated.allow is hooks.allow
    assert ungated.overrides is hooks.overrides and ungated.effective_policy == policy


def test_repr_collection_defaults_to_thinking_but_can_be_enabled_elsewhere():
    thinking_only = LogitHookSet()
    everywhere = LogitHookSet(repr_only_thinking=False)
    for hooks, thinking, expected in (
        (thinking_only, False, 0), (thinking_only, True, 1), (everywhere, False, 1),
    ):
        session = RecordingSession()
        hooks.before_sample(session, SamplingPolicy(), thinking=thinking)
        assert session.contexts[-1].collect == expected


def make_eff(values):
    import ctypes as C
    from ds4py._native.structs import Logits
    data = (C.c_float * len(values))(*values)
    logits = Logits(C.cast(data, C.POINTER(C.c_float)), len(values), len(values))
    eff = EffLogits(C.pointer(logits), C.cast(data, C.POINTER(C.c_uint8)))
    return data, logits, eff


def test_sampling_applies_range_mask_then_overrides_and_masks_candidates():
    import ctypes as C
    data, logits, eff = make_eff([1, 2, 3, 4, 5, 6])
    hooks = LogitHookSet(
        allow=DisjointTokenRanges(((2, 3),)),
        overrides=TokenOverrides(((4, 9.5),)),
    )
    candidates = (C.c_int * 4)(1, 2, 4, 5)
    hooks.sampling(C.pointer(eff), candidates, 4)
    assert math.isinf(data[0]) and data[0] < 0 and math.isinf(data[1])
    assert list(data)[2:5] == [3, 4, 9.5]
    assert math.isinf(data[5]) and data[5] < 0
    assert list(candidates) == [-1, 2, 4, -1]


def test_argmax_collect_copies_payload_and_noncollect_clears_records():
    import ctypes as C
    hooks = LogitHookSet()
    records = (TokenScore * 2)((7, 1.25, -0.5), (8, 2.5, -0.25))
    context = ArgmaxContext(records, 2, 1)
    hooks.argmax(C.pointer(context))
    assert hooks.argmax_records == (
        ArgmaxRecord(7, pytest.approx(1.25), pytest.approx(-0.5)),
        ArgmaxRecord(8, pytest.approx(2.5), pytest.approx(-0.25)),
    )
    records[0].id = 99
    assert hooks.argmax_records[0].id == 7
    context.collect = 0
    hooks.argmax(C.pointer(context))
    assert hooks.argmax_records == ()
