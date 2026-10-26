from pathlib import Path

import pytest

from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py._native.protocol import EngineConfig, ThinkMode
from ds4py.engine.decode import GenerateJob, generate
from ds4py.engine.events import Stopped, Token, ToolCalls
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.stream import RoundStream
from ds4py.engine.transcript import Transcript
from ds4py.engine.worker import EngineWorker, SyncJob
from ds4py.errors import EngineError
from fake_steps import FakeMultiTokenStep, block_chunks

DSML = ("<｜DSML｜tool_calls><｜DSML｜invoke name=\"read\">"
        "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/x</｜DSML｜parameter>"
        "</｜DSML｜invoke></｜DSML｜tool_calls>")


def scripted(tokens):
    tokens = tuple(tokens)
    return lambda prompt, generated: tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS


@pytest.fixture
def workers():
    created = []
    yield created
    for worker in created:
        worker.close()


def run_step(workers, tokens, blocks, *, max_tokens=1000, chat=False, logits_hooks=(), step_type=FakeMultiTokenStep, cancelled=lambda: False):
    worker = EngineWorker(
        FakeNative(script=scripted(tokens)), EngineConfig(Path("f"), ctx=1024),
    )
    workers.append(worker)
    transcript = Transcript()
    transcript.append([5, 6], "user")
    list(worker.submit(SyncJob((5, 6))))
    stream = RoundStream(worker.info, ThinkMode.NONE) if chat else None
    job = GenerateJob(
        SamplingPolicy(), Rng(1), max_tokens, transcript,
        think_mode=ThinkMode.NONE if chat else None, stream=stream,
        logits_hooks=logits_hooks,
    )
    step = step_type(blocks)
    events, token_positions = [], []

    def execute(engine, session):
        def emit(event):
            events.append(event)
            if isinstance(event, Token):
                token_positions.append((event.id, session.pos(), len(transcript)))
        reason = generate(
            engine, session, job, cancelled=cancelled, emit=emit, step=step,
        )
        events.append(Stopped(reason))
        return reason

    error = None
    try:
        reason = worker.call(execute).result()
    except EngineError as exc:
        reason, error = None, exc
    return reason, events, transcript, worker, step, token_positions, error


def token_ids(events):
    return [event.id for event in events if isinstance(event, Token)]


def test_first_stop_token_has_no_eval_or_commit(workers):
    reason, events, transcript, worker, step, positions, error = run_step(workers, [0], [])
    assert error is None and reason == "stop_token" and events == [Stopped("stop_token")]
    assert transcript.tokens == (5, 6) and positions == []
    assert worker.call(lambda e, s: s.pos()).result() == 2
    assert not [call for call in worker.call(lambda e, s: s.calls).result() if call[0] == "eval"]


def test_later_stop_rewinds_before_publishing_kept_tokens(workers):
    a, b, later = 98, 99, 100
    reason, events, transcript, worker, step, positions, error = run_step(
        workers, [a, b, 0, later], [[b, 0, later]],
    )
    assert error is None and reason == "stop_token"
    assert token_ids(events) == [a, b] and transcript.tokens == (5, 6, a, b)
    assert worker.call(lambda e, s: s.pos()).result() == len(transcript) == 4
    assert ("rewind", 4) in worker.call(lambda e, s: s.calls).result()
    assert all(pos == length for _, pos, length in positions)


def test_budget_stop_discards_remaining_block(workers):
    tokens = [98, 99, 100, 101]
    reason, events, transcript, worker, step, positions, error = run_step(
        workers, tokens, [tokens[1:]], max_tokens=2,
    )
    assert error is None and reason == "max_tokens"
    assert token_ids(events) == tokens[:2] and transcript.tokens == (5, 6, *tokens[:2])
    assert worker.call(lambda e, s: s.pos()).result() == len(transcript) == 4
    assert all(pos == length for _, pos, length in positions)


def test_tool_boundary_discards_later_drafts(workers):
    stanza = tokenize_rendered(DSML)
    tokens = stanza + tokenize_rendered("TAIL")
    reason, events, transcript, worker, step, positions, error = run_step(
        workers, tokens, block_chunks(tokens), chat=True,
    )
    assert error is None and reason == "tool_call_complete"
    assert token_ids(events) == stanza
    assert transcript.tokens == (5, 6, *stanza)
    assert worker.call(lambda e, s: s.pos()).result() == len(transcript)
    assert len([event for event in events if isinstance(event, ToolCalls)]) == 1
    assert all(pos == length for _, pos, length in positions)


def test_drafts_are_capped_at_sixteen_and_sampling_resumes(workers):
    tokens = list(range(1, 41))
    blocks = [tokens[1:], *block_chunks(tokens[17:])]
    reason, events, transcript, worker, step, positions, error = run_step(workers, tokens, blocks)
    assert error is None and reason == "stop_token"
    assert token_ids(events) == tokens and transcript.tokens == (5, 6, *tokens)
    assert step.block_sizes == [17, 17, 6]
    assert worker.call(lambda e, s: s.pos()).result() == len(transcript)


def test_empty_engine_block_raises_without_new_events(workers):
    reason, events, transcript, worker, step, positions, error = run_step(workers, [98], [None])
    assert reason is None and isinstance(error, EngineError)
    assert str(error) == "decode returned no tokens"
    assert events == [] and transcript.tokens == (5, 6) and positions == []
    assert worker.call(lambda e, s: s.pos()).result() == 2


def test_empty_block_after_kept_block_raises_without_changing_position(workers):
    reason, events, transcript, worker, step, positions, error = run_step(
        workers, [98, 99], [[], None],
    )
    assert reason is None and isinstance(error, EngineError)
    assert str(error) == "decode returned no tokens"
    assert token_ids(events) == [98] and transcript.tokens == (5, 6, 98)
    assert worker.call(lambda e, s: s.pos()).result() == len(transcript) == 3


def test_logits_hooks_fall_back_to_plain_step(workers):
    class NonconformingStep(FakeMultiTokenStep):
        hooks_every_sample = False

    reason, events, transcript, _worker, step, _positions, error = run_step(
        workers, [98, 99], [[99]], logits_hooks=(lambda view: None,),
        step_type=NonconformingStep,
    )
    assert error is None and reason == "stop_token"
    assert token_ids(events) == [98, 99] and transcript.tokens == (5, 6, 98, 99)
    assert step.block_sizes == []


def test_no_logits_hooks_keep_the_multi_token_step(workers):
    reason, events, transcript, _worker, step, _positions, error = run_step(
        workers, [98, 99], [[99]],
    )
    assert error is None and reason == "stop_token"
    assert token_ids(events) == [98, 99] and transcript.tokens == (5, 6, 98, 99)
    assert step.block_sizes == [2]


def test_hook_changed_draft_cuts_block_and_hooks_see_each_committed_prefix(workers):
    seen = []

    def hook(view):
        seen.append((view.position, view.sampled_tokens))
        if view.sampled_tokens == (98,):
            view.bias(100, 2.0)

    reason, events, transcript, worker, step, _, error = run_step(
        workers, [98, 99, 101], [[99, 101]], logits_hooks=(hook,),
    )
    assert error is None and reason == "stop_token"
    assert token_ids(events) == [98, 100, 101]
    assert transcript.tokens == (5, 6, 98, 100, 101)
    assert seen == [(2, ()), (3, (98,)), (4, (98, 100)), (5, (98, 100, 101))]
    assert step.block_sizes == [2, 1]
    assert worker.call(lambda e, s: s.pos()).result() == len(transcript)


def test_cancellation_mid_block_rewinds_the_uncommitted_tail(workers):
    checks = iter([False, False, False, True])
    reason, events, transcript, worker, _, positions, error = run_step(
        workers, [98, 99, 100, 101], [[99, 100, 101]],
        cancelled=lambda: next(checks, True),
    )
    assert error is None and reason == 'interrupted'
    assert token_ids(events) == [98, 99] and transcript.tokens == (5, 6, 98, 99)
    assert worker.call(lambda e, s: s.pos()).result() == len(transcript)
    assert ('rewind', 4) in worker.call(lambda e, s: s.calls).result()
    assert all(pos == length for _, pos, length in positions)
