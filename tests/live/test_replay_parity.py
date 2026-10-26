"""Opt-in, read-only chat parity against token-level ds4-agent traces.

The C trace owns every next prompt. No ds4-py tool is executed, and no Python
self-feeding can introduce a token that was not present in the C trace.
"""
import os
from pathlib import Path

import pytest

import ds4py
from ds4py.engine.decode import GenerateJob
from ds4py.engine.events import Stopped, Token
from ds4py.engine.replay import read_replay
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.stream import RoundStream
from ds4py.engine.tool_syntax import make_parser
from ds4py.engine.transcript import Transcript
from ds4py.engine.worker import SyncJob


def _parse_c(engine, round, chunks):
    parser = make_parser(engine.info)
    mode = int(round.think)
    parser.in_think = mode in (1, 2, 3, 4) or 1001 <= mode <= 1100
    streamed = False
    for chunk in chunks:
        streamed = True
        parser.feed(chunk)
    if not streamed:
        parser.feed(b"")
    return parser.finish()


def _token_description(engine, tokens, index):
    if index >= len(tokens):
        return "id=<missing> bytes=<missing>"
    data = engine.token_text(tokens[index])
    return f"id={tokens[index]} bytes={data!r}"


def _assert_tokens(engine, ordinal, round, actual, source):
    expected = round.sampled
    if actual == expected:
        return
    index = next(
        (i for i, (left, right) in enumerate(zip(expected, actual)) if left != right),
        min(len(expected), len(actual)),
    )
    raise AssertionError(
        f"{source} mismatch at trace round {ordinal} (tool_round={round.number}), "
        f"token {index}; prompt={len(round.prompt)} "
        f"C sampled={len(expected)} ds4py={len(actual)}\n"
        f"C:      {_token_description(engine, expected, index)}\n"
        f"ds4py:  {_token_description(engine, actual, index)}"
    )


def _check_round(engine, ordinal, round, events, transcript):
    assert events and isinstance(events[-1], Stopped), (
        f"trace round {ordinal} did not drain to a Stopped event"
    )
    emitted = tuple(event.id for event in events if isinstance(event, Token))
    additions = transcript.tokens[len(round.prompt):]
    _assert_tokens(engine, ordinal, round, emitted, "emitted token")
    _assert_tokens(engine, ordinal, round, additions, "transcript addition")

    data = b"".join(engine.token_text(token) for token in round.sampled)
    c_snapshot = _parse_c(engine, round, [data])
    stopped = events[-1]
    if not round.prefix_only and not round.context_limited:
        assert stopped.reason in ("stop_token", "tool_call_complete", "parser_error"), (
            f"trace round {ordinal} (tool_round={round.number}) did not stop "
            f"naturally at C's boundary: {stopped.reason}"
        )
        if stopped.reason == "tool_call_complete":
            assert c_snapshot.state == "DONE" and not c_snapshot.dsml_in_think, (
                f"trace round {ordinal}: ds4py reported tool_call_complete, "
                f"but C's sampled text parsed as {c_snapshot.state}: {c_snapshot.error}"
            )
        elif stopped.reason == "parser_error":
            assert c_snapshot.state == "ERROR" or c_snapshot.dsml_in_think, (
                f"trace round {ordinal}: ds4py reported parser_error, "
                f"but C's sampled text parsed as {c_snapshot.state}"
            )

    if c_snapshot.state != "DONE" or not c_snapshot.calls:
        return
    bytewise = _parse_c(
        engine, round, (data[index:index + 1] for index in range(len(data)))
    )
    assert bytewise == c_snapshot, f"trace round {ordinal}: bytewise parser mismatch"
    for split in range(len(data) + 1):
        observed = _parse_c(engine, round, [data[:split], data[split:]])
        assert observed == c_snapshot, (
            f"trace round {ordinal}: completed tool stanza differs at byte split {split}"
        )


def _run_round(engine, policy, rng, ordinal, round):
    reserve = 1
    bounded = round.prefix_only or round.context_limited
    if round.context_limited:
        reserve = engine.config.ctx - len(round.prompt) - len(round.sampled)
        if reserve < 0:
            raise AssertionError(
                f"trace round {ordinal} has negative derived context reserve {reserve}"
            )
    # Deliberately no preflight: C's prefix and sampled token count are the
    # parity contract even when native would finish an old edit differently.
    max_tokens = len(round.sampled) if bounded else len(round.sampled) + 1

    transcript = Transcript()
    if round.prompt:
        transcript.append(round.prompt, "system")
    list(engine._worker.submit(SyncJob(round.prompt)))
    stream = RoundStream(engine.info, round.think)
    events = list(engine._worker.submit(GenerateJob(
        policy, rng, max_tokens, transcript,
        reserve=reserve, think_mode=round.think, stream=stream,
        control=stream.control,
    )))
    # For a C-owned context/edit boundary, an unfinished parser at the token
    # limit is not itself a parity failure; only the partial prefix must match.
    _check_round(engine, ordinal, round, events, transcript)


def _replay_trace(engine, trace):
    # The documented capture command uses --temp 0 --seed 1. Keep one RNG
    # across rounds, as ds4-agent does, even though the default is greedy.
    policy = SamplingPolicy.for_engine(engine.info, temperature=0.0, seed=1)
    rng = Rng(1)
    for ordinal, round in enumerate(trace.rounds, 1):
        _run_round(engine, policy, rng, ordinal, round)


@pytest.mark.live
def test_chat_matches_c_trace_without_executing_tools(request):
    model = os.environ.get("DS4_TEST_MODEL")
    trace_path = os.environ.get("DS4_REPLAY_TRACE")
    if not model or not trace_path:
        pytest.skip("set DS4_TEST_MODEL and DS4_REPLAY_TRACE for live replay parity")
    live_lib = request.getfixturevalue("live_lib")
    trace = read_replay(trace_path)
    traced_model = Path(trace.model).expanduser().resolve()
    selected_model = Path(model).expanduser().resolve()
    assert traced_model == selected_model, (
        f"trace model {traced_model} != DS4_TEST_MODEL {selected_model}"
    )
    with ds4py.open(selected_model, ctx=trace.ctx, lib=live_lib, seed=1) as engine:
        _replay_trace(engine, trace)
