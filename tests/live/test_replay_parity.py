"""Opt-in, read-only chat parity against token-level ds4-agent traces.

The C trace owns every next prompt. No ds4-py tool is executed, and no Python
self-feeding can introduce a token that was not present in the C trace.
"""
import os
from pathlib import Path
import sys

import pytest

import ds4py
from ds4py.engine.replay import read_replay

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.replay_m3 import replay_trace


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
        replay_trace(engine, trace)
