"""Gate 6 replays complete C traces without executing tools."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import ds4py
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.replay_c_traces import (
    ReplayResult,
    generation_limit,
    replay_with_engine,
    select_c_traces,
)
from bench.report import build_report, update_replay_report
from bench.restore_check import RestoreSummary


MODEL = "/models/fake.gguf"
STAMP = "2026-10-21 09:05:07.042 "
TOOL_TEXT = (
    "<tool_call><function=read><parameter=path>/tmp/x</parameter>"
    "</function></tool_call>"
)


def write_c_trace(path, text):
    prompt = tokenize_rendered("SYS")
    sampled = tokenize_rendered(text)
    lines = [
        STAMP + f"agent worker start ctx=4096 backend=cpu model={MODEL} trace=c.trace",
        STAMP + f"tokens label=initial_system_prompt start=0 len={len(prompt)}",
    ]
    lines.extend(STAMP + f"token index={index} id={token}" for index, token in enumerate(prompt))
    lines.extend([
        STAMP + f"prefill tool_round=0 transcript={len(prompt)} prompt={len(prompt)} cached=0 suffix={len(prompt)} think=none",
        STAMP + f"tokens label=prefill_suffix start=0 len={len(prompt)}",
    ])
    lines.extend(STAMP + f"token index={index} id={token}" for index, token in enumerate(prompt))
    lines.extend(
        STAMP + f"token index={index} id={token}"
        for index, token in enumerate(sampled, 1)
    )
    lines.append(
        STAMP + f"generation finished tool_round=0 generated={len(sampled)} carried=0 context_limited=0"
    )
    path.write_text("\n".join(lines) + "\n")
    return sampled


def engine_for(sampled, mismatch_at=None):
    def script(prompt, generated):
        index = len(generated)
        if index >= len(sampled):
            return FAKE_EOS
        if index == mismatch_at:
            return sampled[index] + 1
        return sampled[index]

    info = replace(FakeNative()._info, family="qwen", is_qwen4=True)
    return ds4py.open(MODEL, ctx=4096, native=FakeNative(info=info, script=script))


def test_tool_round_replays_every_token_and_stops_without_tool_execution(tmp_path):
    path = tmp_path / "tool.trace"
    sampled = write_c_trace(path, TOOL_TEXT)
    assert len(sampled) > 16
    with engine_for(sampled) as engine:
        assert generation_limit(engine, type("Round", (), {
            "sampled": sampled, "prefix_only": False, "context_limited": False,
            "prompt": (1, 2),
        })()) == len(sampled) + 1
        results = replay_with_engine(
            engine, [("tool-task", path)], require_full_set=False,
        )

    assert len(results) == 1
    result = results[0]
    assert result.passed, result.detail
    assert result.tokens == len(sampled)
    assert result.tool_call
    assert result.terminal_reason == "tool_call_complete"


def test_token_mismatch_fails_with_task_round_and_token_index(tmp_path):
    path = tmp_path / "mismatch.trace"
    sampled = write_c_trace(path, TOOL_TEXT)

    with engine_for(sampled, mismatch_at=1) as engine:
        results = replay_with_engine(
            engine, [("mismatch-task", path)], require_full_set=False,
        )

    assert not results[0].passed
    assert "mismatch-task" in results[0].detail
    assert "round 0" in results[0].detail
    assert "token 1" in results[0].detail


def test_malformed_c_record_becomes_failing_gate_evidence(tmp_path):
    path = tmp_path / "malformed.trace"
    write_c_trace(path, "hello")
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")

    with engine_for(tokenize_rendered("hello")) as engine:
        results = replay_with_engine(
            engine, [("broken-task", path)], require_full_set=False,
        )

    assert not results[0].passed
    assert "broken-task" in results[0].detail
    assert "round -1" in results[0].detail
    assert str(path) in results[0].detail


def test_all_twelve_c_runs_replay_and_tool_coverage_is_required(tmp_path):
    sampled = None
    selected = []
    for index in range(12):
        path = tmp_path / f"task-{index}.trace"
        sampled = write_c_trace(path, TOOL_TEXT)
        selected.append((f"task-{index}", path))

    with engine_for(sampled) as engine:
        results = replay_with_engine(engine, selected)

    assert len(results) == 12
    assert all(result.passed for result in results)
    assert sum(result.tool_call for result in results) == 12


def test_full_set_without_a_tool_round_fails_gate_six(tmp_path):
    selected = []
    for index in range(12):
        path = tmp_path / f"text-{index}.trace"
        sampled = write_c_trace(path, "hello")
        selected.append((f"task-{index}", path))

    with engine_for(sampled) as engine:
        results = replay_with_engine(engine, selected)

    assert not results[-1].passed
    assert "no tool-call round" in results[-1].detail
    assert not build_report([], RestoreSummary(), results).gates["gate-6"].passed


def test_selection_includes_every_c_attempt_trace_and_requires_twelve_tasks(tmp_path):
    paths = []
    for index in range(12):
        first = tmp_path / f"task-{index}-1.trace"
        second = tmp_path / f"task-{index}-2.trace"
        write_c_trace(first, "first")
        write_c_trace(second, "second")
        paths.append((first, second))
    tasks = [
        SimpleNamespace(
            task_id=f"task-{index}", agent="c", attempts=2,
            attempt_trace_paths=paths[index], trace_path=paths[index][-1],
        )
        for index in range(12)
    ]
    tasks.append(SimpleNamespace(
        task_id="task-0", agent="ds4py", attempts=1,
        attempt_trace_paths=(tmp_path / "python.jsonl",), trace_path=tmp_path / "python.jsonl",
    ))

    selected = select_c_traces(tasks)

    assert len(selected) == 24
    assert [path for _task, path in selected] == [path for pair in paths for path in pair]


def test_gate_six_cannot_pass_without_tool_call_evidence():
    result = ReplayResult("task", 0, True, "text replay", tool_call=False)
    assert not build_report([], RestoreSummary(), [result]).gates["gate-6"].passed
    covered = ReplayResult("task", 0, True, "tool replay", tool_call=True)
    assert build_report([], RestoreSummary(), [covered]).gates["gate-6"].passed


def test_standalone_replay_updates_only_its_evidence_and_gate(tmp_path):
    path = tmp_path / "bakeoff.json"
    original = build_report([], RestoreSummary())
    original.write_json(path)
    gate1 = json.loads(path.read_text())["gates"]["gate-1"]
    result = ReplayResult("task", 0, True, "tool replay", tool_call=True)

    gate = update_replay_report(path, [result])

    data = json.loads(path.read_text())
    assert gate.passed
    assert data["gates"]["gate-1"] == gate1
    assert data["gates"]["gate-6"]["passed"]
    assert data["replay"][0]["task_id"] == "task"
    assert data["replay"][0]["tool_call"]
