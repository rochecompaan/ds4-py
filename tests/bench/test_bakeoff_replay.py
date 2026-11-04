"""Exact C replay remains diagnostic evidence, not Gate 6 acceptance."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import ds4py
from ds4py._native.fake import FAKE_EOS, FakeNative, FakeSession
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.engine.session import Engine

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
TOOL_TEXT = "<tool_call><function=read><parameter=path>/tmp/x</parameter></function></tool_call>"


def write_c_trace(path: Any, text: Any) -> Any:
    prompt = tokenize_rendered("SYS")
    sampled = tokenize_rendered(text)
    lines = [
        STAMP + f"agent worker start ctx=4096 backend=cpu model={MODEL} trace=c.trace",
        STAMP + f"tokens label=initial_system_prompt start=0 len={len(prompt)}",
    ]
    lines.extend(STAMP + f"token index={index} id={token}" for index, token in enumerate(prompt))
    lines.extend(
        [
            STAMP
            + f"prefill tool_round=0 transcript={len(prompt)} prompt={len(prompt)} cached=0 suffix={len(prompt)} think=none",
            STAMP + f"tokens label=prefill_suffix start=0 len={len(prompt)}",
        ]
    )
    lines.extend(STAMP + f"token index={index} id={token}" for index, token in enumerate(prompt))
    lines.extend(STAMP + f"token index={index} id={token}" for index, token in enumerate(sampled, 1))
    lines.append(STAMP + f"generation finished tool_round=0 generated={len(sampled)} carried=0 context_limited=0")
    path.write_text("\n".join(lines) + "\n")
    return sampled


def engine_for(sampled: Any, mismatch_at: Any = None) -> Engine:
    def script(prompt: Any, generated: Any) -> Any:
        index = len(generated)
        if index >= len(sampled):
            return FAKE_EOS
        if index == mismatch_at:
            return sampled[index] + 1
        return sampled[index]

    info = replace(FakeNative()._info, family="qwen", is_qwen4=True)
    return ds4py.open(MODEL, ctx=4096, native=FakeNative(info=info, script=script))


def test_tool_round_diagnostic_replays_tokens_without_tool_execution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool.trace"
    sampled = write_c_trace(path, TOOL_TEXT)
    assert len(sampled) > 16
    with engine_for(sampled) as engine:
        assert (
            generation_limit(
                engine,
                cast(
                    "Any",
                    type(
                        "Round",
                        (),
                        {
                            "sampled": sampled,
                            "prefix_only": False,
                            "context_limited": False,
                            "prompt": (1, 2),
                        },
                    ),
                )(),
            )
            == len(sampled) + 1
        )
        results = replay_with_engine(
            engine,
            [("tool-task", path)],
            require_full_set=False,
        )

    assert len(results) == 1
    result = results[0]
    assert result.passed, result.detail
    assert result.tokens == len(sampled)
    assert result.tool_call
    assert result.terminal_reason == "tool_call_complete"


def test_replay_preserves_c_cached_prefill_boundary(tmp_path: Path) -> None:
    system = tokenize_rendered("SYS")
    suffix = tokenize_rendered("TASK")
    sampled = tokenize_rendered("hello")
    prompt = system + suffix
    path = tmp_path / "cached.trace"
    lines = [
        STAMP + f"agent worker start ctx=4096 backend=cpu model={MODEL} trace=c.trace",
        STAMP + f"tokens label=initial_system_prompt start=0 len={len(system)}",
    ]
    lines.extend(STAMP + f"token index={index} id={token}" for index, token in enumerate(system))
    lines.extend(
        [
            STAMP
            + f"prefill tool_round=0 transcript={len(prompt)} prompt={len(prompt)} "
            + f"cached={len(system)} suffix={len(suffix)} think=none",
            STAMP + f"tokens label=prefill_suffix start={len(system)} len={len(prompt)}",
        ]
    )
    lines.extend(STAMP + f"token index={index} id={token}" for index, token in enumerate(suffix, len(system)))
    lines.extend(STAMP + f"token index={index} id={token}" for index, token in enumerate(sampled, 1))
    lines.append(STAMP + f"generation finished tool_round=0 generated={len(sampled)} carried=0 context_limited=0")
    path.write_text("\n".join(lines) + "\n")

    with engine_for(sampled) as engine:
        results = replay_with_engine(
            engine,
            [("cached-task", path)],
            require_full_set=False,
        )

        def session_calls(_engine: Any, session: Any) -> list[tuple[Any, ...]]:
            assert isinstance(session, FakeSession)
            return list(session.calls)

        calls = engine._worker.call(session_calls).result()
        native_tokens = engine._worker.call(lambda _engine, session: session.tokens()).result()

    assert results[0].passed, results[0].detail
    assert calls[:2] == [("sync", len(system)), ("sync", len(suffix))]
    assert native_tokens[: len(prompt)] == prompt


def test_malformed_c_record_becomes_failing_gate_evidence(tmp_path: Path) -> None:
    path = tmp_path / "malformed.trace"
    write_c_trace(path, "hello")
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")

    with engine_for(tokenize_rendered("hello")) as engine:
        results = replay_with_engine(
            engine,
            [("broken-task", path)],
            require_full_set=False,
        )

    assert not results[0].passed
    assert "broken-task" in results[0].detail
    assert "round -1" in results[0].detail
    assert str(path) in results[0].detail


def test_all_twelve_c_runs_replay_and_tool_coverage_is_required(tmp_path: Path) -> None:
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


def test_full_set_without_a_tool_round_is_a_failed_diagnostic(tmp_path: Path) -> None:
    sampled = None
    selected = []
    for index in range(12):
        path = tmp_path / f"text-{index}.trace"
        sampled = write_c_trace(path, "hello")
        selected.append((f"task-{index}", path))

    with engine_for(sampled) as engine:
        results = replay_with_engine(engine, selected)

    assert not results[-1].passed
    assert "no tool-call round" in results[-1].detail


def test_selection_includes_every_c_attempt_without_full_set_requirement(
    tmp_path: Path,
) -> None:
    paths = []
    for index in range(2):
        first = tmp_path / f"task-{index}-1.trace"
        second = tmp_path / f"task-{index}-2.trace"
        write_c_trace(first, "first")
        write_c_trace(second, "second")
        paths.append((first, second))
    tasks = [
        SimpleNamespace(
            task_id=f"task-{index}",
            agent="c",
            attempts=2,
            attempt_trace_paths=paths[index],
            trace_path=paths[index][-1],
        )
        for index in range(2)
    ]
    tasks.append(
        SimpleNamespace(
            task_id="task-0",
            agent="ds4py",
            attempts=1,
            attempt_trace_paths=(tmp_path / "python.jsonl",),
            trace_path=tmp_path / "python.jsonl",
        )
    )

    selected = select_c_traces(cast("Any", tasks))

    assert len(selected) == 4
    assert [path for _task, path in selected] == [path for pair in paths for path in pair]


def test_c_diagnostic_cannot_pass_gate_six() -> None:
    covered = ReplayResult("task", 0, True, "tool replay", tool_call=True)
    gate = build_report([], RestoreSummary(), [covered]).gates["gate-6"]
    assert not gate.passed
    assert not gate.valid


def test_standalone_replay_updates_diagnostic_and_preserves_gates(tmp_path: Path) -> None:
    path = tmp_path / "bakeoff.json"
    original = build_report([], RestoreSummary())
    original.write_json(path)
    original_gates = json.loads(path.read_text())["gates"]
    result = ReplayResult("task", 0, True, "tool replay", tool_call=True)

    diagnostic = update_replay_report(path, [result])

    data = json.loads(path.read_text())
    assert diagnostic.passed
    assert data["gates"] == original_gates
    assert data["replay_diagnostic"][0]["task_id"] == "task"
    assert data["replay_diagnostic"][0]["tool_call"]
