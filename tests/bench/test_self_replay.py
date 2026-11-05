"""Exact complete replay of ds4-py traces under recorded provenance."""

import hashlib
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import ds4py
from ds4py._native.fake import FAKE_EOS, FakeNative, FakeSession
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.engine.replay import read_replay
from ds4py.engine.session import Engine

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.provenance import RoundProvenance, TraceProvenance
from bench.self_replay import (
    SelfReplayContext,
    SelfReplayResult,
    replay_ds4py_task_results,
    replay_ds4py_with_engine,
    select_ds4py_traces,
)

MODEL = "/models/fake.gguf"
TOOL_TEXT = "<tool_call><function=read><parameter=path>/tmp/x</parameter></function></tool_call>"
CONTEXT = SelfReplayContext(
    bakeoff_session_id="session-1",
    instance_id="instance-1",
    model_sha256="model",
    libds4_sha256="library",
)


def test_self_replay_task_results_uses_one_engine_and_selected_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bench.self_replay as replay

    trace = tmp_path / "ds4py.trace"
    trace.write_text("")
    item = SimpleNamespace(agent="ds4py", task_id="task", attempts=1, attempt_trace_paths=(trace,))
    provenance = {trace: cast("TraceProvenance", object())}
    engine = SimpleNamespace()
    opened = {}
    delegated = {}

    def open_engine(*args: Any, **kwargs: Any) -> Any:
        opened.update(kwargs)
        return nullcontext(engine)

    def drive(engine_arg: Any, selected: Any, **kwargs: Any) -> Any:
        delegated.update(selected=selected, provenance=kwargs["provenance"], context=kwargs["context"])
        return (SelfReplayResult("task", 0, passed=True, detail="matched"),)

    monkeypatch.setattr(ds4py, "open", open_engine)
    monkeypatch.setattr(replay, "replay_ds4py_with_engine", drive)
    config = SimpleNamespace(model=MODEL, backend="cpu", ctx=100000, seed=1)

    results = replay_ds4py_task_results(
        cast("Any", [item]),
        cast("Any", config),
        provenance=provenance,
        context=CONTEXT,
    )

    assert results[0].passed
    assert opened == {"ctx": 100000, "backend": "cpu", "lib": None, "seed": 1}
    assert delegated["selected"] == [("task", trace)]
    assert delegated["provenance"] is provenance
    assert delegated["context"] is CONTEXT


def write_ds4py_trace(path: Path, text: str, *, cached: int = 0) -> tuple[int, ...]:
    prompt = tokenize_rendered("SYS")
    sampled = tokenize_rendered(text)
    record = {
        "session_id": "session",
        "run_id": "run",
        "step_index": 0,
        "round_index": 0,
        "prompt": list(prompt),
        "think": "none",
        "cached": cached,
        "prefill_suffix": list(prompt[cached:]),
        "sampled": list(sampled),
    }
    path.write_text(json.dumps(record) + "\n")
    return tuple(sampled)


def engine_for(sampled: tuple[int, ...], mismatch_at: int | None = None) -> Engine:
    def script(prompt: Any, generated: Any) -> Any:
        index = len(generated)
        if index >= len(sampled):
            return FAKE_EOS
        if index == mismatch_at:
            return sampled[index] + 1
        return sampled[index]

    info = replace(FakeNative()._info, family="qwen", is_qwen4=True)
    return ds4py.open(MODEL, ctx=4096, native=FakeNative(info=info, script=script))


def provenance_for(path: Path, **changes: object) -> TraceProvenance:
    trace = read_replay(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    values: dict[str, object] = {
        "bakeoff_session_id": CONTEXT.bakeoff_session_id,
        "instance_id": CONTEXT.instance_id,
        "gpu_model": "gpu",
        "gpu_driver_version": "driver",
        "ds4_revision": "ds4",
        "ds4py_revision": "python",
        "libds4_sha256": CONTEXT.libds4_sha256,
        "model_path": MODEL,
        "model_sha256": CONTEXT.model_sha256,
        "backend": "cpu",
        "ctx": 4096,
        "q8_f16_cache_mb": 5120,
        "temperature": 0.0,
        "seed": 1,
        "response_budget": 8192,
        "reset_policy": "sequential",
        "trace_sha256": digest,
        "source_trace_sha256": digest,
        "rounds": tuple(RoundProvenance(round_.number, round_.think, "prompt", (), ()) for round_ in trace.rounds),
    }
    values.update(changes)
    return TraceProvenance(**values)  # ty: ignore[invalid-argument-type]


def replay(engine: Engine, task_id: str, path: Path, **changes: object) -> tuple[Any, ...]:
    provenance = provenance_for(path, **changes)
    return replay_ds4py_with_engine(
        engine,
        [(task_id, path)],
        provenance={path: provenance},
        context=CONTEXT,
    )


def test_tool_round_self_replays_every_token_without_tool_execution(tmp_path: Path) -> None:
    path = tmp_path / "tool.jsonl"
    sampled = write_ds4py_trace(path, TOOL_TEXT)
    assert len(sampled) > 16

    with engine_for(sampled) as engine:
        results = replay(engine, "tool-task", path)

    assert len(results) == 1
    result = results[0]
    assert result.passed, result.detail
    assert result.valid
    assert result.tokens == len(sampled)
    assert result.tool_call
    assert result.terminal_reason == "tool_call_complete"


def test_self_replay_preserves_cached_sync_and_resets_between_attempts(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    sampled = write_ds4py_trace(first, "hello", cached=1)
    write_ds4py_trace(second, "hello", cached=1)
    provenance = {path: provenance_for(path) for path in (first, second)}

    with engine_for(sampled) as engine:
        results = replay_ds4py_with_engine(
            engine,
            [("task", first), ("task", second)],
            provenance=provenance,
            context=CONTEXT,
        )

        def session_calls(_engine: Any, session: Any) -> list[tuple[Any, ...]]:
            assert isinstance(session, FakeSession)
            return list(session.calls)

        calls = engine._worker.call(session_calls).result()

    assert all(result.passed for result in results)
    assert calls[:2] == [("sync", 1), ("sync", 2)]
    assert calls.count(("rewind", 0)) == 1


@pytest.mark.parametrize(("text", "tool_call"), [(TOOL_TEXT, True), ("hello", False)])
def test_self_replay_mismatch_has_task_round_token_and_round_type(
    tmp_path: Path,
    text: str,
    tool_call: bool,
) -> None:
    path = tmp_path / "mismatch.jsonl"
    sampled = write_ds4py_trace(path, text)

    with engine_for(sampled, mismatch_at=1) as engine:
        results = replay(engine, "mismatch-task", path)

    assert not results[0].passed
    assert results[0].valid
    assert "mismatch-task" in results[0].detail
    assert "round 0" in results[0].detail
    assert "token 1" in results[0].detail
    assert results[0].tool_call is tool_call


@pytest.mark.parametrize(
    "changes",
    [
        {"instance_id": "other-instance"},
        {"bakeoff_session_id": "other-session"},
        {"libds4_sha256": "other-library"},
    ],
)
def test_self_replay_provenance_mismatch_is_invalid_not_token_mismatch(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    path = tmp_path / "provenance.jsonl"
    sampled = write_ds4py_trace(path, "hello")

    with engine_for(sampled) as engine:
        results = replay(engine, "task", path, **changes)

        def eval_calls(_engine: Any, session: Any) -> list[tuple[Any, ...]]:
            assert isinstance(session, FakeSession)
            return [call for call in session.calls if call[0] == "eval"]

        assert engine._worker.call(eval_calls).result() == []

    assert not results[0].passed
    assert not results[0].valid
    assert "token 0" not in results[0].detail


def test_selection_includes_every_ds4py_attempt_without_full_set_requirement(tmp_path: Path) -> None:
    tasks = []
    expected = []
    for index in range(2):
        paths = (tmp_path / f"task-{index}-1.jsonl", tmp_path / f"task-{index}-2.jsonl")
        expected.extend(paths)
        tasks.append(
            SimpleNamespace(
                task_id=f"task-{index}",
                agent="ds4py",
                attempts=2,
                attempt_trace_paths=paths,
            )
        )
    tasks.append(SimpleNamespace(task_id="task-c", agent="c", attempts=1, attempt_trace_paths=(tmp_path / "c.trace",)))

    selected = select_ds4py_traces(cast("Any", tasks))

    assert len(selected) == 4
    assert [path for _task, path in selected] == expected
