import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.engine.events import Stopped, Synced, Token, ToolCalls
from ds4py.engine.lifecycle import SessionLifecycle
from ds4py.engine.plan import Plan
from ds4py.engine.replay import ReplayError, read_replay
from ds4py.prefill_trace import PrefillTraceWriter

TOOL_TEXT = "<tool_call><function=read><parameter=path>/tmp/x</parameter></function></tool_call>"


def scripted_rounds(*texts: Any) -> Any:
    responses = [tuple(tokenize_rendered(text)) for text in texts]
    state = {"round": 0, "generated": 0}

    def script(prompt: Any, generated: Any) -> Any:
        if not generated and state["generated"]:
            state["round"] += 1
        state["generated"] = len(generated)
        response = responses[state["round"]]
        return response[len(generated)] if len(generated) < len(response) else FAKE_EOS

    return script


def test_text_and_tool_rounds_append_replayable_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "prefill.jsonl"
    info = replace(FakeNative()._info, family="qwen", is_qwen4=True)
    native = FakeNative(info=info, script=scripted_rounds("hello", TOOL_TEXT))
    observed = []

    with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=native) as engine:
        session = engine.session()

        def observe(event: Any) -> None:
            if event.kind == "Synced":
                synced = event.payload["event"]
                observed.append((tuple(session.transcript.tokens), synced.plan.reused))

        session.on_event(observe)
        monkeypatch.setattr(
            session,
            "append_tool_results",
            lambda results: (_ for _ in ()).throw(AssertionError("trace executed a tool")),
        )
        with PrefillTraceWriter(session, path, think=engine.think) as writer:
            session.start_chat()
            session.emit("run_start", run_id="run-text")
            text_events = list(session.run_round())
            session.emit("run_end", run_id="run-text", status="success")
            session.append_user("use the tool")
            session.emit("run_start", run_id="run-tool")
            tool_events = list(session.run_round())
            session.emit("run_end", run_id="run-tool", status="success")
            assert writer.round_index == 2

    assert any(isinstance(event, ToolCalls) for event in tool_events)
    trace = read_replay(path)
    assert trace.ctx is None and trace.model is None
    assert len(trace.rounds) == 2
    assert len(path.read_text().splitlines()) == 2

    for index, (round_, events, (prompt, cached)) in enumerate(
        zip(
            trace.rounds,
            (text_events, tool_events),
            observed,
            strict=True,
        )
    ):
        sampled = tuple(event.id for event in events if isinstance(event, Token))
        assert round_.number == index
        assert round_.prompt == prompt
        assert round_.cached == cached
        assert round_.sampled == sampled
        assert round_.think == ThinkMode.NONE
        assert not round_.context_limited and not round_.prefix_only

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["run_id"] for record in records] == ["run-text", "run-tool"]
    assert [record["step_index"] for record in records] == [0, 0]
    assert [record["round_index"] for record in records] == [0, 1]
    for record, round_ in zip(records, trace.rounds, strict=True):
        assert tuple(record["prefill_suffix"]) == round_.prompt[round_.cached :]


class TraceSession(SessionLifecycle):
    def __init__(self, tokens: Any) -> None:
        super().__init__()
        self.transcript = SimpleNamespace(tokens=tokens)


def synced(pos: Any, reused: Any = 0) -> Synced:
    return Synced(pos, Plan("extend", reused, pos - reused, True, None), 0)


def emit_round(session: Any, run_id: Any, *, fail: Any = False, repeated_sync: Any = False) -> None:
    session.emit("run_start", run_id=run_id)
    session.emit("Synced", event=synced(2, 1))
    session.emit("Token", event=Token(3, b"a"))
    if repeated_sync:
        session.emit("Synced", event=synced(3, 2))
        session.emit("Token", event=Token(4, b"b"))
    if fail:
        session.emit("run_end", run_id=run_id, status="error")
    else:
        session.emit("Stopped", event=Stopped("stop_token"))
        session.emit("run_end", run_id=run_id, status="success")


def test_reopening_trace_resumes_file_wide_round_numbering(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    session: Any = TraceSession([1, 2, 3, 4])
    for run_id in ("first", "second"):
        with PrefillTraceWriter(session, path, think=1):
            emit_round(session, run_id)

    trace = read_replay(path)
    assert [round_.number for round_ in trace.rounds] == [0, 1]


def test_reopening_trace_without_final_newline_preserves_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(VALID))
    session: Any = TraceSession([1, 2, 3, 4])

    with PrefillTraceWriter(session, path, think=1):
        emit_round(session, "second")

    trace = read_replay(path)
    assert [round_.number for round_ in trace.rounds] == [0, 1]


def test_writer_rejects_an_existing_c_text_trace(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_bytes(Path("tests/fixtures/replay/basic.trace").read_bytes())
    session: Any = TraceSession([1, 2, 3, 4])

    with pytest.raises(ReplayError, match=r"invalid JSON|missing field"):
        PrefillTraceWriter(session, path, think=1)


def test_failed_round_keeps_committed_tokens_and_ignores_save_only_sync(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    session: Any = TraceSession([1, 2, 3, 4])
    with PrefillTraceWriter(session, path, think=1):
        emit_round(session, "failed", fail=True)
        session.emit("Synced", event=synced(3, 2))

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["sampled"] == [3]


def test_repeated_sync_preserves_original_prompt_and_all_samples(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    session: Any = TraceSession([1, 2, 3, 4])
    with PrefillTraceWriter(session, path, think=1):
        emit_round(session, "resumed", repeated_sync=True)

    record = read_replay(path).rounds[0]
    assert record.prompt == (1, 2)
    assert record.cached == 1
    assert record.sampled == (3, 4)


def test_trace_write_failure_publishes_an_error_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session: Any = TraceSession([1, 2, 3, 4])
    events = []
    session.on_event(lambda event: events.append(event) if event.kind == "trace_error" else None)
    trace = tmp_path / "missing" / "trace.jsonl"
    original_open = Path.open

    def fail_trace_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == trace:
            raise OSError("trace write failed")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_trace_open)
    with PrefillTraceWriter(session, trace, think=1):
        emit_round(session, "run")

    assert len(events) == 1
    assert isinstance(events[0].payload["error"], OSError)


VALID = {
    "session_id": "session",
    "run_id": "run",
    "step_index": 0,
    "round_index": 0,
    "prompt": [1, 2, 3],
    "think": 1,
    "cached": 1,
    "prefill_suffix": [2, 3],
    "sampled": [4],
}


def write_record(tmp_path: Path, value: Any) -> Any:
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(value) + "\n")
    return path


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda value: value.pop("prompt"), "missing field"),
        (lambda value: value.update(cached=2), r"cached \+ prefill_suffix"),
        (lambda value: value.update(prefill_suffix=[9, 3]), "suffix does not match"),
        (lambda value: value.update(sampled="4"), "sampled"),
        (lambda value: value.update(round_index=True), "round_index"),
    ],
)
def test_ds4py_replay_records_fail_loudly(tmp_path: Path, mutate: Any, fragment: Any) -> None:
    value = dict(VALID)
    mutate(value)
    path = write_record(tmp_path, value)
    with pytest.raises(ReplayError, match=fragment) as caught:
        read_replay(path)
    assert f"{path}:1:" in str(caught.value)


def test_ds4py_replay_rejects_regressing_harness_steps_within_a_run(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    first = dict(VALID, step_index=2)
    second = dict(VALID, step_index=1, round_index=1)
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")

    with pytest.raises(ReplayError, match="step_index must increase"):
        read_replay(path)


def test_ds4py_replay_rejects_regressing_steps_in_an_interleaved_run(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    records = [
        dict(VALID, run_id="A", step_index=2, round_index=0),
        dict(VALID, run_id="B", step_index=1, round_index=1),
        dict(VALID, run_id="A", step_index=1, round_index=2),
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ReplayError, match="step_index must increase"):
        read_replay(path)


def test_ds4py_replay_rejects_malformed_json_with_its_line(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(VALID) + "\n{broken}\n")
    with pytest.raises(ReplayError, match="invalid JSON") as caught:
        read_replay(path)
    assert f"{path}:2:" in str(caught.value)
