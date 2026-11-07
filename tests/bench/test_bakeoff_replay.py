"""Trace selection and shared replay checks do not reintroduce exact C Gate 6."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from ds4py._native.fake_chat import tokenize_rendered
from ds4py.engine.events import Stopped, Token
from ds4py.engine.tool_syntax.model import Snapshot, ToolCall
from ds4py.engine.transcript import Transcript

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench import trace_selection
from bench.report import build_report
from bench.restore_check import RestoreSummary
from bench.trace_selection import ReplayResult, select_c_traces

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


def test_tool_round_parser_validation_is_linear_in_response_bytes(monkeypatch: Any) -> None:
    data = TOOL_TEXT.replace("/tmp/x", "x" * 20_000).encode()
    snapshot = Snapshot(
        state="DONE",
        raw=data,
        calls=(ToolCall(name="read", args={"path": "x" * 20_000}, raw=data.decode()),),
        error=None,
        param_close_prefix=False,
        in_think=False,
        dsml_in_think=False,
    )
    created = 0
    fed_bytes = 0

    class FakeParser:
        def __init__(self) -> None:
            nonlocal created
            created += 1
            self.in_think = False

        def feed(self, chunk: bytes) -> Snapshot:
            nonlocal fed_bytes
            fed_bytes += len(chunk)
            return snapshot

        def finish(self) -> Snapshot:
            return snapshot

    monkeypatch.setattr(trace_selection, "make_parser", lambda _info: FakeParser())
    engine = SimpleNamespace(info=object(), token_text=lambda _token: data)
    round_ = SimpleNamespace(
        prompt=(),
        sampled=(1,),
        think=0,
        prefix_only=False,
        context_limited=False,
    )
    transcript = Transcript()
    transcript.append((1,), "assistant")

    result = trace_selection._check_round(
        cast("Any", engine),
        "large-tool",
        0,
        cast("Any", round_),
        (Token(1, data), Stopped("tool_call_complete")),
        transcript,
    )

    assert result == (True, "tool_call_complete")
    assert created == 2
    assert fed_bytes == 2 * len(data)


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
