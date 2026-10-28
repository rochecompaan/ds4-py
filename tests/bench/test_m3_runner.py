"""Sequential M3 runner and gates 1–5 with fake executables."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import textwrap

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.m3 import clone_workspace, prepare_exercise
from bench.metrics import ToolAccounting
from bench.replay_m3 import ReplayResult
from bench.report import build_report
from bench.restore_check import CheckResult, RestoreResult, RestoreSummary
from bench.run_m3 import (
    AgentConfig,
    TaskRunResult,
    build_agent_command,
    run_bakeoff,
    run_task,
    run_tasks_sequentially,
)


def config(tmp_path, agent="ds4py"):
    executable = tmp_path / f"{agent}-fake.py"
    if not executable.exists():
        executable.write_text(FAKE_AGENT)
    return AgentConfig(
        agent=agent, executable=(sys.executable, str(executable)), model="/model.gguf",
        backend="cpu", ctx=100000, temperature=0, seed=1, tokens=256,
    )


def result(task_id="task", agent="ds4py", **changes):
    save = Path("/saves") / task_id / agent
    values = dict(
        task_id=task_id, agent=agent, passed=True, attempts=1, wall_seconds=10,
        progress_events=3, tokens=20, tool_calls=1, avg_prefill_reuse=0.8,
        non_extend_rounds=0, save_dir=save if agent == "ds4py" else None,
        trace_path=Path("/traces") / task_id / agent,
        attempt_save_dirs=(save,) if agent == "ds4py" else (),
        tool_accounting=ToolAccounting(1, 1),
        grader_outputs=("grader secret output",),
    )
    values.update(changes)
    return TaskRunResult(**values)


def healthy_restore(results):
    runs = tuple(
        RestoreResult(save_dir=save, passed=True)
        for item in results if item.agent == "ds4py"
        for save in item.attempt_save_dirs
    )
    return RestoreSummary(
        runs=runs,
        interruption=CheckResult(True, "resumed"),
        approval=CheckResult(True, "approved"),
    )


def test_agent_commands_share_model_sampling_and_task_prompt(tmp_path):
    prompt = "solve this exact task"
    trace = tmp_path / "trace.jsonl"
    c_command = build_agent_command(config(tmp_path, "c"), prompt, trace)
    python = build_agent_command(config(tmp_path), prompt, trace)
    approval = build_agent_command(config(tmp_path), prompt, trace, approval=True)

    for command in (c_command, python, approval):
        for flag, value in (
            ("--model", "/model.gguf"), ("--backend", "cpu"),
            ("--ctx", "100000"), ("--temp", "0"), ("--seed", "1"),
            ("--tokens", "256"), ("--prompt", prompt), ("--trace", str(trace)),
        ):
            assert command[command.index(flag) + 1] == value
    assert "--yolo" not in c_command
    assert "--non-interactive" in python and "--yolo" in python
    assert "--yolo" not in approval


def test_failed_first_attempt_feeds_grader_output_to_exactly_one_retry(tmp_path):
    source = make_exercise(tmp_path)
    workspace = prepare_exercise(source, tmp_path / "staged")
    outcomes = iter([
        subprocess.CompletedProcess([], 1, stdout="hidden failure", stderr=""),
        subprocess.CompletedProcess([], 0, stdout="hidden pass", stderr=""),
    ])

    completed = run_task(
        workspace, config(tmp_path), tmp_path / "results",
        grade_runner=lambda *args: next(outcomes),
    )

    assert completed.attempts == 2 and completed.passed
    assert completed.grader_outputs == ("hidden failure", "hidden pass")
    assert len(completed.attempt_save_dirs) == 2
    assert all(path.is_dir() for path in completed.attempt_save_dirs)
    assert completed.tokens == 2
    invocations = read_invocations(tmp_path)
    assert len(invocations) == 2
    assert "hidden failure" not in invocations[0]["prompt"]
    assert "hidden failure" in invocations[1]["prompt"]


def test_c_and_python_run_sequentially_on_equivalent_tasks(tmp_path):
    source = make_exercise(tmp_path)
    staged = prepare_exercise(source, tmp_path / "staged")

    results = run_tasks_sequentially(
        [staged], [config(tmp_path, "c"), config(tmp_path)], tmp_path / "results",
        grade_runner=lambda command, cwd, env: subprocess.CompletedProcess(command, 0, "pass", ""),
    )

    assert [(item.task_id, item.agent) for item in results] == [("tiny", "c"), ("tiny", "ds4py")]
    assert all(item.passed for item in results)
    invocations = read_invocations(tmp_path)
    assert [item["agent"] for item in invocations] == ["c", "ds4py"]
    assert not (tmp_path / "concurrency-violation").exists()
    assert not (results[0].save_dir or Path("missing")).exists() if results[0].save_dir else True


def test_gates_one_to_five_pass_with_complete_evidence():
    c = result(agent="c", wall_seconds=10)
    python = result(wall_seconds=15, non_extend_rounds=2)
    python = replace(python, allowed_non_extend_rounds=2)
    report = build_report([c, python], healthy_restore([c, python]))

    assert all(report.gates[f"gate-{number}"].passed for number in range(1, 6))
    assert not report.gates["gate-6"].passed
    assert "not run" in report.gates["gate-6"].detail


def test_gate_one_allows_exactly_one_task_of_pass_slack():
    c = [result(f"task-{index}", "c") for index in range(3)]
    python = [result(f"task-{index}", passed=index != 0) for index in range(3)]
    assert build_report(c + python, healthy_restore(c + python)).gates["gate-1"].passed
    python[1] = replace(python[1], passed=False)
    report = build_report(c + python, healthy_restore(c + python))
    assert not report.gates["gate-1"].passed
    assert "attempt" in report.gates["gate-1"].detail


def test_gate_two_allows_bad_argument_retry_but_not_parser_error_or_missing_outcome():
    c = result(agent="c")
    python = replace(result(), tool_accounting=ToolAccounting(2, 1, 0, 1))
    assert build_report([c, python], healthy_restore([c, python])).gates["gate-2"].passed

    parser_error = replace(python, parser_errors=1)
    assert not build_report([c, parser_error], healthy_restore([c, parser_error])).gates["gate-2"].passed
    missing = replace(python, tool_accounting=ToolAccounting(2, 1, 0, 0))
    assert not build_report([c, missing], healthy_restore([c, missing])).gates["gate-2"].passed


def test_gate_three_enforces_time_and_only_unexplained_non_extensions():
    c = result(agent="c", wall_seconds=10)
    python = result(wall_seconds=15.01)
    assert not build_report([c, python], healthy_restore([c, python])).gates["gate-3"].passed

    on_time = replace(python, wall_seconds=15, invalid_non_extend_rounds=1, non_extend_rounds=1)
    assert not build_report([c, on_time], healthy_restore([c, on_time])).gates["gate-3"].passed


def test_gate_four_requires_interrupt_resume_and_explicit_approval():
    results = [result(agent="c"), result()]
    restore = healthy_restore(results)
    restore = replace(restore, approval=CheckResult(False, "EOF denied"))
    report = build_report(results, restore)
    assert not report.gates["gate-4"].passed
    assert "EOF denied" in report.gates["gate-4"].detail


def test_gate_five_requires_restore_for_every_attempt():
    python = replace(result(attempts=2), attempts=2)
    saves = (Path("/saves/first"), Path("/saves/second"))
    python = replace(python, attempts=2, attempt_save_dirs=saves, save_dir=saves[-1])
    c = result(agent="c")
    restore = healthy_restore([c, python])
    restore = replace(restore, runs=restore.runs[:1])

    assert not build_report([c, python], restore).gates["gate-5"].passed


def test_gate_five_rejects_an_attempt_with_no_save_directory():
    c = result(agent="c")
    save = Path("/saves/only-one")
    python = replace(result(), attempts=2, save_dir=save, attempt_save_dirs=(save,))
    restore = healthy_restore([c, python])

    report = build_report([c, python], restore)

    assert not report.gates["gate-5"].passed
    assert "attempt" in report.gates["gate-5"].detail


def test_report_json_has_attempt_feedback_and_table_has_six_deltas_not_logs():
    c = result(agent="c", wall_seconds=10, progress_events=2, tokens=10)
    python = result(wall_seconds=12, progress_events=5, tokens=14)
    report = build_report([c, python], healthy_restore([c, python]))

    data = report.to_dict()
    python_data = next(item for item in data["tasks"] if item["agent"] == "ds4py")
    assert python_data["grader_outputs"] == ["grader secret output"]
    table = report.compact_table()
    for metric in (
        "wall_seconds", "progress_events", "tokens", "tool_calls",
        "avg_prefill_reuse", "non_extend_rounds",
    ):
        assert metric in table
    assert "delta" in table and "grader secret output" not in table


def make_exercise(root):
    source = root / "benchmark/python/exercises/practice/tiny"
    (source / ".docs").mkdir(parents=True)
    (source / ".docs/instructions.md").write_text("Return good.\n")
    (source / "solution.py").write_text("def value():\n    return 'good'\n")
    (source / "tiny_test.py").write_text("def test_value():\n    assert True\n")
    return source


def read_invocations(root):
    path = root / "invocations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


FAKE_AGENT = r'''
import fcntl
import json
import os
from pathlib import Path
import sys
import time

args = sys.argv[1:]
def option(name):
    return args[args.index(name) + 1]

root = Path(os.environ["M3_ATTEMPT_ROOT"])
root.mkdir(parents=True, exist_ok=True)
agent = os.environ["M3_AGENT_NAME"]
lock_path = Path(os.environ["M3_LOCK_PATH"])
lock_path.parent.mkdir(parents=True, exist_ok=True)
with lock_path.open("w") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_path.with_name("concurrency-violation").write_text("overlap")
        raise
    time.sleep(0.02)
    events = [
        {"kind": "run_start", "run_id": "run-1", "payload": {}},
        {"kind": "Synced", "run_id": "run-1", "payload": {"event": {
            "plan": {"kind": "rebuild", "reused": 0, "to_prefill": 2}
        }}},
        {"kind": "Prefill", "run_id": "run-1", "payload": {"event": {"done": 2, "total": 2}}},
        {"kind": "Token", "run_id": "run-1", "payload": {"event": {"id": 1}}},
        {"kind": "Stopped", "run_id": "run-1", "payload": {"event": {"reason": "stop_token"}}},
        {"kind": "run_end", "run_id": "run-1", "payload": {"status": "success"}},
    ]
    events_text = "\n".join(json.dumps(item) for item in events) + "\n"
    Path(os.environ["M3_EVENTS_PATH"]).write_text(events_text)
    trace = {
        "session_id": "fake", "run_id": "run-1", "step_index": 0,
        "round_index": 0, "prompt": [1, 2], "think": 0, "cached": 0,
        "prefill_suffix": [1, 2], "sampled": [1],
    }
    Path(option("--trace")).write_text(json.dumps(trace) + "\n")
    if agent == "ds4py":
        save = Path(os.environ["DS4PY_HOME"]) / "sessions" / ("save-" + os.environ["M3_ATTEMPT"])
        save.mkdir(parents=True)
        (save / "save.json").write_text("{}")
        (save / "events.jsonl").write_text(events_text)
    invocation = {"agent": agent, "prompt": option("--prompt"), "argv": args}
    with Path(os.environ["M3_INVOCATIONS"]).open("a") as stream:
        stream.write(json.dumps(invocation) + "\n")
    fcntl.flock(lock, fcntl.LOCK_UN)
'''


def test_complete_runner_writes_gate_six_after_acceptance(tmp_path, monkeypatch):
    import bench.run_m3 as runner

    order = []
    staged = [SimpleNamespace(task_id="repo-glm-close")]
    monkeypatch.setattr(runner, "load_manifest", lambda _path: object())
    monkeypatch.setattr(runner, "prepare_manifest", lambda *args: staged)
    monkeypatch.setattr(
        runner, "run_tasks_sequentially",
        lambda *args, **kwargs: order.append("tasks") or (),
    )
    monkeypatch.setattr(
        runner, "check_result_saves",
        lambda *args: order.append("restores") or (),
    )
    monkeypatch.setattr(runner, "clone_workspace", lambda *args: object())
    monkeypatch.setattr(
        runner, "interruption_acceptance",
        lambda *args: order.append("interruption") or CheckResult(True, "ok"),
    )
    monkeypatch.setattr(
        runner, "approval_acceptance",
        lambda *args, **kwargs: order.append("approval") or CheckResult(True, "ok"),
    )
    replay = ReplayResult("task", 0, True, "tool replay", tool_call=True)
    monkeypatch.setattr(
        runner, "replay_task_results",
        lambda *args: order.append("replay") or (replay,),
    )

    report = run_bakeoff(
        tmp_path / "manifest", tmp_path / "benchmark", tmp_path,
        (config(tmp_path),), tmp_path / "results",
    )

    assert order == ["tasks", "restores", "interruption", "approval", "replay"]
    assert report.gates["gate-6"].passed
    assert (tmp_path / "results/m3.json").is_file()
    assert "gate-6: PASS" in (tmp_path / "results/report.txt").read_text()
