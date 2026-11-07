"""Report-reading commands deserialize real typed benchmark contracts."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.restore_check import RestoreResult, check_command
from bench.run_bakeoff import TaskRunResult


def task_record(task_id: str, agent: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "agent": agent,
        "passed": True,
        "attempts": 1,
        "summary": {
            "wall_seconds": 1.5,
            "progress_events": 2,
            "tokens": 3,
            "tool_calls": 0,
            "avg_prefill_reuse": 0.5,
            "non_extend_rounds": 0,
            "invalid_non_extend_rounds": 0,
            "allowed_non_extend_rounds": 0,
            "non_extension_reasons": [],
            "parser_errors": 0,
            "terminal_reasons": ["stop_token" if agent == "ds4py" else "recorded_boundary"],
            "tool_accounting": {
                "parsed": 0,
                "executed": 0,
                "refused": 0,
                "retried": 0,
            }
            if agent == "ds4py"
            else None,
            "reuse_ratios": [0.5],
        },
        "grader_outputs": [],
        "attempt_save_dirs": ["/save"] if agent == "ds4py" else [],
        "attempt_trace_paths": [f"/{task_id}.trace"],
        "errors": [],
        "workspace": f"/workspaces/{task_id}",
    }


def write_report(path: Path, tasks: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"tasks": tasks, "gates": {}}))


def test_task_deserialization_requires_terminal_evidence() -> None:
    record = task_record("task", "ds4py")
    summary = record["summary"]
    del summary["terminal_reasons"]

    with pytest.raises(ValueError, match="terminal_reasons"):
        TaskRunResult.from_dict(record)


@pytest.mark.parametrize(
    ("options", "tokens", "request_limit"),
    [([], 8192, 100), (["--tokens", "4096", "--request-limit", "7"], 4096, 7)],
)
def test_restore_check_deserializes_tasks_and_complete_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
    tokens: int,
    request_limit: int,
) -> None:
    report = tmp_path / "bakeoff.json"
    write_report(report, [task_record("task", "ds4py")])
    captured: dict[str, Any] = {}

    def check(tasks: Any, config: Any, results_root: Any) -> tuple[RestoreResult, ...]:
        captured.update(tasks=tasks, config=config, results_root=results_root)
        return (RestoreResult(tmp_path / "save", True),)

    monkeypatch.setattr("bench.restore_check.check_result_saves", check)

    status = check_command(
        [
            "--report",
            str(report),
            "--model",
            "/model.gguf",
            *options,
        ]
    )

    assert status == 0
    assert len(captured["tasks"]) == 1
    assert isinstance(captured["tasks"][0], TaskRunResult)
    assert captured["tasks"][0].summary.reuse_ratios == (0.5,)
    assert captured["tasks"][0].attempt_save_dirs == (Path("/save"),)
    assert captured["config"].tokens == tokens
    assert captured["config"].request_limit == request_limit


def test_restore_recheck_inherits_schema_two_session_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "bakeoff.json"
    write_report(report, [task_record("task", "ds4py")])
    data = json.loads(report.read_text())
    identity = {
        "bakeoff_session_id": "session-1",
        "instance_id": "instance-1",
        "gpu_model": "NVIDIA H100",
        "gpu_driver_version": "580.00",
        "ds4_revision": "ds4-revision",
        "ds4py_revision": "ds4py-revision",
        "libds4_sha256": "library-sha",
        "model_sha256": "model-sha",
    }
    data.update(schema_version=2, **identity)
    data["provenance"] = [{"c": identity}]
    report.write_text(json.dumps(data))
    captured: dict[str, Any] = {}

    def check(tasks: Any, config: Any, results_root: Any, **kwargs: Any) -> tuple[RestoreResult, ...]:
        captured.update(kwargs)
        return (RestoreResult(tmp_path / "save", True),)

    monkeypatch.setattr("bench.restore_check.check_result_saves", check)

    status = check_command(["--report", str(report), "--model", "/model.gguf"])

    assert status == 0
    environment = captured["provenance_environment"]
    assert environment["BAKEOFF_SESSION_ID"] == "session-1"
    assert environment["DS4_BAKEOFF_INSTANCE_ID"] == "instance-1"
    assert environment["DS4_BAKEOFF_MODEL_SHA256"] == "model-sha"
    assert environment["DS4_BAKEOFF_LIBDS4_SHA256"] == "library-sha"
