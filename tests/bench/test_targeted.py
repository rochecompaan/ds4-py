"""Independent targeted bakeoff repetitions with retained evidence."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

import bench.run_targeted as targeted
from bench.metrics import EventSummary
from bench.run_bakeoff import TaskRunResult
from bench.workspaces import Workspace


def workspace(root: Path, task_id: str) -> Workspace:
    agent_dir = root / task_id / "agent"
    grader_dir = root / task_id / "grader"
    agent_dir.mkdir(parents=True)
    grader_dir.mkdir(parents=True)
    return Workspace(
        task_id=task_id,
        agent_dir=agent_dir,
        grader_dir=grader_dir,
        test_paths=(),
        max_attempts=2,
    )


def completed(
    workspace: Workspace,
    *,
    passed: bool = True,
    errors: tuple[str, ...] = (),
) -> TaskRunResult:
    summary = EventSummary(
        wall_seconds=0.0,
        progress_events=0,
        tokens=0,
        tool_calls=0,
        avg_prefill_reuse=None,
        non_extend_rounds=0,
    )
    return TaskRunResult(
        task_id=workspace.task_id,
        agent="ds4py",
        passed=passed,
        attempts=1,
        summary=summary,
        attempt_save_dirs=(),
        attempt_trace_paths=(),
        errors=errors,
        workspace=workspace.agent_dir,
    )


def test_targeted_runs_each_task_three_times_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    staged = [
        workspace(tmp_path / "staged-source", "scale-generator"),
        workspace(tmp_path / "staged-source", "repo-ctx-command"),
    ]
    calls: list[tuple[Workspace, Any, Path]] = []

    monkeypatch.setattr(
        targeted,
        "load_manifest",
        lambda _path: SimpleNamespace(ctx=100_000, temperature=0.0, seed=1),
    )
    monkeypatch.setattr(targeted, "prepare_manifest", lambda *args: staged)

    def fake_run_task(
        workspace: Workspace,
        config: Any,
        results_root: Path,
        **kwargs: Any,
    ) -> TaskRunResult:
        calls.append((workspace, config, Path(results_root)))
        return completed(workspace)

    monkeypatch.setattr(targeted, "run_task", fake_run_task)
    results = tmp_path / "results"

    exit_code = targeted.main(
        [
            "--task",
            "scale-generator",
            "--task",
            "repo-ctx-command",
            "--repetitions",
            "3",
            "--model",
            "/model.gguf",
            "--results",
            str(results),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 6
    assert [call[0].task_id for call in calls] == [
        "scale-generator",
        "scale-generator",
        "scale-generator",
        "repo-ctx-command",
        "repo-ctx-command",
        "repo-ctx-command",
    ]
    assert all(call[0].max_attempts == 1 for call in calls)
    assert len({call[0].agent_dir for call in calls}) == 6
    assert [call[2] for call in calls] == [
        results / task_id / f"run-{repetition}"
        for task_id in ("scale-generator", "repo-ctx-command")
        for repetition in range(1, 4)
    ]
    assert all(call[1].tokens == 8192 and call[1].request_limit == 100 for call in calls)

    data = json.loads((results / "targeted.json").read_text())
    assert len(data) == 6
    assert [item["repetition"] for item in data] == [1, 2, 3, 1, 2, 3]
    assert [item["task_id"] for item in data] == [call[0].task_id for call in calls]
    assert all(isinstance(item["workspace"], str) for item in data)
    output = capsys.readouterr().out
    assert output.count("PASS") == 6


def test_targeted_prepends_task_preamble_to_each_independent_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = [workspace(tmp_path / "staged-source", "scale-generator")]
    (staged[0].agent_dir / "TASK.md").write_text("Original task.\n")
    preamble = tmp_path / "preamble.md"
    preamble.write_text("Write a best-effort implementation, then run local checks.\n")
    prompts: list[str] = []

    monkeypatch.setattr(
        targeted,
        "load_manifest",
        lambda _path: SimpleNamespace(ctx=100_000, temperature=0.0, seed=1),
    )
    monkeypatch.setattr(targeted, "prepare_manifest", lambda *args: staged)

    def fake_run_task(workspace: Workspace, *_args: Any) -> TaskRunResult:
        prompts.append((workspace.agent_dir / "TASK.md").read_text())
        return completed(workspace)

    monkeypatch.setattr(targeted, "run_task", fake_run_task)

    exit_code = targeted.main(
        [
            "--task",
            "scale-generator",
            "--repetitions",
            "2",
            "--model",
            "/model.gguf",
            "--task-preamble",
            str(preamble),
            "--results",
            str(tmp_path / "results"),
        ]
    )

    expected = "Write a best-effort implementation, then run local checks.\n\nOriginal task.\n"
    assert exit_code == 0
    assert prompts == [expected, expected]
    assert (staged[0].agent_dir / "TASK.md").read_text() == expected


def test_targeted_c_agent_uses_requested_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = [workspace(tmp_path / "staged-source", "scale-generator")]
    configs: list[Any] = []

    monkeypatch.setattr(
        targeted,
        "load_manifest",
        lambda _path: SimpleNamespace(ctx=100_000, temperature=0.0, seed=1),
    )
    monkeypatch.setattr(targeted, "prepare_manifest", lambda *args: staged)

    def fake_run_task(workspace: Workspace, config: Any, *_args: Any) -> TaskRunResult:
        configs.append(config)
        return completed(workspace)

    monkeypatch.setattr(targeted, "run_task", fake_run_task)

    exit_code = targeted.main(
        [
            "--task",
            "scale-generator",
            "--repetitions",
            "2",
            "--model",
            "/model.gguf",
            "--c-agent",
            "/opt/ds4-agent",
            "--results",
            str(tmp_path / "results"),
        ]
    )

    assert exit_code == 0
    assert len(configs) == 2
    assert all(config.agent == "c" for config in configs)
    assert all(config.executable == ("/opt/ds4-agent",) for config in configs)


@pytest.mark.parametrize(
    ("passed", "errors"),
    [
        (False, ()),
        (True, ("attempt 1 request_limit",)),
    ],
)
def test_targeted_returns_nonzero_for_grader_or_attempt_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passed: bool,
    errors: tuple[str, ...],
) -> None:
    staged = [workspace(tmp_path / "staged-source", "scale-generator")]
    monkeypatch.setattr(
        targeted,
        "load_manifest",
        lambda _path: SimpleNamespace(ctx=100_000, temperature=0.0, seed=1),
    )
    monkeypatch.setattr(targeted, "prepare_manifest", lambda *args: staged)
    monkeypatch.setattr(
        targeted,
        "run_task",
        lambda workspace, *args: completed(workspace, passed=passed, errors=errors),
    )

    exit_code = targeted.main(
        [
            "--task",
            "scale-generator",
            "--repetitions",
            "1",
            "--model",
            "/model.gguf",
            "--results",
            str(tmp_path / "results"),
        ]
    )

    assert exit_code == 1
    assert len(json.loads((tmp_path / "results" / "targeted.json").read_text())) == 1
