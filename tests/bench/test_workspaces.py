"""Workspace isolation, hidden grading, and retry behavior for the agent bake-off."""

from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.workspaces import (
    BenchmarkError,
    PreparationError,
    clone_workspace,
    grade_workspace,
    load_manifest,
    prepare_exercise,
    prepare_repository_task,
    run_attempts,
    verify_benchmark,
)


EXERCISE = Path("python/exercises/practice/tiny")
HIDDEN_TEST = EXERCISE.name + "_test.py"


def make_exercise(root, initial="good"):
    source = root / "benchmark" / EXERCISE
    (source / ".docs").mkdir(parents=True)
    (source / ".docs" / "instructions.md").write_text(
        "Implement value() so it returns 'good'.\n"
    )
    (source / "solution.py").write_text(f"def value():\n    return {initial!r}\n")
    (source / HIDDEN_TEST).write_text(
        "from solution import value\n\n"
        "def test_value():\n"
        "    assert value() == 'good'\n"
    )
    return source


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def commit_repo(root):
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    git(root, "config", "user.email", "bakeoff@example.invalid")
    git(root, "config", "user.name", "Bakeoff Test")
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


def test_exercise_has_visible_starter_and_temporarily_mounted_hidden_tests(tmp_path):
    source = make_exercise(tmp_path)
    workspace = prepare_exercise(source, tmp_path / "staged")

    assert (workspace.agent_dir / "TASK.md").read_text() == (
        source / ".docs" / "instructions.md"
    ).read_text()
    assert (workspace.agent_dir / "solution.py").is_file()
    assert not (workspace.agent_dir / HIDDEN_TEST).exists()
    assert (workspace.grader_dir / HIDDEN_TEST).is_file()

    result = grade_workspace(workspace)

    assert result.passed, result.output
    assert "1 passed" in result.output
    assert not (workspace.agent_dir / HIDDEN_TEST).exists()


def test_first_failure_output_is_feedback_for_attempt_two(tmp_path):
    source = make_exercise(tmp_path, initial="bad")
    workspace = prepare_exercise(source, tmp_path / "staged")
    calls = []

    def agent(agent_dir, attempt, feedback):
        calls.append((attempt, feedback))
        assert not (agent_dir / HIDDEN_TEST).exists()
        value = "bad" if attempt == 1 else "good"
        (agent_dir / "solution.py").write_text(f"def value():\n    return {value!r}\n")

    grades = run_attempts(workspace, agent)

    assert len(grades) == 2
    assert not grades[0].passed and grades[1].passed
    assert calls[0] == (1, "")
    assert calls[1][0] == 2
    assert "test_value" in calls[1][1] and "assert" in calls[1][1]
    assert not (workspace.agent_dir / HIDDEN_TEST).exists()


def test_attempts_stop_at_two_and_cleanup_survives_failures(tmp_path):
    source = make_exercise(tmp_path, initial="bad")
    workspace = prepare_exercise(source, tmp_path / "staged")
    calls = []

    def agent(agent_dir, attempt, feedback):
        calls.append((attempt, feedback))

    def fail(command, cwd, env):
        assert cwd == workspace.grader_dir
        assert (workspace.agent_dir / HIDDEN_TEST).is_file()
        return subprocess.CompletedProcess(command, 1, stdout="still failing", stderr="")

    grades = run_attempts(workspace, agent, runner=fail)

    assert [attempt for attempt, _feedback in calls] == [1, 2]
    assert calls[1][1] == "still failing"
    assert len(grades) == 2 and not any(grade.passed for grade in grades)
    assert not (workspace.agent_dir / HIDDEN_TEST).exists()


def test_agents_receive_separate_but_equivalent_workspace_copies(tmp_path):
    source = make_exercise(tmp_path)
    staged = prepare_exercise(source, tmp_path / "staged")
    c_workspace = clone_workspace(staged, tmp_path / "c")
    python_workspace = clone_workspace(staged, tmp_path / "ds4py")

    def files(root):
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()
        }

    assert files(c_workspace.agent_dir) == files(python_workspace.agent_dir)
    assert c_workspace.agent_dir != python_workspace.agent_dir
    (c_workspace.agent_dir / "solution.py").write_text("def value():\n    return 'C'\n")
    assert "return 'good'" in (python_workspace.agent_dir / "solution.py").read_text()


def test_wrong_benchmark_commit_fails_clearly(tmp_path):
    source = make_exercise(tmp_path)
    benchmark = source.parents[3]
    actual = commit_repo(benchmark)
    expected = "0" * 40

    with pytest.raises(BenchmarkError) as caught:
        verify_benchmark(benchmark, expected)

    message = str(caught.value)
    assert expected in message and actual in message


def test_ctx_repository_copy_keeps_hidden_grader_outside(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "ds4py").mkdir(parents=True)
    (repo / "src" / "ds4py" / "__init__.py").write_text("")
    (repo / "tests" / "engine" / "tool_syntax").mkdir(parents=True)
    (repo / "tests" / "engine" / "tool_syntax" / "test_glm.py").write_text("# visible\n")
    (repo / "bench" / "graders").mkdir(parents=True)
    (repo / "bench" / "graders" / "test_ctx_command.py").write_text("# secret\n")
    commit_repo(repo)
    external_grader = tmp_path / "external" / "test_ctx_command.py"
    external_grader.parent.mkdir()
    external_grader.write_text("# hidden behavioral grader\n")

    workspace = prepare_repository_task(
        repo, tmp_path / "staged", task_id="repo-ctx-command",
        prompt="Add /ctx without generation.", grader=external_grader,
    )

    assert (workspace.agent_dir / "TASK.md").read_text() == "Add /ctx without generation."
    assert not (workspace.agent_dir / "bench").exists()
    assert not (workspace.agent_dir / "tests" / "bakeoff_hidden" / "test_ctx_command.py").exists()
    assert (workspace.grader_dir / "tests" / "bakeoff_hidden" / "test_ctx_command.py").is_file()
    assert (workspace.agent_dir / "tests" / "engine" / "tool_syntax" / "test_glm.py").is_file()


def test_glm_patch_fails_focused_test_and_clean_tree_passes(tmp_path):
    repo = Path(__file__).parents[2]
    common = dict(
        prompt="Fix the GLM close boundary without editing tests.",
        visible_test="tests/engine/tool_syntax/test_glm.py",
    )
    clean = prepare_repository_task(
        repo, tmp_path / "clean", task_id="repo-glm-clean", **common,
    )
    patched = prepare_repository_task(
        repo, tmp_path / "patched", task_id="repo-glm-close",
        patch=repo / "bench" / "patches" / "glm-off-by-one.patch", **common,
    )

    clean_result = grade_workspace(clean)
    patched_result = grade_workspace(patched)

    assert clean_result.passed, clean_result.output
    assert not patched_result.passed
    assert "test_glm" in patched_result.output


def test_workspace_cannot_bypass_grading_with_startup_or_pytest_configuration(tmp_path):
    source = make_exercise(tmp_path, initial="bad")
    workspace = prepare_exercise(source, tmp_path / "staged")
    agent_dir = workspace.agent_dir
    (agent_dir / "conftest.py").write_text(
        "import solution\nsolution.value = lambda: 'good'\n"
    )
    (agent_dir / "workspace_bypass.py").write_text(
        "import solution\nsolution.value = lambda: 'good'\n"
    )
    (agent_dir / "pytest.ini").write_text("[pytest]\naddopts = -p workspace_bypass\n")
    (agent_dir / "src").mkdir()
    (agent_dir / "src" / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(agent_dir / 'solution.py')!r}).write_text("
        '"def value():\\n    return \'good\'\\n")\n'
    )

    result = grade_workspace(workspace)

    assert not result.passed
    assert "return 'bad'" in (agent_dir / "solution.py").read_text()


@pytest.mark.parametrize("content", ["", "---\n", "- not\n- a\n- mapping\n", "scalar\n"])
def test_empty_or_non_mapping_manifest_is_a_path_aware_preparation_error(tmp_path, content):
    path = tmp_path / "manifest.yaml"
    path.write_text(content)

    with pytest.raises(PreparationError) as caught:
        load_manifest(path)

    assert str(path) in str(caught.value)
    assert "mapping" in str(caught.value)
