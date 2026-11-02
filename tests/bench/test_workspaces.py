"""Workspace isolation, hidden grading, and retry behavior for the agent bake-off."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    validate_manifest_sources,
    verify_benchmark,
)

EXERCISE = Path("python/exercises/practice/tiny")
HIDDEN_TEST = EXERCISE.name + "_test.py"


def make_exercise(root: Any, initial: Any = "good") -> Any:
    source = root / "benchmark" / EXERCISE
    (source / ".docs").mkdir(parents=True)
    (source / ".meta").mkdir()
    (source / "__pycache__").mkdir()
    (source / ".docs" / "instructions.md").write_text("Implement value() so it returns 'good'.\n")
    (source / ".docs" / "instructions.append.md").write_text("Return the value without changing its type.\n")
    (source / ".meta" / "config.json").write_text(
        json.dumps(
            {
                "files": {
                    "solution": ["solution.py", "support.py"],
                    "test": [HIDDEN_TEST],
                    "example": [".meta/example.py"],
                }
            }
        )
    )
    (source / ".meta" / "example.py").write_text("def value():\n    return 'good'\n")
    (source / "solution.py").write_text(f"def value():\n    return {initial!r}\n")
    (source / "support.py").write_text("SUPPORT = True\n")
    (source / HIDDEN_TEST).write_text("from solution import value\n\ndef test_value():\n    assert value() == 'good'\n")
    (source / "__pycache__" / "tiny_test.cpython-314-pytest-9.1.1.pyc").write_bytes(b"compiled hidden test")
    return source


def git(cwd: Any, *args: Any) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit_repo(root: Any) -> Any:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    git(root, "config", "user.email", "bakeoff@example.invalid")
    git(root, "config", "user.name", "Bakeoff Test")
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


def test_exercise_stages_only_student_inputs_with_visible_docs(tmp_path: Path) -> None:
    source = make_exercise(tmp_path)
    workspace = prepare_exercise(source, tmp_path / "staged")

    assert (workspace.agent_dir / "TASK.md").read_text() == (source / ".docs" / "instructions.md").read_text()
    assert {
        str(path.relative_to(workspace.agent_dir)) for path in workspace.agent_dir.rglob("*") if path.is_file()
    } == {
        "TASK.md",
        "docs/instructions.md",
        "docs/instructions.append.md",
        "solution.py",
        "support.py",
    }
    assert not (workspace.agent_dir / HIDDEN_TEST).exists()
    assert (workspace.grader_dir / HIDDEN_TEST).is_file()


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(None, id="missing-config"),
        pytest.param("{not json", id="invalid-json"),
        pytest.param({"files": {}}, id="missing-solution"),
        pytest.param({"files": {"solution": "solution.py"}}, id="not-a-list"),
        pytest.param({"files": {"solution": []}}, id="empty-list"),
        pytest.param({"files": {"solution": [42]}}, id="non-string-entry"),
        pytest.param({"files": {"solution": ["bad\0name.py"]}}, id="nul-path"),
    ],
)
def test_exercise_rejects_missing_or_malformed_student_configuration(
    tmp_path: Path,
    config: Any,
) -> None:
    source = make_exercise(tmp_path)
    config_path = source / ".meta" / "config.json"
    if config is None:
        config_path.unlink()
    elif isinstance(config, str):
        config_path.write_text(config)
    else:
        config_path.write_text(json.dumps(config))

    with pytest.raises(PreparationError, match=r"\.meta/config\.json"):
        prepare_exercise(source, tmp_path / "staged")


@pytest.mark.parametrize(
    "relative",
    [
        pytest.param("../outside.py", id="parent-traversal"),
        pytest.param("/absolute.py", id="absolute"),
        pytest.param(HIDDEN_TEST, id="hidden-test"),
        pytest.param(".meta/example.py", id="grading-metadata"),
        pytest.param("__pycache__/solution.pyc", id="bytecode-cache"),
        pytest.param("missing.py", id="missing-file"),
        pytest.param(".", id="directory"),
    ],
)
def test_exercise_rejects_unsafe_or_grading_student_paths(
    tmp_path: Path,
    relative: str,
) -> None:
    source = make_exercise(tmp_path)
    (source.parent / "outside.py").write_text("OUTSIDE = True\n")
    (source / "__pycache__" / "solution.pyc").write_bytes(b"bytecode")
    (source / ".meta" / "config.json").write_text(json.dumps({"files": {"solution": [relative]}}))
    destination = tmp_path / "staged"

    with pytest.raises(PreparationError):
        prepare_exercise(source, destination)
    assert not destination.exists()


def test_exercise_rejects_solution_symlink_escaping_source(tmp_path: Path) -> None:
    source = make_exercise(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("OUTSIDE = True\n")
    (source / "linked.py").symlink_to(outside)
    (source / ".meta" / "config.json").write_text(json.dumps({"files": {"solution": ["linked.py"]}}))
    destination = tmp_path / "staged"

    with pytest.raises(PreparationError):
        prepare_exercise(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    ("declared", "target", "is_directory"),
    [
        pytest.param("innocent.py", ".meta/example.py", False, id="example-file-alias"),
        pytest.param("innocent.py", HIDDEN_TEST, False, id="test-file-alias"),
        pytest.param("alias/example.py", ".meta", True, id="example-directory-alias"),
        pytest.param(f"alias/{HIDDEN_TEST}", ".", True, id="test-directory-alias"),
    ],
)
def test_exercise_rejects_symlink_aliases_to_grading_inputs(
    tmp_path: Path,
    declared: str,
    target: str,
    is_directory: bool,
) -> None:
    source = make_exercise(tmp_path)
    link = source / (Path(declared).parts[0] if is_directory else declared)
    link.symlink_to(source / target, target_is_directory=is_directory)
    (source / ".meta" / "config.json").write_text(json.dumps({"files": {"solution": [declared]}}))
    destination = tmp_path / "staged"

    with pytest.raises(PreparationError):
        prepare_exercise(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("name", ["leak.txt", "instructions.md"])
def test_exercise_rejects_documentation_symlinks_outside_source(
    tmp_path: Path,
    name: str,
) -> None:
    source = make_exercise(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("EXTERNAL_SECRET\n")
    document = source / ".docs" / name
    document.unlink(missing_ok=True)
    document.symlink_to(outside)
    destination = tmp_path / "staged"

    with pytest.raises(PreparationError):
        prepare_exercise(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("target", [".meta/example.py", HIDDEN_TEST])
def test_exercise_rejects_documentation_aliases_to_grading_inputs(
    tmp_path: Path,
    target: str,
) -> None:
    source = make_exercise(tmp_path)
    (source / ".docs" / "alias.txt").symlink_to(source / target)
    destination = tmp_path / "staged"

    with pytest.raises(PreparationError):
        prepare_exercise(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "relative",
    [
        "__pycache__/hidden.pyc",
        ".pytest_cache/secret",
    ],
)
def test_exercise_rejects_documentation_artifacts(
    tmp_path: Path,
    relative: str,
) -> None:
    source = make_exercise(tmp_path)
    artifact = source / ".docs" / relative
    artifact.parent.mkdir()
    artifact.write_bytes(b"GRADING_ARTIFACT")
    destination = tmp_path / "staged"

    with pytest.raises(PreparationError):
        prepare_exercise(source, destination)
    assert not destination.exists()


def test_real_grading_leaves_no_hidden_source_or_bytecode_in_agent_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    source = make_exercise(tmp_path)
    workspace = prepare_exercise(source, tmp_path / "staged")

    def assert_no_grading_artifacts() -> None:
        artifacts = [
            path
            for path in workspace.agent_dir.rglob("*")
            if (
                Path(HIDDEN_TEST).stem in path.name
                or "__pycache__" in path.parts
                or ".pytest_cache" in path.parts
                or path.suffix == ".pyc"
            )
        ]
        assert artifacts == []

    result = grade_workspace(workspace)

    assert result.passed, result.output
    assert_no_grading_artifacts()
    assert (workspace.grader_dir / HIDDEN_TEST).is_file()

    result = grade_workspace(workspace)

    assert result.passed, result.output
    assert "1 passed" in result.output
    assert_no_grading_artifacts()


def test_first_failure_output_is_feedback_for_attempt_two(tmp_path: Path) -> None:
    source = make_exercise(tmp_path, initial="bad")
    workspace = prepare_exercise(source, tmp_path / "staged")
    calls = []

    def agent(agent_dir: Any, attempt: Any, feedback: Any) -> None:
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


def test_attempts_stop_at_two_and_cleanup_survives_failures(tmp_path: Path) -> None:
    source = make_exercise(tmp_path, initial="bad")
    workspace = prepare_exercise(source, tmp_path / "staged")
    calls = []

    def agent(agent_dir: Any, attempt: Any, feedback: Any) -> None:
        calls.append((attempt, feedback))

    def fail(command: Any, cwd: Any, env: Any) -> Any:
        assert cwd == workspace.grader_dir
        assert not (workspace.agent_dir / HIDDEN_TEST).exists()
        assert str(workspace.grader_dir / HIDDEN_TEST) in command
        return subprocess.CompletedProcess(command, 1, stdout="still failing", stderr="")

    grades = run_attempts(workspace, agent, runner=fail)

    assert [attempt for attempt, _feedback in calls] == [1, 2]
    assert calls[1][1] == "still failing"
    assert len(grades) == 2 and not any(grade.passed for grade in grades)
    assert not (workspace.agent_dir / HIDDEN_TEST).exists()


def test_agents_receive_separate_but_equivalent_workspace_copies(
    tmp_path: Path,
) -> None:
    source = make_exercise(tmp_path)
    staged = prepare_exercise(source, tmp_path / "staged")
    c_workspace = clone_workspace(staged, tmp_path / "c")
    python_workspace = clone_workspace(staged, tmp_path / "ds4py")

    def files(root: Any) -> Any:
        return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    assert files(c_workspace.agent_dir) == files(python_workspace.agent_dir)
    assert c_workspace.agent_dir != python_workspace.agent_dir
    (c_workspace.agent_dir / "solution.py").write_text("def value():\n    return 'C'\n")
    assert "return 'good'" in (python_workspace.agent_dir / "solution.py").read_text()


def test_check_only_source_validation_rejects_missing_student_config(tmp_path: Path) -> None:
    source = make_exercise(tmp_path)
    (source / ".meta" / "config.json").unlink()
    benchmark = source.parents[3]
    revision = commit_repo(benchmark)
    manifest = replace(
        load_manifest(Path(__file__).parents[2] / "bench" / "bakeoff.yaml"),
        commit=revision,
        exercises=(str(EXERCISE),) * 10,
    )

    with pytest.raises(PreparationError, match=r"\.meta/config\.json"):
        validate_manifest_sources(manifest, benchmark)


def test_wrong_benchmark_commit_fails_clearly(tmp_path: Path) -> None:
    source = make_exercise(tmp_path)
    benchmark = source.parents[3]
    actual = commit_repo(benchmark)
    expected = "0" * 40

    with pytest.raises(BenchmarkError) as caught:
        verify_benchmark(benchmark, expected)

    message = str(caught.value)
    assert expected in message and actual in message


def test_ctx_repository_copy_keeps_hidden_grader_outside(tmp_path: Path) -> None:
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
        repo,
        tmp_path / "staged",
        task_id="repo-ctx-command",
        prompt="Add /ctx without generation.",
        grader=external_grader,
    )

    assert (workspace.agent_dir / "TASK.md").read_text() == "Add /ctx without generation."
    assert not (workspace.agent_dir / "bench").exists()
    assert not (workspace.agent_dir / "tests" / "bakeoff_hidden" / "test_ctx_command.py").exists()
    assert (workspace.grader_dir / "tests" / "bakeoff_hidden" / "test_ctx_command.py").is_file()
    assert (workspace.agent_dir / "tests" / "engine" / "tool_syntax" / "test_glm.py").is_file()


def test_glm_patch_fails_focused_test_and_clean_tree_passes(tmp_path: Path) -> None:
    repo = Path(__file__).parents[2]
    common: dict[str, Any] = dict(  # noqa: C408 - Keyword-built task context is clearer here.
        prompt="Fix the GLM close boundary without editing tests.",
        visible_test="tests/engine/tool_syntax/test_glm.py",
    )
    clean = prepare_repository_task(
        repo,
        tmp_path / "clean",
        task_id="repo-glm-clean",
        **common,
    )
    patched = prepare_repository_task(
        repo,
        tmp_path / "patched",
        task_id="repo-glm-close",
        patch=repo / "bench" / "patches" / "glm-off-by-one.patch",
        **common,
    )

    clean_result = grade_workspace(clean)
    patched_result = grade_workspace(patched)

    assert clean_result.passed, clean_result.output
    assert not patched_result.passed
    assert "test_glm" in patched_result.output


def test_workspace_cannot_bypass_grading_with_startup_or_pytest_configuration(
    tmp_path: Path,
) -> None:
    source = make_exercise(tmp_path, initial="bad")
    workspace = prepare_exercise(source, tmp_path / "staged")
    agent_dir = workspace.agent_dir
    (agent_dir / "conftest.py").write_text("import solution\nsolution.value = lambda: 'good'\n")
    (agent_dir / "workspace_bypass.py").write_text("import solution\nsolution.value = lambda: 'good'\n")
    (agent_dir / "pytest.ini").write_text("[pytest]\naddopts = -p workspace_bypass\n")
    (agent_dir / "src").mkdir()
    (agent_dir / "src" / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(agent_dir / 'solution.py')!r}).write_text("
        "\"def value():\\n    return 'good'\\n\")\n"
    )

    result = grade_workspace(workspace)

    assert not result.passed
    assert "return 'bad'" in (agent_dir / "solution.py").read_text()


@pytest.mark.parametrize("content", ["", "---\n", "- not\n- a\n- mapping\n", "scalar\n"])
def test_empty_or_non_mapping_manifest_is_a_path_aware_preparation_error(tmp_path: Path, content: Any) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(content)

    with pytest.raises(PreparationError) as caught:
        load_manifest(path)

    assert str(path) in str(caught.value)
    assert "mapping" in str(caught.value)
