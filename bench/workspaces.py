"""Pinned bakeoff workspace staging and hidden-test grading.

Agents see only ``Workspace.agent_dir``. Test files live in the separate
``grader_dir`` and are mounted only for the duration of one grading call.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from io import BytesIO
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


class BenchError(RuntimeError):
    """Base error for benchmark preparation and grading."""


class BenchmarkError(BenchError):
    """The external benchmark checkout does not match the manifest."""


class PreparationError(BenchError):
    """A task cannot be staged without changing its grading contract."""


@dataclass(frozen=True)
class GradeResult:
    passed: bool
    output: str


@dataclass(frozen=True)
class Manifest:
    path: Path
    repository: str
    commit: str
    selection_seed: int
    model: str
    ctx: int
    temperature: float
    seed: int
    max_attempts: int
    exercises: tuple[str, ...]
    repository_tasks: tuple[dict, ...]


@dataclass(frozen=True)
class Workspace:
    task_id: str
    agent_dir: Path
    grader_dir: Path
    test_paths: tuple[str, ...]
    hidden_tests: tuple[str, ...] = ()
    max_attempts: int = 2

    def __post_init__(self):
        if self.max_attempts not in {1, 2}:
            raise ValueError("Bakeoff workspaces allow one or two attempts")


def _reset_directory(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _is_test_file(path):
    name = path.name
    return path.suffix == ".py" and (
        name.endswith("_test.py") or name.startswith("test_") or name == "tests.py"
    )


def _copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare_exercise(source, destination, *, max_attempts=2):
    """Stage one Exercism task with its tests held outside the agent view."""
    source = Path(source)
    destination = Path(destination)
    instructions = source / ".docs" / "instructions.md"
    if not instructions.is_file():
        raise PreparationError(f"{source}: missing .docs/instructions.md")
    _reset_directory(destination)
    agent_dir = destination / "workspace"
    grader_dir = destination / "grader"
    agent_dir.mkdir()
    grader_dir.mkdir()
    hidden = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(source)
        if _is_test_file(path):
            _copy_file(path, grader_dir / relative)
            hidden.append(str(relative))
        else:
            _copy_file(path, agent_dir / relative)
    if not hidden:
        raise PreparationError(f"{source}: no Python exercise tests found")
    (agent_dir / "TASK.md").write_text(instructions.read_text())
    return Workspace(
        source.name, agent_dir, grader_dir, tuple(hidden), tuple(hidden), max_attempts,
    )


def archive_repository(repo_root, destination):
    """Copy the current committed repository state without its ``.git`` directory."""
    result = subprocess.run(
        ["git", "archive", "HEAD"], cwd=repo_root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise PreparationError(
            f"git archive HEAD failed in {repo_root}: {result.stderr.decode(errors='replace')}"
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(result.stdout), mode="r:") as archive:
        archive.extractall(destination, filter="data")


def prepare_repository_task(
    repo_root, destination, *, task_id, prompt, patch=None, visible_test=None,
    grader=None, max_attempts=2,
):
    """Stage a repository task from ``HEAD`` and optionally plant one patch."""
    destination = Path(destination)
    _reset_directory(destination)
    agent_dir = destination / "workspace"
    grader_dir = destination / "grader"
    grader_dir.mkdir()
    archive_repository(repo_root, agent_dir)
    # The archived bench directory contains this grader and must never be visible.
    shutil.rmtree(agent_dir / "bench", ignore_errors=True)
    if patch is not None:
        patch = Path(patch).resolve()
        result = subprocess.run(
            ["git", "apply", "--whitespace=error", str(patch)], cwd=agent_dir,
            capture_output=True, text=True,
        )
        if result.returncode:
            raise PreparationError(
                f"could not apply {patch} to {task_id}: {result.stdout}{result.stderr}"
            )
    (agent_dir / "TASK.md").write_text(prompt)
    hidden = []
    if grader is not None:
        grader = Path(grader)
        relative = Path("tests") / "bakeoff_hidden" / grader.name
        _copy_file(grader, grader_dir / relative)
        hidden.append(str(relative))
    test_paths = list(hidden)
    if visible_test is not None:
        if not (agent_dir / visible_test).is_file():
            raise PreparationError(f"{task_id}: visible test is missing: {visible_test}")
        test_paths.append(visible_test)
    if not test_paths:
        raise PreparationError(f"{task_id}: no grading tests configured")
    return Workspace(
        task_id, agent_dir, grader_dir, tuple(test_paths), tuple(hidden), max_attempts,
    )


def clone_workspace(workspace, destination):
    """Make an independent agent/grader copy from one pristine staged task."""
    destination = Path(destination)
    if destination.exists():
        raise PreparationError(f"refusing to replace existing agent workspace: {destination}")
    destination.mkdir(parents=True)
    agent_dir = destination / "workspace"
    grader_dir = destination / "grader"
    shutil.copytree(workspace.agent_dir, agent_dir)
    shutil.copytree(workspace.grader_dir, grader_dir)
    return replace(workspace, agent_dir=agent_dir, grader_dir=grader_dir)


@contextmanager
def _mounted_tests(workspace):
    backups = {}
    mounted = []
    try:
        for name in workspace.hidden_tests:
            source = workspace.grader_dir / name
            destination = workspace.agent_dir / name
            if not source.is_file():
                raise PreparationError(f"{workspace.task_id}: hidden test is missing: {name}")
            if destination.is_symlink() or not destination.resolve().is_relative_to(
                workspace.agent_dir.resolve()
            ):
                raise PreparationError(f"{workspace.task_id}: unsafe test mount: {name}")
            backups[destination] = destination.read_bytes() if destination.is_file() else None
            _copy_file(source, destination)
            mounted.append(destination)
        yield
    finally:
        for destination in reversed(mounted):
            original = backups[destination]
            if original is None:
                destination.unlink(missing_ok=True)
            else:
                destination.write_bytes(original)
            parent = destination.parent
            while parent != workspace.agent_dir:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent


def _combined_output(completed):
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    separator = "\n" if stdout and stderr and not stdout.endswith("\n") else ""
    return stdout + separator + stderr


def grade_workspace(workspace, *, runner=None):
    """Mount hidden tests, capture one grading result, then unmount them."""
    if not workspace.test_paths:
        raise PreparationError(f"{workspace.task_id}: no grading tests configured")
    grader_config = workspace.grader_dir / "pytest.ini"
    grader_config.write_text("[pytest]\naddopts =\n")
    # Import pytest before adding the workspace package path. In particular,
    # Python must not import an agent-provided sitecustomize during startup.
    bootstrap = (
        "import sys; import pytest; "
        "sys.path.insert(0, sys.argv.pop(1)); "
        "raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    command = [
        sys.executable, "-c", bootstrap, str(workspace.agent_dir / "src"),
        "-q", "-c", str(grader_config),
        f"--rootdir={workspace.grader_dir}", "--noconftest",
        "-p", "no:cacheprovider",
        *(str(workspace.agent_dir / name) for name in workspace.test_paths),
    ]
    environment = os.environ.copy()
    agent_src = (workspace.agent_dir / "src").resolve()
    pythonpath = [
        entry for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
        if entry and Path(entry).resolve() != agent_src
    ]
    if pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    else:
        environment.pop("PYTHONPATH", None)
    if runner is None:
        def runner(command, cwd, env):
            return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    with _mounted_tests(workspace):
        completed = runner(command, workspace.grader_dir, environment)
    return GradeResult(completed.returncode == 0, _combined_output(completed))


def run_attempts(workspace, agent, *, runner=None):
    """Run at most two agent/grade rounds, feeding failure output to the next call."""
    grades = []
    feedback = ""
    for attempt in range(1, workspace.max_attempts + 1):
        agent(workspace.agent_dir, attempt, feedback)
        grade = grade_workspace(workspace, runner=runner)
        grades.append(grade)
        if grade.passed:
            break
        feedback = grade.output
    return grades


def verify_benchmark(benchmark, expected_commit):
    """Require the existing external checkout to be at the pinned commit."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=benchmark, capture_output=True, text=True,
    )
    if result.returncode:
        raise BenchmarkError(f"{benchmark}: not a readable git checkout: {result.stderr.strip()}")
    actual = result.stdout.strip()
    if actual != expected_commit:
        raise BenchmarkError(
            f"benchmark commit mismatch: expected {expected_commit}, got {actual}"
        )
    return actual


def load_manifest(path):
    """Load the fixed task list; preparation performs behavioral validation."""
    import yaml

    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise PreparationError(f"{path}: manifest must contain a mapping")
    required = {
        "repository", "commit", "selection_seed", "model", "ctx", "temperature",
        "seed", "max_attempts", "exercises", "repository_tasks",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise PreparationError(f"{path}: missing manifest keys: {', '.join(missing)}")
    if len(data["exercises"]) != 10 or len(data["repository_tasks"]) != 2:
        raise PreparationError(f"{path}: bakeoff requires ten exercises and two repository tasks")
    return Manifest(
        path=path,
        repository=str(data["repository"]),
        commit=str(data["commit"]),
        selection_seed=int(data["selection_seed"]),
        model=str(data["model"]),
        ctx=int(data["ctx"]),
        temperature=float(data["temperature"]),
        seed=int(data["seed"]),
        max_attempts=int(data["max_attempts"]),
        exercises=tuple(str(item) for item in data["exercises"]),
        repository_tasks=tuple(dict(item) for item in data["repository_tasks"]),
    )


def validate_manifest_sources(manifest, benchmark):
    """Check the pinned checkout and required exercise inputs without staging."""
    benchmark = Path(benchmark)
    verify_benchmark(benchmark, manifest.commit)
    for exercise in manifest.exercises:
        source = benchmark / exercise
        if not (source / ".docs" / "instructions.md").is_file():
            raise PreparationError(f"{source}: missing .docs/instructions.md")
        if not any(path.is_file() and _is_test_file(path) for path in source.rglob("*")):
            raise PreparationError(f"{source}: no Python exercise tests found")


def prepare_manifest(manifest, output, benchmark, repo_root):
    """Materialize all twelve pinned tasks in manifest order."""
    validate_manifest_sources(manifest, benchmark)
    output = Path(output)
    _reset_directory(output)
    workspaces = []
    for exercise in manifest.exercises:
        workspaces.append(prepare_exercise(
            Path(benchmark) / exercise, output / Path(exercise).name,
            max_attempts=manifest.max_attempts,
        ))
    for spec in manifest.repository_tasks:
        patch = spec.get("patch")
        grader = spec.get("grader")
        workspaces.append(prepare_repository_task(
            repo_root, output / spec["id"], task_id=spec["id"], prompt=spec["prompt"],
            patch=manifest.path.parent / patch if patch else None,
            visible_test=spec.get("visible_test"),
            grader=manifest.path.parent / grader if grader else None,
            max_attempts=manifest.max_attempts,
        ))
    return workspaces
