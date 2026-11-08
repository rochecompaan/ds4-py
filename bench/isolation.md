# Bakeoff workspace isolation and hidden grading

There are two different guarantees here: **repeatable workspace isolation**, which the bakeoff implements, and **strong security isolation**, which it does not currently provide.

## Current isolation model

- **Each agent/task gets a fresh workspace.**
  `clone_workspace()` copies a pristine, sanitized task into a separate directory for each agent and task, preventing one run from contaminating another.

- **The agent runs only with its workspace as the working directory.**
  `run_task()` starts each attempt with `cwd=workspace.agent_dir` and an attempt-specific `DS4PY_HOME` for session data.

- **Runs are sequential.**
  The runner waits for the agent process to exit before grading, then finishes that run before cloning or starting the next one.

- **Grading is hardened against workspace configuration.**
  `grade_workspace()` starts Pytest from the grader directory with grader-owned configuration, `--noconftest`, and no cache plugin. Pytest is imported before the agent’s source path is added, preventing an agent-provided `sitecustomize.py` from modifying the solution during startup.

## How hidden grader answers are handled

During exercise staging, `prepare_exercise()` uses an allowlist rather than copying every non-test file:

- `.meta/config.json` declares the student solution and support files but is not copied into the workspace.
- Source `.docs/` becomes visible `docs/`; `TASK.md` still contains the main instructions.
- Only `TASK.md`, `docs/`, and declared student files go into `workspace/`. Everything else is denied by default.
- `.meta/`, example solutions, test sources, bytecode, `__pycache__/`, and pytest caches never go into the agent workspace. Declared and documentation paths, including resolved symlink targets, are validated before staging.
- `--check-only` applies the same student-input and documentation validation without writing workspaces.
- Exercise tests go into a separate `grader/` directory.
- Repository-task preparation is separate; its archived `bench/` directory is removed so the agent cannot see the bundled grader.

During grading:

1. The agent process has already exited.
2. Tests remain in `grader/`; they are never mounted or copied into the workspace.
3. Pytest imports the student solution through explicit import paths.
4. `PYTHONDONTWRITEBYTECODE=1` prevents solution or test imports from leaving bytecode in the agent workspace.

Tests, grading metadata, and derived grading artifacts are absent during normal agent execution and between attempts.

**Important exception:** after a failed first attempt, `run_task()` deliberately includes the grader’s stdout/stderr in the second-attempt prompt. Therefore, expected values or assertions printed by Pytest may become visible. Grader output is also retained in the results.

## What this does not guarantee

This is **logical workspace separation, not a security sandbox**:

- The agent inherits the host environment and runs as the same OS user.
- There is no container, filesystem sandbox, network restriction, or separate user.
- A determined agent could read `../grader`, another checkout, or files outside its workspace.
- `DS4PY_HOME` isolates ds4-py state; it does not hide the real home directory.
- Code under test can inspect the hidden tests while grading.
