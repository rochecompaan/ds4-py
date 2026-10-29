# Bakeoff workspace isolation and hidden grading

There are two different guarantees here: **repeatable workspace isolation**, which the bakeoff implements, and **strong security isolation**, which it does not currently provide.

## Current isolation model

- **Each agent/task gets a fresh workspace.**
  `clone_workspace()` copies a pristine task into a separate directory for each agent and task, preventing one run from contaminating another (`bench/workspaces.py:172`).

- **The agent runs only with its workspace as the working directory.**
  Each attempt is a subprocess started with `cwd=workspace.agent_dir`. It also gets an attempt-specific `DS4PY_HOME` for session data (`bench/run_bakeoff.py:274-301`).

- **Runs are sequential.**
  The runner waits for the agent process to exit before grading, then finishes that run before cloning or starting the next one (`bench/run_bakeoff.py:298`, `bench/run_bakeoff.py:343`).

- **Grading is hardened against workspace configuration.**
  Pytest starts from the grader directory with grader-owned configuration, `--noconftest`, and no cache plugin. Pytest is imported before the agent’s source path is added, preventing an agent-provided `sitecustomize.py` from modifying the solution during startup (`bench/workspaces.py:230-265`).

## How hidden grader answers are handled

During staging:

- Instructions and starter source go into `workspace/`.
- Exercise tests go into a separate `grader/` directory (`bench/workspaces.py:84-110`).
- For repository tasks, the archived `bench/` directory is removed so the agent cannot see the bundled grader (`bench/workspaces.py:140-159`).

During grading:

1. The agent process has already exited.
2. Hidden tests are temporarily copied into the workspace.
3. Pytest runs.
4. A `finally` block removes the tests or restores any files they replaced (`bench/workspaces.py:185-216`).

So the tests are absent during normal agent execution and between attempts.

**Important exception:** after a failed first attempt, the grader’s stdout/stderr is deliberately included in the second-attempt prompt (`bench/run_bakeoff.py:281-285`). Therefore, expected values or assertions printed by Pytest may become visible. Grader output is also retained in the results.

## What this does not guarantee

This is **logical workspace separation, not a security sandbox**:

- The agent inherits the host environment and runs as the same OS user.
- There is no container, filesystem sandbox, network restriction, or separate user.
- A determined agent could read `../grader`, another checkout, or files outside its workspace.
- `DS4PY_HOME` isolates ds4-py state; it does not hide the real home directory.
- Code under test can inspect the hidden tests while grading.
