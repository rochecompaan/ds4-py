# Bakeoff Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace milestone-oriented names in persistent benchmark code and user-facing documentation with domain names while preserving behavior.

**Architecture:** Rename modules and artifacts together, then update every import, environment variable, path, and user-facing reference. Keep the implementation and tests unchanged apart from names and descriptive text.

**Tech Stack:** Python, pytest, YAML, Markdown, git.

**Spec:** User request and scope approval in the current session on 2026-10-29.

## Global Constraints

- Leave `docs/specs/` and existing historical plans unchanged.
- Leave the current branch and pull request names unchanged.
- Do not add compatibility aliases for the old benchmark names.
- Do not change benchmark behavior, gate rules, or report schemas.
- Use existing tests for regression coverage. Do not add tests that only assert filenames or static documentation.

## Review Focus

- Stale module imports: test collection must fail before the production rename and pass afterward.
- Mismatched producer and consumer environment names: runner and restore tests must pass.
- Stale manifest and report paths: command smoke tests and runner tests must pass.
- A changed hidden-grader path: workspace isolation tests must pass.
- Stale user-facing references: the final scoped search must return no milestone names.

---

### Task 1: Rename the benchmark and its tests

**Files:**
- Rename: `bench/m3.py` to `bench/workspaces.py`
- Rename: `bench/prepare_m3.py` to `bench/prepare_bakeoff.py`
- Rename: `bench/run_m3.py` to `bench/run_bakeoff.py`
- Rename: `bench/replay_m3.py` to `bench/replay_c_traces.py`
- Rename: `bench/m3.yaml` to `bench/bakeoff.yaml`
- Rename: `tests/bench/test_m3_metrics.py` to `tests/bench/test_metrics.py`
- Rename: `tests/bench/test_m3_replay.py` to `tests/bench/test_bakeoff_replay.py`
- Rename: `tests/bench/test_m3_restore.py` to `tests/bench/test_bakeoff_restore.py`
- Rename: `tests/bench/test_m3_runner.py` to `tests/bench/test_runner.py`
- Rename: `tests/bench/test_m3_workspace.py` to `tests/bench/test_workspaces.py`
- Rename: `tests/live/test_m3_bakeoff.py` to `tests/live/test_bakeoff.py`
- Modify: Other files in `bench/`, `tests/`, `src/ds4py/model.py`, and `src/ds4py/tui_commands.py` that reference old names

**Interfaces:**
- Consumes: Current behavior tests and fake agent drivers.
- Produces: `bench.workspaces`, `bench.run_bakeoff`, `bench.replay_c_traces`, `bakeoff.yaml`, `bakeoff.json`, `BAKEOFF_*`, `DS4_BAKEOFF_BACKEND`, and `DS4_RUN_BAKEOFF`.

- [ ] Rename the test files and update their imports, environment variables, report paths, and descriptive names.
- [ ] Run `uv run --extra agent pytest tests/bench tests/live/test_bakeoff.py --collect-only -q` and confirm collection fails because the desired production modules do not exist.
- [ ] Rename the benchmark files and update all imports, paths, environment variables, hidden-test paths, errors, comments, and docstrings.
- [ ] Run `uv run --extra agent pytest tests/bench tests/agent tests/live/test_bakeoff.py -q` and require all selected tests to pass.
- [ ] Run the three renamed commands with `--help` and require each to exit successfully.

### Task 2: Update persistent documentation and verify

**Files:**
- Modify: `README.md`, `docs/agent.md`, `bench/README.md`, `bench/isolation.md`, `.gitignore`

- [ ] Replace command paths, artifact names, environment variables, and milestone-oriented prose with the new names.
- [ ] Search tracked persistent files for stale milestone names and old paths, excluding specs and historical plans.
- [ ] Run `git diff --check`.
- [ ] Run `uv run --extra agent pytest -q` and require the complete suite to pass.

### Task 3: Commit the refactor

- [ ] Review `git status`, `git diff --stat`, and the rename summary.
- [ ] Stage only the planned renames and reference updates.
- [ ] Commit with `refactor(bench): replace milestone names with domain names`.
- [ ] Report the commit and verification evidence for user review.
