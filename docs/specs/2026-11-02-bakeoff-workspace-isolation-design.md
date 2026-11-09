# Bake-off workspace isolation design

## Intent

Keep every grading-related artifact out of the agent-observable exercise workspace. Use ordinary, non-hidden directory names for material that the agent is allowed to see.

The retained `tree-building` attempt showed why this matters: staging exposed `.meta/config.json`, `.meta/example.py`, and compiled pytest bytecode. The agent spent its response budget trying to reconstruct the external tests instead of implementing the task.

## Goals

- Stage only student-deliverable exercise inputs.
- Keep test sources, example solutions, grading metadata, bytecode, and caches out of the agent workspace before attempt 1 and after every grading call.
- Emit task documentation under `docs/`, not `.docs/`.
- Use `meta/` as the non-hidden name for any metadata explicitly allowed in the future. Current grading-only metadata is not allowed, so current exercises do not emit a `meta/` directory.
- Give both agent arms the same sanitized inputs.
- Preserve external grading and the existing second-attempt feedback contract.

## Non-goals

- Rename directories in the pinned upstream Exercism checkout.
- Change repository-task staging, which already has a separate preparation path.
- Add prompt instructions as a substitute for isolation.
- Add OS-level sandboxing. The existing documentation already states that logical workspace separation is not a security sandbox.

## Constraints and current behavior

- Upstream exercises use `.docs/` and `.meta/`; those names remain valid at the source-input boundary.
- `prepare_exercise()` currently copies every non-test file, which reproduces the metadata, example-solution, and test-bytecode disclosures.
- Commit `3bb3a05` already removed grading-time test mounting. `grade_workspace()` runs tests from `grader_dir` and adds the agent solution paths to the test process import path.
- `.docs/instructions.append.md` is legitimate task documentation even when it mentions tests. It remains visible under `docs/`.

## Chosen approach

Use an allowlist at the staging boundary.

1. Read the upstream `.meta/config.json` as staging input; never copy it into the agent workspace.
2. Copy `.docs/` to agent-visible `docs/` and continue generating `TASK.md` from the main instructions.
3. Copy the files declared by `files.solution`, including declared support files needed by the student.
4. Keep discovered exercise tests in `grader_dir`; do not copy any test source or derived test artifact into the agent workspace.
5. Exclude everything else by default, including `.meta/`, example solutions, `__pycache__/`, pytest caches, and bytecode.
6. Validate declared paths as existing, relative input files that cannot escape the source directory or reintroduce test/metadata paths.

A denylist was considered and rejected because it would only block artifact types already known to us. An allowlist makes new or unexpected files private by default.

## Verification

Add a behavioral regression fixture containing:

- legitimate instructions and solution files;
- `.meta/config.json` and `.meta/example.py`;
- a hidden test source;
- compiled test bytecode and cache artifacts.

Verify that after staging, the agent has `TASK.md`, `docs/`, and declared solution/support files, but none of the grading material. Run real grading twice and verify that no hidden source, bytecode, or cache appears afterward. Keep the existing retry and grader-hardening tests passing.

Run the focused bench tests and then the complete non-live suite. After implementation, prepare the real exercises and inspect their agent-visible file sets before any model run. A fresh targeted `tree-building` run and full bake-off are required for new Gate 2 evidence because the sanitized inputs change the benchmark contract; previous results remain historical evidence.
