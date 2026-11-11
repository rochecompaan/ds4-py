# Python Quality Gates Design

## Intent

Establish one strict, reproducible quality workflow before the codebase grows further. Make the same static checks run locally on macOS and Linux, in agent workflows, and in CI. Use PR #6's completed annotation work as the baseline rather than adding a parallel or weaker toolchain.

## Goals

- Replace Pyright with ty as the only project type checker.
- Use Ruff for both broad linting and formatting across `src/`, `bench/`, and `tests/`.
- Run all static checks through prek with a checked-in `.pre-commit-config.yaml`.
- Keep tool behavior consistent between the existing devenv + uv environment and GitHub Actions.
- Make `Any` an explicit, narrowly scoped exception at dynamic boundaries, not a default annotation.
- Adopt modern Python 3.14 typing and `typing_extensions` 4.16 features where they improve contracts.
- Validate TOML, YAML, JSON, and GitHub Actions configuration, not only Python.
- Add concise root `AGENTS.md` instructions and keep detailed commands in `docs/development.md`.
- Preserve runtime behavior except for mechanical formatting and explicitly verified type-only improvements.

## Non-goals

- Mutation testing, crap analysis, release automation, or deployment changes.
- Permanent dual type checking with both Pyright and ty.
- Enabling every Ruff or ty rule without regard for false positives and project conventions.
- Automatic unsafe fixes or automatic commits to pull requests.
- Rewriting public APIs solely to make a linter happy.
- Replacing annotations mechanically without checking their runtime and public API contracts.
- Changing native model execution, replay semantics, persistence formats, or agent capabilities.

## Anti-goals

The change fails if it produces a large collection of blanket suppressions, leaves CI advisory-only, creates separate local and CI tool versions, or makes routine development slower without improving defect detection. It also fails if formatting and typing edits hide an unreviewed behavioral change.

## Constraints

- Base branch and stacked PR base: `m3-harness-agent` at PR #6 head `6b5d9b912a02e00901eee4cc5f7aaa92b991043a`.
- Feature branch: `build/python-quality-gates` in an isolated worktree.
- Python 3.14 or later, using the minimum supported version for type analysis.
- Existing devenv + uv workflow remains canonical on macOS and Linux. Enter `devenv shell` before dependency synchronization, static checks, tests, builds, and all other project commands. Python tools are locked by `uv.lock`; do not introduce a second Nix-generated hook configuration.
- CI uses frozen lockfile resolution and read-only repository permissions.
- PR #6's portable suite is the behavioral baseline: 1,425 passed, 1 skipped, 17 deselected when the developer's personal skill catalog is isolated from `HOME`.
- No production change is intended for the baseline's dependence on a developer's personal skills; verification commands will use an isolated `HOME` on affected workstations.

## Approaches Considered

### Ruff rule selection

1. **Enable `ALL` immediately.** Rejected for this PR. PR #6 produced 10,938 diagnostics without ignores and 4,188 even with datamodel-code-generator-style exceptions and stable-only preview selection. This would shift the PR from quality infrastructure to mass policy rewrites.
2. **Use only the conventional default groups.** Rejected as too weak for a new project that already has complete annotations.
3. **Use broad categories plus explicit high-value groups and controlled preview rules.** Selected. This follows Ruff's own repository more closely than legacy `ALL` configurations, while adding annotation, async, pytest, path, security, and type-specific checks.

### ty strictness

1. **Use default ty only.** Rejected because it misses useful opt-in checks.
2. **Use `all = "error"`.** Rejected because it automatically enables rules that upstream documents as false-positive-prone and makes upgrades less predictable.
3. **Explicitly enable the useful strict rules and keep the known noisy rules disabled.** Selected. On PR #6 this is stricter than the recommended migration configuration while remaining a manageable remediation effort.

### Any policy

1. **Ignore `ANN401` globally.** Rejected because it loses useful constraints in ordinary modules.
2. **Ban all dynamic typing everywhere.** Rejected because ctypes and some third-party boundaries genuinely need escape hatches.
3. **Default-deny explicit `Any`, with path-scoped exceptions at reviewed dynamic boundaries.** Selected.

### Change organization

1. **Independent branch from `main`.** Rejected because this work depends on PR #6's annotations and agent modules.
2. **Push quality changes directly to PR #6.** Rejected to keep implementation and quality review separate.
3. **Create a stacked branch and PR based on `m3-harness-agent`.** Selected. Rebase or retarget after PR #6 merges.

## Ruff Design

Configure Ruff centrally in `pyproject.toml`:

- `line-length = 120`.
- `preview = true` with `explicit-preview-rules = true` so upgrades do not implicitly enable every new preview rule.
- Select `correctness`, `suspicious`, `complexity`, `performance`, and `style` categories.
- Add `ANN`, `ASYNC`, `BLE`, `DTZ`, `FBT`, `FURB`, `G`, `LOG`, `PGH`, `PIE`, `PL`, `PT`, `PTH`, `PYI`, `RET`, `RSE`, `RUF`, `S`, `SIM`, `SLOT`, `TC`, `TID`, `TRY`, `UP`, and `YTT` as applicable after resolving overlaps.
- Ignore only formatter conflicts and deliberately documented policies, initially `COM812`, `ISC001`, and `E501`.
- Set McCabe `max-complexity = 15`; do not disable `C901` globally.
- Enable Ruff formatting for `src/`, `bench/`, and `tests/`.
- Use safe fixes locally, never `--unsafe-fixes` in hooks.
- Use narrow per-file ignores for tests, CLI output, generated files, and dynamic boundaries. Each exception must have a reason and must not become a permanent blanket category without review.

## Any and Dynamic Boundary Design

Enable both annotation and import-level controls:

- `ANN401` rejects `Any` in function arguments and return annotations.
- `TID251` with `typing.Any` in `banned-api` prevents bypassing the policy through variable annotations, aliases, and nested annotations.
- A dedicated prek `pygrep` hook bans `TypeGuard` in every Python file, independently of path-scoped `TID251` exceptions for `Any`; use `TypeIs` for type-narrowing validators without compatibility aliases.

Default-deny `Any` in production code. Start the exception review with `src/ds4py/_native/**` and direct native interfaces such as `raw.py`, but do not automatically exempt every module that currently imports `Any`. JSON, event, and external API boundaries receive exceptions only when a precise static contract is not practical. Tests and benchmark code may use broader, explicitly scoped exceptions for test doubles and dynamic fixtures.

Use this priority when removing `Any`:

1. A precise domain type, `Protocol`, `TypedDict`, or recursive `JsonValue` alias.
2. A generic when input and output types must be preserved.
3. `object` when arbitrary values are accepted and narrowed before use.
4. `Any` only at a reviewed dynamic boundary.

ty's `unsound-assignment`, `unsound-return-statement`, and `unsound-yield` checks limit leakage from remaining dynamic boundaries into fully static types.

## Modern Typing Design

Add `typing-extensions>=4.16.0,<5` as a direct production dependency; do not rely on a transitive dependency. Use the lockfile for the resolved version.

- Use lowercase `typing_extensions.sentinel`, not the soft-deprecated `Sentinel` alias.
- Use the sentinel object itself as a type, for example `JsonValue | _UNSET`, and narrow with `is` / `is not`; do not use `Literal[_UNSET]`.
- Candidate replacements include the existing `_UNSET`, `_UNKNOWN`, `_END`, and `_STOP` objects, after checking identity, repr, serialization, and copy behavior.
- Use standard-library `typing.TypeIs` on Python 3.14 for validators that genuinely prove a type at runtime. `TypeGuard` is prohibited in both stdlib and `typing_extensions` forms. Review each validator's runtime proof when migrating; do not use `TypeIs` as a suppression mechanism.
- Use `typing_extensions.TypedDict` for `closed` and `extra_items` support. Prefer class syntax; use the dictionary-based functional form when needed, not removed keyword-field syntax.
- Use standard-library `Required`, `NotRequired`, and `ReadOnly` for item contracts. `closed=True` is a static contract and does not replace runtime validation.
- Use PEP 695 `type` aliases and generic syntax where they clarify contracts.
- Use standard-library `override`, `Self`, `Never`, and `assert_never` where applicable. `Never` is limited to non-returning and unreachable paths; no new uses are required merely for modernity.

## ty Design

Configure `[tool.ty]` in `pyproject.toml`:

- Include `src`, `bench`, and `tests`.
- Set `environment.python-version = "3.14"`.
- Set `terminal.error-on-warning = true`.
- Enable the strict analysis options `strict-equality-semantics` and `strict-generic-narrowing`.
- Enable `blanket-ignore-comment`, `dynamic-function-decorator-return`, `missing-type-argument`, `possibly-unresolved-reference`, `unsound-assignment`, `unsound-return-statement`, `unsound-yield`, and `unsupported-dynamic-base` as errors.
- Enable `disjoint-cast`, `invalid-legacy-positional-parameter`, `mismatched-sentinel-name`, `missing-override-decorator`, `truthiness-test-of-none-union`, `redundant-condition-strict`, `unused-ignore-comment`, and `redundant-cast` as errors.
- Leave `division-by-zero`, `possibly-missing-attribute`, and `possibly-missing-import` disabled by default because upstream documents significant false-positive rates.
- Do not enable `missing-direct-dependency` in the initial gate; its uv integration can be evaluated separately.
- Fix diagnostics with precise types, narrowing, protocols, generic signatures, and runtime validators. Avoid whole-module suppressions and unjustified casts.

Remove the Pyright dependency, configuration, and CI invocation after ty and the complete test suite pass.

## prek and Configuration Validation

Use a checked-in `.pre-commit-config.yaml`, not devenv's generated `git-hooks.nix` configuration. Add prek to the development dependency group so local and CI execution both use `uv run prek` and the same lockfile.

Hooks will cover:

- Ruff lint with safe fixes and Ruff formatting through project-local uv commands.
- Whole-project ty checking through the locked project environment; do not pass only changed filenames to ty.
- A dedicated `pygrep` hook that rejects `TypeGuard` in every Python file, including files with dynamic-boundary exceptions.
- `pyproject-fmt` with full dependency versions preserved and automatic Python classifier generation disabled.
- `yamllint` with line length 120, optional document start, and truthy key checking disabled for GitHub's `on:` key while retaining value checks.
- JSON syntax and stable pretty formatting without sorting keys or escaping Unicode unnecessarily.
- YAML and TOML syntax checks.
- General hygiene: EOF, trailing whitespace, mixed line endings, merge conflicts, case conflicts, and related low-risk checks.
- GitHub workflow schema validation and required job timeouts through `check-jsonschema`.
- actionlint through its official hook so CI does not depend on a system-installed binary.

Exclude patch files from whitespace rewrites and avoid fights with generated lockfiles. Pin every remote hook revision. Do not configure pre-commit.ci or autofix.ci.

## CI Design

Replace the existing Pyright-only `typing` job with a `quality` job that runs:

```bash
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
```

Keep the existing Linux/macOS test matrix and native build jobs. Add explicit `timeout-minutes` to every job. Pin GitHub Actions to full commit SHAs, retain least-privilege permissions, and disable persisted checkout credentials where they are not needed. Static checks run once on Ubuntu; runtime tests continue to cover both operating systems.

## AGENTS.md and Development Documentation

Add a concise root `AGENTS.md` that tells coding agents:

- How to enter devenv, synchronize dependencies, and install prek hooks.
- The canonical quality, type-check, and portable test commands.
- The requirement to check all of `src/`, `bench/`, and `tests/`.
- The `Any` exception policy, the `TypeGuard` prohibition in favor of `TypeIs`, and preference for precise types, generics, and `object`.
- The rules for targeted suppressions, modern typing, safe fixes, and behavioral verification.
- When to use focused tests versus the full portable suite.

Keep native build, backend, model, and replay instructions in `docs/development.md`; link to them instead of duplicating them.

## Implementation Sequence

1. Add the locked dependencies and central Ruff/ty configuration; remove Pyright only when the replacement gate is ready.
2. Add prek, YAML/JSON/TOML validation, and their formatting configuration.
3. Apply mechanical Ruff safe fixes and formatting in a separate commit.
4. Resolve remaining Ruff diagnostics by subsystem, keeping tests focused on accidental behavior changes.
5. Migrate type checking to ty by subsystem. Reduce `Any`, add precise payload types, use modern typing, and add `@override` and narrowing where required.
6. Update CI, `AGENTS.md`, and `docs/development.md`.
7. Run the complete local and CI-equivalent verification and review the final diff for hidden behavioral changes.

Mechanical formatting, manual lint fixes, type migration, and workflow/documentation changes should remain separate commits within the stacked PR.

## Verification

Required final checks locally inside devenv on macOS, with equivalent frozen uv commands in Linux CI:

- `devenv shell -- uv run --frozen --extra agent prek run --all-files --show-diff-on-failure`
- `devenv shell -- uv run --frozen --extra agent ty check`
- `devenv shell -- uv run --frozen --extra agent pytest -m "not build and not live" -q`
- `devenv shell -- uv build --wheel` and the existing wheel-content check
- `git diff --check`

Add focused runtime tests whenever a type-oriented change also affects sentinel behavior, validation, signatures, imports, or control flow. Do not add tests that merely restate tool configuration. Existing native, agent, engine, persistence, and benchmark suites are the primary regression evidence. Native build and live CI coverage remains in place; no claim of live model parity will be made from portable tests alone.
