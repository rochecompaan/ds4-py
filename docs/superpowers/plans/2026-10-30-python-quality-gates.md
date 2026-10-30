# Python Quality Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PR #6's minimal Pyright/Ruff checks with a strict, reproducible ty, Ruff, prek, configuration-lint, CI, and agent-documentation workflow.

**Architecture:** Keep all tool configuration in `pyproject.toml` and a checked-in prek configuration, run Python tools through the locked uv environment inside devenv, and remediate diagnostics subsystem by subsystem. Centralize JSON contracts and `TypeIs` validators in one small module, keep `Any` behind path-scoped dynamic boundaries, and preserve PR #6 behavior through focused and full regression tests.

**Tech Stack:** Python 3.14+, devenv, uv, ty 0.0.95, Ruff 0.17.3, prek 0.5.7, typing-extensions 4.16, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-30-python-quality-gates-design.md`

## Global Constraints

- Base all work on `m3-harness-agent` at `6b5d9b912a02e00901eee4cc5f7aaa92b991043a`; keep the stacked PR base unchanged.
- Run every local dependency, static-check, test, and build command inside `devenv shell`; use `UV_PYTHON_DOWNLOADS=never` and the Nix Python supplied by devenv.
- Use `uv run --frozen --extra agent` for project checks and `uv sync --frozen --extra agent` after lockfile changes; CI uses equivalent frozen uv commands.
- Check `src/`, `bench/`, and `tests/`; preserve runtime behavior and public signatures unless a focused test justifies a correction.
- `TypeGuard` is forbidden in every Python file, including native and test exceptions; use `TypeIs` only for validators that prove their claimed type at runtime.
- `Any` is default-denied in production through `ANN401` and `TID251`; only reviewed dynamic boundaries receive path-scoped exceptions.
- Prefer precise domain types and `TypedDict`, then generics, then `object`; retain `Any` only where no practical static contract exists.
- Use lowercase `typing_extensions.sentinel`, standard-library `TypeIs` and `override`, and `typing_extensions.TypedDict` for `closed` / `extra_items`.
- Do not use unsafe automatic fixes, blanket suppression comments, permanent Pyright/ty dual checking, pre-commit.ci, or autofix.ci.
- Keep mechanical formatting, subsystem remediation, and workflow/documentation changes in separate commits.

## Review Focus

- Nested JSON with non-string keys, tuples, custom objects, and recursive containers must be accepted or rejected according to the `JsonValue` contract; Task 2 adds runtime acceptance/rejection tests and ty-checked narrowing.
- Sentinel replacement must preserve identity comparisons and queue/unset behavior; Tasks 4 and 5 run focused worker/model/system-notes tests and add copy/repr tests only where those behaviors are used.
- Test-double and callback signature edits must not change monkeypatching, cancellation, streaming, or restore behavior; Tasks 3-6 run the corresponding native, engine, agent, and benchmark suites after every subsystem.
- Patch files and JSON fixtures must not be semantically altered by hygiene hooks; Task 1 scopes whitespace hooks and Task 7 runs tool-syntax tests plus final diff review.
- A developer's personal `~/.claude/skills` catalog must not contaminate verification; Task 7 runs the full portable suite with an empty isolated `HOME` and documents that environmental requirement for affected local runs.

---

### Task 1: Toolchain and prek configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `.pre-commit-config.yaml`
- Create: `.yamllint.yaml`

**Interfaces:**
- Consumes: PR #6's existing uv environment and annotation-only Ruff configuration.
- Produces: Locked `ty`, `prek`, and direct `typing-extensions` dependencies; final Ruff/ty rule configuration; `devenv shell -- uv run --frozen --extra agent prek run --all-files` as the eventual static entry point.

- [x] **Step 1: Add and lock the required tools**

Run `devenv shell -- uv add --dev ty==0.0.95 prek==0.5.7` and `devenv shell -- uv add 'typing-extensions>=4.16.0,<5'`. Keep Pyright temporarily so the old gate remains available during migration. Inspect the resolved dependency-group and production dependency versions in `pyproject.toml` and `uv.lock`.

- [x] **Step 2: Configure Ruff and ty from the spec**

Set Ruff line length 120, `preview = true`, `explicit-preview-rules = true`, the approved category/group selection, `COM812`/`ISC001`/`E501` ignores, and McCabe complexity 15. Add `typing.Any` to `lint.flake8-tidy-imports.banned-api`. Configure ty for all three source roots, Python 3.14, warnings-as-errors, strict analysis, and every approved opt-in rule; leave the three noisy rules disabled.

- [x] **Step 3: Add prek and YAML configuration**

Add pinned hooks for project-local Ruff and ty commands, `pyproject-fmt` with `--keep-full-version --no-generate-python-version-classifiers`, yamllint, JSON/TOML/YAML checks, general hygiene, `check-github-workflows`, `check-github-workflows-require-timeout`, and official actionlint. Configure yamllint for line length 120, optional document start, and `truthy.check-keys = false`; configure JSON pretty formatting with unsorted keys and preserved Unicode. Add a `pygrep` hook whose regex is `\bTypeGuard\b`, whose message requires `TypeIs`, and which has no per-file exceptions. Exclude `docs/patches/**` from whitespace rewriting and prevent generated-lockfile formatting conflicts.

- [x] **Step 4: Record the initial diagnostic baseline without modifying files or weakening the rules**

Run:

```bash
devenv shell -- uv run --frozen --extra agent ruff check src bench tests --statistics
devenv shell -- uv run --frozen --extra agent ty check src bench tests --output-format concise
devenv shell -- uv run --frozen --extra agent prek run ty forbid-type-guard actionlint check-json check-yaml check-toml check-github-workflows --all-files --show-diff-on-failure
```

Use only non-mutating hooks at this stage; defer formatting and safe-fix hooks to Task 2. Expected: Ruff, ty, and the timeout/schema checks report existing diagnostics; `TypeGuard`, actionlint, and syntax checks execute successfully, and no command is missing. Do not add broad ignores to make this step pass.

- [x] **Step 5: Commit the toolchain configuration**

```bash
git add pyproject.toml uv.lock .pre-commit-config.yaml .yamllint.yaml
git commit -m "build: configure strict Python quality tools"
```

### Task 2: Mechanical Ruff remediation and shared JSON contracts

**Files:**
- Modify: Python files under `src/`, `bench/`, and `tests/` that Ruff safely rewrites or formats
- Create: `src/ds4py/_json.py`
- Create: `tests/test_json.py`

**Interfaces:**
- Consumes: Task 1's Ruff configuration and modern typing dependencies.
- Produces:
  - `type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]`
  - `type JsonObject = dict[str, JsonValue]`
  - `is_json_value(value: object) -> TypeIs[JsonValue]`
  - `is_json_object(value: object) -> TypeIs[JsonObject]`

- [x] **Step 1: Apply only safe mechanical Ruff changes**

Run `devenv shell -- uv run --frozen --extra agent ruff check src bench tests --fix`, review every changed class of fix, then run `ruff format src bench tests`. Do not use `--unsafe-fixes` and do not manually mix behavioral edits into the mechanical commit.

- [x] **Step 2: Verify the mechanical changes before adding new code**

Run `HOME=/tmp/ds4-quality-empty-home devenv shell -- uv run --frozen --extra agent pytest -m "not build and not live" -q`. Expected: all portable tests pass. Commit only the mechanical changes as `style: apply Ruff formatting and safe fixes`.

- [x] **Step 3: Write failing JSON validator tests**

In `tests/test_json.py`, cover every scalar, nested lists/objects, empty containers, rejection of tuples and custom objects, rejection of non-string object keys, and rejection when one deeply nested value is invalid. Include a ty-checked helper that accepts `object`, narrows it with `is_json_value`, and passes the result to a `JsonValue` consumer.

- [x] **Step 4: Run the JSON tests to verify they fail**

Run `devenv shell -- uv run --frozen --extra agent pytest tests/test_json.py -q`. Expected: collection or import failure because `ds4py._json` does not exist.

- [x] **Step 5: Implement the JSON aliases and recursive TypeIs validators**

Create `src/ds4py/_json.py` with the four public interfaces above. Accept only JSON-shaped runtime values, require string object keys, recurse through lists and dictionaries, and use standard-library `TypeIs`. Do not add `Any`, `TypeGuard`, or runtime behavior unrelated to validation.

- [x] **Step 6: Verify the new module and commit**

Run the focused pytest test, `ruff check src/ds4py/_json.py tests/test_json.py`, `ruff format --check` on those files, and `ty check src/ds4py/_json.py tests/test_json.py`. Expected: all pass. Commit `feat: add typed JSON boundary validators`.

### Task 3: Native, ABI, and raw static gate

**Files:**
- Modify: `src/ds4py/_native/**`
- Modify: `src/ds4py/abi/**`
- Modify: `src/ds4py/raw.py`
- Modify: `.pre-commit-config.yaml` and `pyproject.toml` only for reviewed native-boundary exceptions

**Interfaces:**
- Consumes: Task 1 rules and Task 2 JSON contracts where raw payloads need them.
- Produces: `ruff check src/ds4py/_native src/ds4py/abi src/ds4py/raw.py` and `ty check` on the same paths with zero diagnostics; a minimal documented native `Any` exception list.

- [ ] **Step 1: Triage native diagnostics by boundary**

Separate genuine ctypes/dynamic signatures from ordinary code. Keep `Any` only in the smallest necessary `_native` or raw interfaces; use `object`, precise handle/pointer types, `JsonValue`, `Generator`/`AsyncGenerator`, and parameterized generics elsewhere. Replace string-based deprecated ctypes annotations where the runtime contract is preserved.

- [ ] **Step 2: Resolve native and ABI Ruff diagnostics**

Fix imports, context-manager annotations, subprocess/path/security findings, exception behavior, and complexity without changing native calls or file formats. Add only path-scoped, commented exceptions for unavoidable ctypes or dynamic signatures.

- [ ] **Step 3: Resolve native and ABI ty diagnostics**

Add exact generic arguments, overrides, narrowing, and sound return contracts. Do not suppress `unsound-*` findings in ordinary modules; validate or wrap dynamic values at the boundary.

- [ ] **Step 4: Run the native regression gate**

```bash
devenv shell -- uv run --frozen --extra agent ruff check src/ds4py/_native src/ds4py/abi src/ds4py/raw.py
devenv shell -- uv run --frozen --extra agent ty check src/ds4py/_native src/ds4py/abi src/ds4py/raw.py
devenv shell -- uv run --frozen --extra agent pytest tests/native tests/abi tests/test_raw.py -m "not live and not build" -q
```

Expected: all pass. Commit `refactor: harden native type boundaries`.

### Task 4: Engine static gate and worker sentinels

**Files:**
- Modify: `src/ds4py/engine/**`
- Test: existing `tests/engine/**`; add focused tests only if sentinel or control-flow behavior changes

**Interfaces:**
- Consumes: Tasks 1-2's rules, JSON contracts, and dynamic-boundary policy.
- Produces: zero Ruff/ty diagnostics for `src/ds4py/engine`; typed worker/session queues and events; reviewed lowercase sentinels for `_END` and `_STOP`.

- [ ] **Step 1: Type engine payload, queue, and state contracts**

Replace avoidable `Any` with `object`, generics, `JsonValue`, `TypedDict`, and exact callback/queue types. Add missing generic arguments and `@override`; preserve iterator, async, cancellation, and listener signatures.

- [ ] **Step 2: Modernize worker sentinels and narrowing**

Replace worker `_END`/`_STOP` object sentinels with lowercase `sentinel` where identity behavior remains compatible. Use the sentinel itself in union types and narrow with identity. Add focused tests only for identity, repr, copying, or queue behavior that the existing suite does not already pin.

- [ ] **Step 3: Resolve strict condition and soundness diagnostics**

Review every `redundant-condition-strict`, `truthiness-test-of-none-union`, `unsound-*`, and invalid assignment report against actual state transitions. Correct real unreachable or ambiguous branches; use a narrow documented suppression only after demonstrating a checker false positive.

- [ ] **Step 4: Run the engine regression gate**

Run Ruff and ty on `src/ds4py/engine`, then `pytest tests/engine -m "not live and not build" -q`, all inside devenv. Expected: zero static diagnostics and all engine tests pass. Commit `refactor: strengthen engine type contracts`.

### Task 5: Agent, persistence, model, and TUI static gate

**Files:**
- Modify: `src/ds4py/agent.py`, `agent_cli.py`, `binding.py`, `cli.py`, `errors.py`, `message_rendering.py`, `model.py`, `noninteractive.py`, `prefill_trace.py`, `runtime.py`, `runtime_restore.py`, `store.py`, `tui.py`, `tui_commands.py`, `tui_render.py`
- Modify: `src/ds4py/capabilities/**`
- Test: existing `tests/agent/**`; add focused tests only for changed validator or sentinel behavior

**Interfaces:**
- Consumes: `JsonValue`, `JsonObject`, `is_json_value`, and `is_json_object` from Task 2.
- Produces: zero Ruff/ty diagnostics for the agent-facing production modules; typed persisted payloads and capability boundaries; modern model/system-notes sentinels.

- [ ] **Step 1: Replace dynamic payload annotations with precise contracts**

Use Task 2's JSON aliases for tool arguments, persisted data, and JSON parsing. Introduce closed `TypedDict` contracts for stable record shapes, using `Required`/`NotRequired`/`ReadOnly` where appropriate. Use `object` for arbitrary values that are narrowed, and generics where result types must be preserved.

- [ ] **Step 2: Modernize model and system-notes sentinels**

Review `_UNSET`, `_UNKNOWN`, and model `_END` for public repr/copy/pickle implications before replacing them with lowercase `sentinel`. Add focused tests for any relied-upon behavior not already covered; otherwise keep the change type-only and covered by existing tests.

- [ ] **Step 3: Correct overrides, callbacks, async contexts, and restore boundaries**

Add `@override`, precise `RunContext`/result types, `Generator`/`AsyncGenerator` annotations, and narrowing for restore results and stream events. Resolve disjoint casts and redundant conditions without hiding real state or history bugs.

- [ ] **Step 4: Run the agent regression gate**

Run Ruff and ty over all files owned by this task, then `pytest tests/agent -m "not live and not build" -q` inside devenv with the isolated `HOME`. Expected: zero static diagnostics and all agent tests pass. Commit `refactor: strengthen agent and persistence types`.

### Task 6: Benchmark and complete test-suite static gate

**Files:**
- Modify: `bench/**`
- Modify: `tests/**`
- Modify: `pyproject.toml` only for documented test/benchmark-specific exceptions

**Interfaces:**
- Consumes: all production contracts from Tasks 2-5.
- Produces: zero Ruff/ty diagnostics across `src`, `bench`, and `tests`; the complete static gate used by prek and CI.

- [ ] **Step 1: Align benchmark tooling with the typed production contracts**

Replace dynamic manifest/report values with JSON aliases or exact records, fix subprocess/path/PTy annotations, remove redundant or disjoint casts, and keep CLI-only exceptions scoped to the relevant files.

- [ ] **Step 2: Correct tests and test doubles without changing assertions**

Annotate queues, fixtures, callbacks, monkeypatched methods, and generic containers to match production signatures. Fix test-only import resolution explicitly rather than suppressing unresolved imports. Retain documented test exceptions for assertions, private access, dynamic doubles, and `Any`; the independent `TypeGuard` ban still applies.

- [ ] **Step 3: Handle intentional exceptional tests narrowly**

For intentional cases such as division by zero, use the smallest possible targeted suppression with a reason. Do not enable the globally noisy ty rules and do not weaken an entire test directory's soundness checks to resolve isolated diagnostics.

- [ ] **Step 4: Run the complete static and behavioral gate**

```bash
devenv shell -- uv run --frozen --extra agent ruff check src bench tests
devenv shell -- uv run --frozen --extra agent ruff format --check src bench tests
devenv shell -- uv run --frozen --extra agent ty check
HOME=/tmp/ds4-quality-empty-home devenv shell -- uv run --frozen --extra agent pytest -m "not build and not live" -q
```

Expected: zero static diagnostics and the full portable suite passes. Commit `test: align suite with strict type checks`.

### Task 7: CI, agent instructions, and final verification

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml` and `uv.lock` to remove Pyright
- Create: `AGENTS.md`
- Modify: `docs/development.md`
- Modify: `.pre-commit-config.yaml` only if final CI parity requires adjustment

**Interfaces:**
- Consumes: Task 6's completely green local gate.
- Produces: one CI `quality` job running prek, no Pyright dependency/configuration, documented devenv-first commands, and the final stacked-PR verification evidence.

- [ ] **Step 1: Replace the old typing job and harden workflow behavior**

Rename `typing` to `quality` and run `uv run --frozen --extra agent prek run --all-files --show-diff-on-failure` on Python 3.14 Ubuntu. Remove Pyright commands. Preserve test/build matrices, add suitable timeouts to every job, pin actions to full commit SHAs, and disable persisted credentials where unnecessary.

- [ ] **Step 2: Remove Pyright and synchronize documentation**

Remove its dependency and `[tool.pyright]` table, regenerate the lockfile, then run `pyproject-fmt --check`. In `AGENTS.md`, document devenv setup, `uv sync --frozen --extra agent`, `uv run prek install`, canonical quality/test commands, `Any` and `TypeGuard` policies, modern typing, safe fixes, and focused versus full verification. Link native/live details in `docs/development.md` rather than duplicating them.

- [ ] **Step 3: Run all configuration and packaging checks**

Inside devenv, run prek with all files, direct ty, actionlint, `uv build --wheel`, the existing `Makefile.lib` wheel-content assertion, and `git diff --check`. Expected: all pass; GitHub schema, timeout, YAML, JSON, TOML, and `TypeGuard` hooks all execute successfully.

- [ ] **Step 4: Run final behavioral and hygiene verification**

Run the full portable pytest suite with the empty isolated `HOME`, then rerun prek to prove no hook modifies the final tree. Inspect `git diff --stat`, `git diff --check`, and all type-oriented changes for accidental behavior edits. Expected: 1,425 or more passing tests depending on added focused tests, no unexpected skips beyond baseline, and a clean worktree after commits.

- [ ] **Step 5: Commit and prepare the stacked PR**

```bash
git add .github/workflows/ci.yml pyproject.toml uv.lock AGENTS.md docs/development.md .pre-commit-config.yaml
git commit -m "build: enforce Python quality gates in CI"
```

Push `build/python-quality-gates` and open the PR with base `m3-harness-agent`, listing the exact local verification results. Do not claim live model parity from portable tests; allow the existing native build and unit matrices to provide their own CI evidence.
