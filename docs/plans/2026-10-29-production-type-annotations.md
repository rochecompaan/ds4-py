# Python Type Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Annotate every Python function under `src/`, `bench/`, and `tests/`, and make CI reject missing parameter and return annotations.

**Architecture:** Keep runtime behavior unchanged. Add explicit types at module boundaries and inside implementations, using existing protocols and domain types before introducing new aliases. Use Pyright for type consistency and missing parameter annotations, and Ruff's flake8-annotations rules for return annotations and variadic parameters.

**Tech Stack:** Python 3.14+, Pyright, Ruff, uv, pytest, GitHub Actions.

**Spec:** `docs/specs/2026-10-21-ds4-py-design.md`, plus the approved `fb58158` review requirement: the permanent annotation gate covers every Python file in `src/`, `bench/`, and `tests/`, including fixtures and test helpers.

## Global Constraints

- Do not change runtime behavior, public signatures, defaults, or return values.
- Add annotations and type-only imports; use casts only where the type system cannot observe an existing runtime guarantee.
- Prefer existing protocols and domain types. Use `Any` only at genuinely dynamic boundaries such as JSON, ctypes, and extension payloads.
- Configure Pyright with `reportMissingParameterType = "error"`; default mode leaves this diagnostic off.
- Configure Ruff rules `ANN001`, `ANN002`, `ANN003`, `ANN201`, `ANN202`, `ANN204`, `ANN205`, and `ANN206` for `src/`, `bench/`, and `tests/`.
- Do not add tests that restate configuration. Verify the gate by running Pyright and Ruff, and run existing tests to catch accidental behavior changes.

## Review Focus

- Native and ctypes boundaries: handles, pointers, callback functions, and streams must retain their runtime representations. Verify with native unit tests.
- Engine callbacks and async APIs: parameter and return types must preserve generator, awaitable, and listener contracts. Verify with engine and agent tests.
- JSON and event payloads: use existing payload types or explicit JSON-shaped aliases rather than claiming narrower types than runtime validation provides.
- Benchmark subprocess and PTY code: path, environment, file descriptor, and timeout types must match actual accepted values, including `None`.
- Test fixtures and helpers: annotate pytest fixture parameters, yielded fixture values, nested callbacks, and test doubles without changing test setup or assertions.
- Annotation-only edits can still break runtime imports. Keep type-only imports behind `TYPE_CHECKING` where needed and run the full non-live suite.

---

### Task 1: Add the Python annotation gate

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `uv run --extra agent pyright` for `src/`, `bench/`, and `tests/`.
- Produces: `uv run --extra agent ruff check src bench tests` for missing annotations.

- [x] **Step 1: Add Pyright and Ruff to the development dependency group**

Use `uv add --group dev pyright ruff` so the lock file records the resolved versions.

- [x] **Step 2: Configure the annotation rules**

Set Pyright's include paths to `src`, `bench`, and `tests`, retain standard type checking, and set `reportMissingParameterType = "error"`. Enable only the Ruff annotation rules listed in Global Constraints.

- [x] **Step 3: Add one CI typing job**

Run both commands with the agent extra on Python 3.14. Keep this separate from the unit-test matrix so annotation failures run once per pull request.

- [x] **Step 4: Record the baseline failures**

Run both commands and confirm that they report the existing unannotated production functions. This is the failing check that the remaining tasks must fix.

- [x] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .github/workflows/ci.yml
git commit -m "build(ci): add production annotation checks"
```

### Task 2: Annotate native, ABI, and raw interfaces

**Files:**
- Modify: `src/ds4py/_native/api.py`
- Modify: `src/ds4py/_native/bindings.py`
- Modify: `src/ds4py/_native/chat.py`
- Modify: `src/ds4py/_native/fake.py`
- Modify: `src/ds4py/_native/fake_chat.py`
- Modify: `src/ds4py/_native/loader.py`
- Modify: `src/ds4py/_native/logits.py`
- Modify: `src/ds4py/_native/modelcheck.py`
- Modify: `src/ds4py/_native/payload.py`
- Modify: `src/ds4py/_native/protocol.py`
- Modify: `src/ds4py/_native/rng.py`
- Modify: `src/ds4py/_native/structs.py`
- Modify: `src/ds4py/_native/typemap.py`
- Modify: `src/ds4py/abi/build.py`
- Modify: `src/ds4py/abi/check.py`
- Modify: `src/ds4py/abi/manifest.py`
- Modify: `src/ds4py/abi/pins.py`
- Modify: `src/ds4py/abi/probe.py`
- Modify: `src/ds4py/abi/shaders.py`
- Modify: `src/ds4py/raw.py`

**Interfaces:**
- Consumes: existing native protocols, ctypes structures, ABI manifests, and stream wrappers.
- Produces: explicit parameter and return types without changing native calls or file formats.

- [x] **Step 1: Annotate native protocol, payload, fake, and loader modules**

Use the existing `Protocol` declarations as the contract for implementations. Annotate context managers, iterators, callbacks, and `*args`/`**kwargs` explicitly.

- [x] **Step 2: Annotate ctypes API, structs, type maps, and raw bindings**

Keep `ctypes` pointer and handle types aligned with the existing runtime values. Isolate unavoidable dynamic values at the binding boundary.

- [x] **Step 3: Annotate ABI build and probe modules**

Use `Path`, mapping, sequence, and subprocess types that match current callers. Preserve optional path and environment behavior.

- [x] **Step 4: Run focused static and runtime checks**

Run Pyright and Ruff on the files in this task, then run `uv run --extra agent pytest tests/native tests/abi tests/test_raw.py -m "not live and not build" -q`.

- [x] **Step 5: Commit**

```bash
git add src/ds4py/_native src/ds4py/abi src/ds4py/raw.py
git commit -m "refactor(native): complete production type annotations"
```

### Task 3: Annotate the engine package

**Files:**
- Modify: all Python modules directly under `src/ds4py/engine/`
- Modify: all Python modules under `src/ds4py/engine/tool_syntax/`

**Interfaces:**
- Consumes: annotated native and raw interfaces from Task 2.
- Produces: typed session, worker, chat, prompt, replay, transcript, event, sampling, and tool-syntax contracts for the agent layer.

- [x] **Step 1: Annotate tool parsing and rendering**

Cover decoder inputs, token sequences, mappings, call records, and parser results. Preserve existing dictionary and list shapes.

- [x] **Step 2: Annotate session, lifecycle, worker, and stream code**

Use explicit callback, awaitable, generator, queue, lock, and optional job types. Do not weaken cancellation or event-listener contracts.

- [x] **Step 3: Annotate chat, prompt, replay, transcript, model, and planning helpers**

Use existing segment, message, token, and event types. Add small aliases only when they remove repeated structural types without changing validation.

- [x] **Step 4: Run focused static and runtime checks**

Run Pyright and Ruff on `src/ds4py/engine`, then run `uv run --extra agent pytest tests/engine -m "not live and not build" -q`.

- [x] **Step 5: Commit**

```bash
git add src/ds4py/engine
git commit -m "refactor(engine): complete production type annotations"
```

### Task 4: Annotate agent, capability, model, store, and TUI modules

**Files:**
- Modify: `src/ds4py/agent.py`
- Modify: `src/ds4py/agent_cli.py`
- Modify: `src/ds4py/binding.py`
- Modify: `src/ds4py/cli.py`
- Modify: `src/ds4py/message_rendering.py`
- Modify: `src/ds4py/model.py`
- Modify: `src/ds4py/noninteractive.py`
- Modify: `src/ds4py/prefill_trace.py`
- Modify: `src/ds4py/runtime.py`
- Modify: `src/ds4py/runtime_restore.py`
- Modify: `src/ds4py/store.py`
- Modify: `src/ds4py/tui.py`
- Modify: `src/ds4py/tui_commands.py`
- Modify: `src/ds4py/tui_render.py`
- Modify: all Python modules under `src/ds4py/capabilities/`

**Interfaces:**
- Consumes: typed engine and native contracts from Tasks 2 and 3.
- Produces: fully annotated public agent factories, runtime APIs, capability hooks, persistence functions, and interactive interfaces.

- [x] **Step 1: Annotate capabilities and persistence**

Cover session hooks, event payloads, shell process state, run records, system notes, save records, and atomic write callbacks.

- [x] **Step 2: Annotate model and message rendering**

Preserve pydantic-ai model request, response, usage, streaming, and note-provider contracts. Use the library's public types where available.

- [x] **Step 3: Annotate CLI, TUI, commands, and rendering**

Cover prompt-toolkit documents, completion events, input and output interfaces, command handlers, application results, and render fragments.

- [x] **Step 4: Verify the five modules from `fb58158` and remaining top-level modules**

Run both static checks on all files in this task. Fix residual missing annotations without changing established contracts.

- [x] **Step 5: Run focused runtime checks**

Run `uv run --extra agent pytest tests/agent -m "not live and not build" -q`.

- [x] **Step 6: Commit**

```bash
git add src/ds4py
git commit -m "refactor(agent): complete production type annotations"
```

### Task 5: Annotate benchmark tooling and verify the production gate

**Files:**
- Modify: `bench/graders/test_ctx_command.py`
- Modify: `bench/metrics.py`
- Modify: `bench/prepare_bakeoff.py`
- Modify: `bench/replay_c_traces.py`
- Modify: `bench/report.py`
- Modify: `bench/restore_check.py`
- Modify: `bench/run_bakeoff.py`
- Modify: `bench/workspaces.py`
- Modify: `README.md` or the existing development documentation that lists verification commands

**Interfaces:**
- Consumes: typed runtime, event, store, and replay contracts.
- Produces: typed benchmark configuration, subprocess, PTY, metrics, restore, replay, and workspace helpers.

- [x] **Step 1: Annotate metrics, reports, and trace replay**

Use explicit event, summary, path, and numeric types. Keep malformed-input handling and numeric coercion unchanged.

- [x] **Step 2: Annotate runner, restore, PTY, and workspace helpers**

Cover command sequences, environment mappings, timeout values, file descriptors, attempt results, and optional callbacks.

- [x] **Step 3: Document the two static verification commands**

Add the Pyright and Ruff commands beside the existing test command in the established development documentation.

- [x] **Step 4: Run benchmark tests and the complete production gate**

Run `uv run --extra agent pytest tests/bench -m "not live and not build" -q`, then `uv run --extra agent pyright` and `uv run --extra agent ruff check src bench`. All three must pass.

- [x] **Step 5: Run final regression and hygiene checks**

Run `uv run --extra agent pytest -m "not live and not build" -q` and `git diff --check`. Record the exact results.

- [x] **Step 6: Commit**

```bash
git add bench README.md docs pyproject.toml uv.lock .github/workflows/ci.yml src
git commit -m "refactor(bench): complete production type annotations"
```

### Task 6: Annotate all tests, fixtures, and test helpers

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_raw.py`
- Modify: all Python modules under `tests/agent/`
- Modify: all Python modules under `tests/abi/`
- Modify: all Python modules under `tests/bench/`
- Modify: all Python modules under `tests/engine/`
- Modify: all Python modules under `tests/fixtures/`
- Modify: all Python modules under `tests/live/`
- Modify: all Python modules under `tests/native/`

**Interfaces:**
- Consumes: fully annotated production APIs and benchmark helpers.
- Produces: explicit pytest fixture, test-double, callback, helper, and test-function signatures throughout the test suite.

- [x] **Step 1: Annotate shared fixtures and test doubles**

Use `pytest.MonkeyPatch`, `pytest.CaptureFixture[str]`, `pytest.FixtureRequest`, `Path`, and existing fake-engine contracts where they match. Annotate fixture yield and iterator types explicitly.

- [x] **Step 2: Annotate agent, ABI, and benchmark tests**

Cover test parameters, nested helpers, callbacks, scripts, and fake runtime methods. Keep every assertion and fixture dependency unchanged.

- [x] **Step 3: Annotate engine, native, live, and extension-fixture tests**

Cover tool-syntax fixtures, native contract helpers, callback signatures, and extension entry-point fixtures. Do not import test-only types into production modules.

- [x] **Step 4: Run the complete static gate**

Run `uv run --extra agent pyright` and `uv run --extra agent ruff check src bench tests`. Both must pass with zero errors.

- [x] **Step 5: Run final regression and hygiene checks**

Run `uv run --extra agent pytest -m "not live and not build" -q` and `git diff --check`. Record the exact results.

- [x] **Step 6: Commit**

```bash
git add tests
git commit -m "test: complete test suite type annotations"
```
