# PR 3 Review Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the seven agreed local PR 3 changes without adding a workaround for the deferred engine SDK issue.

**Architecture:** Extend the current native boundary and install generations. A new shader module owns copied assets and their integrity checks. The worker enforces a byte budget without changing checkpoint ownership or eviction policy.

**Tech Stack:** Python 3.14+, ctypes, pytest, uv, C compiler, Make, GitHub Actions.

**Spec:** `docs/specs/2026-10-23-pr3-review-feedback-design.md`, which amends `docs/specs/2026-10-21-ds4-py-design.md`.

## Global Constraints

- Require Python 3.14 or later and use 3.14 as the development default.
- Keep the core free of Python runtime dependencies.
- Keep upstream at `5379c1c9ac315f1540371b06a3db88491d4059ff`.
- Pin the fork to `3e61e09bbe4599ca71197d8ec21933a93e24f022`.
- Keep `~/.ds4/kvcache` as the compatible default when `DS4PY_HOME` is unset.
- Set the default checkpoint budget to `1 << 30` bytes, including `system`.
- Do not evict checkpoints automatically.
- Keep every native engine call on the existing worker thread.
- Keep the Metal SDK fix outside this repository.
- Do not push, post GitHub replies, or resolve review threads.

## Review Focus

- Task 4: A deleted checkout or moved install must not change shader resolution.
- Task 4: Missing assets, malformed manifest paths, and mismatched overrides must fail before native use.
- Task 5: Failed opens and later opens from another bundle must leave no stale environment values.
- Task 7: Exact-limit requests, `system`, replacement failures, and inaccurate size estimates must preserve accounting and ownership.
- Task 3: Unset or empty roots and explicit library overrides must retain the documented path precedence.

## Starting state and execution

Worktree: `.worktrees/m1-native-layer`, branch `m1-native-layer`.
Base: `aaed23f0ed8c36a955b97025ad0ea24d1f328661`.
The worktree was clean before these planning documents.
Baseline: `.venv/bin/python -m pytest -q -m "not build and not live"`.
Result on the existing Python 3.11 environment: 313 passed, 15 deselected.

The parent implements directly in this session. Do not delegate implementation.
Review this plan with the user before product changes.
Keep changes uncommitted until the user requests commits or integration.

Apply a red/green test cycle to production behavior.
Do not add tests that merely assert workflow text, version strings, pin literals, or documentation.
Use direct checks for those static changes.

## File boundaries

Keep storage resolution in the existing loader.
Create only one production module: `src/ds4py/abi/shaders.py`.
It handles shader files and returns environment dictionaries without changing `os.environ`.
Keep environment mutation beside the native-open code.
Keep checkpoint accounting in the worker.
Do not restructure the native adapters or build system beyond these changes.

---

### Task 1: Raise the Python floor

Files: `pyproject.toml`, `.python-version`, `uv.lock`, `.github/workflows/ci.yml`, and the active design.

Interfaces: Package metadata requires `>=3.14`. The development default is `3.14`.

- [ ] Set the floor and default. Replace the unsupported CI entries with Linux 3.14, Linux 3.15, and macOS 3.14. Use 3.14 for build jobs.
- [ ] Update the active design's Python requirement without rewriting archived plans or drafts.
- [ ] Regenerate the lockfile with `uv lock --python 3.14`.
- [ ] Run `uv sync --python 3.14 --locked` and `uv run --python 3.14 pytest -q -m "not build and not live"`. Expect exit 0.
- [ ] Run `uv build --wheel`. Inspect the wheel's `Requires-Python` and `ds4py/abi/Makefile.lib` entries.
- [ ] Use TOML parsing and the existing CI syntax checks instead of new static-content tests.

### Task 2: Refresh the fork pin and header fixtures

Files: `src/ds4py/abi/pins.py`, `tests/fixtures/ds4-headers/fork/{ds4.h,ds4_kvstore.h,ds4_ssd.h}`, and the active design.

Interfaces: `FORK.commit` names the full approved revision. `KNOWN_HEADERS` recognizes the actual pinned headers.

- [ ] Obtain a clean source checkout at `3e61e09bbe4599ca71197d8ec21933a93e24f022`. Do not overwrite a checkout with user changes.
- [ ] Refresh all three fork headers from that revision. Preserve the fixture's license and update it only if the source license changed.
- [ ] Update the fork pin and its computed header fingerprints. Preserve the upstream pin and fingerprints.
- [ ] Run `uv run --python 3.14 pytest tests/abi/test_probe.py tests/native/test_bindings.py -q -m "not build and not live"`.
- [ ] Compare fixture bytes with the exact source revision. Do not add a test for the commit literal.
- [ ] Run the native build commands in the final evidence section before completion.

### Task 3: Make the library root configurable

Files: `src/ds4py/_native/loader.py`, `tests/native/test_loader.py`, and the active design.

Interfaces: `default_lib_dir() -> Path` honors `DS4PY_HOME`. Existing `resolve_lib_path(lib)` precedence stays intact.

- [ ] Add behavior tests for unset and empty roots, a custom root, leading-`~` expansion, and environment changes between calls.
- [ ] Add precedence cases proving `lib` argument > `DS4_LIB` > `DS4PY_HOME/lib` > `~/.ds4py/lib`.
- [ ] Run `uv run --python 3.14 pytest tests/native/test_loader.py -q`. Expect new cases to fail for the missing root behavior.
- [ ] Extend `default_lib_dir()` without creating directories or caching the environment value.
- [ ] Run the loader tests again. Expect all cases to pass.
- [ ] Document the future sessions, saves, and KV layout. Do not implement those later persistence features.

### Task 4: Publish and validate Metal shader bundles

Files: Create `src/ds4py/abi/shaders.py` and `tests/abi/test_shaders.py`.
Modify `src/ds4py/abi/{build,manifest}.py`, `src/ds4py/_native/loader.py`, `tests/abi/{test_build,test_manifest}.py`, and `tests/native/test_loader.py`.

Interfaces:

- `copy_metal_sources(source_root: Path, destination_root: Path) -> dict[str, str]` copies `source_root/metal/*.metal` into `destination_root/metal/`. Keys are relative paths such as `metal/example.metal`. Values are SHA-256 hashes of copied bytes.
- `metal_source_env(lib_dir: Path, hashes: Mapping[str, str], overrides: Mapping[str, str] | None = None) -> dict[str, str]` validates the bundle and effective overrides, then returns native shader variables. `None` means no overrides. It does not read or change the process environment.
- `Manifest.metal_sources: dict[str, str]` defaults to an empty mapping. Metal requires a nonempty valid mapping.
- Extend `install(lib, manifest, out_dir, *, metal_root: Path | None = None)` to stage shader files before switching the library symlink. `metal_root` contains the captured `metal/` directory.

- [ ] Add tests with small fake shader files. Assert copied bytes and computed hashes, not merely JSON field presence.
- [ ] Test removal or modification of the original checkout, complete-bundle relocation, and resolution through the installed library symlink.
- [ ] Test missing or modified bundled files, absent Metal records, invalid hashes, absolute paths, parent traversal, and bundled symlinks that escape the install directory.
- [ ] Test valid explicit overrides with matching contents and rejection of mismatched override contents. Preserve valid override strings.
- [ ] Extend publication failure tests. The old generation must remain usable, and an unpublished partial generation must not replace it.
- [ ] Run the shader, build, manifest, and loader tests. Expect the new behavior tests to fail before implementation.
- [ ] Implement copying and validation in the new shader module. Raise actionable `BuildError` during staging and `AbiMismatch` during runtime validation.
- [ ] Capture shaders into temporary build storage before native compilation. Pass their hashes into the manifest.
- [ ] Copy the captured files into the new install generation. Compare their copied hashes with the manifest before publication.
- [ ] Keep the existing atomic symlink switch and old-generation retention. No extra content-addressed storage or garbage collector is needed.
- [ ] Validate Metal assets before `_dlopen`. Keep old CPU manifests usable without shader records.
- [ ] Run `uv run --python 3.14 pytest tests/abi/test_shaders.py tests/abi/test_build.py tests/abi/test_manifest.py tests/native/test_loader.py -q -m "not build and not live"`. Expect exit 0.

### Task 5: Restore shader environment variables after native open

Files: `src/ds4py/_native/api.py`, `tests/native/test_api.py`, and the active design.

Interfaces: Native Metal open consumes Task 4's `metal_source_env()` with the real library directory, manifest hashes, and current environment.

- [ ] Extend the native-open fixture to provide real temporary shader assets and matching manifest hashes.
- [ ] Test the values visible during native open and exact environment restoration afterward.
- [ ] Cover native-open errors, Python exceptions, later opens using another install, valid explicit overrides, and the CPU path.
- [ ] Test a shader change after library loading. Native open must recheck effective files before calling C.
- [ ] Run `uv run --python 3.14 pytest tests/native/test_api.py -q`. Expect the cleanup and stale-source cases to fail.
- [ ] Use the real library directory rather than `manifest.source_path`.
- [ ] Add missing shader variables only around native open, under `_open_lock`. Restore prior values in `finally` without changing valid explicit overrides.
- [ ] Preserve the existing `DS4_LOCK_FILE` restoration and instance-lock cleanup paths.
- [ ] Run the native API tests again. Expect exit 0.
- [ ] Keep the native adapter focused. Extract a small private context manager only if it makes cleanup clearer.

### Task 6: Supply bundle paths to the live C parity subprocess

Files: `tests/live/test_raw_parity.py`.

Interfaces: Reuse Task 4's pure environment-dictionary resolver.

- [ ] Read the manifest once. Resolve the real library directory for asset paths.
- [ ] Build a copy of the current environment for the C subprocess. Add validated shader variables only for Metal.
- [ ] Pass that dictionary through `subprocess.run(env=...)`. Do not change `cwd`, model arguments, or parent environment.
- [ ] Run `uv run --python 3.14 pytest tests/live/test_raw_parity.py --collect-only -q`.
- [ ] Do not add a mock test that only asserts the live script's structure. Task 4 tests cover the reusable path and override logic.
- [ ] Run the existing live test when matching engine, model, and Metal hardware are available. Otherwise report that it was not run.

### Task 7: Enforce the checkpoint byte budget

Files: `src/ds4py/engine/worker.py`, `tests/engine/test_worker.py`, any affected session tests, and the active design.

Interfaces: `EngineWorker(..., checkpoint_budget_bytes: int = 1 << 30, ...)` replaces `max_checkpoints`. Requests still report `Checkpointed` or raise `CheckpointError`.

- [ ] Replace count-limit tests with byte-budget cases using small `FakeNative` snapshots.
- [ ] Cover exact-fit admission, over-budget rejection, `system` accounting, and a zero budget.
- [ ] Cover replacement of an existing key. Subtract the old snapshot from the proposed retained total, but keep it until the replacement succeeds.
- [ ] Test dropping and closing snapshots, failed saves, and a returned snapshot larger than its preflight size estimate.
- [ ] Assert that rejected replacements keep the old snapshot, free rejected new snapshots, and do not evict other keys.
- [ ] Test rejection of negative budgets and non-integer values, including booleans.
- [ ] Run `uv run --python 3.14 pytest tests/engine/test_worker.py tests/engine/test_session.py -q`. Expect the new budget cases to fail.
- [ ] Validate the constructor argument. Reject creation unconditionally when the budget is zero. Use the existing `session.payload_bytes()` query for preflight admission.
- [ ] Compute retained payload bytes from stored snapshots. Include `system`; exclude the replaced key when calculating the candidate total.
- [ ] Recheck the actual `snapshot.nbytes` before storing it. Free a rejected candidate and leave existing state unchanged.
- [ ] Keep the old checkpoint alive until a replacement is accepted. Preserve thread affinity and existing drop/close behavior.
- [ ] Run the worker and session tests again. Expect exit 0.
- [ ] Update the design's count-limit and memory-risk sections. State that total RAM and temporary allocation peaks remain outside the budget.

## Final evidence

- [ ] Run `uv run --python 3.14 pytest -q -m "not build and not live"`.
- [ ] Set `DS4_SRC_UPSTREAM` and `DS4_SRC_FORK` to clean checkouts at their exact pins.
- [ ] Run `uv run --python 3.14 pytest -q -m build` with those variables.
- [ ] Run CPU `ds4py build` and `ds4py check` against each pin in separate output directories. For the fork extension checks, retain the existing CI `PYTHONPATH=tests/fixtures/ext_site` setup.
- [ ] Run `uv lock --check`, `uv build --wheel`, and the wheel metadata/resource inspection.
- [ ] Run `git diff --check`. Review the complete diff against the amended design.
- [ ] Report actual commands and results. Separate unavailable Metal/live checks from successful CPU checks.
- [ ] State that the engine-side Metal SDK issue remains pending. Do not treat the Python floor or Linux success as evidence of that fix.
