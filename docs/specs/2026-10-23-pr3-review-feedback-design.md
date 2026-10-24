# PR 3 review feedback design

Date: 2026-10-23

This document records the eight review decisions approved in conversation.
It amends `docs/specs/2026-10-21-ds4-py-design.md` where the requirements differ.
Implementation stays on the existing `m1-native-layer` worktree.

## Scope

Implement the seven local changes below.
Keep the Metal language-version fix in the C engine.
Do not add a Python workaround for that engine issue.
Do not post replies, resolve GitHub threads, or push changes as part of this work.

## Approved requirements

### Python support

Require Python 3.14 or later.
Use Python 3.14 as the development default.
Update package metadata, the version file, CI, the lockfile, and the active design.
Drop support for Python 3.11 through 3.13.
The version change does not prove that the engine SDK issue is fixed.

### Fork revision and headers

Pin the fork to `3e61e09bbe4599ca71197d8ec21933a93e24f022`.
Refresh its header fixtures from that exact revision.
Refresh the known header fingerprints to match the pinned sources.
Keep upstream at `5379c1c9ac315f1540371b06a3db88491d4059ff`.
Run the existing header checks and native build checks for both sources.

### Storage location

Use `DS4PY_HOME` to change the root for ds4py files.
With no override, keep libraries under `~/.ds4py/lib`.
Keep the compatible KV default at `~/.ds4/kvcache`.

With an override, use its `lib/`, `sessions/`, `saves/`, and `kvcache/` subdirectories.
Explicit library arguments and `DS4_LIB` take precedence over the root.
Do not move existing files.
This PR implements library path resolution only.
Document the same layout for future session, save-record, and KV persistence work.

### Metal shader bundle

Copy the matching `metal/*.metal` sources during the build.
Publish them beside the library inside its install directory.
Record their relative paths and SHA-256 hashes in the manifest.
Keep `source_path` as build provenance, not a runtime dependency.

Runtime code must use the bundled sources.
Updating or deleting the original checkout must not affect an existing installation.
Moving a complete install directory must preserve its shader paths.
Missing or changed bundled files must cause a clear rebuild error.
Old Metal manifests without shader records require rebuilding.
CPU installations do not require shader assets.

Explicit shader overrides remain supported.
Their contents must match the hashes recorded for the corresponding shaders.
An override cannot silently load a different shader version.

### Temporary environment changes

Add missing shader-path variables only during native Metal open.
Restore the original environment on success and failure.
Preserve explicit overrides that pass the shader checks.
Keep the existing open lock around process-wide environment changes.
Do not claim that these temporary changes isolate other threads or child processes.

### Live parity subprocess

Give the C parity subprocess its shader paths through an explicit environment dictionary.
Use the shared shader-path resolver.
For Metal, resolve paths from the installed bundle rather than the original checkout.
The bundle requirement supersedes the earlier proposal to use `manifest.source_path`.
Do not change the working directory.
Relative model and executable paths must keep their current meaning.
CPU parity does not require Metal sources.

### Checkpoint byte budget

Replace the checkpoint count limit with `checkpoint_budget_bytes`.
The default is `1 << 30` bytes, or 1 GiB.
Count every retained snapshot, including `system`, using its `nbytes`.
Do not evict checkpoints automatically.

Reject a request that exceeds the budget with `CheckpointError`.
If a replacement is rejected, keep the previous checkpoint.
Free any newly created snapshot that cannot be retained.
Dropping checkpoints and closing the worker must release their snapshots.
A zero budget disables checkpoint creation.
The budget covers retained snapshot payloads, not total process RAM or temporary allocation peaks.

## Implementation choices

These are implementation choices for review, not additional user requirements.

Use the existing install generations and atomic library-symlink switch.
A generation contains its library, manifest, and `metal/` directory.
Retain old generations according to the existing reader-safety policy.

Put shader copying, hash checks, and environment-dictionary construction in `ds4py.abi.shaders`.
Keep that module independent of native library handles.
Record assets as a `metal_sources: dict[str, str]` manifest field.
An empty mapping remains valid for CPU manifests.

Treat an empty `DS4PY_HOME` as unset and expand a leading `~`.
Resolve the root at call time.
Path resolution does not create directories.

Use the existing payload-size query before allocating a checkpoint.
Also check the returned snapshot size before retaining it.
The second check protects against an inaccurate size estimate.
Reject negative or non-integer budgets, including booleans.

Use Python 3.14 for Linux and macOS CI.
Keep a second Linux unit-test entry on Python 3.15.
Use existing tests and direct package checks for static version, workflow, and pin changes.

## Deferred engine issue

The Metal language-version fix must land in the C engine first.
Adopt its fixed revision in a later pin update.
Test Python 3.14 on an affected macOS version after that update.
Do not report that issue as fixed from Linux checks or the Python version change.

## Acceptance evidence

Run behavior tests for shader integrity, environment cleanup, storage precedence, and checkpoint accounting.
Run the full unit suite on Python 3.14.
Run CPU build checks against both pinned engine revisions.
Build the wheel and inspect its metadata and bundled Makefile.
Report the Metal and live parity checks separately when hardware or model files are unavailable.
