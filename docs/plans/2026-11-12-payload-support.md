# Generic File-Payload Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve complete session persistence on engines without native file-payload support by writing stripped saves and switching through rendered restores, while leaving supported engines and M4 in-memory compaction behavior unchanged.

**Architecture:** Add one optional core ABI declaration and a native-boundary helper that maps an absent, freshly probed symbol to legacy support and otherwise returns the native boolean. `Session` exposes the helper through its existing worker; `AgentRuntime` branches only on `False` around native file staging and KV loading. A stripped save publishes its full harness snapshot and zero-payload record before unlinking stale KV.

**Tech Stack:** Python 3.14, ctypes, pytest, existing `EngineWorker`/`Session` native confinement, `SessionStore`, harness `FileStepStore`, devenv + uv.

**Spec:** `docs/specs/2026-11-10-m3-harness-integration-design.md` (sections 2.3, 10, 12.1, 14.1 item 2, and 14.2)

## Global Constraints

- Stack on `feat/harness-composition`; commit locally on `feat/payload-support`, with no push, PR, co-author, or tool trailers.
- Add exactly one optional core binding: `bool ds4_session_supports_payload(ds4_session *s)`, with `source="core"` and `required=False`.
- A missing manifest record retains the existing rebuild-required ABI failure; only a rebuilt manifest recording an absent symbol receives legacy `True` behavior.
- Runtime obtains the boolean through the native worker. Only `False` selects stripped/rendered persistence; there is no third state.
- On `False`, save performs no native staging or sync-for-payload and writes no KV file. The complete snapshot and record are published before stale `session.kv` removal.
- On `False`, a planned KV switch becomes rendered after record/model/snapshot/provenance validation and before native load, identity hooks, recorder relocation, or session identity mutation.
- Do not guard M4 checkpoints, summary generation, transcript replacement, or `/compact` with the file-payload predicate. `/compact` output must not claim KV persistence.
- Do not change default behavior on engines supporting payloads. Do not reject save or switch based on speculative decoding mode.
- No downstream identifiers in `src/`; the final `grep -rniE "satyrn|dflash|qwen35" src/` must print nothing.
- Out of scope: `bench/restore_plan.py`, `/list`, `/strip`, `/del`, the system-prefix cache, other repositories, and live/Metal runs.
- Follow RED-GREEN: write each focused test, watch it fail for the missing behavior, implement the smallest production change, and rerun it.

## Review Focus

- A crash immediately before or after stale-KV unlink must leave the newly published zero-payload snapshot/record restorable; Task 2 has a crash test on each side.
- A legacy manifest with no probe record must not be confused with a rebuilt manifest recording absence; Task 1 tests both loader paths.
- The native predicate must remain a strict boolean policy and must execute on the worker; Task 1 tests absent/true/false values and Task 2/3 exercise the public worker-confined path.
- File-payload incapability must not disable in-memory checkpoint/summary/compaction operations; Tasks 2 and 3 compact with explicit `False` before stripped save and rendered switch.
- Supported exact KV restore and validation failures must remain unchanged; Task 3 tests explicit native `True`, no load on fallback, and corrupt-record/model/provenance validation before fallback.

---

### Task 1: Optional Core Probe and Worker-Confined Native Policy

**Files:**
- Modify: `src/ds4py/_native/bindings.py`
- Modify: `src/ds4py/_native/api.py`
- Modify: `src/ds4py/_native/protocol.py`
- Modify: `src/ds4py/_native/fake.py`
- Modify: `src/ds4py/engine/session.py`
- Modify: `tests/native/test_bindings.py`
- Modify: `tests/native/test_loader.py`
- Modify: `tests/abi/test_probe.py`
- Modify: `tests/abi/test_manifest.py`
- Modify: `tests/abi/test_check.py`
- Modify: `tests/abi/test_build.py` (build-marked assertion only; no build/live run in this PR)
- Create: `tests/native/test_file_payloads.py`
- Modify: `tests/agent/test_runtime.py` (test helper only)

**Interfaces:**
- Produces: core `Binding("ds4_session_supports_payload", "bool ds4_session_supports_payload(ds4_session *s)", required=False)`.
- Produces: `file_payloads_supported(library: _PayloadLibrary, session: int) -> bool` in `_native/api.py`, where `_PayloadLibrary` is a structural `Protocol` with `has`/`fn`. The C declaration requires a raw session pointer in addition to the library adapter named by the spec; absent returns literal `True`, present returns `bool_result(library.fn(...)(session))`.
- Produces: `NativeSession.file_payloads_supported() -> bool`, delegating to the helper with its library and session.
- Produces: `Session.file_payloads_supported() -> bool`, evaluating `SessionHandle.file_payloads_supported()` inside `EngineWorker.call(...).result()`.
- Produces: `FakeNative(payload_support: bool | None = None)`. `None` models a rebuilt library with the optional symbol absent and therefore returns `True`; a bool models a present native predicate returning that value. `FakeSession` enforces its normal worker-thread check.
- Consumes: existing `Library.has`/`Library.fn`, `bool_result`, and worker-confined `SessionHandle` implementations.

- [ ] **Step 1: Write failing binding, manifest, and helper tests**

Update `test_core_bindings_are_the_m1_list` to expect 53 declarations, all required except `ds4_session_supports_payload`, whose source is `core`. Update the extension concatenation slice to `len(bindings.CORE_BINDINGS)`. Add a downstream-collision test using `install_providers` with the same name and require the existing redeclaration error. Preserve assertions that unrelated optional extension records keep their source and required metadata.

Update the ABI assertions in the same RED step with this replacement invariant: required core bindings compile against both pins; `ds4_session_supports_payload` compiles against the fork pin only and is `absent` in an upstream manifest; the `check` report gains `optional: ds4_session_supports_payload  present  (core)` and its required-ok count stays 52. Cover `tests/abi/test_probe.py`, `test_manifest.py`, and `test_check.py`; update the build-marked upstream assertion in `test_build.py` so all records except the optional upstream-absent probe remain `ok`.

Add loader tests:
- `test_rebuilt_library_without_payload_probe_loads_absent_and_uses_legacy_support`: build with `all_bindings()` and upstream headers, load, assert `not loaded.has(...)`, and assert the helper returns `True`.
- `test_legacy_manifest_without_payload_probe_record_requires_rebuild`: remove the probe record with `pop(name, None)`, then require `AbiMismatch` matching the probe name, `no probe result recorded`, and `ds4py build`.

Create helper tests with a small library double whose `has`/`fn` behavior is explicit: absent returns `True`; present native `False` and `True` return exactly those values. Add a fake-engine test showing `None`, `False`, and `True` are returned through `Session.file_payloads_supported()` from the worker.

- [ ] **Step 2: Run the native tests to verify RED**

Run: `uv run --frozen --extra agent pytest tests/native/test_bindings.py tests/native/test_loader.py tests/native/test_file_payloads.py tests/abi/test_probe.py tests/abi/test_manifest.py tests/abi/test_check.py -q`

Expected: failures for the missing 53rd binding/helper and the updated optional-core ABI invariant; the legacy-manifest test must fail because loading incorrectly succeeds, not because the fixture lacks the new record.

- [ ] **Step 3: Implement the declaration, helper, session seam, and fake capability**

Add the optional core declaration. Implement the helper at the native boundary and the `NativeSession`/`SessionHandle`/high-level `Session` forwarding needed to invoke it through the worker. Add the fake capability and allow the agent test `fake_engine(..., payload_support=...)` helper to pass it into `FakeNative`; keep the default `None` so every existing test retains legacy native saves.

- [ ] **Step 4: Run native and existing payload/switch smoke tests to verify GREEN**

Run: `uv run --frozen --extra agent pytest tests/native/test_bindings.py tests/native/test_loader.py tests/native/test_file_payloads.py tests/abi/test_probe.py tests/abi/test_manifest.py tests/abi/test_check.py tests/engine/test_payload_jobs.py tests/agent/test_restore.py::test_valid_kv_switch_restores_binding_without_prefill -q`

Expected: PASS, including existing native save/load behavior with the fake probe absent.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/_native/bindings.py src/ds4py/_native/api.py src/ds4py/_native/protocol.py src/ds4py/_native/fake.py src/ds4py/engine/session.py tests/native/test_bindings.py tests/native/test_loader.py tests/native/test_file_payloads.py tests/abi/test_probe.py tests/abi/test_manifest.py tests/abi/test_check.py tests/abi/test_build.py tests/agent/test_runtime.py
git commit -m "feat: add optional file payload capability probe"
```

### Task 2: Complete Stripped Saves and Crash-Safe Stale-KV Removal

**Files:**
- Modify: `src/ds4py/runtime.py`
- Create: `tests/agent/test_payload_support.py`
- Modify: `tests/agent/test_noninteractive.py`
- Modify: `tests/agent/test_compaction_persistence.py`

**Interfaces:**
- Consumes: `Session.file_payloads_supported() -> bool` from Task 1.
- Produces: on `False`, `AgentRuntime.save()` returns a full `SaveRecord` with `payload_bytes == 0`, full token digest/provenance/messages, and no new KV artifact.
- Preserves: existing `_PayloadWrite`, `write_kv_stream_atomic`, token/byte consistency checks, snapshot IDs, and record format on `True`.

- [ ] **Step 1: Write the failing complete stripped-save test**

In `test_payload_support.py`, run a real fake-agent prompt with `payload_support=False`, monkeypatch native staging to fail if called, and save. Assert `payload_bytes == 0`, no `session.kv`, full transcript length/digest/segments, model/quant provenance, a complete harness snapshot with exactly the runtime messages, and a saved event with zero payload bytes. Also save with explicit `True` and assert a nonzero KV payload remains, pinning no default-path regression.

- [ ] **Step 2: Write failing replacement/crash tests on both sides of unlink**

First create a native save with messages/tokens from one prompt, then advance the same runtime with `payload_support=False` so the replacement record and messages differ. Inject crashes at these exact points:
- immediately after `write_record_atomic` publishes the new record but before `Path.unlink` removes stale KV;
- immediately after the stale KV unlink completes.

For each side, assert the new record is zero-payload, the new full snapshot exists, the prior native save's step history remains, and a subsequent switch restores the new messages through rendering while ignoring any stale KV. For the before-unlink case assert the old KV remains; for the after-unlink case assert it is absent. Add a record-publication failure case proving the old record and KV remain until the new record is committed.

- [ ] **Step 3: Write failing noninteractive stripped-save success/failure tests**

Use a real runtime with `payload_support=False` and `run_noninteractive`:
- successful output returns 0 and leaves a complete zero-payload save;
- terminal response failure (`max_tokens` or parser error) returns 1, retains its error diagnostic, and still leaves a complete zero-payload save.

Assert no KV file and `record.tokens > 0` in both cases so an empty-session save cannot satisfy the tests.

- [ ] **Step 4: Write the failing compacted-session stripped-save tests**

Allow the existing compaction runtime fixture to be indirectly parameterized with fake `payload_support`, keeping default `None` for all current tests. With explicit `False`, run two prompts and execute the real `/compact` command before saving. Assert compaction succeeds without a file-payload guard, command text reports only compaction counts and never KV/payload persistence, and the compacted save has full tokens/messages/provenance with `payload_bytes == 0` and no KV. Retain the existing supported compacted KV test and add explicit-`True` coverage if needed to distinguish a present native probe from legacy absence.

- [ ] **Step 5: Run the new tests to verify RED**

Run: `uv run --frozen --extra agent pytest tests/agent/test_payload_support.py tests/agent/test_noninteractive.py tests/agent/test_compaction_persistence.py -q`

Expected: stripped-save and compacted-save tests fail because native staging still writes a nonzero KV file; crash tests fail because stale removal/publication ordering is absent. Existing noninteractive and supported compaction tests remain green.

- [ ] **Step 6: Implement the save guard and ordered stale cleanup**

After active-run draining and the empty-transcript rejection, obtain the worker-confined boolean. Only on `True`, execute the existing `save_payload`/`write_kv_stream_atomic` path and consistency checks. On `False`, set `payload_bytes = 0` without staging or syncing for payload. Always write the complete snapshot and record. Only after `write_record_atomic` succeeds on `False`, call `session.kv.unlink(missing_ok=True)`, then emit `saved` and return. Do not suppress unlink errors: after such an error the published record remains authoritative. Update the save docstring to distinguish native KV saves from complete stripped saves with `payload_bytes=0`; never describe a stripped save as an empty session.

- [ ] **Step 7: Run save and regression tests to verify GREEN**

Run: `uv run --frozen --extra agent pytest tests/agent/test_payload_support.py tests/agent/test_noninteractive.py tests/agent/test_compaction_persistence.py tests/agent/test_restore.py tests/agent/test_store.py tests/agent/test_store_failures.py -q`

Expected: PASS; Task 2's compacted-session test stops at stripped-save assertions, while Task 3 owns its rendered-switch counterpart.

- [ ] **Step 8: Commit**

```bash
git add src/ds4py/runtime.py tests/agent/test_payload_support.py tests/agent/test_noninteractive.py tests/agent/test_compaction_persistence.py
git commit -m "feat: preserve stripped saves without file payloads"
```

### Task 3: Render Planned KV Switches Before Identity Mutation

**Files:**
- Modify: `src/ds4py/runtime.py`
- Modify: `src/ds4py/runtime_restore.py` only if a plan-conversion helper improves type clarity; no restore semantics change required.
- Modify: `tests/agent/test_payload_support.py`
- Modify: `tests/agent/test_compaction_persistence.py`

**Interfaces:**
- Consumes: `RestorePlan(path, payload_bytes, mismatch)` and validated snapshots from the existing switch flow.
- Produces: on explicit `False` with `plan.path == "kv"`, `RestorePlan("render", None, "the live engine does not support file payloads")` before identity hooks or `Session.switch`.
- Preserves: `plan_restore` immediate rendering for zero records and all exact KV token/digest checks on `True`.

- [ ] **Step 1: Write failing forced-render switch tests**

Create a native save, then open a fresh fake engine with explicit `False`. Make `Session.load_payload` raise if called. Switch and assert saved messages/provenance are installed, live native transcript is empty until the next request, no `load_payload` call occurs, the switch event says `restore_path="render"` and names unsupported file payloads, and the next prompt performs a full rebuild prefill. Repeat with explicit `True` and assert exact loaded tokens, zero prefill during switch, and an extend plan on the next request.

- [ ] **Step 2: Add validation-order controls and the failing compacted rendered-switch test**

With explicit `False` and an otherwise restorable KV record, independently corrupt model identity, named snapshot/provenance, and KV headers. Assert the existing errors still occur and the target identity is not adopted. Include a zero-payload record with a stale/corrupt `session.kv` and assert switch neither opens nor reads that artifact. These are characterization controls for behavior that already exists; the new RED behavior is the forced-render test above, not a claim that these controls initially fail.

Add a compacted-session explicit-`False` test that starts from the stripped compacted save covered in Task 2, switches after `/new`, and asserts rendered messages/provenance, no native payload load, and full compacted rebuild on the next prompt. Keep explicit-`True` compacted exact KV coverage as the supported control.

- [ ] **Step 3: Run switch tests to verify RED**

Run: `uv run --frozen --extra agent pytest tests/agent/test_payload_support.py -q`

Expected: the explicit-false native-record test fails because `load_payload` is called; validation and zero-record cases retain their existing errors/render behavior.

- [ ] **Step 4: Implement the pre-identity plan conversion**

Keep record/model/snapshot lookup, `plan_restore`, and `validate_restore_snapshot` in their current order. After validation and before identity hooks or recorder relocation, if the plan is KV and `Session.file_payloads_supported()` is `False`, replace it with the rendered plan and unsupported-payload mismatch. Pass the converted plan into the existing restore callback and switch event. Do not probe or reopen a zero-payload artifact.

- [ ] **Step 5: Run switch, compacted-persistence, and identity-lifecycle regressions to verify GREEN**

Run: `uv run --frozen --extra agent pytest tests/agent/test_payload_support.py tests/agent/test_compaction_persistence.py tests/agent/test_restore.py tests/agent/test_runtime.py -q`

Expected: PASS, including before/after identity hook and restore-failure tests.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/runtime.py src/ds4py/runtime_restore.py tests/agent/test_payload_support.py tests/agent/test_compaction_persistence.py
git commit -m "feat: render switches when file payloads are unsupported"
```

### Task 4: Full Verification and Independent Review

**Files:**
- Modify: `docs/plans/2026-11-12-payload-support.md` (check off completed steps and record any evidence-based rulings)
- Modify: review-finding files only when a P0 or worthwhile P1 finding has concrete evidence.
- No production compaction changes expected; Tasks 2 and 3 own the M4 contact-point coverage.

**Interfaces:**
- Consumes: completed save/switch behavior and compacted-session tests from Tasks 2 and 3.
- Preserves: merged M4 checkpoint, summary snapshot restore, transcript replacement, and `/compact` behavior.

- [ ] **Step 1: Run focused M4 regressions and review the final diff against the spec**

Run: `uv run --frozen --extra agent pytest tests/agent/test_compaction_persistence.py tests/agent/test_compaction_capability.py tests/engine/test_compaction.py tests/agent/test_payload_support.py -q`

Expected: PASS. Confirm the diff contains no file-payload predicate in checkpoint, summary, replacement, or `/compact` code and no out-of-scope files.

- [ ] **Step 2: Run all required checks**

From the worktree, inside devenv:

```bash
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
HOME=$(mktemp -d) uv run --frozen --extra agent pytest -m "not build and not live" -q
grep -rniE "satyrn|dflash|qwen35" src/
```

Expected: static gate PASS; portable suite PASS with counts recorded; grep prints nothing and exits 1.

- [ ] **Step 3: Run `/review-loop` and address evidence-backed findings**

Give one fresh reviewer the `thermo-nuclear-code-quality-review` skill for structure/maintainability and keep another on correctness/tests against the spec and merged M4 contracts. Fix P0 findings and worthwhile P1 findings with new RED-GREEN tests; ask the user before any restructuring beyond this approved scope. Review again only after a material fix.

- [ ] **Step 4: Commit review fixes and report**

If review produced changes, commit them locally with a focused message and rerun all three required checks. Report every commit hash, exact check results and test counts, and every departure from the spec with its reason. Do not push or open a PR.
