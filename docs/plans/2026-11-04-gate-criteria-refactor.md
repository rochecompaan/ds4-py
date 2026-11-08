# M3 gate criteria refactor implementation plan

- **Date:** 2026-11-04
- **Design:** `docs/specs/2026-11-04-gate-criteria-refactor-design.md`
- **Branch:** `gate-criteria-refactor`
- **Base:** `bakeoff-workspace-isolation` at `be45b7cdd9a12dcf399ccf75b5da689d09b7d657`
- **Execution status:** Phase 1 only. Do not implement until the user approves the design and this plan.

## Goal

Move the six M3 gates to checks that isolate ds4-py's driver, lifecycle, cache, and restore behavior. Use same-host logits equivalence for C/ds4-py comparisons and exact replay only for ds4-py's own traces. Keep exact C replay as a diagnostic.

## Constraints

- Phase 2 changes are limited to `bench/`, `tests/`, and gate documentation.
- Do not change prompts, sampling, native source, `libds4`, or existing token/trace records.
- Add provenance and diagnostic sidecars instead of rewriting native C traces.
- Write focused failing tests before comparison, validation, deserialization, report, and fake-engine behavior changes.
- Do not add automated tests for document text, static configuration, or C source structure. Verify those directly.
- Do not run instance lifecycle commands. The user must approve and provision any live instance.

## Baseline

The new worktree's non-live bench baseline was run before these documents:

```text
uv run --extra agent pytest tests/bench -m "not live and not build" -q
120 passed, 8 warnings in 13.83s
```

The warnings are existing harness deprecation warnings and one personal-skill catalog warning. There were no baseline failures.

---

## Task 1: Add provenance and pure comparison primitives

**Files**

- Create `bench/provenance.py`.
- Create `bench/comparison.py`.
- Create `tests/bench/test_provenance.py`.
- Create `tests/bench/test_comparison.py`.

### Step 1: Write failing comparison tests

Add these behavior cases before implementation:

1. `test_topk_requires_identical_ordered_ids_and_positions`.
2. `test_topk_accepts_small_raw_logit_delta_and_rejects_tolerance_boundary`.
3. `test_topk_rejects_missing_or_short_capture`.
4. `test_rate_comparison_uses_aggregated_tokens_and_seconds`.
5. `test_rate_comparison_rejects_zero_denominator_and_relative_difference_above_tolerance`.
6. `test_cached_fraction_uses_absolute_tolerance`.
7. `test_provenance_rejects_different_instance_session_model_or_library`.
8. `test_provenance_rejects_different_session_shape_or_missing_chunk_plan`.
9. `test_invalid_provenance_is_distinct_from_numeric_mismatch`.

Run:

```bash
uv run --frozen --extra agent pytest \
  tests/bench/test_comparison.py tests/bench/test_provenance.py -q
```

Expected: collection fails because the two production modules do not exist.

### Step 2: Implement the value types and comparison functions

In `bench/comparison.py`, define:

```python
@dataclass(frozen=True)
class TopLogit:
    token: int
    logit: float


@dataclass(frozen=True)
class StepCapture:
    position: int
    top: tuple[TopLogit, ...]
    selected_token: int


@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    detail: str
    valid: bool = True
    max_abs_delta: float | None = None


@dataclass(frozen=True)
class RateEvidence:
    task_id: str
    prefill_tokens: int
    prefill_seconds: float
    decode_steps: int
    decode_seconds: float


TOP_K = 12
MAX_LOGIT_STEPS = 16
LOGIT_ABS_TOLERANCE = 1e-5
RATE_RELATIVE_TOLERANCE = 0.25
CACHED_FRACTION_ABS_TOLERANCE = 0.15


def compare_step(c: StepCapture, ds4py: StepCapture) -> ComparisonResult: ...


def compare_rate(c: RateEvidence, ds4py: RateEvidence) -> ComparisonResult: ...


def compare_cached_fraction(c: float, ds4py: float) -> ComparisonResult: ...
```

`ComparisonResult` keeps comparison code independent of `report.py`. The report converts it into `GateResult`. Missing, short, non-finite, or position-mismatched captures return `valid == false`. A valid numeric difference outside tolerance returns `valid == true` and `passed == false`.

`compare_step` must require equal positions, exactly 12 ordered entries, identical token IDs, finite values, and `max_abs_delta < 1e-5`. Use the actual length for short zero-token rounds only at orchestration level; never accept a short top-k list.

Rate aggregation is `sum(tokens) / sum(seconds)`, not a mean of per-round rates. Relative difference is `abs(ds4py-c)/c` and must be `<= 0.25`. Different task IDs, non-finite values, and zero C rates are invalid. Cached-fraction difference must be `<= 0.15`; non-finite fractions are invalid.

In `bench/provenance.py`, define:

```python
@dataclass(frozen=True)
class SyncCall:
    label: str
    input_tokens: int
    cached_tokens: int
    prefill_tokens: int
    position: int


@dataclass(frozen=True)
class ProgressChunk:
    done: int
    total: int


@dataclass(frozen=True)
class RoundProvenance:
    round_index: int
    think: int
    prompt_sha256: str
    sync_calls: tuple[SyncCall, ...]
    progress_chunks: tuple[ProgressChunk, ...]


@dataclass(frozen=True)
class TraceProvenance:
    bakeoff_session_id: str
    instance_id: str
    gpu_model: str
    gpu_driver_version: str
    ds4_revision: str
    ds4py_revision: str
    libds4_sha256: str
    model_path: str
    model_sha256: str
    backend: str
    ctx: int
    temperature: float
    seed: int
    response_budget: int
    reset_policy: str
    trace_sha256: str
    source_trace_sha256: str
    rounds: tuple[RoundProvenance, ...]


def write_trace_provenance(
    trace_path: Path,
    provenance: TraceProvenance,
) -> Path: ...


def validate_session_pair(c: TraceProvenance, ds4py: TraceProvenance) -> tuple[str, ...]: ...


def validate_pair(c: TraceProvenance, ds4py: TraceProvenance) -> tuple[str, ...]: ...
```

`write_trace_provenance` writes `<trace>.provenance.json`. Both validators return all mismatch reasons; they do not raise on the first mismatch. `validate_session_pair` checks instance, bake-off session, GPU, model, library, backend, context, sampling, and response budget. `validate_pair` calls it and then checks source trace, per-round think and prompt hash, reset policy, round cached/prefill lengths, and resulting positions. Paired drivers must have the same `source_trace_sha256`; their own `trace_sha256` values identify different raw outputs and need not match. Full progress chunks are required only for reference-driver comparisons. Actual task histories use `validate_session_pair` because their prompts and tool histories may differ.

### Step 3: Re-run focused tests and static checks

```bash
uv run --frozen --extra agent pytest \
  tests/bench/test_comparison.py tests/bench/test_provenance.py -q
uv run --frozen --extra agent ruff check \
  bench/comparison.py bench/provenance.py \
  tests/bench/test_comparison.py tests/bench/test_provenance.py
```

Expected: all focused tests pass; Ruff reports no errors.

---

## Task 2: Add terminal, cap, rate, and cached-fraction metrics to Gates 1–4

**Files**

- Modify `bench/models.py`.
- Modify `bench/metrics.py`.
- Modify `bench/report.py`.
- Modify `bench/run_bakeoff.py` only for new evidence arguments and response-budget passing.
- Modify `tests/bench/test_metrics.py`.
- Modify `tests/bench/test_runner.py`.
- Modify `tests/bench/test_report_commands.py` for schema changes.

### Step 1: Write failing lifecycle and report tests

Add or update tests to prove:

1. Gate 1's one-task slack is unchanged.
2. Gate 2 rejects parser errors, missing outcomes, unknown terminal reasons, and inconsistent tool-call terminal evidence.
3. Gate 2 accepts bad-argument retries and ordinary `max_tokens` stops.
4. Gate 2 reports per-arm and per-task cap counts without using those counts for pass/fail.
5. A first-attempt pass has no retry; a failed first attempt has exactly one retry containing the existing grader feedback.
6. Gate 3 no longer fails on `wall_seconds > 1.5 * C`.
7. Gate 3 fails on rate difference `> 0.25`, cached-fraction difference `> 0.15`, or unexplained non-extension.
8. Gate 4's current interrupt/resume assertions are unchanged.
9. Missing paired rates or mismatched provenance produce `valid == false`, not a numeric mismatch.
10. The compact report prints cap metrics, paired rates, and `INVALID` where required.

Update `test_gate_three_enforces_time_and_only_unexplained_non_extensions` into separate rate/cache and non-extension tests. Keep its non-extension assertion. Update `test_gates_one_to_five_pass_with_complete_evidence` and the report-table test with the new required evidence.

Run:

```bash
uv run --frozen --extra agent pytest \
  tests/bench/test_metrics.py tests/bench/test_runner.py \
  tests/bench/test_report_commands.py -q
```

Expected: the new cases fail because terminal/rate/provenance fields and new gate logic are absent.

### Step 2: Extend event and C-trace summaries

Add `terminal_reasons: tuple[str, ...]` to `EventSummary`, with an empty default only for construction paths that explicitly represent no rounds. Finished real attempts must not use that default to hide missing events.

In `summarize_events`:

- Append every `Stopped.reason` in event order.
- Reject unknown terminal reasons.
- Validate terminal/tool consistency without treating `max_tokens` as an error.
- Preserve all existing parser/tool/non-extension checks.

Add:

```python
def classify_c_terminal(round_: ReplayRound, response_budget: int) -> str:
    """Return context_limited, prefix_only, max_tokens, or recorded_boundary."""
```

Classification order is `context_limited`, `prefix_only`, `max_tokens`, then `recorded_boundary`. A sampled length greater than the response budget is malformed evidence. Equality uses the runner contract; do not infer cap behavior from generated text.

Pass `response_budget` into `metrics_from_trace`. `combine_summaries` concatenates terminal reasons and combines all existing metrics without dropping the new evidence.

### Step 3: Refactor report Gates 1–4

Extend `GateResult`:

```python
@dataclass(frozen=True)
class GateResult:
    passed: bool
    detail: str
    valid: bool = True
```

`build_report` receives paired rate evidence and provenance in addition to existing task/restore evidence. Keep Gate 1 unchanged.

Gate 2 evaluates only hard lifecycle evidence by default. Cap counts are report metrics. If `cap_parity` is supplied because `--cap-parity` was explicit, include its result in Gate 2; otherwise record it as not run without failing.

Gate 3 consumes `RateEvidence` from Task 4's driver runs and actual task cached fractions. Remove the wall-time ratio condition. Keep `invalid_non_extend_rounds` rejection. Wall time remains in `METRICS` for diagnostics.

Add rate and cap sections to `BakeoffReport.to_dict` and `compact_table`. Print `INVALID` when `valid` is false, then `FAIL` or `PASS`.

### Step 4: Re-run focused tests

```bash
uv run --frozen --extra agent pytest \
  tests/bench/test_metrics.py tests/bench/test_runner.py \
  tests/bench/test_report_commands.py -q
```

Expected: all three files pass, including all unchanged Gate 1, Gate 2, and Gate 4 assertions.

---

## Task 3: Promote the two reference drivers and record common NDJSON evidence

**Files**

- Modify `bench/c_replay_diagnostic.c`.
- Create `bench/ds4py_replay_diagnostic.py` from the approved raw probe logic.
- Create `bench/driver_cases.py`.
- Modify `bench/README.md` with the build and standalone run commands.
- Create `tests/bench/test_driver_cases.py` for manifest and NDJSON boundary validation only.

### Step 1: Define a shared case manifest and evidence parser

Write failing tests for:

- all selected rounds are exported in order;
- `steps == min(16, len(round.sampled))`;
- cached prefix is never greater than prompt length;
- malformed NDJSON, duplicate steps, wrong top-k length, negative duration, and wrong position are rejected;
- zero-token rounds produce sync evidence and zero steps, not a fabricated token.

Implement `bench/driver_cases.py`:

```python
@dataclass(frozen=True)
class DriverCase:
    task_id: str
    attempt: int
    round_index: int
    token_file: Path
    prompt_tokens: int
    split_at: int
    steps: int
    think: int


def export_trace_cases(
    trace: ReplayTrace,
    *,
    task_id: str,
    attempt: int,
    destination: Path,
    max_steps: int = MAX_LOGIT_STEPS,
) -> Path: ...


@dataclass(frozen=True)
class DriverRoundEvidence:
    source_trace: Path
    case: DriverCase
    sync_calls: tuple[SyncCall, ...]
    progress_chunks: tuple[ProgressChunk, ...]
    steps: tuple[StepCapture, ...]
    prefill_seconds: float
    decode_seconds: float
    terminal_reason: str | None


def read_driver_ndjson(path: Path) -> tuple[DriverRoundEvidence, ...]: ...
```

Use one manifest per source trace. Use a simple TSV manifest so the C driver needs no JSON parser. Reject tabs and newlines in paths/IDs. Each token file contains prompt IDs followed by the recorded sampled IDs needed by the diagnostic. One engine load per driver and manifest is allowed; every case still gets a fresh native session.

The NDJSON record types are:

- `round_start`: task, attempt, round, case shape;
- `sync`: label, input/cached/prefill lengths, position, elapsed seconds;
- `prefill`: `done` and `total` progress;
- `step`: step index, pre-sample position, ordered top-12 raw logits, selected token;
- `round_end`: total prefill/decode tokens and seconds, terminal reason where available.

### Step 2: Extend the C diagnostic without native changes

Keep the existing single-case CLI for investigation. Add `--cases FILE` for a manifest.

Use:

- `ds4_session_set_progress` to emit actual `Prefill` chunks;
- `clock_gettime(CLOCK_MONOTONIC)` for sync and decode intervals;
- one loaded `ds4_engine` per manifest;
- a new `ds4_session` per case to provide the fresh-session shape;
- the existing `ds4_session_top_logprobs` and `ds4_session_token_logprob` calls;
- the existing greedy `ds4_decode` behavior and recorded think mode.

Do not change sampler parameters to make outputs agree. Stop NDJSON timing before file writes and exclude engine/session setup.

Document the existing single-case build command and the new manifest mode in `bench/README.md`:

```bash
cc -O2 -std=c11 -Wall -Wextra -Werror -I"$DS4_SRC" \
  bench/c_replay_diagnostic.c \
  -L"$DS4PY_HOME/lib" -Wl,-rpath,"$DS4PY_HOME/lib" -lds4 \
  -o "$DIAG/c-replay-diagnostic"
```

### Step 3: Add the supported ds4-py probe

Port `probe_ds4py.py` into `bench/ds4py_replay_diagnostic.py` with the same manifest and NDJSON contract.

- Load one `Engine` per manifest.
- Use `ResetJob` between cases for a fresh native session without model reload.
- Sync `prompt[:cached]`, then `prompt`.
- Use the existing `RoundStream`, `GenerateJob`, and `SamplingPolicy.for_engine(... temperature=0.0, seed=1)` path.
- Capture top-12 raw logits before each sample through the existing logits view.
- Emit progress and monotonic timing equivalent to the C driver.
- Do not execute tools or change production files.

Run boundary tests and direct CLI checks:

```bash
uv run --frozen --extra agent pytest tests/bench/test_driver_cases.py -q
uv run --frozen --extra agent python bench/ds4py_replay_diagnostic.py --help
```

Expected: parser tests pass and `--help` exits zero. The `--help` check is direct verification, not a new static-structure test.

---

## Task 4: Implement same-host Gate 6 and ds4-py self-replay

**Files**

- Create `bench/driver_equivalence.py`.
- Create `bench/self_replay.py`.
- Modify `bench/replay_c_traces.py` only to make standalone output diagnostic-only and share replay helpers where useful.
- Modify `bench/report.py` for the two Gate 6 parts.
- Rewrite the affected cases in `tests/bench/test_bakeoff_replay.py`.
- Create `tests/bench/test_driver_equivalence.py`.
- Create `tests/bench/test_self_replay.py`.

### Step 1: Write failing Gate 6 comparison tests

Use synthetic NDJSON and fake engines. Prove:

1. Every selected C attempt and round is present in Part A.
2. Equal ordered top-12 captures pass for all compared steps.
3. A ranking reorder fails even when all values are close.
4. A delta at or above `1e-5` fails.
5. Different prefill boundaries, reset policies, instance IDs, session IDs, or library hashes are invalid.
6. A zero-token round is sync-only; a full set with no decode steps is invalid.
7. Rates from the same evidence feed Gate 3 without another model run.
8. Self-replay reproduces complete ds4-py token sequences and terminal boundaries.
9. Self-replay reports task/round/token index on mismatch and never executes a tool.
10. Self-replay on a different host/session is invalid rather than a token mismatch.
11. Standalone exact C replay leaves Gate 6 unchanged.

Run:

```bash
uv run --frozen --extra agent pytest \
  tests/bench/test_driver_equivalence.py \
  tests/bench/test_self_replay.py tests/bench/test_bakeoff_replay.py -q
```

Expected: new modules are missing and old standalone Gate 6 behavior fails the revised test.

### Step 2: Implement cross-driver orchestration

In `bench/driver_equivalence.py`, implement:

```python
def compare_driver_rounds(
    c_rounds: Sequence[DriverRoundEvidence],
    ds4py_rounds: Sequence[DriverRoundEvidence],
    *,
    c_provenance: TraceProvenance,
    ds4py_provenance: TraceProvenance,
) -> DriverEquivalenceResult: ...


def rates_by_task(
    rounds: Sequence[DriverRoundEvidence],
) -> dict[str, RateEvidence]: ...
```

The runner invokes each reference driver once per source-trace manifest, writes raw NDJSON and provenance sidecars, and then calls the pure comparator. It must not drop failed rounds from the report.

Gate 6 Part A passes only when all selected rounds are covered, provenance is valid, every compared step passes, and the set has decode coverage. No tool-call-round coverage requirement remains.

### Step 3: Implement self-replay

Reuse the existing replay reader/runner where it already implements the required worker behavior. Do not translate traces into another format.

`bench/self_replay.py` selects every ds4-py attempt trace, preserves sequential session state and cached boundaries, generates the complete recorded round, and compares all committed IDs and terminal boundaries. It uses no executable tools.

Move the old exact-token mismatch tests to this module's tests. Exact C replay helpers can remain in `bench/replay_c_traces.py`; shared helpers must not change their behavior.

Change the standalone C replay command to write `replay-diagnostic.json` or the `replay_diagnostic` report section. It must not update `gates.gate-6`.

### Step 4: Implement the combined Gate 6 result

Gate 6 passes only when Part A and Part B both pass and are valid. The detail reports each part separately. Keep `replay_diagnostic` outside the gate.

Re-run the focused command from Step 1. Expected: all pass.

---

## Task 5: Replace Gate 5 token equality with plan, position, and first-step logits

**Files**

- Modify `bench/restore_check.py`.
- Modify `bench/report.py` only if restore evidence serialization needs new fields.
- Modify `tests/bench/test_bakeoff_restore.py`.
- Reuse `StepCapture`, `ComparisonResult`, and `compare_step` from Task 1.

### Step 1: Write failing restore comparison tests

Update the old exact-token tests to prove:

1. Case 1 must have zero switch prefill, `extend`, and `Synced.pos == len(followup_prompt)`.
2. Case 2 must have zero switch prefill, `rebuild`, real prefill, and the same final prompt position.
3. Equal first-step positions, ordered top-12 IDs, and delta below `1e-5` pass even when later sampled tokens differ.
4. A first-step ranking reorder or delta at the tolerance fails.
5. Different restore hosts/sessions are invalid.
6. Different later terminal reasons are recorded but do not fail Gate 5.
7. Missing first-step evidence or a case-1 `rebuild` fails.
8. Existing save completeness, prompt composition, save-in-`finally`, and disposable-workspace tests remain unchanged.

Run:

```bash
uv run --frozen --extra agent pytest tests/bench/test_bakeoff_restore.py -q
```

Expected: new first-step/position cases fail and the old 256-token equality assertion is no longer authoritative.

### Step 2: Capture restore logits through the public hook

Extend `RestoreRun` with:

```python
followup_position: int
first_step: StepCapture | None
provenance_path: Path | None
```

In `_restore_operation`:

- Keep the existing public `/switch` path.
- Capture `Synced.plan.kind`, `Synced.pos`, and prefill counts.
- After switch and before `runtime.run(FOLLOWUP)`, call `runtime.model.session.add_logits_hook`.
- The hook copies the first call's position and top-12 raw logits into local state, then returns without writing on later calls.
- The hook performs no blocking I/O, engine/session calls, or harness calls.
- After the run, write restore metadata/sidecar JSON. Do not add logits to the ordinary trace stream.

Add `validate_restore_pair` in `bench/restore_check.py`. It calls `validate_session_pair`, then requires the same original save-record hash and fixed follow-up prompt hash. It deliberately does not require equal sync kinds, cached lengths, or prefill lengths: case 1 must `extend` and case 2 must `rebuild`. Those plan requirements remain separate assertions.

In `check_restore`, replace `_compare_restore`'s 256-token loop with plan, position, restore-provenance, and `compare_step` checks. Retain the full token traces as diagnostic artifacts; do not delete them.

### Step 3: Re-run restore and adjacent agent tests

```bash
uv run --frozen --extra agent pytest \
  tests/bench/test_bakeoff_restore.py tests/agent/test_restore.py \
  tests/agent/test_prefill_trace.py -q
```

Expected: all pass. Existing trace-parser tests require no production change.

---

## Task 6: Integrate the runner, commands, live guards, and documentation

**Files**

- Modify `bench/run_bakeoff.py`.
- Modify `bench/report.py` and `bench/models.py` for final schema wiring.
- Modify `bench/README.md` and `bench/gate-repairs.md`.
- Modify `tests/bench/test_runner.py` and `tests/bench/test_report_commands.py` integration fixtures.
- Modify `tests/live/test_bakeoff.py`.
- Retire `tests/live/test_replay_parity.py` as an acceptance test and create `tests/live/test_driver_equivalence.py` for an explicitly guarded one-trace diagnostic run.
- Add a supersession note to the M3 gate section in `docs/specs/2026-10-21-ds4-py-design.md` pointing to the new design. Preserve the old M3 plan as history.

### Step 1: Wire one session through all phases

Use this order:

1. Capture session/host/library provenance and generate one `bakeoff_session_id`.
2. Run C and ds4-py tasks sequentially.
3. Run restore cases and Gate 4 without concurrent engines.
4. Export all selected C case manifests and run C/ds4-py reference drivers sequentially.
5. Run ds4-py self-replay sequentially.
6. Build one report with Gates 1–6 and optional diagnostics.

Add runner options:

```text
--instance-id ID              (or DS4_BAKEOFF_INSTANCE_ID)
--c-driver PATH               built c-replay-diagnostic
--cap-parity                  optional Gate 2 subcheck
```

The default `bakeoff_session_id` is generated once and printed. All raw evidence and sidecars inherit it. The runner refuses mixed-host/session evidence and refuses to overwrite an existing result root, as it does now.

Targeted manifests run the same comparison logic but write `certification: "targeted"`; the full manifest writes `certification: "full-12-task"`.

### Step 2: Update commands and live guards

Document:

- C driver build;
- targeted live run;
- full live run;
- restore recheck;
- new driver-equivalence/self-replay recheck;
- exact C replay as diagnostic-only;
- provenance fields and invalid comparisons;
- non-live checks.

`tests/live/test_bakeoff.py` must still skip unless `DS4_RUN_BAKEOFF=1` and all required model/agent/benchmark variables are present. Add the instance ID and C driver to the required live inputs. Replace the old `replay` assertion with assertions for both Gate 6 evidence parts and provenance.

Retire the old exact C live parity test as a gate. Keep its manual procedure in documentation under diagnostics. `tests/live/test_driver_equivalence.py` runs one retained C trace through both reference drivers only when `DS4_RUN_DRIVER_EQUIVALENCE=1`, `DS4_TEST_MODEL`, `DS4_REPLAY_TRACE`, `DS4_BAKEOFF_INSTANCE_ID`, and `DS4_C_DRIVER` are all present. It checks Part A and writes provenance; the full bake-off remains the certification path. Do not make `-m live` alone start any GPU work.

### Step 3: Run integration and collection checks

```bash
uv run --frozen --extra agent pytest tests/bench -m "not live and not build" -q
env -u DS4_RUN_BAKEOFF -u DS4_TEST_MODEL -u DS4_AGENT -u POLYGLOT_BENCHMARK \
  uv run --frozen --extra agent pytest tests/live/test_bakeoff.py -q
uv run --frozen --extra agent python bench/run_bakeoff.py --help
uv run --frozen --extra agent python bench/replay_c_traces.py --help
```

Expected: all non-live bench tests pass, the live test skips without GPU work, and both CLI checks exit zero.

---

## Task 7: Run full verification, then approved same-host live certification

### Step 1: Run the complete static and portable gates

From the worktree root:

```bash
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
uv run --frozen --extra agent pytest -m "not build and not live" -q
git diff --check
```

Expected: all commands exit zero. Record exact test counts. Do not claim live verification from these results.

### Step 2: Build the reference C driver on the approved host

After the user approves and provisions an instance:

```bash
export DS4_SRC=/workspace/ds4
export DS4PY_HOME=/workspace/native
export DIAG=/tmp/ds4-gate-criteria-driver
mkdir -p "$DIAG"
cc -O2 -std=c11 -Wall -Wextra -Werror -I"$DS4_SRC" \
  bench/c_replay_diagnostic.c \
  -L"$DS4PY_HOME/lib" -Wl,-rpath,"$DS4PY_HOME/lib" -lds4 \
  -o "$DIAG/c-replay-diagnostic"
```

Expected: compile and link exit zero with no warnings.

### Step 3: Run a targeted same-host check

Use the prepared four-task manifest and a new result root:

```bash
export DS4_BAKEOFF_INSTANCE_ID="$INSTANCE_ID"
uv run --frozen --extra agent python bench/run_bakeoff.py \
  --manifest bench/gate-repairs.yaml \
  --benchmark "$POLYGLOT_BENCHMARK" \
  --repo-root "$PWD" \
  --results /workspace/results/gate-criteria-targeted \
  --model "$DS4_TEST_MODEL" \
  --agent "$DS4_AGENT" \
  --backend cuda \
  --tokens 8192 \
  --instance-id "$INSTANCE_ID" \
  --c-driver "$DIAG/c-replay-diagnostic"
```

Expected: all selected gates are valid and pass. The report says `certification: "targeted"`. If it fails, inspect raw provenance, sync plans, first differing step, and rate evidence before any code change. Do not start the full run to hide a targeted failure.

### Step 4: Run full 12-task certification after targeted success

```bash
uv run --frozen --extra agent python bench/run_bakeoff.py \
  --manifest bench/bakeoff.yaml \
  --benchmark "$POLYGLOT_BENCHMARK" \
  --repo-root "$PWD" \
  --results /workspace/results/gate-criteria-full \
  --model "$DS4_TEST_MODEL" \
  --agent "$DS4_AGENT" \
  --backend cuda \
  --tokens 8192 \
  --instance-id "$INSTANCE_ID" \
  --c-driver "$DIAG/c-replay-diagnostic"
```

Expected: all six gates are valid and pass on one instance in one bake-off session. Archive `bakeoff.json`, `report.txt`, provenance sidecars, raw NDJSON, and logs. Do not archive model payloads, KV files, or local environments in git.

Optional: add `--cap-parity` to a separate explicit run if the user requests the extra cap-budget check. Do not enable it silently.

No instance start, stop, destroy, or creation command is part of this plan. Ask the user before any lifecycle action.

---

## Self-review

- **Placeholder scan:** no `TBD`, `TODO`, or undecided interface remains. The only approval choice is the explicitly stated Gate 3 tolerance review.
- **Spec coverage:** Gate 1 and Gate 4 remain unchanged; Gate 2 separates lifecycle from cap metrics; Gate 3 removes wall-ratio gating; Gate 5 checks extend/position/first-step logits; Gate 6 has same-host logits and self-replay; exact C replay remains a diagnostic.
- **Testing Value Gate:** new tests cover comparison, validation, report, and fake-engine behavior. GPU behavior has live commands. Documentation, CLI help, and C compilation use direct verification.
- **Guardrails:** Phase 1 writes only this plan and its design. No `bench/` or `src/` implementation, evidence-file mutation, or instance lifecycle action is included.
- **Open dependency:** the source worktree's uncommitted `FINDINGS.md` and raw evidence are cited but not copied. The user retains control of those files.
