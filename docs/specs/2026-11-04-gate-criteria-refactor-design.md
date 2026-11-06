# M3 gate criteria refactor design

- **Date:** 2026-11-04
- **Status:** Phase 1 proposal; implementation is not approved
- **Branch:** `gate-criteria-refactor`
- **Base:** `bakeoff-workspace-isolation` at `be45b7cdd9a12dcf399ccf75b5da689d09b7d657`
- **Scope:** bake-off evidence, comparison logic, reports, and reference drivers

## 1. Purpose

M3 must show that ds4-py is a stable driver at the seam it owns. It must not require greedy token equality when the native library produces different results after a prefill-shape change or on another host.

The six gates will test these separate properties:

1. The agent can solve the tasks.
2. The harness handles parser, tool, retry, and terminal events correctly.
3. The driver has comparable throughput and uses the KV cache correctly.
4. An interrupted run can resume.
5. A saved session can restore and extend the correct state.
6. Both drivers expose the same logits under the same conditions, and ds4-py can replay its own traces.

This design replaces criteria that failed for reasons outside the ds4-py seam. It does not weaken parser, lifecycle, restore, or self-consistency requirements.

## 2. Evidence and design choice

The primary findings are in `docs/reports/evidence/2026-11-04-logits-probe/FINDINGS.md` in the `bakeoff-workspace-isolation` worktree. That file and its raw evidence belong to the user's uncommitted source-worktree changes. This Phase 1 branch does not copy or alter them.

The evidence shows:

- With the same sync shape, the ds4-py worker and the C driver produced identical ordered top-12 logits and greedy tokens for 16 steps in every probed case.
- The largest observed raw-logit difference in those matched cases was about `5e-8`.
- Split and single-shot prefill changed logits by up to `1.511933` and changed near-tie argmax results in both drivers.
- A C driver on another host did not reproduce C's own retained trace.
- C-driven greedy generation entered repetition loops on ds4-py prompts. The loops are not proof of a ds4-py defect.

The old Gate 6 also passed only 59 of 200 rounds in the cross-feed diagnostic. Required capped rounds failed near their first tokens. These results do not isolate a ds4-py error.

### Chosen approach

Use same-host, same-session logits equivalence for cross-driver checks. Use exact replay only for ds4-py's own traces under the same recorded conditions. Keep exact C replay as a diagnostic.

### Rejected approaches

- **Keep cross-host exact C replay as a gate.** The C driver does not meet this criterion against its own retained traces.
- **Remove token-level checks entirely.** This would lose the useful same-host driver-equivalence and self-replay signals.
- **Compare only task outcomes.** Task outcomes do not isolate parser, restore, cache, or driver defects.

## 3. Common definitions

### 3.1 Bake-off session and instance

- A **bake-off session** is one runner invocation with one generated `bakeoff_session_id`. Both arms, all probes, restore checks, and self-replay inherit that ID.
- An **instance** is one physical or virtual host identified by an explicit `instance_id`. The live runner accepts `--instance-id` or `DS4_BAKEOFF_INSTANCE_ID`. It must not guess an instance ID.
- A **session shape** includes the model, context, backend, think mode, sampling policy, prompt tokens, cached boundary, sync-call sequence, session-reset policy, and decode position.
- **Invalid** means the evidence cannot support a comparison. Invalid evidence makes the affected gate not pass and makes the command exit nonzero. The report distinguishes it from a valid numeric mismatch.

### 3.2 Reference drivers

Two supported diagnostic drivers produce the same NDJSON evidence shape:

1. `bench/c_replay_diagnostic.c` uses the native C API directly.
2. `bench/ds4py_replay_diagnostic.py` is the supported form of `docs/reports/evidence/2026-11-04-logits-probe/raw/probe_ds4py.py` and uses the ds4-py worker.

The C file is already in `bench/`. Phase 2 promotes it from a standalone diagnostic to a documented built tool. It remains diagnostic code; it is not part of `libds4`.

Build it with the tested command shape:

```bash
cc -O2 -std=c11 -Wall -Wextra -Werror -I"$DS4_SRC" \
  bench/c_replay_diagnostic.c \
  -L"$DS4PY_HOME/lib" -Wl,-rpath,"$DS4PY_HOME/lib" -lds4 \
  -o "$DIAG/c-replay-diagnostic"
```

Both drivers record pre-sample position, ordered top-12 raw logits, selected token, sync calls, progress chunks, and monotonic timing. Model loading, session creation/reset, token-file reads, and NDJSON writes are outside the timed prefill and decode intervals. Both timers cover equivalent driver work, including logits capture.

### 3.3 Provenance

Every trace has a sibling `<trace>.provenance.json` record linked by the trace SHA-256. Existing prompt, sampled-token, event, and terminal records do not change. A provenance record contains:

- `bakeoff_session_id` and `instance_id`;
- GPU model and GPU driver version;
- `ds4` source revision, ds4-py source revision, and `libds4` SHA-256;
- model path and model SHA-256;
- the output trace SHA-256 and, for a replay, the shared source-trace SHA-256;
- backend, context, Q8/FP16 cache limit in MiB, temperature, seed, and response budget;
- the session-reset policy;
- each round's think mode, prompt SHA-256, input length, cached length, prefill length, sync-call sequence, and resulting position;
- actual progress chunks when the driver exposes them.

A native C task trace supplies its recorded cached/suffix plan. The reference C driver uses `ds4_session_set_progress` to record actual chunks. The ds4-py worker already emits equivalent `Prefill` progress. Chunk-sensitive comparisons require full chunk records from both reference drivers. A cached/suffix record alone is enough for task-level cached-prefix metrics, which do not compare chunk-sensitive logits or timings.

The report repeats the session-level provenance and links every evidence set to its trace record. Paired reference drivers must name the same source-trace hash and per-round prompt hashes; their own output trace hashes may differ. Token, logit, or timing comparisons with different instance IDs, session IDs, source traces, prompt hashes, model/library identities, or required session-shape fields are invalid. The runner must not silently compare them.

### 3.4 Shared numeric rules

- `TOP_K = 12`.
- `MAX_LOGIT_STEPS = 16`.
- `LOGIT_ABS_TOLERANCE = 1e-5`. The largest absolute difference must be strictly below this value.
- A top-k ranking match means the same ordered token IDs at every compared step. It does not mean set equality.
- The `1e-5` tolerance is 200 times the observed `5e-8` matched-driver difference. It is far below observed prefill-shape differences of `0.593376` and `1.511933`.

Both reference drivers set `DS4_CUDA_Q8_F16_CACHE_MB=5120` before native engine open. The bake-off session exports the same value to its subprocesses. Every provenance sidecar records it as the required `q8_f16_cache_mb` field, and any cross-driver mismatch makes the comparison invalid. The 5 GiB limit is below the smallest observed automatic cutoff of 5.77 GiB, leaving more GPU memory headroom while retaining the cached path.

At ds4 revision `5379c1c9ac315f1540371b06a3db88491d4059ff`, an unset limit means `UINT64_MAX` (`ds4_cuda.cu:1560-1563`). Native code then calls `cudaMemGetInfo` for each allocation and stops caching when the request exceeds free memory or would leave less than the reserve (`ds4_cuda.cu:1647-1675`). The printed `cached` value is the accumulated Q8-to-FP16 expansion allocation, not the KV cache size. The fixed limit makes the cutoff independent of small free-VRAM differences between process starts.

This cache limit changes logits because it changes kernel selection. A cached projection uses FP16 weights and activations with `cublasGemmEx`; after the cutoff, the same projection falls back to native Q8 kernels (`ds4_cuda.cu:16470-16504`). These arithmetic paths can round and accumulate differently. The 2026-11-04 probe's `5e-8` agreements occurred under equal cache values; old/new disagreements later correlated with unequal values. The kernel-path difference is confirmed in source; no more specific accumulation-order cause is claimed.

## 4. Gate decisions

### Gate 1: task comparison — unchanged

For the same selected tasks in one bake-off session:

```text
ds4py_passes >= c_passes - 1
```

The report still separates pass counts, attempt counts, grader outcomes, and diagnostics. Timing and token counts do not affect this gate.

### Gate 2: lifecycle — split hard requirements from cap metrics

#### Hard pass/fail requirements

For every ds4-py attempt:

- No `Stopped` event has `reason == "parser_error"`.
- Every parsed tool call has exactly one accounted outcome: executed, refused by workspace enforcement, or returned to the model as a retry.
- Bad-argument retries remain allowed and reported.
- Each model round has a known terminal reason and consistent terminal evidence. `tool_call_complete` requires a parsed call. `max_tokens` requires the configured response-budget boundary. Unexpected interrupts in ordinary task runs fail.
- A passed first attempt has no second attempt. A failed first attempt has at most one retry. The second attempt contains the first grader's feedback through the existing runner path. Grader feedback is not hidden or replaced.
- Missing or malformed lifecycle evidence fails the gate; it must not become a zero count.

The existing unit test that checks the exact retry prompt remains. The live report checks the recorded attempt/grader consistency rather than adding another model run.

#### Reported metric, not a gate

The report counts `max_tokens` exits per arm and per task. C uses the recorded response-budget boundary under the runner contract. It does not classify a round as capped only because two different arms produced similar text.

Cap-exit frequency differences are a rendering/sampling product decision, not by themselves a harness defect. They do not change Gate 2's pass/fail result.

#### Optional cap-parity check

`--cap-parity` enables an additional same-host check. For each arm's round-0 prompt, both reference drivers receive the same prompt, sync shape, and response budget. Each prompt must have the same cap outcome in both drivers: both reach the cap or both stop before it. Token equality is not required.

This subcheck is off by default. When explicitly enabled, its result is required for Gate 2 and its provenance is validated.

### Gate 3: throughput and KV reuse — replace wall-time ratio

The `1.5x` per-task wall-time rule is removed. Wall time stays in the report as a diagnostic.

#### Paired rates

The Gate 6 reference-driver runs also collect Gate 3 timing evidence. For each task, each driver processes the same retained C rounds with the same fresh-session and split-prefill shape. Aggregate tokens and elapsed time before calculating rates:

```text
prefill_tokens_per_second = total_prefilled_tokens / total_prefill_seconds
decode_tokens_per_second  = total_decode_steps / total_decode_seconds
relative_rate_difference  = abs(ds4py_rate - c_rate) / c_rate
```

Both relative rate differences must be at most `0.25`. A zero denominator or missing paired interval is invalid. Process startup and model loading are excluded.

The old evidence does not contain paired C and ds4-py prefill/decode durations, so `0.25` is a provisional initial tolerance, not a claim about a measured rate distribution. The old token-per-wall-second proxy has relative differences from `0.151` to `0.238` across `docs/reports/evidence/2026-11-01-bakeoff-8192-b28bb4d/bakeoff.json` and `docs/reports/evidence/2026-11-03-gate-repairs-0ec1288/gate-repairs-0ec1288/bakeoff.json`. A `0.25` tolerance covers that proxy while the new gate collects its first direct paired measurements. The report retains raw tokens and seconds so a later design can tighten this threshold from real rate data.

#### Cached-prefix fraction

For each common task, calculate the existing mean per-round cached-prefix fraction for each arm:

```text
C:      cached / prompt_length
ds4-py: plan.reused / (plan.reused + plan.to_prefill)
```

Zero denominators remain excluded. The absolute difference between arm means must be at most `0.15`.

The retained full bake-off has a largest observed absolute difference of `0.1365`; the gate-repair run has `0.1348`. The `0.15` threshold therefore comes from existing data. It accommodates different tool histories without making exact prompt lengths a requirement.

#### Extension-only KV reuse

This requirement stays unchanged for actual ds4-py task runs. Every later sync must use `extend`, except a round with an explicit `compaction` or `strategy` reason. Missing reasons and invalid plan kinds fail Gate 3.

### Gate 4: interrupt and resume — unchanged

One ds4-py run must be interrupted with Ctrl+C during prefill and resumed by the next prompt. The existing structured event driver and terminal evidence remain authoritative.

The criteria do not change. The repaired branch still needs a plain live re-run to certify Gate 4. A unit-test pass is not live certification.

### Gate 5: records and restore — hold KV construction history constant

Gate 5 must prove that restore does not corrupt state and that the stripped-save rebuild renders the identical prompt. It must not compare logits from an extended KV history with logits from a freshly rebuilt KV history. Those histories can produce different ordered logits even when both save and restore are correct.

The Phase A investigation established this distinction. Same-history continuation versus save/restore had identical ordered token IDs and maximum absolute raw-logit difference `0.0` in both ds4-py and C. Cross-history extend comparisons differed by about `0.30` to `1.13`, while stripped-save rebuild versus C differed by only `4.53e-08` and `4.89e-08`. Independent ds4-py and C saves of the same prefix also produced byte-identical native payloads. The old extend-versus-rebuild logit comparison was therefore a gate design error, not evidence of a ds4-py product defect.

The following requirements stay:

- Every ds4-py attempt leaves a complete save record, events, snapshots, and KV payload.
- Each case starts in a new ds4-py process and uses the public `/switch` command.
- Switch itself performs no prefill.
- Register the existing public `Session.add_logits_hook` after `/switch` and before the fixed follow-up. Capture only the first pre-sample logits call and store it outside the normal trace event format.
- Record each case's own terminal reason. Do not require the two continuations to have identical terminal reasons.
- Require all compared runs to come from the same save, fixed follow-up, instance, bake-off session, model, library, cache configuration, and sampling policy.

Gate 5 has four checks:

1. **5.1 plan.** Case 1 loads the original save and its fixed follow-up uses `Synced.plan.kind == "extend"` with only the suffix prefilled. Case 2 uses the payload-stripped copy and its follow-up uses `rebuild` with a real prefill.
2. **5.2 position.** Both cases have equal first-step positions. In each case, the follow-up `Synced.pos` and first pre-sample logits position equal the rendered follow-up prompt length. A plan label alone is not position evidence.
3. **5.3 restore fidelity.** Compare case 1 with a same-history reference, not with case 2. In one fresh ds4-py process, replay the original attempt's recorded sync and decode history, then prefill the fixed follow-up suffix and capture step-0 top-12 logits. The replay uses the retained original trace as the recorded history and validates its prompts, sampled IDs, sync positions, and token counts against the save's original-run events and saved token hash. It does not load the KV payload. Require equal positions, identical ordered top-12 token IDs, and maximum absolute raw-logit difference below `1e-5` between this reference and case 1. The full replay includes the original decode steps and therefore costs roughly the original attempt's generation time.
4. **5.4 rebuild equivalence.** Run the C diagnostic with `split_at = 0` on the exact rebuilt prompt from case 2 and compare its step-0 top-12 logits with case 2. Use the same model, context, cache, sampling, and response-budget provenance; only step 0 is captured and compared. Require equal positions, identical ordered top-12 token IDs, and maximum absolute raw-logit difference below `1e-5` through the existing driver-equivalence comparator.

Gate 5 passes only when file, switch, provenance, and all four checks pass. The extend-versus-rebuild logit comparison is removed.

### Gate 6: driver equivalence and self-replay — replace cross-host exact C replay

Gate 6 has two required parts.

#### Part A: same-host C/ds4-py logits equivalence

For every round of every retained C attempt trace:

1. Start a fresh native session in each reference driver. Reuse the loaded model across rounds; do not reload it for every round.
2. Apply the same two-stage sync: the recorded cached prefix, then the full prompt. Record all sync calls and progress chunks.
3. Use the recorded think mode, context, model, backend, and greedy sampling policy.
4. Compare at most the first `16` greedy steps: `steps = min(16, len(round.sampled))`.
5. At every compared step, require equal positions, identical ordered top-12 token IDs, and maximum absolute raw-logit difference below `1e-5`.

A zero-token round has no decode logits. Its sync and position evidence is still required and it counts as a sync-only check. The complete set must contain at least one decode step. No tool is executed.

The two drivers may run one after the other, but they must use the same instance, bake-off session, library, case manifest, and session-reset policy. Any mismatch is invalid.

#### Part B: ds4-py self-replay

For every recorded ds4-py attempt trace:

- Reproduce its complete committed-token sequence and terminal boundary exactly on the instance and in the bake-off session that recorded it.
- Preserve the original sequential session shape and recorded prefill boundaries. Do not use the fresh-per-round mode from Part A.
- Use an empty executable toolset and fail if tool execution is requested.
- Include short rounds, capped rounds, natural rounds, and tool-call rounds. Do not cap self-replay at 16 tokens.

A self-replay mismatch is a valid Gate 6 failure. A provenance mismatch is invalid, not a token mismatch.

#### Diagnostic retained

`bench/replay_c_traces.py` remains runnable for exact C replay investigations. It no longer computes or updates Gate 6. Its standalone command writes diagnostic evidence only. Cross-host exact C replay is retired as an acceptance criterion.

## 5. Report and result behavior

The report schema gains a `schema_version` and separate sections for:

- session provenance and trace provenance links;
- per-arm terminal reasons and `max_tokens` counts;
- paired prefill/decode tokens, seconds, and rates;
- cached-prefix fractions and extension exceptions;
- restore plan, position, and first-step logits evidence;
- cross-driver logits evidence;
- ds4-py self-replay evidence;
- optional exact-C-replay diagnostics.

`GateResult` gains `valid: bool = True`. An invalid gate has `passed == false` and `valid == false`. The compact report prints `INVALID`, `FAIL`, or `PASS`. Missing evidence is invalid. A completed comparison outside tolerance is a failure.

All six gates must be valid and passing for live certification. A targeted manifest can exercise the same gate logic, but its report must say that it is not full 12-task certification.

## 6. Existing test impact

### Tests that assert old criteria and must change

| File | Tests or behavior | Required change |
| --- | --- | --- |
| `tests/bench/test_runner.py` | `test_gate_three_enforces_time_and_only_unexplained_non_extensions` | Replace the wall-ratio assertion with paired-rate and cached-fraction cases. Keep unexplained non-extension rejection. |
| `tests/bench/test_runner.py` | `test_gates_one_to_five_pass_with_complete_evidence`, `test_report_json_has_attempt_feedback_and_table_has_six_deltas_not_logs`, `test_complete_runner_writes_gate_six_after_acceptance` | Update fixtures and expected evidence for terminal counts, rates, provenance, self-replay, and the new Gate 6 sections. |
| `tests/bench/test_bakeoff_replay.py` | `test_tool_round_replays_every_token_and_stops_without_tool_execution`, `test_token_mismatch_fails_with_task_round_token_index_and_round_type` | Move exact complete-token expectations to ds4-py self-replay. Cross-driver tests assert top-k/logits instead. |
| `tests/bench/test_bakeoff_replay.py` | `test_all_twelve_c_runs_replay_and_tool_coverage_is_required`, `test_full_set_without_a_tool_round_fails_gate_six`, `test_gate_six_cannot_pass_without_tool_call_evidence`, `test_selection_includes_every_c_attempt_trace_and_requires_twelve_tasks` | Require every selected C round and nonzero decode coverage for Part A. Remove tool-call coverage as a Gate 6 condition. Full 12-task coverage belongs to certification, not the reusable selection function. |
| `tests/bench/test_bakeoff_replay.py` | `test_standalone_replay_updates_only_its_evidence_and_gate` | Require exact C replay to update diagnostic evidence only and leave Gate 6 unchanged. |
| `tests/bench/test_bakeoff_restore.py` | `test_restore_cases_compare_tokens_and_enforce_prefill_paths`, `test_restore_token_mismatch_fails_with_index`, `test_short_restore_sequences_must_share_terminal_reason` | Replace token/terminal equality with plan, position, and first-step logits checks. Add ranking, tolerance, and provenance failures. |
| `tests/bench/test_metrics.py` | `test_event_metrics_cover_six_comparison_values_and_tool_accounting` and summary fixtures | Add terminal reasons, cap counts, cached fractions, and timing fields. Preserve tool accounting. |
| `tests/bench/test_report_commands.py` | Report deserialization and standalone replay command tests | Read the new schema and verify diagnostic-only replay output. |
| `tests/live/test_bakeoff.py` | `len(written["replay"]) > 0` and live configuration | Assert both new Gate 6 evidence parts, provenance, and the C driver requirement. Keep the explicit opt-in guard. |
| `tests/live/test_replay_parity.py` | Exact C token parity as a live acceptance test | Retire this acceptance test. Use `bench/replay_c_traces.py` as the manual diagnostic and add a guarded live driver-equivalence entry point. |

### Tests whose behavior stays

| File | Behavior that remains covered |
| --- | --- |
| `tests/bench/test_runner.py` | Gate 1 slack; Gate 2 parser/tool accounting; Gate 4; required saves; equivalent model/sampling/task parameters; sequential arms; exact grader-feedback retry prompt. Fixtures may gain required fields without weakening these assertions. |
| `tests/bench/test_bakeoff_replay.py` | Malformed C records become evidence failures; cached prefill boundaries are preserved; every selected C attempt is included; replay never executes tools. |
| `tests/bench/test_bakeoff_restore.py` | Save validation, stripped copies, no switch prefill, real case-2 prefill, non-interactive prompt composition, save-in-`finally`, disposable workspaces, and interruption behavior. |
| `tests/bench/test_metrics.py` | Canonical tool outcomes, retry semantics, non-extension exceptions, zero-denominator handling, malformed-event rejection, and C cached-prefix validation. |
| `tests/bench/test_interruption_driver.py` | All Gate 4 lifecycle and PTY cases. |
| `tests/bench/test_targeted.py` | Targeted task execution, repetition, and grader behavior. |
| `tests/bench/test_workspaces.py` | Workspace isolation and hidden-grader behavior. |
| `tests/engine/test_replay.py`, `tests/agent/test_prefill_trace.py` | Existing replay parsing and trace token records. Provenance sidecars do not change these formats. |
| `tests/abi/test_live_checks.py` | Native restore and terminal checks unrelated to bake-off report criteria. |

New automated tests belong to pure comparison, evidence-validation, deserialization, and report logic, plus fake-engine self-replay behavior. The live GPU gates are verified by running them. No new test will assert documentation text, CLI YAML/configuration literals, or C source structure.

## 7. What does not change

- No prompt rendering, system prompt, task preamble, tool result, or grader changes.
- No sampling-policy or production decode changes.
- No native `ds4` source or `libds4` changes.
- No changes to the existing trace prompt, token, event, or terminal records. Add provenance and diagnostic sidecars only.
- No changes to Gate 1's pass-count rule, Gate 4's interruption rule, or the extension-only KV rule.
- No deletion of the exact C replay diagnostic.
- No production `src/` change is planned. The work is in `bench/`, `tests/`, and documentation.
- No instance creation, start, stop, or destruction is authorized by this design.

## 8. Phase 2 verification and stop conditions

After approval, implement with focused failing tests before each comparison/report change. Then run:

```bash
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
uv run --frozen --extra agent pytest -m "not build and not live" -q
git diff --check
```

Compile the C diagnostic with `-Werror` on the live host. Run a targeted same-host check first. If it passes, run the full 12-task bake-off on the same instance. The user must approve and provision the instance. A portable-suite pass is not live certification.

## 9. Open questions and evidence handling

1. **Gate 3 initial tolerances:** approve the provisional `0.25` relative-rate tolerance and the evidence-backed `0.15` cached-prefix tolerance. The rate threshold should be reviewed after the first direct paired-rate report; it should not be tightened without that data.
2. **Source evidence:** the `FINDINGS.md` file and raw probe evidence remain uncommitted in `bakeoff-workspace-isolation`. The user should decide when to commit or archive them. Phase 1 deliberately leaves them untouched.

Phase 1 ends with the specification and plan commit. Phase 2 must not start until the user approves this design and plan.
