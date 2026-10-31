# DS4 bake-off failure diagnosis: 8,192 tokens per response

Date: 2026-10-31

## Executive summary

The 8,192-token run improved ds4-py from 8/12 to 10/12 passing task graders; C remained at 12/12. ds4-py needed 14 attempts rather than 18. All six acceptance gates still failed.

The larger budget fixed the grader results for `zipper`, `tree-building`, and `repo-glm-close`, but it did not produce a clean run. Eleven of 14 ds4-py attempts exited 1: eight ended a response at `max_tokens`, one ended with `parser_error`, and two hit Pydantic AI's default `request_limit=50`. Seven tasks passed their graders despite a nonzero attempt.

The two remaining grader failures still do not represent failed model-written implementations. `scale-generator` made no source edit before truncation. `repo-ctx-command` made no source or test edit in either attempt; its second attempt reached the request limit first. Its grader therefore still reports the baseline `unknown command: /ctx` error.

The previous diagnostic fixes worked in important ways:

- canonical JSON tool accounting now records 303 parsed calls, 298 executions, and five retries instead of reporting zero executions;
- `max_tokens` and `parser_error` now produce truthful nonzero statuses;
- all 228 C replay rounds are read and attempted rather than failing UTF-8 decoding;
- restore no longer fails with the earlier `Skills needs a workspace` error.

New or remaining blockers are the implicit request limit, invalid saves on that error path, an under-budgeted restore probe, an interruption trigger that still fires during startup, high model output volume, and systematic non-tool replay divergence. Increasing the response budget again without addressing these issues is not the recommended next step.

## Run configuration and retained evidence

The run used the runner's new default of 8,192 tokens for both agents, with no `--tokens` override.

| Setting | Value |
|---|---|
| ds4-py revision | `e73ec15584825d51799eb089eecf02f4c180cba0` |
| ds4 revision | `5379c1c9ac315f1540371b06a3db88491d4059ff` |
| Polyglot benchmark revision | `7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f` |
| Model | DeepSeek V4 Flash IQ2XXS, 86,720,111,488 bytes |
| Backend | CUDA |
| Context | 100,000 tokens |
| Temperature / seed | 0 / 1 |
| Response budget | 8,192 sampled tokens |

The Vast.ai instance was destroyed after collection. The local evidence is:

- extracted files: `/home/roche/projects/satyrn/ds4-py-results/tokens-8192/`
- main results: `tokens-8192/results/bakeoff.json` and `report.txt`
- runner configuration: `tokens-8192/_runner/`
- workspace changes: `tokens-8192/diffs/`
- compressed archive: `tokens-8192-evidence.tar.gz` (30 MB)
- checksum manifest: `tokens-8192.local.sha256`

All 375 retained files matched the remote files by relative path and SHA-256. The archive also passed gzip integrity checking. Large KV caches, snapshots, staged repositories, and duplicate workspaces were excluded; traces, session events, logs, save metadata, restore evidence, reports, and final workspace diffs remain.

## Results and changes from the 2,048-token run

| Measure | 2,048 tokens | 8,192 tokens |
|---|---:|---:|
| C graders passed | 12/12 | 12/12 |
| ds4-py graders passed | 8/12 | 10/12 |
| ds4-py attempts | 18 | 14 |
| ds4-py nonzero attempts | 13 | 11 |
| ds4-py `max_tokens` exits | 13 | 8 |
| ds4-py `parser_error` exits | 1 | 1 |
| ds4-py request-limit exits | 0 observed | 2 |
| Acceptance gates passed | 0/6 | 0/6 |

The improvement is real but incomplete. A passing grader still does not imply a successful agent run.

### ds4-py task summary

| Task | Grader | Attempts | Tokens | Wall seconds | Process outcome |
|---|---|---:|---:|---:|---|
| `scale-generator` | FAIL | 2 | 17,042 | 426.8 | `parser_error`, `max_tokens` |
| `affine-cipher` | PASS | 1 | 19,510 | 457.8 | `max_tokens` |
| `zipper` | PASS | 1 | 16,744 | 409.4 | request limit |
| `grep` | PASS | 1 | 13,984 | 336.5 | `max_tokens` |
| `go-counting` | PASS | 1 | 14,788 | 355.6 | `max_tokens` |
| `proverb` | PASS | 1 | 2,206 | 72.7 | clean |
| `tree-building` | PASS | 1 | 15,399 | 365.5 | `max_tokens` |
| `dot-dsl` | PASS | 1 | 10,344 | 256.2 | clean |
| `paasio` | PASS | 1 | 16,935 | 401.5 | `max_tokens` |
| `variable-length-quantity` | PASS | 1 | 14,652 | 350.6 | `max_tokens` |
| `repo-glm-close` | PASS | 1 | 15,449 | 394.1 | clean |
| `repo-ctx-command` | FAIL | 2 | 26,024 | 687.1 | `max_tokens`, request limit |

## Gate 1: task comparison — FAIL

C passed 12/12 tasks and ds4-py passed 10/12.

### `scale-generator`

Attempt 1 used all 8,192 tokens and truncated inside its first `write_file` content, producing `parser_error`; the edit did not execute. Attempt 2 used all 8,192 tokens while reasoning and ended with `max_tokens`. Neither attempt ran tests. The final source is still the starter, and there is no ds4-py source diff to review.

The C and ds4-py trajectories differ before the budget failure:

- C starts with its `bash` tool running `pwd && ls -la`, sees `.meta`, and reads `.meta/example.py` before writing and testing.
- ds4-py starts with `list_directory`, which intentionally omits hidden entries, and never reads `.meta/example.py`. Its second attempt eventually runs `ls -la` through the harness's `run_command` tool but still does not inspect the example.

This is evidence of different tool affordances, not a broken filesystem tool. ds4-py already has equivalent shell capability through `Shell`/`run_command`. Remove `list_directory` from ds4-py's exposed `FileSystem` tools so the model must use a command such as `ls -la` for directory discovery. This gives C and ds4-py the same way to discover hidden entries without adding prompt-specific instructions or changing the upstream harness. Update the M3 specification because it currently lists `list_directory` as a default tool. This change does not guarantee identical trajectories; three independent targeted runs must each confirm discovery, a source edit, tests, and a clean terminal.

### `repo-ctx-command`

Both graders fail with:

```text
ds4py.tui_commands.CommandError: unknown command: /ctx
```

The final ds4-py workspace has no source or test changes; its diff contains harness tool-output artifacts only. Attempt 1 reached `max_tokens`; attempt 2 reached the 50-request limit. The 59 parsed calls were therefore inspection and test activity rather than a failed implementation edit.

C modified `runtime.py`, `tui_commands.py`, and their tests and passed. The immediate ds4-py blocker is again exhaustion before implementation, not evidence that a model-written `/ctx` implementation is wrong. The move from a grader pass at 2,048 tokens to a failure at 8,192 also shows that increasing the budget changes trajectories unpredictably and is not a sufficient fix.

## Gate 2: lifecycle and tool accounting — FAIL

The accounting repair is visible in the report: ds4-py has 303 parsed calls, 298 executed outcomes, and five retries. `zipper`, `dot-dsl`, and `repo-ctx-command` account for the parsed/executed difference through retries. This replaces the 2,048 run's incorrect zero-execution measurements.

Lifecycle health remains unacceptable. The 11 nonzero attempts comprise:

| Error | Attempts | Tasks |
|---|---:|---|
| `max_tokens` | 8 | `scale-generator` attempt 2; `affine-cipher`; `grep`; `go-counting`; `tree-building`; `paasio`; `variable-length-quantity`; `repo-ctx-command` attempt 1 |
| `parser_error` | 1 | `scale-generator` attempt 1 |
| `request_limit=50` | 2 | `zipper`; `repo-ctx-command` attempt 2 |

Seven passing tasks have a nonzero attempt. The new terminal-error handling correctly prevents a late truncated response from being reported as success after an earlier edit or test.

The two request-limit errors have the message:

```text
The next request would exceed the request_limit of 50.
```

This is a separate budget from the 8,192-token response limit. It is currently implicit in the Pydantic AI harness rather than an explicit bake-off setting.

### Required lifecycle fixes

1. Set an explicit request limit for benchmark runs and report it alongside token, temperature, and seed settings. Keep a finite safety limit, but choose it from the intended task policy rather than the dependency default.
2. Preserve `request_limit` as a distinct terminal category in Gate 2 evidence; do not collapse it into an unexplained exit 1.
3. Bound reasoning and encourage earlier inspection, edit, and test cycles. A larger per-response cap alone does not bound the number of requests or total run time.
4. Keep the existing `max_tokens` and `parser_error` rejection. Do not convert these failures back into successful runs to make Gate 2 pass.

## Gate 3: timing and prefix reuse — FAIL

Nine tasks exceed 1.5 times C's wall time, up from five in the 2,048-token run.

| Task | C wall seconds | ds4-py wall seconds | Wall ratio | Token ratio |
|---|---:|---:|---:|---:|
| `affine-cipher` | 184.8 | 457.8 | 2.48x | 2.76x |
| `dot-dsl` | 51.2 | 256.2 | 5.01x | 7.45x |
| `grep` | 118.9 | 336.5 | 2.83x | 3.17x |
| `paasio` | 62.3 | 401.5 | 6.44x | 8.51x |
| `proverb` | 42.7 | 72.7 | 1.70x | 2.25x |
| `repo-ctx-command` | 302.2 | 687.1 | 2.27x | 2.54x |
| `repo-glm-close` | 103.9 | 394.1 | 3.79x | 4.64x |
| `variable-length-quantity` | 53.4 | 350.6 | 6.57x | 9.91x |
| `zipper` | 90.3 | 409.4 | 4.54x | 6.38x |

Across all 12 tasks, ds4-py generated 183,077 tokens in 4,513.9 seconds; C generated 72,768 tokens in 2,006.1 seconds. These totals include all attempts and are not clean engine benchmarks.

Output volume is the dominant observed pattern. Every task over the wall-time gate uses at least 2.25 times C's tokens. `repo-glm-close` has better ds4-py prefix reuse than C (0.946 versus 0.915) but is still 3.79 times slower while generating 4.64 times as many tokens. The evidence therefore does not isolate a slower Python engine kernel.

Do not raise the global budget again to address performance. First reduce unnecessary reasoning and repeated inspection, require clean attempts, and then profile prefill and decode throughput separately if a gap remains.

## Gate 4: interruption — FAIL

The probe still sends Ctrl-C on the first regular-expression match for `prefill|tokens/s`. In this run, `prefill_cap=4096` appears during model startup. Ctrl-C consequently raises `KeyboardInterrupt` while `open_runtime`/`open_engine` waits for worker readiness, before visible generation. The driver then exits while waiting for `idle`.

The later switch to `idle` for follow-up and exit did not fix the first trigger. Gate 4 still does not exercise its documented generation-cancellation contract.

Replace the first trigger with an actual post-submission generation signal, such as an explicit TUI running/streaming state or a traced token event. Ignore startup prefill-capacity messages. Then retain the `idle` checks for the follow-up and clean exit. Test startup cancellation separately if it is a supported contract; do not use it as a substitute for generation cancellation.

## Gate 5: save, restore, and continuation — FAIL

There are 14 saved ds4-py attempts. The old workspace reconstruction error is gone, but no restore case passes:

| Case-1 outcome | Count | Interpretation |
|---|---:|---|
| Save/snapshot mismatch | 2 | `zipper`; `repo-ctx-command` attempt 2 |
| Follow-up `max_tokens` | 10 | Restore and switch occur, but 256 tokens are exhausted |
| Follow-up `parser_error` | 2 | `scale-generator` attempt 1; `repo-ctx-command` attempt 1 |

Case 2 is `not run` for all 14 attempts, so the run supplies no passing second-case continuation evidence.

### Invalid saves on the request-limit path

Both request-limit failures leave the same inconsistency: `save.json` holds 100 keyed messages while the restore snapshot has 101. Restore rejects the save before the follow-up with:

```text
cannot restore: the save record holds 100 keyed messages, the snapshot has 101
```

Associate the save/rebuild path with the request-limit exception and make its persisted history atomic. Add an error-path regression that requires keyed messages and the canonical snapshot to describe the same history after a usage-limit failure. Do not relax restore validation to accept the mismatch.

### Restore probe budget

The other 12 restores reach the continuation, but `bench/restore_check.py` invokes the follow-up with `--tokens 256` while thinking remains enabled. The prompt is `Continue the previous task: state the next concrete action, then stop.` Ten responses hit `max_tokens` and two hit `parser_error` under that cap.

Use a dedicated, sufficient continuation budget or disable thinking for a deterministic restore probe. Keep the follow-up's clean terminal requirement, but do not treat a 256-token thinking probe as evidence that KV restoration itself failed. Run case 2 only after case 1 restores and terminates cleanly.

## Gate 6: C trace replay — FAIL

The byte-safe trace reader works: all 228 rounds are parsed and replayed, including the previously unreadable traces. The results expose a deeper parity failure:

| Replay result | Count |
|---|---:|
| Passed tool-call rounds | 62 |
| Failed non-tool rounds | 166 |
| Total | 228 |

All 62 tool-call rounds pass; all 166 non-tool rounds diverge. Twenty-six diverge at token 0, the median first mismatch is token 10, and the latest is token 193. One example is `scale-generator` attempt 1, round 0, where C emits token 411 (` an`) and ds4-py emits token 270 (` the`) at position 10.

This is no longer a UTF-8 or trace-selection failure. Replay submits the exact C prompt token IDs, uses temperature 0 and seed 1, and reuses one RNG per C attempt. The perfect tool/non-tool split strongly suggests a systemic sampling, session-state, or numerical-parity difference; tool parsing itself has positive evidence from the 62 exact matches. The current evidence does not distinguish those underlying causes.

Do not skip non-tool rounds or weaken token comparison. Capture a minimal failing round with both implementations on the same backend and native library and record:

1. the prompt hash, session position, prefill plan, and cache reuse;
2. the complete sampler inputs at each step: temperature, top-k, top-p, min-p, and mutable RNG state;
3. the first divergent logits and selected token;
4. the resulting KV/session state before the next token.

If logits match but selected tokens differ, fix sampler/RNG parity. If logits differ at token 0, investigate prefill, KV restoration, backend numerics, and native-library configuration. If they diverge only after several tokens, inspect per-token session evaluation and cache updates. Add the minimal failing round as a regression once the cause is fixed.

## Recommended order of work

1. **Make all budgets explicit and preserve save integrity.** Configure the benchmark request limit, classify it in reports, and fix the 100/101 message/snapshot failure on that path.
2. **Change tool discovery rather than raising the cap again.** Remove `list_directory` from ds4-py's exposed `FileSystem` tools so the model uses the existing `Shell`/`run_command` capability for full directory listings, including hidden files. Do not add prompt-specific discovery instructions or patch the upstream harness. Update the M3 specification to reflect the tool change.
3. **Re-run only `scale-generator` and `repo-ctx-command` first.** Run each failed task at least three times in a fresh process and workspace, without stopping after the first pass. Every run must produce a source edit, run the tests, and end cleanly. For `repo-ctx-command`, also require `/ctx` registration and its hidden grader to pass in every run. Treat three passes as a basic repeatability check, not statistical proof, before starting a full bake-off.
4. **Repair the acceptance probes.** Trigger Gate 4 on visible generation, and give Gate 5 a suitable continuation budget while retaining strict restore validation.
5. **Isolate replay parity with one non-tool round.** Use sampler/logit/session evidence to select the fix; preserve the successful tool-call replay coverage.
6. **Run the full bake-off only after targeted probes pass.** Reassess performance from clean attempts and separate output volume from engine throughput.

No production code was changed during this analysis; this report is documentation only.

## Live follow-up after the remediation fixes

The targeted runs and probes used ds4-py `f5abe799be725508c7fad4ad58af7ce5d190b442`, ds4 `5379c1c9ac315f1540371b06a3db88491d4059ff`, and Polyglot benchmark `7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f`. The model, CUDA backend, 100,000-token context, temperature 0, seed 1, 8,192-token response budget, and request limit 100 stayed fixed. The native library SHA-256 was `866920c49bc280098ae0f45cff43e287251b442616bfc638d747ba859fc3f830`.

The host used the targeted runner through `/workspace/resume_targeted.py`, which calls the same `run_task` implementation. The wrapper resumed the missing third scale run after correcting `PYTHONPATH=/workspace/ds4-py`; it did not rerun completed cases. The canonical local command remains:

```bash
python bench/run_targeted.py \
  --task scale-generator --task repo-ctx-command --repetitions 3 \
  --model /workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --benchmark /workspace/polyglot-benchmark \
  --repo-root /workspace/ds4-py \
  --results /workspace/results/targeted-8192-f5abe79
```

### Targeted results: 0/6 clean passes

| Task | Run | Grader | Agent outcome |
|---|---:|---|---|
| `scale-generator` | 1 | PASS | `max_tokens` |
| `scale-generator` | 2 | PASS | `max_tokens` |
| `scale-generator` | 3 | FAIL | Clean exit, but no edit and 0/17 tests passed |
| `repo-ctx-command` | 1 | PASS | `request_limit` after 100 requests |
| `repo-ctx-command` | 2 | PASS | `max_tokens` |
| `repo-ctx-command` | 3 | PASS | `max_tokens` |

A passing grader still does not make an attempt clean. The three repository runs passed the grader, but none terminated successfully. The third scale run shows the opposite case: a clean terminal state with no useful work.

### Acceptance probes: Gate 4 and all six restores failed

Gate 4 reached `streaming` and sent Ctrl-C, but generation continued through 7,032 `Token` events, three model steps, and two completed tool calls. No `run_end` or interruption event appeared before the PTY timeout. A separate fake-runtime PTY probe delivered Ctrl-C to the TUI handler, so key delivery works; the live cancellation path still needs diagnosis.

Restore case 2 ran only after a clean case 1. The failures were:

| Save | First failing check |
|---|---|
| `scale-generator` run 1 | Case 2 reached the context reserve after case 1 rebuilt 99,164 prompt tokens |
| `scale-generator` run 2 | Case 1 ended with `max_tokens` |
| `scale-generator` run 3 | Both cases terminated, but case 1 rebuilt instead of extending and continuation token 41 differed |
| `repo-ctx-command` run 1 | Case 1 rejected saved history with unprocessed tool calls |
| `repo-ctx-command` run 2 | Case 1 ended with `max_tokens` |
| `repo-ctx-command` run 3 | Both cases terminated, but case 1 rebuilt instead of extending and continuation token 68 differed |

These are real restore and continuation blockers, not the earlier 256-token probe-budget failure.

### Replay diagnosis

Two diagnostics used the same model and native library, with worker and direct-native calls compared independently.

1. A control replay used current ds4-py run 3, round 0. At context 8,192, both paths reproduced all 16 traced tokens. At context 256, a shortened 239-token prompt also produced identical worker and native tokens; this shortened case is only a cross-implementation control, not a comparison with the original trace.
2. The known historical failure used C `scale-generator` attempt 1, round 0 at its original 100,000-token context. Both paths diverged at token 10 with both full prefill and the traced 1,839/948 split prefill. C emitted token 411 (` an`); both replay paths emitted token 270 (` the`). At token 10, the current replay logits were 30.991417 for ` the` and 30.933065 for ` an`. Temperature was 0 and the RNG state was 1. Worker and native logits, selected tokens, positions, and RNG states matched at every captured step.

The replay therefore isolates the mismatch from the Python worker: current worker and direct-native behavior is identical. It does not yet justify an engine patch. The remaining question is why the historical C run selected ` an` when the current native replay prefers ` the`. A current C-side capture of the same prompt must record its logits and sampler inputs before choosing a production fix.

### Evidence and next gate

The concise local evidence is under `docs/reports/evidence/2026-10-31-targeted-8192/`: six targeted records, acceptance summaries and failure logs, and detailed worker/native replay comparisons. Large KV files, session events, and workspaces remain excluded.

Do not start the full bake-off. The prerequisites now are clean targeted termination, working live interruption, valid and equivalent restore continuations, and a C-side replay capture that identifies the historical mismatch.
