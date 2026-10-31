# DS4 bake-off failure diagnosis: 2,048 tokens per response

Date: 2026-10-30

## Executive summary

The 2,048-token run improved ds4-py from 1/12 to 8/12 passing task graders. C improved from 10/12 to 12/12. All six acceptance gates still failed.

The four remaining ds4-py task failures have one immediate cause: the model exhausts the per-response token limit before emitting an edit. All eight attempts across these tasks ended their last round at exactly 2,048 sampled tokens. Seven ended in thinking with `max_tokens`; the eighth truncated a tool-call marker and ended with `parser_error`. None wrote a file or ran a test. Their grader failures therefore describe the untouched starter code, not a failed model-written implementation.

Independent failures remain in terminal-status handling, harness tool accounting, C trace encoding, restore setup, startup interruption, and task-level performance. Increasing the token limit again without fixing these issues would leave the run invalid.

## Local evidence

The remote instance has been destroyed. The analysis uses verified local copies:

- 512-token run: `/home/roche/projects/satyrn/ds4-py-results/tokens-512`
- 2,048-token run: `/home/roche/projects/satyrn/ds4-py-results/tokens-2048`
- SHA-256 manifests: `/home/roche/projects/satyrn/ds4-py-results/tokens-*.local.sha256`
- Prior report: `docs/reports/2026-10-30-bakeoff-512.md`

Before the instance was destroyed, 1,020 retained result files from the 512 run and 894 from the 2,048 run matched the remote files by relative path and SHA-256. Runner logs and scripts are under each run's `_runner` directory. Large session KV caches, staged repositories, and duplicate workspaces were pruned locally; transcripts, reports, scores, traces, errors, save metadata, and restore evidence remain.

## Results and terminal health

| Measure | 512 tokens | 2,048 tokens |
|---|---:|---:|
| C graders passed | 10/12 | 12/12 |
| ds4-py graders passed | 1/12 | 8/12 |
| ds4-py attempts | 24 | 18 |
| ds4-py `max_tokens` terminals | 20 | 13 |
| ds4-py `parser_error` terminals | 3 | 1 |
| ds4-py clean `stop_token` terminals | 0 | 4 |

A passing grader does not imply a clean agent run. In the 2,048-token run, 13 attempts exited 1. Five tasks passed their graders despite at least one nonzero attempt: `affine-cipher`, `grep`, `go-counting`, `paasio`, and `repo-ctx-command`. The last of these passed despite both attempts exiting 1. Gate 2 correctly rejects this lifecycle evidence even where Gate 1 records a test pass.

## Remaining task failures

| Task | Attempt 1 terminal | Attempt 2 terminal | Last-round sampled tokens | Edits or tests |
|---|---|---|---:|---|
| `scale-generator` | `max_tokens` | `max_tokens` | 2,048 / 2,048 | None |
| `zipper` | `max_tokens` | `parser_error` | 2,048 / 2,048 | None |
| `tree-building` | `max_tokens` | `max_tokens` | 2,048 / 2,048 | None |
| `repo-glm-close` | `max_tokens` | `max_tokens` | 2,048 / 2,048 | None |

The attempts performed reads, directory listings, and searches, then spent the final budget on internal reasoning. Representative evidence:

- `tokens-2048/scale-generator/ds4py/attempt-2/trace.jsonl`: final `sampled` array has 2,048 entries.
- `tokens-2048/scale-generator/ds4py/attempt-2/home/sessions/457a10ea8abb470aa18d2dd86320537e/events.jsonl`: final thinking part ends mid-scale; `Stopped` reports `max_tokens`; `run_end` reports an error.
- `tokens-2048/zipper/ds4py/attempt-2/home/sessions/c9bd386108d9446594416626463ebad6/events.jsonl`: the model reaches `Let's write code`, emits only an incomplete DSML tool-call opening, enters `ToolSyntax: ERROR`, and stops with `parser_error`.
- Equivalent terminal evidence exists in both attempts for `tree-building` and `repo-glm-close`.

The grader symptoms—unimplemented `Scale` methods, `Zipper.from_tree` returning `None`, existing tree-validation failures, and existing GLM parser-state failures—are baseline symptoms. There is no ds4-py patch for these four tasks to review.

### Required response-lifecycle fixes

1. Give the next diagnostic run a substantially larger response budget, or bound/disable thinking separately if the agent supports it. A targeted 8,192-token probe is a reasonable next experiment, not a proven production setting.
2. Treat truncation during a tool call as a failed attempt. `zipper` attempt 2 records `run_end: success` after `parser_error`; this can silently accept incomplete output.
3. Keep `max_tokens` and `parser_error` visible in task status even if a grader passes after an earlier edit.
4. Re-run the four failed tasks before another full bake-off. Confirm that each writes code, runs tests, and reaches a clean terminal state.

## Acceptance-gate diagnosis

### Gate 1: task comparison — FAIL

C passed 12/12 and ds4-py passed 8/12. The immediate blocker for the four remaining tasks is response truncation, as detailed above. The current evidence does not show four independent algorithmic failures by ds4-py.

### Gate 2: events and tool accounting — FAIL

The harness reports zero executed tool outcomes for every ds4-py task. The raw session events disprove that measurement:

- All 250 `FunctionToolCallEvent` records across the 18 attempts have matching `FunctionToolResultEvent` records.
- The per-attempt call/result counts match the harness's parsed-call counts exactly.
- The failed four tasks used read-only tools successfully; other tasks successfully edited files and ran commands.

This is a harness reader/schema mismatch, not evidence that ds4-py tools did not execute. Update accounting to parse the JSON event schema and pair results by tool-call ID, or use the step-level `tool_effects.jsonl` started/completed records. Do not infer execution from a missing `part.outcome` value.

Gate 2 also exposes the real lifecycle defects: 13 nonzero attempt exits and one `parser_error` reported as a successful run.

### Gate 3: timing and reuse — FAIL

Five tasks exceed 1.5 times C's wall time:

| Task | ds4-py tokens | C tokens | Token ratio | Wall ratio |
|---|---:|---:|---:|---:|
| `zipper` | 4,682 | 2,231 | 2.10x | 2.14x |
| `grep` | 4,210 | 1,472 | 2.86x | 2.89x |
| `dot-dsl` | 6,252 | 2,695 | 2.32x | 2.08x |
| `paasio` | 10,568 | 2,151 | 4.91x | 2.11x |
| `variable-length-quantity` | 9,412 | 2,040 | 4.61x | 3.74x |

The dominant task-level pattern is much greater output volume and more model rounds, including long thinking. `variable-length-quantity` remains 3.74 times slower despite one clean attempt, so retries alone do not explain the gap. These results do not isolate a slower engine kernel from agent behavior. Fix truncation first, then compare clean attempts and run controlled prefill/decode benchmarks if the gap remains.

`scale-generator` and `go-counting` also have incomplete C timing/reuse metrics because their C traces fail strict UTF-8 decoding. That is the same root cause as Gate 6.

### Gate 4: interruption — FAIL

The PTY driver sends Ctrl-C after seeing startup output containing `prefill_cap=4096`. The process is still inside `open_runtime`/`open_engine`, waiting for engine-worker readiness. SIGINT raises `KeyboardInterrupt` at `threading.wait`, the process exits, and the driver never observes the expected `interrupted` terminal event.

The immediate failure is startup interruption handling and probe timing, not the preceding CUDA host-registration warning. The current probe does not exercise generation-time cancellation. If the contract requires graceful startup cancellation, handle SIGINT during engine startup and emit the expected terminal state. If Gate 4 is meant to test generation cancellation, wait for an actual generation marker before sending Ctrl-C and test startup interruption separately.

### Gate 5: restore and continuation — FAIL

All 18 restore case-1 checks exit 1 before continuation. Every log has the same error: `Skills` needs a workspace for the saved `local:` workspace. The restore path must attach a matching `LocalWorkspace` capability or pass an explicit workspace when reconstructing the agent. Case 2 is marked `not run` in every entry, so the run provides no evidence about the second restore case.

Save metadata and restored session events exist. The failure occurs while reconstructing runtime capabilities, before a continued model response; it is not evidence of missing KV state or corrupt save payloads.

Representative evidence: `tokens-2048/restores/scale-generator/attempt-2/restore/case-1.log`.

### Gate 6: C trace replay — FAIL

Replay selection fails before selecting a tool-call round because it reads a C trace with strict UTF-8 decoding. The C trace writer places each token's raw bytes directly in a text field. A multi-byte character can be split across token records, leaving invalid UTF-8 in the file:

- `tokens-2048/scale-generator/c/attempt-1/trace.txt`, byte 161671: `e2 99` is logged in one token and `af` in the next.
- `tokens-2048/go-counting/c/attempt-1/trace.txt`, byte 157797: `e1 bb` is logged in one token and `9d` in the next.

This also explains the two C metric failures in Gate 3. Write token text with byte-safe escaping, or read the trace as bytes and parse the existing `hex=` field. For replay, use the hex bytes rather than lossy replacement decoding so the reconstructed token content remains exact.

## Recommended order of work

1. Fix response-budget handling and nonzero status for `max_tokens`/`parser_error`.
2. Fix ds4-py JSON event accounting and C byte-trace parsing so the next run produces valid diagnostics.
3. Fix restore workspace reconstruction and run both restore cases as targeted probes.
4. Clarify and repair the startup versus generation interruption probe.
5. Run the four failed tasks with a larger diagnostic response budget. Require edits, tests, and clean terminals.
6. Run the full bake-off only after those probes pass. Reassess wall time using clean attempts; profile engine throughput separately if needed.

No source changes were made during this diagnosis.
