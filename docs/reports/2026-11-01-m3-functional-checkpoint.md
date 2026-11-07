> Superseded by the [2026-11-06 M3 targeted certification](2026-11-06-m3-targeted-certification.md).

# M3 functional checkpoint: full bake-off after prompt alignment

**Status:** Functional implementation checkpoint. **M3 is not live-verified and is not complete.** Gate 1 passed; Gates 2–6 failed. Whole-branch review and integration are intentionally left to a separate session.

- **ds4-py revision:** `b28bb4d75a22200e59c5e7f6343db9565d2d6703`
- **Run date:** 2026-11-01
- **Result root:** `/workspace/results/bakeoff-8192-b28bb4d` on Vast.ai instance `58555396`

## Executive summary

- ds4-py passed **12/12 task graders**; C passed **11/12**.
- Both arms used 13 attempts. ds4-py's `tree-building` passed on its second attempt after its first ended at the response-token cap. C's `repo-glm-close` failed twice.
- ds4-py generated **104,318 tokens**, compared with **138,579** for C. These totals cover different tool histories and task attempts, so they are not a controlled efficiency comparison.
- The previous full run generated 183,077 ds4-py tokens. The reduction to 104,318 is about **43%**, across the combined remediation and prompt-alignment changes; this run does not isolate either change's contribution.
- ds4-py reported **no parser errors**. Replay passed every tool-call round but no non-tool round.
- The agent is functional at the task-completion level. Restore, interruption, lifecycle, timing, and exact non-tool replay still need work before M3 acceptance.

## Configuration and retained evidence

The run used the same model, native backend, 100,000-token context, temperature 0, seed 1, and **8,192-token response cap** for both arms. The model was `ds4f-q2.gguf` (DeepSeek V4 Flash Q2, 80.76 GiB resident weights on this host), running on one RTX PRO 6000 Blackwell with about 96 GiB VRAM. The harness was `pydantic-ai-harness==0.59.0`; C used pinned upstream ds4 commit `85e92bf`.

Concise, tracked evidence is in [`evidence/2026-11-01-bakeoff-8192-b28bb4d/`](evidence/2026-11-01-bakeoff-8192-b28bb4d/):

- `bakeoff.json`: complete task, gate, restore, and replay results.
- `config.txt`: exact run configuration.
- `failure-evidence-manifest.json`: paths, sizes, and SHA-256 hashes for the focused archive.
- `ARCHIVE.sha256`: archive checksum.

The focused local archive is:

`/home/roche/projects/satyrn/ds4-py-results/bakeoff-8192-b28bb4d-failure-evidence/bakeoff-8192-b28bb4d-failure-evidence.tar.gz`

SHA-256: `e500f151150bbe946fae69efd306cc30860da5ebeac0898ddc2c6aeee5b4acab`

The archive contains 127 files, 65,076,057 raw bytes, and 4,146,023 compressed bytes. All 127 extracted files passed local size and SHA-256 verification. It retains the runner's raw `report.txt` and `bakeoff.log`, failure-specific agent logs, C/Python traces, events, restore records, and named save snapshots. It excludes KV payloads, task workspaces, staged repositories, duplicate step histories, and `__pycache__`.

## Task results

| Task | C grader | C attempts | C tokens | C seconds | ds4-py grader | ds4-py attempts | ds4-py tokens | ds4-py seconds | ds4-py/C time |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| `scale-generator` | PASS | 1 | 7,959 | 211.3 | PASS | 1 | 12,753 | 320.8 | 1.52x |
| `affine-cipher` | PASS | 1 | 4,597 | 131.3 | PASS | 1 | 17,397 | 426.1 | 3.25x |
| `zipper` | PASS | 1 | 2,236 | 76.7 | PASS | 1 | 5,722 | 158.7 | 2.07x |
| `grep` | PASS | 1 | 4,515 | 121.8 | PASS | 1 | 2,574 | 81.8 | 0.67x |
| `go-counting` | PASS | 1 | 38,832 | 978.9 | PASS | 1 | 13,855 | 344.9 | 0.35x |
| `proverb` | PASS | 1 | 1,017 | 39.3 | PASS | 1 | 1,095 | 45.2 | 1.15x |
| `tree-building` | PASS | 1 | 5,094 | 135.0 | PASS | 2 | 13,299 | 344.6 | 2.55x |
| `dot-dsl` | PASS | 1 | 7,397 | 192.2 | PASS | 1 | 7,700 | 198.7 | 1.03x |
| `paasio` | PASS | 1 | 5,799 | 161.8 | PASS | 1 | 4,457 | 122.0 | 0.75x |
| `variable-length-quantity` | PASS | 1 | 1,216 | 45.0 | PASS | 1 | 1,678 | 59.4 | 1.32x |
| `repo-glm-close` | FAIL | 2 | 50,348 | 1,292.7 | PASS | 1 | 15,740 | 442.2 | 0.34x |
| `repo-ctx-command` | PASS | 1 | 9,569 | 277.2 | PASS | 1 | 8,048 | 233.2 | 0.84x |
| **Total** | **11/12** | **13** | **138,579** | **3,663.1** | **12/12** | **13** | **104,318** | **2,777.7** | — |

C's large `go-counting` and failed two-attempt `repo-glm-close` runs make aggregate wall-time comparisons misleading. Gate 3 therefore uses per-task limits.

## Gate 1: task comparison — PASS

ds4-py passed all 12 graders and met the requirement to remain within one task of C. Both arms used 13 attempts. This is strong evidence of functional task-solving, but it does not override the separate lifecycle and restore requirements.

## Gate 2: lifecycle and tool accounting — FAIL

`tree-building` ds4-py attempt 1 exited 1 with:

`ResponseTerminalError: model response ended with max_tokens`

Its grader failed, and the runner supplied that output to attempt 2, which passed with a clean terminal. The failed attempt used the full 8,192-token response cap. The follow-up must diagnose why this trajectory exhausted its budget; it must not convert a genuinely truncated response into a false success.

No parser-error stops were reported, and the retained task summaries account for parsed, executed, refused, and retried tool calls.

**Follow-up:** `TODO-b72c629c`.

## Gate 3: timing and prefix reuse — FAIL

Four ds4-py tasks exceeded 1.5 times C wall time:

| Task | C seconds | ds4-py seconds | Ratio | C tokens | ds4-py tokens |
|---|---:|---:|---:|---:|---:|
| `scale-generator` | 211.261 | 320.774 | 1.518x | 7,959 | 12,753 |
| `affine-cipher` | 131.269 | 426.122 | 3.246x | 4,597 | 17,397 |
| `tree-building` | 134.971 | 344.570 | 2.553x | 5,094 | 13,299 |
| `zipper` | 76.694 | 158.664 | 2.069x | 2,236 | 5,722 |

The gate detail identifies wall time; it does not establish how much came from token volume, decode throughput, or tool overhead. Profile those separately after the correctness fixes. Do not relax the per-task limit or use C's slow aggregate outliers to hide local regressions.

**Follow-up:** `TODO-e2ba04bb`.

## Gate 4: interruption — FAIL

The PTY driver timed out. Its log shows that the TUI continued rendering model output around the interruption attempt. This evidence alone does not distinguish a driver lifecycle-signal bug from a cancellation defect.

Use the retained `gate-4/interruption/interactive.log`, `trace.jsonl`, and session events to identify where the expected transition was lost. Acceptance requires Ctrl+C during actual generation, a defined interrupted/idle state, a successful follow-up, and clean exit.

**Follow-up:** `TODO-ec34b362`.

## Gate 5: save, restore, and continuation — FAIL

Thirteen ds4-py saves were checked:

- Twelve completed case-1 follow-ups used a **rebuild** plan instead of the required KV extension. Each diverged from its stripped-save case-2 continuation at a recorded token index.
- The twelve corresponding case-2 stripped-save checks passed, including fresh rendering.
- `tree-building` attempt 1 case 1 exited 1 at `max_tokens`, so case 2 was not run for that save.

The representative `scale-generator` mismatch occurred at token 41 (`case 1=16`, `case 2=295`). Its original save record, named snapshot, events, stripped save record, and both follow-up traces are retained.

Acceptance requires no prefill during case-1 switch, extension on its follow-up, fresh rendering for case 2, and matching committed tokens and terminal boundaries at temperature 0. The token-cap exit should be diagnosed with Gate 2 rather than hidden by the restore check.

**Follow-up:** `TODO-0d89a160`.

## Gate 6: C trace replay — FAIL

Only **91/287** replay checks passed, but the split is exact:

| Round type | Passed | Failed |
|---|---:|---:|
| Tool-call rounds | 91 | 0 |
| Non-tool rounds | 0 | 196 |

The retained C `scale-generator` attempt-1 trace contains both useful cases:

- Round 0, a non-tool round, diverged at token 10: C sampled token 411 (`b' an'`), while ds4-py sampled token 270 (`b' the'`).
- Round 1, a 137-token tool-call round, passed exactly.

This is strong evidence for the structured tool-call path and a precise starting point for ordinary-generation diagnosis. It does not by itself identify whether the remaining difference is in prompt construction, sampling configuration, or native decode state. Do not weaken the exact replay comparison or substitute ds4-py self-replay.

**Follow-up:** `TODO-24103f08`.

## Follow-up order and integration status

1. Diagnose the Gate 2 token-cap lifecycle failure.
2. Repair and prove Gate 4 interruption behavior.
3. Fix Gate 5 KV restore and stripped-rebuild agreement.
4. Isolate Gate 6 non-tool replay divergence.
5. Profile Gate 3 performance after correctness is stable.

The branch may be reviewed and integrated as a **functional implementation checkpoint**, not as an accepted M3 delivery. These follow-ups remain open, and M3 remains **not live-verified** until a full run passes all six gates.
