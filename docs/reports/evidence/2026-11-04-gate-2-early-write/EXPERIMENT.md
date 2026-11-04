# Gate 2 early-write task-preamble experiment

Date: 2026-11-04
Instance: `59078792`
Result: **negative; do not adopt the preamble as a Gate 2 repair**

## Question

Does a shared task-level instruction to write a best-effort implementation early prevent ds4-py's long canonical-recall rounds without changing the production system prompt?

Both arms received this exact text before the task instructions:

```text
After inspecting the task, relevant source files, and available documentation or examples, write a best-effort implementation. Do not spend extended reasoning trying to recall a canonical solution or reconstruct unavailable tests. Run focused local checks based on the documented requirements, then refine the implementation. Prefer concrete progress with tools over repeated deliberation.
```

The production system prompt and `src/ds4py/agent_prompt.py` were unchanged.

## Configuration

- ds4 native source: `5379c1c9ac315f1540371b06a3db88491d4059ff`
- ds4-py base revision: `e90a259f3cabad09272324c96bd6887ea3f6a713`, plus the retained task-preamble runner patch
- Benchmark: `7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f`
- Model SHA-256: `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB
- Driver: `580.173.02`
- Response budget: 8,192 tokens per round
- Request cap: 12
- Tasks: `scale-generator`, `affine-cipher`, `tree-building`, and `zipper`
- Three fresh one-attempt repetitions per task and arm; 24 live task attempts total
- No Gate 4 interruption, Gate 5 restore checks, or full bake-off

`summary.json` records every attempt and verifies the exact shared preamble in all 24 decoded input traces. The C trace's token text supplied the token-ID mapping used to check the ds4-py prompt trace.

## Results

| Arm | Task | Grader passes | Ended with `max_tokens` | Median wall time | Median reported tokens |
|---|---|---:|---:|---:|---:|
| C | scale-generator | 2/3 | 0/3 | 210.0 s | 8,402 |
| C | affine-cipher | 3/3 | 0/3 | 114.2 s | 4,289 |
| C | tree-building | 0/3 | 0/3 | 265.8 s | 10,456 |
| C | zipper | 0/3 | 0/3 | 154.3 s | 6,051 |
| ds4-py | scale-generator | 1/3 | 2/3 | 214.3 s | 8,440 |
| ds4-py | affine-cipher | 0/3 | 3/3 | 213.7 s | 8,427 |
| ds4-py | tree-building | 0/3 | 2/3 | 212.4 s | 8,446 |
| ds4-py | zipper | 0/3 | 2/3 | 210.3 s | 8,378 |

Overall:

- **C:** 5/12 passed; all 12 agent processes exited without a terminal error. Failures were incorrect implementations.
- **ds4-py:** 1/12 passed; **9/12 ended with `max_tokens`** and recorded `attempt 1 exited 1`.

The C arm did not pass `tree-building` or `zipper`, so these runs do not provide a successful same-prompt C timing baseline for those tasks. The table describes these runs; it does not establish a token-equivalence requirement.

## Trajectory evidence

The instruction was present, but the model did not reliably follow it. Representative capped ds4-py rounds still began:

- `scale-generator`: "Let me recall exact canonical."
- `affine-cipher`: "Let's recall canonical solution."
- `tree-building`: "Let me recall the canonical solution for the Python \"tree building\" exercise."
- `zipper`: "Let me recall the canonical Exercism Python zipper solution."

Those final 8,192-token rounds contained no tool call. Examples in the retained traces include repeated reconstruction of expected test values and long deliberation about canonical implementations. The first live `scale-generator` attempt wrote no implementation before the cap. The sole ds4-py pass was `scale-generator` run 2; it used ten rounds and passed without a terminal error.

## Setup-failure exclusion

The first ds4-py launch omitted the documented `DS4_LIB=/workspace/native/lib/libds4.so` setting. All 12 processes exited before inference, with zero tokens and no event stream, because they looked for `libds4.so` under each per-attempt home. Those records are retained separately as:

```text
raw/results/gate2-early-write-ds4py/
```

They are invalid behavioral evidence and are excluded from every result above. Native loading was then verified explicitly, and the live arm was run as:

```text
raw/results/gate2-early-write-ds4py-live/
```

The corrected run used the same model, task preamble, task set, and three-repetition design as the C arm.

## Decision and next step

Reject this preamble as a Gate 2 repair. It neither prevented the canonical-recall loops nor produced reliable passing implementations. Do not add it to the production system prompt or default task contract.

A further prompt-wording iteration is not justified by this evidence. A future experiment should instead test a mechanism that changes the transition from extended reasoning to concrete tool use, with a passing same-prompt C control for any task used to make timing claims. Gate 2 remains open.

## Evidence

- `summary.json`: per-attempt outcomes, terminal status, token counts, and aggregates
- `raw/results/gate2-early-write-c/`: C targeted results, traces, logs, graders, and workspaces
- `raw/results/gate2-early-write-ds4py-live/`: valid ds4-py targeted evidence
- `raw/results/gate2-early-write-ds4py/`: excluded pre-inference setup failures
- `raw/provenance.txt`: host, source, model, and artifact provenance
- `setup-host.sh`, `run-experiment.sh`, and `run-ds4py-rerun.sh`: retained commands
- External archive: `/home/roche/projects/satyrn/ds4-py-evidence/gate2-early-write-59078792-non-kv.tar.gz`
- Archive SHA-256: `abe4d6b106bf49d724b1a90515d0881530dad6b7e0f883623fa594dbd67ed60d`

The archive contains 1,003 entries and excludes `*.kv` files. Its gzip integrity and downloaded SHA-256 were verified before extraction. Instance `59078792` was stopped after evidence retrieval.
