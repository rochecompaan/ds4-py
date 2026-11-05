# ds4-agent bake-off

This directory contains the 12-task, two-agent bakeoff comparison and its six gates.
The normal test suite uses fakes. Only the explicit commands below load the live
model or run the agents.

See [Workspace isolation and hidden grading](isolation.md) for details on task
isolation and grader visibility. See [Targeted gate-repair run](gate-repairs.md)
for the prepared four-task GPU procedure and standalone C replay diagnostic.

## Fixed inputs and machine requirements

- Model: the same absolute `ds4f-q2` GGUF for C, ds4-py, restore checks, and C
  replay. The Q2 artifact is approximately **81 GB**. Use a machine with at
  least **96 GB RAM**, plus enough free memory for the engine and benchmark.
- Backend: use the same backend for every phase. The examples use `cpu`; set
  `DS4_BAKEOFF_BACKEND` consistently if you select another supported backend.
- Benchmark checkout: `https://github.com/Aider-AI/polyglot-benchmark.git` at
  commit `7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f`.
- Agent selection: 10 deterministic Exercism tasks with selection seed
  `20261021`, plus the two pinned repository tasks in `bench/bakeoff.yaml`.
- Sampling: temperature 0, seed 1, context 100000, at most two attempts per
  task. Both agents must use the same model, backend, prompt, and sampling.
- Response budget: 8192 tokens per model response by default for both agents.
  Thinking consumes this budget. Use `--tokens` to set a different shared limit;
  a larger explicit value may still be necessary for thinking-heavy tasks.
- Run agents and replay sequentially. Do not run another agent, model, replay,
  or memory-heavy benchmark process at the same time.

Prepare the local paths:

```bash
export DS4_TEST_MODEL=/absolute/path/ds4f-q2.gguf
export DS4_AGENT=/absolute/path/ds4-agent
export POLYGLOT_BENCHMARK=/absolute/path/polyglot-benchmark
export DS4_BAKEOFF_BACKEND=cpu
export DS4_BAKEOFF_INSTANCE_ID=your-instance-id
export DS4_C_DRIVER=/absolute/path/c-replay-diagnostic
```

## Prepare

This checks the pinned checkout, task selection, and patches without running
the bake-off:

```bash
uv run --extra agent python bench/prepare_bakeoff.py \
  --manifest bench/bakeoff.yaml \
  --benchmark "$POLYGLOT_BENCHMARK" \
  --check-only
```

## CUDA host-registration opt-out

For the ds4 `5379c1c` CUDA bake-off and sanity wrappers, apply the pinned native
patch before building `ds4-agent`, `libds4.so`, and the C diagnostic. This opts
out of zero-copy host registration and selects the existing device-cache path:

```bash
git -C /absolute/path/to/ds4 apply --check \
  "$PWD/bench/patches/cuda-no-copy-host-register-opt-out.patch"
git -C /absolute/path/to/ds4 apply --whitespace=error \
  "$PWD/bench/patches/cuda-no-copy-host-register-opt-out.patch"

export DS4_CUDA_Q8_F16_CACHE_MB=5120
export DS4_CUDA_DISABLE_HOST_REGISTER=1
export DS4_BAKEOFF_NATIVE_PATCHES=bench/patches/cuda-no-copy-host-register-opt-out.patch
```

Use the same three exports in the bake-off and sanity launch shells. The patch
manifest is a colon-separated list of repo-relative paths. Each sidecar records
its patch names and SHA-256 hashes and the raw host-register environment value.
A paired comparison is invalid if either value differs. Do not set the opt-out
without applying the patch; do not claim the patch in an unpatched run.

## Run all six gates

`bench/results` must not already exist. The runner refuses to replace an old
report. The runner generates and prints one `bakeoff_session_id`. Tasks run
first, then every attempt restore, the interruption/resume check, sequential C
and ds4-py reference drivers, and finally ds4-py self-replay. No later phase
starts while an earlier engine is open. Every raw trace and sidecar inherits the
same instance, model, library, revision, and sampling provenance.

```bash
uv run --extra agent python bench/run_bakeoff.py \
  --manifest bench/bakeoff.yaml \
  --benchmark "$POLYGLOT_BENCHMARK" \
  --repo-root "$PWD" \
  --results bench/results \
  --model "$DS4_TEST_MODEL" \
  --agent "$DS4_AGENT" \
  --backend "$DS4_BAKEOFF_BACKEND" \
  --instance-id "$DS4_BAKEOFF_INSTANCE_ID" \
  --c-driver "$DS4_C_DRIVER"
```

A 12-task manifest writes `certification: "full-12-task"`. A targeted manifest
uses the same gate logic but writes `certification: "targeted"`; it is not full
M3 acceptance. Add `--cap-parity` only when equal `max_tokens` outcomes per task
are an explicit extra requirement.

The command exits nonzero if any gate fails. Outputs:

- `bench/results/bakeoff.json`: schema version 2, certification and session
  identity, per-task metrics, restore evidence, paired rates, provenance,
  reference-driver equivalence, ds4-py self-replay, and gates 1–6;
- `bench/results/report.txt`: compact gate and metric tables;
- `bench/results/`: staged workspaces, per-attempt logs, traces and provenance
  sidecars, saves, restore traces, reference NDJSON, and acceptance logs.
  Exact C replay is absent unless it is run separately as a diagnostic.

## Run selected tasks three times

Use targeted runs to check a fix before paying for another full bake-off. The
command runs only ds4-py, with one attempt per repetition and the explicit
8192-token and 100-request budgets. Each repetition gets a fresh workspace and
result root. The results directory must not already exist.

```bash
uv run --extra agent python bench/run_targeted.py \
  --task scale-generator \
  --task repo-ctx-command \
  --repetitions 3 \
  --results bench/targeted-results \
  --model "$DS4_TEST_MODEL" \
  --backend "$DS4_BAKEOFF_BACKEND"
```

`targeted.json` retains every result, including failures. The command exits
nonzero for a grader failure or a recorded attempt error. Do not start another
full bake-off until each selected task passes all three repetitions.

## Recheck restores

This starts fresh processes for both restore paths of every ds4-py attempt. For
a schema-2 report it inherits the original session, instance, model, and library
identity from the report's provenance section. It writes
`bench/results/restore-check.json` and does not replace the main report:

```bash
uv run --extra agent python bench/restore_check.py check \
  --report bench/results/bakeoff.json \
  --model "$DS4_TEST_MODEL" \
  --backend "$DS4_BAKEOFF_BACKEND" \
  --ctx 100000
```

Case 1 loads KV and extends on follow-up. Case 2 uses a payload-stripped copy,
renders fresh, and performs a full rebuild prefill on follow-up. Gate 5 checks
the restore plans, prompt position, save/session provenance, and the first
pre-sample top-12 logits. Later tokens and terminal reasons stay in the report
as diagnostics; they are not required to match.

## Diagnostic: exact C replay

This manual check reads the original C `--trace` files directly. It does not
translate traces or replay ds4-py against itself. It loads the GGUF once,
replays every round from every C attempt at temperature 0, and compares complete
committed IDs and terminal boundaries. Tool-call parsing never executes a tool.
It is a diagnostic for historical mismatches, not an acceptance gate.

```bash
uv run --extra agent python bench/replay_c_traces.py \
  --report bench/results/bakeoff.json \
  --model "$DS4_TEST_MODEL" \
  --backend "$DS4_BAKEOFF_BACKEND"
```

This updates only `replay_diagnostic` in `bench/results/bakeoff.json`; it does
not change any gate. It fails if a trace is malformed, a token differs, or a
round does not reproduce its natural/recorded terminal boundary.

## Replay reference drivers

The C and ds4-py diagnostics share a TSV case manifest and an NDJSON evidence
contract. They replay raw token IDs without executing tools. The main runner
invokes them for every selected C attempt, writes both provenance sidecars,
and uses the same evidence for Gate 3 rates and Gate 6 Part A. Each driver loads
one engine per manifest. C creates a fresh native session per case; ds4-py uses
`ResetJob` between cases. Run the drivers sequentially and keep native logs on
stderr. Stdout contains only NDJSON evidence.

Build the C driver against the same native source and library used by ds4-py:

```bash
export DS4_SRC=/absolute/path/to/ds4
export DS4PY_HOME=/absolute/path/to/native
export DIAG="$PWD/bench/results/diagnostics"
mkdir -p "$DIAG"
cc -O2 -std=c11 -Wall -Wextra -Werror -I"$DS4_SRC" \
  bench/c_replay_diagnostic.c \
  -L"$DS4PY_HOME/lib" -Wl,-rpath,"$DS4PY_HOME/lib" -lds4 \
  -o "$DIAG/c-replay-diagnostic"
```

The original single-case CLI remains available for the retained logits probes.
The token file contains prompt IDs followed by sampled IDs. Set
`PROMPT_TOKENS` and `SPLIT_AT` to the values recorded for the case:

```bash
"$DIAG/c-replay-diagnostic" \
  --model "$DS4_TEST_MODEL" \
  --tokens "$DIAG/case.tokens" \
  --prompt-tokens "$PROMPT_TOKENS" \
  --split-at "$SPLIT_AT" \
  --steps 16 \
  --think high \
  --backend "$DS4_BAKEOFF_BACKEND" \
  > "$DIAG/single-case.ndjson"
```

For a full retained C trace, export one manifest and one token file per round:

```bash
export TRACE=/absolute/path/to/c.trace
export TASK_ID=scale-generator
export ATTEMPT=1
export CASE_DIR="$DIAG/cases"
uv run --frozen --extra agent python - <<'PY'
import os
from pathlib import Path

from bench.driver_cases import export_trace_cases
from ds4py.engine.replay import read_replay

manifest = export_trace_cases(
    read_replay(Path(os.environ["TRACE"])),
    task_id=os.environ["TASK_ID"],
    attempt=int(os.environ["ATTEMPT"]),
    destination=Path(os.environ["CASE_DIR"]),
)
print(manifest)
PY
export CASES="$CASE_DIR/cases.tsv"
```

The manifest limits each round to `min(16, len(round.sampled))` steps. Its token
files contain the prompt followed by only those recorded sampled IDs. Run both
drivers with the same manifest, model, backend, context, and source trace:

```bash
"$DIAG/c-replay-diagnostic" \
  --model "$DS4_TEST_MODEL" \
  --cases "$CASES" \
  --source-trace "$TRACE" \
  --backend "$DS4_BAKEOFF_BACKEND" \
  --ctx 100000 \
  > "$DIAG/c-rounds.ndjson"

uv run --frozen --extra agent python bench/ds4py_replay_diagnostic.py \
  --model "$DS4_TEST_MODEL" \
  --cases "$CASES" \
  --source-trace "$TRACE" \
  --backend "$DS4_BAKEOFF_BACKEND" \
  --ctx 100000 \
  --lib "$DS4PY_HOME/lib/libds4.so" \
  > "$DIAG/ds4py-rounds.ndjson"
```

Each manifest round emits `round_start`, split-prefill `sync` and `prefill`
records, pre-sample top-12 `step` records, and `round_end`. Zero-token rounds
emit sync evidence and no fabricated step. Use
`bench.driver_cases.read_driver_ndjson` to validate either output; incomplete or
inconsistent streams are rejected.

## Live pytest entry point

The test below is another explicit way to invoke the complete runner. It skips
unless its opt-in flag and all five inputs are present. `-m live` alone cannot
start it.

```bash
export DS4_RUN_BAKEOFF=1
uv run --extra agent pytest tests/live/test_bakeoff.py -m live -vv -s
```

The test reads the final report and requires all six gates to be valid and pass.

For one retained C trace, the following runs only reference-driver Part A and
writes fresh provenance. It skips unless its separate flag and all four inputs
are present:

```bash
export DS4_RUN_DRIVER_EQUIVALENCE=1
export DS4_REPLAY_TRACE=/absolute/path/to/c.trace
uv run --frozen --extra agent pytest tests/live/test_reference_driver.py -m live -vv -s
```

Set `DS4_BAKEOFF_DS4_REVISION` and `DS4_BAKEOFF_DS4PY_REVISION` when the
reference binary is outside its source checkout. Unknown identity is invalid
provenance. Full 12-task certification still requires the complete runner.
To recheck both reference equivalence and ds4-py self-replay, rerun the targeted
or full runner with a fresh `--results` root. That command runs Part A and Part B
in order under one new session ID; it does not mix evidence across sessions.

## Non-live checks

```bash
uv run --extra agent pytest tests/bench -m "not live and not build" -q
env -u DS4_RUN_BAKEOFF -u DS4_RUN_DRIVER_EQUIVALENCE \
  -u DS4_TEST_MODEL -u DS4_AGENT -u POLYGLOT_BENCHMARK \
  uv run --extra agent pytest tests/live/test_bakeoff.py tests/live/test_reference_driver.py -q
```

A guarded skip confirms only that the live suite did not start. It is not live
model evidence. Until the full run has passed on the required model and
machine, the bakeoff remains **not live-verified**, even when every non-live test passes.
