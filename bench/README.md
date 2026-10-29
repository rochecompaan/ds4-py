# ds4-agent bake-off

This directory contains the 12-task, two-agent bakeoff comparison and its six gates.
The normal test suite uses fakes. Only the explicit commands below load the live
model or run the agents.

See [Workspace isolation and hidden grading](isolation.md) for details on task
isolation and grader visibility.

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
- Run agents and replay sequentially. Do not run another agent, model, replay,
  or memory-heavy benchmark process at the same time.

Prepare the local paths:

```bash
export DS4_TEST_MODEL=/absolute/path/ds4f-q2.gguf
export DS4_AGENT=/absolute/path/ds4-agent
export POLYGLOT_BENCHMARK=/absolute/path/polyglot-benchmark
export DS4_BAKEOFF_BACKEND=cpu
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

## Run all six gates

`bench/results` must not already exist. The runner refuses to replace an old
report. Normal ds4-py attempts use `--yolo`; gate 4 uses a separate PTY approval
path. Tasks run first, then interruption/approval checks, every attempt restore,
and finally C replay. No later phase starts while an earlier engine is open.

```bash
uv run --extra agent python bench/run_bakeoff.py \
  --manifest bench/bakeoff.yaml \
  --benchmark "$POLYGLOT_BENCHMARK" \
  --repo-root "$PWD" \
  --results bench/results \
  --model "$DS4_TEST_MODEL" \
  --agent "$DS4_AGENT" \
  --backend "$DS4_BAKEOFF_BACKEND"
```

The command exits nonzero if any gate fails. Outputs:

- `bench/results/bakeoff.json`: per-task metrics, restore evidence, every C replay
  round, and gates 1–6;
- `bench/results/report.txt`: compact gate and metric tables;
- `bench/results/`: staged workspaces, per-attempt logs, traces, saves, restore
  traces, and acceptance logs.

## Recheck restores

This starts fresh processes for both restore paths of every ds4-py attempt. It
writes `bench/results/restore-check.json` and does not replace the main report:

```bash
uv run --extra agent python bench/restore_check.py check \
  --report bench/results/bakeoff.json \
  --model "$DS4_TEST_MODEL" \
  --backend "$DS4_BAKEOFF_BACKEND" \
  --ctx 100000
```

Case 1 loads KV and extends on follow-up. Case 2 uses a payload-stripped copy,
renders fresh, and performs a full rebuild prefill on follow-up. Both compare
up to the first 256 committed token IDs and the terminal boundary.

## Replay C traces again

Gate 6 reads the original C `--trace` files directly. It does not translate
traces or replay ds4-py against itself. It loads the GGUF once, replays every
round from every C attempt at temperature 0, and compares complete committed
IDs and terminal boundaries. Tool-call parsing never executes a tool.

```bash
uv run --extra agent python bench/replay_c_traces.py \
  --report bench/results/bakeoff.json \
  --model "$DS4_TEST_MODEL" \
  --backend "$DS4_BAKEOFF_BACKEND"
```

This updates only `replay` and `gate-6` in `bench/results/bakeoff.json`. It fails if
the 12-run C set has no tool-call round, a trace is malformed, a token differs,
or a round does not reproduce its natural/recorded terminal boundary.

## Live pytest entry point

The test below is another explicit way to invoke the complete runner. It skips
unless all four opt-in variables are present. `-m live` alone cannot start it.

```bash
export DS4_RUN_BAKEOFF=1
uv run --extra agent pytest tests/live/test_bakeoff.py -m live -vv -s
```

The test reads `bench/results/bakeoff.json` and requires all six gates to pass.

## Non-live checks

```bash
uv run --extra agent pytest tests/bench -m "not live and not build" -q
env -u DS4_RUN_BAKEOFF -u DS4_TEST_MODEL -u DS4_AGENT -u POLYGLOT_BENCHMARK \
  uv run --extra agent pytest tests/live/test_bakeoff.py -q
```

A guarded skip confirms only that the live suite did not start. It is not live
model evidence. Until the full run has passed on the required model and
machine, the bakeoff remains **not live-verified**, even when every non-live test passes.
