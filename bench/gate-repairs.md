# Targeted gate-repair run

**Status:** prepared locally; live GPU execution is pending.

This run checks the current fixes without paying for all 12 tasks. It compares C
and ds4-py on `scale-generator`, `affine-cipher`, `tree-building`, and `zipper`,
then runs the restore, interruption, reference-driver, and ds4-py self-replay
checks for that subset. It also retains a standalone C-side replay diagnostic.

A targeted report is marked `certification: "targeted"` and is not full M3
acceptance. Gates 1 and 6 cover only this subset. Gate 3 compares paired
reference-driver prefill/decode rates and cached-prefix fractions on identical
inputs, not wall time.

## Fixed inputs

- ds4 native source: `5379c1c9ac315f1540371b06a3db88491d4059ff`
- Polyglot benchmark: `7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f`
- Model: `/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- CUDA, context 100000, temperature 0, seed 1, response budget 8192,
  request limit 100, at most two attempts per task
- Run agents and model probes sequentially on a dedicated instance.

The manifest is `bench/gate-repairs.yaml`. Its explicit `targeted: true` permits
the four-task subset. Normal manifests still require ten exercises and two
repository tasks.

## Evidence layout

The commands below derive `RUN_ID=gate-repairs-<short-revision>` from the
uploaded revision marker and retain these remote paths:

- `/workspace/results/$RUN_ID/`: task, restore, interruption, reference-driver, self-replay, and gate evidence
- `/workspace/results/$RUN_ID-diagnostic/`: standalone C replay capture
- `/workspace/results/$RUN_ID.config`: revisions and native build identity
- `/workspace/results/$RUN_ID.runner.log`: complete runner output

The main runner creates `$RESULTS` itself and refuses to replace it. The C probe
uses a separate directory so it can run first without triggering that guard.

## 1. Upload the committed ds4-py revision

Run locally from this worktree after committing the preparation changes. Set the
SSH target and port supplied by Vast.ai:

```bash
export VAST_SSH="root@INSTANCE_IP"
export VAST_PORT="SSH_PORT"
export DS4PY_REVISION="$(git rev-parse HEAD)"
git diff --quiet
git diff --cached --quiet
git archive HEAD | ssh -p "$VAST_PORT" "$VAST_SSH" \
  'rm -rf /workspace/ds4-py && mkdir -p /workspace/ds4-py && tar -x -C /workspace/ds4-py'
ssh -p "$VAST_PORT" "$VAST_SSH" \
  "printf '%s\\n' '$DS4PY_REVISION' > /workspace/ds4-py/.bakeoff-revision"
```

Open a remote shell for all remaining remote commands:

```bash
ssh -p "$VAST_PORT" "$VAST_SSH"
```

## 2. Build and preflight on the instance

The instance must already contain the ds4 source, model, and Polyglot checkout
at the paths above. These commands rebuild the shared library and C agent from
the same ds4 source revision:

```bash
set -euo pipefail
cd /workspace/ds4-py
export DS4_SRC=/workspace/ds4
export POLYGLOT_BENCHMARK=/workspace/polyglot-benchmark
export DS4PY_HOME=/workspace/native
export DS4_LIB="$DS4PY_HOME/lib/libds4.so"
export DS4_AGENT="$DS4_SRC/ds4-agent"
export DS4_TEST_MODEL="$DS4_SRC/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
export DS4_BAKEOFF_BACKEND=cuda
export CUDA_ARCH=native
export DS4PY_REVISION="$(cat /workspace/ds4-py/.bakeoff-revision)"
export DS4_BAKEOFF_INSTANCE_ID="${INSTANCE_ID:?set the approved instance ID}"
export DS4_BAKEOFF_DS4_REVISION="$(git -C "$DS4_SRC" rev-parse HEAD)"
export DS4_BAKEOFF_DS4PY_REVISION="$DS4PY_REVISION"
export RUN_ID="gate-repairs-${DS4PY_REVISION:0:7}"
export RESULTS="/workspace/results/$RUN_ID"
export DIAG="/workspace/results/$RUN_ID-diagnostic"
export DS4_C_DRIVER="$DIAG/c-replay-diagnostic"

test "$(git -C "$DS4_SRC" rev-parse HEAD)" = \
  5379c1c9ac315f1540371b06a3db88491d4059ff
test "$(git -C "$POLYGLOT_BENCHMARK" rev-parse HEAD)" = \
  7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f
test -f "$DS4_TEST_MODEL"
test ! -e "$RESULTS"
test ! -e "$DIAG"

uv sync --frozen --extra agent
uv run ds4py build --src "$DS4_SRC" --backend cuda
make -C "$DS4_SRC" cuda CUDA_ARCH="$CUDA_ARCH"
uv run ds4py check
uv run --extra agent python bench/prepare_bakeoff.py \
  --manifest bench/gate-repairs.yaml \
  --benchmark "$POLYGLOT_BENCHMARK" \
  --repo-root /workspace/ds4-py \
  --check-only
```

Record the exact native artifacts before opening the model:

```bash
{
  echo "ds4py_revision=$DS4PY_REVISION"
  echo "ds4_revision=$(git -C "$DS4_SRC" rev-parse HEAD)"
  echo "polyglot_revision=$(git -C "$POLYGLOT_BENCHMARK" rev-parse HEAD)"
  echo "model=$DS4_TEST_MODEL"
  echo "backend=$DS4_BAKEOFF_BACKEND ctx=100000 tokens=8192 request_limit=100 temperature=0 seed=1 tasks=4 agents=c,ds4py"
  sha256sum "$DS4_LIB"
  file "$DS4_AGENT" "$DS4_LIB"
} > "/workspace/results/$RUN_ID.config"
cat "/workspace/results/$RUN_ID.config"
```

## 3. Capture the standalone C replay diagnostic

The retained case has 2,787 prompt tokens, an original cached prefix of 1,839,
and 16 C token IDs. The first 16 IDs match the final full-bake-off C trace. The
probe loads the same shared `libds4.so` as ds4-py and repeats the original split
prefill before greedy generation.

```bash
export CASE="$PWD/docs/reports/evidence/2026-10-31-targeted-8192/replay/c-scale-round0/case.json"
mkdir -p "$DIAG"
cp "$CASE" "$DIAG/case.json"
python3 - "$CASE" "$DIAG/case.tokens" <<'PY'
import json
import sys

with open(sys.argv[1]) as source:
    case = json.load(source)
assert len(case["prompt"]) == 2787
assert case["cached"] == 1839
assert len(case["sampled"]) >= 16
with open(sys.argv[2], "w") as output:
    for token in case["prompt"] + case["sampled"][:16]:
        print(token, file=output)
PY

cc -O2 -std=c11 -Wall -Wextra -Werror -I"$DS4_SRC" \
  bench/c_replay_diagnostic.c \
  -L"$DS4PY_HOME/lib" -Wl,-rpath,"$DS4PY_HOME/lib" -lds4 \
  -o "$DIAG/c-replay-diagnostic"
"$DIAG/c-replay-diagnostic" \
  --model "$DS4_TEST_MODEL" \
  --tokens "$DIAG/case.tokens" \
  --prompt-tokens 2787 \
  --split-at 1839 \
  --steps 16 \
  --ctx 100000 \
  --backend cuda \
  --think high \
  > "$DIAG/capture.jsonl" 2> "$DIAG/native.log"
tail -n 1 "$DIAG/capture.jsonl"
```

Compare `capture.jsonl` with the retained
`docs/reports/evidence/2026-10-31-targeted-8192/replay/c-scale-round0/native-split.json`:

- `first_divergence: 10`, with the same selected IDs and logits as the retained
  ds4-py capture, points to a C-baseline/native-build difference. Rebuild the C
  reference against the same library before changing engine behavior.
- `first_divergence: -1` means the current shared library reproduced all 16
  historical C IDs. Compare a fresh ds4-py capture before choosing the next
  repair.
- Any other divergence requires comparison of the sync records and first
  differing logits. Do not patch the sampler or relax exact replay from the
  summary alone.

## 4. Run the four-task two-arm checks

Keep the same remote shell and environment. The runner may exit nonzero while a
gate remains open; retain its report and logs rather than rerunning over them.

```bash
set +e
uv run --extra agent python bench/run_bakeoff.py \
  --manifest bench/gate-repairs.yaml \
  --benchmark "$POLYGLOT_BENCHMARK" \
  --repo-root /workspace/ds4-py \
  --results "$RESULTS" \
  --model "$DS4_TEST_MODEL" \
  --agent "$DS4_AGENT" \
  --backend "$DS4_BAKEOFF_BACKEND" \
  --tokens 8192 \
  --instance-id "$DS4_BAKEOFF_INSTANCE_ID" \
  --c-driver "$DS4_C_DRIVER" \
  2>&1 | tee "/workspace/results/$RUN_ID.runner.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e
printf 'runner_exit=%s\n' "$RUN_STATUS" | tee -a "/workspace/results/$RUN_ID.config"
cat "$RESULTS/report.txt"
```

For Gate 3, inspect each task's paired prefill/decode rates, cached-prefix
fractions, and provenance. For Gate 6, inspect both reference-driver equivalence
and ds4-py self-replay. Later agent traces need not match the C agent because
tool calls and tool results change subsequent prompts. Do not infer a native
throughput fix from wall time alone.

## 5. Collect the evidence

Run locally with the same SSH variables and the revision used above:

```bash
export RUN_ID="gate-repairs-DS4PY_SHORT_REVISION"
export EVIDENCE_ROOT="/home/roche/projects/satyrn/ds4-py-results"
mkdir -p "$EVIDENCE_ROOT"
rsync -az --info=progress2 -e "ssh -p $VAST_PORT" \
  "$VAST_SSH:/workspace/results/$RUN_ID" \
  "$VAST_SSH:/workspace/results/$RUN_ID-diagnostic" \
  "$VAST_SSH:/workspace/results/$RUN_ID.config" \
  "$VAST_SSH:/workspace/results/$RUN_ID.runner.log" \
  "$EVIDENCE_ROOT/"
```

Keep the instance available until the reports and diagnostic have been inspected
and any follow-up fix has been targeted. Do not start the full 12-task bake-off
from this subset's overall exit status.
