#!/usr/bin/env bash
# Run the approved four-task, three-repetition, shared-preamble experiment.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export DS4PY_HOME=/workspace/native
export DS4_LIB="$DS4PY_HOME/lib/libds4.so"
ROOT=/workspace
PYROOT=$ROOT/ds4-py
EVIDENCE=$PYROOT/docs/reports/evidence/2026-11-04-gate-2-early-write
MODEL=$ROOT/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
RESULTS=$ROOT/results
mkdir -p "$RESULTS"
cd "$PYROOT"
COMMON=(
  --manifest "$PYROOT/bench/gate-repairs.yaml"
  --benchmark "$ROOT/polyglot-benchmark"
  --repo-root "$PYROOT"
  --model "$MODEL"
  --backend cuda
  --tokens 8192
  --task-preamble "$EVIDENCE/task-preamble.md"
  --repetitions 3
  --task scale-generator
  --task affine-cipher
  --task tree-building
  --task zipper
)
date -u > "$RESULTS/experiment-started-at.txt"
uv run python bench/run_targeted.py "${COMMON[@]}" \
  --c-agent "$ROOT/ds4/ds4-agent" \
  --results "$RESULTS/gate2-early-write-c" > "$ROOT/run-c.log" 2>&1
C_STATUS=$?
printf '%s\n' "$C_STATUS" > "$RESULTS/c-exit.txt"
uv run python bench/run_targeted.py "${COMMON[@]}" \
  --results "$RESULTS/gate2-early-write-ds4py" > "$ROOT/run-ds4py.log" 2>&1
DS4PY_STATUS=$?
printf '%s\n' "$DS4PY_STATUS" > "$RESULTS/ds4py-exit.txt"
date -u > "$RESULTS/experiment-finished-at.txt"
printf 'c_exit=%s\nds4py_exit=%s\n' "$C_STATUS" "$DS4PY_STATUS"
if [ "$C_STATUS" -ne 0 ] || [ "$DS4PY_STATUS" -ne 0 ]; then exit 1; fi
