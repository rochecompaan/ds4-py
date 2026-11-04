#!/usr/bin/env bash
# Rerun ds4-py after the initial attempt omitted DS4_LIB and never opened the model.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export DS4PY_HOME=/workspace/native
export DS4_LIB="$DS4PY_HOME/lib/libds4.so"
ROOT=/workspace
PYROOT=$ROOT/ds4-py
EVIDENCE=$PYROOT/docs/reports/evidence/2026-11-04-gate-2-early-write
MODEL=$ROOT/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
RESULTS=$ROOT/results/gate2-early-write-ds4py-live
cd "$PYROOT"
date -u > /workspace/results/ds4py-live-started-at.txt
uv run python bench/run_targeted.py \
  --manifest "$PYROOT/bench/gate-repairs.yaml" \
  --benchmark "$ROOT/polyglot-benchmark" \
  --repo-root "$PYROOT" \
  --model "$MODEL" \
  --backend cuda \
  --tokens 8192 \
  --task-preamble "$EVIDENCE/task-preamble.md" \
  --repetitions 3 \
  --task scale-generator \
  --task affine-cipher \
  --task tree-building \
  --task zipper \
  --results "$RESULTS" > /workspace/run-ds4py-live.log 2>&1
STATUS=$?
printf '%s\n' "$STATUS" > /workspace/results/ds4py-live-exit.txt
date -u > /workspace/results/ds4py-live-finished-at.txt
printf 'ds4py_live_exit=%s\n' "$STATUS"
exit "$STATUS"
