#!/usr/bin/env bash
# One-off setup for the Gate 2 early-write task-prompt experiment.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT=/workspace
SRC=$ROOT/ds4
PYROOT=$ROOT/ds4-py
BENCHMARK=$ROOT/polyglot-benchmark
REVISION=5379c1c9ac315f1540371b06a3db88491d4059ff
BENCHMARK_REVISION=7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f
BASE_REVISION=${1:?ds4-py base revision required}
MODEL=DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
URL="https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/$MODEL"
EXPECTED=ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0
PREAMBLE=$PYROOT/docs/reports/evidence/2026-11-04-gate-2-early-write/task-preamble.md
NATIVE_PATCH=bench/patches/cuda-no-copy-host-register-opt-out.patch
export DS4_BAKEOFF_NATIVE_PATCHES=$NATIVE_PATCH
export DS4_CUDA_DISABLE_HOST_REGISTER=1
mkdir -p "$ROOT" "$ROOT/results"
date -u > "$ROOT/setup-started-at.txt"
df -B1 / > "$ROOT/disk-before.txt"
printf '%s\n' "$BASE_REVISION" > "$PYROOT/.bakeoff-revision"
if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/antirez/ds4.git "$SRC" > "$ROOT/clone.log" 2>&1
fi
git -C "$SRC" checkout "$REVISION" >> "$ROOT/clone.log" 2>&1
git -C "$SRC" apply --check "$PYROOT/$NATIVE_PATCH"
git -C "$SRC" apply --whitespace=error "$PYROOT/$NATIVE_PATCH"
if [ ! -d "$BENCHMARK/.git" ]; then
  git clone https://github.com/Aider-AI/polyglot-benchmark.git "$BENCHMARK" >> "$ROOT/clone.log" 2>&1
fi
git -C "$BENCHMARK" checkout "$BENCHMARK_REVISION" >> "$ROOT/clone.log" 2>&1
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh > "$ROOT/uv-install.log" 2>&1
fi

build_c() {
  local status
  if make -C "$SRC" cuda CUDA_ARCH=native > "$ROOT/build.log" 2>&1; then status=0; else status=$?; fi
  printf '%s\n' "$status" > "$ROOT/build-exit.txt"
}
setup_ds4py_env() {
  local status
  if (cd "$PYROOT" && uv sync --frozen --extra agent) > "$ROOT/ds4py-env.log" 2>&1; then status=0; else status=$?; fi
  printf '%s\n' "$status" > "$ROOT/ds4py-env-exit.txt"
}
download_model() {
  local status
  mkdir -p "$SRC/gguf"
  if curl --fail --location --silent --show-error --retry 5 --retry-delay 5 --continue-at - \
    --output "$SRC/gguf/$MODEL" "$URL" > "$ROOT/model-download.log" 2>&1; then status=0; else status=$?; fi
  printf '%s\n' "$status" > "$ROOT/model-download-exit.txt"
}
build_c & BUILD_PID=$!
setup_ds4py_env & ENV_PID=$!
download_model & MODEL_PID=$!
wait "$BUILD_PID" || true
wait "$ENV_PID" || true
wait "$MODEL_PID" || true
[ "$(cat "$ROOT/build-exit.txt")" = 0 ]
[ "$(cat "$ROOT/ds4py-env-exit.txt")" = 0 ]
[ "$(cat "$ROOT/model-download-exit.txt")" = 0 ]
cd "$PYROOT"
rm -rf "$ROOT/native/lib/build"/cuda-*
CUDA_ARCH=native uv run ds4py build --src "$SRC" --backend cuda --out "$ROOT/native/lib" > "$ROOT/libds4-build.log" 2>&1
printf '%s  %s\n' "$EXPECTED" "$SRC/gguf/$MODEL" | sha256sum --check > "$ROOT/model-sha256.txt" 2>&1
{
  date -u
  uname -a
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  nvcc --version | tail -n 1
  echo "ds4_revision=$(git -C "$SRC" rev-parse HEAD)"
  echo "benchmark_revision=$(git -C "$BENCHMARK" rev-parse HEAD)"
  echo "ds4py_base_revision=$BASE_REVISION"
  echo "DS4_CUDA_DISABLE_HOST_REGISTER=$DS4_CUDA_DISABLE_HOST_REGISTER"
  echo "DS4_BAKEOFF_NATIVE_PATCHES=$DS4_BAKEOFF_NATIVE_PATCHES"
  stat -c 'model_size=%s' "$SRC/gguf/$MODEL"
  sha256sum "$SRC/ds4-agent" "$ROOT/native/lib/libds4.so" "$SRC/gguf/$MODEL" \
    "$PYROOT/$NATIVE_PATCH" "$PYROOT/bench/run_targeted.py" "$PREAMBLE"
} > "$ROOT/provenance.txt"
df -B1 / > "$ROOT/disk-after.txt"
date -u > "$ROOT/setup-finished-at.txt"
cat "$ROOT/provenance.txt"
cat "$ROOT/disk-after.txt"
