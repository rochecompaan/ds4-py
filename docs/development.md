# Development guide

This guide covers environment setup, portable tests, native library checks, and live replay parity.
Run the commands from the root of a ds4-py checkout.
Check out the revision that you want to test before you start.

## Enter the development environment

Follow [devenv's installation guide](https://devenv.sh/getting-started/) to install Nix and devenv.
On macOS, use the recommended Nix installer:

```bash
curl -sSfL https://artifacts.nixos.org/nix-installer | sh -s -- install
```

Devenv recommends this installer because it can handle OS upgrades and has better support for Apple silicon.
From the ds4-py checkout, enter the project environment:

```bash
devenv shell
```

The environment provides Python, `uv`, a C compiler, GNU Make, Git, curl, GNU coreutils, unzip, and actionlint.
`devenv.lock` pins the Nix inputs.
`uv` uses the Nix Python and does not download another interpreter.
It stores the project's virtual environment in `.devenv/state/venv`.
Shell startup does not install the Python dependencies.
Run all remaining commands inside this environment.

For automatic activation, install direnv and enable its hook in your shell.
Then run `direnv allow` in this checkout.
The included `.envrc` loads devenv.

## Run portable tests

Install the Python dependencies and run the portable tests:

```bash
uv sync --dev
uv run pytest -m "not build and not live" -q
```

The above `pytest` run needs no ds4 source tree, native library, or model.
It tests the Python code and synthetic native contracts.
It does not prove that model inference works on your hardware.

When you change packaging or CI configuration, run the relevant checks:

```bash
uv build --wheel
unzip -l dist/*.whl
actionlint .github/workflows/ci.yml
```

## Choose a backend

The backend is the compute implementation that runs model inference.
Use the same backend for `libds4`, `ds4-agent`, trace capture, and replay.
Pass the backend explicitly because the build defaults to Metal on macOS and CPU on other systems.

| `BACKEND` value | Choose this backend when | Extra requirements |
| --- | --- | --- |
| `cpu` | You want a CPU baseline or have no supported GPU. | None beyond the development environment. |
| `metal` | Your Mac supports Metal. | macOS, the macOS SDK, and Apple Command Line Tools. |
| `cuda` | Your machine has a supported NVIDIA GPU. | The CUDA toolkit and its `nvcc` compiler. |

`ds4py build` does not support ROCm yet.
Set `BACKEND` to one value from the table:

```bash
export BACKEND=cpu  # Change this to metal or cuda when appropriate.
```

For Metal, `devenv.nix` selects Apple SDK 15 for the pinned ds4 source's macOS 15 Metal APIs.
It also sets `MACOSX_DEPLOYMENT_TARGET` to `15.0`.
These settings avoid the deployment-target conflict between devenv's default SDK and PyTorch on macOS.

For CUDA, set the GPU architecture before the native builds:

```bash
export CUDA_ARCH=native
```

The `native` value targets the local NVIDIA GPU.
Replace it with an explicit value such as `sm_86` when your setup requires one.

## Get the matching ds4 source

Build `libds4` and `ds4-agent` from the same ds4 source revision.
These commands read the upstream repository and commit from [`src/ds4py/abi/pins.py`](../src/ds4py/abi/pins.py).
Do not substitute the current ds4 `main` branch.

```bash
export DS4_SRC="$PWD/../ds4-src"
read -r DS4_REPO DS4_COMMIT <<< "$(
  uv run python -c \
    'from ds4py.abi import pins; print(pins.UPSTREAM.repo, pins.UPSTREAM.commit)'
)"

if [ -d "$DS4_SRC/.git" ]; then
  git -C "$DS4_SRC" fetch origin
else
  git clone "$DS4_REPO" "$DS4_SRC"
fi
git -C "$DS4_SRC" checkout -f "$DS4_COMMIT"
```

For fork-specific work, replace `pins.UPSTREAM` with `pins.FORK` in the Python command.
The source pin then selects the matching fork repository and commit.

### Apply the Metal language-version patch

On macOS with `BACKEND=metal`, apply the patch that matches the selected source pin.
Apply it once to the clean source checkout before building `libds4` or `ds4-agent`.

For `pins.UPSTREAM`, run:

```bash
git -C "$DS4_SRC" apply \
  "$PWD/docs/patches/ds4-metal-language-version-upstream.patch"
```

For `pins.FORK`, run:

```bash
git -C "$DS4_SRC" apply \
  "$PWD/docs/patches/ds4-metal-language-version-fork.patch"
```

Do not apply both patches.
If you reset or check out the source again, apply the matching patch again.
Keep the same patched source tree for both native builds.
CPU and CUDA builds do not need these patches.

Metal chooses its default shader language from the SDK of the host executable.
The host executable is Nix Python in this environment.
The current Nix Python links against macOS SDK 14.4.
Metal can therefore select a language below Metal 3.2, even when `libds4` uses the macOS 15 SDK.
The macOS 15 deployment target does not change Nix Python's SDK metadata.

The pinned ds4 shaders use `memory_order_seq_cst` and `thread_scope_device`.
These features require Metal 3.2 on macOS 15.
Without explicit language selection, the native build can succeed but model loading or replay fails during shader compilation.
The patches select Metal 3.2 when the SDK and operating system support it.
They select Metal 4.0 on macOS 26 or later when the SDK supports it.

## Build and check the native library

Set `DS4PY_HOME` to keep this test installation inside the checkout.
The build command installs the library, its manifest, and any backend support files.
The check command validates the native interface without opening a model.

```bash
export DS4PY_HOME="$PWD/.ds4py"
uv run ds4py build --src "$DS4_SRC" --backend "$BACKEND"
uv run ds4py check
```

Keep `DS4PY_HOME` set in the shell that runs the live tests.
The test fixtures use its library by default.
If you unset it, set `DS4_LIB` to the built library path instead.

## Build the matching agent

The live replay test compares Python token IDs with a trace from the C agent.
Build the agent from `DS4_SRC` with the backend that you selected.
Run the command for that backend:

```bash
case "$BACKEND" in
  cpu)
    make -C "$DS4_SRC" cpu
    ;;
  metal)
    make -C "$DS4_SRC" ds4-agent
    ;;
  cuda)
    make -C "$DS4_SRC" cuda CUDA_ARCH="${CUDA_ARCH:-native}"
    ;;
esac

export DS4_AGENT="$DS4_SRC/ds4-agent"
```

On macOS, the `ds4-agent` target builds the Metal agent.
The CPU and CUDA targets also build the other ds4 executables.
The CUDA command uses `CUDA_ARCH` from the environment or `native` when it is unset.

## Capture and replay a live trace

Use the exact `frozen-1221-quantized-DS4-Flash.gguf` model for the standard parity check.
Change the path below to its location on your machine.
Make sure that the machine has enough free memory for the model.
Metal reads shader files from the agent's working directory.
Run the agent from `DS4_SRC` in a subshell so your shell stays in the ds4-py checkout.

```bash
export MODEL="$HOME/models/frozen-1221-quantized-DS4-Flash.gguf"
export TRACE="$PWD/c-agent.trace"
rm -f "$TRACE"

(
  cd "$DS4_SRC"
  "$DS4_AGENT" --non-interactive --model "$MODEL" --backend "$BACKEND" \
    --ctx 4096 --tokens 2048 --temp 0 --seed 1 \
    --prompt 'Read README.md and report the project name.' \
    --trace "$TRACE"
)
```

Remove the old trace before each capture because `--trace` appends to an existing file.
The high token limit lets the model reach a natural stop or tool boundary.
The trace records the model path, prompt tokens, sampled tokens, and round boundaries.

Replay the trace through ds4-py:

```bash
DS4_TEST_MODEL="$MODEL" DS4_REPLAY_TRACE="$TRACE" \
  uv run pytest -m live tests/live/test_replay_parity.py -v
```

`DS4_TEST_MODEL` must resolve to the same model path that the trace records.
Replay uses the prompts from the C trace and does not execute parsed tool calls.
A successful replay ends with `1 passed`; a skipped test does not prove live parity.

The 27B model can produce `ds4-agent: failed to save system prompt KV: unsupported routed quantization for KV save`.
The agent continues without saving that KV cache.
This warning does not invalidate replay when capture completes and `pytest` reports `1 passed`.

See [`tests/live/REPLAY.md`](../tests/live/REPLAY.md) for trace semantics and bake-off captures.
When you report a result, include the backend, operating system, GPU or chip, and test output.
If a command fails, include its complete output rather than only the final error line.
