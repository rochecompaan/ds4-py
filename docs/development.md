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
uv sync --frozen --extra agent
uv run --frozen --extra agent pytest -m "not build and not live" -q
```

The above `pytest` run needs no ds4 source tree, native library, or model.
It tests the Python code and synthetic native contracts.
It does not prove that model inference works on your hardware.

Install the prek hook once, then run the same static quality gate used by CI:

```bash
uv run prek install
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
```

The gate includes Ruff lint and formatting, ty, TOML, YAML, JSON, GitHub workflow schema checks, actionlint, and general file hygiene.
Run ty directly when you need its complete diagnostics without the other hooks:

```bash
uv run --frozen --extra agent ty check
```

Use `TypeIs`, not `TypeGuard`, for custom narrowing validators.
Prefer precise types, generics, and `object`; keep `Any` only at reviewed dynamic boundaries.
See [`AGENTS.md`](../AGENTS.md) for the complete typing and lint policy.

When you change packaging or CI configuration, run the relevant checks:

```bash
uv build --wheel
unzip -l dist/*.whl
actionlint .github/workflows/ci.yml
```

## Context compaction

The default agent registers `LLMCompaction` as its automatic context owner.
Before a model request, it checks the bound transcript against the configured
context size. The default soft trigger is 85 percent (`used >= int(ctx * 0.85)`).
It also triggers when free space is no greater than the larger of
`min(8192, ctx // 8)` and `summary_reserve() + 128`. The summary reserve includes
the rendered reason-bearing summary suffix, 64 safety tokens, and a summary
budget of `max(256, min(ctx // 8, 4096))` tokens (1024→256, 8192→1024,
65536→4096).

Trigger sizing and private summary execution share `summary_budget` and exact
`summary_suffix` rendering, not policy. The conservative trigger appends
`Compaction reason: context pressure before continuing the current task` to its
reserve prompt and retains the +64 safety and +128 threshold margins. Execution
uses the fixed, reason-free summary prompt against the caller's exact token
prefix, followed by the no-think assistant opener. It leaves one spare native
slot, rejects fewer than 256 remaining output slots, and caps output at
`min(remaining_room, summary_budget(ctx))`. The trigger reserve is intentionally
not the private generation request; no reason is passed to summary jobs.

The same owner receives oversized tool-result hooks and response-reserve
requests. When a response reaches its reserve boundary, generation pauses,
compaction attempts to make room, and generation resumes in the same
`ModelResponse`. Already delivered response text is not emitted again. Summary
tokens count once as output usage for automatic compaction, but never become
response text. Threshold, tool-overflow, and reserve recovery count successfully
emitted `SummaryToken` output even if the owner subsequently fails or is
cancelled. On success, only the unobserved remainder of
`RoomResult.generated_tokens` is added, so emitted summary output is never
charged twice; a failure before emission adds nothing.
Custom owners that emit no summary events retain their result-based accounting.
If a tool result still does not fit, the owner returns a tool
error instead of inserting that oversized result. If reserve recovery cannot
make room, the response fails with `ContextExhausted`.

### Manual compaction and custom owners

In the interactive shell, `/compact` takes no arguments. It reports the token
change on success or `Nothing to compact` when there is no removable history.
It is rejected while a run is active, including while an interactive tool is
waiting for an answer; it neither queues compaction nor treats the command as
the answer. The programmatic entry point is `await runtime.compact()`, with the
same idle-only restriction.

`ContextComposition(owner=custom_strategy)` replaces the default automatic
owner with a `ContextStrategy` capability. Exactly one automatic owner is
registered: the selected capability receives the tool-overflow and
response-reserve hooks as well as model-request hooks. The selection survives
`/new` and `/switch`.

`runtime.components.compactor` remains a concrete `LLMCompaction` manual
service, even when a custom automatic owner is selected. Its
`compact_messages(messages, reason="manual")` method operates on the supplied
history in place. The runtime's manual entry point uses this service, not the
custom owner. Replacing the automatic owner does not register a second
automatic compactor. Use the runtime entry point for its operation guard.

Summary preparation and native replacement run off the event loop through
retained, shielded executor operations; all native calls still go through the
single engine worker. Provenance preparation and binder/message publication
stay on the loop. Session listeners (including manual-compaction progress) run
on that same loop. The TUI keeps one input application running while an async
slash command executes, so the compacting footer actually paints, Ctrl-C can
signal the operation, and registry-marked immediate commands still busy-reject.
Other input retains the existing queue/answer routing; cancelling the command
does not cancel the input loop or create a competing prompt.

One compaction attempt owns its entire drain. Overlapping attempts reject
before emitting progress. While a runtime manual compaction is active,
run/compact/save/new/switch/fork reject immediately rather than queue or cancel
it. `runtime.cancel()` signals that operation; `runtime.close()` cancels and
rejects new operations from the start of closing and awaits its drain before
closing session state. Concurrent closes share that teardown boundary.
Cancellation, including repeated `Task.cancel()` or AnyIO deadlines, never
abandons the executor work. A native replacement that already succeeded
publishes its matching prepared Python
state before cancellation propagates; a failed replacement publishes nothing.
The terminal event follows drain/publication and still reports cancellation,
not a fabricated successful compaction.

Nonstream reserve recovery uses the private `get_current_run_context` accessor
from the exactly pinned harness to select canonical history. This is the
nonstream equivalent of `request_stream`'s explicit context, not another history
store: a request-only copy cannot preserve a commit followed by an owner error.
No custom-owner method signature or dependency version changes are required.

### Compaction session events

Every started attempt emits `before_compact(reason, old_tokens)` before tail
planning, then exactly one `compaction_finished(reason, outcome)` in `finally`.
The terminal outcome is `compacted`, `noop`, `failed`, or `cancelled`. A busy
rejection starts no attempt and emits neither event.

Only a successful commit emits `compacted(reason, old_tokens, new_tokens,
generated_tokens)`, before `compaction_finished`. No-op, failure, and
cancellation never fabricate success or reset the system-note provider. An
internal error after native replacement also terminates with `failed`; this
notification does not promise rollback of a replacement that already succeeded.

The footer restores its previous idle/running status from the terminal event,
not a command name or a success notification. A later `run_end` takes precedence
so a late finish cannot revive a completed run. All three displayed success
counts must be non-negative integers (not booleans); the footer validates the
complete count payload and terminal outcome before mutating their state.

### History replacement and persistence

Summary generation uses the session's model with thinking disabled and greedy,
tool-masked decoding. The engine receives only the selected token prefix; the
capability selects the message tail. The tail budget is
`max(1, min(ctx // 10, 50_000))` tokens, adjusted to whole messages and complete
tool exchanges. Retained token IDs are copied from the transcript, not
reconstructed from message text, including an open response tail during reserve
recovery.

Compaction replaces the live canonical message list in place with one summary
request followed by the retained messages. It preserves system instructions,
assigns a fresh `ds4.key` to the summary request, and preserves the retained
messages' stable keys and token provenance. The native transcript becomes the
system prefix, rendered summary request, and verbatim retained tail. It is not
an extra summary appended to otherwise unchanged history.

The commit prepares message provenance first, performs native replacement as
the only fallible mutation, then installs the prepared binder and message
state. A failed native replacement leaves the previous history and transcript
intact. An interrupted or failed replacement sync restores the prior native
prefix, preserving any not-yet-evaluated transcript suffix, without emitting
another usage-accounting sync. Restoration must complete successfully; an
exception or interrupted restoration poisons the worker, retaining the original
replacement failure as exception context, so mismatched state cannot be reused.
There is no Python-history rollback protocol. Summary preparation restores the
borrowed native state before replacement. A summary restoration failure is
surfaced and poisons the worker rather than continuing with private summary
state.

Saving writes the compacted current history and matching native payload.
Switch and fork preserve its keys and provenance. Existing step snapshots
retain the removed messages, so compaction is not a deletion or redaction of
older run records. Save-record v1 and old step snapshot formats are unchanged.
An exact compatible native KV payload is reused with no restore prefill; the
next request extends the loaded state. With an absent, stripped, or
incompatible payload, the compacted messages are rendered again and the next
request performs a rebuild prefill.

### Verification limits

Portable agent, engine, persistence, and conformance tests use synthetic native
contracts, including FakeNative. They do not establish native inference or live
replay parity; full native inference and live replay parity remain deferred.
The live setup below describes how to collect that separate evidence, not a
completed compaction parity result.

Two scope gaps remain:

- There is no command-level `/strip`. Existing payload removal and restore
  paths cover the equivalent stripped-payload behavior in portable tests.
- C's second nearly-full restored-context `summary_bottom` cutoff and its
  retained-unsummarized-token path are not implemented. When no history can
  safely be removed, compaction is a no-op; when summary generation cannot fit,
  it fails with prior state preserved instead of implementing that C fallback.

The trigger rules and covered continuation behavior do not imply complete C
parity. Other deferred milestone work is not made complete by compaction.

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

### Apply the CUDA host-register opt-out patch

Apply this patch on a CUDA host when `cudaHostRegister` fails or stalls on the no-copy path.
The vast.ai hosts showed this symptom.
Apply it once to the clean source checkout before building `libds4` or `ds4-agent`.

```bash
git -C "$DS4_SRC" apply --check \
  "$PWD/docs/patches/cuda-no-copy-host-register-opt-out.patch"
git -C "$DS4_SRC" apply --whitespace=error \
  "$PWD/docs/patches/cuda-no-copy-host-register-opt-out.patch"
```

The patch adds the `DS4_CUDA_DISABLE_HOST_REGISTER=1` runtime opt-out.
Set this variable in each shell that starts `ds4-agent` or loads `libds4`:

```bash
export DS4_CUDA_DISABLE_HOST_REGISTER=1
```

The opt-out selects the existing device-cache path instead of no-copy host registration.
The patch is a no-op when the variable is unset.
If you reset or check out the source again, apply the patch again.
Keep the same patched source tree for both native builds.
CPU and Metal builds do not need this patch.

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
