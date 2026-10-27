# Live `ds4-agent` replay parity

This test compares token IDs (numbers for model input and output units). Every
round starts from the exact prompt tokens recorded by C. The test never executes
a parsed call. Python tool output cannot change the next prompt.

The section-21 bake-off capture target is the frozen 1221 quantized DS4-Flash
model specified for that work. Use that exact GGUF model file for bake-off
captures. Do not substitute another model. The generic M2 reader and runner also
accept GLM, Qwen, and DeepSeek V4.1 traces for focused live checks. Those checks
do not replace the section-21 bake-off run.

## Capture a simple tool trace

Build `ds4-agent` and `libds4` from the same ds4 source revision. Use the same
backend (the compute implementation) for both binaries. If the library is not in
ds4-py's default library directory, set `DS4_LIB`. Replace the paths and backend
below with the values for that installation. Use the actual filename of the
frozen model. The placeholder is not a second model choice.
If you use Metal with devenv's Nix Python, apply the matching language-version
patch before these builds. Follow [the development guide's patch
step](../../docs/development.md#apply-the-metal-language-version-patch).
The patch prevents Nix Python from selecting a shader language older than
Metal 3.2 during replay.
Metal reads shader files from the agent's working directory.
Run the agent from `DS4_SRC` in a subshell so your shell stays in the ds4-py checkout.

```bash
export DS4_SRC=/absolute/path/to/ds4
export DS4_AGENT="$DS4_SRC/ds4-agent"
export DS4_LIB=/absolute/path/to/libds4.so
export MODEL=/absolute/path/to/frozen-1221-quantized-DS4-Flash.gguf
export BACKEND=cpu  # Or cuda or metal; it must match libds4.
export TRACE="$PWD/c-agent.trace"

# --trace appends, so start with a fresh file.
rm -f "$TRACE"
(
  cd "$DS4_SRC"
  "$DS4_AGENT" --non-interactive --model "$MODEL" --backend "$BACKEND" \
    --ctx 4096 --tokens 2048 --temp 0 --seed 1 \
    --prompt 'Read README.md and report the project name.' \
    --trace "$TRACE"
)
```

Set `--tokens` (`-n`) high. This lets the model reach a natural stop or a tool
boundary before the caller's output limit. The trace records the context size,
per-round think mode, prompt tokens, sampled tokens, early edit stops, and
context-limited boundaries. Replay uses these recorded values.

## Replay the trace

Run this command from the ds4-py checkout. `DS4_TEST_MODEL` must resolve to the
same GGUF path recorded in the trace. The test fails if the paths differ. Replay
uses the trace's context size and the capture's `--temp 0 --seed 1` policy.

```bash
DS4_TEST_MODEL="$MODEL" DS4_REPLAY_TRACE="$TRACE" \
  uv run pytest -m live tests/live/test_replay_parity.py -v
```

This is a live test (a test that loads a real model). If either variable is
unset, the test skips with a clear reason. Synthetic reader and parser tests
still run in the default suite without these assets. They do not show live replay
parity.

## Replay bake-off traces individually

Each section-21 `ds4-agent` run already uses `--trace`. To replay one run, set
`TRACE` to its trace file. Set `MODEL` to the absolute frozen-model path used for
that capture. Then run the replay command above.

Replay each trace separately. Do not concatenate runs. `--trace` appends, and
each replay expects one worker start.
