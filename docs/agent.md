# Agent guide

The agent interface connects ds4py to the Pydantic AI harness. It provides
workspace tools, approvals, saved sessions, an interactive shell, and token
traces.

## Install the dependencies

Install the core package and the agent extra from the repository root:

```bash
uv sync --extra agent
```

The extra pins `pydantic-ai-harness[cli]==0.59.0` and installs prompt-toolkit.
The core engine, `ds4py raw`, `ds4py build`, and `ds4py check` do not import the
harness or prompt-toolkit. Use `uv sync` without the extra for those commands.

## Start an interactive session

Run this command from the workspace that the agent can access:

```bash
uv run --extra agent ds4py \
  --model /absolute/path/ds4f-q2.gguf \
  --backend cpu
```

The current directory is the workspace root. The default thinking mode is
`high`. Use `--think`, `--think-level`, `--think-max`, or `--nothink` to select
a different mode before the session starts.

The interactive editor uses these keys:

- Press Enter to submit the prompt.
- Press Escape and then Enter to insert a newline.
- Press Ctrl-C to cancel the active model round.
- Press Ctrl-D to exit.

The footer shows the session, token use, and current state. Slash completion
lists the available commands.

## Run one prompt

A noninteractive run needs `--prompt` or `--prompt-file`:

```bash
uv run --extra agent ds4py \
  --model /absolute/path/ds4f-q2.gguf \
  --backend cpu \
  --non-interactive \
  --prompt "Summarize the failing test" \
  --trace /tmp/ds4py-prefill.jsonl
```

The runner saves the session in a `finally` block. This preserves run records
for success, interruption, and failure. The process prints the model response
and returns a nonzero status when the run fails.

## Control workspace access

The default approval policy asks before every shell command. It also asks
before a file mutation outside the workspace. File mutations inside the
workspace do not need approval.

The interactive prompt accepts these answers:

- `y`: Approve one request.
- `n`: Deny one request.
- `t`: Approve later requests for the same tool.
- `a`: Approve later requests for the same command or parent path.
- `s`: Deny the request and stop the session.

A denied tool returns a denied tool result to the model. The tool body does not
run.

Noninteractive runs have no approval prompt. They deny a request that needs
approval unless you pass `--yolo`:

```bash
uv run --extra agent ds4py \
  --model /absolute/path/ds4f-q2.gguf \
  --non-interactive \
  --prompt "Run the unit tests" \
  --yolo
```

`--yolo` bypasses approval prompts. It does not bypass workspace read-only
rules. If the agent must not mutate files, pass `--read-only`:

```bash
uv run --extra agent ds4py \
  --model /absolute/path/ds4f-q2.gguf \
  --read-only \
  --yolo
```

Read-only file rules stay active with `--yolo`. Shell commands can have side
effects, so use a sandbox when you cannot allow them.

## Store and restore sessions

`DS4PY_HOME` selects the storage directory. The default is `~/.ds4py`. The
directory contains native libraries and saved sessions:

```bash
export DS4PY_HOME=/absolute/path/ds4py-home
```

Each session has a directory under `$DS4PY_HOME/sessions/`. It contains the
save record, native payload, step snapshots, and `events.jsonl`.

The M3 interactive shell supports these commands:

- `/save`: Save the current session.
- `/switch <prefix>`: Restore the saved session with that unique ID prefix.
- `/new`: Close session state and start a new session.
- `/hints on|off`: Enable or disable programming hints.
- `/exit`: Close session state and exit.

The save operation syncs the native session before it writes files. It writes
the native payload and complete step snapshot before the save record. A
corrupt or incomplete record causes an error instead of an empty session.

A KV cache is the native key-value state for the prompt. `/switch` has two
restore paths:

1. Case 1 loads a valid KV payload. The switch performs no prefill. The next
   request extends the loaded state.
2. Case 2 renders the saved conversation when the payload is absent or
   incompatible. The next request performs a full rebuild prefill.

Both paths bind restored segments to the saved record. Token or key mismatches
cause an error.

## Read token traces

The `--trace` option writes ds4py prefill records as JSON Lines. JSON Lines is
a text format with one JSON object per line. Each record contains the prompt,
sampled token IDs, thinking mode, and round metadata.

The C agent uses its own text trace format. ds4py has strict readers for both
formats. Gate 6 replays the original C traces without translating them. Replay
parses tool calls but never executes tools.

Read [the bake-off guide](../bench/README.md) for the replay command and trace
locations.

## Run the live bake-off

The live suite needs the same `ds4f-q2` GGUF for every phase. The model is
about 81 GB. The documented machine assumption is at least 96 GB RAM.

Set the required paths and the explicit opt-in:

```bash
export DS4_TEST_MODEL=/absolute/path/ds4f-q2.gguf
export DS4_AGENT=/absolute/path/ds4-agent
export POLYGLOT_BENCHMARK=/absolute/path/polyglot-benchmark
export DS4_RUN_M3_BAKEOFF=1
```

Make sure that the benchmark checkout uses commit
`7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f`. Then run the preparation command:

```bash
uv run --extra agent python bench/prepare_m3.py \
  --manifest bench/m3.yaml --check-only
```

If the preparation succeeds, run the complete suite:

```bash
uv run --extra agent pytest tests/live/test_m3_bakeoff.py -m live -vv -s
```

The suite writes `bench/results/m3.json`. All six gates must pass before M3 is
live-verified. A guarded skip is not live evidence.

## Milestone limits

M4 owns real compaction, the 50,000-token reminder, and the remaining
persistence commands. M5 owns interactive `/think`. The M3 shell does not
provide those commands.

The runtime has a fork primitive for later milestones. M3 does not expose an
interactive `/fork` command.
