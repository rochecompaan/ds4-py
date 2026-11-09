# ds4py

ds4py provides Python bindings and an engine interface for DS4. The optional
agent interface uses Pydantic AI and prompt-toolkit.

## Install

Install the core package:

```bash
uv sync
```

Install the agent interface:

```bash
uv sync --extra agent
```

The core commands do not require the agent dependencies:

```bash
uv run ds4py raw --help
uv run ds4py check --help
uv run ds4py build --help
```

## Start the agent

Start an interactive session in the current workspace:

```bash
uv run --extra agent ds4py \
  --model /absolute/path/ds4f-q2.gguf \
  --backend cpu
```

Run one prompt and write a token trace:

```bash
uv run --extra agent ds4py \
  --model /absolute/path/ds4f-q2.gguf \
  --backend cpu \
  --non-interactive \
  --prompt "Explain this repository" \
  --trace /tmp/ds4py-prefill.jsonl
```

Read [the agent guide](docs/agent.md) for workspace access, sessions, restore
paths, and storage.

### Compact context

The default agent uses `LLMCompaction` to summarize older history automatically
at 85 percent context use, or earlier to reserve room for a summary. It also
attempts recovery when a tool result or an in-progress response needs room.
Reserve recovery continues the same model response without repeating output.

Use `/compact` with no arguments to compact manually while idle. The command
is rejected during an active run, including while a tool is waiting for an
answer. Manual compaction remains available with a custom automatic context
owner.

Compaction replaces the current stored history with a summary and a verbatim
recent tail, preserving retained message keys and token provenance. Saving
persists that compacted history; older step snapshots still contain removed
history. A valid native KV payload restores without prefill. Save-record v1
and old step snapshot formats are unchanged.

See [the development guide](docs/development.md#context-compaction) for owner
composition, persistence, and limits. Full native inference and live replay
parity remain deferred. There is no `/strip` command, and C's second
nearly-full restored-context `summary_bottom` cutoff and retained-unsummarized-token
path are not implemented; these cases instead use a safe no-op or fail with
state preserved. This is not complete C parity.

## Acceptance

Read [the agent bake-off guide](bench/README.md) for the fixed inputs, machine
requirements, and live commands. The normal test suite does not load a model:

```bash
uv run --extra agent pytest -m "not live and not build"
```

The [design specification](docs/specs/2026-10-21-ds4-py-design.md) defines the
engine and milestone boundaries.
