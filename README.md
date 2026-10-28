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

Read [the agent guide](docs/agent.md) for approvals, sessions, restore paths,
and storage.

## Acceptance

Read [the M3 bake-off guide](bench/README.md) for the fixed inputs, machine
requirements, and live commands. The normal test suite does not load a model:

```bash
uv run --extra agent pytest -m "not live and not build"
```

The [design specification](docs/specs/2026-10-21-ds4-py-design.md) defines the
engine and milestone boundaries.
