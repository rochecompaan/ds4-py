# Post-8,192 bake-off fixes design

Status: approved
Date: 2026-10-31
Branch: `m3-harness-agent`

## Evidence and baseline

The requirements come from `docs/reports/2026-10-31-bakeoff-8192-analysis.md`.
The live run tested ds4-py revision `e73ec15584825d51799eb089eecf02f4c180cba0`
with 8,192 tokens per response, temperature 0, seed 1, and CUDA.

The run improved ds4-py to 10/12 passing graders, but 11 of 14 attempts exited
nonzero. Two attempts hit Pydantic AI's implicit `request_limit=50`. Their
failure snapshots contain one message beyond the bound transcript, so the
resulting saves fail restore validation. All ordinary replay rounds diverge,
while all tool-call rounds match.

## Goals

1. Make request budgets explicit and keep failed-run saves internally consistent.
2. Remove `list_directory` from ds4-py so directory discovery uses the existing shell.
3. Run each failed task at least three times in a fresh process and workspace.
4. Test interruption after real token generation and restore with the configured response budget.
5. Diagnose replay parity from evidence before changing engine code.

## Non-goals

- No prompt-specific instructions for hidden-file discovery.
- No patch to `pydantic-ai-harness` or its upstream `list_directory` behavior.
- No weaker restore, interruption, replay, or clean-terminal checks.
- No automatic continuation after `max_tokens` or `parser_error`.
- No claim that three targeted passes provide statistical proof.
- No full bake-off before targeted correctness and acceptance probes pass.
- No replay implementation change before a minimal diagnostic identifies the fault.
- No performance patch from failed attempts or another blanket token increase.
- No automatic creation of a paid GPU instance.

## Design

### 1. Explicit request limits and failed-save integrity

`AgentRuntime` accepts `request_limit: int = 50` and always passes
`UsageLimits(request_limit=...)` to `Agent.run`. The existing production default
therefore remains 50, but it is explicit in ds4-py rather than inherited.

The ds4-py CLI adds `--request-limit` with a positive integer value and default
100. `open_runtime` passes it to `AgentRuntime`. The bake-off `AgentConfig`
carries `request_limit=100`; command construction adds the flag only for ds4-py,
never for C. Restore subprocesses receive both `request_limit` and the existing
response-token setting. Gate 2 error text names `request_limit` when the agent
log contains that terminal error.

Failure reconciliation must distinguish bound history from an unexecuted suffix.
`MessageBinder.adopt_failed(messages)` returns the longest history prefix backed
by bound token ranges. It preserves a bound trailing tool request but drops the
first request that never reached model preparation. `_fail` uses this method
instead of adopting every snapshot message. Successful runs continue to use the
existing authoritative `adopt` path.

A request-limit regression must run a real harness agent through a tool call,
fail before the next model request, save, and validate the save. The saved
messages and keyed token ranges must describe the same history. The run must
still exit with an error; this design does not turn request exhaustion into
success.

### 2. Shell-based directory discovery

`build_agent_components` configures `FileSystem` with exactly these tools:

```python
("read_file", "write_file", "edit_file", "search_files", "find_files", "create_directory", "file_info")
```

`list_directory` is absent. `Shell` remains enabled, so the model can use
`run_command` with `ls -la` to see hidden entries such as `.meta`. Other file
and repository tools remain unchanged.

Update the M3 design's capability table and composition example. Do not add
prompt instructions and do not change the dependency. A behavior test must
enumerate the real toolset and prove that `run_command` can list a hidden file.

### 3. Three independent targeted runs

Add `bench/run_targeted.py`. Its default task IDs are `scale-generator` and
`repo-ctx-command`, and its default repetition count is three. It reuses
`prepare_manifest` and `run_task` but invokes each repetition with
`max_attempts=1` and a fresh result root of the form
`results/<task-id>/run-<n>/`.

Each call clones a fresh workspace and starts a new ds4-py process. The targeted
entry point does not stop after a pass and does not run the full six-gate report.
It writes `targeted.json` with every `TaskRunResult` and exits nonzero if any
grader fails or any attempt records an error.

Every targeted run must produce a source edit, run the grader, and end cleanly.
For `repo-ctx-command`, the grader must prove `/ctx` registration. Three passes
are a repeatability check only.

### 4. Acceptance probes

`FooterModel` retains `running` at `run_start`, changes to `streaming` on the
first engine `Token`, and returns to `idle` at `run_end`. Startup and prefill
messages do not produce `streaming`.

Gate 4 waits for `\bstreaming\b` before sending Ctrl-C. It then waits for `idle`
before submitting the follow-up and waits for `idle` again before `/exit`.
This tests generation cancellation rather than model startup.

Restore case 1 passes `config.tokens` instead of hard-coded 256 and passes
`config.request_limit`. Case 2 still runs only after a clean case-1 continuation.
Save validation remains strict.

### 5. Live replay diagnosis and later full run

After local verification, use a user-supplied GPU instance for the targeted
runs and acceptance probes. Do not start a full bake-off unless all three runs
of both tasks pass cleanly and the repaired probes pass.

Select one short non-tool replay failure. On the same model, backend, and native
library, capture prompt identity, session position, prefill/cache behavior,
sampling arguments, mutable RNG state, logits, and selected tokens. Compare C
and ds4-py at the first divergence:

- same logits but different tokens implicates sampling or RNG state;
- different token-0 logits implicates prefill, KV state, backend numerics, or library configuration;
- later divergence implicates per-token evaluation or cache updates.

Keep all 62 passing tool-call checks and all non-tool comparisons. Add a minimal
regression only after the evidence identifies a production fix.

## Testing and commits

Production changes follow red-green TDD. Configuration and documentation-only
changes use direct verification under the Testing Value Gate. Each task is
committed separately. Before live work, run:

```bash
uv run --extra agent pytest -m "not build and not live" -q
uv run --extra agent pyright
uv run --extra agent ruff check src bench tests
uv run --extra agent ruff format --check bench/run_targeted.py tests/bench/test_targeted.py
git diff --check
```

Local fake-based tests are not live gate evidence. Performance reassessment
remains blocked until clean live attempts exist.
