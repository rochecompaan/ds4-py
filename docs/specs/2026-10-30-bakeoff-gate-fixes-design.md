# Bake-off gate fixes design

Status: approved
Date: 2026-10-30
Branch: `m3-harness-agent`

## Evidence and baseline

The evidence is recorded in:

- `docs/reports/2026-10-30-bakeoff-512.md`
- `docs/reports/2026-10-30-bakeoff-2048-analysis.md`

The live runs tested revision `6b5d9b9` plus five modified files. The current
`m3-harness-agent` baseline contains that revision and two later lifecycle
commits. The fix branch starts at `7cdb1c4`, which also preserves the benchmark
invocation and progress-logging changes used by the live runs.

The isolated baseline passed 1,437 tests with 17 skips. Existing harness
warnings remain baseline warnings; this work must not add new ones.

Two reported defects are already fixed by `6f40746` and need no duplicate
implementation:

- restore subprocesses run from the original task workspace, allowing `Skills`
  to acquire a workspace;
- Gate 4 uses the interactive TUI, interrupts visible generation, and waits for
  `idle` before the follow-up prompt and exit.

Their focused regression tests are
`test_restore_processes_run_from_the_original_task_workspace` and
`test_interruption_acceptance_uses_tui_idle_lifecycle`.

## Goals

1. Give thinking-enabled agents a usable default response budget without hiding
   exhaustion when an explicit budget is still too small.
2. Prevent `max_tokens` and `parser_error` responses from becoming successful
   agent runs, including when partial text or an earlier edit exists.
3. Count outcomes from the JSON events that ds4-py actually emits.
4. Read C traces containing arbitrary token bytes without losing replay data.
5. Keep each behavioral fix small, tested, and separately committed.

## Non-goals

- No automatic continuation after a token limit. A caller's limit must remain a
  real limit.
- No ds4-py-only budget increase. Both agents receive the same configured
  budget.
- No change to the external C writer in this repository.
- No performance patch based on the current failed runs. Wall-time analysis
  resumes only after clean attempts exist.
- No new live-model claim without a new full live run.

## Design

### 1. Response budget and terminal failures

`bench/run_bakeoff.py` currently defaults `--tokens` to 512. The 2048-token run
shows that even four times that default can be consumed by thinking before a
tool call. Raise the shared runner default to 8192 and document in
`bench/README.md` that both agents receive this value, that thinking consumes
it, and that a larger explicit value may be necessary. This is a benchmark
configuration change, not a claim that every task will fit.

The model adapter currently maps `max_tokens` to a `length` finish reason and
`parser_error` to an `error` finish reason, then returns a normal
`ModelResponse`. Pydantic AI rejects a thinking-only `length` response, but it
can accept partial non-empty text. It also has no general rejection for this
adapter's `error` finish reason. Consequently, an incomplete or malformed
response can reach `AgentRuntime` as success.

Reject these terminal reasons at the existing `DS4StreamedResponse` boundary:

- retain the finish-reason mapping for diagnostics;
- after terminal event dispatch and normal stream cleanup, raise a typed
  response-terminal error for `max_tokens` and `parser_error`;
- include the terminal reason in the error message;
- leave raw engine and `Session.run_round` behavior unchanged;
- leave `stop_token`, `tool_call_complete`, interruption, and context-reserve
  behavior unchanged.

`AgentRuntime.run` already converts an exception from its harness run into an
`error` run record and a nonzero noninteractive exit. Use that path rather than
scanning a whole trace after success. Post-run scanning could incorrectly fail
a successful run that had recovered from an earlier malformed round.

This raises the benchmark default while preserving truthful failure for any
explicit or default budget that is exhausted.

### 2. Canonical JSON tool accounting

`DS4RunRecord` writes harness events using their class names. Current Pydantic
AI emits `FunctionToolCallEvent` and `FunctionToolResultEvent`, while
`bench/metrics.py` only counts outcomes from `ToolResultEvent`. Engine
`ToolCalls` events therefore report parsed calls while executed outcomes remain
zero.

Extend `summarize_events` to understand the canonical harness events:

- register calls by their nonempty `tool_call_id`;
- resolve results against the same ID;
- classify successful, denied, failed, and interrupted outcomes without
  counting retry prompts as executions;
- reject duplicate resolutions, unknown IDs, malformed parts, and calls left
  without outcomes as metric evidence errors;
- keep engine `ToolCalls` as the parsed-call source so the matching harness call
  event does not double-count parsing;
- retain existing legacy `ToolResultEvent` support unless evidence shows it is
  no longer a supported input contract.

Pairing by ID is required even though the observed evidence happened to contain
250 matched pairs in order. Order alone is not the contract.

### 3. Byte-safe C trace reading

C token text can contain a prefix of a multibyte UTF-8 sequence. Replay only
needs the logged token IDs and metadata, but `engine/replay.py` currently
decodes the entire file as strict UTF-8 before parsing those fields. Eight live
C traces consequently fail before replay starts.

Read trace bytes and decode with UTF-8 `surrogateescape` at both existing text
entry points. This preserves every original byte one-to-one, including invalid
or split sequences, while leaving valid metadata and decimal token IDs
unchanged. Do not use replacement characters and do not rewrite the token
parser or replay protocol.

The regression fixture will include the observed `e2 99` / `af` and `e1 bb` /
`9d` splits and verify that parsing preserves the sampled IDs needed by replay.

### 4. Restore and interruption

No new code. Keep the existing `6f40746` implementation and tests, and verify
them in the final suite.

### 5. Performance follow-up

Keep the performance todo open. After correctness fixes, rerun both agents
with the same sufficient budget and compare attempts with edits and tests.
Separate agent output volume from engine throughput before proposing a
performance change.

## Alternatives considered

### Raise only the benchmark budget

This avoids some failures but does not fix false success when a response is
truncated or malformed. It also leaves the default at an unusably low value if
only documentation changes.

### Continue automatically after `max_tokens`

This would make the configured limit cease to be a limit and would complicate
response provenance across multiple engine rounds. Reject the incomplete
response instead.

### Scan completed traces for any bad terminal event

This is decoupled from the response that caused it and can misclassify a later
successful run after a recovered error. Reject the offending response at its
stream boundary.

### Replace invalid UTF-8 or rewrite the C parser around hex

Replacement loses bytes. A hex-only parser adds a second protocol even though
replay does not consume token display text. Surrogateescape is the smaller,
lossless reader fix.

## Testing and commits

Each production behavior fix follows red-green verification:

1. **Budget commit:** change the shared default and documentation. Per the
   Testing Value Gate, do not add a test that only restates static CLI
   configuration. Verify parser construction directly and run existing bench
   tests.
2. **Terminal-error commit:** add failing model and runtime regression tests
   for partial text under `max_tokens` and malformed tool syntax under
   `parser_error`; verify error run records and nonzero noninteractive behavior.
3. **Metrics commit:** add failing canonical call/result fixtures, including
   out-of-order results, retries, denials, and missing outcomes; then implement
   ID-based accounting.
4. **Trace commit:** add a failing byte-level trace fixture; then implement
   surrogateescape decoding and verify replay IDs.
5. Run focused tests after each change and commit that change separately.
6. Run the complete suite, lint, formatting, and type checks before declaring
   the branch complete.

Live bake-off, restore, replay, interruption, and performance gates require the
large model and benchmark environment. Local fake-based tests are not live
gate evidence.
