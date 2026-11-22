# Tool profiles design

| | |
| --- | --- |
| Date | 2026-11-22 |
| Status | Approved |
| Branch | `feat/tool-profiles` |
| Base | PR A at `934ee5b917fb69f301a90e5f0dc4b2404192331d` |
| Reference | `ds4-mellum` at `d2d9338`, read-only |

## Intent

Add a generic `auto | native | pi` tool profile to the existing ds4-py agent.
The Pi profile presents the four operations Mellum was trained on while reusing
the existing workspace, filesystem, shell, event, retry, and output-limit paths.

## Goals

1. Thread the profile through CLI, engine options, runtime, and composition.
2. Expose exactly `read`, `bash`, `edit`, and `write` with reference-compatible
   schemas and Pi-specific instructions.
3. Convert only the adapter-level arguments and preserve existing result text.
4. Resolve `auto` to native for existing families and Pi for Hermes as a
   development hypothesis, not a measured result.
5. Deliver a real PR A → PR B stack with portable verification and reviews.

## Non-goals

- No live model comparison or claim that the Hermes `auto` mapping is final.
- No new filesystem/shell execution implementation or Pi result-text parity.
- No changes to the Hermes parser or native engine.
- No downstream product identifiers or policy in ds4-py.

## Constraints

- Existing-family and explicit-native output must remain unchanged.
- Keep devenv + `uv --frozen --extra agent` and test-first behavior changes.
- Pi schemas match the reference except the approved read character wording
  (`50,000 characters`) and bash timeout wording (the configured 30-second
  default).
- Workspace containment, read-only policy, events, retries, and output limits
  stay in the existing toolsets.
- A read defaults to 2,000 lines; offsets 0 and 1 both select the first line.
- Batch edits match every `oldText` against the original file, not
  incrementally; ambiguous or overlapping matches are rejected before writing.
  Omitted bash timeout uses the configured default of 30 seconds.

## Approaches

### Chosen: a toolset adapter at the existing execution boundary

Add a Pi capability that wraps the existing filesystem and shell toolsets. It
prepares exact Pi schemas, maps calls back to the original tools, and converts
arguments at that boundary. Composition selects native or Pi components and
instructions without changing the model or history pipeline.

This preserves execution policy and result text while limiting the new code to
schema presentation, argument conversion, and pre-write edit validation.

### Rejected: independent implementations of the four tools

Copying filesystem or shell behavior would risk diverging in containment,
read-only enforcement, events, retries, timeouts, and truncation.

### Rejected: renaming tools without an adapter layer

Simple renaming cannot express Pi's offset, batch-edit, default-limit, and
timeout contracts, and would leave the native instruction text contradictory.

## Design

### Profile plumbing

Define `ToolProfile = Literal["auto", "native", "pi"]`. Add a defaulted
`EngineOpenOptions.tool_profile`, CLI `--tool-profile`, a keyword-only runtime
option, and a composition argument. Existing callers get `auto` without code
changes.

### Pi capability

Build one wrapper over the existing filesystem and shell toolsets. Its tool
map is ordered `read`, `bash`, `edit`, `write`. Exact Pi `ToolDefinition`s use
typed validators; calls are mapped back to the original names with the original
run context and tool policy.

- `read`: map offsets 0/1 to zero-based 0, other positive offsets to `offset-1`,
  and supply `limit=2000` when omitted.
- `bash`: forward an explicit timeout as `timeout_seconds`; omission uses the
  existing 30-second default.
- `edit`: resolve every `edits[{oldText,newText}]` against the original decoded
  bytes, reject ambiguous/overlapping ranges, compute one non-overlapping
  splice, and submit the original→final content as one whole-file replacement
  through the existing atomic edit path. This preserves non-incremental Pi
  semantics and existing events/hash/approval behavior.
- `write`: forward only path/content, with no optimistic-hash argument.

Do not register native filesystem, shell-job, skills, question, or
`read_tool_result` tools in Pi mode; `ToolOutputLimits` is therefore omitted.
The toolset's own 2,000-line/50,000-character bounds remain. Keep workspace,
context, compaction, events, run records, instrumentation, and notes
capabilities. Filter `edit`/`write` from tool discovery in read-only runs. Pi
definitions set no `tool_kind`, avoiding a mismatch with the reference's
`number` schema.

### Instructions and selection

`PI_CODING_AGENT_INSTRUCTIONS` names only the four Pi operations. Explicit
`system` text still takes precedence. Explicit `native` and `pi` work for every
family. `auto` resolves Hermes to Pi and all other families to native as a
development hypothesis awaiting the downstream comparison. The comparison must
select `--tool-profile native|pi` explicitly, and the dependency must be
re-pinned if the final ruling changes `auto`.

## Departures from the older plan

- Base is PR A's current head `934ee5b`, not `95d52cb` or an older pin.
- Thread through the current `EngineOpenOptions` and `AgentRuntime` APIs rather
  than the old line-number layout.
- Use installed harness 0.59.0's existing `Replacement` list and atomic edit
  implementation; no new file-edit engine.
- Keep existing ds4-py result text rather than copying C result formatting.
  Pi mode omits `ToolOutputLimits` spill/`read_tool_result` and the native
  read-file-specific edit-error enrichment to preserve exactly four tools.
- The Hermes `auto` mapping is explicitly provisional, not a measurement claim.
- The open-decisions report is absent from the downstream main branch; its
  feature-branch copy supplied no additional PR B ruling.

## Verification

Use focused red/green tests for plumbing, exact definitions, every adapter,
real workspace round trips, and selection. Run the complete static and portable
gates with empty-`HOME` isolation. Do not claim live tool-use parity.
