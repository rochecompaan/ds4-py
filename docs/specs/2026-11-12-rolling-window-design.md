# RollingWindow full-parity design

**Status:** proposed

**Parity baseline:** `pi-context-paging` v0.3.11, commit
`4409e5e4ffa0b626c11e72afe666f0b6e55a0e5f`.

The baseline's `README.md`, `docs/architecture.md`, source, and tests are the
normative behavior reference. This document specifies how that behavior maps to
ds4-py. Where the two conflict, this document wins; otherwise full behavioral
parity is required, including the later correctness and usability fixes. The
only approved exception is native image attachment, which remains M5 scope.

## Intent

Add a rolling, view-only context strategy that keeps long sessions useful
without summarizing or deleting their history. Port the complete behavior of
`pi-context-paging`, not only the earlier M4 summary. Exact ds4 token accounting
and KV binding replace Pi's estimates, while the stable-cut, recovery, privacy,
notice, persistence, command, and statistics contracts remain intact.

## Goals

- Provide `RollingWindow` as an alternative automatic `ContextStrategy` selected
  through `ContextComposition`.
- Keep canonical harness history unchanged and send a bounded request view.
- Preserve stable, forward-only cuts and append-only KV reuse between advances.
- Recover public history exactly through all four baseline tools and their
  current bounded output protocol.
- Preserve task context through the current conversation-bearing paging notice.
- Handle tool overflow, reserve pauses, interruptions, save/restore, and forks.
- Provide the baseline settings, `/context-paging` command, footer status, and
  read-only statistics with ds4-native storage and usage.
- Make the context extension conformance items executable against `FakeNative`.

## Non-goals

- A Node process, TypeScript runtime, HTTP service, or second model session.
- A second canonical history or an events-derived history.
- Porting Pi's evaluation harness, Claude bridge, npm packaging, or provider
  plugins.
- Adding branch navigation, interactive model selection, bookmarks, web tools,
  subagents, or M5 features.
- Saving sessions that contain images. The existing refusal remains.
- Native image attachment from `read_context_output(mode="image")`. Image
  metadata, handles, privacy, and search behavior are in scope; the attachment
  itself is the approved M5 exception described below.
- Bit-identical token counts across different tokenizers. Parity applies to the
  selection and recovery contracts; ds4's own exact rendered counts govern its
  budget and KV behavior.

## Approved image-scope exception

The current Python binder, renderer, provenance, transcript, and KV path reject
images pending M5. Implement all baseline image metadata and exact public part
handles now, including image-safe search, previews, JSON, and part lists. Never
serialize image bytes or base64 into text. A valid explicit
`read_context_output(mode="image")` request raises
`EngineError("image recovery requires M5 vision support")`; invalid handles and
non-image selections retain their baseline validation errors. M5 can enable the
same handle without changing the text protocol. No `BinaryContent` return or
partial vision binding is added in this milestone.

## Anti-goals

- A minimal FIFO window without stable frontiers, protected overflow, recovery
  exclusions, crash recovery, or bounded read-back.
- Recalculating eviction on every request or restoring evicted history merely
  because later input became smaller.
- Returning thinking, signatures, paging implementation chatter, or base64 image
  data through recovery.
- Writing notices or request-only recovery replacements into canonical history.
- Using character-length estimates for budget, reserve, or binding decisions.

## Constraints

- Use the existing `MessageBinder`, `PromptBuilder`, `Session`, `Session.state`,
  `FileStepStore`, event bus, command registry, and status events.
- Exactly one registered automatic context owner handles request fitting and
  reserve recovery. Manual `/compact` remains a separate concrete service.
- Register recovery tools from agent construction, including while paging is
  disabled, so enabling or disabling never changes the tool set or forces a
  prefill by itself.
- Keep public modules focused. Selection, history projection, public recovery,
  output paging, tool adaptation, settings, and statistics are separate changes.
- Use precise domain types, `TypeIs`, PEP 695 syntax, and safe lint fixes only.
- Start every behavioral change with a focused failing test and observe RED.
- Keep v1 save records readable and preserve existing LLMCompaction behavior.

## Approaches

### Behavior-first native port — selected

Port observable algorithms and protocols into focused Python modules connected
to ds4's existing ownership boundaries. Use language-neutral differential
fixtures from the pinned TypeScript baseline where message and token models
allow direct comparison. Add ds4-native integration tests for exact rendering,
KV, persistence, reserve recovery, and lifecycle.

This preserves the later fixes without duplicating binder, persistence, or
runtime responsibilities.

### Module-by-module transliteration — rejected

Copying each TypeScript function and file shape would simplify source diffs but
would import Pi message shapes, provider calibration, and lifecycle ownership
into Python. It would make integration harder and create parallel binding and
history mechanisms.

## Architecture

Add a `ds4py.context_paging` package for deterministic policy and recovery, and
one capability adapter in `ds4py.capabilities.rolling_window`.

| Module | Responsibility |
| --- | --- |
| `context_paging/model.py` | Immutable history items, public content records, cut frontiers, selection results, settings, and error types |
| `context_paging/settings.py` | Pure settings resolution, positive-integer validation, effective budget/target arithmetic |
| `context_paging/history.py` | Canonical Pydantic messages to atomic user/model-turn items; exchange validation and restore interruption classification |
| `context_paging/public_history.py` | Public recovery projection, privacy filtering, paging/nested-paging exclusion, metadata and file/tool/failure filters |
| `context_paging/selection.py` | Prefix/completed/active grouping, remembered-frontier resolution, FIFO advances, keyless fallback, protected overflow and recovery modes |
| `context_paging/notice.py` | Frozen notices, copied evicted conversation, clipping and continuation handles |
| `context_paging/navigator.py` | Cached public search corpus, stable IDs and sequences, search/browse/load |
| `context_paging/recovery_parts.py` | Baseline-compatible public JSON shapes, JSON Pointer selection, part descriptions and image metadata |
| `context_paging/recovery_response.py` | Serialized UTF-8 byte budgets, deferred loads, plain-text/JSON pages, UTF-16 offsets and continuation handles |
| `context_paging/tools.py` | Four tool functions, strict schemas, disabled refusal, Pydantic `ToolReturn`, and the explicit M5 image-mode error |
| `context_paging/stats.py` | Read-only saved-history, latest-request, and cache statistics |
| `capabilities/rolling_window.py` | Session state, exact token accounting, capability hooks, owner fallback, lifecycle, command service and status events |

The deterministic modules do not import the TUI, native worker, runtime, or
filesystem. The capability and small store adapters own those effects.

## Public history and stable identity

`HistorySource` reads the runtime's canonical `ctx.messages`; it never treats
`request_context.messages`, `events.jsonl`, or KV segments as history. Immutable
history projections reference canonical content but do not form a second stored
ledger.

Map messages as follows:

- A keyed `ModelRequest` with user content becomes one user history item. Its
  public DTO retains the baseline `/userMessage/content` shape.
- A keyed `ModelResponse` and its immediately following matching
  `ToolReturnPart`/`RetryPromptPart` results become one atomic model-turn item.
  Its public DTO retains `/assistantMessage/content/<original index>` and tool
  result paths. Omitted assistant slots preserve original indices.
- System and instruction parts belong to the resident prefix. A request envelope
  may retain active injected instructions as in the baseline.
- Tool calls and results match by `tool_call_id`. Duplicate, orphan, and
  mismatched live exchanges fail with the baseline error codes. Incomplete
  exchanges follow this table instead of one blanket failure rule:

  | Exchange | Request selection | Public recovery |
  | --- | --- | --- |
  | Newest normal incomplete exchange | Keep the protected active request | Omit until it completes |
  | Older normal incomplete exchange | Fail `INCOMPLETE_TOOL_RESULTS` | Fail `INCOMPLETE_TOOL_RESULTS` |
  | Live aborted/error exchange | Normalize it out of request candidates | Retain it with failure metadata |
  | Restore-discovered incomplete exchange | Classify it at the restore boundary | Retain it as interrupted |

- Failure is true only for an aborted/error/interrupted response, tool outcome
  `failed`, `denied`, or `interrupted`, a `RetryPromptPart` error result, or a
  restore-discovered interruption. A normal length-capped response with
  `state='incomplete'` is not by itself a failure. No synthetic tool result is
  created.
- Only restore-like boundaries (`switched` and `forked`) may classify newly
  found incomplete saved exchanges as interrupted and retain them for recovery.
  Later user messages must not make those exchanges fatal.

`metadata['ds4.key']` supplies history IDs. A model-turn ID is the response key;
a tool-call ID disambiguates a cut within a turn. Recovery sequence numbers are
projection positions, not persistent identifiers.

The public projection:

- permits user content, assistant `TextPart`, ordinary `ToolCallPart`, and
  ordinary tool results;
- omits `ThinkingPart`, signatures, provider-private values, unknown assistant
  part types, and all four paging calls/results;
- removes provider signatures from retained public calls without rewriting
  arguments or text;
- hides paging-only turns and retains ordinary work in mixed turns;
- supports complete host-level `nestedCalls` metadata only when every nested
  call is a paging call, preserving mixed or malformed wrappers;
- never recursively strips application-owned fields or text.

## Exact request accounting

Add one public observation-only binder entry:

```python
project_request_view(
    *,
    canonical_messages: Sequence[ModelMessage],
    candidate_messages: Sequence[ModelMessage],
    params: ModelRequestParameters,
) -> RequestProjection
```

`canonical_messages` supplies original identities, content hashes, and exchange
provenance. `candidate_messages` is the sole message list rendered into the
projected request. Neither list is mutated. The binder reads its already
registered `note_provider` and takes an isolated `preview()`; the caller does not
pass a second provider or apply that preview.

`RequestProjection` is immutable and contains `tokens`, `total_tokens`, the
binding `change`, `resident_tokens`, `notice_tokens`, and `units`. Each
`RequestUnitSpan` has `unit_key`, `start`, `end`, `request_keys`, and an optional
`response_key`; offsets address `tokens`. A completed exchange uses its canonical
response `ds4.key` as `unit_key`. A newest request-only unit uses its first
canonical request key. For a view-only replacement, `ds4.view_of` supplies the
canonical identity. Resident preamble and notice costs are separate totals, not
anonymous units. The projection exposes no mutable planner state.

The implementation uses the existing `plan_transition()` path. The capability's
`RequestMeter` consumes this API and must not duplicate `MessageRenderer` or
planner logic. It reuses resident system/tools/instructions, sampled response
token records, request grouping, assistant prefixes, and notes. It must not
tokenize an approximation separately from `MessageBinder`.
`Session.plan(projection.tokens)` supplies the conformance-test transition kind
and cost; RollingWindow does not need a second public `plan()` API.

Cache measurements only by immutable inputs: configuration signature, message
key and content hash, grouping role, notice content, and frontier. A selection
that fails or is abandoned publishes none of these mutations. The binder's
existing `project_request()` remains the final exact fit check before a view is
returned. `Session.ctx_used` supplies the live reserve measurement.

Pi's provider-estimate calibration is not ported. Its accounting invariants are:
removed units remove their own cost, notice cost remains included, outgoing-only
content keeps its cost, and lower later costs never reopen a committed cut.

## Selection and cut semantics

Port the baseline selector's complete state machine:

1. Validate all original exchanges before applying a remembered cut.
2. Normalize restore-interrupted exchanges without losing outgoing-only
   alignment.
3. Group prefix units, completed turns, the protected active request, active
   exchanges, and trailing outgoing content.
4. Resolve a remembered frontier in raw canonical history and validated groups.
5. Keep the existing frozen notice and view until exact projected tokens exceed
   the effective budget. Equality, or a value between target and budget, does
   not advance.
6. Advance FIFO toward the effective target through prefix units, completed
   turns, then eligible active-turn exchanges.
7. Protect the active request and newest valid unread tool-result exchange.
   Interrupted exchanges do not gain unread-result protection.
8. Preserve keyed partial-turn remainders and later evict each completed
   remainder atomically.
9. If eviction ends at keyless content, snap back to the last safe keyed
   boundary. Never move backward past an already committed frontier.
10. On unusable provenance, use stateless FIFO toward the budget and report the
    baseline fallback reason without committing a frontier.

`token_budget` defaults to `128_000`; implicit `trim_to` is
`max(1, token_budget * 5 // 8)`. Effective budget is capped by the engine context
and the target scales with exact integer arithmetic. Explicit targets at or
above budget retain a stable cut but trim to the effective budget and produce
one warning per resolved pair per runtime process.

Selection errors retain the baseline codes: invalid settings/model context,
resident input too large, active request too large, and invalid message
structure. A failed selection commits no cut and leaves canonical history
untouched. Add `ContextSelectionError(EngineError)` with a structured `code`
field and the baseline diagnostic. `AgentRuntime.run` is the before-request
error boundary: it records the failed run and `run_end`, then re-raises the same
exception. The interactive and non-interactive frontends render its structured
code and diagnostic through the normal `EngineError` path; neither matches on
message text nor converts the failure into a successful empty result.
A reserve failure raises `ContextExhausted` with the selection error as its
cause, preserving the code without changing the existing exception's public
shape. The tool fallback retains the exact text `Tool error: tool result still
does not fit`; it does not append a code to the model-visible result.

Protected-overflow first allows the full unread exchange above the normal
rolling budget when it fits the model context. If it cannot fit, replace
recoverable oversized results with exact read handles and paging results with
exclusion notices. Preserve live call/result pairing. Normal frozen-notice
behavior resumes without a fabricated frontier advance.

## Paging notice

The notice is a transient user request inserted after the resident system prefix
and before retained canonical messages. It has a dedicated metadata marker and
no canonical-history ownership.

Port the current wording and behavior, including:

- no-summary explanation and recovery guidance;
- stable recent history and ordinary tool-result handles;
- every fully evicted user request and final public reply from the newest three
  evicted turns;
- no conversation contribution from prefix or partial-turn eviction;
- newest-first selection displayed oldest-first;
- 6,000 UTF-16-code-unit entry clips, 500-code-unit minimum useful clip,
  32,000-code-unit section cap, and the smaller room left by the effective
  budget and protected floor;
- exact `read_context_output` handles for clipped text;
- notice cost in both trigger and destination;
- frozen content between advances, with bounded resizing when a remembered
  notice no longer fits.

## Capability, fallback, and reserve recovery

Keep `ContextComposition.owner` as the selection seam. Its RollingWindow form is:

```python
ContextComposition(
    owner=RollingWindow(
        model.binder,
        enabled=None,          # None resolves settings; a bool overrides them
        token_budget=None,     # positive int overrides settings
        trim_to=None,          # positive int or None for the adaptive default
    )
)
```

The constructor needs no compactor. `build_agent_components` detects a
RollingWindow owner and injects the concrete `LLMCompaction` through an internal
`set_disabled_owner()` composition step before the agent is built. No factory or
second `ContextComposition.strategy` field is added. The shared agent CLI parser
used by the interactive TUI and `--non-interactive` exposes
`--context-strategy compaction|rolling`, `--context-token-budget`, and
`--context-trim-to`; `raw` is not an agent path and gains none of these flags.

`RollingWindow` is the one registered owner when selected. Its session state
holds the enabled override, frontier/frozen notice, restore-interrupted IDs,
navigator cache identity, latest input, and accounting state. Mutable session
data does not live in a per-run copy.

At composition, `build_agent_components` gives it the existing concrete
`LLMCompaction` as its disabled-state delegate:

- enabled: RollingWindow handles request selection, tool fit, and reserve
  recovery; automatic compaction does not run;
- disabled: the delegate handles those hooks with existing LLMCompaction
  behavior; recovery tools refuse execution;
- manual `/compact`: always calls the concrete compactor directly.

This preserves one owner and Pi's "off restores ordinary compaction" behavior
without registering two competing capabilities.

For view-only reserve recovery, advance the cut in the actual request-message
view, rebind through the existing retained reserve operations, and continue the
same `ModelResponse`. Do not append the view to canonical history. If no legal
cut can satisfy the reserve, raise `ContextExhausted`.

RollingWindow overrides the inherited pending-tool-fit path. It never calls
`_reject_until_fit` on canonical `ctx.messages`. Instead it builds the candidate
request view, runs the same selector, and verifies exact fit with
`project_request_view`. To reject an oversized trailing ordinary result, clone
its containing request with a fresh view-only `ds4.key`, set `ds4.view_of` to the
canonical request key, and change only the clone's part to `Tool error: tool
result still does not fit`. Repeat the baseline's minimal trailing-rejection
loop on view-only clones. If every allowed replacement still overflows, raise
`ContextExhausted`.

`request_context.messages` carries the selected view to the model binder while
`ctx.messages` remains canonical. Runtime adoption retains canonical request
history plus the new response and discards all marked view-only request clones
and notices. Save-record `request_view` serialization is the only place those
transient request messages persist.

## Lifecycle, settings, command, and status

`RollingWindowSettings` resolves `enabled`, `tokenBudget`, and `trimToTokens`
from the baseline `contextPaging` JSON object. Supported locations are:

1. trusted project `<project>/.ds4py/settings.json`;
2. global `<SessionStore.root>/settings.json` (the default root resolves
   `DS4PY_HOME`, then `~/.ds4py`);
3. programmatic RollingWindow constructor values, which are explicit ds4
   overrides and take highest precedence when supplied.

Following the explicit store root keeps tests and benchmarks isolated from the
developer's real global settings.

`AgentRuntime` and the shared interactive/non-interactive agent CLI parser gain a
`project_trusted` flag, exposed as `--trust-project-settings`. It defaults to
false. Project settings are ignored unless trusted; the flag grants no other
permission. Values accept only real booleans and positive integers. Invalid
project values fall through to global, then defaults. Direct policy inputs raise
instead of falling back.

Match the baseline failure behavior: a settings read/parse failure first
publishes `paging off` and emits a `context_paging_error` diagnostic, then
re-raises the settings error. Runtime/CLI startup aborts; it does not silently
continue with compaction. An enabled-state setting that was loaded successfully
never remains half-applied.

Install `/context-paging [on|off|status|stats]` only when RollingWindow is the
selected owner:

- no argument and `status` report state without resetting it;
- an actual `on`/`off` change resets cut and accounting and publishes status;
- repeated choices and `stats` preserve cut and accounting;
- invalid arguments return usage without state changes;
- `stats` is read-only and makes no native/provider call;
- command execution is idle-only under the existing registry admission rules.

`install_builtin_commands` inspects `runtime.model.context_owner`. When it is
RollingWindow, it registers a thin idle-only handler that calls
`runtime.context_paging(action)`; the runtime delegates to the capability. The
capability still has no TUI import.

Publish exactly `paging on` or `paging off` under the `context-paging` status
key. Clear it on every `session_end` (including `/new`, `/switch`, and close),
then publish the new state after successful session initialization. Trim-target
warnings use a separate `context_paging_warning` event with a `message` payload;
the interactive and non-interactive frontends render it like other warnings.
Warning deduplication remains capability-owned.

Lifecycle behavior:

- `/new` clears override, cut, accounting, navigation, and restore IDs;
- `/switch` clears override/accounting, classifies restored interruptions, and
  derives the cut from the request view bound in the save record;
- `forked` retains canonical keys but resets override/accounting and performs
  the same restore classification and bound-view cut derivation from the
  binder's current request view, preserving KV;
- successful manual compaction resets cut and accounting but not the enabled
  override; canceled attempts change nothing;
- `model_changed`, if emitted by an embedding application, resets cut and
  accounting but preserves override and warning history;
- events and saved-entry notifications are idempotent so one transition cannot
  reset a newly established frontier twice.

No branch-navigation API is added. Future branch events can use the same
restore-classification seam.

## Save-record and restore contract

A paged native transcript can contain a transient notice or request-only tool
result replacements. The existing v1 assumption that every bound message exists
in the canonical snapshot is therefore insufficient.

Introduce save format v2 with an optional `request_view` field containing the
complete serialized model request view used to produce the KV payload. Serialize
it with Pydantic AI's model-message type adapter in JSON mode, preserving all
parts and metadata. Resident tool definitions and instructions are not
duplicated into this field: the existing current-preamble check decides whether
case 1 remains valid, and a changed preamble takes case 2.

The canonical complete snapshot remains unchanged and authoritative.
`request_view` is binding provenance, not a second history and never feeds
recovery or stats. V1 records omit the field and retain their current meaning.
V2 compaction saves may omit it; RollingWindow saves include it whenever the
bound view differs from canonical history.

Unchanged view messages retain their canonical `ds4.key`. Every synthetic or
content-replaced view message receives a fresh view-only `ds4.key` plus:

- `metadata['ds4.context_paging']`, identifying `notice` or `recovery`;
- `metadata['ds4.view_of']` for a replacement, naming its canonical message key.

This prevents `MessageBinder.adopt()` from replacing canonical content with a
recovery substitution. Notices have no `view_of`. Canonical cut identities never
use these synthetic view keys.

Save order remains KV -> complete canonical snapshot -> save record. Saving a
RollingWindow session writes:

- the full canonical `_messages` snapshot, including paged-out content;
- `request_view`, including transient notices/replacements when present;
- token segments and hashes for exactly that request view.

On case-1 restore, validate and bind `request_view` to the payload, then adopt
the full canonical snapshot independently in the runtime. Every keyed segment
must resolve exactly once in `request_view` with a matching content hash. Every
unmarked view message must also occur in the canonical snapshot with the same
key/hash; every view-only key must carry a valid paging marker and, for a
replacement, an existing `view_of` target. The canonical snapshot may contain
arbitrary additional paged-out messages. The view need not contain every
canonical message.

On case 2, ignore `request_view` and fresh-render full canonical history.
Missing or malformed view provenance fails before live state changes. The
`StoreError` names the session directory, record run/step, and the missing or
mismatched view key, canonical target, or snapshot as applicable.

RollingWindow reconstructs its frontier from the first canonical message bound
in the loaded view and retains the loaded notice; it does not persist a separate
frontier. The first follow-up extends the loaded KV. A valid view with omitted
canonical messages must contain its paging marker. With no `request_view` (v1),
a full equal view, a fork before the first request, or a post-compaction reset,
start with an empty frontier and let the next exact budget check decide whether
a cut is needed. This preserves a full loaded view without an unnecessary
prefill.

Case 1 binds `request_view` only when the selected owner is an enabled
RollingWindow. A disabled RollingWindow, LLMCompaction, and every other owner
take case 2 and fresh-render full canonical history. In particular, the disabled
delegate never receives a paging notice or view-only replacement as compaction
history.

Image-containing sessions still refuse save. Request-view serialization adds no
image-storage exception.

## Recovery tools and wire protocol

Implement the baseline tool names and strict schemas exactly:

- `search_history(query, files?, tools?, failed?, limit=5, load=false)`;
- `browse_history(historyId?, sequence?, direction, count=5, stride=1)`;
- `load_history(historyIds[1..3])`;
- `read_context_output(historyId, source?, part?, mode?, contentIndex?,
  toolCallId?, offset?, limit?)`.

Keep the baseline maxima, defaults, optional-selector rules, camelCase JSON,
stable references, previews, filters, errors, and result ordering. Public DTOs
use the baseline paths and field names rather than exposing Pydantic's internal
serialization layout.

The complete serialized text reply is at most 16,384 UTF-8 bytes. Part
descriptions are at most 4,096 bytes. Load at most three items, smallest first
for fitting while retaining requested output order. Large items defer to bounded
part descriptions and read handles; one large item cannot block a smaller one.

`read_context_output` retains:

- `text`, `parts`, and `image` modes in the public schema;
- legacy `assistant` and `toolResult` selectors with unique-choice defaults;
- public JSON Pointer handles and bounded choice/error lists;
- a JSON header block followed by an unescaped page block;
- `text` pages for strings/plain text blocks and exact JSON for other values;
- offsets and limits in UTF-16 code units, independent of Python string indices;
- continuation arguments and no split surrogate pairs;
- the approved M5 error for a valid explicit image-mode request.

Use an explicit UTF-16 codec/index adapter; Python code-point slicing is not a
compatible substitute. Serialize replies through one byte-budget function that
counts content, metadata, escaping, and continuation fields. Do not duplicate
payloads into a `details` field.

Pydantic adaptation returns model-visible text blocks. Text serialization
exposes only image metadata and handles. Binary data never enters JSON,
previews, search, or ordinary text reads. Image mode follows the approved M5
exception above rather than returning a partial or text-encoded attachment.

All four tools refuse with `Context paging recovery tools are disabled.` while
paging is off. They read canonical history at execution time and invalidate the
navigator only when visible stable IDs change.

## Statistics

Preserve the baseline report sections and unavailable/zero semantics.

- **Stored size:** total physical bytes of regular files under the live
  session's `steps/` directory, excluding each run's `events.jsonl` telemetry.
  Include `run.json`, `snapshot-keys.jsonl`, snapshots, tool-effect records, and
  externalized media. Count each content-addressed media file once. KV, save
  records, and events are excluded because Pi's corresponding value measures
  history storage, not KV or telemetry.
- **History tokens:** an explicitly estimated (`~`) whole-session value over all
  saved runs/snapshots, deduplicating repeated snapshots by stable message key
  and content hash. Distinct revisions and summaries remain distinct. Port the
  baseline's pure estimate: sum UTF-16 code units and ceiling-divide by four;
  count user/text/tool-result content, thinking, and tool name plus JSON
  arguments; use 4,800 units per image. This estimate runs without an engine,
  is for display only, and never enters selection.
- **Input tokens:** latest exact selected request total, then the latest valid
  saved response's Pydantic `usage.input_tokens` after restore. That field
  already includes cache-read and cache-write tokens, so do not add the cache
  counters again. Output is excluded. Mark only estimated fallback values with
  `~`.
- **Paging budget/model window:** effective values and current percentage;
  disabled shows `inactive`.
- **Cache totals:** sum `cacheRead` and `cacheWrite` once per stable saved
  response/tool usage record across all runs. Any missing/invalid required
  counter makes that total unavailable; explicit measured zero remains zero.

Reading snapshots must tolerate the step store's documented damaged-file
behavior. Stats does not mutate navigation, cut state, accounting, settings,
status, native state, or canonical history.

## Verification and parity evidence

### Differential contract fixtures

Add `scripts/generate_context_paging_fixtures.mjs` as a committed, manually run
generator. It accepts the baseline checkout path, verifies commit `4409e5e`, and
uses that checkout's own Node dependencies. It needs no ds4 `package.json`,
Python, runtime, or CI dependency. Commit selected language-neutral expected
JSON under `tests/fixtures/context_paging/` with a manifest naming the baseline
and generator command. The generator is reproducibility tooling, not a second
test runner or a production dependency.

Cover:

- settings precedence and integer boundaries;
- exchange projection and public privacy/exclusion shapes;
- stable, partial, keyless, fallback, and protected selection frontiers using
  injected additive token costs;
- notice conversation selection, clipping, and handles;
- search/browse/load ordering and filters;
- reply byte limits, deferred parts, UTF-16 pages, selectors, image metadata,
  and the explicit M5 image-mode error.

Where Pi estimates and ds4 exact rendering cannot share numeric inputs, compare
structure and decisions, then pin ds4's exact totals separately against
`FakeNative`.

### ds4 integration and regression tests

Use real capabilities and `FakeNative` to prove:

- full history remains canonical while request views page;
- exact budget/notice counts and stable KV prefixes between cuts;
- cut restore/rebuild costs match the following sync plan;
- tool overflow and mid-response reserve recovery continue correctly, including
  byte-for-byte unchanged canonical `ctx.messages` after a rejected tool result;
- disabled fallback uses LLMCompaction and manual compaction remains available;
- paging calls/results, thinking, signatures, and nested paging-only wrappers
  never leak through recovery;
- restored incomplete exchanges page safely while malformed live exchanges fail;
- `/new`, `/switch`, fork, compaction, and repeated events follow lifecycle
  rules without losing keys or duplicating resets;
- v1 records still restore, v2 records preserve full history plus exact view,
  bad view provenance is atomic, and disabled-Rolling/non-Rolling restores take
  case 2 without feeding transient messages to compaction;
- all conformance items 1–9 pass, with RollingWindow satisfying items 1–5;
- command/status/stats behavior and disabled tool refusal;
- no prefill on exact view restore and no notice leakage into canonical history.

Run focused RED/GREEN cycles, then:

```bash
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
uv run --frozen --extra agent pytest -m "not build and not live" -q
```

The M4 live criterion remains a long RollingWindow session within budget. Do not
claim native/live parity from portable tests.

## Builder choices

The builder may choose private helper names, concrete dataclass layouts, cache
key encoding, async filesystem offload, and the smallest Pydantic `ToolReturn`
adapter that preserves the wire contract. These choices must not alter the
public behavior, ownership boundaries, save format, or verification requirements
above.
