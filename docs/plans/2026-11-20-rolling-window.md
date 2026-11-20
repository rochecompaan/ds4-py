# RollingWindow Full-Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the complete view-only RollingWindow context strategy, including stable cuts, exact ds4 token/KV accounting, public history recovery, lifecycle and settings, v2 request-view persistence, statistics, and native/live verification.

**Architecture:** Keep deterministic policy and recovery in `ds4py.context_paging`; keep native, runtime, filesystem, command, and status effects in one `RollingWindow` capability and small existing-runtime adapters. Canonical harness history remains authoritative. A selected request view is measured and bound through `MessageBinder`, while transient notices and recovery replacements never enter canonical history.

**Tech Stack:** Python 3.14, Pydantic AI/Pydantic AI Harness, existing `MessageBinder`/`Session`/`FileStepStore`, `FakeNative`, pytest, devenv and uv. Node is used only by the manually run baseline fixture generator. CUDA inference and live replay run on the separately provisioned Salad host.

**Spec:** [`docs/specs/2026-11-12-rolling-window-design.md`](../specs/2026-11-12-rolling-window-design.md). The parity baseline is `pi-context-paging` commit `4409e5e4ffa0b626c11e72afe666f0b6e55a0e5f`; later upstream changes are not normative.

**Base:** Start at `7d80c7c` on `feat/rolling-window` in `/home/roche/projects/satyrn/ds4-py/.worktrees/rolling-window`. Do not edit the main checkout.

## Global Constraints

- Preserve full behavioral parity with pinned `pi-context-paging` commit `4409e5e4ffa0b626c11e72afe666f0b6e55a0e5f`, except native image attachment, which raises `EngineError("image recovery requires M5 vision support")` for a valid explicit image-mode read.
- Keep canonical `ctx.messages` complete and unchanged by paging. Do not derive history from events, KV segments, or request-only replacements.
- Use exact binder-rendered token counts and KV transitions for request selection. Character-length estimates are allowed only for display-only history-token statistics and the baseline notice conversation cap.
- Keep exactly one automatic context owner. RollingWindow delegates to the concrete `LLMCompaction` only while disabled; manual `/compact` always uses that concrete compactor.
- Register all four recovery tools whenever RollingWindow is constructed, including while disabled. Enabling or disabling changes policy, not the tool set or native context size.
- Use the baseline defaults: `token_budget = 128_000`, implicit `trim_to = max(1, token_budget * 5 // 8)`, and settings-enabled defaults to `True`. Accept only real booleans and integers in `1 <= value <= 2**53 - 1` for settings integers.
- Keep the public recovery protocol at or below 16,384 UTF-8 bytes, part descriptions at or below 4,096 bytes, and text offsets/limits in UTF-16 code units. Never expose thinking, signatures, paging implementation chatter, image bytes, or base64 through public recovery.
- Keep v1 save records readable. The canonical snapshot remains authoritative; v2 `request_view` is binding provenance only.
- Do not add a Node production dependency, a second model session/history, native image attachment, branch navigation, or a command-level `/strip` implementation.
- Keep modules focused. New pure modules follow the boundaries in the spec; `RollingWindow` remains the only new registered capability, with session-lifecycle helpers kept out of its public interface.
- Use `TypeIs`, precise domain types, PEP 695 syntax, and safe lint fixes. Do not add a second Python environment or hook runner.
- Start every behavioral change with a focused failing test and observe RED. Keep formatting and behavioral changes in separate commits.
- Run commands from this worktree inside `devenv shell`. Run the complete static gate and portable suite before delivery.
- Local verification has no GPU. Do not claim native inference or live replay parity from `FakeNative`; collect that evidence on the provisioned Salad CUDA host.

## Review Focus

- **View-scoped restore provenance:** a valid paged save can omit canonical messages and contain synthetic keys. Task 9 pins atomic case-1 validation, independent canonical validation, and case-2 fallback for disabled/non-Rolling owners.
- **Small effective budgets:** notices must not consume the protected active request's floor or silently exceed the model window. Task 5 pins the 500-code-unit useful clip, the model-capped 10% conversation allowance, and bounded notice resizing.
- **Unicode paging boundaries:** Python code-point slicing can split the baseline's UTF-16 positions, and character counts do not bound UTF-8 replies. Task 6 pins astral characters, unpaired-surrogate avoidance, exact byte limits, and continuation arguments.
- **Settings and lifecycle failures:** a parse error or repeated event must not leave half-applied settings or erase a newer frontier. Task 10 pins status-then-raise startup failure, atomic settings resolution, event idempotence, and warning deduplication.
- **Damaged or old usage records:** aggregate multi-sync usage is not one exact request input. Task 11 pins `ds4.request_sync_count`, exact input metadata, unavailable fallbacks, and damaged snapshot behavior.
- **Note-driven projection drift:** hints, dates, and restored notes can change exact request tokens without changing message hashes. Tasks 4 and 8 pin isolated previews and cache invalidation across every emitted-note input.

## File Structure

- `scripts/generate_context_paging_fixtures.mjs` and `tests/fixtures/context_paging/` provide pinned, language-neutral differential evidence; Node is not a production or CI dependency.
- `src/ds4py/context_paging/model.py` defines immutable history, public DTO, frontier, selection, notice, and settings records.
- `src/ds4py/context_paging/settings.py` resolves trusted project, global, and programmatic settings.
- `src/ds4py/context_paging/history.py` validates exchanges and maps canonical messages to atomic history items; it also validates restored views and derives their frontiers.
- `src/ds4py/context_paging/public_history.py` projects safe public history and applies paging/privacy/nested-wrapper exclusions.
- `src/ds4py/binding_projection.py` defines exact request projections and unit spans without expanding the binder facade's responsibilities.
- `src/ds4py/context_paging/selection.py` resolves remembered frontiers, advances stable cuts, and handles protected overflow/fallback modes.
- `src/ds4py/context_paging/notice.py` builds, freezes, clips, and resizes transient paging notices.
- `src/ds4py/context_paging/recovery_parts.py` implements public JSON shapes, JSON Pointer selection, and image metadata.
- `src/ds4py/context_paging/recovery_response.py` implements UTF-16 pages and complete UTF-8 byte-budget serialization.
- `src/ds4py/context_paging/navigator.py` owns cached public search/browse/load views and stable references.
- `src/ds4py/context_paging/tools.py` adapts the four strict tool schemas to model-visible returns and disabled/M5 errors.
- `src/ds4py/context_paging/stats.py` computes read-only storage, history, request, budget, and cache reports.
- `src/ds4py/capabilities/rolling_window.py` is the sole new context owner and toolset provider.
- `src/ds4py/capabilities/rolling_window_session.py` contains session-scoped lifecycle/status/settings helpers, not a second capability.
- Existing `binding.py`, `model.py`, `runtime.py`, `runtime_restore.py`, `store.py`, `agent.py`, `agent_cli.py`, `tui_commands.py`, `tui.py`, and `noninteractive.py` remain integration facades rather than receiving duplicate policy implementations.

---

### Task 1: Generate the pinned differential contract fixtures

**Files:**
- Create: `scripts/generate_context_paging_fixtures.mjs`
- Create: `tests/fixtures/context_paging/manifest.json`
- Create: selected JSON fixtures under `tests/fixtures/context_paging/`
- Add: `docs/plans/2026-11-20-rolling-window.md`

**Interfaces:**
- Produces: `node scripts/generate_context_paging_fixtures.mjs <baseline-checkout>`; the command refuses any baseline whose `HEAD` is not `4409e5e4ffa0b626c11e72afe666f0b6e55a0e5f`.
- Produces: a manifest naming the full baseline commit, generator command, fixture files, and source test references.
- Consumes: the baseline checkout's own Node dependencies; no ds4 `package.json` or Python runtime.

- [ ] **Step 1: Prepare the exact baseline and specify fixture cases**

Create a detached baseline worktree outside the repository:

```bash
git -C /home/roche/projects/pi/extensions/pi-context-paging worktree add --detach /tmp/pi-context-paging-4409e5e 4409e5e4ffa0b626c11e72afe666f0b6e55a0e5f
```

Install the detached checkout's own locked Node dependencies with `npm ci --prefix /tmp/pi-context-paging-4409e5e`; do not add Node dependencies to ds4-py. Enumerate fixtures for settings precedence and `2**53 - 1`/`2**53`; exchange/privacy/exclusion shapes; stable/partial/keyless/fallback/protected cuts with additive costs; notice selection/clipping/low-budget caps; navigator ordering and filters; bounded replies/deferred parts/UTF-16 selectors; image metadata, unconditional image-item load deferral, and the M5 error.

- [ ] **Step 2: Write the commit guard and verify RED on the wrong baseline**

Create the generator entry point with argument parsing and the full-commit guard, but no fixture generation yet. Run it against `/home/roche/projects/pi/extensions/pi-context-paging`, which is not at the pinned commit.

Expected: a nonzero exit naming the expected and actual commits, with no fixture changes. This is the generator's direct behavioral RED check; absence of the script is not accepted as the failure.

- [ ] **Step 3: Implement and run the generator**

Use small Node entry points that import the pinned baseline's own modules and serialize language-neutral inputs/expected outputs. Do not port implementation logic into the generator. Write deterministic JSON with sorted fixture IDs and stable formatting.

Run:

```bash
node scripts/generate_context_paging_fixtures.mjs /tmp/pi-context-paging-4409e5e
node scripts/generate_context_paging_fixtures.mjs /tmp/pi-context-paging-4409e5e
git diff --exit-code -- tests/fixtures/context_paging
```

Expected: both pinned runs succeed; the second produces no diff.

- [ ] **Step 4: Check fixture coverage directly**

Compare every bullet in the spec's “Differential contract fixtures” section with a manifest entry. Validate all JSON files with the repository's JSON hook or `python -m json.tool`. No new pytest suite is needed for manifest text or one-off script structure; the meaningful verification is executing the generator and consuming fixtures in later behavioral tests.

- [ ] **Step 5: Commit the plan and reproducibility inputs**

```bash
git add docs/plans/2026-11-20-rolling-window.md scripts/generate_context_paging_fixtures.mjs tests/fixtures/context_paging
git commit -m "test: add pinned context paging fixtures"
```

---

### Task 2: Add immutable domain records and settings resolution

**Files:**
- Create: `src/ds4py/context_paging/__init__.py`
- Create: `src/ds4py/context_paging/model.py`
- Create: `src/ds4py/context_paging/settings.py`
- Create: `tests/context_paging/test_settings.py`

**Interfaces:**
- Produces: immutable `RollingWindowSettings(enabled: bool, token_budget: int, trim_to: int, trim_to_explicit: bool)` plus the history/public/frontier records consumed by later tasks.
- Produces: `default_trim_to(token_budget: int) -> int` using exact integer arithmetic.
- Produces: `resolve_settings(*, project_dir: Path, store_root: Path, project_trusted: bool, enabled: bool | None = None, token_budget: int | None = None, trim_to: int | None = None) -> RollingWindowSettings`.
- Precedence: valid trusted project `<project>/.ds4py/settings.json`, then global `<store_root>/settings.json`, then defaults; explicit constructor values override the corresponding resolved value. `trim_to=None` means adaptive when no explicit target was supplied.

- [ ] **Step 1: Write failing settings tests**

Consume the settings fixtures and add literal boundary tests. Assert:

1. defaults are `True`, `128_000`, and `80_000` with `trim_to_explicit=False`;
2. trusted project values beat global values, untrusted project values are ignored, and each programmatic override wins independently;
3. invalid project values fall through to global/default values, while direct integer fields reject booleans, floats, zero, negatives, and `2**53`, and `enabled` rejects non-booleans;
4. `2**53 - 1` is accepted and adaptive multiplication does not overflow or round through a float;
5. an explicitly supplied target retains `trim_to_explicit=True`; an adaptive target does not;
6. malformed/unreadable files raise the original settings error rather than silently selecting compaction.

- [ ] **Step 2: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_settings.py -q
```

Expected: FAIL because the package, records, and resolver do not exist.

- [ ] **Step 3: Implement records and settings resolution**

Validate dynamic JSON at the file boundary. Distinguish booleans from integers despite Python's `bool` subclassing `int`. Read the global file under the explicit store root so tests never use the developer's real home. Keep file I/O and diagnostics separate: this task raises; Task 10 adds status/event publication.

- [ ] **Step 4: Run focused tests**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_settings.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/context_paging tests/context_paging/test_settings.py
git commit -m "feat: add context paging settings"
```

---

### Task 3: Project canonical history into safe public items

**Files:**
- Create: `src/ds4py/context_paging/history.py`
- Create: `src/ds4py/context_paging/public_history.py`
- Create: `tests/context_paging/test_history.py`
- Create: `tests/context_paging/test_public_history.py`

**Interfaces:**
- Produces: `project_history(messages: Sequence[ModelMessage], *, restore_boundary: Literal["switched", "forked"] | None = None) -> HistoryProjection`.
- Produces: `project_public_history(history: HistoryProjection) -> tuple[PublicHistoryItem, ...]`.
- `HistoryProjection` retains canonical keys, atomic user/model-turn items, prefix/completed/active roles, exchange provenance, and original assistant part indices. Public sequence numbers are projection positions, not persistent IDs.

- [ ] **Step 1: Write failing exchange and privacy tests**

Use the generated fixtures plus direct Pydantic message fixtures. Assert:

1. keyed user requests and response/tool-result exchanges retain baseline DTO paths and original assistant indices;
2. newest normal incomplete exchanges remain protected for selection but are omitted publicly; older incomplete exchanges fail `INCOMPLETE_TOOL_RESULTS`;
3. live abort/error exchanges normalize out of request candidates but remain public with failure metadata; only `switched`/`forked` restore boundaries classify saved incomplete exchanges as interrupted;
4. denied/interrupted tools, failed retry prompts, and restore interruptions are failures, while a normal length-capped response with `state='incomplete'` is not;
5. thinking, signatures, provider-private values, unknown assistant parts, and all four paging calls/results never appear;
6. paging-only turns disappear, mixed turns retain ordinary work, and `nestedCalls` is read per result from `ToolReturnPart.metadata` only;
7. a nested wrapper is hidden only when every nested call is paging; mixed, malformed, and absent wrappers preserve ordinary public output;
8. duplicate, orphan, and mismatched live exchanges fail atomically without creating synthetic results.

- [ ] **Step 2: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_history.py tests/context_paging/test_public_history.py -q
```

Expected: FAIL because canonical/public projection does not exist.

- [ ] **Step 3: Implement canonical and public projection**

Match calls/results by `tool_call_id`; validate all original exchanges before any remembered cut is applied. Keep canonical content immutable and use projections/references rather than a second stored ledger. Remove provider signatures only from retained public calls; never recursively strip application-owned fields or text.

- [ ] **Step 4: Run focused tests**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_history.py tests/context_paging/test_public_history.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/context_paging/history.py src/ds4py/context_paging/public_history.py src/ds4py/context_paging/model.py tests/context_paging/test_history.py tests/context_paging/test_public_history.py
git commit -m "feat: project public context history"
```

---

### Task 4: Expose exact observation-only request projection

**Files:**
- Create: `src/ds4py/binding_projection.py`
- Modify: `src/ds4py/binding.py` (`MessageBinder.project_request_view`)
- Create: `tests/agent/test_request_view_projection.py`
- Modify: `tests/agent/test_binding_projection.py`
- Modify: `tests/agent/test_system_notes.py`

**Interfaces:**
- Produces: immutable `RequestUnitSpan(unit_key: str, start: int, end: int, request_keys: tuple[str, ...], response_key: str | None)`.
- Produces: immutable `RequestProjection(tokens: tuple[int, ...], total_tokens: int, change: BindingChange, resident_tokens: int, notice_tokens: int, units: tuple[RequestUnitSpan, ...])`, with `total_tokens = len(tokens)` and assistant-prefix tokens included only in the total.
- Produces: `MessageBinder.project_request_view(*, canonical_messages: Sequence[ModelMessage], candidate_messages: Sequence[ModelMessage], params: ModelRequestParameters) -> RequestProjection`.
- The candidate list alone is rendered. Canonical messages supply original identities, hashes, `ds4.view_of`, and exchange provenance. Neither input list nor the live note provider is mutated.

- [ ] **Step 1: Write failing exact-projection tests**

Assert:

1. projected tokens equal the following successful `Session.plan(projection.tokens)`/prepared request tokens, including the assistant prefix;
2. completed units use canonical response keys, newest request-only units use the first canonical request key, and replacements map through `ds4.view_of`;
3. resident preamble and marked notice costs are separate totals, not anonymous history units;
4. removed units remove their own cost, notice cost remains included, and outgoing-only content retains its cost;
5. changing candidate history changes only the affected unit spans and `BindingChange`;
6. hints on/off, pending/restored note content, note state, and clock-derived values change the isolated preview and resulting measurement without applying that preview to the live provider;
7. failed projection publishes no message keys, notes, planner state, or native changes.

- [ ] **Step 2: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/agent/test_request_view_projection.py tests/agent/test_binding_projection.py tests/agent/test_system_notes.py -q
```

Expected: FAIL because the view projection API and span records do not exist.

- [ ] **Step 3: Implement the projection through the existing planner**

Reuse `plan_transition()`, current registered `note_provider.preview()`, request grouping, sampled token records, assistant prefixes, and effective instructions. Do not tokenize a second approximation or call the mutating prepare path. Derive spans from planned boundaries and canonical provenance. Keep projection caches out of the binder; Task 8's meter keys any cache by immutable configuration/content/frontier/note-preview inputs.

- [ ] **Step 4: Run focused and neighboring binder tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_request_view_projection.py tests/agent/test_binding_projection.py tests/agent/test_system_notes.py tests/agent/test_binding.py tests/engine/test_binding_transaction.py -q
```

Expected: PASS with no change to existing `project_request()` behavior.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/binding_projection.py src/ds4py/binding.py tests/agent/test_request_view_projection.py tests/agent/test_binding_projection.py tests/agent/test_system_notes.py
git commit -m "feat: project exact request views"
```

---

### Task 5: Implement stable selection and paging notices

**Files:**
- Modify: `src/ds4py/errors.py` (`ContextSelectionError`)
- Create: `src/ds4py/context_paging/selection.py`
- Create: `src/ds4py/context_paging/notice.py`
- Create: `tests/context_paging/test_selection.py`
- Create: `tests/context_paging/test_notice.py`

**Interfaces:**
- Produces: `ContextSelectionError(EngineError)` with structured `code`; `str(error)` is `[CODE] diagnostic`.
- Produces: a `SelectionMeter` protocol implemented by Task 8's exact request meter and by additive fake costs in differential tests.
- Produces: `select_view(history: HistoryProjection, *, settings: RollingWindowSettings, model_context: int, meter: SelectionMeter, remembered: SelectionState | None = None, overflow: OverflowMode = "normal") -> SelectionResult`.
- Produces: frozen `PagingNotice` records and bounded resizing for a remembered notice. Selection results contain candidate messages, exact measurement, resolved effective budget/target, optional frontier/notice, and fallback/warning diagnostics.
- Selection failure commits no frontier or notice and never mutates canonical history.

- [ ] **Step 1: Write failing selector tests**

Use additive-cost fixtures and literal state transitions to assert:

1. exact equality with budget and values between target and budget retain the frozen view; only `>` budget advances;
2. FIFO advances through prefix units, completed turns, then eligible active exchanges, while protecting the active request and newest valid unread result;
3. whole-turn and keyed partial remainders are atomic, keyless cuts snap back to a safe keyed boundary, and later smaller inputs never reopen a committed cut;
4. unusable provenance uses stateless FIFO with the baseline fallback reason and no committed frontier;
5. invalid model context/settings, oversized resident input, oversized active request, and malformed structure preserve their baseline codes and diagnostics;
6. protected unread overflow fits above normal budget when possible, otherwise selects recoverable-result replacement/exclusion mode without fabricating a frontier advance;
7. explicit targets at/above budget trim to effective budget and emit one warning per resolved pair per runtime process.

- [ ] **Step 2: Write failing notice tests**

Assert newest-three evicted-turn selection, newest-first selection displayed oldest-first, full user requests/final public replies only, no prefix/partial-turn conversation, exact continuation handles, 6,000-unit clips, the 500-unit minimum useful clip, and `conversationCharacterLimit = min(32_000, floor(effective_budget_tokens * 0.4))` followed by the remaining-room cap. Include configured budget above model window, frozen content between advances, and bounded resizing.

- [ ] **Step 3: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_selection.py tests/context_paging/test_notice.py -q
```

Expected: FAIL because selection/frontiers/notices do not exist.

- [ ] **Step 4: Implement selection and notice construction**

Resolve remembered frontiers against raw canonical history before grouping. Advance only forward and remeasure exact destination views, including notices. Keep policy independent of native/session/TUI imports. Use UTF-16 code units for baseline conversation limits and exact read handles for clipped content.

- [ ] **Step 5: Run focused tests**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_selection.py tests/context_paging/test_notice.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/errors.py src/ds4py/context_paging/selection.py src/ds4py/context_paging/notice.py tests/context_paging/test_selection.py tests/context_paging/test_notice.py
git commit -m "feat: select stable rolling context views"
```

---

### Task 6: Implement bounded recovery serialization and public parts

**Files:**
- Create: `src/ds4py/context_paging/recovery_parts.py`
- Create: `src/ds4py/context_paging/recovery_response.py`
- Create: `tests/context_paging/test_recovery_parts.py`
- Create: `tests/context_paging/test_recovery_response.py`

**Interfaces:**
- Produces: public DTO serialization, bounded JSON Pointer selection, part descriptions, explicit public part handles, and image metadata without binary payloads.
- Produces: UTF-16 length/slicing helpers that never split surrogate pairs and continuation arguments for the next page.
- Produces: one complete-reply serializer with a 16,384-byte budget and one part-description serializer with a 4,096-byte budget. Text pages contain strings/plain text; other values use exact JSON. A JSON header precedes an unescaped page block.

- [ ] **Step 1: Write failing wire-protocol tests**

Consume fixtures and assert:

1. ASCII and astral Unicode pages use UTF-16 offsets/limits, never Python code-point positions, and do not split surrogate pairs;
2. complete replies, escaping, metadata, and continuation fields fit the stated UTF-8 byte budget even when the first candidate exceeds it;
3. legacy `assistant`/`toolResult` selectors use unique-choice defaults, explicit `part`/JSON Pointer handles resolve exactly, and bounded errors list valid choices;
4. large items defer to part descriptions while smaller requested items still load; requested output order is retained after smallest-first fitting;
5. any item containing an image defers regardless of size; metadata/handles remain available while bytes/base64 never enter JSON, previews, search text, or text reads;
6. no payload is duplicated into a `details` field.

- [ ] **Step 2: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_recovery_parts.py tests/context_paging/test_recovery_response.py -q
```

Expected: FAIL because public parts and bounded serialization do not exist.

- [ ] **Step 3: Implement the explicit UTF-16 and byte-budget adapters**

Use a UTF-16 codec/index adapter, not code-point slicing. Serialize each complete candidate and measure `len(candidate.encode('utf-8'))`; shrink through the baseline continuation protocol rather than truncating encoded bytes. Keep public JSON shapes independent of Pydantic's internal serialization layout.

- [ ] **Step 4: Run focused tests**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_recovery_parts.py tests/context_paging/test_recovery_response.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/context_paging/recovery_parts.py src/ds4py/context_paging/recovery_response.py tests/context_paging/test_recovery_parts.py tests/context_paging/test_recovery_response.py
git commit -m "feat: add bounded context recovery pages"
```

---

### Task 7: Add history navigation and all four recovery tools

**Files:**
- Create: `src/ds4py/context_paging/navigator.py`
- Create: `src/ds4py/context_paging/tools.py`
- Create: `tests/context_paging/test_navigator.py`
- Create: `tests/context_paging/test_tools.py`

**Interfaces:**
- Produces: `HistoryNavigator` with `search`, `browse`, and `load` operations over the latest public projection; its cache identity changes only when visible stable IDs change.
- Produces tool functions with exact public names/schemas:
  - `search_history(query, files=None, tools=None, failed=None, limit=5, load=False)`;
  - `browse_history(direction, historyId=None, sequence=None, count=5, stride=1)`;
  - `load_history(historyIds)` with one to three IDs;
  - `read_context_output(historyId, source=None, part=None, mode=None, contentIndex=None, toolCallId=None, offset=None, limit=None)`.
- Produces: model-visible `ToolReturn` adaptation without exposing implementation objects. Disabled calls return exactly `Context paging recovery tools are disabled.`

- [ ] **Step 1: Write failing navigator and tool tests**

Assert baseline defaults/maxima, camelCase arguments, stable IDs/sequences/handles, context previews, files/tools/failed filters, browse directions/count/stride, result ordering, and at-most-three smallest-first loading. Verify repeated reads against unchanged visible IDs reuse the corpus, while changed IDs rebuild it.

Through real Pydantic tool definitions/calls, assert strict argument rejection, disabled refusal in all four tools, and all Task 6 wire guarantees. A valid explicit `mode='image'` raises `EngineError("image recovery requires M5 vision support")`; invalid handles and non-image selections keep their baseline validation errors.

- [ ] **Step 2: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_navigator.py tests/context_paging/test_tools.py -q
```

Expected: FAIL because navigation and tool adaptation do not exist.

- [ ] **Step 3: Implement navigation and tool adaptation**

Read canonical history at execution time through a supplied history source. Exclude paging calls/results from the public corpus, including their own invocations. Keep the M5 image exception explicit; do not return `BinaryContent` or a text-encoded attachment.

- [ ] **Step 4: Run focused tests**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_navigator.py tests/context_paging/test_tools.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/context_paging/navigator.py src/ds4py/context_paging/tools.py tests/context_paging/test_navigator.py tests/context_paging/test_tools.py
git commit -m "feat: add context history recovery tools"
```

---

### Task 8: Wire RollingWindow as the single view-only context owner

**Files:**
- Create: `src/ds4py/capabilities/rolling_window.py`
- Modify: `src/ds4py/capabilities/context.py` (`ContextStrategy.request_messages`)
- Modify: `src/ds4py/model.py` (`DS4Model._open_stream`)
- Modify: `src/ds4py/agent.py` (composition and disabled delegate)
- Create: `tests/agent/test_rolling_window.py`
- Create: `tests/agent/test_rolling_window_reserve.py`
- Modify: `tests/agent/test_context_strategy.py`
- Modify: `tests/agent/test_agent_factory.py`
- Modify: `tests/agent/test_conformance.py`

**Interfaces:**
- Produces: `RollingWindow(model.binder, enabled=None, token_budget=None, trim_to=None)` with `reserve_operations=True`.
- Produces: internal `set_disabled_owner(compactor: LLMCompaction) -> None`; composition calls it before building the agent, without adding a factory or second strategy field.
- Produces: `ContextStrategy.request_messages(messages, run_context)`, defaulting to identity for all existing owners. RollingWindow returns its exact stored selected view.
- Produces: RollingWindow's four-tool toolset at construction time, independent of enabled state.
- `before_model_request` leaves `request_context.messages = ctx.messages` and stores selection in session state. Immediately before `MessageBinder.prepare_model_request`, `DS4Model._open_stream` invokes the owner hook after harness normalization.

- [ ] **Step 1: Write failing request-view and composition tests**

Using real capabilities and `FakeNative`, assert:

1. full history remains byte-for-byte canonical while the bound request contains the selected notice/view;
2. marked view-only messages survive post-normalization binder input both with a retained user request first and with a resident system request first;
3. runtime adoption retains canonical requests plus the new response and discards notices/replacements;
4. exact budget/notice counts match Task 4 and stable cuts reuse append-only KV prefixes between advances;
5. projection caches invalidate on hints, pending/restored notes, clock-derived note changes, configuration, content, grouping, and frontier changes; failed/abandoned selections publish no cache or live state;
6. the selected owner is the only registered `ContextStrategy`, and all four tools are present while both enabled and disabled;
7. disabled RollingWindow delegates threshold/tool/reserve behavior to the existing `LLMCompaction`, while manual compaction remains a separate concrete service.

- [ ] **Step 2: Write failing overflow and reserve tests**

Assert oversized ordinary trailing results are replaced only on cloned view requests with fresh `ds4.key`, `ds4.view_of`, and paging markers; only the changed part becomes exactly `Tool error: tool result still does not fit`. The minimal trailing-rejection loop and paging-result exclusion preserve live call/result pairing. If no legal view fits, raise `ContextExhausted`.

During reserve recovery, advance in the actual view, rebind through retained reserve operations, and continue the same `ModelResponse` without re-emitting text or mutating canonical history. Selection errors propagate as `ContextSelectionError`; reserve exhaustion chains the selection error through `ContextExhausted` without changing its public shape.

- [ ] **Step 3: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/agent/test_rolling_window.py tests/agent/test_rolling_window_reserve.py tests/agent/test_context_strategy.py tests/agent/test_agent_factory.py -q
```

Expected: FAIL because RollingWindow and the post-normalization view hook do not exist.

- [ ] **Step 4: Implement the capability and binder-side seam**

Store mutable enabled/frontier/notice/navigation/accounting state in `Session.state`, not a per-run copy. Measure through Task 4, select through Task 5, and use the existing `project_request()` final exact fit check. Never call `_reject_until_fit` on canonical messages and never pass marked views through harness merging. Use retained reserve operations; do not add a second native allocation path.

- [ ] **Step 5: Run focused, conformance, and neighboring tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_rolling_window.py tests/agent/test_rolling_window_reserve.py tests/agent/test_context_strategy.py tests/agent/test_agent_factory.py tests/agent/test_conformance.py tests/agent/test_compaction_capability.py tests/agent/test_model_stream.py -q
```

Expected: PASS, including context extension conformance items 1–5 for RollingWindow and unchanged LLMCompaction behavior.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/capabilities/rolling_window.py src/ds4py/capabilities/context.py src/ds4py/model.py src/ds4py/agent.py tests/agent/test_rolling_window.py tests/agent/test_rolling_window_reserve.py tests/agent/test_context_strategy.py tests/agent/test_agent_factory.py tests/agent/test_conformance.py
git commit -m "feat: add rolling window context owner"
```

---

### Task 9: Add v2 request-view save and atomic restore

**Files:**
- Modify: `src/ds4py/store.py` (`SaveRecord` v1/v2 and optional `request_view`)
- Modify: `src/ds4py/binding.py` (`MessageBinder.request_view`)
- Modify: `src/ds4py/context_paging/history.py` (view validation/frontier derivation)
- Modify: `src/ds4py/runtime_restore.py`
- Modify: `src/ds4py/runtime.py` (save/switch/fork integration)
- Create: `tests/agent/test_rolling_window_restore.py`
- Modify: `tests/agent/test_restore.py`
- Modify: `tests/agent/test_store.py`
- Modify: `tests/agent/test_compaction_persistence.py`

**Interfaces:**
- Produces: `MessageBinder.request_view() -> tuple[ModelMessage, ...]`, returning deep copies of the last successful bound view or `()` before any bind.
- `SaveRecord` accepts v1 records without `request_view`, writes v2, and serializes a present view through `ModelMessagesTypeAdapter` in JSON mode. V2 compaction saves may omit it.
- Produces: pure request-view provenance validation and `derive_frontier(canonical_messages, request_view) -> CutFrontier`, replaying that frontier to require exactly the retained canonical messages/exchanges.
- Case 1 binds a validated `request_view` only for an enabled RollingWindow with compatible payload/preamble. Case 2 ignores view-scoped segments and fresh-renders full canonical history.

- [ ] **Step 1: Write failing save-record and binder-view tests**

Assert v1 round-trip compatibility, v2 optional-field behavior, deep-copy isolation, and empty-view omission. Save records preserve full canonical snapshots plus exact view tokens/segments. Notices have a paging marker and no `view_of`; replacements have fresh view keys, markers, and existing canonical targets. Image-containing sessions retain the existing save refusal.

- [ ] **Step 2: Write failing restore/frontier tests**

Cover:

1. no omitted content, whole-turn omissions, active partial-turn omissions with optional tool-call anchor, and a retained user request first;
2. exact case-1 switch/fork KV extension, including an active remainder that later completes and saving again before any new request;
3. malformed keys/hashes/markers/targets, missing snapshots, and frontier replay mismatches fail atomically with session/run/step/key context;
4. every keyed segment resolves exactly once in the view; unmarked messages occur in canonical history; arbitrary additional canonical messages are allowed;
5. stripped/missing payload, preamble mismatch, disabled RollingWindow, and non-Rolling owners take case 2 without feeding transient messages to compaction; malformed `request_view` provenance is ignored on these case-2 paths rather than making an otherwise valid canonical restore fail;
6. case 2 with `request_view` ignores view-scoped note segments and rebuilds first-turn date/hints from canonical history; unpaged records retain existing note restoration;
7. v1, full equal views, pre-first-request forks, and post-compaction resets start with an empty frontier and avoid unnecessary prefill.

- [ ] **Step 3: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/agent/test_rolling_window_restore.py tests/agent/test_restore.py tests/agent/test_store.py -q
```

Expected: FAIL because v2 views, exact bound-view access, and view-scoped restore do not exist.

- [ ] **Step 4: Implement save/restore case selection and provenance**

Select the restore case before calling canonical segment validation. Validate canonical schema/media/stable keys independently. For eligible case 1, validate view plus segments before live mutation, bind the view to the payload, then adopt full canonical history. Keep save order KV -> canonical snapshot -> save record. Case 2 resets note/restored-note state appropriately and ignores `request_view` provenance.

- [ ] **Step 5: Run focused and persistence regressions**

```bash
uv run --frozen --extra agent pytest tests/agent/test_rolling_window_restore.py tests/agent/test_restore.py tests/agent/test_store.py tests/agent/test_compaction_persistence.py tests/agent/test_conformance_persistence.py tests/agent/test_system_notes.py -q
```

Expected: PASS with no prefill on exact view restore and no notice/replacement leakage into canonical history.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/store.py src/ds4py/binding.py src/ds4py/context_paging/history.py src/ds4py/runtime_restore.py src/ds4py/runtime.py tests/agent/test_rolling_window_restore.py tests/agent/test_restore.py tests/agent/test_store.py tests/agent/test_compaction_persistence.py
git commit -m "feat: persist rolling request views"
```

---

### Task 10: Wire settings, lifecycle, commands, status, and CLI selection

**Files:**
- Create: `src/ds4py/capabilities/rolling_window_session.py`
- Modify: `src/ds4py/capabilities/rolling_window.py`
- Modify: `src/ds4py/runtime.py` (`project_trusted`, `context_paging`, failure boundary)
- Modify: `src/ds4py/agent_cli.py`
- Modify: `src/ds4py/tui_commands.py`
- Modify: `src/ds4py/tui.py`
- Modify: `src/ds4py/noninteractive.py`
- Modify: `docs/development.md` (RollingWindow configuration, composition requirements, commands, and verification limits)
- Create: `tests/agent/test_context_paging_commands.py`
- Create: `tests/agent/test_context_paging_lifecycle.py`
- Create: `tests/agent/test_context_paging_cli.py`
- Modify: `tests/agent/test_context_commands.py`
- Modify: `tests/agent/test_noninteractive.py`
- Modify: `tests/agent/test_tui.py`
- Modify: `tests/agent/test_runtime.py`

**Interfaces:**
- Produces: shared agent CLI `--context-strategy compaction|rolling`, `--context-token-budget`, `--context-trim-to`, and `--trust-project-settings`; programmatic runtime gains `project_trusted=False`.
- Produces: `/context-paging [on|off|status|stats]` only for a RollingWindow owner and `AgentRuntime.context_paging(action)` delegating to it.
- Produces: `context-paging` status values exactly `paging on`/`paging off`, `context_paging_error` diagnostics, and `context_paging_warning(message=...)`.
- CLI Rolling selection resolves settings before `open_engine`, then passes `ctx=min(args.ctx, resolved_token_budget)` through the existing worker/session creation path.

- [ ] **Step 1: Write failing command and lifecycle tests**

Assert no-arg/status/repeated on/off/stats preserve cut/accounting, actual on/off changes reset them, invalid arguments return usage without mutation, stats makes no native/provider call, and command execution remains idle-only. `/new` clears override/cut/accounting/navigation/restore IDs but preserves warning history; switch/fork use Task 9 restoration; compaction start clears accounting, success resets cut, cancellation/failure preserves cut, and override survives. `model_changed` resets cut/accounting but preserves override/warnings. Duplicate lifecycle and saved-entry notifications cannot reset a newer frontier twice.

Assert status clears on every `session_end` and publishes only after successful initialization. Settings read/parse failures publish `paging off` when a session exists, emit their diagnostic, and abort without half-applied enabled state.

- [ ] **Step 2: Write failing CLI/frontend/error tests**

Assert default `compaction` preserves current behavior; Rolling reserves the smaller native window even when initially disabled; on/off never resizes; `raw` gains no flags; preflight settings errors abort before event-bus construction. Trim warnings deduplicate as specified and reach TUI plus `warning: <message>` on noninteractive stderr with unsubscribe in `finally`.

Inject each `ContextSelectionError` before a request. Assert the failed run and `run_end` are recorded, the same exception is re-raised, interactive/noninteractive frontends preserve its code/diagnostic through normal `EngineError` handling, and no successful empty result is fabricated. Tool fallback text remains exactly unchanged.

- [ ] **Step 3: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/agent/test_context_paging_commands.py tests/agent/test_context_paging_lifecycle.py tests/agent/test_context_paging_cli.py tests/agent/test_noninteractive.py tests/agent/test_tui.py -q
```

Expected: FAIL because settings/CLI/command/lifecycle wiring does not exist.

- [ ] **Step 4: Implement session and frontend integration**

Keep TUI imports out of the capability. Use existing registry admission, runtime operation guards, session event subscriptions, and status events. Resolve constructor overrides above file settings, and make programmatic budget clamping to `session.ctx()` explicit. Reserve native context only during CLI Rolling selection; never resize a live session. Document the CLI flags, trusted settings, default-enabled Rolling policy versus default-compaction CLI, disabled delegate, `/context-paging`, and the programmatic requirement to allocate a suitably sized native context in `docs/development.md`.

- [ ] **Step 5: Run focused and neighboring frontend tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_context_paging_commands.py tests/agent/test_context_paging_lifecycle.py tests/agent/test_context_paging_cli.py tests/agent/test_context_commands.py tests/agent/test_noninteractive.py tests/agent/test_tui.py tests/agent/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/capabilities/rolling_window_session.py src/ds4py/capabilities/rolling_window.py src/ds4py/runtime.py src/ds4py/agent_cli.py src/ds4py/tui_commands.py src/ds4py/tui.py src/ds4py/noninteractive.py docs/development.md tests/agent/test_context_paging_commands.py tests/agent/test_context_paging_lifecycle.py tests/agent/test_context_paging_cli.py tests/agent/test_context_commands.py tests/agent/test_noninteractive.py tests/agent/test_tui.py tests/agent/test_runtime.py
git commit -m "feat: add rolling context commands and lifecycle"
```

---

### Task 11: Add read-only statistics and request-input provenance

**Files:**
- Create: `src/ds4py/context_paging/stats.py`
- Modify: `src/ds4py/model.py` (`ds4.request_sync_count` and exact paging input metadata)
- Modify: `src/ds4py/capabilities/rolling_window.py` (statistics service)
- Modify: `src/ds4py/runtime.py` (live store access only if required)
- Create: `tests/context_paging/test_stats.py`
- Create: `tests/agent/test_context_paging_stats.py`
- Modify: `tests/agent/test_summary_usage.py`
- Modify: `tests/agent/test_restore.py`

**Interfaces:**
- Produces: a read-only statistics report with baseline stored-size, history-token, input-token, budget/model-window, and cache sections plus unavailable/zero semantics.
- `DS4StreamedResponse` increments `ds4.request_sync_count` once per `Synced` event. RollingWindow records the selected `RequestProjection.total_tokens` as validated nonnegative safe integer metadata at `ds4.context_paging.input_tokens` on the final canonical response.
- Older records can reconstruct exact input only when `ds4.request_sync_count == 1` and all of `usage.input_tokens`, `usage.cache_read_tokens`, and `usage.cache_write_tokens` exist; otherwise input is `unavailable`.

- [ ] **Step 1: Write failing statistics tests**

Use a real `FileStepStore` and temporary session directories. Assert:

1. stored size counts regular files under `steps/`, including run/snapshot/tool-effect/media files, excludes `events.jsonl`, KV and save records, and counts content-addressed media once;
2. history tokens use UTF-16 units ceiling-divided by four, include user/text/tool/thinking/tool-name/JSON arguments, use 4,800 units per image, and exclude resident prefix and transient notices;
3. repeated snapshots deduplicate by stable message key plus rehydrated content hash, while revisions/summaries remain distinct;
4. externalized text is restored through the store's media-aware `list_snapshots`/`restore_media` path before hashing/estimating, and documented damaged-file behavior is preserved;
5. cached append, exact restore, and reserve recovery report exact selected input; ambiguous multi-sync or missing old counters report `unavailable`, never a falsely exact sum;
6. cache read/write totals deduplicate stable saved response/tool usage records, explicit zero remains zero, and any required invalid/missing counter makes the affected total unavailable;
7. disabled budget/model display is `inactive`, only estimates carry `~`, and stats mutate no navigation/cut/accounting/settings/status/native/canonical state.

- [ ] **Step 2: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_stats.py tests/agent/test_context_paging_stats.py -q
```

Expected: FAIL because statistics and sync provenance do not exist.

- [ ] **Step 3: Implement statistics without native/provider calls**

Read snapshots through `FileStepStore`'s public async media-aware methods; do not parse raw snapshot JSON as a substitute for media restoration. Walk storage metadata without loading KV. Keep this display estimate separate from exact selection. Stamp sync count in the existing response dispatch path and selected input only on the final canonical response.

- [ ] **Step 4: Run focused and usage/restore regressions**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_stats.py tests/agent/test_context_paging_stats.py tests/agent/test_summary_usage.py tests/agent/test_restore.py tests/agent/test_model_stream.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/context_paging/stats.py src/ds4py/model.py src/ds4py/capabilities/rolling_window.py src/ds4py/runtime.py tests/context_paging/test_stats.py tests/agent/test_context_paging_stats.py tests/agent/test_summary_usage.py tests/agent/test_restore.py
git commit -m "feat: add rolling context statistics"
```

---

### Task 12: Complete conformance, portable gates, Salad live evidence, and delivery

**Files:**
- Modify only files needed by RED regression findings; every correction starts with a focused failing test.
- Update the PR body with verification and review evidence; do not commit generated caches, native libraries, models, traces, virtual environments, or `.devenv` state.

**Interfaces:**
- Consumes: all previous tasks, the provisioned Salad CUDA host, matching pinned ds4 source, native library/agent, and model.
- Produces: complete conformance, static/portable gate results, native/live RollingWindow evidence, standard replay parity, review evidence, pushed branch, and PR.

- [ ] **Step 1: Run the complete portable conformance matrix**

Run all context-paging, agent, engine, persistence, and extension conformance tests. Confirm all conformance items 1–9 pass, with RollingWindow satisfying items 1–5. Check each bullet in the spec's “ds4 integration and regression tests” against a named executed test; add a RED test for any uncovered behavior before changing production code.

- [ ] **Step 2: Run the required local gates**

```bash
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
uv run --frozen --extra agent pytest -m "not build and not live" -q
```

Expected: every static hook and portable test passes. Record exact counts and warnings. If the current NixOS `prek` dynamic-linker problem remains, report it as an environment blocker and resolve the canonical runner rather than installing a second environment or silently skipping the gate.

- [ ] **Step 3: Prepare the provisioned Salad CUDA host with the other agent**

Obtain the host access, GPU/CUDA architecture, compatible checkout, model path, and native build locations from the provisioning agent. Use the pinned ds4 source from `ds4py.abi.pins`; do not substitute current ds4 main. Set `BACKEND=cuda` and an explicit appropriate `CUDA_ARCH` (or `native` on that host), then follow `docs/development.md`:

```bash
uv run ds4py build --src "$DS4_SRC" --backend cuda
uv run ds4py check
make -C "$DS4_SRC" cuda CUDA_ARCH="${CUDA_ARCH:-native}"
```

Apply the documented CUDA host-register opt-out patch only if that host exhibits the documented failure/stall. Do not run or claim these GPU steps locally.

- [ ] **Step 4: Run live RollingWindow acceptance and standard replay parity**

On the Salad host, run a long RollingWindow agent session that crosses several stable cuts, uses all four recovery tools, survives tool overflow/reserve recovery, and remains within the effective budget. Save, switch, and fork; verify exact view restore extends KV without unnecessary prefill and disabled Rolling falls back to compaction. Record commands, model/library revisions, native events, budget/input statistics, and outcomes.

Capture and replay the standard C-agent trace as documented:

```bash
DS4_TEST_MODEL="$MODEL" DS4_REPLAY_TRACE="$TRACE" uv run pytest -m live tests/live/test_replay_parity.py -v
```

Expected: the long session meets the M4 live criterion and replay reports exactly `1 passed`, not skipped. Portable tests alone do not satisfy this step.

- [ ] **Step 5: Inspect and commit any final verified corrections**

Run `git diff --check`, inspect `git diff main...HEAD` and `git status --short`, and confirm only planned source/tests/spec/plan files are tracked. Rerun affected focused tests plus both local gates after every correction.

- [ ] **Step 6: Run the required author-side PR review loop**

Load and follow `.agents/skills/pr-review-loop/SKILL.md` before requesting human review. Give fresh reviewers the spec, plan, base/head, diff, acceptance criteria, portable results, and Salad evidence. Record every accepted, declined, and deferred finding and every human reviewer callout. Rerun the loop after pushed code changes.

- [ ] **Step 7: Push and open the PR**

Push `feat/rolling-window` and open a pull request. Include the full-parity scope, M5 image exception, v2 persistence implications, exact local gate results, Salad host/backend/GPU and live outputs, remaining limitations, review evidence head, declined/deferred findings, and human reviewer callouts. Ensure the evidence head matches the PR head before requesting human review.
