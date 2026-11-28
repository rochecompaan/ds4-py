# RollingWindow Full-Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the complete view-only RollingWindow context strategy, including stable cuts, exact ds4 token/KV accounting, public history recovery, lifecycle and settings, v2 request-view persistence, statistics, and native/live verification.

**Architecture:** Keep deterministic policy and recovery in `ds4py.context_paging`; keep native, filesystem, command, and status effects in the orchestrating `RollingWindow` capability and small adapters. Reuse `binding_provenance.py` for all key/hash/segment validation and extract pure save planning before extending persistence. Canonical harness history remains authoritative. A selected request view is measured without a projection cache and bound through `MessageBinder`, while transient notices and recovery replacements never enter canonical history.

**Tech Stack:** Python 3.14, Pydantic AI/Pydantic AI Harness, existing `MessageBinder`/`Session`/`FileStepStore`, `FakeNative`, pytest, devenv and uv. Node is used only by the manually run baseline fixture generator. CUDA inference and live replay run on the separately provisioned Salad host.

**Spec:** [`docs/specs/2026-11-12-rolling-window-design.md`](../specs/2026-11-12-rolling-window-design.md). The parity baseline is `pi-context-paging` commit `4409e5e4ffa0b626c11e72afe666f0b6e55a0e5f`; later upstream changes are not normative.

**Base:** `7d80c7c`; `feat/rolling-window` adds only the design and this plan. Execute in `/home/roche/projects/satyrn/ds4-py/.worktrees/rolling-window`; do not edit the main checkout.

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
- Keep modules focused. `RollingWindow` is the only new registered capability and orchestrates policy rather than reimplementing it. No file may cross 1,000 lines; extract a cohesive responsibility instead.
- Add an identity-default `ContextStrategy.request_messages` hook for ordinary requests only. Keep `reserve_messages` recovery-only and preserve the LLMCompaction override unchanged. RollingWindow overrides both hooks, but neither hook delegates to the other. Do not add a mutable disabled-owner setter.
- Centralize production `ds4.*` keys in `message_metadata.py`. Extend `binding_provenance.py` for view validation; `context_paging` must not duplicate key, hash, marker, or segment rules.
- Do not cache request projections in this milestone. Recompute through the exact planner so every note, clock, content, configuration, grouping, and frontier input is fresh.
- Use `TypeIs`, precise domain types, PEP 695 syntax, and safe lint fixes. Do not add a second Python environment or hook runner.
- Start every behavioral change with a focused failing test and observe RED. Keep formatting and behavioral changes in separate commits.
- Run commands from this worktree inside `devenv shell`. Run the complete static gate and portable suite before delivery.
- Local verification has no GPU. Do not claim native inference or live replay parity from `FakeNative`; collect that evidence on the provisioned Salad CUDA host.

## Review Focus

- **View-scoped restore provenance:** a valid paged save can omit canonical messages and contain synthetic keys. Tasks 3 and 11 pin canonical key/hash/segment validation, atomic case-1 validation, independent canonical validation, and case-2 fallback for disabled/non-Rolling owners.
- **Small effective budgets:** notices must not consume the protected active request's floor or silently exceed the model window. Task 6 pins the 500-code-unit useful clip, the model-capped 10% conversation allowance, and bounded notice resizing.
- **Unicode paging boundaries:** Python code-point slicing can split the baseline's UTF-16 positions, and character counts do not bound UTF-8 replies. Tasks 6 and 7 pin astral characters, unpaired-surrogate avoidance, exact byte limits, and continuation arguments through shared helpers.
- **Settings and lifecycle failures:** a parse error or repeated event must not leave half-applied settings or erase a newer frontier. Task 12 pins status-then-raise startup failure, atomic settings resolution, event idempotence, and warning deduplication.
- **Damaged or old usage records:** aggregate multi-sync usage is not one exact request input. Task 13 pins typed exact/reconstructible/unavailable input provenance and damaged snapshot behavior.
- **Note-driven projection drift:** hints, dates, and restored notes can change exact request tokens without changing message hashes. Tasks 5 and 9 pin isolated previews and fresh recomputation across every emitted-note input, with no projection cache.

## File Structure

- `scripts/generate_context_paging_fixtures.mjs` and `tests/fixtures/context_paging/` provide pinned, language-neutral differential evidence; Node is not a production or CI dependency.
- `src/ds4py/message_metadata.py` is the single production registry for `ds4.*` keys and typed paging marker values.
- `src/ds4py/binding_provenance.py` owns canonical part/key/hash validation, `store.SegmentProvenance` validation, and request-view-to-canonical mappings. No paging module duplicates those rules.
- `src/ds4py/context_paging/settings.py` defines `RollingWindowSettings` and resolves trusted project, global, and programmatic settings.
- `src/ds4py/context_paging/history.py` defines canonical history/exchange records, validates exchanges, and derives restored frontiers from validated provenance.
- `src/ds4py/context_paging/public_history.py` defines public DTOs and applies paging/privacy/nested-wrapper exclusions.
- `src/ds4py/binding_projection.py` defines exact request projections and unit spans without expanding the binder facade's responsibilities.
- `src/ds4py/context_paging/selection.py` defines cut/selection records and `SelectionMeter`, resolves remembered frontiers, and advances stable cuts.
- `src/ds4py/context_paging/meter.py` implements `SelectionMeter` as `RequestMeter` through the exact binder projection, without policy or caching.
- `src/ds4py/context_paging/notice.py` defines `PagingNotice` and builds, freezes, clips, and resizes transient notices.
- `src/ds4py/context_paging/text_units.py` owns UTF-16 length/offset/slicing; `estimates.py` owns the shared four-units-per-token and 4,800-units-per-image display estimates.
- `src/ds4py/context_paging/recovery_parts.py` implements public JSON shapes, JSON Pointer selection, and image metadata.
- `src/ds4py/context_paging/recovery_response.py` implements complete UTF-8 byte-budget serialization using `text_units.py`.
- `src/ds4py/context_paging/navigator.py` owns cached public search/browse/load views and stable references.
- `src/ds4py/context_paging/tools.py` adapts the four strict tool schemas to model-visible returns and disabled/M5 errors.
- `src/ds4py/context_paging/stats.py` computes read-only reports through an explicit snapshot-store provider and returns typed request-input provenance.
- `src/ds4py/capabilities/rolling_window.py` is the sole new context owner and toolset provider; it contains orchestration, not selection/serialization/statistics policy.
- `src/ds4py/capabilities/rolling_window_session.py` contains session-scoped lifecycle/status/settings/command/statistics adapters, not a second capability.
- `src/ds4py/runtime_persistence.py` contains only pure save snapshot/record builders with explicit inputs. `AgentRuntime` retains save/switch/new/fork orchestration and state transitions; no `PersistenceHost` or runtime mirror protocol is added.
- `src/ds4py/runtime_restore.py` owns restore-case selection and pure request-view/canonical validation planning alongside its existing restore helpers.
- Existing `binding.py`, `model.py`, `runtime.py`, `store.py`, `agent.py`, `agent_cli.py`, `tui_commands.py`, `tui.py`, and `noninteractive.py` remain integration/orchestration facades rather than receiving duplicate policy implementations.

---

### Task 1: Generate the pinned differential contract fixtures

**Files:**
- Create: `scripts/generate_context_paging_fixtures.mjs`
- Create: `tests/fixtures/context_paging/manifest.json`
- Create: `tests/fixtures/context_paging/settings.json`
- Create: `tests/fixtures/context_paging/history.json`
- Create: `tests/fixtures/context_paging/selection.json`
- Create: `tests/fixtures/context_paging/notice.json`
- Create: `tests/fixtures/context_paging/navigator.json`
- Create: `tests/fixtures/context_paging/recovery.json`

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
git add tests/fixtures/context_paging/manifest.json tests/fixtures/context_paging/settings.json tests/fixtures/context_paging/history.json tests/fixtures/context_paging/selection.json tests/fixtures/context_paging/notice.json tests/fixtures/context_paging/navigator.json tests/fixtures/context_paging/recovery.json
node scripts/generate_context_paging_fixtures.mjs /tmp/pi-context-paging-4409e5e
git diff --exit-code -- tests/fixtures/context_paging/manifest.json tests/fixtures/context_paging/settings.json tests/fixtures/context_paging/history.json tests/fixtures/context_paging/selection.json tests/fixtures/context_paging/notice.json tests/fixtures/context_paging/navigator.json tests/fixtures/context_paging/recovery.json
```

Expected: both pinned runs succeed; the second produces no unstaged diff against the staged first-run outputs. Staging before the second run is required: `git diff` ignores untracked fixtures and therefore cannot prove idempotence until their first-run contents enter the index.

- [ ] **Step 4: Check fixture coverage directly**

Compare every bullet in the spec's “Differential contract fixtures” section with a manifest entry. Validate all JSON files with the repository's JSON hook or `python -m json.tool`. No new pytest suite is needed for manifest text or one-off script structure; the meaningful verification is executing the generator and consuming fixtures in later behavioral tests.

- [ ] **Step 5: Commit the reproducibility inputs**

```bash
git add scripts/generate_context_paging_fixtures.mjs tests/fixtures/context_paging/manifest.json tests/fixtures/context_paging/settings.json tests/fixtures/context_paging/history.json tests/fixtures/context_paging/selection.json tests/fixtures/context_paging/notice.json tests/fixtures/context_paging/navigator.json tests/fixtures/context_paging/recovery.json
git commit -m "test: add pinned context paging fixtures"
```

---

### Task 2: Add immutable domain records and settings resolution

**Files:**
- Create: `src/ds4py/context_paging/__init__.py`
- Create: `src/ds4py/context_paging/settings.py`
- Create: `tests/context_paging/test_settings.py`

**Interfaces:**
- Produces: immutable `RollingWindowSettings(enabled: bool, token_budget: int, trim_to: int, trim_to_explicit: bool)`.
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

Validate dynamic JSON at the file boundary. Distinguish booleans from integers despite Python's `bool` subclassing `int`. Read the global file under the explicit store root so tests never use the developer's real home. Keep file I/O and diagnostics separate: this task raises; Task 12 adds status/event publication.

- [ ] **Step 4: Run focused tests**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_settings.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/context_paging/__init__.py src/ds4py/context_paging/settings.py tests/context_paging/test_settings.py
git commit -m "feat: add context paging settings"
```

---

### Task 3: Centralize metadata and canonical view provenance

**Files:**
- Create: `src/ds4py/message_metadata.py`
- Modify: `src/ds4py/binding_provenance.py`
- Modify: `src/ds4py/binding.py`
- Modify: `src/ds4py/message_rendering.py`
- Modify: `src/ds4py/model.py`
- Modify: `src/ds4py/capabilities/compaction.py`
- Modify: `src/ds4py/capabilities/compaction_tail.py`
- Create: `tests/agent/test_request_view_provenance.py`
- Test: `tests/agent/test_binding.py`
- Test: `tests/agent/test_restore.py`

**Interfaces:**
- Produces: `DS4_KEY`, `DS4_SOURCE`, `DS4_SAMPLED`, `DS4_CONTEXT_PAGING`, `DS4_VIEW_OF`, `DS4_PAGING_INPUT_TOKENS`, and `DS4_REQUEST_SYNC_COUNT` constants plus typed `PagingMarker = Literal["notice", "recovery"]`.
- Produces: public `validate_messages(messages: Sequence[ModelMessage]) -> None` and `message_content_hash(message: ModelMessage) -> str`, implemented by the existing canonical validators rather than copied.
- Produces: `prepare_request_view_provenance(*, canonical_messages, request_view, segments, loaded) -> PreparedRequestViewProvenance`, returning the existing `PreparedProvenance` plus an immutable validated view-key/canonical-key/marker mapping.
- Consumes: `store.SegmentProvenance` as the persisted segment record and existing `prepare_provenance()` for segment coverage, key uniqueness, order, and content hashes.

- [ ] **Step 1: Write failing request-view provenance tests**

Construct canonical messages, marked notices, keyed replacements, and `SegmentProvenance` records directly. Assert every keyed segment resolves exactly once with the canonical content hash; unmarked view messages resolve to the same canonical key/hash; notices have no target; replacements have fresh keys and existing canonical targets; duplicate/out-of-order keys, mismatched hashes, malformed markers, missing targets, and non-canonical unmarked messages fail atomically. Arbitrary additional canonical messages are valid.

- [ ] **Step 2: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/agent/test_request_view_provenance.py tests/agent/test_binding.py -q
```

Expected: FAIL because the public hash/validator and view-provenance result do not exist.

- [ ] **Step 3: Extend the canonical owner and centralize keys**

Promote narrow wrappers around the existing part validator and content hash, then implement view validation in `binding_provenance.py` on top of `prepare_provenance()`. Keep frontier/exchange policy out of this module. Replace production `ds4.*` literals in the listed existing modules with `message_metadata.py` constants; leave literal wire expectations in tests. This is a mechanical key migration in the same commit as the new provenance behavior only because both establish the single canonical metadata layer; do not change rendered metadata values.

- [ ] **Step 4: Run provenance and neighboring regression tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_request_view_provenance.py tests/agent/test_binding.py tests/agent/test_restore.py tests/agent/test_binding_rewrite.py tests/agent/test_compaction_stream.py -q
```

Expected: PASS with byte-identical metadata behavior.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/message_metadata.py src/ds4py/binding_provenance.py src/ds4py/binding.py src/ds4py/message_rendering.py src/ds4py/model.py src/ds4py/capabilities/compaction.py src/ds4py/capabilities/compaction_tail.py tests/agent/test_request_view_provenance.py
git commit -m "feat: validate rolling request view provenance"
```

---

### Task 4: Project canonical history into safe public items

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

Match calls/results by `tool_call_id`; validate all original exchanges before any remembered cut is applied. Define history/exchange records here and public DTOs in `public_history.py`, not in a shared record dump. Use Task 3's `message_content_hash()` and canonical key validation; do not implement another hasher. Keep canonical content immutable and use projections/references rather than a second stored ledger. Remove provider signatures only from retained public calls; never recursively strip application-owned fields or text.

- [ ] **Step 4: Run focused tests**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_history.py tests/context_paging/test_public_history.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/context_paging/history.py src/ds4py/context_paging/public_history.py tests/context_paging/test_history.py tests/context_paging/test_public_history.py
git commit -m "feat: project public context history"
```

---

### Task 5: Expose exact observation-only request projection

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

Reuse `plan_transition()`, current registered `note_provider.preview()`, request grouping, sampled token records, assistant prefixes, and effective instructions. Do not tokenize a second approximation or call the mutating prepare path. Derive spans from planned boundaries and Task 3's canonical provenance helpers. Recompute every projection and keep projection caches out of both the binder and Task 9's meter; correctness must not depend on enumerating invalidation dimensions.

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

### Task 6: Implement stable selection and paging notices

**Files:**
- Modify: `src/ds4py/errors.py` (`ContextSelectionError`)
- Create: `src/ds4py/context_paging/selection.py`
- Create: `src/ds4py/context_paging/notice.py`
- Create: `src/ds4py/context_paging/text_units.py`
- Create: `src/ds4py/context_paging/estimates.py`
- Create: `tests/context_paging/test_selection.py`
- Create: `tests/context_paging/test_notice.py`
- Create: `tests/context_paging/test_text_units.py`

**Interfaces:**
- Produces: `ContextSelectionError(EngineError)` with structured `code`; `str(error)` is `[CODE] diagnostic`.
- Produces: cut frontier, selection state/result, and overflow records plus a `SelectionMeter` protocol implemented by Task 9's `context_paging/meter.py` and by additive fake costs in differential tests.
- Produces: `select_view(history: HistoryProjection, *, settings: RollingWindowSettings, model_context: int, meter: SelectionMeter, remembered: SelectionState | None = None, overflow: OverflowMode = "normal") -> SelectionResult`.
- Produces: frozen `PagingNotice` records and bounded resizing for a remembered notice. Selection results contain candidate messages, exact measurement, resolved effective budget/target, optional frontier/notice, and fallback/warning diagnostics.
- Produces: shared UTF-16 length/offset/slicing helpers and display-only text/image estimates consumed by this task, Task 7, and Task 13.
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

- [ ] **Step 2: Write failing notice and shared-unit tests**

Assert newest-three evicted-turn selection, newest-first selection displayed oldest-first, full user requests/final public replies only, no prefix/partial-turn conversation, exact continuation handles, 6,000-unit clips, the 500-unit minimum useful clip, and `conversationCharacterLimit = min(32_000, floor(effective_budget_tokens * 0.4))` followed by the remaining-room cap. Include configured budget above model window, frozen content between advances, and bounded resizing.

- [ ] **Step 3: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_selection.py tests/context_paging/test_notice.py tests/context_paging/test_text_units.py -q
```

Expected: FAIL because selection/frontiers/notices do not exist.

- [ ] **Step 4: Implement selection and notice construction**

Resolve remembered frontiers against raw canonical history before grouping. Advance only forward and remeasure exact destination views, including notices. Keep policy independent of native/session/TUI imports. Put UTF-16 operations in `text_units.py` and the four-units-per-token/4,800-units-per-image rules in `estimates.py`; notices must import those shared modules rather than duplicating conversion logic. Use exact read handles for clipped content.

- [ ] **Step 5: Run focused tests**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_selection.py tests/context_paging/test_notice.py tests/context_paging/test_text_units.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/errors.py src/ds4py/context_paging/selection.py src/ds4py/context_paging/notice.py src/ds4py/context_paging/text_units.py src/ds4py/context_paging/estimates.py tests/context_paging/test_selection.py tests/context_paging/test_notice.py tests/context_paging/test_text_units.py
git commit -m "feat: select stable rolling context views"
```

---

### Task 7: Implement bounded recovery serialization and public parts

**Files:**
- Create: `src/ds4py/context_paging/recovery_parts.py`
- Create: `src/ds4py/context_paging/recovery_response.py`
- Modify: `src/ds4py/context_paging/text_units.py`
- Create: `tests/context_paging/test_recovery_parts.py`
- Create: `tests/context_paging/test_recovery_response.py`
- Modify: `tests/context_paging/test_text_units.py`

**Interfaces:**
- Produces: public DTO serialization, bounded JSON Pointer selection, part descriptions, explicit public part handles, and image metadata without binary payloads.
- Consumes: Task 6's `text_units.py` helpers, extending them for continuation arguments without splitting surrogate pairs. This serialization module owns no UTF-16 conversion rules.
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
uv run --frozen --extra agent pytest tests/context_paging/test_recovery_parts.py tests/context_paging/test_recovery_response.py tests/context_paging/test_text_units.py -q
```

Expected: FAIL because public parts and bounded serialization do not exist.

- [ ] **Step 3: Implement the explicit UTF-16 and byte-budget adapters**

Use Task 6's UTF-16 codec/index adapter, not code-point slicing or a second encoder in this module. Serialize each complete candidate and measure `len(candidate.encode('utf-8'))`; shrink through the baseline continuation protocol rather than truncating encoded bytes. Keep public JSON shapes independent of Pydantic's internal serialization layout.

- [ ] **Step 4: Run focused tests**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_recovery_parts.py tests/context_paging/test_recovery_response.py tests/context_paging/test_text_units.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/context_paging/recovery_parts.py src/ds4py/context_paging/recovery_response.py src/ds4py/context_paging/text_units.py tests/context_paging/test_recovery_parts.py tests/context_paging/test_recovery_response.py tests/context_paging/test_text_units.py
git commit -m "feat: add bounded context recovery pages"
```

---

### Task 8: Add history navigation and all four recovery tools

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

Through real Pydantic tool definitions/calls, assert strict argument rejection, disabled refusal in all four tools, and all Task 7 wire guarantees. A valid explicit `mode='image'` raises `EngineError("image recovery requires M5 vision support")`; invalid handles and non-image selections keep their baseline validation errors.

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

### Task 9: Wire RollingWindow as the single view-only context owner

**Files:**
- Create: `src/ds4py/capabilities/rolling_window.py`
- Create: `src/ds4py/context_paging/meter.py`
- Modify: `src/ds4py/capabilities/context.py` (add ordinary-request `ContextStrategy.request_messages`; keep recovery-only `reserve_messages`)
- Modify: `src/ds4py/model.py` (`DS4Model._open_stream`)
- Modify: `src/ds4py/agent.py` (owner factory and disabled delegate)
- Create: `tests/agent/test_rolling_window.py`
- Create: `tests/agent/test_rolling_window_reserve.py`
- Modify: `tests/agent/test_context_strategy.py`
- Modify: `tests/agent/test_agent_factory.py`
- Modify: `tests/agent/test_conformance.py`
- Modify: `tests/agent/test_model_stream.py`

**Interfaces:**
- Produces: `ContextOwnerFactory = Callable[[MessageBinder, LLMCompaction], ContextStrategy]`; `ContextComposition.owner` accepts one existing custom owner instance, one factory, or `None`, preserving current custom-owner callers without a second strategy field.
- Produces: `ContextComposition.rolling(*, enabled: bool | None = None, token_budget: int | None = None, trim_to: int | None = None) -> ContextComposition` using a factory. This is the sole supported RollingWindow composition path for the default agent builder; directly constructed RollingWindow instances do not get a second builder path.
- Produces: `RollingWindow(model.binder, *, disabled_owner: LLMCompaction, enabled=None, token_budget=None, trim_to=None)` with `reserve_operations=True`, constructed by the factory with the same compactor returned as `AgentComponents.compactor`. The generic builder has no RollingWindow instance-adoption branch or owner/compactor reconciliation.
- Produces: `ContextStrategy.request_messages(messages, run_context)` with an identity default for ordinary requests. LLMCompaction inherits it unchanged; its existing recovery-only `reserve_messages` override continues to select `run_context.messages` when present.
- Produces: separate RollingWindow overrides for `request_messages` and `reserve_messages`. The ordinary hook returns the exact stored selected view; the recovery hook returns the actual request view only to `reserve_settlement._recover`. Neither hook delegates to the other, and no `set_disabled_owner` method is added.
- Produces: `RequestMeter` in `context_paging/meter.py`, implementing `SelectionMeter` through Task 5 with no cache or selection policy.
- Produces: RollingWindow's four-tool toolset at construction time, independent of enabled state.
- `before_model_request` leaves `request_context.messages = ctx.messages` and stores selection in session state. Immediately before `MessageBinder.prepare_model_request`, `DS4Model._open_stream` invokes `request_messages` after harness normalization. `reserve_settlement._recover` remains the sole caller of `reserve_messages`.

- [ ] **Step 1: Write failing request-view and composition tests**

Using real capabilities and `FakeNative`, assert:

1. full history remains byte-for-byte canonical while the bound request contains the selected notice/view;
2. marked view-only messages survive post-normalization binder input both with a retained user request first and with a resident system request first;
3. a regression fixture makes canonical history `C`, harness-normalized messages `N`, and Rolling-selected view `S` pairwise distinct: LLMCompaction's ordinary `_open_stream` binds `N`, never the `C` returned by its recovery hook, while RollingWindow binds exactly `S`; capture the actual `prepare_model_request` input for both owners;
4. RollingWindow's final selection measurement supplies `S` itself to `project_request_view`, and the ordinary hook returns that same message sequence with identical keys, parts, metadata, and order; the implementation must not rebuild a different candidate from `N` after selection, and its recovery hook is not called on the ordinary path;
5. runtime adoption retains canonical requests plus the new response and discards notices/replacements;
6. exact budget/notice counts match Task 5 and stable cuts reuse append-only KV prefixes between advances;
7. repeated projections immediately reflect hints, pending/restored notes, clock-derived note changes, configuration, content, grouping, and frontier changes without a projection cache; failed/abandoned selections publish no planner or live state;
8. the selected owner is the only registered `ContextStrategy`, and all four tools are present while both enabled and disabled;
9. disabled RollingWindow delegates threshold/tool/reserve behavior to the existing `LLMCompaction`, while manual compaction remains a separate concrete service;
10. `ContextComposition.rolling()` injects the identical `AgentComponents.compactor` as the disabled delegate, existing custom owner instances still work, and the builder has neither a direct-Rolling adoption branch nor two-phase/post-construction wiring.

- [ ] **Step 2: Write failing overflow and reserve tests**

Assert oversized ordinary trailing results are replaced only on cloned view requests with fresh `ds4.key`, `ds4.view_of`, and paging markers; only the changed part becomes exactly `Tool error: tool result still does not fit`. The minimal trailing-rejection loop and paging-result exclusion preserve live call/result pairing. If no legal view fits, raise `ContextExhausted`.

During reserve recovery, advance in the actual view, rebind through retained reserve operations, and continue the same `ModelResponse` without re-emitting text or mutating canonical history. Selection errors propagate as `ContextSelectionError`; reserve exhaustion chains the selection error through `ContextExhausted` without changing its public shape.

- [ ] **Step 3: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/agent/test_rolling_window.py tests/agent/test_rolling_window_reserve.py tests/agent/test_context_strategy.py tests/agent/test_agent_factory.py tests/agent/test_model_stream.py -q
```

Expected: FAIL because RollingWindow and the separate post-normalization `request_messages` hook are not wired.

- [ ] **Step 4: Implement the capability and binder-side seam**

Store mutable enabled/frontier/notice/navigation/accounting state in `Session.state`, not a per-run copy. Measure through `RequestMeter` and Task 5, select through Task 6, and use the existing `project_request()` final exact fit check. Keep measurement in `meter.py`, selection in `selection.py`, and lifecycle/statistics policy out of this capability. Never call `_reject_until_fit` on canonical messages and never pass marked views through harness merging. Use retained reserve operations; do not add a second native allocation path. Resolve `ContextComposition.rolling()` factories in `build_agent_components` with the concrete compactor already in hand; do not mutate the owner after construction or add a direct-Rolling special case. Leave ordinary LLMCompaction binding on the normalized messages through the identity request hook.

- [ ] **Step 5: Run focused, conformance, and neighboring tests**

```bash
uv run --frozen --extra agent pytest tests/agent/test_rolling_window.py tests/agent/test_rolling_window_reserve.py tests/agent/test_context_strategy.py tests/agent/test_agent_factory.py tests/agent/test_conformance.py tests/agent/test_compaction_capability.py tests/agent/test_model_stream.py -q
```

Expected: PASS, including context extension conformance items 1–5 for RollingWindow and unchanged LLMCompaction behavior.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/capabilities/rolling_window.py src/ds4py/context_paging/meter.py src/ds4py/capabilities/context.py src/ds4py/model.py src/ds4py/agent.py tests/agent/test_rolling_window.py tests/agent/test_rolling_window_reserve.py tests/agent/test_context_strategy.py tests/agent/test_agent_factory.py tests/agent/test_conformance.py tests/agent/test_model_stream.py
git commit -m "feat: add rolling window context owner"
```

---

### Task 10: Extract pure save planning without moving runtime orchestration

**Files:**
- Create: `src/ds4py/runtime_persistence.py`
- Modify: `src/ds4py/runtime.py`
- Test: `tests/agent/test_restore.py`
- Test: `tests/agent/test_runtime.py`
- Test: `tests/agent/test_compaction_persistence.py`

**Interfaces:**
- Produces: `build_save_snapshot(*, run_id, session_id, messages) -> ContinuableSnapshot`, deep-copying canonical messages through explicit inputs.
- Produces: `build_save_record(...) -> SaveRecord` from explicit snapshot, engine/model, identity, payload, token, and segment values. It computes the existing token count/digest but accepts no runtime, host, session, store, or callbacks.
- `AgentRuntime.save` retains activity handling, save-run numbering, KV writes, step-store writes, event ordering, and state transitions. `switch`, `new`, and `fork` remain on `AgentRuntime`; `runtime_restore.py` retains restore helpers. No `PersistenceHost`, host adapter, or one-call runtime delegate is added, and no v2 behavior is introduced in this task.

- [ ] **Step 1: Establish the existing persistence baseline**

```bash
uv run --frozen --extra agent pytest tests/agent/test_restore.py tests/agent/test_runtime.py tests/agent/test_compaction_persistence.py tests/agent/test_conformance_persistence.py -q
```

Expected: PASS before the move. Existing behavioral coverage is the regression gate for this mechanical extraction; add a characterization test only if a moved branch has no existing assertion.

- [ ] **Step 2: Move only pure snapshot and record construction**

Move the existing `ContinuableSnapshot` and `SaveRecord` assembly into the two pure builders. Keep save/switch/new/fork orchestration, save-run numbering, payload writes, restore abort handling, activity claims, identity hooks, and all session/store/native state transitions in their current owners. Do not add request-view cases, a host interface, or unrelated reformatting. Both `runtime.py` and the new module must remain below 1,000 lines.

- [ ] **Step 3: Verify the extraction**

```bash
uv run --frozen --extra agent pytest tests/agent/test_restore.py tests/agent/test_runtime.py tests/agent/test_compaction_persistence.py tests/agent/test_conformance_persistence.py -q
```

Expected: PASS with no test expectation changes.

- [ ] **Step 4: Commit**

```bash
git add src/ds4py/runtime_persistence.py src/ds4py/runtime.py
git commit -m "refactor: extract save record planning"
```

---

### Task 11: Add v2 request-view save and atomic restore

**Files:**
- Modify: `src/ds4py/store.py` (`SaveRecord` v1/v2 and optional `request_view`)
- Modify: `src/ds4py/binding.py` (`MessageBinder.request_view`)
- Modify: `src/ds4py/context_paging/history.py` (frontier derivation from validated provenance)
- Modify: `src/ds4py/runtime_restore.py` (pure restore-case selection and validation planning)
- Modify: `src/ds4py/runtime_persistence.py` (optional request-view input to pure save construction)
- Modify: `src/ds4py/runtime.py` (save/switch/fork orchestration through the pure planners)
- Create: `tests/agent/test_rolling_window_restore.py`
- Modify: `tests/agent/test_restore.py`
- Modify: `tests/agent/test_store.py`
- Modify: `tests/agent/test_compaction_persistence.py`

**Interfaces:**
- Produces: `MessageBinder.request_view() -> tuple[ModelMessage, ...]`, returning deep copies of the last successful bound view or `()` before any bind.
- `SaveRecord` accepts v1 records without `request_view`, writes v2, and serializes a present view through `ModelMessagesTypeAdapter` in JSON mode. V2 compaction saves may omit it.
- Consumes: Task 3's `prepare_request_view_provenance()` and immutable validated mapping; this task adds no key/hash/marker/segment validator to `context_paging`.
- Produces: `derive_frontier(canonical_messages, prepared_view) -> CutFrontier`, replaying that frontier to require exactly the retained canonical messages/exchanges.
- Produces: a pure restore-case planner in `runtime_restore.py`; `AgentRuntime.switch` invokes it before canonical segment validation and retains all live-state orchestration. Case 1 binds a validated `request_view` only for an enabled RollingWindow with compatible payload/preamble. Case 2 ignores view-scoped segments and fresh-renders full canonical history.

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

Select the restore case through the pure planner in `runtime_restore.py` before calling canonical segment validation; `AgentRuntime.switch` remains the orchestrator, not a delegate to a shadow persistence host. Validate canonical schema/media/stable keys independently. For eligible case 1, call Task 3's canonical view validator before live mutation, derive/replay the frontier in `history.py`, bind the view to the payload, then adopt full canonical history. Extend Task 10's pure save builders with the optional exact bound view while `AgentRuntime.save` keeps save order KV -> canonical snapshot -> save record. Case 2 resets note/restored-note state appropriately and ignores `request_view` provenance.

- [ ] **Step 5: Run focused and persistence regressions**

```bash
uv run --frozen --extra agent pytest tests/agent/test_rolling_window_restore.py tests/agent/test_restore.py tests/agent/test_store.py tests/agent/test_compaction_persistence.py tests/agent/test_conformance_persistence.py tests/agent/test_system_notes.py -q
```

Expected: PASS with no prefill on exact view restore and no notice/replacement leakage into canonical history.

- [ ] **Step 6: Commit**

```bash
git add src/ds4py/store.py src/ds4py/binding.py src/ds4py/context_paging/history.py src/ds4py/runtime_restore.py src/ds4py/runtime_persistence.py src/ds4py/runtime.py tests/agent/test_rolling_window_restore.py tests/agent/test_restore.py tests/agent/test_store.py tests/agent/test_compaction_persistence.py
git commit -m "feat: persist rolling request views"
```

---

### Task 12: Wire settings, lifecycle, commands, status, and CLI selection

**Files:**
- Create: `src/ds4py/capabilities/rolling_window_session.py`
- Modify: `src/ds4py/capabilities/rolling_window.py`
- Modify: `src/ds4py/runtime.py` (`project_trusted`, `context_paging`, `context_paging_stats` store adapter, failure boundary)
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
- Produces: `/context-paging [on|off|status]` only for a RollingWindow owner and `AgentRuntime.context_paging(action)` delegating to it; Task 13 adds `stats` to this command without changing runtime persistence.
- Produces: `AgentRuntime.context_paging_stats()` as the final store-adapter seam. It supplies the current `FileStepStore` and session directory to `RollingWindowSession.statistics()`; Task 13 implements the report behind this interface and does not modify `runtime.py`.
- Produces: `context-paging` status values exactly `paging on`/`paging off`, `context_paging_error` diagnostics, and `context_paging_warning(message=...)`.
- CLI Rolling selection resolves settings before `open_engine`, then passes `ctx=min(args.ctx, resolved_token_budget)` through the existing worker/session creation path.

- [ ] **Step 1: Write failing command and lifecycle tests**

Assert no-arg/status/repeated on/off preserve cut/accounting, actual on/off changes reset them, invalid arguments return usage without mutation, and command execution remains idle-only. `/new` clears override/cut/accounting/navigation/restore IDs but preserves warning history; switch/fork use Task 11 restoration; compaction start clears accounting, success resets cut, cancellation/failure preserves cut, and override survives. `model_changed` resets cut/accounting but preserves override/warnings. Duplicate lifecycle and saved-entry notifications cannot reset a newer frontier twice.

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

Keep TUI imports out of the capability. Use existing registry admission, runtime operation guards, session event subscriptions, and status events. Resolve constructor overrides above file settings, and make programmatic budget clamping to `session.ctx()` explicit. Reserve native context only during CLI Rolling selection; never resize a live session. Register on/off/status now and add the final runtime statistics-provider adapter, but leave the `stats` subcommand and report to Task 13. Document the CLI flags, trusted settings, default-enabled Rolling policy versus default-compaction CLI, disabled delegate, `/context-paging`, and the programmatic requirement to allocate a suitably sized native context in `docs/development.md`.

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

### Task 13: Add read-only statistics and request-input provenance

**Files:**
- Create: `src/ds4py/context_paging/stats.py`
- Modify: `src/ds4py/model.py` (sync-count and exact paging input metadata through Task 3's key registry)
- Modify: `src/ds4py/capabilities/rolling_window.py` (latest-input snapshot only)
- Modify: `src/ds4py/capabilities/rolling_window_session.py` (statistics provider adapter)
- Modify: `src/ds4py/tui_commands.py` (add the `stats` subcommand)
- Modify: `docs/development.md` (statistics and provenance semantics)
- Create: `tests/context_paging/test_stats.py`
- Create: `tests/agent/test_context_paging_stats.py`
- Modify: `tests/agent/test_context_paging_commands.py`
- Modify: `tests/agent/test_summary_usage.py`
- Modify: `tests/agent/test_restore.py`

**Interfaces:**
- Produces: `StatisticsStore`, a narrow structural protocol for only the used public async `FileStepStore.list_snapshots` and `restore_media` methods; `FileStepStore` satisfies it directly.
- Produces: immutable `RollingWindowStatisticsState` and `ContextPagingStats` records plus `collect_statistics(*, store: StatisticsStore, session_dir: Path, current: RollingWindowStatisticsState) -> ContextPagingStats`. The function receives storage explicitly and never imports or reaches through `AgentRuntime`.
- Consumes: Task 12's `AgentRuntime.context_paging_stats()` adapter, which supplies the current store/directory. This task makes no `runtime.py` change.
- Produces: immutable `InputTokenReading(provenance: Literal["exact", "reconstructible", "unavailable"], tokens: int | None)` and one `read_request_input(response)` adapter. Exact/reconstructible readings require an integer; unavailable requires `None`. All sync-count/counter rules live in that adapter; report code consumes its typed result.
- Produces: a read-only statistics report with baseline stored-size, history-token, input-token, budget/model-window, and cache sections plus unavailable/zero semantics.
- `DS4StreamedResponse` increments `DS4_REQUEST_SYNC_COUNT` once per `Synced` event. RollingWindow records the selected `RequestProjection.total_tokens` as validated nonnegative safe integer metadata at `DS4_PAGING_INPUT_TOKENS` on the final canonical response.
- Older records return `reconstructible` only when `DS4_REQUEST_SYNC_COUNT == 1` and all of `usage.input_tokens`, `usage.cache_read_tokens`, and `usage.cache_write_tokens` exist; otherwise input is `unavailable`.

- [ ] **Step 1: Write failing statistics tests**

Use a real `FileStepStore` and temporary session directories. Assert:

1. stored size counts regular files under `steps/`, including run/snapshot/tool-effect/media files, excludes `events.jsonl`, KV and save records, and counts content-addressed media once;
2. history tokens use Task 6's shared UTF-16/four-units-per-token estimator, include user/text/tool/thinking/tool-name/JSON arguments, use the shared 4,800-units-per-image rule, and exclude resident prefix and transient notices;
3. repeated snapshots deduplicate by stable message key plus Task 3's rehydrated content hash, while revisions/summaries remain distinct;
4. externalized text is restored through the provider's media-aware `list_snapshots`/media-restoration path before hashing/estimating, and documented damaged-file behavior is preserved;
5. cached append, exact restore, and reserve recovery return `exact` selected input; a valid older single-sync record returns `reconstructible`; ambiguous multi-sync or missing old counters return `unavailable`, never a falsely exact sum;
6. cache read/write totals deduplicate stable saved response/tool usage records, explicit zero remains zero, and any required invalid/missing counter makes the affected total unavailable;
7. disabled budget/model display is `inactive`, only estimates carry `~`, and stats mutate no navigation/cut/accounting/settings/status/native/canonical state;
8. `/context-paging stats` and repeated `stats` preserve cut/accounting, make no native/provider call, and use Task 12's explicit runtime store adapter without adding store logic to statistics.

- [ ] **Step 2: Run the tests and verify RED**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_stats.py tests/agent/test_context_paging_stats.py tests/agent/test_context_paging_commands.py -q
```

Expected: FAIL because statistics and sync provenance do not exist.

- [ ] **Step 3: Implement statistics without native/provider calls**

Read snapshots through the supplied `StatisticsStore`; do not parse raw snapshot JSON as a substitute for media restoration or obtain a store from `AgentRuntime`. Walk storage metadata under the supplied session directory without loading KV. Reuse Task 6's display estimators and Task 3's content hash. Keep this display estimate separate from exact selection. Stamp sync count in the existing response dispatch path and selected input only on the final canonical response. Implement the sync/counter rule once in `read_request_input()` and keep it out of report formatting.

- [ ] **Step 4: Run focused and usage/restore regressions**

```bash
uv run --frozen --extra agent pytest tests/context_paging/test_stats.py tests/agent/test_context_paging_stats.py tests/agent/test_context_paging_commands.py tests/agent/test_summary_usage.py tests/agent/test_restore.py tests/agent/test_model_stream.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ds4py/context_paging/stats.py src/ds4py/model.py src/ds4py/capabilities/rolling_window.py src/ds4py/capabilities/rolling_window_session.py src/ds4py/tui_commands.py docs/development.md tests/context_paging/test_stats.py tests/agent/test_context_paging_stats.py tests/agent/test_context_paging_commands.py tests/agent/test_summary_usage.py tests/agent/test_restore.py
git commit -m "feat: add rolling context statistics"
```

---

### Task 14: Complete conformance, portable gates, Salad live evidence, and delivery

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
