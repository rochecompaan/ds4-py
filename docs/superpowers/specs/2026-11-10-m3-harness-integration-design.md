# M3 harness integration design

| | |
| --- | --- |
| Date | 2026-11-10 |
| Status | Draft for review; no implementation is approved by this document |
| Package | `satyrn-agent`, using generic companion APIs in ds4-py |
| ds4-py baseline | `main` `8a14760c0800d67095a3e25404e394ff983b5691` |
| Satyrn baseline | `5a9d924b5f5ccbd889bfec5f71b353767dbb2d79` |
| M4 evidence | Roché's PR #24, `m4-context-persistence` at `3eecd1ddcbbfb06177a1a85cccd4aca7dbba0128` |
| Governing requirements | Satyrn agent package design §13; generic companion policy in §8, including §8.4; ds4-py ADR 0002 |
| Demo target | Paul can chat with Qwen 27B in the existing ds4-py TUI, use tools, change DFlash and the soft think budget, and see committed-draft acceptance in the footer |

## 1. Intent

Satyrn's first package slice already opens and configures a real ds4-py
`Session`. It has no command-line or interactive front end. ds4-py already has
the required harness, tools, session persistence, TUI, slash-command registry,
and footer, but its CLI composition hardcodes `ds4py.open()` and constructs
`DS4Model` itself.

This slice connects those two existing stacks. It must not create a second
agent, token loop, TUI, native worker, or persistence format. The result is a
`satyrn-agent` command that reuses ds4-py's agent command and TUI while opening
through `satyrn_agent.open()`.

The demo is time-sensitive. The design therefore chooses a narrow explicit
composition seam over a general plugin system, and defers telemetry schemas and
other work that is not needed for the demo.

## 2. Evidence and current state

### 2.1 ds4-py

At the baseline:

- `python -m ds4py` without `build`, `check`, or `raw` reaches
  `agent_cli.main()` → `command()` → `_run_tui()` → `TUI`.
- `agent_cli.open_runtime()` opens `ds4py.open()`, creates its only session,
  computes a `SamplingPolicy`, constructs `DS4Model`, and then constructs
  `AgentRuntime`. There is no configured-session factory parameter.
- `TUI.__init__()` creates `SlashCommandRegistry` and calls
  `install_builtin_commands()`. The registry is already a suitable extension
  point, but there is no post-builtin registration hook in composition.
- `FooterModel` already renders `session.emit("status", key=..., text=...)` in
  its sorted custom-status map. An empty status removes the key.
- `session.state(key, factory)` survives prompts and fork. `Session.new()` and
  `Session.switch()` close and drop scoped states; `Session.fork()` retains
  them. `session_start` is emitted after a new or switched identity is set.
- `Session.set_decode_step()` rejects both an active round and a paused round.
- A stop-sequence-paused chat response is retired by `MessageBinder`'s private
  `_retire_paused()`. That helper appends EOS, marks it with the sampled
  response's key and provenance, and updates the bound response end. Calling
  `chat_ops.retire_paused()` directly would skip that bookkeeping.
- `AgentRuntime.save()` always streams a native payload. The existing save
  format and restore planner already support `payload_bytes=0`: a zero-payload
  save takes the rendered-restore path with full message and provenance
  validation.
- Noninteractive execution always attempts `runtime.save()`, including after a
  run error or interruption. An unsupported native payload must therefore not
  turn an otherwise valid run into an avoidable persistence failure.
- `PrefillTraceWriter` records committed facade `Token` events by harness run
  and model step. It is suitable evidence for the final token-alignment check.

### 2.2 Satyrn

At the baseline:

- `satyrn_agent.open()` mirrors ds4-py's native-open options and returns a
  `SatyrnEngine`. Its first and only `session()` applies native sampling
  defaults, explicit sampling overrides, the optional soft think-budget hook,
  and one resolved DFlash policy.
- The returned object is ds4-py's real `Session`, so `DS4Model` needs no Satyrn
  adapter and no second request loop.
- `SatyrnConfig` already defines `auto`/`on`/`off`, drafter path, draft count,
  sampling overrides, and optional `ThinkBudgetConfig(max_tokens, bias)`.
  `config_from_dict()` validates the equivalent JSON-shaped mapping.
- `DFlashStep` owns thread-safe metrics for committed blocks, drafts offered,
  drafts accepted, replacements, and discarded tails. Counters describe only
  committed prefixes.
- The current policy's initial `off` path returns `PlainStep()` without calling
  `ds4_session_set_verify_mode(..., OFF)`. That is correct for a fresh engine
  but insufficient for an on → off toggle: runtime mode selection must
  explicitly leave MMA verify mode.
- The current think hook is immutable. M3 needs one stable hook whose immutable
  setting can be replaced between prompts; repeatedly adding hooks would apply
  stale budgets and there is no removal API.
- The eight existing Satyrn bindings remain downstream optional bindings.
  ds4-py rejects an extension that redeclares a core binding, so the generic
  payload-support probe must have exactly one owner.

### 2.3 Native payload support and M4

The pinned qwen35 patch series supports dense in-memory snapshots and file
payloads. Its DFlash patch adds an optional trailer for the drafter ring. A
loader with matching geometry restores that ring; a missing or incompatible
drafter skips the trailer and starts with an empty ring. Malformed trailing
bytes still fail. Consequently, DFlash being on or off is not itself a reason
to reject save or switch.

The patched `ds4_session_supports_payload` predicate concerns file payloads.
It is not the predicate for M4's required in-memory
`ds4_session_save_snapshot`/`load_snapshot` operations. A positive probe means
the file-payload API is supported, not that every transient native state is
serializable; the existing sync, staging, and native error checks still apply.

Roché's PR #24 is a documentation-only standard-compaction branch at the cited
head. Its plan adds:

- a `system` in-memory checkpoint after the first system-only sync;
- summary-only generation with native-state restoration;
- transcript replacement using a checkpoint or a rebuild;
- automatic and manual `/compact` compaction, including reserve recovery; and
- compacted-session save, switch, and fork regression coverage.

It explicitly defers RollingWindow, the 50,000-token reminder work, the
system-prefix cache, Bookmarks, `/list`, `/strip`, and `/del`. The companion
implementation must re-read the final merged M4 code rather than implement
against this plan blindly.

Neither ds4-py nor satyrn-agent has `docs/sdd.md`, so no SDD document-size cap
applies to this spec.

## 3. Goals, non-goals, and anti-goals

### 3.1 Goals

1. Ship a `satyrn-agent` console command that reuses ds4-py's parser driver,
   `DS4Model`, `AgentRuntime`, tools, persistence, TUI, and footer.
2. Open the configured session through `satyrn_agent.open()` and pass its fully
   merged sampling policy to `DS4Model` exactly once.
3. Expose DFlash mode, drafter path, draft count, sampling overrides, and soft
   think budget through CLI options and project settings.
4. Add `/dflash` and `/think-budget` commands, including query forms and clear
   validation errors.
5. Show requested/resolved DFlash state and committed acceptance in the footer.
6. Give mutable harness capabilities the existing session-state lifecycle:
   reset on `/new` and `/switch`, retain on fork.
7. Define safe between-prompt mode changes, including retirement of a paused
   response with exact EOS provenance.
8. Guard native file-payload operations while preserving stripped save and
   rendered switch, including after M4 compaction.
9. Prove the combined path with portable public-flow tests and a new real
   Metal TUI test, then perform the manual demo acceptance on this Mac.
10. Preserve ds4-py ADR 0002 exactly.

### 3.2 Non-goals

- A second front end, inference server, native worker, agent loop, or token
  loop.
- A generic plugin discovery framework, entry-point-driven CLI composition, or
  a downstream session/message/save schema.
- Changes to DFlash kernels, model loading, native payloads, or the ds4 patch
  series.
- A generic MTP speculative step in ds4-py.
- Changing the drafter path or draft count after native open. These are
  process-lifetime settings.
- Persisting slash-command changes back to `pyproject.toml` or saved sessions.
- Native think-mode `/think` commands, which remain M5, or a hard reasoning
  stop. `/think-budget` changes only the soft logit bias.
- A DFlash acceptance telemetry schema, wall-time gate, or speed claim.
- Implementation of deferred M4 features in this slice.
- Comparison of ordinary-mode and DFlash-mode output numerics.
- Any approval prompt, approval capability, or `--yolo` option.

### 3.3 Anti-goals

The change fails if it:

- puts Satyrn or DFlash identifiers or policy in ds4-py production code or
  tests;
- bypasses `DS4Model`, `AgentRuntime`, the existing binder, or the native
  worker;
- re-renders sampled text to recover token IDs;
- changes decode mode during a round, or falls back to plain decode after a
  DFlash block or native failure has started;
- emits speculative tokens or acceptance counts before native commit;
- loses sampled EOS provenance when retiring a paused response;
- turns off MMA numerics only cosmetically while leaving native verify mode
  enabled;
- rejects all persistence merely because native file payloads are unavailable;
- applies file-payload support rules to M4's in-memory snapshots; or
- treats a model-free test, a skipped live test, or component-only live tests
  as proof of the complete demo.

## 4. Approaches considered

### 4.1 A. Satyrn command reusing ds4-py internals directly

Satyrn would ship its own command, call `agent_cli.parser()`, and reuse the TUI
but supply its own open path.

**What is duplicated or coupled**

- `open_runtime()` must be reproduced or bypassed because it owns native open,
  sampling construction, `DS4Model`, `AgentRuntime`, interaction services, and
  close ordering.
- Parts of `command()` must also be reproduced if the replacement runtime is to
  retain prefix loading, tracing, interactive/noninteractive selection, and
  cleanup.
- Paused-response retirement and payload guards would either use private
  helpers or add downstream wrappers around ds4-py's runtime behavior.
- The parser and TUI classes themselves can be reused, but their composition
  assumptions become implicit downstream dependencies.

**Required ds4-py changes**

None for a minimal demo. That is the approach's only material advantage.

**Upgradeability**

Poor. A ds4-py change to runtime arguments, sampling, capabilities, persistence,
or cleanup can break Satyrn without changing the parser or TUI APIs that the
command appears to reuse. Private-helper use makes that coupling worse.

### 4.2 B. Small generic ds4-py seam plus Satyrn plug-in — selected

Satyrn passes an explicit hook object to `agent_cli.main()`. The hook customizes
parser options, opens a configured session, and registers extra TUI commands.
ds4-py retains everything else.

**What is duplicated**

Only downstream-specific work that belongs in Satyrn: TOML/CLI mapping to
`SatyrnConfig`, DFlash/think-budget command behavior, and status formatting.
There is no duplicate model, runtime, token loop, save format, or TUI loop.

**Required ds4-py changes**

Small generic companion changes provide:

1. CLI/session/TUI composition hooks;
2. a public binder operation for paused-response retirement; and
3. an optional core file-payload probe plus generic save/switch guards.

Each change has default-path tests and its own small companion PR. M4-related
persistence guards are rebased on and reviewed against the final M4 merge.

**Upgradeability**

Good. Satyrn pins the ds4-py companion commit and tests the public composition
contract. ds4-py's no-hook path remains the current behavior. Upstream changes
to `DS4Model`, `AgentRuntime`, tools, and TUI internals arrive through the same
composition used by ds4-py itself, rather than through a copied downstream
version.

### 4.3 Decision

Choose **B**. It has slightly more coordination cost than A but less total
machinery and a stable generic boundary. It also directly tests ds4-py's goal
of supporting downstream agents without downstream policy entering core.

## 5. Selected architecture and ownership

```text
satyrn-agent console script
        |
        | AgentCLIHooks (explicit, per process)
        v
ds4py.agent_cli.main
  |-- parser + Satyrn parser configurator
  |-- open_runtime
  |     |-- configured-session factory
  |     |     `-- satyrn_agent.open() -> real Session + merged SamplingPolicy
  |     |-- DS4Model(session, sampling=...)
  |     `-- AgentRuntime(model, ...)
  |-- command: prefix / trace / interactive or noninteractive
  `-- TUI
        |-- built-in commands
        `-- post-builtin configurator -> /dflash, /think-budget

Satyrn controller <-> session.state / decode step / think hook / status events
```

The illustrative generic values below define behavior, not mandatory final
names:

```python
@dataclass(frozen=True)
class ConfiguredSession:
    session: Session
    sampling: SamplingPolicy


@dataclass(frozen=True)
class AgentCLIHooks:
    program: str = "ds4py"
    configure_parser: Callable[[ArgumentParser], None] | None = None
    open_session: SessionFactory | None = None
    configure_commands: CommandConfigurator | None = None
```

`SessionFactory` is a context-manager factory called with the parsed arguments
and `project_dir`. Its result supplies the real session and final sampling
policy. `CommandConfigurator` receives the slash registry and runtime after
builtins are installed.

Exact type and module names are builder latitude; the ownership and call order
are not.

### 5.1 ds4-py default path

With no hooks, `open_runtime()` performs the current open and sampling flow.
The configured-session abstraction must not drop `args.temp`, `top_p`, `min_p`,
or seed, and must not alter the default engine or model options.

`open_runtime()` still constructs `DS4Model` and `AgentRuntime`. The context
manager closes the runtime before the session-factory context exits and closes
the engine wrapper. Prefix loading, tracing, question answerer wiring, initial
prompt handling, and noninteractive behavior remain in the existing command
path.

The hook's `program` is used for parser help and command error prefixes,
including noninteractive diagnostics. It changes labels only, not ds4-py's
package identity.

### 5.2 Satyrn path

The `satyrn-agent` script constructs one per-invocation CLI adapter, not a
second runtime controller and not mutable module-global session state. M3
changes the pinned ds4-py dependency to include its `agent` extra and registers
the console script; it keeps ds4-py's pinned harness version rather than
selecting another harness. `SatyrnEngine.session()` constructs the single
runtime controller described below, and the wrapper exposes it to the TUI
command configurator. The CLI adapter's session factory:

1. resolves project and CLI configuration before native open;
2. calls `satyrn_agent.open(..., config=resolved_config)` with the existing
   generic engine options;
3. obtains the wrapper's only configured session;
4. returns that session and `SatyrnEngine.sampling_defaults`, which already
   include family defaults, native defaults, explicit overrides, and seed; and
5. keeps the wrapper alive until after `AgentRuntime.close()`.

There is no second `SamplingPolicy.for_engine()` call and no second application
of sampling overrides.

The command configurator runs after `install_builtin_commands()`. Argparse and
`SlashCommandRegistry` continue to reject duplicate options and commands.
Noninteractive mode uses the same configured-session factory and persistence
guards but does not create the TUI, slash commands, or footer.

## 6. CLI and project settings

### 6.1 Project file

M3 reads **only** `<current-working-directory>/pyproject.toml`, the same
`project_dir` passed to `open_runtime()`. It does not search parent directories.
No file, or a file without `[tool.satyrn]`, supplies an empty mapping. TOML and
configuration errors name the source file and occur before native open.

The table uses the existing configuration field names and validation:

```toml
[tool.satyrn]
dflash = "auto"
drafter_path = "/absolute/path/dflash-Qwen3.8-27B-Q4_0.gguf"
draft_tokens = 0
think_budget = { max_tokens = 512, bias = 2.0 }

[tool.satyrn.sampling]
temperature = 0.0
top_k = 0
top_p = 1.0
min_p = 0.0
```

`tomllib` produces the mapping and the existing external-config validation is
applied to the merged final value. Unknown fields, booleans used as integers,
negative counts, invalid ranges, and non-finite values remain errors.

Precedence is:

1. `SatyrnConfig` defaults;
2. `[tool.satyrn]` values; then
3. explicitly supplied CLI values.

An explicit zero is an override, not a missing value. Project `think_budget`
continues to require both `max_tokens` and `bias`. Relative drafter paths are
expanded relative to the directory containing this TOML file, not relative to
a later native worker directory. A leading `~` is expanded. No environment
variable interpolation is added.

### 6.2 Options

| CLI | Project field | Default and rule |
| --- | --- | --- |
| `--dflash auto\|on\|off` | `dflash` | `auto`; `on` requires an effective drafter path |
| `--drafter PATH` | `drafter_path` | `None`; loaded only at native open |
| `--draft-tokens N` | `draft_tokens` | `0`, selecting the engine default; non-negative integer |
| `--temp FLOAT` | `sampling.temperature` | Unset uses the merged native default; explicit zero is valid |
| `--top-k N` | `sampling.top_k` | Unset uses the merged native default; zero is valid |
| `--top-p FLOAT` | `sampling.top_p` | Unset uses the merged native default; range 0–1 |
| `--min-p FLOAT` | `sampling.min_p` | Unset uses the merged native default; range 0–1 |
| `--think-budget N` | `think_budget.max_tokens` | Disabled unless configured; positive integer |
| `--think-bias FLOAT` | `think_budget.bias` | Finite and positive; source rules below |
| `--no-think-budget` | `think_budget` | Clears any project budget |

The existing `--think`, `--think-level`, `--think-max`, and `--nothink` options
retain their native think-mode meaning and are independent of
`--think-budget`.

For CLI think-budget resolution:

- `--think-budget N` uses the project bias when a project budget exists;
  otherwise it uses **2.0**.
- `--think-bias` is valid when the effective budget comes from either the CLI
  or project settings. It overrides that budget's bias.
- `--think-bias` without an effective budget is an error before native open.
- `--no-think-budget` conflicts with both positive think-budget flags and
  clears the project value.

Drafter path, draft count, and sampling are startup-only. DFlash mode and the
soft think budget can also change through slash commands, but those changes do
not rewrite this configuration.

## 7. Runtime controller and session lifecycle

### 7.1 One controller, one recorder, one stable think hook

Satyrn's first-and-only session setup creates the runtime controller as part of
its existing configuration flow. The controller holds the immutable merged
startup configuration outside session state. Mutable per-live-session values
are in `session.state()` under one downstream-owned key.

Each scoped state contains:

- requested DFlash mode and the concrete resolved step;
- the native-policy owner needed for later between-round selection;
- one thread-safe metrics recorder shared by every `DFlashStep` created for
  that live session;
- current immutable `ThinkBudgetConfig | None`.

The controller, not the scoped state, owns its `session_start` and `run_end`
subscriptions for the engine lifetime. `close_states()` therefore cannot remove
the subscriptions needed to initialize the next identity or refresh its footer.
The controller removes those lifetime subscriptions during
`SatyrnEngine.close()`.

`DFlashStep` gains an optional recorder injection point while retaining its
current private-recorder default for direct users. A mode toggle replaces the
step but reuses the scoped recorder. Thus counts accumulate across toggles in
one live session without copying or delta-adjusting snapshots.

The think-budget hook is installed once and remains stable. On each sample it
reads the controller's current immutable setting, using Python-only atomic or
locked access, and applies the existing soft-bias algorithm. When the setting
is `None`, it is a no-op. The hook never calls `session.state()`, session or
engine methods, cancellation, or blocking I/O on the worker. Installing one
no-op hook while disabled avoids stale duplicate hooks and does not change
sampled logits.

### 7.2 `/new`, `/switch`, fork, and close

- **`/new`:** the old scoped state closes. On `session_start`, the controller
  creates fresh state from the immutable startup config, re-resolves DFlash,
  creates a fresh recorder, resets think budget to startup, and emits fresh
  footer status.
- **`/switch`:** old state closes before restore. On the target
  `session_start`, after the restore callback, the controller performs the same
  fresh initialization. Saved DFlash settings and acceptance are not restored,
  because no downstream persistence schema is added.
- **Fork:** state, requested and resolved mode, recorder, acceptance counts,
  and current think budget are retained. There is no interactive `/fork`
  command in this slice; programmatic `AgentRuntime.fork()` is the test path.
- **Close:** per-session scoped resources close exactly once with the existing
  session cleanup, and the controller removes its lifetime subscriptions during
  `SatyrnEngine.close()`. No replacement worker or recovery path is introduced.

A failed switch that emits its error `session_start` still initializes state
for the identity that ds4-py actually adopted; it must not resurrect the old
scoped state.

## 8. Slash commands and status

### 8.1 `/dflash`

| Form | Behavior |
| --- | --- |
| `/dflash` | Return the current requested/resolved mode and acceptance text |
| `/dflash auto` | Re-resolve support; use DFlash when available, otherwise plain |
| `/dflash on` | Require every DFlash requirement; report the failed requirement on error |
| `/dflash off` | Set native verify mode off and install `PlainStep` |

Any other arity or mode raises `CommandError` with
`usage: /dflash [auto|on|off]`.

Mode selection has one Satyrn policy implementation shared by initial open and
later commands:

- `off` and `auto` fallback explicitly set verify mode `OFF` before installing
  `PlainStep`;
- supported `on`/`auto` sets `DS4_VERIFY_MODE_MMA` and installs a `DFlashStep`
  using the scoped recorder and startup draft count;
- `on` names missing bindings, missing drafter, wrong backend, zero rows, or
  setup failure;
- `auto` falls back only during this between-prompt policy resolution; an
  unexpected native error still propagates; and
- the resolved step remains fixed until another explicit command or session
  reinitialization. `auto` is not reinterpreted between rounds.

A successful command emits status before returning. A failed command leaves the
previous requested/resolved mode and status text intact, subject to the paused
response retirement rule below.

### 8.2 `/think-budget`

| Form | Behavior |
| --- | --- |
| `/think-budget` | Report `think budget off` or `think budget N tokens, bias B` |
| `/think-budget N [BIAS]` | Replace the immutable setting for subsequent samples |
| `/think-budget off` | Disable the soft budget |

`N` must be a positive integer and `BIAS` finite and positive. Invalid input
uses `usage: /think-budget [<max_tokens> [bias]|off]`.

When only `N` is supplied, the command retains the current bias if a budget is
active; otherwise it uses **2.0**. Changes do not inject `</think>`, truncate
text, cancel a round, alter native think mode, or guarantee closure. They take
effect on the next sample, including if a paused response is resumed.

### 8.3 Commands execute between prompts

The existing TUI reads input during a run but queues slash lines instead of
executing them. Therefore these commands run only after the active agent prompt
has ended. The controller nevertheless defensively requires an idle native
round before changing decode or hook settings.

There is no asynchronous mode-change request and no command interruption of an
active model round.

### 8.4 Footer text

Satyrn emits:

```python
session.emit("status", key="dflash", text=status_text)
```

The status has two parts:

1. requested/resolved mode: show only the mode when they are equal, otherwise
   `<requested>/<resolved>`; and
2. committed acceptance: `accept R% (A/O)`, where `A` is drafts accepted and
   `O` is drafts offered. `R` is `100 * A / O`, formatted to one decimal place.
   With `O == 0`, show `accept — (0/0)` rather than claiming 0% acceptance.

Examples:

```text
dflash: on, accept — (0/0)
dflash: on, accept 62.5% (10/16)
dflash: auto/on, accept 62.5% (10/16)
dflash: auto/off, accept — (0/0)
dflash: off, accept 62.5% (10/16)
```

The last example deliberately retains the live session's historical committed
metrics while plain decode is active. Off rounds add no offered or accepted
drafts.

Status refreshes:

- after initial controller creation, and again during TUI command configuration
  after `FooterModel` has subscribed (the open-time event alone would be missed);
- after each successful `/dflash` command;
- on every `session_start`, so `/new` and `/switch` cannot leave stale counters;
  and
- on every `run_end`, including success, interruption, and error, from the
  thread-safe committed metrics snapshot.

M3 does not write these metrics to run records or define a telemetry schema.

## 9. Paused rounds and mode-change invariants

### 9.1 Actual TUI cleanup and direct-consumer paused rounds

The actual M3 `DS4Model`/TUI path cannot retain a paused round between prompts.
`DS4StreamedResponse` cleanup closes the facade generator and then calls
`session.reconcile_round()`; reconciliation truncates to the committed cutoff
and clears `_paused_stream`, `_paused_control`, and paused sampled tokens.
Consequently, `/dflash` in the shipped TUI needs only the between-prompt idle
guard. It does not need to invent a pause-resume lifecycle.

A pause can survive for a consumer that drives `Session.run_round()` directly
and uses `MessageBinder` without `DS4Model` cleanup.
`Session.set_decode_step()` correctly rejects a mode change in that state. To
define how such a pause is closed, the companion exposes a **public generic
binder operation** with the existing private `_retire_paused()` behavior:

1. require an idle round and a paused **chat** response with a bound sampled
   response record;
2. clear that paused chat stream/control state;
3. append exactly the normal assistant EOS;
4. annotate that EOS with the existing sampled response key, content hash, and
   `source="sampled"`; and
5. update the bound response end.

It adds no user or assistant message, performs no rewind or second model
response, and does not resume generation. With no paused chat response it is a
no-op. A paused chat response with no bound sampled-response record fails with
a clear error **before any mutation**, rather than dereferencing an empty
record list. The controller calls the operation defensively before native mode
selection, so it is harmless in the actual TUI and correct for a direct binder
consumer. The next direct-consumer prompt starts a fresh assistant prefix.

The operation does not retire a raw `_paused_control`-only round. That raw
state has no `MessageBinder` response/provenance owner, so
`Session.set_decode_step()` continues to reject the mode change with its
paused-round error. No raw-session retirement API is implied.

Retirement is deliberately completed **before** native mode selection. If
`/dflash on` then fails a support check, the previous mode remains selected
but the old paused response stays closed; the command does not attempt to
recreate its parser state.

This slice does not change `DS4Model` stop-sequence cleanup, EOS handling, or
resumability. Adding resumable harness responses would require a separate
explicit ds4-py behavior design and is not smuggled into this companion seam.

### 9.2 Native invariants

- Active DFlash decode is never interrupted to change modes.
- Every controlled round exit closes or invalidates verify before save,
  switch, checkpoint, snapshot, or compaction work.
- Cancellation and runtime lifecycle operations use the existing stop-and-await
  cleanup before changing session state.
- A DFlash error after block start never selects `PlainStep` as recovery.
- On → off changes both the Python step and native verify mode, so later plain
  rounds use ordinary decode numerics.
- Paused-chat retirement is tested only at the direct
  `Session.run_round()`/`MessageBinder` seam where that state can survive; no
  unreachable paused-TUI fixture is claimed. Raw `_paused_control`-only and
  unbound paused-chat cases reject before mutation.

## 10. File-payload support and persistence

### 10.1 One generic optional binding

ds4-py owns an **optional core binding**:

```c
bool ds4_session_supports_payload(ds4_session *s);
```

Satyrn keeps exactly its existing eight downstream bindings and does not
redeclare this name. The patched build records Satyrn's eight entries with
`source="satyrn"`, `required=False`, and the payload probe as a separate
optional core entry. After they are **rebuilt and re-probed by the new ds4-py**,
upstream libraries without the function continue to load with the probe marked
absent. An older library+manifest pair with no record for the newly declared
probe fails closed with the existing rebuild-required ABI error; absence of a
probe record is not silently treated as an absent symbol.

A generic session query returns `True`, `False`, or `None` on the worker:

- `True`/`False`: the native predicate answered;
- `None`: the optional binding is absent, so support is unknown.

`None` preserves legacy behavior. Only an explicit `False` selects the guarded
fallback. This avoids regressing existing ds4-py users on libraries that lack
the newer predicate.

### 10.2 Save

After the existing stop-and-await active-run rule and empty-transcript check:

- `True` or `None` retains the current sync-and-stream native payload path and
  all token/byte consistency checks.
- `False` performs no native payload staging. It writes the existing KV header
  with `payload_bytes=0`, then writes the complete step snapshot and save
  record with full transcript tokens, digest, segments, and
  `record.payload_bytes=0`.

The zero-payload file is atomically published before the snapshot and record,
as today. The record's token count is still the full transcript length; zero is
only the native payload size. This is a stripped save, not an empty session and
not a failure.

Noninteractive unconditional save therefore has a valid result on an
explicitly unsupported engine.

### 10.3 Switch

After record, model, snapshot, and provenance validation, but before any native
load or identity/state mutation:

- `True` or `None` retains the current KV/render planner and exact token/digest
  checks.
- `False` changes a planned KV restore to the existing rendered restore, with a
  mismatch explanation that the live engine does not support file payloads.

Rendered restore keeps the saved message history and provenance and performs a
full rebuild prefill on the next request. It does not call `load_payload`.
Corrupt records, incompatible models, and invalid provenance remain errors;
the fallback is not a way to bypass validation.

Fork performs no payload operation and retains controller state. On the pinned
qwen35 engine, supported KV save/switch may include the DFlash trailer. Loading
without a matching drafter restores the target state with an empty ring rather
than being rejected.

### 10.4 M4 and remaining persistence commands

The same save/switch rules apply after automatic or manual compaction. In
contrast:

- The file-payload predicate does **not** guard M4's in-memory `system`
  checkpoint, summary generation, transcript replacement, or `/compact`.
  Those paths use required snapshot/sync/rebind contracts, not `session.kv`.
- `/compact` remains available when file payloads are unsupported. Its result
  and progress text must not claim that a native KV payload was persisted. A
  later `/save` can be stripped and a later `/switch` rendered.
- Reserve compaction starts only after the DFlash step has committed its
  accepted prefix and closed verify. Summary generation uses the M4 summary
  job's plain `argmax` path and does not increment DFlash metrics.
- Compacted-session save/switch tests must cover both native KV support and
  the stripped/rendered fallback. The latter can be a portable fault/capability
  double; no unsafe real-native failure is required.
- PR #24 defers the system-prefix cache. If that cache is added later, its
  file-payload loads and saves use this same probe with ordinary rendering and
  prefill as fallback.
- `/list`, `/strip`, and `/del`, when later implemented, need no native support
  guard: `/list` reads save records, `/strip` deletes only `session.kv`, and
  `/del` deletes `session.kv` plus `save.json` while retaining step history.

The ds4-py companion PR that touches M4 code must be based on the final merged
M4 result. Guard placement may follow that code, but these semantics may not be
weakened without a new design decision.

## 11. Tool policy: ADR 0002 remains unchanged

This integration adds no tool-approval policy, prompt, answerer, capability, or
flag. Interactive and noninteractive tools execute without per-command
approval by default, exactly as ds4-py ADR 0002 requires.

Specifically:

- `--yolo` remains invalid;
- no Satyrn equivalent of `--yolo` is added;
- the existing `--read-only` workspace limit remains independent of approval;
- `AskUser` remains a model-question interaction, not a tool-approval policy;
  and
- a downstream extension could add its own policy through capability hooks,
  but this M3 slice does not do so.

## 12. Verification design

### 12.1 Generic ds4-py companion tests

Each ds4-py companion PR follows ds4-py's own RED-GREEN and quality policy and
adds focused tests for its public contract:

**Composition and paused-response PR**

- No-hook parsing, engine open, sampling, runtime arguments, question wiring,
  tracing, and close order match the current behavior.
- Parser and command hooks run in the specified order; duplicate names fail.
- A configured-session factory supplies the session and exact sampling policy
  without a second default merge.
- Interactive and noninteractive paths both use the factory; only interactive
  mode installs TUI commands.
- Public paused-chat retirement produces the same tokens, EOS provenance,
  bound end, and next-prompt behavior as the existing private path, with no
  pause as a no-op. A raw control-only pause keeps rejecting decode changes,
  and an unbound paused chat fails before mutation.
- ADR 0002 tests still reject `--yolo` and no approval arguments enter runtime
  composition.

**Payload-support PR**

- Optional-core and downstream binding names do not collide; a rebuilt library
  with an absent core probe returns `None`, native false/true return their
  values, and unrelated optional bindings retain their source/required
  metadata. A legacy manifest with no record for the new declaration is
  rejected with the rebuild-required error.
- Explicit false writes a complete stripped save with zero payload and full
  snapshot/provenance; true and unknown retain native save behavior.
- Explicit false forces rendered switch before native load; true and unknown
  retain exact KV restore and all validation errors.
- Noninteractive success and failure paths can finish with a stripped save.
- On the final M4 code, compacted sessions use native KV when supported and
  stripped/rendered persistence when unsupported, without adding a payload
  predicate to in-memory checkpoint or summary operations.

### 12.2 Satyrn model-free public-flow tests

New normal-path scenarios use the actual Satyrn package, real ds4-py worker,
session, `DS4Model`, `AgentRuntime`, and TUI where applicable. Only external
native behavior is replaced by the existing deterministic `FakeNative` and
scripted optional DFlash bindings.

Cover:

- TOML absence/table extraction, no-upward-search behavior, path resolution,
  CLI precedence, explicit zeros, every think-bias combination, invalid values,
  and validation before native open;
- default/native/explicit sampling merge order passed once to `DS4Model`;
- real tool-call and tool-result rounds through the configured runtime;
- both commands' query/set/usage/error forms, active-run queuing, and no
  mid-round mutation;
- on/auto/off policy, explicit verify-mode off after an on → off toggle,
  missing-support errors, and no failure-driven plain fallback;
- one recorder across repeated toggles, exact status strings, 0/0 formatting,
  one-decimal rounding, and run-end refresh;
- `/new` and `/switch` resetting startup settings, metrics, and footer on
  `session_start`; fork retaining all of them; close unsubscribing exactly
  once;
- a no-op disabled think hook and a replaced setting observed by subsequent
  samples without any session call from the worker;
- paused-chat retirement through direct `Session.run_round()` with
  `MessageBinder` and caller-supplied stop sequences, including sampled EOS
  provenance and a failed mode change after retirement; raw control-only and
  unbound paused-chat rejection before mutation; and a separate TUI test proving
  actual `DS4Model` cleanup leaves no pause to retire; and
- supported native persistence and explicit-unsupported stripped/rendered
  persistence through the same public runtime.

The existing Satyrn rule remains: new Satyrn code reaches 100% line and branch
coverage through model-free public-flow E2E, with auxiliary unit tests only for
parameterized validation and complex exception/external-service doubles.
Model-free coverage is not live inference evidence.

### 12.3 Existing live tests and what they prove

| Existing suite | Reuse for M3 | What it does not prove |
| --- | --- | --- |
| `tests/e2e/live/test_abi_workflow.py` | Patched build/check/open and repeated real public-session flow | Harness composition or TUI |
| `test_policy.py` | Real `auto`/`on`/`off`, native defaults, explicit-zero overrides, consecutive rounds | Same-session toggling or verify-off after on |
| `test_capabilities.py` | Real native Qwen sampling and soft think-bias effect | Dynamic slash-command settings |
| `test_metrics.py` | Committed-stream acceptance reconciliation and external timing metadata | Footer text or session-scoped aggregation |
| `test_dflash_parity.py` and `test_dflash_boundaries.py` | Real token/parser/RNG/commit behavior, stops, tools, reserve, cancellation | AgentRuntime or persistence integration |
| ds4-py `tests/live/test_bakeoff.py` | Generic DS4Model/AgentRuntime/noninteractive tools and persistence path | Satyrn, DFlash, the interactive TUI, or the pinned Qwen models |

These are component or generic-path tests. No existing live test covers the
new Satyrn + harness + TUI combination. The ABI workflow assertions must also
cover the new optional core probe while retaining the eight Satyrn-owned
entries and their source metadata.

The full ds4-py C bake-off is not a substitute for the M3 Metal gate and is not
a required Qwen demo step. The existing Satyrn real-engine suite is a required
regression suite for this slice.

### 12.4 New live tests

Add `tests/e2e/live/test_harness_tui.py` using the real patched library and
pinned GGUFs, the actual configured runtime/TUI, and prompt-toolkit pipe input
or an equivalent real terminal driver. It must:

1. open with DFlash `on`, greedy sampling, and seed 1, then verify the initial
   footer shows `on` and `accept — (0/0)` before the first prompt;
2. run the checked-in `greedy-parity` corpus prompt and require offered drafts
   and at least one accepted draft;
3. execute at least two user turns and one real tool exchange, asserting both
   the tool-call and tool-result harness events rather than only parsed tool
   syntax;
4. change `/think-budget`, query it, disable it, and continue real generation;
5. toggle `/dflash off`, run a plain round and verify metrics do not advance,
   then toggle `/dflash on` and verify new committed metrics;
6. inspect footer state for requested/resolved modes, acceptance ratio, counts,
   and session-start reset;
7. exercise `/save`, `/new`, and `/switch`, then continue generation through
   the restored history; and
8. compare every emitted committed `Token` ID with assistant transcript
   segments and check native position against transcript length minus an
   unevaluated trailing EOS.

For DFlash-on spans, also assert:

```text
new committed assistant tokens
    == delta committed_blocks + delta drafts_accepted
```

Acceptance shown in the footer must equal the recorder's accepted/offered
counts at the same boundary. The test uses no native double and may not skip a
required branch when the documented live environment is supplied.

Once M4 lands, add a real supported-path compaction case: generate real
history with DFlash on, invoke `/compact`, continue the response or next turn,
check token/native alignment and summary/event behavior, then save and switch.
Unsupported-payload compaction persistence remains a portable guard test.

## 13. Definition of Done: real Metal demo on this Mac

A skip, a model-free pass, or the older component suites alone does not satisfy
this section.

### 13.1 Identity and environment

Use the exact models in satyrn-agent README **Models**:

- target `Qwen3.8-27B-Q4_K_M.gguf`, revision
  `5a7da681f60570ab5b439a587e912d2e5eddb582`, SHA-256
  `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520`;
- drafter `dflash-Qwen3.8-27B-Q4_0.gguf`, revision
  `a316ea527fde32c12487be7574e2db5ac2ad2a7e`, SHA-256
  `0a994382adf17c720c9e5d4292f57344f3b445f058d3e9d5b0f601ae7d6ca3c4`.

Run the README `shasum -a 256 -c` command and require `OK` for both. Set:

```bash
export SATYRN_PROJECT=/Users/koudai/work/satyrn-agent
export SATYRN_TEST_MODEL="$HOME/models/Qwen3.8-27B-Q4_K_M.gguf"
export SATYRN_TEST_DRAFTER="$HOME/models/dflash-Qwen3.8-27B-Q4_0.gguf"
export SATYRN_TEST_LIB=/absolute/path/to/patched/libds4.dylib
```

Build/check the patched library after installing the companion packages. The
ABI manifest must contain all eight Satyrn bindings and the optional core
payload probe; the live library must load against that exact manifest.

Record the Mac model, memory, macOS version, Metal backend, ds4 tree commit,
ds4-py companion commit, Satyrn commit, exact library path, model paths,
revisions, hashes, and all commands.

### 13.2 Launch through project settings plus CLI overrides

Use a scratch read-only demo workspace containing a small `README.md` and a
`pyproject.toml` with:

```toml
[tool.satyrn]
dflash = "auto"
drafter_path = "/Users/koudai/models/dflash-Qwen3.8-27B-Q4_0.gguf"
draft_tokens = 0
think_budget = { max_tokens = 512, bias = 2.0 }

[tool.satyrn.sampling]
temperature = 0.0
top_k = 0
top_p = 1.0
min_p = 0.0
```

From that workspace, launch with DFlash and budget overrides and a committed
trace:

```bash
export DS4PY_HOME=/tmp/satyrn-m3-demo-home
uv run --project "$SATYRN_PROJECT" satyrn-agent \
  --model "$SATYRN_TEST_MODEL" \
  --lib "$SATYRN_TEST_LIB" \
  --metal \
  --dflash on \
  --think-budget 256 \
  --seed 1 \
  --tokens 2048 \
  --read-only \
  --trace /tmp/satyrn-m3-demo.trace.jsonl
```

This one launch proves the project drafter/draft/sampling values, CLI mode and
budget precedence, and project-bias inheritance.

### 13.3 Interactive acceptance script

Perform and record all of the following:

1. **DFlash and acceptance:** submit the corpus prompt
   `Explain speculative decoding in two concise sentences.` Confirm a normal
   answer, `dflash: on`, offered drafts greater than zero, at least one
   accepted draft, and a footer ratio equal to accepted/offered.
2. **Real tools over several turns:** ask the agent to use `read_file` on the
   workspace `README.md` and report its identifying text. Confirm real
   tool-call and tool-result output, then ask a follow-up about the file. No
   approval prompt may appear.
3. **Think budget:** query `/think-budget`; set `/think-budget 2 32` and run the
   checked-in think-budget prompt; then set `/think-budget 256 2`, query it,
   and finally test `/think-budget off`. Generation must continue without an
   injected marker or a hard stop. The existing real capability test supplies
   the controlled proof that this bias changes decode.
4. **Toggle:** run `/dflash off`, confirm the footer says `off`, and run another
   turn. Acceptance counts must not advance. Run `/dflash on`, confirm DFlash
   resumes and new committed acceptance is reported. Do not require off/on
   output text to match numerically.
5. **Persistence and scope:** `/save`, note the ID, then read `payload_bytes`
   from `$DS4PY_HOME/sessions/<id>/save.json` and require it to be positive on
   this supported engine (the `/save` result itself prints ID, title, and token
   count, not payload size). Run `/new` and `/switch <prefix>`. On each new identity,
   confirm the footer resets to startup `on` and `0/0` metrics, and query
   `/think-budget` to confirm the startup `256` token budget with bias `2.0`.
   Continue with a tool or ordinary prompt and verify the restored history is
   used. Fork retention is checked by the automated tests.
6. **Committed tokens:** `/save` again so the final `save.json` and trace cover
   the same history, then parse `/tmp/satyrn-m3-demo.trace.jsonl` and compare
   each model round's `sampled` IDs with the corresponding committed assistant
   transcript segments in the saved records. Assert the
   native-position/trailing-EOS invariant in the live TUI test. For DFlash-on
   rounds, reconcile token count with `committed_blocks + drafts_accepted`;
   for off rounds, require no DFlash metric delta.
7. **M4, when merged:** run the new supported-path real compaction test and
   its stripped-persistence portable counterpart. A `/compact` demo is useful
   evidence but does not replace those assertions.

Attach the trace and the live-test output to the implementation evidence; do
not commit generated GGUFs, native libraries, saved sessions, or traces.

### 13.4 Required gates

For the companion and Satyrn implementation PRs, run their complete static and
portable gates in the canonical devenv + uv workflow, including:

```bash
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
uv run --frozen --extra agent pytest -m "not build and not live" -q
```

Use the satyrn-agent equivalents without the ds4-py `agent` extra, including
its model-free 100% line/branch coverage gate and shell suite. Run the existing
Satyrn abnormal and real-engine E2E suites plus the new harness live tests.
Run `git diff --check` in both repositories.

The ds4-py companion diff's production code and tests must contain no Satyrn
or DFlash identifiers. The no-`--yolo` tests must pass unchanged. Any failure
or guarded skip in the documented real Metal harness run leaves M3 not done.

## 14. Delivery boundaries and builder latitude

### 14.1 Companion PR sequence

1. **ds4-py companion PR: generic harness composition.** Add the explicit
   CLI/session/TUI hooks and public paused-response retirement, with no
   downstream identifiers and no default behavior change.
2. **ds4-py companion PR: generic payload support.** Add the optional core
   probe and stripped/rendered save/switch guards. Rebase the M4-related test
   and guard integration on Roché's final compaction merge.
3. **satyrn-agent M3 PR:** pin the companion commit; add the console command,
   settings resolution, runtime controller, shared recorder, dynamic think
   setting, slash commands, footer, portable scenarios, and live harness
   tests.
4. **Acceptance:** run the complete real Metal script above and record its
   evidence before Paul’s demo.

No product implementation is authorized by this spec branch. Each companion
PR receives its own review, including Roché's review of the generic boundary
and the final M4 contact points.

### 14.2 Builder latitude

The builder may choose exact hook/controller type names, small module
boundaries, lock versus atomic-reference implementation for the think setting,
error wording that preserves the required cause and usage, and test helper
structure. The builder may adjust guard locations to the final M4 code.

The builder may not change the selected generic/downstream boundary, native
worker confinement, session-state lifecycle, command/status contract, config
precedence, verify-mode invariants, stripped-persistence semantics, ADR 0002,
or real Metal Done criteria without another design decision.

There are no intentional open product questions in this design.
