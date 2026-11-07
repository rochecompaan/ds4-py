# M3 targeted certification: 2026-11-06

This report supersedes the [2026-11-01 functional checkpoint](2026-11-01-m3-functional-checkpoint.md). It records the approved targeted-9 result and the repairs that led to it.

## Result

Targeted-9 covered `scale-generator`, `affine-cipher`, `tree-building`, and `zipper`. It passed all six gates. This is a four-task targeted certification, not the 12-task certification. The full certification remains outstanding.

Prefill processes input tokens. Decode produces new tokens. The key-value (KV) cache stores model state from earlier tokens.

| Gate | Result | Key evidence |
|---|---|---|
| 1, task comparison | PASS | C passed `4/4` tasks and ds4-py passed `4/4`. Both arms used five attempts. |
| 2, lifecycle | PASS | ds4-py parsed and executed `38/38` tools. All calls had terminal outcomes. Cap parity passed on `10/10` prompts. |
| 3, rates and cache | PASS | All paired prefill and decode deltas were within `0.25`. All four cache deltas were `0`. |
| 4, interruption | PASS | Ctrl+C interruption, follow-up, and clean exit passed. Structured evidence reported `passed=true` and `valid=true`. |
| 5, restore | PASS | All five save and restore pairs passed checks 5.1 through 5.4. |
| 6A, driver equivalence | PASS | All `77/77` rounds passed with `1,232` decode steps. The maximum logit delta was `4.726562394807843e-07`, below `1e-05`. |
| 6B, self-replay | PASS | All `34/34` rounds passed against the retained ds4-py trace. |

Certification: `targeted`. Result: PASS. Targeted-9 exited `0` in `1h 35m 30s`. It tested revision `3ef9aa5600887e8f9c3e125a6a75e18a04ba310a`. Later commits do not change agent or harness behavior.

The full raw evidence is outside git in `bakeoff-evidence/2026-11-06-gate-refactor-targeted-9/raw/bakeoff/` at the repository root. This directory contains the `bakeoff.json` and `report.txt` summaries. Use these SHA-256 values to make sure that the files on disk match the certified run:

```text
bakeoff.json  0269be1841f3f552f01f555445238617a4edd5e97c5b36db348870140379e31d
report.txt    ac3a0b87a1e502b0ed72d22a70e56f854467dd51697be82ea3ddb3ad7a9bbfcd
```

The [gate criteria design](../specs/2026-11-04-gate-criteria-refactor-design.md) defines the revised checks.

Provenance:

```text
ds4-py revision:       3ef9aa5600887e8f9c3e125a6a75e18a04ba310a
ds4 revision:          5379c1c9ac315f1540371b06a3db88491d4059ff
benchmark revision:    7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f
source archive SHA:    ed1d547a7575b5c33bddfb4ed48042bad871ca97844ebe473878017f761618d2
model SHA:             ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0
libds4.so SHA:         fb89ba682f40acaf44d0206fc3343c86c9d943ae2559a6651baa8303b7cd302b
native patch SHA:      d51a5be286a98b40e1ea1797e3e4170d12b33e1d3fc54ea045c1bd9ed1c3e8d1
run configuration:     response budget 8192, temperature 0, seed 1, context 100000
backend and reset:     cuda, fresh-per-round
host:                  NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 590.48.01
instance:              59386390
```

## How we got here

The 2026-11-03 four-task repair run isolated the cap, wall-time, interruption, and restore failures. Its direct-import and PTY findings were harness defects, not product evidence. The final gate results above supersede its failed or unevaluated gates.

Follow-up trace analysis found repeated recall and deliberation loops, not a native decode-speed deficit: both arms decoded at about 50 tokens/s on that host. This evidence supported terminal and cap checks plus paired rate and cache comparisons, rather than larger response budgets or wall-time ratios.

The two-phase Gate 6 redesign then separated same-host driver equivalence from ds4-py self-replay. Exact C trace replay remains diagnostic because prefill chunking and host changes can flip near-tie token choices. The design spec and retained logits-probe findings record that evidence.

## Repairs

The original failure column comes from the 2026-11-01 checkpoint. The category names the main defect owner. Some harness repairs also supported Gate 5.

| Gate | Original failure | Root cause | Category | Fix commits | Final status |
|---|---|---|---|---|---|
| 1 | PASS: ds4-py passed `12/12` graders. Both arms used 13 attempts. | No defect appeared in this checkpoint. | harness | None | PASS: `4/4` tasks per arm. |
| 2 | FAIL: `tree-building` attempt 1 used all `8,192` response tokens and raised `ResponseTerminalError`. | Lifecycle accounting rejected an expected capped exit. The run also lacked an equivalent C and ds4-py cap check. | harness | `7782489`, `8d94322`, `c498b4f` | PASS: all calls were terminal, and cap parity passed `10/10` prompts. |
| 3 | FAIL: four ds4-py tasks exceeded `1.5` times C wall time. | Total wall time mixed different token volumes and tool histories. A later cache check also paired different sessions. | harness | `7782489`, `72d0cd6` | PASS: paired rates were within `0.25`, and cache deltas were `0`. |
| 4 | FAIL: the PTY driver timed out, and the TUI continued to render output. | Ctrl+C left the active run and prompt in an unfinished state. | agent | `6467239`, `836a239` | PASS: interruption, follow-up, and clean exit passed. |
| 5 | FAIL: 12 follow-ups rebuilt instead of extending KV and diverged from their stripped-save continuations. | Response rebuilds and save round trips lost native sampled tokens. Harness checks also used unequal positions and histories. | agent | `9603c7e`, `5c327d1`, `69624a8`, `a92f565`, `a7b1544`, `3ef9aa5` | PASS: all five pairs passed checks 5.1 through 5.4. |
| 6 | FAIL: exact C trace replay passed `91/287` rounds and `0/196` non-tool rounds. | The old gate required exact replay across hosts and prefill boundaries. Near-tie token choices can differ under those conditions. | harness | `b22449a`, `12e3c2f`, `211953c` | PASS: driver equivalence passed `77/77` rounds, and self-replay passed `34/34` rounds. |

The five agent commits changed these files and defects:

- `6467239`, `src/ds4py/tui.py`: restored the prompt after a terminal interrupt.
- `836a239`, `src/ds4py/runtime.py` and `src/ds4py/tui.py`: finished the active run cleanly at Ctrl+C.
- `a92f565`, `src/ds4py/message_rendering.py`: preserved native token IDs when a response was rebuilt.
- `a7b1544`, `src/ds4py/binding.py` and `src/ds4py/message_rendering.py`: preserved sampled tokens through save and restore round trips.
- `3ef9aa5`, `src/ds4py/binding.py`: retained sampled tokens when equal-content history adoption kept the harness response.

Agent source changes total four files and `+142/-11` lines.

## Run ledger

Raw evidence directories are under `bakeoff-evidence/` at the repository root.

| Run | Outcome | Evidence directory |
|---|---|---|
| targeted-5 | Operational failure: `DS4_LIB` was missing. No ds4-py measurements or `bakeoff.json` were produced. Runtime: `21m 22s`. | `2026-11-05-gate-refactor-targeted-5` |
| targeted-6 | Gates 1 and 3 passed. Gates 2, 4, and 6 failed. Gate 5 was invalid. Runtime: `2h 13m 45s`. | `2026-11-05-gate-refactor-targeted-6` |
| targeted-7 | Operational failure: `DS4_LIB` was missing. All C arms passed, but ds4-py arms failed before traces. Runtime: `15m 21s`. | `2026-11-06-gate-refactor-targeted-7` |
| targeted-8 | Gates 1, 2, 4, and 6 passed. Gate 3 failed, and Gate 5 was invalid. Exit `1`. Runtime: `2h 43m 55s`. | `2026-11-06-gate-refactor-targeted-8` |
| targeted-9 | All six gates passed at `3ef9aa5`. Exit `0`. Runtime: `1h 35m 30s`. | `2026-11-06-gate-refactor-targeted-9` |
| phase-only-a | Engineering verification only. Gate 4 and 13-prompt cap parity passed. Gate 5 failed all six pairs on ordered top-12 mismatch. | `2026-11-06-phase-only-a` |
| phase-only-b | Offline evidence inventory only. No live run or `raw/` tree. | `2026-11-06-phase-only-b` |
| phase-only-c | All six targeted-6 save and restore pairs passed checks 5.1 through 5.4. Runtime: `1h 1m 22s`. | `2026-11-06-phase-only-c` |
| phase-only-d | The capped-save precheck stopped the run because equal-content adoption dropped `ds4.sampled`. No checks 5.1 through 5.4 ran. Runtime: `7m 28s`. | `2026-11-06-phase-only-d` |
| phase-only-e | The capped scale-generator pair passed. The strict precheck stopped the ordinary-stop tree save. A separate stopped-tree diagnostic passed all checks. Runtimes: `14m 47s` and `3m 19s`. | `2026-11-06-phase-only-e` |
