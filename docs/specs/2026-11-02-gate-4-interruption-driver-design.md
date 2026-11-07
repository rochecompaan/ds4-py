# Gate 4 structured interruption driver design

**Status:** Approved
**Date:** 2026-11-02
**Gate:** Gate 4 — interruption and follow-up

## Intent

Repair the Gate 4 acceptance driver so it sends Ctrl+C during actual generation and verifies the complete interrupt, follow-up, and exit lifecycle. Keep the production cancellation path unchanged unless a focused integration test exposes a separate defect.

## Evidence and root cause

The retained Gate 4 run emitted 2,466 token events across seven model rounds and ended with `run_end=success`. It contained no interrupted terminal event. The PTY driver waited for the literal footer text `streaming`, never delivered an effective Ctrl+C, and timed out while model output continued.

The footer is a presentation surface, not a reliable process-control interface. A PTY probe also showed that setting a nonzero terminal size alone did not make the footer appear. The first repair therefore belongs in the acceptance driver, not in native cancellation.

## Goals

- Send Ctrl+C through the real PTY after the active run emits its first `Token` event.
- Require an interrupted terminal event for that run before submitting a follow-up.
- Verify that the follow-up completes and `/exit` closes the TUI cleanly.
- Preserve the PTY transcript and structured events as acceptance evidence.
- Add non-live regression coverage for the driver and the real TUI cancellation path.

## Non-goals

- Do not change native or runtime cancellation behavior without a failing integration test that localizes a second defect.
- Do not make a successful, uninterrupted run satisfy Gate 4.
- Do not relax the timeout or accept visible output as proof of interruption.
- Do not run a live bake-off before the other reported gate repairs have local verification.

## Approaches considered

1. **Structured session events with PTY input — selected.** Watch the existing `events.jsonl` for lifecycle transitions while sending Ctrl+C and prompts through the PTY. This tests the real user path without depending on ANSI rendering.
2. **More ANSI scraping.** Set the PTY size and adjust footer patterns. This remains coupled to rendering and was not sufficient in the probe.
3. **Machine-readable production status output.** Add explicit lifecycle lines or a control descriptor. This would make the driver simpler but changes a production interface only to satisfy its test.

## Design

Extend the Gate 4 driver to combine PTY output with a tolerant follower for the new session's append-only `events.jsonl`:

1. Start with a fresh acceptance home and wait for its session event stream.
2. Correlate events by `run_id`; ignore other runs and tolerate an incomplete final JSON line while the writer is active.
3. On the first `Token` for the initial run, send one Ctrl+C through the PTY.
4. Require `Stopped` with reason `interrupted` for that run, followed by its `run_end`, before sending the follow-up.
5. Require the follow-up run to finish without an error, send `/exit`, and require process exit status 0.
6. On timeout or unexpected process exit, kill and reap the child and retain bounded PTY output plus the relevant lifecycle failure.

Keep the existing generic PTY helper for output-only tests unless sharing this lifecycle logic is simpler than maintaining two implementations.

## Tests

- Reproduce the old failure: visible generation followed by ordinary success must fail rather than time out silently or pass as an interruption.
- Prove the structured driver sends Ctrl+C on a `Token`, waits for the matching interrupted run, and does not act on another run's events.
- Exercise the real TUI with the fake engine: Ctrl+C during generation produces an interrupted round, returns control to the prompt, and permits a successful follow-up and clean exit.
- Run the focused tests, full non-live suite, and full `prek` quality gate.

If the TUI integration test fails inside cancellation, diagnose that failure separately before changing production cancellation code.

## Acceptance

Non-live acceptance requires all new regressions and existing checks to pass. Live Gate 4 acceptance remains pending the single later GPU session; it requires an actual interrupted generation, a successful follow-up, clean exit, and retained structured evidence.
