# Plan: preserve streamed TUI lines

Issue: #36

## Confirmed cause

The supplied pty repro reliably loses text printed delta by delta with `end=""` while a `PromptSession` is active. The control prints the same text once with a trailing newline and keeps it visible. `StreamRenderer.handle_event` currently takes the first path, so prompt-toolkit app redraws overwrite unfinished lines. Saved model output is not involved in the loss.

## Design

Keep the change inside the TUI rendering boundary:

1. Add a styled-fragment pending buffer to `StreamRenderer`.
2. For text and thinking events, append fragments and print only through the last newline. Preserve each fragment's existing style and preserve blank lines. Continue returning the event's original fragments for existing callers.
3. Add `flush()` to print a nonempty pending remainder with a trailing newline. Make it idempotent so adjacent boundaries do not add blank lines.
4. Flush on `PartEndEvent`, immediately before tool call/result lines, and when the TUI receives `run_end`. Flush during TUI shutdown as a final cancellation/error safeguard. Before displaying an error or command message, flush pending model text to preserve ordering.
5. Leave footer rendering, slash-command formatting, tool preview text, and Markdown conversion unchanged. `show_message` will flush only the renderer's current buffer; already queued agent events retain their existing FIFO behavior.

## Tests (red before implementation)

- Incomplete text and thinking deltas produce no terminal write; a newline emits the complete styled line.
- `PartEndEvent` emits an unfinished remainder exactly once.
- Pending text is emitted before a tool line, retaining tool output order.
- A TUI `run_end` event and shutdown flush any remainder without duplicating completed lines.
- The real `show_message` emits a pending remainder before an error or command message.
- Existing style, Markdown, tool preview, footer, and command tests remain unchanged and passing.

Observe renderer writes by monkeypatching `ds4py.tui_render.print_formatted_text`; assert the real formatted fragments and `end` arguments at that output boundary rather than relying on `DummyOutput`.

## Verification

- Focused renderer/TUI tests.
- Original external pty repro adapted to use `StreamRenderer`.
- Real ds4-py TUI with `FakeNative`, driven in a pty through pyte; keep the script outside the repository and include its final screen in the PR. Run it with `uv run --no-project --with pyte python /tmp/<driver>.py <worktree-python>` so pyte remains outside project dependencies.
- `uv run --frozen --extra agent prek run --all-files --show-diff-on-failure`
- `HOME=$(mktemp -d) uv run --frozen --extra agent pytest -m "not build and not live" -q`
- Commit with `Fixes #36`, push, open the PR, and run the required review loop.
