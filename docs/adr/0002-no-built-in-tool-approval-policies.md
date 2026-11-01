---
status: accepted
---

# Core ds4-py does not implement tool approval policies

A tool approval policy controls whether a tool call runs. Core ds4-py does not make that decision or ask for confirmation. It provides tool execution and extension hooks so downstream agents and plugins can implement the policy they need.

## Considered options

- Built-in approval prompts with `--yolo`. Core asks before selected tool calls, and `--yolo` bypasses the prompts. This gives every downstream agent one policy and puts user consent rules in core.
- No approvals and no extension hooks. Core removes approval prompts and blocks custom approval hooks. This removes built-in policy but also prevents downstream agents from choosing their own controls.
- Extension-owned approval policies (chosen). Core provides tool execution and hooks but no approval decisions. Extensions can allow or deny calls, prompt users, and store their own decision state. This keeps policy outside core without preventing stricter downstream behavior.

## Consequences

- The base package has no `DS4Approvals`, approval answerer API, approval prompts, or `--yolo` flag.
- Interactive and noninteractive runs execute tools without confirmation by default.
- An extension can use capability hooks such as `before_tool_execute`. It owns its request and decision types, user interaction, denial results, and stored state.
- Core still enforces explicit workspace limits, such as `LocalWorkspace(read_only=True)`. The caller selects those limits. Core does not turn them into per-call consent decisions.
- Changes to tool execution must preserve enough hook context for extensions to make their own decisions.
