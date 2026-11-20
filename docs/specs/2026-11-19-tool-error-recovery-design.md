# Tool error recovery design

## Intent

Keep a coding run alive when the model repeatedly sends a bad tool call, make an exact-edit miss actionable, and stop the tool prompt from teaching a tool name that is not registered. These are fixes for #25, #26, and #22. Issue #27 and ADR 0002 are unchanged.

## Confirmed causes

- `build_agent_components` leaves both the agent and `DS4FileSystem` retry budgets at the pydantic-ai default of 1. A second `ModelRetry` for the same tool becomes `UnexpectedModelBehavior`; `run_noninteractive` reports that exception and exits 1. A scripted `FunctionModel` reproduced this with two stale `edit_file` calls.
- pydantic-ai-harness 0.59.0 implements exact replacements in `filesystem/_toolset.py::_apply_replacements`. A miss raises only `old_text not found in <path>. No changes were written.` Its `content_hashes` option emits hashes but keeps no last-read state.
- `ToolPromptGenerator._glm` and `._qwen` hard-code a `read` call example regardless of the caller-owned definitions. A prompt containing only `read_file` still teaches `read`.

## Goals and constraints

- One failed tool call and repeated failed tool calls must produce model-visible feedback without ending the run solely because the retry budget ran out.
- A missing `old_text` error must name one nearest file line and its 1-based line number. When an earlier visible `read_file` result supplies a hash for the same path, say whether the current bytes differ from that read.
- Added edit-error detail must be bounded. Do not add line-range editing.
- GLM and Qwen examples must use a placeholder that cannot be mistaken for a registered tool name.
- Use scripted models and temporary files only; no real model runs.
- Make one commit per issue. Run the complete static gate and portable suite.

## Approaches

### Retry exhaustion

1. Raise the retry limits. This postpones termination but retains a fatal boundary, and the #25 follow-up comment does not show that five retries helped.
2. Recover the final retry as a normal tool result. This guarantees that retry exhaustion alone cannot terminate the run and uses the existing `Tool error: ...` convention.

Chosen: option 2. An agent capability recovers only `UnexpectedModelBehavior` directly caused by `ModelRetry` or argument `ValidationError`. Execution-stage exhaustion is recovered on the context's last attempt; the harness's normal success bookkeeping resets that tool's retry count for the next step. Validation-stage exhaustion (unknown tools and invalid arguments) is recovered at the public `CallToolsNode` error boundary: when exactly one tool call remains unanswered, replace the interrupted request with a normal request containing its `Tool error: <original retry message>` result and continue to `ModelRequestNode`. A recovered validation failure leaves that tool's budget exhausted, so a later failure recovers immediately. Ambiguous multi-call failures and all unrelated exceptions still propagate; output-validation retries and event emission are untouched.

### Missing exact-edit text

1. Change pydantic-ai-harness so every filesystem toolset gets the richer error. This is the right upstream default but cannot fix the pinned ds4-py dependency by itself.
2. Wrap ds4-py's filesystem toolset. On the specific single-`old_text` miss, read the current file, add one bounded nearest-line excerpt, and inspect visible message history for the last matching read hash.

Chosen: option 2 now, with the generic diagnostic left available for upstream adoption. Use stdlib line-similarity matching over at most the first 200 characters of each compared line, file lines outer and old-text lines inner, deterministic earliest-line tie breaking, at most 10,000 line comparisons, and a 200-character excerpt cap. Scan read results for the hash header rather than assuming it starts the content, because `ToolOutputLimits` may prepend a spill notice. No state beyond the existing message history is added. If no prior hash is visible, make no staleness claim.

### Prompt example

1. Render an example from a registered definition. This couples example syntax to arbitrary caller schemas and needs a no-tools fallback.
2. Use `$TOOL_NAME`, `$PARAMETER_NAME`, and `$PARAMETER_VALUE`, matching the existing DeepSeek template's explicit placeholders.

Chosen: option 2. Only the GLM and Qwen examples change; serialized tool schemas and all other prompt text stay unchanged.

## Implementation boundary

- Add a focused agent capability for execution- and validation-stage retry-exhaustion recovery and register it in `build_agent_components`.
- Extend `DS4FileSystem`'s existing prepared toolset with edit-error decoration. Hash comparison uses raw current bytes and the last visible `read_file` result for the exact path spelling.
- Replace the hard-coded GLM and Qwen example names and values with placeholders.
- Upstream may later move the nearest-line and staleness diagnostic into pydantic-ai-harness `_apply_replacements`/filesystem toolset. The generic retry-exhaustion policy remains ds4-py's choice.
