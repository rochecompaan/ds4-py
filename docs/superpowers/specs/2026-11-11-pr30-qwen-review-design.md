# PR #30 Qwen Review Follow-up Design

## Intent

Address every review comment on PR #30 without changing the approved issue #29 behavior: malformed or non-JSON-looking Qwen parameter content remains text, while supported non-string JSON values retain their types.

## Goals

- Parse each Qwen parameter once and use that same result for streaming conversion and history-rendering validation.
- Document the deliberate combination of the `ds4-agent` numeric rule and the `ds4-server` complete-JSON rule.
- Run all new edge-case vectors through the existing fixture-driven chunk-equivalence tests.
- Remove duplicate parser coverage and simplify the Qwen renderer branches.
- Reply to each inline GitHub review thread after its corresponding fix or clarification.

## Non-goals

- Do not plumb tool schemas into parser factories in this PR. Schema-declared strings remain a documented follow-up, as already stated in the PR body.
- Do not adopt C `strtod` extensions such as `NaN`, `Infinity`, or hexadecimal floats.
- Do not change GLM or DSML parsing.

## Constraints

- After the pinned leading/trailing trim, values fully matching `agent_qwen_value_is_json`'s ASCII numeric grammar use its 64-byte cap. The grammar is ASCII-only, so character and byte counts are equal for candidates. Other values use a complete strict JSON parse inspired by `ds4_server.c:qwen_param_value_is_json`; trailing content, top-level strings, and non-finite constants remain text. A value that does not fully match after that trim (for example, 64 digits plus `\\r`) belongs to the JSON branch, not the capped numeric branch.
- Preserve existing accepted numeric forms (`+1`, `01`, `1.`, and `.5`) and the existing renderer errors for values that cannot round-trip.
- Keep the existing devenv + uv workflow and run the complete static and portable test gates.

## Approaches

1. **Single parse helper (selected).** Introduce `parse_qwen_value(text: str) -> tuple[object, bool]`, returning the converted value and whether the source is text. This removes duplicated trimming, numeric checks, JSON parsing, and the unguarded second parse. It directly matches the review recommendation.
2. **Retain two passes and restore the exception guard.** This is a smaller textual change but leaves duplicated rules and their future drift risk, so it does not address the full review.
3. **Add schema-aware parsing now.** This would address the remaining valid-JSON string case but requires retaining schemas through sessions, streams, pause/fork/restore, and parser factories. The reviewer explicitly allowed this to remain a follow-up, and the PR body already records it.

## Implementation outline

Use a string numeric regex and one trimming implementation in `parse_qwen_value`. Return the original text for string results; the streaming parser retains responsibility for delimiter unescaping. Keep the single `RecursionError` guard: CPython 3.14.7 raises it for 200,000-deep complete arrays, and the current parser degrades that input to text. Update `render._qwen` to consume the helper flag and assign `value_text` once per branch. Fold the parameterized edge cases into existing or new `qwen.json` entries without duplicating BOM/malformed vectors, delete the duplicate Jinja test, and use fixture labels that distinguish the strict JSON and agent numeric rules.
