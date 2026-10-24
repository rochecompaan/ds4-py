---
status: accepted
---

# ds4-py does not share session files with ds4-agent

The first design required that a session saved by `ds4-agent`, the C agent in antirez/ds4, open in ds4-py, and the other way round. Nobody needs that, and it was the main cost of milestones 2 and 3. A saved session holds the KV cache for an exact list of tokens, so sharing it meant matching C byte for byte: the same tools, the same tools prompt, the same tool output and C's KV file format. That required a port of all 11 C tools, including a Chrome driver, plus golden tests for every tool and prompt. So ds4-py drops session compatibility. Its tools come from the Pydantic AI harness, its prompts are its own, and it saves sessions in its own format under `~/.ds4py/`.

## Considered options

- **Keep two-way compatibility.** This is what the first design did. It costs about 2,000 lines (the tool port and the vendored C prompts) and a set of byte-for-byte fixtures that must be regenerated whenever the upstream pin moves.
- **Read C's KV files but don't write them.** This keeps replay parity, the test that ds4-py's token loop matches C's, on C's saved sessions. It is not needed: `ds4-agent --trace` already records the system prompt, each round's prompt tokens and each sampled token by id. It also works with `--non-interactive`, while saving a KV file needs `/save` in an interactive session.
- **No compatibility, with replay parity read from `--trace`** (chosen). ds4-py reads no C session files at all.

## Consequences

- The C chat templates, the token transcript as the source of truth and the four tool-call parsers stay. ds4-py needs them to run these models at all, whatever its tools are.
- Replay parity still tests the token loop against C. Each replayed round starts from the tokens C saw, including C's tool calls and results, so it does not depend on ds4-py's tools.
- The bake-off's pass-rate gate no longer finds loop bugs. It becomes a minimum bar for the harness tools on these models. With 12 tasks, it can catch tools that clearly do worse than antirez's, not small differences (spec section 21).
- ds4-py saves under `~/.ds4py/`, not `~/.ds4/kvcache/`. Once the two system prompts differ, a shared directory would mix the two agents' sessions in `/list`, and each agent would overwrite the other's `sysprompt.kv` on every start.
- If the bake-off shows that these models need C's tools, a single tool, such as `edit`, can come back as an optional capability. It does not have to match C's bytes.
