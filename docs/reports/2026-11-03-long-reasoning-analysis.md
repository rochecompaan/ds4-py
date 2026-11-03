# Long reasoning and Gate 3 analysis — 2026-11-03

## Scope and conclusion

This report analyzes the four-task targeted bake-off recorded in
[2026-11-03-gate-repairs-targeted-bakeoff.md](2026-11-03-gate-repairs-targeted-bakeoff.md).
The retained evidence came from revision `0ec1288`, with greedy decoding
(`temperature: 0`, seed 1), thinking set to `high`, and an 8,192-token response
budget.

The evidence does **not** point to slow native decode. Both arms decode at about
50 tokens/s on the same model and host. The failed ds4-py attempts instead spend
8,192-token rounds on repeated recall or deliberation without reaching a file
write. Those rounds add both decode time and another attempt's fixed startup
cost.

The primary repair direction is therefore to change agent behavior so it writes
a best-effort implementation earlier. More response budget would let the
unproductive rounds run longer. It would not address the observed failure.

Gate 2 checks errors from **all recorded attempts**, not just first attempts.
A later successful attempt does not clear an earlier `max_tokens` error
(`bench/report.py`, `bench/run_bakeoff.py`). Gate 3 also sums wall time and tokens
across attempts, and requires ds4-py wall time to be at most 1.5 times C
(`bench/metrics.py`, `bench/report.py`).

## Evidence and method

The local evidence root was:

```text
/home/roche/projects/satyrn/ds4-py-evidence/full/gate-repairs-0ec1288
```

For every task, arm, and retained attempt, the analysis used:

- `bakeoff.json` for grader results, attempt counts, errors, tokens, and measured
  end-to-end wall time;
- ds4-py `trace.jsonl` and the matching session `events.jsonl` for prompts,
  sampled tokens, lifecycle timestamps, terminal reasons, tools, and text;
- C `trace.txt` for prefill and generation timings and C `agent.log` for actions
  and reasoning;
- complete `TextDelta` text for every 8,192-token ds4-py round, plus the long
  successful retry rounds and the corresponding long C paths.

“Active time” below means ds4-py `run_start` to `run_end`, or C worker start to
the last `generation finished`. It excludes some process setup, model loading,
grading, and save work. Gate 3 uses the larger end-to-end wall measurement, not
active time. ds4-py decode time is first `Token` to `Stopped`; C decode time is
`prefill sync done` to `generation finished`. These are comparable lifecycle
intervals, though not claims of instruction-level profiling.

## Timing and token evidence

### Gate measurements

| Task | C wall (s) | ds4-py wall (s) | Wall ratio | C tokens | ds4-py tokens | Token ratio | Attempts C / ds4-py |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scale-generator | 103.01 | 383.69 | 3.725x | 4,171 | 16,782 | 4.023x | 1 / 2 |
| affine-cipher | 58.04 | 305.13 | 5.257x | 1,985 | 12,894 | 6.496x | 1 / 2 |
| tree-building | 137.13 | 300.81 | 2.194x | 4,755 | 12,914 | 2.716x | 2 / 2 |
| zipper | 265.74 | 281.82 | 1.060x | 11,871 | 11,811 | 0.995x | 1 / 2 |

The ratios track generated volume and attempts, not a decode-speed deficit.
Aggregate decode throughput by task was 50.0–51.2 tokens/s for C and
49.6–50.1 tokens/s for ds4-py.

Total prefill time remained small: at most 11.6 seconds for C and 10.0 seconds
for ds4-py on any task. Recorded ds4-py tool execution totaled less than 0.2
seconds per task. These costs cannot explain the Gate 3 failures.

The difference between gate wall and active time was about 34–36 seconds for
two ds4-py attempts, about 15 seconds for one C attempt, and about 30 seconds
for two C attempts. This is consistent with the roughly 14-second model startup
shown in the logs plus other per-attempt work. Avoiding an unnecessary attempt
therefore saves more than its generated tokens.

### Per-attempt behavior

| Task | Arm and attempt | Tokens | Rounds | Active (s) | Longest round tokens | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| scale-generator | C 1 | 4,171 | 7 | 87.6 | 2,869 | pass |
| scale-generator | ds4-py 1 | 8,414 | 3 | 173.5 | 8,192 | `max_tokens`; no write |
| scale-generator | ds4-py 2 | 8,368 | 4 | 174.7 | 8,192 | `max_tokens`; no write |
| affine-cipher | C 1 | 1,985 | 8 | 42.9 | 838 | pass |
| affine-cipher | ds4-py 1 | 8,432 | 4 | 175.5 | 8,192 | `max_tokens`; no write |
| affine-cipher | ds4-py 2 | 4,462 | 8 | 95.2 | 2,562 | pass |
| tree-building | C 1 | 4,367 | 14 | 96.3 | 1,510 | grader fail |
| tree-building | C 2 | 388 | 4 | 10.6 | 216 | pass |
| tree-building | ds4-py 1 | 8,448 | 4 | 173.6 | 8,192 | `max_tokens`; no write |
| tree-building | ds4-py 2 | 4,466 | 6 | 93.1 | 3,461 | pass |
| zipper | C 1 | 11,871 | 19 | 250.6 | 6,738 | pass |
| zipper | ds4-py 1 | 8,537 | 5 | 175.6 | 8,192 | `max_tokens`; no write |
| zipper | ds4-py 2 | 3,274 | 9 | 72.2 | 2,089 | pass |

Zipper is the clearest warning against using token totals alone. C and ds4-py
generated almost the same total tokens and had nearly equal decode time. C spent
its tokens in one successful attempt. ds4-py spent 8,537 tokens on a capped
attempt and 3,274 on a successful one, adding a second startup and about 19
seconds of non-active wall time. Its active time was actually slightly lower
than C's, while its end-to-end wall was 6% higher.

## What the long rounds contain

### scale-generator

The first ds4-py long round repeatedly reconstructs an incomplete “canonical”
solution. The same incomplete `interval` snippet recurs throughout the round;
its `def interval` line appears 19 times. The model shifts
between recalling tests and deriving the algorithm, but never emits a write. The
round ends mid-recall at the token cap.

The second attempt receives grader feedback but still does not write. While
checking D dorian, it moves from A at index 0 by a whole step and incorrectly
selects index 1 (`Bb`) instead of index 2 (`B`). It then treats the resulting
mismatch as a contradiction in the task and spends the rest of the round
reconsidering sharp/flat rules until the cap.

C reaches a correct implementation after a 2,869-token longest round and local
checks. The two failed ds4-py attempts are the only task failures without any
successful ds4-py trajectory to use as a timing estimate.

### affine-cipher

After a short, correct summary of the algorithm, the first ds4-py long round
starts listing supposed canonical encode cases. The same block repeats about
16 times. Of 450 non-empty lines, only 41 are unique; this is exact repetitive
output, not useful test execution. The model never reaches an implementation.

C implements and checks the algorithm in 1,985 tokens. The ds4-py retry writes
after receiving grader feedback and passes in 4,462 tokens. Its longest retry
round is still verbose: it repeatedly traces incorrect extended-Euclidean
variants before fixing the inverse. Thus the retry is successful, but not an
efficient first-attempt model.

### tree-building

The first ds4-py long round is mainly deliberation rather than an exact repeat
loop. It repeatedly debates whether to preserve the stub's error strings or use
remembered canonical messages, reconstructs test files, and restarts the design.
It ends mid-analysis before a write.

C uses `google` and `curl` in this trace to retrieve the public Exercism tests,
then writes and runs them. Its first attempt has one punctuation mismatch
(`its` versus `it's`); its second attempt fixes that in 388 tokens. The ds4-py
retry also uses the grader feedback to select the expected messages and passes,
though it still spends 3,461 tokens in its longest round.

### zipper

The first ds4-py long round recalls test methods, then repeats the same pair of
test-like lines 74 times. Of 414 non-empty lines, only 98 are unique. It reaches
the cap without implementing the stub.

C retrieves the public test file and official example. Its longest round is
6,738 tokens, but the text is productive design analysis: it checks mutation and
immutability, navigation after edits, and reconstruction through ancestors. It
then writes and runs the tests. The ds4-py retry writes a correct implementation
after a 2,089-token design round and passes in 3,274 total tokens.

### Cross-task interpretation

All four failed first ds4-py attempts hit `max_tokens` before any file write.
Affine-cipher and zipper show strong exact repetition. Scale-generator and
tree-building are better described as recall and deliberation loops. Long text
alone is not the failure: C's 6,738-token zipper round and the successful ds4-py
retries end in useful tool calls.

C also used public web retrieval for tree-building and zipper, while ds4-py did
not. This is a prompt/tool-affordance difference worth testing, not proof that
ds4-py cannot retrieve public references: ds4-py has `run_command`, but did not
choose that path. C did not need web retrieval for scale-generator or
affine-cipher, so web access cannot be the complete explanation.

The run used greedy decoding, so lowering temperature is not an option. The
current evidence does not establish an engine output defect: the arms have
different prompts, tool schemas, and trajectories, and Gate 6 replay was not
evaluated in this four-task run. A controlled replay remains the right check
before making any engine output change.

## Regression predictions

These are bounded predictions from retained timings, not guarantees of model
behavior.

### Gate 2

A retry that passes the grader does not repair Gate 2 if any recorded attempt
still has a `max_tokens` error. Every attempt in the next evidence set must end
without that error. Scale-generator is the highest risk because both retained
attempts capped.

Increasing the response budget is not a predicted fix. The affine-cipher and
zipper rounds already repeat stable blocks; more budget can extend the loop and
increase Gate 3 wall time. A smaller budget would only turn the same behavior
into an earlier error.

### Gate 3

The Gate 3 wall limits for these C baselines are:

| Task | 1.5x C wall limit (s) | Optimistic one-attempt ds4-py estimate (s) | Prediction |
| --- | ---: | ---: | --- |
| scale-generator | 154.52 | not estimable | No successful ds4-py attempt; any capped round is already too slow. |
| affine-cipher | 87.06 | 112.43 | Still fails; removing only the failed first attempt is insufficient. |
| tree-building | 205.69 | 110.17 | Likely passes if a fresh first attempt resembles the retry. |
| zipper | 398.62 | 89.22 | Already passes; removing the failed first attempt would add margin. |

For the three successful retries, the optimistic estimate substitutes one retry
for the whole ds4-py task and adds the observed per-attempt non-active overhead
of about 17 seconds. This is favorable to ds4-py because the retries had grader
feedback that a fresh first attempt would not have.

Even under that favorable assumption, affine-cipher remains about 25 seconds
over its Gate 3 limit. At the observed decode rate and other retry costs, it
needs roughly 3,200 generated tokens rather than 4,462 to fit comfortably. It
therefore needs both elimination of the capped attempt and shorter successful
reasoning. Tree-building has more room; zipper already demonstrates that nearly
equal token totals can pass when other costs remain controlled.

## Ranked experiments

All behavioral experiments should use fresh, one-attempt targeted runs with
three repetitions per task. Do not use second-attempt success as the main
acceptance evidence. Record time to first write, total tokens, end-to-end wall,
terminal reasons, grader result, and the longest round's text. Do not start
another full bake-off until the selected tasks pass these checks and their
one-attempt wall times fit the Gate 3 limits above.

### 1. Early-write prompt variant

Test a narrow instruction change first:

> Use the provided task and local documentation. Do not spend extended reasoning
> trying to recall a canonical solution or test file. After initial inspection,
> write a best-effort implementation, run local checks, and refine it.

This directly targets all four failed trajectories without changing engine
behavior. Keep the greedy decoding and 8,192-token budget fixed so the comparison
isolates the prompt effect.

Acceptance requires no `max_tokens` errors, a real implementation write, passing
local grader evidence, and wall within the task's Gate 3 limit. Inspect the long
rounds as well as the scalar results; a faster but still repetitive trajectory is
not a durable repair.

### 2. Public-reference/tool-parity variant

If recall loops remain, test an explicit public-reference path for tree-building
and zipper. For example, instruct the agent that it may use available shell
network tools to retrieve public exercise documentation or tests, or provide an
equivalent web tool to both arms. Do not expose benchmark grader files or their
workspace.

Measure whether this moves the agent from recall to retrieval and shortens time
to first write. This experiment addresses a difference visible in the C logs,
but it should not replace experiment 1: C solved scale-generator and
affine-cipher without web retrieval.

### 3. Targeted native parity diagnostic

Before considering an engine output repair, replay the retained C traces through
the ds4-py native path and compare committed token IDs and terminal boundaries.
The four-task bake-off did not evaluate Gate 6, and the 16-token standalone
diagnostic is not evidence about these long trajectories.

Exact replay would support a context/agent-behavior explanation. A divergence
would identify a concrete round for engine debugging. This is a diagnostic, not
permission to change sampling or engine code without a minimal failing case.

### 4. Think-level ablation

Only after checking that a different think level changes this model's rendered
prompt, compare `high` with `medium` under the same prompt and budget. Keep both
arms on the same setting for a fair bake-off. This may reduce deliberation, but
it can also remove useful design work such as C's long zipper round.

Do not use a larger response budget as the repair variant. The retained text
shows why that can increase cost without producing an action.
