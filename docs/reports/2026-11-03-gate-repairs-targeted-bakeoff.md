# Gate repair targeted bake-off

## Scope

This run measured four tasks on Vast.ai instance `58938373`. It tested the current workspace isolation, lifecycle, and restore repairs against a fresh C baseline. It was not a full bake-off.

The main run used ds4-py revision `0ec1288436dbcac53966a03698a5ca5f40afa328`. The C source used revision `5379c1c9ac315f1540371b06a3db88491d4059ff`. The benchmark used revision `7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f`.

Both agents used CUDA, context 100000, response budget 8192 tokens, temperature 0, and seed 1. The shared native library had SHA-256 `6126230f8b502178e1902bd5c7b78db31ce42cbf3f9ec177e5e6a093680f542c`.

## Result summary

C passed all four tasks. ds4-py passed three tasks and failed `scale-generator`. Gate 1 passed. Gates 2 through 5 still need repairs. The targeted manifest cannot evaluate Gate 6.

| Gate | Assessment | Evidence |
|---|---|---|
| 1. Task comparison | PASS | C passed 4/4. ds4-py passed 3/4. C used 5 attempts. ds4-py used 8. |
| 2. Lifecycle | FAIL | Every first ds4-py attempt hit the response budget. Both `scale-generator` attempts hit it. |
| 3. Performance | FAIL | ds4-py exceeded 1.5 times the C wall time on three tasks. |
| 4. Interruption | FAIL | The first run stopped and saved, but the follow-up run did not start. |
| 5. Restore | FAIL | All eight restore case 1 processes failed before native replay. |
| 6. Replay | NOT EVALUATED | The selector requires 12 distinct C tasks. This manifest supplied 4. |

The raw report marks Gate 6 as failed because selection found no eligible case. This report treats that result as a scope limit, not as evidence of a replay regression.

## Task measurements

Wall time includes all attempts for the task.

| Task | C attempts | C result | ds4-py attempts | ds4-py result | C tokens | ds4-py tokens | C seconds | ds4-py seconds |
|---|---:|---|---:|---|---:|---:|---:|---:|
| `scale-generator` | 1 | PASS | 2 | FAIL | 4171 | 16782 | 103.012 | 383.693 |
| `affine-cipher` | 1 | PASS | 2 | PASS | 1985 | 12894 | 58.043 | 305.134 |
| `tree-building` | 2 | PASS | 2 | PASS | 4755 | 12914 | 137.125 | 300.809 |
| `zipper` | 1 | PASS | 2 | PASS | 11871 | 11811 | 265.744 | 281.817 |

`zipper` passed Gate 3 with almost equal token counts and wall times. The other tasks retained large token differences. This run did not isolate a native generation-throughput defect.

## Gate 2: response budget exhaustion

Every first ds4-py attempt ended with `ds4py: model response ended with max_tokens`. The second attempts passed `affine-cipher`, `tree-building`, and `zipper`. Both `scale-generator` attempts hit the budget and failed.

The isolated workspace did not remove the long generation behavior. The staged grader remained outside each agent workspace. The agent workspace contained only the student files, `TASK.md`, and visible `docs`.

The current evidence does not justify another budget increase. It also does not justify a native performance change by itself. The next analysis must identify the long reasoning rounds and the work that they repeat.

## Gate 4: interruption and follow-up

The first run emitted `Stopped(reason=interrupted)` and `run_end`. It then emitted save events. This confirms that the first interruption and save phase completed.

The driver then wrote the follow-up prompt. No new `run_start` occurred. The process raised `KeyboardInterrupt` in `PromptSession.prompt_async` and emitted `session_end`. The PTY driver reported `process exited while waiting for follow-up run_start`.

The remaining defect is the prompt lifecycle after interruption. The first cancellation phase is not the current failure point.

## Gate 5: restore setup

All eight save directories exist. Their restore case 1 logs contain the same error:

```text
pydantic_ai.exceptions.UserError: `Skills` needs a workspace.
```

The restored message history names a `local:` workspace. The restore process did not attach a capability that provides that workspace. The failure occurred before native replay, so this run provides no evidence about token-level restore correctness.

The Gate 5 summary text says `missing or failed restore evidence`. In this run, the specific condition is failed restore setup, not missing saves.

## Gate 6: targeted evidence

The bake-off replay selector reported `gate 6 needs 12 distinct C tasks, found 4`. It did not run a replay comparison. A full 12-task manifest is required for formal Gate 6 acceptance.

The standalone C control used revision `d163c2c` and the same native library. It completed the 16-token split-prefill control with `first_divergence = -1`. This is a passing targeted control, not full Gate 6 acceptance.

## Harness repairs during this run

The first execution attempt exposed two direct-script defects. `restore_check.py` failed to import `bench` from its worker process. `bench/pty.py` also shadowed the standard-library `pty` module.

Revision `0ec1288` added the repository root to the restore worker import path. It renamed the PTY driver to `bench/pty_driver.py`. Two regression tests reproduce the original failures. The full local suite passed with 1575 tests and 17 skips.

The earlier `d163c2c` task results did not execute ds4-py and are not product evidence. The completed `0ec1288` results supersede them.

## Next repairs

1. Attach the saved local workspace when constructing the restore agent. Add an integration test that restores history containing `local:` workspace metadata.
2. Repair the TUI prompt lifecycle after interruption. Use a real PTY test that requires a follow-up `run_start` and a clean exit.
3. Analyze the long generation rounds before changing the response budget or native engine. Record the repeated work and its token cost.
4. Run Gate 6 with all 12 tasks after the lifecycle and restore repairs. Keep the standalone C control as targeted evidence.

## Evidence

The project evidence directory is `bakeoff-evidence/2026-11-03-gate-repairs-0ec1288/` at the repository root. It contains the main report, configuration, agent logs, restore logs, interruption events, and C diagnostic.

The full non-KV evidence archive is `/home/roche/projects/satyrn/ds4-py-evidence/gate-repairs-0ec1288-non-kv.tar.gz`. It also contains the raw runner and terminal logs. Its SHA-256 is `400c186e3dd5fd32e30f2c9f9e00fbdacb19a37d3c3f8eaff89688fe8849b4cd`.

The remote directory `/workspace/results/gate-repairs-0ec1288/` retains the complete evidence, including KV saves. The archive excludes those 1.5 GB of binary saves.
