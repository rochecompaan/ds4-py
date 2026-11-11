# Fatal-State Poisoning Design

## Intent

Let a multi-token decode step mark its owning `EngineWorker` unusable when a native commit or cleanup failure leaves native state uncertain. The worker remains the only native owner, but no later ordinary operation may enter native code. Owner-driven cleanup remains available.

This implements section 8.5 of the agent package design while preserving the tentative-publication guarantees from section 8.2.

## Goals

- Add `DecodeContext.poison(error: BaseException) -> Never` as a generic fatal-error operation.
- Poison the existing owning worker and re-raise the exact supplied exception without wrapping or replacing it. `poison()` is a terminal, single-use operation for the decode step.
- Reject every later native-capable `_Call` and submitted `Job`, including work queued before poisoning, before invoking its callback, job handler, native double, or real native binding.
- Cover public engine/session facades, rounds, sync/eval/rewind, checkpoints/snapshots, payload jobs, extension callbacks, and `on_worker` through the common worker dispatch boundary.
- Complete every rejected future or event channel promptly with an error; no waiter may hang.
- Preserve the initial native error, cleanup error identity, and normal Python exception chaining.
- Keep `Engine.close()` and worker shutdown available and release snapshots, the session, the engine, and the worker thread exactly once.
- Keep session lifecycle operations from clearing poisoning or providing implicit recovery.

## Non-goals

- Automatically deciding whether an arbitrary native exception makes state uncertain; the decode step owns that decision and calls `poison()` explicitly.
- Adding recovery, a replacement worker/thread, native structs, bindings, or native APIs.
- Changing direct `PlainStep`, adding staging allocations to it, or selecting it after a block starts or after a failure.
- Changing native eval, rewind, commit, or cleanup ordering inside a decode step.
- Adding product-specific names or policy.
- Claiming real-engine parity from `FakeNative` tests.

## Anti-goals

The change fails if a supported decode-step exit path performs further native cleanup after poisoning, if a rejected caller waits forever, if publication pretends an uncertain staged prefix committed, or if `Session.new()`, reset, switch, or fork makes the worker usable again.

The worker gate is an operation-routing guarantee, not a containment boundary for arbitrary extension code. A decode step receives a borrowed raw `SessionHandle`; it must not suppress the exception from `poison()` and continue using that handle. The design does not claim to sandbox a deliberately non-cooperative step inside the already-running native call.

## Constraints

- Base the branch on latest `main` at PR #16 (`36f73e9`).
- Keep all native ownership and cleanup on the existing worker thread.
- Use focused RED-GREEN TDD for every runtime behavior change.
- Preserve existing tests unless an assertion conflicts with section 8.5; replace only such an assertion with a stronger safety-contract assertion.
- Run the complete static gate, portable regression suite, focused engine suites, and `git diff --check` inside the canonical devenv + uv workflow.
- Keep the completed branch available without merging or pushing.

## Approaches Considered

### Worker dispatch gate and direct poisoning callback

Selected. `EngineWorker` passes a private poisoning callback through `generate()` into `DecodeContext`. The worker stores the fatal exception and checks its state before every queued call or job. `_STOP` and `ExitStack` cleanup bypass that ordinary-operation gate.

This gives every entry route the same behavior and avoids duplicating checks across public methods. It also avoids an exception wrapper on the initial failure.

### Facade-level guards in addition to worker guards

Rejected as unnecessary for this change. They could reject before queue admission, but would duplicate state inspection across many methods while direct worker jobs and callbacks still require the central gate. Queue admission and result semantics can remain unchanged when dispatch rejects promptly.

### A fatal wrapper exception caught by the worker

Rejected. Unwrapping or replacing it would make initial exception identity, traceback, and cleanup chaining harder to preserve. Directly re-raising the supplied exception is simpler and stronger.

## API and Error Design

`DecodeContext.poison(error)` performs only Python state changes:

1. Invoke the owning worker's private poison callback, which records `error` as the worker's fatal exception.
2. Raise that exact `error` object.

A decode step calls `poison()` exactly once on its terminal abnormal path. The step must not catch the resulting exception to attempt another native operation or a second poisoning call.

Worker-free unit contexts may omit the private callback; `poison()` still re-raises the exact exception. Normal worker-backed generation always supplies the callback.

Add public `WorkerPoisonedError(EngineError)` and export it from `ds4py`. Every later rejected operation raises a new `WorkerPoisonedError("engine worker is poisoned")` with the recorded fatal exception as its explicit `__cause__`.

If no cleanup error occurred, callers can follow `rejected.__cause__` directly to the original native error. If cleanup failed and that cleanup exception was supplied to `poison()`, `rejected.__cause__` is the exact cleanup exception and its normal `__context__` is the original native error. The initial job receives that same original or cleanup exception unchanged.

## Dispatch and Completion Design

The worker checks poisoning only on its owner thread, after dequeuing an ordinary item and before any user callback or job handler:

- For `_Call`, mark the future running, then perform the check inside the existing exception-transfer block. A rejected future receives `WorkerPoisonedError`, so `result()` and `exception()` terminate normally.
- For `_Submission`, perform the check inside the existing job exception handler. Publish the error, then always publish `_END` and set `finished` in `finally`. Event iteration raises promptly and cleanup barriers cannot hang.
- Check `_STOP` before poisoning. `close()` therefore still stops and joins the thread after previously admitted work has failed promptly.

Poisoning is permanent. Session `new()` and reset reach a rejected reset job. A failed `new()` retains the existing failure ordering: the old ID remains, scoped states have been closed, `session_end` was emitted, and no `session_start` is emitted. This does not recover or replace the worker. Pure-Python `switch()` and `fork()` may still change identity, but neither clears the recorded error; a switch restore or any later native operation remains rejected.

## Decode Failure and Publication Design

A decode step calls `poison()` only after its required native cleanup has succeeded or failed. Once called, the operation raises and the step must not attempt native eval, rewind, commit, or cleanup again, including from its own `finally` blocks. A `finally` block after poisoning may perform only Python-local bookkeeping. Tests pin this contract for the supported multi-token failure path; they do not represent the raw handle as a security boundary against deliberately non-cooperative code.

`generate()` retains its existing abnormal exit: an active overlay is discarded using Python state only, and no finalization or publication occurs. Staged tokens and buffered events therefore remain unpublished when native state is uncertain. The existing worker catches and transfers the exact exception without converting the step to `PlainStep`.

A step that can still establish a certain kept native prefix without a fatal commit/cleanup failure retains section 8.2 behavior: commit or rewind that prefix first, publish exactly it, and do not poison. Poisoning is reserved for uncertainty, not ordinary hook, cancellation, draft, or parser exits.

## Shutdown Design

Poisoning does not alter `Engine.close()`, `Session.close()`, `EngineWorker.close()`, or the worker's `ExitStack` ownership. Shutdown closes retained snapshots, the owned session, and the engine once on the worker thread, then terminates and joins that thread. Repeated owner close calls remain idempotent and do not execute native cleanup twice.

## Test Design

Add focused fatal-state tests using a multi-token step and instrumented `FakeNative` doubles:

- Fail during a staged native operation, invoke `poison()`, and assert the exact initial exception, no staged token/event publication, and no native call from the poisoned job's remaining Python-only `finally` path.
- Exercise original-plus-cleanup failure and assert exception identity and `original is cleanup.__context__`; also inspect later rejection causes. Separate rejected operations receive distinct `WorkerPoisonedError` objects with the same recorded cause.
- Parameterize submitted rounds, sync, reset, checkpoint create/drop, payload save/load, direct calls that would eval/rewind/snapshot, public engine/session methods, `on_worker`, and extension callbacks. Include calls and jobs admitted before poisoning but dequeued afterward. Assert the native call log and callback counters do not advance.
- For every route, assert `WorkerPoisonedError`, prompt future/event completion, and the stored failure chain. Exercise both `Future.result()` and `Future.exception()`, plus event iteration and `events.wait()`, to prove that no waiter hangs.
- Attempt `new()`, reset, switch, and fork and prove that later native work is still rejected without fallback to `PlainStep`. For failed `new()`, assert the unchanged ID, closed scoped state, absent `session_start`, and persistent poisoning.
- Create a snapshot before poisoning, close the public engine and worker repeatedly, and assert one snapshot close, one session close, one engine close, all on the owner thread, plus a terminated worker thread.
- Run focused worker, decode, lifecycle, interruption, snapshot/checkpoint, payload, multi-token, and tentative-decode suites before the complete portable suite.

Fake-native coverage proves routing, completion, publication, and ownership contracts only; downstream real-engine acceptance remains required.
