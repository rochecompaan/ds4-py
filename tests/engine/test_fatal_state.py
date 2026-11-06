"""Fatal decode failures permanently disable ordinary native work."""

from pathlib import Path
from typing import Any, cast, override

import pytest

from ds4py._native.fake import FakeEngine, FakeNative, FakeSession
from ds4py._native.protocol import EngineConfig, SessionHandle
from ds4py.engine.decode import DecodeContext, GenerateJob
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.transcript import Transcript
from ds4py.engine.worker import EngineWorker, SyncJob
from ds4py.errors import EngineError


class FaultNative(FakeNative):
    """Fail one native eval and optionally fail the attempted rewind cleanup."""

    def __init__(
        self,
        original: EngineError,
        *,
        cleanup: EngineError | None = None,
        partial: bool = False,
    ) -> None:
        super().__init__()
        self.original = original
        self.cleanup = cleanup
        self.partial = partial
        self.fault_calls: list[tuple[str, int]] = []
        self.session: FakeSession | None = None

    @override
    def open_engine(self, config: EngineConfig) -> FakeEngine:
        engine = super().open_engine(config)
        create_session = engine.create_session

        def create(ctx: int) -> FakeSession:
            session = create_session(ctx)
            self.session = session
            evaluate = session.eval
            rewind = session.rewind

            def eval_with_fault(token: int) -> None:
                self.fault_calls.append(("eval", token))
                if token == 99:
                    if self.partial:
                        evaluate(token)
                    raise self.original
                evaluate(token)

            def rewind_with_fault(position: int) -> None:
                self.fault_calls.append(("rewind", position))
                if self.cleanup is not None:
                    raise self.cleanup
                rewind(position)

            dynamic = cast("Any", session)
            dynamic.eval = eval_with_fault
            dynamic.rewind = rewind_with_fault
            return session

        cast("Any", engine).create_session = create
        return engine


class UncertainMultiTokenStep:
    hooks_every_sample = True

    def __init__(self, *, attempt_cleanup: bool = False) -> None:
        self.attempt_cleanup = attempt_cleanup
        self.run_calls = 0
        self.finally_calls = 0

    def run(self, session: SessionHandle, ctx: DecodeContext) -> None:
        self.run_calls += 1
        block_start = session.pos()
        ctx.begin_staging()
        try:
            for token in (98, 99):
                try:
                    verdict = ctx.accept(token)
                    session.eval(token)
                    ctx.stage(token, verdict)
                except EngineError as original:
                    if self.attempt_cleanup:
                        try:
                            session.rewind(block_start)
                        except EngineError as cleanup:
                            ctx.poison(cleanup)
                    ctx.poison(original)
        finally:
            # Poisoning is terminal: only Python-local bookkeeping is allowed here.
            self.finally_calls += 1


def run_uncertain_block(
    native: FaultNative,
    step: UncertainMultiTokenStep,
) -> tuple[Transcript, EngineWorker, EngineError]:
    worker = EngineWorker(native, EngineConfig(Path("f.gguf"), ctx=64))
    transcript = Transcript()
    transcript.append([5, 6], "user")
    list(worker.submit(SyncJob((5, 6))))
    job = GenerateJob(SamplingPolicy(), Rng(1), 10, transcript, decode_step=step)
    events = worker.submit(job)
    with pytest.raises(EngineError) as caught:
        next(events)
    return transcript, worker, caught.value


@pytest.mark.parametrize("partial", [False, True], ids=["before-side-effect", "after-side-effect"])
def test_multi_token_native_failure_poisoning_discards_staged_tokens(partial: bool) -> None:
    original = EngineError("native eval failed")
    native = FaultNative(original, partial=partial)
    step = UncertainMultiTokenStep()
    transcript, worker, error = run_uncertain_block(native, step)
    try:
        assert error is original
        assert step.run_calls == 1
        assert step.finally_calls == 1
        assert transcript.tokens == (5, 6)
        assert native.fault_calls == [("eval", 98), ("eval", 99)]
    finally:
        worker.close()


def test_poison_preserves_original_and_cleanup_exception_chaining() -> None:
    original = EngineError("native eval failed")
    cleanup = EngineError("native rewind cleanup failed")
    native = FaultNative(original, cleanup=cleanup)
    step = UncertainMultiTokenStep(attempt_cleanup=True)
    transcript, worker, error = run_uncertain_block(native, step)
    try:
        assert error is cleanup
        assert cleanup.__context__ is original
        assert step.run_calls == 1
        assert step.finally_calls == 1
        assert transcript.tokens == (5, 6)
        assert native.fault_calls == [("eval", 98), ("eval", 99), ("rewind", 2)]
    finally:
        worker.close()
