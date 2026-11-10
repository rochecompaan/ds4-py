"""Gated native work must not block the loop or outlive its operation owner."""

import asyncio
import threading
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import replace as replace_dataclass
from pathlib import Path
from typing import Literal

import anyio
import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from test_compaction_capability import script
from test_compaction_stream import PREFIX
from test_context_commands import make_runtime, seed

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative, FakeSession
from ds4py._native.protocol import CancelFn, ProgressFn, SyncStatus
from ds4py.capabilities.compaction import CompactionReason, LLMCompaction
from ds4py.capabilities.context import ContextRequest
from ds4py.engine.events import Synced
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.reserve import ReserveAuthority
from ds4py.engine.sampling import SamplingPolicy
from ds4py.engine.transcript import TranscriptReplacement
from ds4py.errors import EngineError
from ds4py.model import DS4Model
from ds4py.tui_render import FooterModel


class Gate:
    """A bounded thread gate; timeout is a deadlock guard, never synchronization."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.timed_out = False
        self.used = False

    def hold(self) -> None:
        if self.used:
            return
        self.used = True
        self.loop.call_soon_threadsafe(self.entered.set)
        self.timed_out = not self.release.wait(3)


async def record_heartbeat(gate: Gate, seen: asyncio.Event) -> None:
    await gate.entered.wait()
    await asyncio.sleep(0)
    seen.set()


async def cancel_gated(
    task: asyncio.Task[None],
    scope: anyio.CancelScope,
    mode: str,
    close_operation: Callable[[], Coroutine[object, object, None]],
) -> asyncio.Task[None] | None:
    close = None
    if mode in {"anyio", "deadline"}:
        scope.deadline = anyio.current_time()
    elif mode == "close":
        close = asyncio.create_task(close_operation())
    elif mode == "repeat":
        task.cancel()
    await asyncio.sleep(0)
    if mode == "repeat":
        task.cancel()
    await asyncio.sleep(0)
    assert not task.done(), "operation owner returned before native drain"
    if close is not None:
        assert not close.done()
    return close


async def settle(task: asyncio.Task[None], close: asyncio.Task[None] | None) -> None:
    await asyncio.gather(task, return_exceptions=True)
    if close is not None:
        await asyncio.gather(close, return_exceptions=True)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.mark.parametrize("reason", ["manual", "threshold", "tool_result", "reserve"])
@pytest.mark.parametrize("phase", ["prefill", "argmax", "replacement"])
def test_native_gates_leave_loop_and_footer_responsive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: CompactionReason, phase: str
) -> None:
    # Inline native work prevents the queued footer consumer from painting.
    async def scenario() -> None:
        with ds4py.open("f.gguf", ctx=16384, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
            runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
            seed(runtime)
            session = runtime.model.session
            gate = Gate()
            threads: list[int] = []
            painted = asyncio.Event()
            heartbeat_seen = asyncio.Event()
            queue: asyncio.Queue[SessionEvent] = asyncio.Queue()
            footer = FooterModel(runtime)

            def listener(event: SessionEvent) -> None:
                threads.append(threading.get_ident())
                # A guarded observation from a listener must not wait for the
                # offloaded producer's mutation lock.
                if event.kind in {"Synced", "Prefill"}:
                    session.open_assistant_start()
                queue.put_nowait(event)

            session.on_event(listener)

            async def consumer() -> None:
                while True:
                    event = await queue.get()
                    if event.kind == "before_compact":
                        assert "compacting" in str(footer.render(160)).lower()
                        painted.set()

            pulse = asyncio.create_task(record_heartbeat(gate, heartbeat_seen))
            consume = asyncio.create_task(consumer())
            sync = FakeSession.sync
            argmax = FakeSession.argmax
            summary_seen = False

            def gated_sync(
                native: FakeSession,
                tokens: Sequence[int],
                *,
                progress: ProgressFn | None = None,
                cancel: CancelFn | None = None,
            ) -> SyncStatus:
                nonlocal summary_seen
                # Fake tokenizer represents ASCII as byte + 1.
                private = bytes(t - 1 for t in tokens if 1 <= t <= 256)
                summary = b"Internal ds4-agent context compaction request." in private
                if summary:
                    summary_seen = True
                replacement = summary_seen and not summary and b"durable state" in private
                if (phase == "prefill" and summary) or (phase == "replacement" and replacement):
                    gate.hold()
                return sync(native, tokens, progress=progress, cancel=cancel)

            def gated_argmax(native: FakeSession) -> int:
                if phase == "argmax":
                    gate.hold()
                return argmax(native)

            monkeypatch.setattr(FakeSession, "sync", gated_sync)
            monkeypatch.setattr(FakeSession, "argmax", gated_argmax)

            async def invoke() -> None:
                owner = runtime.components.compactor
                if reason == "manual":
                    await runtime.compact()
                elif reason == "threshold":
                    owner.soft = 0
                    ctx = RunContext(deps=None, model=runtime.model, usage=RunUsage(), messages=runtime._messages)
                    request = ModelRequestContext(
                        model=runtime.model,
                        messages=runtime._messages,
                        model_settings=None,
                        model_request_parameters=ModelRequestParameters(),
                    )
                    await owner.before_model_request(ctx, request)
                    assert ctx.usage.output_tokens == 13
                else:
                    result = await owner.make_room(runtime._messages, ContextRequest(1, reason))
                    assert result.generated_tokens == 13

            task = asyncio.create_task(invoke())
            try:
                await asyncio.wait_for(gate.entered.wait(), 5)
                await asyncio.wait_for(painted.wait(), 5)
                await asyncio.wait_for(heartbeat_seen.wait(), 5)
                assert not gate.timed_out, "native work blocked the owning loop"
                assert not task.done()
                assert set(threads) == {threading.get_ident()}
                with pytest.raises(EngineError, match="active round"):
                    session.add_logits_hook(lambda _view: None)
                gate.release.set()
                await asyncio.wait_for(task, 5)
                part = runtime.messages[0].parts[-1]
                assert isinstance(part, UserPromptPart) and part.content == "durable state"
                assert runtime.model.binder.prepare(runtime.messages).kind == "unchanged"
                assert footer.status == "idle"
            finally:
                gate.release.set()
                await asyncio.gather(task, return_exceptions=True)
                consume.cancel()
                pulse.cancel()
                await asyncio.gather(consume, pulse, return_exceptions=True)
                footer.close()
                await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["summary", "replacement_before", "replacement_after"])
@pytest.mark.parametrize("cancellation", ["repeat", "anyio", "close", "signal"])
def test_manual_cancellation_drains_and_publishes_only_committed_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: Literal["summary", "replacement_before", "replacement_after"],
    cancellation: str,
) -> None:
    # Abandoning executor work allows late mutation; skipping publication after
    # a successful native commit leaves canonical history paired with wrong KV.
    async def scenario() -> None:
        with ds4py.open("f.gguf", ctx=16384, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
            runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
            seed(runtime)
            session = runtime.model.session
            session._worker.call(lambda e, n: n.sync(session.transcript.tokens)).result()
            before = runtime.messages, session.transcript.tokens, runtime.model.binder.records()
            gate = Gate()
            events: list[SessionEvent] = []
            runtime.on_event(events.append)
            original_summary = FakeSession.argmax
            original_replace = session.replace_transcript

            def summary(native: FakeSession) -> int:
                if phase == "summary":
                    gate.hold()
                return original_summary(native)

            def replace(value: TranscriptReplacement, *, reason: str) -> object:
                if phase == "replacement_before":
                    gate.hold()
                result = original_replace(value, reason=reason)
                if phase == "replacement_after":
                    gate.hold()
                return result

            monkeypatch.setattr(FakeSession, "argmax", summary)
            monkeypatch.setattr(session, "replace_transcript", replace)
            scope = anyio.CancelScope()

            async def compact() -> None:
                with scope:
                    await runtime.compact()

            task = asyncio.create_task(compact())
            close: asyncio.Task[None] | None = None
            try:
                await asyncio.wait_for(gate.entered.wait(), 5)
                assert not gate.timed_out, "native work blocked cancellation"
                for operation in (
                    runtime.compact(),
                    runtime.run("overlap"),
                    runtime.save(),
                    runtime.new(),
                    runtime.switch("absent"),
                    runtime.fork(),
                    runtime.components.compactor.compact_messages(runtime._messages, reason="manual"),
                    LLMCompaction(runtime.model.binder).compact_messages(runtime._messages, reason="threshold"),
                ):
                    with pytest.raises(EngineError, match="active"):
                        await operation
                assert sum(e.kind == "before_compact" for e in events) == 1
                if cancellation == "signal":
                    runtime.cancel()
                close = await cancel_gated(task, scope, cancellation, runtime.close)
                gate.release.set()
                result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
                if cancellation in {"repeat", "close"}:
                    assert isinstance(result[0], asyncio.CancelledError)
                if close is not None:
                    await asyncio.wait_for(close, 5)
                committed = phase == "replacement_after"
                if committed:
                    assert runtime.messages != before[0]
                    assert runtime.model.binder.records() != before[2]
                    assert session.transcript.tokens != before[1]
                else:
                    assert (runtime.messages, session.transcript.tokens, runtime.model.binder.records()) == before
                assert sum(e.kind == "compaction_finished" for e in events) == 1
                snapshot = runtime.messages, session.transcript.tokens, runtime.model.binder.records()
                await asyncio.sleep(0)
                assert (runtime.messages, session.transcript.tokens, runtime.model.binder.records()) == snapshot
                assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == session.transcript.tokens
                assert not session._worker.cancel_event.is_set()
                if cancellation != "close":
                    # Restore wrappers so the next successful operation proves
                    # neither native cancellation nor ownership remains sticky.
                    monkeypatch.setattr(session, "replace_transcript", original_replace)
                    await runtime.run("continue")
                    saved = await runtime.save()
                    expected = runtime.messages, session.transcript.tokens
                    await runtime.new()
                    await runtime.switch(saved.session_id)
                    assert (runtime.messages, session.transcript.tokens) == expected
            finally:
                gate.release.set()
                await settle(task, close)
                await runtime.close()

    asyncio.run(scenario())


def reserve_script() -> Callable[[Sequence[int], Sequence[int]], int]:
    sampled = 0

    def tokens(prompt: Sequence[int], generated: Sequence[int]) -> int:
        nonlocal sampled
        private = bytes(t - 1 for t in prompt if 1 <= t <= 256)
        if b"Internal ds4-agent context compaction request." in private:
            text = b"durable state"
            return text[len(generated)] + 1 if len(generated) < len(text) else FAKE_EOS
        sampled += 1
        return 98 if sampled <= 12 else FAKE_EOS

    return tokens


@pytest.mark.parametrize("phase", ["summary", "replacement_before", "replacement_after"])
@pytest.mark.parametrize("cancellation", ["close", "repeat", "deadline"])
def test_reserve_cancellation_drains_before_response_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    cancellation: str,
) -> None:
    # Closing the suspended round before its retained owner drains causes late
    # native writes, expired reserve permission, or duplicate response binding.
    async def scenario() -> None:
        native = FakeNative(
            script=reserve_script(), info=replace_dataclass(FakeNative()._info, family="qwen", is_qwen4=True)
        )
        with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=native) as engine:
            session = engine.session()
            model = DS4Model(session, sampling=engine.sampling_defaults)
            model.context_owner = LLMCompaction(model.binder)
            messages: list[ModelMessage] = [
                ModelRequest([UserPromptPart("x" * 4000)]),
                ModelResponse([TextPart("old answer")]),
                ModelRequest([UserPromptPart("r" * 1500)]),
            ]
            gate = Gate()
            argmax = FakeSession.argmax
            replacement = session.replace_transcript
            notifications: list[SessionEvent] = []
            listener_threads: list[int] = []
            rejected: list[str] = []

            def listener(event: SessionEvent) -> None:
                listener_threads.append(threading.get_ident())
                notifications.append(event)
                if isinstance(event.payload.get("event"), Synced):
                    # Callback delivery must never inherit the selected owner's
                    # executor authority, nor deadlock trying to reject it.
                    try:
                        session.prepare_summary(0)
                    except EngineError:
                        rejected.append(event.kind)

            def gated_argmax(handle: FakeSession) -> int:
                if phase == "summary":
                    gate.hold()
                return argmax(handle)

            def gated_replace(
                value: TranscriptReplacement, *, reason: str, authority: ReserveAuthority | None = None
            ) -> object:
                if phase == "replacement_before":
                    gate.hold()
                result = replacement(value, reason=reason, authority=authority)
                if phase == "replacement_after":
                    gate.hold()
                return result

            monkeypatch.setattr(FakeSession, "argmax", gated_argmax)
            monkeypatch.setattr(session, "replace_transcript", gated_replace)
            scope = anyio.CancelScope()
            async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
                session.on_event(listener)
                metadata = stream.get().metadata
                assert metadata is not None
                key = metadata["ds4.key"]
                session._worker.call(
                    lambda e, n: setattr(n, "_ctx", len(session.transcript) + len(PREFIX) + 8)
                ).result()

                async def consume() -> None:
                    with scope:
                        async for _ in stream:
                            pass

                consumer = asyncio.create_task(consume())
                close: asyncio.Task[None] | None = None
                try:
                    await asyncio.wait_for(gate.entered.wait(), 5)
                    assert not gate.timed_out
                    close = await cancel_gated(consumer, scope, cancellation, stream.close_stream)
                    gate.release.set()
                    result = await asyncio.wait_for(asyncio.gather(consumer, return_exceptions=True), 5)
                    if cancellation != "deadline":
                        assert isinstance(result[0], asyncio.CancelledError)
                    if close is not None:
                        await asyncio.wait_for(close, 5)
                finally:
                    gate.release.set()
                    await settle(consumer, close)
                response = stream.get()
                assert response.parts == [TextPart("a" * 7)]
                assert response.metadata is not None and response.metadata["ds4.key"] == key
                assert response.usage.output_tokens == (7 if phase == "summary" else 20)
                assert stream._bound and stream._reconciled
                assert stream._cleanup_task is not None and stream._cleanup_task.done()
                assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == session.transcript.tokens
                assert model.binder.records()[-1].end == len(session.transcript)
                assert model.binder.records()[-1].key == key
                part = messages[0].parts[0]
                assert isinstance(part, UserPromptPart)
                assert (part.content == "durable state") == (phase == "replacement_after")
                terminal = [e.payload["outcome"] for e in notifications if e.kind == "compaction_finished"]
                assert terminal == ["cancelled"]
                assert rejected
                assert set(listener_threads) == {threading.get_ident()}
                assert not session._worker.cancel_event.is_set()
            session._worker.call(lambda e, n: setattr(n, "_ctx", 8192)).result()
            async with model.request_stream(messages, None, ModelRequestParameters()) as next_stream:
                async for _ in next_stream:
                    pass

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["summary", "replacement"])
def test_native_error_restores_unchanged_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    # A native sync may mutate KV and then fail without producing Synced.
    async def scenario() -> None:
        with ds4py.open("f.gguf", ctx=16384, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
            runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
            seed(runtime)
            session = runtime.model.session
            session._worker.call(lambda e, n: n.sync(session.transcript.tokens)).result()
            before = runtime.messages, session.transcript.tokens, runtime.model.binder.records()
            sync = FakeSession.sync
            events: list[SessionEvent] = []
            runtime.on_event(events.append)

            def fail(
                native: FakeSession,
                tokens: Sequence[int],
                *,
                progress: ProgressFn | None = None,
                cancel: CancelFn | None = None,
            ) -> SyncStatus:
                status = sync(native, tokens, progress=progress, cancel=cancel)
                text = bytes(t - 1 for t in tokens if 1 <= t <= 256)
                summary = b"Internal ds4-agent" in text
                if (phase == "summary" and summary) or (
                    phase == "replacement" and b"durable state" in text and not summary
                ):
                    raise EngineError(f"{phase} native failure")
                return status

            monkeypatch.setattr(FakeSession, "sync", fail)
            try:
                with pytest.raises(EngineError, match=f"{phase} native failure"):
                    await runtime.compact()
                assert (runtime.messages, session.transcript.tokens, runtime.model.binder.records()) == before
                assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == before[1]
                assert [e.payload["outcome"] for e in events if e.kind == "compaction_finished"] == ["failed"]
                assert not any(e.kind == "compacted" for e in events)
                monkeypatch.setattr(FakeSession, "sync", sync)
                await runtime.run("continue")
                saved = await runtime.save()
                expected = runtime.messages, session.transcript.tokens
                await runtime.new()
                await runtime.switch(saved.session_id)
                assert (runtime.messages, session.transcript.tokens) == expected
            finally:
                await runtime.close()

    asyncio.run(scenario())


def test_repeated_close_cancellation_cannot_abandon_reconciliation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Returning from close before the retained cleanup finishes permits late
    # transcript/provenance mutation after the caller starts another operation.
    async def scenario() -> None:
        with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
            session = engine.session()
            model = DS4Model(session, sampling=engine.sampling_defaults)
            gate = Gate()
            reconcile = session.reconcile_round

            def gated_reconcile(count: int) -> None:
                gate.hold()
                reconcile(count)

            monkeypatch.setattr(session, "reconcile_round", gated_reconcile)
            async with model.request_stream(
                [ModelRequest([UserPromptPart("go")])], None, ModelRequestParameters()
            ) as stream:
                iterator = stream.__aiter__()
                await anext(iterator)
                close = asyncio.create_task(stream.close_stream())
                try:
                    await asyncio.wait_for(gate.entered.wait(), 5)
                    close.cancel()
                    await asyncio.sleep(0)
                    close.cancel()
                    await asyncio.sleep(0)
                    assert not close.done(), "close abandoned retained reconciliation"
                    gate.release.set()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(close, 5)
                    assert stream._cleanup_task is not None and stream._cleanup_task.done()
                    assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == session.transcript.tokens
                    assert stream._bound and stream._reconciled
                finally:
                    gate.release.set()
                    await asyncio.gather(close, return_exceptions=True)
                    await stream._finish()

    asyncio.run(scenario())


def test_cancel_before_native_submission_does_not_poison_next_run(tmp_path: Path) -> None:
    async def scenario() -> None:
        with ds4py.open("f.gguf", ctx=16384, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
            runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
            seed(runtime)
            unsubscribe = runtime.on_event(lambda e: runtime.cancel(), kinds={"before_compact"})
            try:
                task = asyncio.create_task(runtime.compact())
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert not runtime.model.session._worker.cancel_event.is_set()
                unsubscribe()
                assert (await runtime.run("continue")).output == "durable state"
            finally:
                unsubscribe()
                await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(("phase", "expected_usage", "expected_summary"), [("summary", 7, 0), ("replacement", 20, 13)])
def test_reserve_failure_accounts_only_emitted_summary_output(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    expected_usage: int,
    expected_summary: int,
) -> None:
    # RoomResult may never exist; emitted private output still belongs to this
    # response, while a failure before summary emission must not invent usage.
    async def scenario() -> None:
        tokens = reserve_script()

        def fail_summary(prompt: Sequence[int], generated: Sequence[int]) -> int:
            text = bytes(t - 1 for t in prompt if 1 <= t <= 256)
            if phase == "summary" and b"Internal ds4-agent context compaction request." in text:
                raise EngineError("summary native failure")
            return tokens(prompt, generated)

        native = FakeNative(
            script=fail_summary, info=replace_dataclass(FakeNative()._info, family="qwen", is_qwen4=True)
        )
        with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=native) as engine:
            session = engine.session()
            model = DS4Model(session, sampling=engine.sampling_defaults)
            model.context_owner = LLMCompaction(model.binder)
            messages: list[ModelMessage] = [
                ModelRequest([UserPromptPart("x" * 4000)]),
                ModelResponse([TextPart("old answer")]),
                ModelRequest([UserPromptPart("r" * 1500)]),
            ]
            events: list[SessionEvent] = []
            session.on_event(events.append)

            def fail_replace(
                value: TranscriptReplacement, *, reason: str, authority: ReserveAuthority | None = None
            ) -> object:
                raise EngineError("replacement native failure")

            if phase == "replacement":
                monkeypatch.setattr(session, "replace_transcript", fail_replace)
            async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
                session._worker.call(
                    lambda e, n: setattr(n, "_ctx", len(session.transcript) + len(PREFIX) + 8)
                ).result()
                with pytest.raises(EngineError, match=f"{phase} native failure"):
                    async for _ in stream:
                        pass
            response = stream.get()
            assert response.parts == [TextPart("a" * 7)]
            assert response.usage.output_tokens == expected_usage
            assert sum(e.kind == "SummaryToken" for e in events) == expected_summary
            assert [e.payload["outcome"] for e in events if e.kind == "compaction_finished"] == ["failed"]
            assert not any(e.kind == "compacted" for e in events)
            assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == session.transcript.tokens
            assert model.binder.records()[-1].end == len(session.transcript)

    asyncio.run(scenario())


def test_runtime_cancel_signals_blocked_native_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Task cancellation alone cannot interrupt native work; the Session signal
    # must reach the worker while the manual operation is still owned.
    async def scenario() -> None:
        with ds4py.open("f.gguf", ctx=16384, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
            runtime = make_runtime(DS4Model(engine.session(), sampling=SamplingPolicy(seed=1)), tmp_path)
            seed(runtime)
            session = runtime.model.session
            entered = asyncio.Event()
            loop = asyncio.get_running_loop()
            argmax = FakeSession.argmax

            def until_cancel(native: FakeSession) -> int:
                loop.call_soon_threadsafe(entered.set)
                assert session._worker.cancel_event.wait(3), "runtime.cancel did not signal native work"
                return argmax(native)

            monkeypatch.setattr(FakeSession, "argmax", until_cancel)
            task = asyncio.create_task(runtime.compact())
            try:
                await asyncio.wait_for(entered.wait(), 5)
                runtime.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
                assert not session._worker.cancel_event.is_set()
                monkeypatch.setattr(FakeSession, "argmax", argmax)
                assert (await runtime.compact()).summary == "durable state"
            finally:
                runtime.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await runtime.close()

    asyncio.run(scenario())
