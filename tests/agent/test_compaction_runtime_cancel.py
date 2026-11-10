"""Cancellation inside real-runtime reserve work must survive adoption and saves."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness import HarnessDeprecationWarning
from test_compaction_async import Gate
from test_compaction_persistence import dump
from test_runtime import fake_engine, fixed_clock, make_model

from ds4py._native.fake import FAKE_EOS, FakeSession
from ds4py.engine.events import Synced
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.transcript import TranscriptReplacement
from ds4py.runtime import AgentRuntime
from ds4py.store import SessionStore


@pytest.mark.parametrize("phase", ["summary", "replacement_before", "replacement_after"])
def test_runtime_reserve_cancellation_adopts_and_roundtrips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    # A stream-only follow-up request misses runtime adoption of the committed
    # canonical list and its failed-run snapshot, then persistence/restoration.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    stage = "first"
    sampled = 0

    def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
        nonlocal sampled
        text = bytes(t - 1 for t in prompt if 1 <= t <= 256)
        if b"Internal ds4-agent context compaction request." in text:
            summary = b"durable state"
            return summary[len(generated)] + 1 if len(generated) < len(summary) else FAKE_EOS
        output = {"first": b"first", "reserve": b"before after", "continue": b"continued"}[stage]
        token = output[sampled] + 1 if sampled < len(output) else FAKE_EOS
        sampled += 1
        return token

    async def scenario() -> None:
        nonlocal stage, sampled
        with fake_engine(script=script) as engine:
            model = make_model(engine)
            with pytest.warns(HarnessDeprecationWarning, match="deprecated and ignored: commands/paths start"):
                runtime = AgentRuntime(model, tmp_path, store=SessionStore(tmp_path / "store"), clock=fixed_clock)
            session = model.session
            armed = False
            events: list[SessionEvent] = []

            def force_reserve(notification: SessionEvent) -> None:
                nonlocal armed
                events.append(notification)
                event = notification.payload.get("event")
                if isinstance(event, Synced) and stage == "reserve" and not armed:
                    armed = True
                    capacity = event.pos + 8
                    session._worker.call(lambda e, native: setattr(native, "_ctx", capacity)).result()

            runtime.on_event(force_reserve)
            gate = Gate()
            argmax = FakeSession.argmax
            replace = session.replace_transcript

            def summary(native: FakeSession) -> int:
                if phase == "summary" and stage == "reserve":
                    gate.hold()
                return argmax(native)

            def replacement(value: TranscriptReplacement, *, reason: str) -> object:
                if phase == "replacement_before":
                    gate.hold()
                result = replace(value, reason=reason)
                if phase == "replacement_after":
                    gate.hold()
                return result

            monkeypatch.setattr(FakeSession, "argmax", summary)
            monkeypatch.setattr(session, "replace_transcript", replacement)
            tasks: list[asyncio.Task[object]] = []
            try:
                await runtime.run("x" * 14000)
                stage, sampled = "reserve", 0
                events.clear()
                task = asyncio.create_task(runtime.run("r" * 1500))
                tasks.append(task)
                await asyncio.wait_for(gate.entered.wait(), 5)
                assert not gate.timed_out
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                gate.release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
                assert runtime.status == "interrupted"
                assert [e.payload["outcome"] for e in events if e.kind == "compaction_finished"] == ["cancelled"]
                assert not any(e.kind == "compacted" for e in events)
                assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == session.transcript.tokens
                records = model.binder.records()
                assert [(m.metadata or {}).get("ds4.key") for m in runtime.messages] == [r.key for r in records]
                response = runtime.messages[-1]
                assert isinstance(response, ModelResponse)
                assert response.parts == [TextPart("before ")]
                assert response.usage.output_tokens == (7 if phase == "summary" else 20)
                syncs = [event for e in events if isinstance(event := e.payload.get("event"), Synced)]
                assert response.usage.input_tokens == sum(e.prefilled for e in syncs)
                assert response.usage.cache_read_tokens == sum(e.plan.reused for e in syncs)
                users = [p.content for m in runtime.messages for p in m.parts if isinstance(p, UserPromptPart)]
                assert users == (
                    ["durable state", "r" * 1500] if phase == "replacement_after" else ["x" * 14000, "r" * 1500]
                )
                run_id = next(e.payload["run_id"] for e in events if e.kind == "run_start")
                assert isinstance(run_id, str)
                snapshot = await runtime._holder().current.latest_snapshot(run_id=run_id, include_interrupted=True)
                assert snapshot is not None and dump(snapshot.messages) == dump(runtime.messages)
                stable = dump(runtime.messages), session.transcript.tokens, records
                await asyncio.sleep(0)
                assert (dump(runtime.messages), session.transcript.tokens, model.binder.records()) == stable
                assert not session._worker.cancel_event.is_set()
                session._worker.call(lambda e, native: setattr(native, "_ctx", 65536)).result()
                monkeypatch.setattr(session, "replace_transcript", replace)
                # Save the interrupted state itself before continuation, then
                # save/switch again after a successful subsequent runtime run.
                saved = await runtime.save()
                await runtime.new()
                await runtime.switch(saved.session_id)
                assert (dump(runtime.messages), session.transcript.tokens, model.binder.records()) == stable
                stage, sampled = "continue", 0
                assert (await runtime.run("continue")).output == "continued"
                saved = await runtime.save()
                final = dump(runtime.messages), session.transcript.tokens, model.binder.records()
                provenance = model.binder.export_segments()
                await runtime.new()
                await runtime.switch(saved.session_id)
                assert (dump(runtime.messages), session.transcript.tokens, model.binder.records()) == final
                assert model.binder.export_segments() == provenance
                assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == session.transcript.tokens
                assert session.plan().to_prefill == 0
            finally:
                gate.release.set()
                await asyncio.gather(*tasks, return_exceptions=True)
                await runtime.close()

    asyncio.run(scenario())
