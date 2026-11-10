"""Compacted live histories survive version-1 save, restore, and fork."""

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, UserPromptPart
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.step_persistence import FileStepStore
from test_runtime import fake_engine, fixed_clock, make_model, scripted_rounds

from ds4py.capabilities.compaction import LLMCompaction
from ds4py.engine.events import Synced
from ds4py.engine.lifecycle import SessionEvent
from ds4py.runtime import AgentRuntime
from ds4py.store import KV_HEADER, KV_MAGIC, KV_VERSION, SessionStore, token_digest


def dump(messages: list[ModelMessage]) -> bytes:
    return ModelMessagesTypeAdapter.dump_json(messages)


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AgentRuntime]:
    # Personal skill catalogs must not change the prompt or add YAML warnings.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    project = tmp_path / "project"
    project.mkdir()
    with fake_engine(script=scripted_rounds("first answer", "recent answer", "durable state", "continued")) as engine:
        # Existing harness cwd deprecations are expected at runtime construction;
        # capture only that warning, not new warnings from persistence operations.
        with pytest.warns(HarnessDeprecationWarning, match="deprecated and ignored: commands/paths start"):
            instance = AgentRuntime(make_model(engine), project, store=SessionStore(home), clock=fixed_clock)
        try:
            yield instance
        finally:
            asyncio.run(instance.close())


async def compact_history(runtime: AgentRuntime) -> tuple[str, bytes]:
    await runtime.run("x" * 14000)
    last = await runtime.run("recent")
    original = dump(runtime.messages)
    compactor = runtime.components.compactor
    assert isinstance(compactor, LLMCompaction)
    result = await compactor.compact_messages(runtime._messages, reason="manual")
    assert (result.summary, result.dropped, result.kept, result.generated_tokens) == ("durable state", 2, 2, 13)
    assert len(runtime.messages) == len(runtime.model.binder.records()) == 3
    summary = runtime.messages[0]
    assert isinstance(summary, ModelRequest)
    assert any(isinstance(part, UserPromptPart) and part.content == "durable state" for part in summary.parts)
    return last.run_id, original


def collect_syncs(runtime: AgentRuntime) -> list[Synced]:
    syncs: list[Synced] = []

    def collect(notification: SessionEvent) -> None:
        event = notification.payload.get("event")
        if isinstance(event, Synced):
            syncs.append(event)

    runtime.on_event(collect, kinds={"Synced"})
    return syncs


def test_compact_save_switch_restores_exact_kv_and_preserves_old_run(runtime: AgentRuntime) -> None:
    async def scenario() -> None:
        old_run, original = await compact_history(runtime)
        binder = runtime.model.binder
        session = runtime.model.session
        messages = dump(runtime.messages)
        tokens = session.transcript.tokens
        records = binder.records()
        segments = binder.export_segments()
        saved = await runtime.save()
        directory = runtime.store.session_dir(saved.session_id)
        data = json.loads((directory / "save.json").read_text())
        assert saved.version == data["version"] == 1
        assert set(data) == {
            "version",
            "session_id",
            "conversation_id",
            "run_id",
            "step_index",
            "model_id",
            "quant_bits",
            "ctx",
            "title",
            "created_at",
            "last_used",
            "payload_bytes",
            "tokens",
            "tokens_sha256",
            "segments",
        }
        assert KV_HEADER.unpack((directory / "session.kv").read_bytes()[: KV_HEADER.size]) == (
            KV_MAGIC,
            KV_VERSION,
            saved.payload_bytes,
        )
        assert saved.tokens == len(tokens)
        assert saved.tokens_sha256 == token_digest(tokens)
        assert len(saved.segments) == len(segments) == 7
        assert all(not segment.kind.startswith("system_note") for segment in saved.segments)
        steps = FileStepStore(directory / "steps")
        snapshot = await steps.latest_snapshot(run_id=saved.run_id)
        assert snapshot is not None and dump(snapshot.messages) == messages

        await runtime.new()
        syncs = collect_syncs(runtime)
        assert await runtime.switch(saved.session_id) == saved
        assert syncs == []  # Loading and installing provenance never prefills.
        assert dump(runtime.messages) == messages
        assert session.transcript.tokens == tokens
        assert binder.records() == records
        assert binder.export_segments() == segments
        assert session.plan().reused == len(tokens)
        assert session.plan().to_prefill == 0
        # Payload loading retains live KV, not the previous named checkpoints.
        assert session.history.checkpoint_tokens == {}

        continuation = await runtime.run("continue")
        assert continuation.output == "continued"
        assert continuation.usage.input_tokens == 33  # 8 text + 8 user wrapper + 17 assistant opener.
        assert continuation.usage.output_tokens == 9
        assert len(syncs) == 1
        assert (syncs[0].plan.kind, syncs[0].plan.reused, syncs[0].prefilled) == ("extend", len(tokens), 33)
        assert len(session.transcript) == len(tokens) + 43  # Also includes 9 answer tokens and EOS.
        assert session.transcript.tokens[: len(tokens)] == tokens
        assert binder.records()[:3] == records
        assert len(runtime.messages) == len(binder.records()) == 5
        assert len(binder.export_segments()) == 11
        continued_messages = dump(runtime.messages)
        continued_records = binder.records()
        resaved = await runtime.save()
        await runtime.new()
        await runtime.switch(resaved.session_id)
        assert dump(runtime.messages) == continued_messages
        assert binder.records() == continued_records
        old_snapshot = await steps.latest_snapshot(run_id=old_run)
        assert old_snapshot is not None and dump(old_snapshot.messages) == original
        old_keys = {(message.metadata or {}).get("ds4.key") for message in old_snapshot.messages}
        live_keys = {record.key for record in records}
        assert len(old_snapshot.messages) == 4
        assert len(old_keys - live_keys) == 2  # Old runs need not have ranges in the compacted KV.

    asyncio.run(scenario())


@pytest.mark.parametrize("payload", ["missing", "zero"])
def test_compacted_history_renders_fresh_without_payload(runtime: AgentRuntime, payload: str) -> None:
    async def scenario() -> None:
        await compact_history(runtime)
        messages = dump(runtime.messages)
        tokens = runtime.model.session.transcript.tokens
        preamble = runtime.model.binder.records()[0].start
        saved = await runtime.save()
        path = runtime.store.session_dir(saved.session_id) / "session.kv"
        if payload == "missing":
            path.unlink()
        else:
            runtime.store.write_kv_atomic(path, b"")
        await runtime.new()
        syncs = collect_syncs(runtime)
        await runtime.switch(saved.session_id)
        assert dump(runtime.messages) == messages
        assert runtime.model.session.transcript.tokens == ()
        assert runtime.model.binder.records() == ()
        assert syncs == []
        result = await runtime.run("continue")
        assert result.output == "continued"
        assert len(syncs) == 2  # Fresh preamble setup, then history plus the new request.
        assert (syncs[0].plan.kind, syncs[0].plan.reused, syncs[0].prefilled) == ("rebuild", 0, preamble)
        assert (syncs[1].plan.kind, syncs[1].plan.reused, syncs[1].prefilled) == (
            "extend",
            preamble,
            len(tokens) - preamble + 33,
        )
        assert sum(event.prefilled for event in syncs) == len(tokens) + 33
        assert result.usage.input_tokens == len(tokens) - preamble + 33
        assert runtime.model.session.transcript.tokens[: len(tokens)] == tokens
        assert len(runtime.model.session.transcript) == len(tokens) + 43
        # Without loaded binding, the harness can merge adjacent requests; both
        # user parts must still render, with a complete new provenance mapping.
        assert len(runtime.messages) == len(runtime.model.binder.records()) == 4
        assert [
            part.content for message in runtime.messages for part in message.parts if isinstance(part, UserPromptPart)
        ] == [
            "durable state",
            "recent",
            "continue",
        ]
        rerendered = dump(runtime.messages)
        rebound = runtime.model.binder.records()
        resaved = await runtime.save()
        await runtime.new()
        await runtime.switch(resaved.session_id)
        assert dump(runtime.messages) == rerendered
        assert runtime.model.binder.records() == rebound

    asyncio.run(scenario())


def test_fork_keeps_compacted_binding_and_live_kv(runtime: AgentRuntime) -> None:
    async def scenario() -> None:
        await compact_history(runtime)
        saved = await runtime.save()
        binder = runtime.model.binder
        session = runtime.model.session
        messages, tokens = dump(runtime.messages), session.transcript.tokens
        records, segments = binder.records(), binder.export_segments()
        syncs = collect_syncs(runtime)
        forked = await runtime.fork()
        assert forked != saved.session_id
        assert syncs == []
        assert dump(runtime.messages) == messages
        assert session.transcript.tokens == tokens
        assert binder.records() == records
        assert binder.export_segments() == segments
        assert session.plan().to_prefill == 0
        result = await runtime.run("continue")
        assert result.output == "continued"
        assert (syncs[0].plan.kind, syncs[0].plan.reused, syncs[0].prefilled) == ("extend", len(tokens), 33)
        assert result.usage.input_tokens == 33
        assert len(runtime.messages) == len(binder.records()) == 5
        assert session.transcript.tokens[: len(tokens)] == tokens
        fork_record = await runtime.save()
        assert fork_record.session_id == forked
        assert fork_record.tokens == len(tokens) + 43
        assert runtime.store.find(saved.session_id)[0] == saved

    asyncio.run(scenario())
