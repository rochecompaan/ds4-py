"""Restore conformance for summary requests, invalid provenance, and abandoned work."""

import asyncio
from dataclasses import replace

import pytest
import test_compaction_persistence as persistence
from pydantic_ai.messages import ModelRequest, ModelResponse
from test_compaction_persistence import collect_syncs, compact_history, dump

from ds4py.errors import EngineError, StoreError
from ds4py.runtime import AgentRuntime
from ds4py.store import KV_HEADER, SegmentProvenance

runtime = persistence.runtime  # Share the isolated real-runtime fixture.


def test_load_payload_and_binder_restore_install_summary_request_provenance(runtime: AgentRuntime) -> None:
    async def scenario() -> None:
        await compact_history(runtime)
        saved = await runtime.save()
        binder = runtime.model.binder
        session = runtime.model.session
        messages = runtime.messages
        summary = messages[0]
        assert isinstance(summary, ModelRequest)
        assert not hasattr(summary, "provider_response_id")
        assert isinstance(messages[-1], ModelResponse)
        tokens = session.transcript.tokens
        records, segments = binder.records(), binder.export_segments()
        await runtime.run("abandon")
        syncs = collect_syncs(runtime)
        with (runtime.store.session_dir(saved.session_id) / "session.kv").open("rb") as file:
            file.seek(KV_HEADER.size)
            loaded = session.load_payload(file, payload_bytes=saved.payload_bytes)
        binder.restore(messages, saved.segments)
        assert syncs == []
        assert tuple(loaded) == tokens == session.transcript.tokens
        assert binder.records() == records
        assert binder.export_segments() == segments
        assert session.history.checkpoint_tokens == {}
        assert session.plan().reused == len(tokens)
        assert session.plan().to_prefill == 0
        # Keep runtime history paired with the explicitly loaded payload.
        runtime._messages = messages
        result = await runtime.run("continue")
        assert result.output == "continued"
        assert result.usage.input_tokens == 33
        assert len(syncs) == 1 and syncs[0].prefilled == 33
        assert syncs[0].plan.reused == len(tokens)
        assert len(session.transcript) == len(tokens) + 43
        assert binder.records()[:3] == records
        assert len(runtime.messages) == len(binder.records()) == 5

    asyncio.run(scenario())


def test_save_record_holds_the_canonical_exported_provenance(
    runtime: AgentRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        await compact_history(runtime)
        binder = runtime.model.binder
        exported: list[tuple[SegmentProvenance, ...]] = []
        export_segments = binder.export_segments

        def capture() -> tuple[SegmentProvenance, ...]:
            segments = export_segments()
            exported.append(segments)
            return segments

        monkeypatch.setattr(binder, "export_segments", capture)
        saved = await runtime.save()
        assert exported
        assert {type(segment) for segment in saved.segments} == {SegmentProvenance}
        assert saved.segments is exported[-1]

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["switch", "binder"])
def test_bad_summary_key_is_rejected_before_live_state_changes(runtime: AgentRuntime, boundary: str) -> None:
    async def scenario() -> None:
        await compact_history(runtime)
        saved = await runtime.save()
        summary_key = (runtime.messages[0].metadata or {})["ds4.key"]
        bad_segments = tuple(
            replace(segment, key="absent-summary") if segment.key == summary_key else segment
            for segment in saved.segments
        )
        bad = replace(saved, segments=bad_segments)
        if boundary == "switch":
            runtime.store.write_record_atomic(runtime.store.session_dir(saved.session_id) / "save.json", bad)
            await runtime.new()
            await runtime.run("unrelated session")
        binder = runtime.model.binder
        session = runtime.model.session
        before = (
            runtime.session_id,
            dump(runtime.messages),
            session.transcript.tokens,
            session.transcript.segments,
            binder.records(),
            binder.export_segments(),
        )
        native = session._worker.call(lambda e, n: tuple(n.tokens())).result()
        syncs = collect_syncs(runtime)
        if boundary == "switch":
            with pytest.raises(StoreError, match=r"absent-summary.*absent from the snapshot"):
                await runtime.switch(saved.session_id)
        else:
            with pytest.raises(EngineError, match=r"absent-summary.*absent from the snapshot"):
                binder.restore(runtime.messages, bad_segments)
        assert (
            runtime.session_id,
            dump(runtime.messages),
            session.transcript.tokens,
            session.transcript.segments,
            binder.records(),
            binder.export_segments(),
        ) == before
        assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == native
        assert syncs == []

    asyncio.run(scenario())


@pytest.mark.parametrize("detour", ["same-session", "other-session"])
def test_abandoned_post_save_continuation_never_crosses_sessions(runtime: AgentRuntime, detour: str) -> None:
    async def scenario() -> None:
        await compact_history(runtime)
        saved = await runtime.save()
        messages = dump(runtime.messages)
        tokens = runtime.model.session.transcript.tokens
        records = runtime.model.binder.records()
        await runtime.run("discard this unsaved continuation")
        discarded_keys = {record.key for record in runtime.model.binder.records()[3:]}
        other_id: str | None = None
        other_messages = b""
        other_tokens: tuple[int, ...] = ()
        if detour == "other-session":
            other_id = await runtime.new()
            await runtime.run("independent task")
            other_messages = dump(runtime.messages)
            other_tokens = runtime.model.session.transcript.tokens
            await runtime.save()
        syncs = collect_syncs(runtime)
        await runtime.switch(saved.session_id)
        assert syncs == []
        assert dump(runtime.messages) == messages
        assert runtime.model.session.transcript.tokens == tokens
        assert runtime.model.binder.records() == records
        result = await runtime.run("continue")
        assert result.output == "continued"
        assert result.usage.input_tokens == 33
        assert len(syncs) == 1
        assert (syncs[0].plan.kind, syncs[0].plan.reused, syncs[0].prefilled) == ("extend", len(tokens), 33)
        assert len(runtime.model.session.transcript) == len(tokens) + 43
        assert len(runtime.messages) == len(runtime.model.binder.records()) == 5
        assert not discarded_keys.intersection(record.key for record in runtime.model.binder.records())
        if other_id is not None:
            syncs.clear()
            await runtime.switch(other_id)
            assert syncs == []
            assert dump(runtime.messages) == other_messages
            assert runtime.model.session.transcript.tokens == other_tokens
            assert not {record.key for record in records}.intersection(
                record.key for record in runtime.model.binder.records()
            )
            result = await runtime.run("continue")
            assert result.output == "continued"
            assert result.usage.input_tokens == 33
            assert len(syncs) == 1 and syncs[0].plan.reused == len(other_tokens)
            assert len(runtime.messages) == len(runtime.model.binder.records()) == 4

    asyncio.run(scenario())
