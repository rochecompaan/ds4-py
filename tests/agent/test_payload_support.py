"""Generic stripped-save behavior when native file payloads are unsupported."""

import asyncio
from pathlib import Path
from typing import Any, Never, cast

import pytest
from pydantic_ai_harness.step_persistence import FileStepStore
from test_restore import dump, read_events
from test_runtime import fake_engine, fixed_clock, make_model, project_dir, scripted_rounds

from ds4py._native.fake import FakeNative
from ds4py.engine.session import Session
from ds4py.runtime import AgentRuntime
from ds4py.store import SaveRecord, SessionStore, token_digest


def latest_snapshot(store: SessionStore, record: SaveRecord) -> Any:
    steps = FileStepStore(store.session_dir(record.session_id) / "steps")
    return asyncio.run(steps.latest_snapshot(run_id=record.run_id, include_interrupted=True))


def test_unsupported_file_payloads_write_a_complete_stripped_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("first answer"), payload_support=False) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project_dir(tmp_path), store=store, clock=fixed_clock)
        saved_events: list[Any] = []
        runtime.on_event(saved_events.append, kinds={"saved"})

        def forbid_native_save(*args: Any, **kwargs: Any) -> Never:
            raise AssertionError("native payload staging must not run")

        monkeypatch.setattr(Session, "save_payload", forbid_native_save)
        try:
            asyncio.run(runtime.run("hello"))
            messages = dump(runtime.messages)
            tokens = list(model.session.transcript.tokens)
            segments = model.binder.export_segments()
            record = asyncio.run(runtime.save())
            snapshot = latest_snapshot(store, record)
        finally:
            asyncio.run(runtime.close())

    session_dir = store.session_dir(record.session_id)
    assert record.payload_bytes == 0
    assert record.tokens == len(tokens) > 0
    assert record.tokens_sha256 == token_digest(tokens)
    assert record.segments == segments
    assert record.model_id == engine.info.model_id
    assert record.quant_bits == engine.info.routed_quant_bits
    assert not (session_dir / "session.kv").exists()
    assert snapshot is not None
    assert snapshot.state == "complete"
    assert dump(snapshot.messages) == messages
    assert len(saved_events) == 1
    assert saved_events[0].payload["payload_bytes"] == 0


def test_explicit_native_payload_support_retains_kv_save(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("first answer"), payload_support=True) as engine:
        runtime = AgentRuntime(make_model(engine), project_dir(tmp_path), store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.run("hello"))
            record = asyncio.run(runtime.save())
        finally:
            asyncio.run(runtime.close())

    assert record.payload_bytes > 0
    assert (store.session_dir(record.session_id) / "session.kv").stat().st_size > record.payload_bytes


class SimulatedCrash(Exception):
    pass


@pytest.mark.parametrize("crash_point", ["before_unlink", "after_unlink"])
def test_stripped_replacement_publishes_record_on_both_sides_of_stale_kv_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
) -> None:
    store = SessionStore(tmp_path / "home")
    project = project_dir(tmp_path)
    with fake_engine(script=scripted_rounds("first answer", "second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.run("first"))
            first_record = asyncio.run(runtime.save())
            first_snapshot = latest_snapshot(store, first_record)
            session_dir = store.session_dir(first_record.session_id)
            kv_path = session_dir / "session.kv"
            original_kv = kv_path.read_bytes()

            asyncio.run(runtime.run("second"))
            replacement_messages = dump(runtime.messages)
            replacement_tokens = list(model.session.transcript.tokens)
            cast("FakeNative", engine.native).payload_support = False

            if crash_point == "before_unlink":
                original_write = store.write_record_atomic

                def crash_after_publication(path: Any, record: Any) -> None:
                    original_write(path, record)
                    raise SimulatedCrash("after record publication")

                monkeypatch.setattr(store, "write_record_atomic", crash_after_publication)
            else:
                original_unlink = Path.unlink

                def crash_after_removal(path: Path, *args: Any, **kwargs: Any) -> None:
                    original_unlink(path, *args, **kwargs)
                    if path == kv_path:
                        raise SimulatedCrash("after stale KV removal")

                monkeypatch.setattr(Path, "unlink", crash_after_removal)

            with pytest.raises(SimulatedCrash):
                asyncio.run(runtime.save())
        finally:
            asyncio.run(runtime.close())

    replacement = store.read_record(session_dir / "save.json")
    assert replacement.run_id != first_record.run_id
    assert replacement.payload_bytes == 0
    assert replacement.tokens == len(replacement_tokens)
    assert replacement.tokens_sha256 == token_digest(replacement_tokens)
    assert kv_path.exists() is (crash_point == "before_unlink")
    if crash_point == "before_unlink":
        assert kv_path.read_bytes() == original_kv
    replacement_snapshot = latest_snapshot(store, replacement)
    assert replacement_snapshot is not None
    assert dump(replacement_snapshot.messages) == replacement_messages
    assert latest_snapshot(store, first_record) == first_snapshot

    with fake_engine(payload_support=False) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.switch(replacement.session_id))
            assert dump(runtime.messages) == replacement_messages
            assert model.session.transcript.tokens == ()
        finally:
            asyncio.run(runtime.close())

    switched = [event for event in read_events(session_dir / "events.jsonl") if event["kind"] == "switched"]
    assert switched[-1]["payload"]["restore_path"] == "render"


def test_failed_stripped_record_publication_keeps_the_old_record_and_kv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("first answer", "second answer")) as engine:
        runtime = AgentRuntime(make_model(engine), project_dir(tmp_path), store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.run("first"))
            first_record = asyncio.run(runtime.save())
            kv_path = store.session_dir(first_record.session_id) / "session.kv"
            original_kv = kv_path.read_bytes()
            asyncio.run(runtime.run("second"))
            cast("FakeNative", engine.native).payload_support = False

            def refuse_publication(path: Any, record: Any) -> Never:
                raise SimulatedCrash("record publication failed")

            monkeypatch.setattr(store, "write_record_atomic", refuse_publication)
            with pytest.raises(SimulatedCrash, match="record publication failed"):
                asyncio.run(runtime.save())
        finally:
            asyncio.run(runtime.close())

    assert store.read_record(kv_path.with_name("save.json")) == first_record
    assert kv_path.read_bytes() == original_kv
