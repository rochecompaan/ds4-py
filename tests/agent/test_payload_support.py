"""Generic stripped-save behavior when native file payloads are unsupported."""

import asyncio
import json
from pathlib import Path
from typing import Any, Never, cast

import pytest
from pydantic_ai_harness.step_persistence import FileStepStore
from test_restore import dump, engine_events, read_events, saved_session
from test_runtime import fake_engine, fixed_clock, make_model, project_dir, scripted_rounds

from ds4py._native.fake import FakeNative
from ds4py.engine.session import Session
from ds4py.errors import StoreError
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
            assert first_snapshot is not None
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


def test_unsupported_engine_renders_a_planned_kv_switch_before_native_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, store, record, saved = saved_session(tmp_path, script=("first answer",))

    def forbid_native_load(*args: Any, **kwargs: Any) -> Never:
        raise AssertionError("native payload load must not run")

    monkeypatch.setattr(Session, "load_payload", forbid_native_load)
    order: list[str] = []
    original_probe = Session.file_payloads_supported

    def record_probe(session: Session) -> bool:
        order.append("probe")
        return original_probe(session)

    monkeypatch.setattr(Session, "file_payloads_supported", record_probe)
    with fake_engine(script=scripted_rounds("second answer"), payload_support=False) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(
            model,
            project,
            store=store,
            clock=fixed_clock,
            before_identity_change=lambda: order.append("before"),
        )
        try:
            during_switch: list[Any] = []
            runtime.on_event(during_switch.append, kinds={"Prefill", "Synced"})
            asyncio.run(runtime.switch(record.session_id))
            assert order == ["probe", "before"]
            assert during_switch == []
            assert dump(runtime.messages) == dump(saved["messages"])
            assert model.session.transcript.tokens == ()

            requests: list[Any] = []
            runtime.on_event(requests.append, kinds={"Synced"})
            asyncio.run(runtime.run("follow up"))
            synced = engine_events(requests, "Synced")
            assert synced[0].plan.kind == "rebuild"
            assert synced[0].prefilled == synced[0].pos
            assert list(model.session.transcript.tokens[: len(saved["tokens"])]) == saved["tokens"]
            assert dump(runtime.messages[: len(saved["messages"])]) == dump(saved["messages"])
        finally:
            asyncio.run(runtime.close())

    switched = [
        event
        for event in read_events(store.session_dir(record.session_id) / "events.jsonl")
        if event["kind"] == "switched"
    ]
    assert switched[-1]["payload"]["restore_path"] == "render"
    assert switched[-1]["payload"]["mismatch"] == "the live engine does not support file payloads"


def test_explicit_payload_support_retains_exact_kv_switch(tmp_path: Path) -> None:
    project, store, record, saved = saved_session(tmp_path, script=("first answer",))
    with fake_engine(script=scripted_rounds("second answer"), payload_support=True) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            during_switch: list[Any] = []
            runtime.on_event(during_switch.append, kinds={"Prefill", "Synced"})
            asyncio.run(runtime.switch(record.session_id))
            assert during_switch == []
            assert list(model.session.transcript.tokens) == saved["tokens"]
            assert model.binder.export_segments() == record.segments
            assert dump(runtime.messages) == dump(saved["messages"])

            requests: list[Any] = []
            runtime.on_event(requests.append, kinds={"Synced"})
            asyncio.run(runtime.run("follow up"))
            synced = engine_events(requests, "Synced")
            assert synced[0].plan.kind == "extend"
            assert synced[0].plan.reused == len(saved["tokens"])
        finally:
            asyncio.run(runtime.close())


@pytest.mark.parametrize("damage", ["model", "provenance", "kv"])
def test_switch_validation_errors_precede_capability_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    project, store, record, _saved = saved_session(tmp_path, script=("first answer",))
    record_path = store.session_dir(record.session_id) / "save.json"
    if damage in {"model", "provenance"}:
        data = json.loads(record_path.read_text())
        if damage == "model":
            data["model_id"] += 1
        else:
            keyed = [index for index, segment in enumerate(data["segments"]) if segment["key"]]
            data["segments"][keyed[-1]]["content_hash"] = "f" * 64
        record_path.write_text(json.dumps(data))
    else:
        (record_path.with_name("session.kv")).write_bytes(b"not a KV header")

    def forbid_early_probe(*args: Any, **kwargs: Any) -> Never:
        raise AssertionError("capability probe must follow validation")

    monkeypatch.setattr(Session, "file_payloads_supported", forbid_early_probe)
    expected = {"model": "model id", "provenance": "cannot restore", "kv": r"session\.kv"}[damage]
    with fake_engine(payload_support=False) as engine:
        runtime = AgentRuntime(make_model(engine), project, store=store, clock=fixed_clock)
        original_session_id = runtime.session_id
        try:
            with pytest.raises(StoreError, match=expected):
                asyncio.run(runtime.switch(record.session_id))
            assert runtime.session_id == original_session_id
            assert runtime.messages == []
        finally:
            asyncio.run(runtime.close())


def test_zero_payload_switch_never_probes_or_reads_a_stale_kv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore(tmp_path / "home")
    project = project_dir(tmp_path)
    with fake_engine(script=scripted_rounds("first answer"), payload_support=False) as engine:
        runtime = AgentRuntime(make_model(engine), project, store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.run("hello"))
            record = asyncio.run(runtime.save())
            messages = dump(runtime.messages)
        finally:
            asyncio.run(runtime.close())

    kv_path = store.session_dir(record.session_id) / "session.kv"
    kv_path.write_bytes(b"stale corrupt KV")

    def forbid_artifact_access(*args: Any, **kwargs: Any) -> Never:
        raise AssertionError("zero-payload restore must not probe or read KV")

    monkeypatch.setattr(Session, "file_payloads_supported", forbid_artifact_access)
    monkeypatch.setattr(Session, "load_payload", forbid_artifact_access)
    monkeypatch.setattr(store, "read_kv_header", forbid_artifact_access)
    with fake_engine(payload_support=False) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.switch(record.session_id))
            assert dump(runtime.messages) == messages
            assert model.session.transcript.tokens == ()
        finally:
            asyncio.run(runtime.close())
