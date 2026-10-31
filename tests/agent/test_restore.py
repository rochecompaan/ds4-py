"""Save, switch, new, fork, and close behavior (Task 9)."""

import asyncio
import json
import shutil
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never, cast

import pytest
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, TextPart

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.engine.sampling import SamplingPolicy
from ds4py.engine.session import Engine
from ds4py.errors import EngineError, StoreError
from ds4py.model import DS4Model
from ds4py.runtime import AgentRuntime
from ds4py.store import KV_HEADER, KV_MAGIC, KV_VERSION, SessionStore, token_digest

READ_CALL = "<tool_call><function=read_file><parameter=path>hello.txt</parameter></function></tool_call>"


def fixed_clock() -> datetime:
    return datetime(2026, 10, 21, 9, 30, 0, tzinfo=UTC)


def fake_engine(*, script: Any = None, family: Any = "qwen", model_id: Any = 7, quant_bits: Any = None) -> Engine:
    info = replace(
        FakeNative()._info,
        name="fake-ds4",
        model_id=model_id,
        family=family,
        is_deepseek41=False,
        is_glm_dsa=False,
        is_qwen4=family == "qwen",
    )
    if quant_bits is not None:
        info = replace(info, routed_quant_bits=quant_bits)
    native = FakeNative(info=info) if script is None else FakeNative(info=info, script=script)
    return ds4py.open("f.gguf", ctx=65536, think=ThinkMode.NONE, native=native)


def scripted_rounds(*texts: Any) -> Any:
    rounds = [tuple(tokenize_rendered(text)) for text in texts]
    state = {"round": -1}

    def next_token(prompt: Any, generated: Any) -> Any:
        if not generated:
            state["round"] += 1
        sequence = rounds[min(state["round"], len(rounds) - 1)]
        return sequence[len(generated)] if len(generated) < len(sequence) else FAKE_EOS

    return next_token


def project_dir(tmp_path: Path) -> Any:
    project = tmp_path / "project"
    project.mkdir()
    (project / "hello.txt").write_text("world")
    return project


def make_model(engine: Any, **kwargs: Any) -> DS4Model:
    return DS4Model(engine.session(), sampling=SamplingPolicy(seed=1), **kwargs)


def dump(messages: Any) -> str:
    return ModelMessagesTypeAdapter.dump_json(list(messages)).decode()


def capture(runtime: Any, kinds: Any) -> Any:
    seen = []

    def listener(event: Any) -> None:
        seen.append(event)

    runtime.on_event(listener, kinds=kinds)
    return seen


def engine_events(seen: Any, kind: Any) -> Any:
    return [event.payload["event"] for event in seen if event.kind == kind]


def read_events(path: Any) -> Any:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def saved_session(tmp_path: Path, script: Any = (READ_CALL, "first answer"), *, hints: Any = False) -> Any:
    """Run one prompt and save; return the runtime parts still needed."""
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    state = {}

    with fake_engine(script=scripted_rounds(*script)) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        cast("Any", runtime._notes).hints_enabled = hints
        try:
            asyncio.run(runtime.run("read hello"))
            record = asyncio.run(runtime.save())
            state.update(
                session_id=runtime.session_id,
                messages=list(runtime.messages),
                tokens=list(model.session.transcript.tokens),
            )
        finally:
            asyncio.run(runtime.close())
    return project, store, record, state


def test_save_and_switch_with_skill_history_preserves_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    skill = home / ".claude" / "skills" / "test-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: test-skill\ndescription: Exercise deferred capability instructions.\n---\nUse the test skill.\n"
    )
    monkeypatch.setenv("HOME", str(home))
    project, store, record, saved = saved_session(tmp_path, script=("first answer",))
    assert any("test-skill" in str(message) for message in saved["messages"])

    with fake_engine() as engine:
        runtime = AgentRuntime(make_model(engine), project, store=store, clock=fixed_clock)
        try:
            restored = asyncio.run(runtime.switch(record.session_id))
            assert restored == record
            assert dump(runtime.messages) == dump(saved["messages"])
        finally:
            asyncio.run(runtime.close())


def test_partial_response_does_not_replace_bound_history_in_saved_snapshot(tmp_path: Path) -> None:
    # The harness may expose a partial processed response while the binder's
    # completed response is the version represented by the sampled tokens.
    project, store, record, saved = saved_session(tmp_path, script=("first answer",))
    with fake_engine() as engine:
        runtime = AgentRuntime(make_model(engine), project, store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.switch(record.session_id))
            steps = _steps(store.session_dir(record.session_id))
            snapshot = asyncio.run(steps.latest_snapshot(run_id=record.run_id))
            response = next(message for message in snapshot.messages if isinstance(message, ModelResponse))
            partial = deepcopy(response)
            partial.parts = partial.parts[:1]
            cast("TextPart", partial.parts[0]).content = "partial"
            candidates = [partial if message is response else message for message in snapshot.messages]
            snapshot.messages = candidates
            asyncio.run(runtime._adopt_history(record.run_id, candidates, snapshot))
            updated = asyncio.run(runtime.save())
        finally:
            asyncio.run(runtime.close())

    with fake_engine() as engine:
        runtime = AgentRuntime(make_model(engine), project, store=store, clock=fixed_clock)
        try:
            restored = asyncio.run(runtime.switch(updated.session_id))
            assert restored == updated
            assert dump(runtime.messages) == dump(saved["messages"])
        finally:
            asyncio.run(runtime.close())


def test_valid_kv_switch_restores_binding_without_prefill(tmp_path: Path) -> None:
    project, store, record, saved = saved_session(tmp_path)
    session_id = saved["session_id"]
    assert record.session_id == session_id
    assert record.run_id == f"{session_id}.save-1"
    assert record.payload_bytes > 0
    assert record.tokens == len(saved["tokens"])
    assert record.tokens_sha256 == token_digest(saved["tokens"])

    with fake_engine(script=scripted_rounds("second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            during_switch = capture(runtime, {"Prefill", "Synced"})
            restored = asyncio.run(runtime.switch(session_id[:8]))

            assert restored == record
            assert runtime.session_id == session_id
            assert runtime.title == record.title
            assert during_switch == []  # nothing was rendered or prefilled

            loaded = list(model.session.transcript.tokens)
            assert loaded == saved["tokens"]
            assert len(loaded) == record.tokens
            assert token_digest(loaded) == record.tokens_sha256
            assert dump(runtime.messages) == dump(saved["messages"])
            assert all(cast("Any", message.metadata)["ds4.key"] for message in runtime.messages)

            # Every saved provenance range is installed on the loaded transcript
            # and every keyed range still names its snapshot message.
            def installed_view(segments: Any) -> Any:
                return [
                    (
                        segment.start,
                        segment.end,
                        "note" if segment.kind.startswith("system_note") else segment.kind,
                        segment.key,
                        segment.content_hash,
                        segment.source,
                    )
                    for segment in segments
                ]

            installed = [
                (
                    segment.start,
                    segment.end,
                    segment.kind,
                    segment.meta.get("key"),
                    segment.meta.get("content_hash"),
                    segment.meta.get("source"),
                )
                for segment in model.session.transcript.segments
            ]
            assert installed == installed_view(record.segments)
            assert [
                segment.meta.get("subtype") for segment in model.session.transcript.segments if segment.kind == "note"
            ] == ["date"]
            responses = [m for m in runtime.messages if isinstance(m, ModelResponse)]
            assert responses and all(cast("Any", m.metadata)["ds4.key"] for m in responses)
            keyed = [segment for segment in record.segments if segment.key]
            assert keyed and all(segment.key for segment in keyed)

            # The next request extends the loaded KV instead of re-rendering it.
            requests = capture(runtime, {"Synced"})
            asyncio.run(runtime.run("second answer"))
            synced = engine_events(requests, "Synced")
            assert synced and synced[0].plan.kind == "extend"
            # The loaded payload holds every saved token, so nothing re-renders:
            # only the follow-up request is prefilled.
            assert synced[0].plan.reused == len(saved["tokens"])
            assert 0 < synced[0].prefilled == synced[0].pos - len(saved["tokens"])
        finally:
            asyncio.run(runtime.close())


def zero_payload_kv(store: Any, session_id: Any) -> None:
    store.write_kv_atomic(store.session_dir(session_id) / "session.kv", b"")


def test_native_restore_does_not_emit_a_second_date_note(tmp_path: Path) -> None:
    project, store, record, _saved = saved_session(tmp_path, script=("first answer",))
    later_clock = lambda: datetime(2026, 10, 22, 9, 30, 0, tzinfo=UTC)

    with fake_engine(script=scripted_rounds("second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=later_clock)
        try:
            asyncio.run(runtime.switch(record.session_id))
            asyncio.run(runtime.run("follow up"))

            assert [
                segment.meta.get("subtype") for segment in model.session.transcript.segments if segment.kind == "note"
            ] == ["date"]
        finally:
            asyncio.run(runtime.close())


@pytest.mark.parametrize("hints", [False, True])
def test_stripped_payload_switch_renders_fresh_with_full_prefill(tmp_path: Path, hints: Any) -> None:
    project, store, record, saved = saved_session(tmp_path, hints=hints)
    session_id = saved["session_id"]

    # Case 1 output with the same follow-up is the deterministic reference.
    reference = {}
    with fake_engine(script=scripted_rounds("second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.switch(session_id))
            reference["output"] = asyncio.run(runtime.run("second answer")).output
        finally:
            asyncio.run(runtime.close())

    zero_payload_kv(store, session_id)

    with fake_engine(script=scripted_rounds("second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            during_switch = capture(runtime, {"Prefill", "Synced"})
            restored = asyncio.run(runtime.switch(session_id))

            assert restored == record
            assert during_switch == []  # the switch itself never renders
            assert len(model.session.transcript) == 0
            assert not model.session.transcript.chat
            assert dump(runtime.messages) == dump(saved["messages"])
            assert cast("Any", runtime._notes).hints_enabled is hints

            requests = capture(runtime, {"Synced"})
            output = asyncio.run(runtime.run("second answer")).output
            synced = engine_events(requests, "Synced")
            assert synced[0].plan.kind == "rebuild"
            assert synced[0].prefilled == synced[0].pos  # full prefill
            rebuilt = list(model.session.transcript.tokens[: len(saved["tokens"])])
            assert rebuilt == saved["tokens"]
            note_subtypes = [
                segment.meta.get("subtype") for segment in model.session.transcript.segments if segment.kind == "note"
            ]
            assert note_subtypes == (["date", "hints_full"] if hints else ["date"])
            assert dump(runtime.messages[: len(saved["messages"])]) == dump(saved["messages"])
            assert output == reference["output"]
        finally:
            asyncio.run(runtime.close())

    events = read_events(store.session_dir(session_id) / "events.jsonl")
    switched = [line for line in events if line["kind"] == "switched"]
    assert switched and switched[-1]["payload"]["restore_path"] == "render"
    assert "payload" in switched[-1]["payload"]["mismatch"]


def test_newer_kv_than_record_takes_the_render_path_and_reports_the_mismatch(tmp_path: Path) -> None:
    project, store, record, saved = saved_session(tmp_path)
    session_id = saved["session_id"]
    kv_path = store.session_dir(session_id) / "session.kv"
    # A crash between the KV write and the record write leaves a newer payload.
    store.write_kv_atomic(kv_path, b"x" * (record.payload_bytes + 2048))

    with fake_engine(script=scripted_rounds("second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.switch(session_id))
            assert len(model.session.transcript) == 0
            assert dump(runtime.messages) == dump(saved["messages"])
        finally:
            asyncio.run(runtime.close())

    switched = [
        line for line in read_events(store.session_dir(session_id) / "events.jsonl") if line["kind"] == "switched"
    ]
    assert switched[-1]["payload"]["restore_path"] == "render"
    assert str(record.payload_bytes) in switched[-1]["payload"]["mismatch"]


def test_token_hash_mismatch_renders_and_reports_the_actual_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, store, _record, saved = saved_session(tmp_path)
    session_id = saved["session_id"]

    with fake_engine(script=scripted_rounds("second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        original_load = model.session.load_payload

        def load_with_wrong_digest(file: Any, **kwargs: Any) -> Any:
            loaded = list(original_load(file, **kwargs))
            loaded[0] += 1
            return loaded

        monkeypatch.setattr(model.session, "load_payload", load_with_wrong_digest)
        try:
            asyncio.run(runtime.switch(session_id))
            assert len(model.session.transcript) == 0
            asyncio.run(runtime.run("second answer"))
            rebuilt = list(model.session.transcript.tokens[: len(saved["tokens"])])
            assert rebuilt == saved["tokens"]
        finally:
            asyncio.run(runtime.close())

    switched = [
        line for line in read_events(store.session_dir(session_id) / "events.jsonl") if line["kind"] == "switched"
    ]
    assert switched[-1]["payload"]["restore_path"] == "render"
    assert "hash" in switched[-1]["payload"]["mismatch"]


def test_invalid_snapshot_does_not_commit_switch_or_overwrite_target(tmp_path: Path) -> None:
    project, store, record, saved = saved_session(tmp_path)
    steps = _steps(store.session_dir(saved["session_id"]))
    snapshot = asyncio.run(steps.latest_snapshot(run_id=record.run_id))
    response = next(message for message in snapshot.messages if isinstance(message, ModelResponse))
    metadata = dict(response.metadata or {})
    metadata.pop("ds4.key")
    response.metadata = metadata
    _replace_snapshot(steps, snapshot)
    target_path = store.session_dir(saved["session_id"]) / "save.json"
    target_before = target_path.read_bytes()

    class State:
        closed = 0

        def close(self) -> None:
            self.closed += 1

    with fake_engine(script=scripted_rounds("old answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        state = State()

        async def scenario() -> None:
            await runtime.run("old conversation")
            old_id = runtime.session_id
            old_messages = list(runtime.messages)
            model.session.state("test", lambda: state)

            with pytest.raises(StoreError, match="key"):
                await runtime.switch(saved["session_id"])

            assert runtime.session_id == old_id
            assert dump(runtime.messages) == dump(old_messages)
            assert state.closed == 0
            await runtime.save()
            assert target_path.read_bytes() == target_before

        try:
            asyncio.run(scenario())
        finally:
            asyncio.run(runtime.close())


def test_native_restore_failure_never_saves_old_history_under_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, store, _record, saved = saved_session(tmp_path)
    target_path = store.session_dir(saved["session_id"]) / "save.json"
    target_before = target_path.read_bytes()

    with fake_engine(script=scripted_rounds("old answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)

        async def scenario() -> None:
            await runtime.run("old conversation")

            def reject_load(file: Any, **kwargs: Any) -> Never:
                raise EngineError("native restore rejected")

            monkeypatch.setattr(model.session, "load_payload", reject_load)
            with pytest.raises(EngineError, match="native restore rejected"):
                await runtime.switch(saved["session_id"])

            assert runtime.session_id == saved["session_id"]
            assert dump(runtime.messages) == dump(saved["messages"])
            assert model.session.transcript.tokens == ()
            assert model.binder._records == []
            with pytest.raises(StoreError, match="empty transcript"):
                await runtime.save()
            assert target_path.read_bytes() == target_before

        try:
            asyncio.run(scenario())
        finally:
            asyncio.run(runtime.close())


def test_missing_named_snapshot_fails_with_the_run_id(tmp_path: Path) -> None:
    project, store, record, saved = saved_session(tmp_path)
    shutil.rmtree(store.session_dir(saved["session_id"]) / "steps" / record.run_id)

    with fake_engine(script=scripted_rounds("second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            with pytest.raises(StoreError, match=record.run_id):
                asyncio.run(runtime.switch(saved["session_id"]))
            assert runtime.messages == []
        finally:
            asyncio.run(runtime.close())


@pytest.mark.parametrize("restore_path", ["kv", "render"])
def test_missing_segment_key_fails_with_the_key(tmp_path: Path, restore_path: Any) -> None:
    project, store, record, saved = saved_session(tmp_path)
    if restore_path == "render":
        zero_payload_kv(store, saved["session_id"])
    keyed = [index for index, segment in enumerate(record.segments) if segment.key]
    assert keyed
    target_key = record.segments[keyed[-1]].key
    changed = [replace(segment, key="0" * 32) if segment.key == target_key else segment for segment in record.segments]
    broken = replace(record, segments=tuple(changed))
    store.write_record_atomic(store.session_dir(saved["session_id"]) / "save.json", broken)

    with fake_engine(script=scripted_rounds("second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            with pytest.raises(StoreError, match="0" * 32):
                asyncio.run(runtime.switch(saved["session_id"]))
            assert runtime.messages == []
        finally:
            asyncio.run(runtime.close())


@pytest.mark.parametrize("restore_path", ["kv", "render"])
def test_first_request_key_tamper_is_rejected(tmp_path: Path, restore_path: Any) -> None:
    project, store, record, saved = saved_session(tmp_path)
    if restore_path == "render":
        zero_payload_kv(store, saved["session_id"])
    first_key = next(segment.key for segment in record.segments if segment.key)
    changed = [replace(segment, key="0" * 32) if segment.key == first_key else segment for segment in record.segments]
    store.write_record_atomic(
        store.session_dir(saved["session_id"]) / "save.json",
        replace(record, segments=tuple(changed)),
    )

    with fake_engine(script=scripted_rounds("second answer")) as engine:
        runtime = AgentRuntime(make_model(engine), project, store=store, clock=fixed_clock)
        try:
            with pytest.raises(StoreError, match="0" * 32):
                asyncio.run(runtime.switch(saved["session_id"]))
        finally:
            asyncio.run(runtime.close())


@pytest.mark.parametrize("restore_path", ["kv", "render"])
def test_missing_response_key_is_rejected(tmp_path: Path, restore_path: Any) -> None:
    project, store, record, saved = saved_session(tmp_path)
    if restore_path == "render":
        zero_payload_kv(store, saved["session_id"])
    steps = _steps(store.session_dir(saved["session_id"]))
    snapshot = asyncio.run(steps.latest_snapshot(run_id=record.run_id))
    response = next(message for message in snapshot.messages if isinstance(message, ModelResponse))
    metadata = dict(response.metadata or {})
    metadata.pop("ds4.key")
    response.metadata = metadata
    _replace_snapshot(steps, snapshot)

    with fake_engine(script=scripted_rounds("second answer")) as engine:
        runtime = AgentRuntime(make_model(engine), project, store=store, clock=fixed_clock)
        try:
            with pytest.raises(StoreError, match="key"):
                asyncio.run(runtime.switch(saved["session_id"]))
        finally:
            asyncio.run(runtime.close())


@pytest.mark.parametrize("restore_path", ["kv", "render"])
def test_reordered_requests_are_rejected(tmp_path: Path, restore_path: Any) -> None:
    project, store, record, saved = saved_session(tmp_path)
    if restore_path == "render":
        zero_payload_kv(store, saved["session_id"])
    steps = _steps(store.session_dir(saved["session_id"]))
    snapshot = asyncio.run(steps.latest_snapshot(run_id=record.run_id))
    request_indexes = [index for index, message in enumerate(snapshot.messages) if isinstance(message, ModelRequest)]
    assert len(request_indexes) >= 2
    first, second = request_indexes[:2]
    snapshot.messages[first], snapshot.messages[second] = (
        snapshot.messages[second],
        snapshot.messages[first],
    )
    _replace_snapshot(steps, snapshot)

    with fake_engine(script=scripted_rounds("second answer")) as engine:
        runtime = AgentRuntime(make_model(engine), project, store=store, clock=fixed_clock)
        try:
            with pytest.raises(StoreError):
                asyncio.run(runtime.switch(saved["session_id"]))
        finally:
            asyncio.run(runtime.close())


@pytest.mark.parametrize("restore_path", ["kv", "render"])
def test_wrong_content_hash_fails_with_the_key(tmp_path: Path, restore_path: Any) -> None:
    project, store, record, saved = saved_session(tmp_path)
    if restore_path == "render":
        zero_payload_kv(store, saved["session_id"])
    keyed = [index for index, segment in enumerate(record.segments) if segment.key]
    changed = list(record.segments)
    target = changed[keyed[-1]]
    changed[keyed[-1]] = replace(target, content_hash="f" * 64)
    broken = replace(record, segments=tuple(changed))
    store.write_record_atomic(store.session_dir(saved["session_id"]) / "save.json", broken)

    with fake_engine(script=scripted_rounds("second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            with pytest.raises(StoreError, match=target.key):
                asyncio.run(runtime.switch(saved["session_id"]))
        finally:
            asyncio.run(runtime.close())


def test_model_or_quant_mismatch_is_an_error_naming_the_identity(tmp_path: Path) -> None:
    project, store, _record, saved = saved_session(tmp_path)

    with fake_engine(script=scripted_rounds("second answer"), model_id=9) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            with pytest.raises(StoreError, match="model id"):
                asyncio.run(runtime.switch(saved["session_id"]))
        finally:
            asyncio.run(runtime.close())

    with fake_engine(script=scripted_rounds("second answer"), quant_bits=5) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            with pytest.raises(StoreError, match="quant bits"):
                asyncio.run(runtime.switch(saved["session_id"]))
        finally:
            asyncio.run(runtime.close())


def test_corrupt_kv_fails_with_the_path(tmp_path: Path) -> None:
    project, store, _record, saved = saved_session(tmp_path)
    kv_path = store.session_dir(saved["session_id"]) / "session.kv"
    kv_path.write_bytes(b"not-a-kv-file" * 8)

    with fake_engine(script=scripted_rounds("second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            with pytest.raises(StoreError, match=r"session\.kv"):
                asyncio.run(runtime.switch(saved["session_id"]))
        finally:
            asyncio.run(runtime.close())


def test_save_writes_kv_then_complete_snapshot_then_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    prompt = "  read\n\n\thello   world\tand " + "x" * 100

    with fake_engine(script=scripted_rounds(READ_CALL, "first answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.run(prompt))
            session_id = runtime.session_id
            failed = StoreError("save record refused")
            original = store.write_record_atomic

            def refuse(path: Any, record: Any) -> Never:
                raise failed

            monkeypatch.setattr(store, "write_record_atomic", refuse)
            with pytest.raises(StoreError, match="save record refused"):
                asyncio.run(runtime.save())

            session_dir = store.session_dir(session_id)
            assert (session_dir / "session.kv").is_file()
            assert not (session_dir / "save.json").exists()
            run_id = f"{session_id}.save-1"
            assert (session_dir / "steps" / run_id).is_dir()

            monkeypatch.setattr(store, "write_record_atomic", original)
            record = asyncio.run(runtime.save())
            saved_tokens = list(model.session.transcript.tokens)
        finally:
            asyncio.run(runtime.close())

    assert record.run_id == f"{session_id}.save-2"
    assert record.session_id == session_id
    assert record.conversation_id == session_id
    assert record.title == " ".join(prompt.split())[:80]
    assert record.tokens == len(saved_tokens)
    assert record.tokens_sha256 == token_digest(saved_tokens)
    assert record.model_id == model.session.engine.info.model_id
    assert record.quant_bits == model.session.engine.info.routed_quant_bits
    assert record.created_at and record.last_used
    assert record.segments

    snapshot = asyncio.run(_steps(store.session_dir(session_id)).latest_snapshot(run_id=record.run_id))
    assert snapshot is not None
    assert snapshot.state == "complete"
    assert snapshot.step_index == record.step_index
    assert any(segment.key for segment in record.segments)
    assert all(segment.key for segment in record.segments if segment.kind in ("user", "assistant", "tool"))

    header = (store.session_dir(session_id) / "session.kv").read_bytes()[: KV_HEADER.size]
    magic, version, payload_bytes = KV_HEADER.unpack(header)
    assert (magic, version, payload_bytes) == (KV_MAGIC, KV_VERSION, record.payload_bytes)


def _steps(directory: Any) -> Any:
    from pydantic_ai_harness.step_persistence import FileStepStore

    return FileStepStore(Path(directory) / "steps")


def _replace_snapshot(steps: Any, snapshot: Any) -> None:
    """Append a tampered step and remove the original so switch selects it."""
    directory = steps._run_dir(snapshot.run_id) / "snapshots"
    original = set(directory.glob("*.json"))
    asyncio.run(steps.save_snapshot(snapshot))
    created = set(directory.glob("*.json")) - original
    assert len(created) == 1
    for path in original:
        path.unlink()


class Sentry:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_new_clears_live_history_and_keeps_old_files(tmp_path: Path) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds(READ_CALL, "first answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.run("read hello"))
            record = asyncio.run(runtime.save())
            old_id = runtime.session_id
            old_dir = store.session_dir(old_id)
            holder = cast("Any", model.session.state("step_store", lambda: None))
            sentry = model.session.state("sentry", Sentry)

            new_id = asyncio.run(runtime.new())

            assert new_id != old_id
            assert runtime.session_id == new_id
            assert runtime.messages == []
            assert len(model.session.transcript) == 0
            assert not model.session.transcript.chat
            assert sentry.closed and holder.closed
            assert (old_dir / "save.json").is_file()
            assert (old_dir / "session.kv").is_file()
            assert (old_dir / "steps" / record.run_id).is_dir()
        finally:
            asyncio.run(runtime.close())


def test_fork_keeps_live_state_and_writes_the_new_directory(tmp_path: Path) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds(READ_CALL, "first answer", "second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        try:
            asyncio.run(runtime.run("read hello"))
            messages = list(runtime.messages)
            tokens = list(model.session.transcript.tokens)
            segments = model.session.transcript.segments
            old_id = runtime.session_id
            old_dir = store.session_dir(old_id)
            holder = cast("Any", model.session.state("step_store", lambda: None))

            forked = asyncio.run(runtime.fork())

            assert forked != old_id
            assert runtime.session_id == forked
            assert dump(runtime.messages) == dump(messages)
            assert list(model.session.transcript.tokens) == tokens
            assert model.session.transcript.segments == segments
            assert model.session.state("step_store", lambda: None) is holder

            requests = capture(runtime, {"Synced"})
            asyncio.run(runtime.run("second answer"))
            synced = engine_events(requests, "Synced")
            assert synced[0].plan.kind == "extend"
            # Every evaluated token of the forked transcript is reused.
            assert synced[0].plan.reused == len(tokens) - 1

            record = asyncio.run(runtime.save())
            new_dir = store.session_dir(forked)
            assert record.session_id == forked
            assert (new_dir / "save.json").is_file()
            assert (new_dir / "session.kv").is_file()
            assert (new_dir / "steps" / record.run_id).is_dir()
            assert not (old_dir / "save.json").exists()
            assert not (old_dir / "session.kv").exists()
        finally:
            asyncio.run(runtime.close())


def test_close_closes_step_stores_and_scoped_state(tmp_path: Path) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds(READ_CALL, "first answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project, store=store, clock=fixed_clock)
        asyncio.run(runtime.run("read hello"))
        holder = cast("Any", model.session.state("step_store", lambda: None))
        sentry = model.session.state("sentry", Sentry)

        asyncio.run(runtime.close())

        assert sentry.closed
        assert holder.closed
        assert runtime.status == "closed"
        asyncio.run(runtime.close())  # idempotent
