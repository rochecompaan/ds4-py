import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import ds4py
from ds4py._native.fake import PAYLOAD_STRIDE, FakeNative, FakeSession, reply
from ds4py.engine.events import Loaded, Prefill, Saved, Token
from ds4py.engine.sampling import SamplingPolicy
from ds4py.engine.session import Engine, Session
from ds4py.engine.tool_prompt import ToolDefinition
from ds4py.engine.worker import SYSTEM, SavePayloadJob
from ds4py.errors import EngineError, StoreError
from ds4py.store import KV_HEADER, SessionStore


def fake_engine(
    *,
    ctx: int = 1024,
    model_id: int = 0,
    quant_bits: int = 0,
    payload_support: bool | None = None,
    restore_enabled: bool = True,
) -> Engine:
    info = replace(FakeNative()._info, model_id=model_id, routed_quant_bits=quant_bits)
    native = FakeNative(
        info=info,
        payload_support=payload_support,
        restore_enabled=restore_enabled,
        script=reply(b"same"),
    )
    return ds4py.open("f.gguf", ctx=ctx, native=native)


def capture(session: Session) -> list[Any]:
    events: list[Any] = []
    session.on_event(events.append)
    return events


def engine_events(events: list[Any], kind: str) -> list[Any]:
    return [event.payload["event"] for event in events if event.kind == kind]


def positive_prefills(events: list[Any]) -> list[Prefill]:
    return [event for event in engine_events(events, "Prefill") if event.total > 0]


def cache_files(home: Path) -> list[Path]:
    directory = home / "sysprompt"
    return sorted(directory.glob("*.kv")) if directory.is_dir() else []


def start_with_tools(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tools: tuple[ToolDefinition, ...] = (),
    instructions: str | None = None,
    previous_paths: tuple[Path, ...] = (),
    **engine_options: Any,
) -> tuple[tuple[int, ...], tuple[Any, ...], list[Any], Path]:
    monkeypatch.setenv("DS4PY_HOME", str(home))
    with fake_engine(**engine_options) as engine:
        session = engine.session()
        events = capture(session)
        session.start_chat(tools, instructions)
        prefix = session.transcript.tokens
        segments = session.transcript.segments
    files = cache_files(home)
    new_files = set(files) - set(previous_paths)
    assert len(new_files) == 1
    return prefix, segments, events, new_files.pop()


def test_second_startup_loads_same_prefix_without_prefill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    prefix, segments, cold_events, path = start_with_tools(home, monkeypatch)

    assert SessionStore().read_kv_header(path) == len(prefix) * PAYLOAD_STRIDE
    assert positive_prefills(cold_events)[0].cached_prefix == 0

    with fake_engine() as engine:
        session = engine.session()
        events = capture(session)
        session.start_chat()

        assert session.transcript.tokens == prefix
        assert session.transcript.segments == segments
        assert session.history.checkpoint_tokens == {SYSTEM: len(prefix)}
        assert session.plan().kind == "extend"
        assert session.plan().reused == len(prefix)
        assert session.plan().to_prefill == 0
        assert len(engine_events(events, "Loaded")) == 1
        assert isinstance(engine_events(events, "Loaded")[0], Loaded)
        assert positive_prefills(events) == []


def test_new_session_reuses_cache_and_changed_prefix_gets_new_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    prefix, _segments, _events, first_path = start_with_tools(home, monkeypatch)

    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        session.new()
        events = capture(session)
        session.start_chat()
        assert session.transcript.tokens == prefix
        assert len(engine_events(events, "Loaded")) == 1
        assert positive_prefills(events) == []

    changed_tool = ToolDefinition("different", "A different tool", {"type": "object"})
    changed_prefix, _segments, changed_events, second_path = start_with_tools(
        home,
        monkeypatch,
        tools=(changed_tool,),
        instructions="different instructions",
        previous_paths=(first_path,),
    )
    assert changed_prefix != prefix
    assert second_path != first_path
    assert len(cache_files(home)) == 2
    assert positive_prefills(changed_events)[0].cached_prefix == 0


def test_cache_key_covers_engine_identity_context_and_exact_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases: tuple[tuple[str, dict[str, Any], str | None], ...] = (
        ("baseline", {}, None),
        ("model", {"model_id": 9}, None),
        ("quantization", {"quant_bits": 4}, None),
        ("context", {"ctx": 2048}, None),
        ("tokens-ab", {}, "ab"),
        ("tokens-ba", {}, "ba"),
    )
    names = []
    for name, options, instructions in cases:
        home = tmp_path / name
        _prefix, _segments, _events, path = start_with_tools(
            home,
            monkeypatch,
            instructions=instructions,
            **options,
        )
        assert path.parent == home / "sysprompt"
        names.append(path.name)

    assert len(set(names)) == len(cases)


@pytest.mark.parametrize(
    "damage",
    ["unreadable", "bad_magic", "bad_version", "short_header", "short_payload", "extra_data", "corrupt_payload"],
)
def test_invalid_caches_are_rebuilt_and_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    home = tmp_path / "home"
    prefix, segments, _events, path = start_with_tools(home, monkeypatch)
    valid = path.read_bytes()

    if damage == "unreadable":
        path.chmod(0)
    elif damage == "bad_magic":
        path.write_bytes(b"NOTDS4KV" + valid[8:])
    elif damage == "bad_version":
        path.write_bytes(valid[:8] + (99).to_bytes(4, "big") + valid[12:])
    elif damage == "short_header":
        path.write_bytes(valid[: KV_HEADER.size - 1])
    elif damage == "short_payload":
        path.write_bytes(valid[:-1])
    elif damage == "extra_data":
        path.write_bytes(valid + b"extra")
    else:
        damaged = bytearray(valid)
        damaged[KV_HEADER.size] ^= 0xFF
        path.write_bytes(damaged)

    with fake_engine() as engine:
        session = engine.session()
        events = capture(session)
        session.start_chat()
        assert session.transcript.tokens == prefix
        assert session.transcript.segments == segments
        prefills = positive_prefills(events)
        assert len(prefills) == 1
        assert prefills[0].cached_prefix == 0
        assert prefills[0].total == len(prefix)

    assert path.read_bytes() == valid
    assert SessionStore().read_kv_header(path) == len(prefix) * PAYLOAD_STRIDE

    with fake_engine() as engine:
        session = engine.session()
        events = capture(session)
        session.start_chat()
        assert positive_prefills(events) == []
        assert len(engine_events(events, "Loaded")) == 1


def test_mismatched_cache_is_rebuilt_and_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _a_prefix, _a_segments, _a_events, a_path = start_with_tools(tmp_path / "a", monkeypatch)
    b_prefix, b_segments, _b_events, b_path = start_with_tools(
        tmp_path / "b",
        monkeypatch,
        instructions="prefix B",
    )
    valid_b = b_path.read_bytes()
    b_path.write_bytes(a_path.read_bytes())

    with fake_engine() as engine:
        session = engine.session()
        events = capture(session)
        session.start_chat(instructions="prefix B")
        assert session.transcript.tokens == b_prefix
        assert session.transcript.segments == b_segments
        synced = engine_events(events, "Synced")
        assert synced[0].plan.kind == "rebuild"
        assert synced[0].plan.reused == 0
        assert synced[0].prefilled == len(b_prefix)

    assert b_path.read_bytes() == valid_b
    with fake_engine() as engine:
        session = engine.session()
        events = capture(session)
        session.start_chat(instructions="prefix B")
        assert positive_prefills(events) == []
        assert len(engine_events(events, "Loaded")) == 1


def test_unsupported_file_payloads_prefill_without_touching_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    prefix, segments, _events, path = start_with_tools(home, monkeypatch)
    cached = path.read_bytes()

    def forbid(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unsupported native file payloads must not be used")

    monkeypatch.setattr(FakeSession, "load_payload", forbid)
    monkeypatch.setattr(FakeSession, "stage_payload", forbid)
    with fake_engine(payload_support=False) as engine:
        session = engine.session()
        events = capture(session)
        session.start_chat()
        assert session.transcript.tokens == prefix
        assert session.transcript.segments == segments
        assert positive_prefills(events)[0].cached_prefix == 0

    assert path.read_bytes() == cached
    assert cache_files(home) == [path]

    fresh = tmp_path / "fresh"
    monkeypatch.setenv("DS4PY_HOME", str(fresh))
    with fake_engine(payload_support=False) as engine:
        engine.session().start_chat()
    assert not (fresh / "sysprompt").exists()


def test_restore_disabled_prefills_without_touching_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    prefix, segments, _events, path = start_with_tools(home, monkeypatch)
    cached = path.read_bytes()

    def forbid(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("disabled native restore must not load or stage cache payloads")

    monkeypatch.setattr(FakeSession, "load_payload", forbid)
    monkeypatch.setattr(FakeSession, "stage_payload", forbid)
    with fake_engine(restore_enabled=False) as engine:
        session = engine.session()
        events = capture(session)
        session.start_chat()
        assert session.transcript.tokens == prefix
        assert session.transcript.segments == segments
        assert positive_prefills(events)[0].cached_prefix == 0

    assert path.read_bytes() == cached
    assert cache_files(home) == [path]


def test_failed_native_cache_load_resets_before_prefill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    prefix, _segments, _events, _path = start_with_tools(home, monkeypatch)
    original_load = FakeSession.load_payload

    def fail_after_loading(self: Any, file: Any, payload_bytes: Any) -> None:
        original_load(self, file, payload_bytes)
        raise EngineError("native load failed after changing state")

    monkeypatch.setattr(FakeSession, "load_payload", fail_after_loading)
    with fake_engine() as engine:
        session = engine.session()
        events = capture(session)
        session.start_chat()
        synced = engine_events(events, "Synced")
        assert synced[0].plan.kind == "rebuild"
        assert synced[0].plan.reused == 0
        assert synced[0].prefilled == len(prefix)


def test_warm_cache_preserves_interrupted_chat_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    _prefix, _segments, _events, path = start_with_tools(home, monkeypatch)
    cached = path.read_bytes()

    with fake_engine() as engine:
        session = engine.session()
        session.cancel()
        with pytest.raises(EngineError, match=r"^chat start was interrupted$"):
            session.start_chat()
        assert session.transcript.tokens == ()

    assert path.read_bytes() == cached


def test_cache_save_rejects_wrong_token_count(monkeypatch: pytest.MonkeyPatch) -> None:
    original_events = Session._job_events

    def wrong_saved_count(self: Session, job: Any, **kwargs: Any) -> Any:
        for event in original_events(self, job, **kwargs):
            if isinstance(job, SavePayloadJob) and isinstance(event, Saved):
                yield replace(event, tokens=event.tokens - 1)
            else:
                yield event

    monkeypatch.setattr(Session, "_job_events", wrong_saved_count)
    with fake_engine() as engine:
        session = engine.session()
        with pytest.raises(EngineError, match="token count"):
            session.start_chat()


def test_cache_write_failure_does_not_abort_chat_start(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise StoreError("unwritable cache")

    monkeypatch.setattr(SessionStore, "write_kv_stream_atomic", fail)
    with fake_engine() as engine:
        session = engine.session()
        session.start_chat()
        assert session.transcript.tokens
        assert session.history.checkpoint_tokens == {SYSTEM: len(session.transcript)}
        assert session.plan().kind == "extend"


def test_interrupted_system_prefill_keeps_existing_error_contract(tmp_path: Path) -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.cancel()
        with pytest.raises(EngineError, match=r"^chat start was interrupted$"):
            session.start_chat()
        assert session.transcript.tokens == ()

    assert cache_files(tmp_path / "ds4py-home") == []


def test_concurrent_cold_starts_publish_one_complete_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("DS4PY_HOME", str(home))
    barrier = threading.Barrier(2)

    def start() -> tuple[tuple[int, ...], list[Any]]:
        barrier.wait()
        with fake_engine() as engine:
            session = engine.session()
            events = capture(session)
            session.start_chat()
            return session.transcript.tokens, events

    with ThreadPoolExecutor(max_workers=2) as pool:
        cold = list(pool.map(lambda _: start(), range(2)))

    assert cold[0][0] == cold[1][0]
    assert all(positive_prefills(events)[0].cached_prefix == 0 for _prefix, events in cold)
    files = cache_files(home)
    assert len(files) == 1
    assert SessionStore().read_kv_header(files[0]) == len(cold[0][0]) * PAYLOAD_STRIDE
    assert list(files[0].parent.glob("*.tmp")) == []

    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        warm = list(pool.map(lambda _: start(), range(2)))
    assert all(prefix == cold[0][0] for prefix, _events in warm)
    assert all(positive_prefills(events) == [] for _prefix, events in warm)


def test_greedy_tokens_match_without_cache_and_with_cold_and_warm_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(home: Path, *, supported: bool | None) -> tuple[list[int], tuple[int, ...], list[Prefill]]:
        monkeypatch.setenv("DS4PY_HOME", str(home))
        with fake_engine(payload_support=supported) as engine:
            session = engine.session()
            events = capture(session)
            session.append_user("hello")
            system_length = max(segment.end for segment in session.transcript.segments if segment.kind == "system")
            generated = list(session.run_round(sampling=SamplingPolicy(temperature=0, seed=1), max_tokens=10))
            token_ids = [event.id for event in generated if isinstance(event, Token)]
            system_prefills = [event for event in positive_prefills(events) if event.cached_prefix < system_length]
            return token_ids, session.transcript.tokens, system_prefills

    without_cache = run(tmp_path / "without", supported=False)
    cold = run(tmp_path / "with", supported=True)
    warm = run(tmp_path / "with", supported=True)

    assert cold[:2] == without_cache[:2]
    assert warm[:2] == without_cache[:2]
    assert cold[2]
    assert warm[2] == []
