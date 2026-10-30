"""Behavior of the stdlib save record and KV store primitives."""

from pathlib import Path
from typing import Any

import pytest

from ds4py._native.protocol import EngineInfo
from ds4py.errors import StoreError
from ds4py.store import (
    KV_HEADER,
    KV_MAGIC,
    KV_VERSION,
    SaveRecord,
    SegmentProvenance,
    SessionStore,
    token_digest,
)

DIGEST_123 = "7b0b5ea3ff36958c8e32ccf24b71da9ac68e51d0881bf75e62b837ec9ea6f3a5"


def record(**overrides: Any) -> SaveRecord:
    fields: dict[str, Any] = dict(  # noqa: C408 - Keyword-built fixture schema is clearer here.
        session_id="session-1",
        conversation_id="conversation-1",
        run_id="session-1.save-2",
        step_index=7,
        model_id=41,
        quant_bits=4,
        ctx=65536,
        title="Fix the failing test",
        created_at="2026-10-21T10:00:00+00:00",
        last_used="2026-10-21T10:05:00+00:00",
        payload_bytes=11,
        tokens=3,
        tokens_sha256=DIGEST_123,
        segments=(
            SegmentProvenance(0, 2, "system", "", "system-hash", "rendered"),
            SegmentProvenance(2, 3, "user", "user-1", "user-hash", "rendered"),
        ),
    )
    fields.update(overrides)
    return SaveRecord(**fields)


def engine_info(**overrides: Any) -> EngineInfo:
    fields: dict[str, Any] = dict(  # noqa: C408 - Keyword-built fixture schema is clearer here.
        name="fake-model",
        model_id=41,
        family="deepseek41",
        is_deepseek41=True,
        is_glm_dsa=False,
        is_qwen4=False,
        has_vision=False,
        routed_quant_bits=4,
        mtp_draft_tokens=1,
        vocab_size=1024,
    )
    fields.update(overrides)
    return EngineInfo(**fields)


def test_kv_payload_round_trips_through_the_declared_header(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    path = store.session_dir("session-1") / "session.kv"
    payload = b"payload\x00bytes"

    store.write_kv_atomic(path, payload)

    magic, version, declared = KV_HEADER.unpack(path.read_bytes()[: KV_HEADER.size])
    assert (magic, version, declared) == (KV_MAGIC, KV_VERSION, len(payload))
    assert path.stat().st_size == KV_HEADER.size + len(payload)
    assert store.read_kv(path) == payload


def test_zero_length_payload_round_trips_without_looking_missing(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    path = store.session_dir("empty") / "session.kv"

    store.write_kv_atomic(path, b"")

    assert store.read_kv(path) == b""
    magic, version, declared = KV_HEADER.unpack(path.read_bytes())
    assert (magic, version, declared) == (KV_MAGIC, KV_VERSION, 0)


def test_save_record_round_trip_preserves_every_field(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    path = store.session_dir("session-1") / "save.json"
    original = record()

    store.write_record_atomic(path, original)

    assert store.read_record(path) == original


def test_save_record_from_dict_preserves_all_segment_provenance() -> None:
    original = record()

    restored = SaveRecord.from_dict(original.to_dict())

    assert restored == original
    assert restored.segments == (
        SegmentProvenance(0, 2, "system", "", "system-hash", "rendered"),
        SegmentProvenance(2, 3, "user", "user-1", "user-hash", "rendered"),
    )


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ([1, 2, 3], DIGEST_123),
        ([], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (
            [1, 0, 4294967295],
            "4e13f2779dfa7ecf14d05a2d34ede84e12fb9f25c3be9c5e9928dd3e5315b617",
        ),
    ],
)
def test_token_digest_is_sha256_over_fixed_big_endian_uint32(tokens: Any, expected: Any) -> None:
    assert token_digest(tokens) == expected


def test_default_root_reads_ds4py_home_at_call_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("DS4PY_HOME", str(first))
    store = SessionStore()

    assert store.root == first
    assert store.session_dir("abc") == first / "sessions" / "abc"

    monkeypatch.setenv("DS4PY_HOME", str(second))
    assert store.root == second
    assert not second.exists()


def test_default_root_expands_a_home_redirect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DS4PY_HOME", "~/state")

    store = SessionStore()

    assert store.root == tmp_path / "home" / "state"
    assert not store.root.exists()


@pytest.mark.parametrize("value", [None, ""])
def test_default_root_falls_back_to_dot_ds4py(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: Any) -> None:
    if value is None:
        monkeypatch.delenv("DS4PY_HOME", raising=False)
    else:
        monkeypatch.setenv("DS4PY_HOME", value)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert SessionStore().root == tmp_path / "home" / ".ds4py"


def test_find_resolves_a_unique_session_id_prefix(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    for session_id in ("aaaa1111", "aaaa2222", "bbbb3333"):
        store.write_record_atomic(
            store.session_dir(session_id) / "save.json",
            record(session_id=session_id, conversation_id=session_id),
        )

    found, path = store.find("bbbb")

    assert found.session_id == "bbbb3333"
    assert path == store.session_dir("bbbb3333") / "save.json"


def test_find_rejects_a_prefix_matching_an_exact_id_and_a_longer_id(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    for session_id in ("abc", "abcdef"):
        store.write_record_atomic(
            store.session_dir(session_id) / "save.json",
            record(session_id=session_id, conversation_id=session_id),
        )

    with pytest.raises(StoreError) as caught:
        store.find("abc")

    assert "ambiguous" in str(caught.value)
    assert "abc" in str(caught.value)
    assert "abcdef" in str(caught.value)


def test_validate_for_engine_accepts_the_matching_model_and_quant(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    store.validate_for_engine(record(), engine_info())
