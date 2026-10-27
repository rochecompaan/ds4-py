"""Path-specific failures from corrupt save records and interrupted writes."""

import json
from pathlib import Path

import pytest

import ds4py.store as store_module
from ds4py._native.protocol import EngineInfo
from ds4py.errors import StoreError
from ds4py.store import (
    KV_HEADER,
    KV_MAGIC,
    KV_VERSION,
    SaveRecord,
    SegmentProvenance,
    SessionStore,
)

DIGEST_123 = "7b0b5ea3ff36958c8e32ccf24b71da9ac68e51d0881bf75e62b837ec9ea6f3a5"
DIGEST_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def record(**overrides):
    fields = dict(
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


def engine_info(**overrides):
    fields = dict(
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


def write_raw_record(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    ("contents", "reason"),
    [
        (b"NOTMAGIC" + KV_HEADER.pack(KV_MAGIC, KV_VERSION, 3)[8:] + b"abc", "magic"),
        (
            KV_HEADER.pack(KV_MAGIC, KV_VERSION + 1, 3) + b"abc",
            "version",
        ),
        (KV_HEADER.pack(KV_MAGIC, KV_VERSION, 3)[:10], "truncated header"),
        (KV_HEADER.pack(KV_MAGIC, KV_VERSION, 4) + b"abc", "truncated payload"),
        (KV_HEADER.pack(KV_MAGIC, KV_VERSION, 3) + b"abcd", "extra"),
    ],
)
def test_corrupt_kv_files_fail_with_the_path_and_reason(tmp_path, contents, reason):
    store = SessionStore(tmp_path)
    path = store.session_dir("session-1") / "session.kv"
    path.parent.mkdir(parents=True)
    path.write_bytes(contents)

    with pytest.raises(StoreError) as caught:
        store.read_kv(path)

    assert str(path) in str(caught.value)
    assert reason in str(caught.value)


@pytest.mark.parametrize(
    "contents",
    [b"{not json", b"\xff\xfe", b"[]", b"null"],
)
def test_malformed_record_json_fails_with_the_path(tmp_path, contents):
    store = SessionStore(tmp_path)
    path = store.session_dir("session-1") / "save.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(contents)

    with pytest.raises(StoreError) as caught:
        store.read_record(path)

    assert str(path) in str(caught.value)
    assert "record" in str(caught.value)


def test_deeply_nested_record_json_fails_as_a_malformed_record(tmp_path):
    store = SessionStore(tmp_path)
    path = store.session_dir("session-1") / "save.json"
    path.parent.mkdir(parents=True)
    path.write_text("[" * 100_000 + "0" + "]" * 100_000, encoding="utf-8")

    with pytest.raises(StoreError) as caught:
        store.read_record(path)

    assert str(path) in str(caught.value)
    assert "record" in str(caught.value)


def test_huge_token_count_integer_fails_as_a_malformed_record(tmp_path):
    store = SessionStore(tmp_path)
    path = store.session_dir("session-1") / "save.json"
    data = record().to_dict()
    data["tokens"] = "PLACEHOLDER"
    raw = json.dumps(data).replace('"PLACEHOLDER"', "9" * 5000)
    path.parent.mkdir(parents=True)
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(StoreError) as caught:
        store.read_record(path)

    assert str(path) in str(caught.value)
    assert "record" in str(caught.value)


def test_kv_header_declaring_maximum_payload_fails_as_truncated(tmp_path):
    store = SessionStore(tmp_path)
    path = store.session_dir("session-1") / "session.kv"
    path.parent.mkdir(parents=True)
    path.write_bytes(KV_HEADER.pack(KV_MAGIC, KV_VERSION, 2**64 - 1))

    with pytest.raises(StoreError) as caught:
        store.read_kv(path)

    assert str(path) in str(caught.value)
    assert "truncated payload" in str(caught.value)


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("version", 99, "version"),
        ("tokens", -1, "tokens"),
        ("tokens", True, "tokens"),
        ("tokens", 1.5, "tokens"),
        ("tokens_sha256", "not-a-digest", "tokens_sha256"),
        ("tokens_sha256", "", "tokens_sha256"),
        ("payload_bytes", -1, "payload_bytes"),
        ("payload_bytes", False, "payload_bytes"),
    ],
)
def test_invalid_record_counts_and_digests_fail_with_the_path(
    tmp_path, key, value, reason,
):
    store = SessionStore(tmp_path)
    path = store.session_dir("session-1") / "save.json"
    data = record().to_dict()
    data[key] = value
    write_raw_record(path, data)

    with pytest.raises(StoreError) as caught:
        store.read_record(path)

    assert str(path) in str(caught.value)
    assert reason in str(caught.value)


@pytest.mark.parametrize(
    ("tokens", "tokens_sha256", "segments", "reason"),
    [
        (3, DIGEST_123, [(0, 2, "system", "", "h", "rendered")], "gap"),
        (3, DIGEST_123, [(1, 3, "system", "", "h", "rendered")], "start"),
        (3, DIGEST_123, [(0, 4, "system", "", "h", "rendered")], "beyond"),
        (
            3,
            DIGEST_123,
            [(0, 2, "system", "", "h", "rendered"),
             (1, 3, "user", "k", "h", "rendered")],
            "overlap",
        ),
        (3, DIGEST_123, [], "segments"),
        (0, DIGEST_EMPTY, [(0, 1, "system", "", "h", "rendered")], "segments"),
    ],
)
def test_segments_must_cover_the_token_range(
    tmp_path, tokens, tokens_sha256, segments, reason,
):
    store = SessionStore(tmp_path)
    path = store.session_dir("session-1") / "save.json"
    data = record().to_dict()
    data["tokens"] = tokens
    data["tokens_sha256"] = tokens_sha256
    data["segments"] = [
        dict(zip(
            ("start", "end", "kind", "key", "content_hash", "source"), segment,
        ))
        for segment in segments
    ]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(StoreError) as caught:
        store.read_record(path)

    assert str(path) in str(caught.value)
    assert "segments" in str(caught.value)
    assert reason in str(caught.value)


def test_find_rejects_an_unknown_prefix(tmp_path):
    store = SessionStore(tmp_path)

    with pytest.raises(StoreError, match="unknown"):
        store.find("missing")


def test_find_rejects_an_ambiguous_prefix(tmp_path):
    store = SessionStore(tmp_path)
    for session_id in ("abc1111", "abc2222"):
        store.write_record_atomic(
            store.session_dir(session_id) / "save.json",
            record(session_id=session_id, conversation_id=session_id),
        )

    with pytest.raises(StoreError) as caught:
        store.find("abc")

    assert "ambiguous" in str(caught.value)
    assert "abc1111" in str(caught.value)
    assert "abc2222" in str(caught.value)


def test_find_ignores_sessions_without_a_save_record(tmp_path):
    store = SessionStore(tmp_path)
    store.session_dir("untouched").mkdir(parents=True)

    with pytest.raises(StoreError, match="unknown"):
        store.find("untouched")


def test_model_mismatch_is_an_error_naming_the_session(tmp_path):
    store = SessionStore(tmp_path)

    with pytest.raises(StoreError) as caught:
        store.validate_for_engine(record(), engine_info(model_id=42))

    assert "session-1" in str(caught.value)
    assert "model" in str(caught.value)


def test_quant_mismatch_is_an_error_naming_the_session(tmp_path):
    store = SessionStore(tmp_path)

    with pytest.raises(StoreError) as caught:
        store.validate_for_engine(record(), engine_info(routed_quant_bits=8))

    assert "session-1" in str(caught.value)
    assert "quant" in str(caught.value)


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_interrupted_kv_write_leaves_the_prior_file_intact(
    tmp_path, monkeypatch, failure,
):
    store = SessionStore(tmp_path)
    path = store.session_dir("session-1") / "session.kv"
    store.write_kv_atomic(path, b"old payload")

    def fail(*args, **kwargs):
        raise OSError("interrupted")

    monkeypatch.setattr(store_module.os, failure, fail)

    with pytest.raises(StoreError) as caught:
        store.write_kv_atomic(path, b"new payload")

    assert str(path) in str(caught.value)
    monkeypatch.undo()
    assert store.read_kv(path) == b"old payload"
    assert [entry.name for entry in path.parent.iterdir()] == ["session.kv"]


def test_interrupted_record_write_leaves_the_prior_record_intact(
    tmp_path, monkeypatch,
):
    store = SessionStore(tmp_path)
    path = store.session_dir("session-1") / "save.json"
    old = record()
    store.write_record_atomic(path, old)
    new = record(run_id="session-1.save-3", title="A newer save")

    def fail(*args, **kwargs):
        raise OSError("interrupted")

    monkeypatch.setattr(store_module.os, "replace", fail)
    with pytest.raises(StoreError) as caught:
        store.write_record_atomic(path, new)

    assert str(path) in str(caught.value)
    monkeypatch.undo()
    assert store.read_record(path) == old
    assert [entry.name for entry in path.parent.iterdir()] == ["save.json"]


def test_failed_record_replacement_keeps_the_old_record_for_mismatch_fallback(
    tmp_path, monkeypatch,
):
    store = SessionStore(tmp_path)
    kv_path = store.session_dir("session-1") / "session.kv"
    record_path = store.session_dir("session-1") / "save.json"
    old = record(payload_bytes=5)
    store.write_kv_atomic(kv_path, b"old!!")
    store.write_record_atomic(record_path, old)
    new_payload = b"a much newer payload"
    store.write_kv_atomic(kv_path, new_payload)
    new = record(payload_bytes=len(new_payload), run_id="session-1.save-3")

    def fail(*args, **kwargs):
        raise OSError("snapshot write failed after KV replacement")

    monkeypatch.setattr(store_module.os, "replace", fail)
    with pytest.raises(StoreError):
        store.write_record_atomic(record_path, new)

    monkeypatch.undo()
    assert store.read_kv(kv_path) == new_payload
    fallback = store.read_record(record_path)
    assert fallback == old
    assert fallback.run_id == "session-1.save-2"
