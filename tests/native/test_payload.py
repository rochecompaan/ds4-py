"""Native payload round trips, corruption handling, and Python-owned files."""

import ctypes
from pathlib import Path

import pytest

from ds4py._native import payload
from ds4py._native.fake import FakeEngine, FakeNative
from ds4py._native.protocol import EngineConfig
from ds4py.errors import EngineError

HEADER = b"DS4PYKV\0" + bytes(12)


def engine(**kwargs):
    return FakeNative(**kwargs).open_engine(EngineConfig(Path("fake.gguf")))


def staged(engine, tokens):
    session = engine.create_session(64)
    session.sync(tokens)
    payload = session.stage_payload()
    assert payload.bytes == session.payload_bytes()
    return session, payload


def write_staged(path, payload):
    with path.open("w+b") as file:
        file.write(HEADER)
        payload.write(file)
        assert file.tell() == len(HEADER) + payload.bytes
    payload.close()


def test_staged_payload_round_trips_tokens_and_position_without_tokenizing(
    tmp_path, monkeypatch,
):
    first = engine()
    source, payload = staged(first, [5, 6, 7, 8])
    path = tmp_path / "session.kv"
    write_staged(path, payload)

    def fail_tokenize(*args, **kwargs):
        raise AssertionError("payload load must not tokenize")

    monkeypatch.setattr(FakeEngine, "tokenize", fail_tokenize)
    target = engine().create_session(64)
    with path.open("rb") as file:
        assert file.read(len(HEADER)) == HEADER
        target.load_payload(file, payload.bytes)
        assert file.tell() == len(HEADER) + payload.bytes
    assert target.tokens() == [5, 6, 7, 8]
    assert target.pos() == 4


def test_zero_length_payload_is_distinct_from_missing_or_truncated(tmp_path):
    session = engine().create_session(64)
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    with empty.open("rb") as file:
        session.load_payload(file, 0)
    assert session.tokens() == [] and session.pos() == 0

    source, payload = staged(engine(), [5, 6])
    path = tmp_path / "session.kv"
    write_staged(path, payload)
    with path.open("rb") as file:
        file.seek(len(HEADER))
        with pytest.raises(EngineError, match="truncated|corrupt|payload"):
            session.load_payload(file, payload.bytes + 1)
    assert session.tokens() == [] and session.pos() == 0


@pytest.mark.parametrize(
    "damage", ["magic", "version", "count", "checksum", "padding", "truncate"],
)
def test_corrupt_payloads_raise_and_do_not_replace_existing_tokens(damage, tmp_path):
    _, payload = staged(engine(), [5, 6])
    path = tmp_path / f"{damage}.kv"
    write_staged(path, payload)
    data = bytearray(path.read_bytes()[len(HEADER):])
    if damage == "magic":
        data[0] ^= 0xFF
    elif damage == "version":
        data[4] ^= 0xFF
    elif damage == "count":
        data[8] ^= 0xFF
    elif damage == "checksum":
        data[12] ^= 0xFF
    elif damage == "padding":
        data[-1] = 1
    else:
        del data[-1]
    path.write_bytes(HEADER + data)

    target = engine().create_session(64)
    target.sync([9, 9])
    with path.open("rb") as file:
        file.seek(len(HEADER))
        with pytest.raises(EngineError, match="truncated|corrupt|payload"):
            target.load_payload(file, payload.bytes)
    assert target.tokens() == [9, 9] and target.pos() == 2


def test_staged_payload_closes_once_and_refuses_later_writes(tmp_path):
    session, payload = staged(engine(), [5, 6])
    path = tmp_path / "session.kv"
    with path.open("w+b") as file:
        payload.close()
        payload.close()
        with pytest.raises(EngineError, match="closed"):
            payload.write(file)
    assert session.payload_bytes() == payload.bytes


def test_native_stream_rewinds_a_read_ahead_buffer_before_reading(tmp_path):
    path = tmp_path / "stream.bin"
    path.write_bytes(b"headerpayloadtail")
    libc = ctypes.CDLL(None)
    fread = libc.fread
    fread.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
    fread.restype = ctypes.c_size_t

    with path.open("rb") as file:
        assert file.read(6) == b"header"    # buffers the whole file ahead of the caller
        with payload.native_stream(file, "rb") as stream:
            buffer = ctypes.create_string_buffer(7)
            assert fread(buffer, 1, 7, stream) == 7
            assert buffer.raw == b"payload"
        assert file.tell() == 13
        assert file.read() == b"tail"
        assert file.read() == b""


def test_native_stream_releases_only_the_duplicate_and_syncs_positions(tmp_path):
    path = tmp_path / "stream.bin"
    path.write_bytes(b"abcdef")
    libc = ctypes.CDLL(None)
    fread, fwrite = libc.fread, libc.fwrite
    fread.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
    fread.restype = ctypes.c_size_t
    fwrite.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
    fwrite.restype = ctypes.c_size_t

    with path.open("r+b") as file:
        assert file.read(2) == b"ab"
        with payload.native_stream(file, "rb") as stream:
            buffer = ctypes.create_string_buffer(3)
            assert fread(buffer, 1, 3, stream) == 3
            assert buffer.raw == b"cde"
        assert not file.closed
        assert file.read() == b"f"

        file.seek(0, 2)
        with payload.native_stream(file, "wb") as stream:
            assert fwrite(b"XY", 1, 2, stream) == 2
        assert not file.closed
        file.write(b"Z")
    assert path.read_bytes() == b"abcdefXYZ"


def test_native_stage_and_write_release_only_the_duplicate_descriptor(
    stub_engine, tmp_path,
):
    engine, c = stub_engine
    session = engine.create_session(64)
    session.sync([5, 6, 7])
    staged = session.stage_payload()
    assert staged.bytes == session.payload_bytes()

    path = tmp_path / "session.kv"
    with path.open("w+b") as file:
        file.write(HEADER)
        staged.write(file)
        assert file.tell() == len(HEADER) + staged.bytes
        staged.close()
        staged.close()
        with pytest.raises(EngineError, match="closed"):
            staged.write(file)
        file.write(b"trailer")
        assert not file.closed      # only the duplicated descriptor reached native
    assert c.stub_payload_writes() == 1
    assert c.stub_payload_frees() == 1
    data = path.read_bytes()
    assert data[: len(HEADER)] == HEADER and data.endswith(b"trailer")
    assert len(data) == len(HEADER) + staged.bytes + len(b"trailer")


def test_native_load_consumes_only_the_declared_bytes(stub_engine, tmp_path):
    engine, _ = stub_engine
    source = engine.create_session(64)
    source.sync([5, 6, 7, 8])
    staged = source.stage_payload()
    path = tmp_path / "session.kv"
    with path.open("w+b") as file:
        file.write(HEADER)
        staged.write(file)
        file.write(b"next")
    staged.close()

    target = engine.create_session(64)
    with path.open("rb") as file:
        assert file.read(len(HEADER)) == HEADER
        target.load_payload(file, staged.bytes)
        assert file.tell() == len(HEADER) + staged.bytes
        assert file.read() == b"next"       # the caller's file keeps its own position
        assert file.read() == b""           # and its read-ahead was discarded
    assert target.tokens() == [5, 6, 7, 8]


def test_native_stage_error_uses_the_buffer_and_frees_the_partial(stub_engine):
    engine, c = stub_engine
    session = engine.create_session(64)
    session.sync([1])
    c.stub_fail_payload(1, 0, 0)
    with pytest.raises(EngineError, match="stage rejected: checkpoint is invalid"):
        session.stage_payload()
    assert c.stub_payload_frees() == 1
    assert c.stub_payload_writes() == 0


def test_native_write_error_uses_the_buffer_and_keeps_the_file_usable(
    stub_engine, tmp_path,
):
    engine, c = stub_engine
    session = engine.create_session(64)
    session.sync([1, 2])
    staged = session.stage_payload()
    c.stub_fail_payload(0, 1, 0)
    path = tmp_path / "session.kv"
    with path.open("w+b") as file:
        file.write(HEADER)
        try:
            with pytest.raises(EngineError, match="write rejected: disk full"):
                staged.write(file)
        finally:
            staged.close()
        assert file.tell() == len(HEADER)
        file.write(b"tail")
        assert not file.closed
    assert path.read_bytes() == HEADER + b"tail"
    assert c.stub_payload_frees() == 1
    assert c.stub_payload_writes() == 0


def test_native_load_error_uses_the_buffer_and_leaves_state_unchanged(
    stub_engine, tmp_path,
):
    engine, c = stub_engine
    source = engine.create_session(64)
    source.sync([5, 6])
    staged = source.stage_payload()
    path = tmp_path / "session.kv"
    with path.open("w+b") as file:
        file.write(HEADER)
        staged.write(file)
        file.write(b"tail")
    staged.close()

    target = engine.create_session(64)
    target.sync([9])
    c.stub_fail_payload(0, 0, 1)
    with path.open("rb") as file:
        file.seek(len(HEADER))
        with pytest.raises(EngineError, match="load rejected: corrupt payload"):
            target.load_payload(file, staged.bytes)
        assert file.tell() == len(HEADER)
        remainder = file.read()
        assert len(remainder) == staged.bytes + len(b"tail")
        assert remainder.endswith(b"tail")
    assert target.tokens() == [9]
    assert c.stub_payload_loads() == 0
