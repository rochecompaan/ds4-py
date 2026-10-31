"""Session save/load jobs over the single native worker."""
from typing import Any
from pathlib import Path

import pytest

import ds4py
from ds4py.engine.session import Engine
from ds4py._native.fake import FakeNative, FakeSession, reply
from ds4py.engine.events import Loaded, Saved, Stopped, Synced
from ds4py.engine.plan import Plan
from ds4py.errors import CheckpointError, EngineError

HEADER = b"DS4PYKV\0" + bytes(12)


def fake_engine(**kwargs: Any) -> Engine:
    return ds4py.open("f.gguf", ctx=4096, native=FakeNative(**kwargs))


def chat_session(engine: Any) -> Any:
    session = engine.session()
    session.append_user("hi")
    list(session.run_round(max_tokens=1))
    assert session.transcript.segments[-1].kind == "eos"
    return session


def save_to(path: Any, session: Any) -> Any:
    with path.open("w+b") as file:
        saved = session.save_payload(file, header=HEADER)
        assert not file.closed
    return saved


def test_save_syncs_the_trailing_eos_before_writing_header_and_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced = []
    original = FakeSession.sync

    def tracking_sync(self: Any, tokens: Any, **kwargs: Any) -> Any:
        synced.append(tuple(tokens))
        return original(self, tokens, **kwargs)

    monkeypatch.setattr(FakeSession, "sync", tracking_sync)
    with fake_engine(script=reply(b"ok")) as engine:
        session = chat_session(engine)
        kinds = []
        session.on_event(lambda event: kinds.append(event.kind), kinds=("Synced", "Saved"))
        path = tmp_path / "session.kv"
        saved = save_to(path, session)

        assert synced[-1] == session.transcript.tokens
        assert kinds[-2:] == ["Synced", "Saved"]
        assert saved.tokens == len(session.transcript.tokens)
        assert saved.payload_bytes > 0
        data = path.read_bytes()
        assert data[: len(HEADER)] == HEADER
        assert len(data) == len(HEADER) + saved.payload_bytes


def test_save_accepts_a_caller_header_without_a_store_length_policy(tmp_path: Path) -> None:
    header = b"short"
    path = tmp_path / "session.kv"
    with fake_engine() as engine:
        session = engine.session()
        session.append_tokens([5, 6], "user")
        with path.open("w+b") as file:
            saved = session.save_payload(file, header=header)
            assert file.tell() == len(header) + saved.payload_bytes

    data = path.read_bytes()
    assert data[:len(header)] == header
    assert len(data) == len(header) + saved.payload_bytes


def test_save_closes_the_staged_payload_on_success_and_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_objects = []
    original = FakeSession.stage_payload

    class Tracked:
        def __init__(self, inner: Any) -> None:
            self.inner, self.closes = inner, 0

        @property
        def bytes(self) -> Any:
            return self.inner.bytes

        def write(self, file: Any) -> Any:
            if fail:
                raise EngineError("staged write failed")
            return self.inner.write(file)

        def close(self) -> None:
            self.closes += 1
            self.inner.close()

    def stage(self: Any) -> Any:
        tracked = Tracked(original(self))
        staged_objects.append(tracked)
        return tracked

    monkeypatch.setattr(FakeSession, "stage_payload", stage)
    for fail in (False, True):
        with fake_engine() as engine:
            session = engine.session()
            session.append_tokens([5, 6], "user")
            path = tmp_path / f"session-{fail}.kv"
            with path.open("w+b") as file:
                if fail:
                    with pytest.raises(EngineError, match="staged write failed"):
                        session.save_payload(file, header=HEADER)
                else:
                    session.save_payload(file, header=HEADER)
                assert not file.closed
                file.write(b"tail")
    assert [tracked.closes for tracked in staged_objects] == [1, 1]


def test_load_adopts_native_tokens_restores_chat_and_permits_extension(tmp_path: Path) -> None:
    with fake_engine(script=reply(b"ok")) as engine:
        session = chat_session(engine)
        saved_tokens = session.transcript.tokens
        path = tmp_path / "session.kv"
        saved = save_to(path, session)

        session.new()
        assert session.transcript.tokens == ()
        loaded_events = []
        session.on_event(
            lambda event: loaded_events.append(event.payload["event"]), kinds=("Loaded",),
        )

        with path.open("r+b") as file:
            assert file.read(len(HEADER)) == HEADER
            loaded = session.load_payload(file, payload_bytes=saved.payload_bytes)
            assert not file.closed

        assert loaded == list(saved_tokens)
        assert session.transcript.tokens == saved_tokens
        assert session.transcript.chat
        assert [(segment.start, segment.end) for segment in session.transcript.segments] == [
            (0, len(saved_tokens)),
        ]
        assert session.ctx_used == len(saved_tokens)
        assert loaded_events == [Loaded(len(saved_tokens), saved.payload_bytes)]
        assert session.plan() == Plan("extend", len(saved_tokens), 0, True, None)

        session.append_user("more")
        events = list(session.run_round(max_tokens=0))
        synced = next(event for event in events if isinstance(event, Synced))
        assert synced.plan.kind == "extend"
        assert synced.plan.reused == len(saved_tokens)
        assert events[-1] == Stopped("max_tokens")


def test_load_clears_checkpoints_that_point_at_the_previous_transcript(tmp_path: Path) -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.append_tokens([5, 6, 7], "user")
        list(session.run_round(max_tokens=0))
        session.checkpoint("old")
        path = tmp_path / "session.kv"
        saved = save_to(path, session)

        with path.open("r+b") as file:
            file.seek(len(HEADER))
            session.load_payload(file, payload_bytes=saved.payload_bytes)
        with pytest.raises(CheckpointError, match="unknown checkpoint"):
            session.drop_checkpoint("old")


def test_load_failure_keeps_the_existing_transcript_and_file_usable(tmp_path: Path) -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.append_tokens([5, 6], "user")
        list(session.run_round(max_tokens=0))
        path = tmp_path / "truncated.kv"
        with path.open("w+b") as file:
            saved = session.save_payload(file, header=HEADER)
        expected = session.transcript.tokens

        with path.open("r+b") as file:
            file.seek(len(HEADER))
            file.truncate(len(HEADER) + saved.payload_bytes - 1)
            file.seek(len(HEADER))
            with pytest.raises(EngineError, match="truncated|corrupt|payload"):
                session.load_payload(file, payload_bytes=saved.payload_bytes)
            assert not file.closed
        assert session.transcript.tokens == expected
