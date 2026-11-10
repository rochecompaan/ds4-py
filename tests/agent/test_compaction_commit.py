"""Ordered compaction: prepare provenance, replace natively, then commit records and messages."""

import asyncio
import threading
from typing import Any, override

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

import ds4py
import ds4py.compaction_commit as commit_module
from ds4py._native.fake import FakeNative
from ds4py.binding import MessageBinder
from ds4py.compaction_commit import commit_compaction
from ds4py.engine.session import Engine, Session
from ds4py.engine.transcript import Segment, TranscriptReplacement
from ds4py.errors import EngineError


def fake_engine() -> Engine:
    return ds4py.open("f.gguf", ctx=8192, native=FakeNative())


def user(text: str) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)])


def key(message: ModelMessage) -> str:
    value = (message.metadata or {}).get("ds4.key")
    assert isinstance(value, str)
    return value


def message_history() -> list[ModelMessage]:
    return [
        user("first"),
        ModelResponse([TextPart("first answer")]),
        user("second"),
        ModelResponse([TextPart("second answer")]),
    ]


def replacement_for(session: Session, keep: list[ModelMessage]) -> TranscriptReplacement:
    """Stand in for the tail planner: keep the preamble and the retained messages' tokens."""
    keys = {key(message) for message in keep}
    tokens: list[int] = []
    segments: list[Segment] = []
    for segment in session.transcript.segments:
        segment_key = segment.meta.get("key")
        if segment_key is not None and segment_key not in keys:
            continue
        start = len(tokens)
        tokens.extend(session.transcript.tokens[segment.start : segment.end])
        segments.append(Segment(start, len(tokens), segment.kind, segment.meta))
    return TranscriptReplacement(tuple(tokens), tuple(segments))


class RecordingList(list[ModelMessage]):
    """Record the in-place message assignment that finishes the commit."""

    def __init__(self, items: list[ModelMessage], calls: list[str]) -> None:
        super().__init__(items)
        self.calls = calls

    @override
    def __setitem__(self, index: Any, value: Any) -> None:
        self.calls.append("messages")
        super().__setitem__(index, value)


def test_commit_prepares_before_native_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        calls: list[str] = []
        messages = RecordingList(message_history(), calls)
        binder.prepare(messages)
        compacted_messages = list(messages[2:])
        replacement = replacement_for(session, compacted_messages)
        calls.clear()
        prepared_records: list[tuple[object, ...]] = []
        threads: list[int] = []

        prepare = commit_module.prepare_provenance

        def record_prepare(snapshot: Any, segments: Any, *, loaded: int | None) -> Any:
            assert loaded == len(replacement.tokens)
            assert list(snapshot) == compacted_messages
            assert segments == replacement.segments
            calls.append("prepare")
            threads.append(threading.get_ident())
            return prepare(snapshot, segments, loaded=loaded)

        replace_transcript = session.replace_transcript

        def record_replace(rep: TranscriptReplacement, *, reason: str) -> Any:
            calls.append("replace")
            threads.append(threading.get_ident())
            return replace_transcript(rep, reason=reason)

        apply_compaction = binder.apply_compaction

        def record_apply(prepared: Any) -> None:
            calls.append("apply_compaction")
            threads.append(threading.get_ident())
            prepared_records.append(prepared.records)
            apply_compaction(prepared)

        monkeypatch.setattr(commit_module, "prepare_provenance", record_prepare)
        monkeypatch.setattr(session, "replace_transcript", record_replace)
        monkeypatch.setattr(binder, "apply_compaction", record_apply)

        plan = asyncio.run(commit_compaction(binder, messages, compacted_messages, replacement))

        assert calls == ["prepare", "replace", "apply_compaction", "messages"]
        assert threads[0] == threads[2] == threading.get_ident()
        assert threads[1] != threading.get_ident()
        assert plan.reused + plan.to_prefill == len(replacement.tokens)
        assert messages == compacted_messages
        assert session.transcript.tokens == replacement.tokens
        assert session.transcript.segments == replacement.segments
        assert binder.records() == tuple(prepared_records[0])


def test_native_failure_needs_no_rollback() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        messages = message_history()
        binder.prepare(messages)
        compacted_messages = list(messages[2:])
        replacement = replacement_for(session, compacted_messages)
        original_messages = list(messages)
        original_segments = binder.export_segments()
        original_records = binder.records()
        session.cancel()

        with pytest.raises(EngineError, match="transcript replacement"):
            asyncio.run(commit_compaction(binder, messages, compacted_messages, replacement))

        assert messages == original_messages
        assert binder.export_segments() == original_segments
        assert binder.records() == original_records
