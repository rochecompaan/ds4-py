"""DS4RunRecord telemetry encoding and failure semantics (Task 9)."""
from typing import Any
from pathlib import Path

import asyncio
from dataclasses import replace
from types import SimpleNamespace
import json

import pytest

import ds4py
from ds4py.engine.session import Engine
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.capabilities.run_record import DS4RunRecord


def fake_engine(*, script: Any=None) -> Engine:
    info = replace(FakeNative()._info, name="fake-ds4", is_deepseek41=False,
                   is_glm_dsa=False, is_qwen4=True, family="qwen")
    native = FakeNative(info=info) if script is None else FakeNative(info=info, script=script)
    return ds4py.open("f.gguf", ctx=65536, think=ThinkMode.NONE, native=native)


def scripted_text(text: Any) -> Any:
    tokens = tuple(tokenize_rendered(text))
    return lambda prompt, generated: (
        tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS
    )


def read_lines(path: Any) -> Any:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_engine_events_are_one_flushed_json_line_each(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with fake_engine(script=scripted_text("ok")) as engine:
        session = engine.session()
        record = DS4RunRecord(session, path)
        asyncio.run(record.for_run(SimpleNamespace(run_id="run-1")))
        session.start_chat()
        session.emit("run_start")
        list(session.run_round())
        session.emit("run_end", status="success")
        session_id = session.id

    lines = read_lines(path)
    # The engine's close publishes a trailing session_end; stop at this round's end.
    lines = lines[:[line["kind"] for line in lines].index("run_end") + 1]
    assert lines[0]["kind"] == "run_start" and lines[0]["run_id"] == "run-1"
    assert lines[-1]["kind"] == "run_end" and lines[-1]["payload"]["status"] == "success"
    for line in lines:
        assert set(line) >= {"ts", "kind", "session_id", "run_id", "payload"}
        assert line["session_id"] == session_id
        assert line["run_id"] == "run-1"

    kinds = [line["kind"] for line in lines]
    assert "Synced" in kinds and "Token" in kinds
    synced = next(line for line in lines if line["kind"] == "Synced")
    event = synced["payload"]["event"]
    assert event["plan"]["kind"] == "rebuild"
    assert event["pos"] > 0 and event["prefilled"] == event["pos"]

    tokens = [line["payload"]["event"] for line in lines if line["kind"] == "Token"]
    assert tokens
    for token in tokens:
        assert isinstance(token["id"], int)
        assert all(char in "0123456789abcdef" for char in token["text_bytes"])
        assert len(token["text_bytes"]) % 2 == 0
    assert "b'" not in path.read_text()


def test_record_write_failure_is_reported_and_never_hidden(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.mkdir(parents=True)
    with fake_engine(script=scripted_text("ok")) as engine:
        session = engine.session()
        record = DS4RunRecord(session, path)

        # A bus listener cannot break the engine: the failure is recorded instead.
        session.emit("run_start", run_id="run-1")
        assert record.failure is not None
        assert str(path) in str(record.failure)

        # The harness hook does surface it, so the run cannot report success.
        ctx = SimpleNamespace(conversation_id=session.id, run_id="run-1")
        with pytest.raises(OSError):
            asyncio.run(record.on_event(ctx, event=SimpleNamespace(parts=[])))
        assert record.failure is not None


def test_relocate_sends_later_lines_to_the_current_session_directory(tmp_path: Path) -> None:
    first = tmp_path / "first" / "events.jsonl"
    second = tmp_path / "second" / "events.jsonl"
    with fake_engine(script=scripted_text("ok")) as engine:
        session = engine.session()
        record = DS4RunRecord(session, first)
        session.emit("saved", session_id=session.id)
        record.relocate(second)
        session.emit("switched", session_id=session.id, restore_path="kv")

    assert [line["kind"] for line in read_lines(first)] == ["saved"]
    kinds = [line["kind"] for line in read_lines(second)]
    assert kinds[0] == "switched" and "saved" not in kinds
    assert read_lines(second)[0]["payload"]["restore_path"] == "kv"
