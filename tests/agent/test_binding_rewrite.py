from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any, cast

import pytest
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative
from ds4py.binding import MessageBinder
from ds4py.engine.prompt import PromptBuilder
from ds4py.engine.session import Engine
from ds4py.engine.worker import SyncJob
from ds4py.errors import CheckpointError, EngineError


def fake_engine(family: Any = "deepseek4", *, think: Any = ThinkMode.NONE, **fake: Any) -> Engine:
    info = replace(
        FakeNative()._info,
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )
    return ds4py.open(
        "f.gguf",
        ctx=8192,
        think=think,
        native=FakeNative(info=info, **fake),
    )


def user(text: Any) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)])


def tool(name: Any) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Run {name}.",
        parameters_json_schema={"type": "object", "properties": {}},
    )


def reserialize(messages: Any) -> list[Any]:
    return ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))


def native_calls(session: Any) -> list[Any]:
    return session._worker.call(lambda engine, native: list(native.calls)).result()


def test_identical_reserialized_messages_keep_bound_prefix() -> None:
    messages = [user("first"), ModelResponse([TextPart("answer")]), user("second")]
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        binder.prepare(messages, tools=[tool("read")], instructions="rules")
        before = session.transcript.tokens
        provenance = binder.export_segments()

        change = binder.prepare(reserialize(messages), tools=[tool("read")], instructions="rules")

        assert change.kind == "unchanged"
        assert change.start == len(before)
        assert session.transcript.tokens == before
        assert binder.export_segments() == provenance


def test_changed_message_renders_only_from_its_segment_start() -> None:
    original = [user("short"), ModelResponse([TextPart("same answer")]), user("same later")]
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        binder.prepare(original)
        old_tokens = session.transcript.tokens
        old_provenance = binder.export_segments()
        first_start = next(p.start for p in old_provenance if p.kind == "user")
        metadata = original[2].metadata
        assert metadata is not None
        later_key = metadata["ds4.key"]
        old_later_start = next(p.start for p in old_provenance if p.key == later_key)

        changed = deepcopy(original)
        cast("UserPromptPart", changed[0].parts[0]).content = "a considerably longer first message"
        change = binder.prepare(changed)

        assert change.kind == "rewrite"
        assert change.start == first_start
        assert session.transcript.tokens[:first_start] == old_tokens[:first_start]
        new_later_start = next(p.start for p in binder.export_segments() if p.key == later_key)
        assert new_later_start > old_later_start


def test_same_key_with_changed_hash_does_not_reuse_old_segment() -> None:
    original = [user("original")]
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        binder.prepare(original)
        key = cast("Any", original[0].metadata)["ds4.key"]
        changed = deepcopy(original)
        cast("UserPromptPart", changed[0].parts[0]).content = "changed"
        assert cast("Any", changed[0].metadata)["ds4.key"] == key

        change = binder.prepare(changed)

        assert change.kind == "rewrite"
        assert b"changed" in b"".join(engine.token_text(token) for token in session.transcript.tokens)


def test_runtime_ids_and_timestamps_do_not_invalidate_unchanged_rendered_content() -> None:
    original = [user("same"), ModelResponse([TextPart("answer")])]
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        binder.prepare(original)
        before = session.transcript.tokens
        changed = deepcopy(original)
        request = cast("ModelRequest", changed[0])
        request.run_id = "new-run"
        request.conversation_id = "new-conversation"
        request.timestamp = datetime.now(UTC)
        cast("UserPromptPart", request.parts[0]).timestamp = datetime.now(UTC)
        response = cast("ModelResponse", changed[1])
        response.run_id = "another-run"
        response.timestamp = datetime.now(UTC)
        response.usage = RequestUsage(input_tokens=999, output_tokens=888)
        response.history_digest = "different"
        response.provider_response_id = "different"

        change = binder.prepare(changed)

        assert change.kind == "unchanged"
        assert session.transcript.tokens == before


def test_tool_definition_change_rebuilds_from_tool_prefix() -> None:
    messages = [user("same")]
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        binder.prepare(messages, tools=[tool("first")], instructions="rules")

        change = binder.prepare(messages, tools=[tool("second")], instructions="rules")

        assert change.kind == "rewrite"
        assert change.start == 0
        text = b"".join(engine.token_text(token) for token in session.transcript.tokens)
        assert b"second" in text and b"first" not in text


def test_think_change_rebuilds_system_prefix() -> None:
    messages = [user("same")]
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        binder.prepare(messages)
        engine.think = ThinkMode.MAX

        change = binder.prepare(messages)

        assert change.kind == "rewrite"
        assert change.start == 0
        assert b"Reasoning Effort" in b"".join(engine.token_text(token) for token in session.transcript.tokens)


def test_count_tokens_does_not_mutate_or_sync() -> None:
    messages = [user("first")]
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        binder.prepare(messages, tools=[tool("read")], instructions="rules")
        before = session.transcript.tokens
        before_segments = session.transcript.segments
        before_calls = native_calls(session)

        count = binder.count_tokens([*messages, user("next")], tools=[tool("read")], instructions="rules")

        builder = PromptBuilder(engine)
        user_delta = builder.message(before, "user", "next")
        prefix = builder.assistant_prefix((*before, *user_delta))
        assert count == len(before) + len(user_delta) + len(prefix)
        assert session.transcript.tokens == before
        assert session.transcript.segments == before_segments
        assert native_calls(session) == before_calls == []


def test_provenance_covers_every_token_and_marks_sampled_response() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        request = user("hello")
        binder.prepare([request])
        list(session.run_round(max_tokens=10))
        response = ModelResponse([TextPart("ok")], metadata={"ds4.source": "sampled"})
        before = session.transcript.tokens

        binder.bind_response(response)
        provenance = binder.export_segments()

        assert session.transcript.tokens == before
        assert provenance[0].start == 0
        assert provenance[-1].end == len(before)
        assert all(left.end == right.start for left, right in pairwise(provenance))
        sampled = [p for p in provenance if p.source == "sampled"]
        assert [p.kind for p in sampled] == ["assistant", "eos"]
        assert {p.key for p in sampled} == {cast("Any", response.metadata)["ds4.key"]}
        assert len({p.content_hash for p in sampled}) == 1
        assert any(p.kind == "assistant_prefix" and not p.key for p in provenance)


def test_unchanged_messages_leave_paused_sampled_response_resumable() -> None:
    stop = tuple(byte + 1 for byte in b"stop")

    def script(prompt: Any, generated: Any) -> Any:
        text = (
            b"next"
            if any(tuple(prompt[index : index + len(stop)]) == stop for index in range(len(prompt) - len(stop) + 1))
            else b"stop"
        )
        return text[len(generated)] + 1 if len(generated) < len(text) else 0

    with fake_engine(script=script) as engine:
        session = engine.session()
        binder = MessageBinder(session)
        request = user("hello")
        response = ModelResponse([TextPart("stop")])
        binder.prepare([request])
        list(session.run_round(max_tokens=10, stop_sequences=[b"stop"]))
        binder.bind_response(response)
        paused = session._paused_stream
        assert paused is not None
        before = session.transcript.tokens

        change = binder.prepare([request, response])

        assert change.kind == "unchanged"
        assert session._paused_stream is paused
        assert session.transcript.tokens == before
        assert not [s for s in session.transcript.segments if s.kind == "eos"]
        list(session.run_round(max_tokens=10, stop_sequences=[b"stop"]))
        assert session._paused_stream is None
        assert session.transcript.segments[-1].kind == "eos"


def test_completed_pause_resume_refreshes_eos_provenance_and_bound_end() -> None:
    stop = tuple(byte + 1 for byte in b"stop")

    def script(prompt: Any, generated: Any) -> Any:
        text = (
            b"next"
            if any(tuple(prompt[index : index + len(stop)]) == stop for index in range(len(prompt) - len(stop) + 1))
            else b"stop"
        )
        return text[len(generated)] + 1 if len(generated) < len(text) else 0

    with fake_engine(script=script) as engine:
        session = engine.session()
        binder = MessageBinder(session)
        request = user("hello")
        response = ModelResponse([TextPart("stop")])
        binder.prepare([request])
        list(session.run_round(max_tokens=10, stop_sequences=[b"stop"]))
        binder.bind_response(response)
        binder.prepare([request, response])
        list(session.run_round(max_tokens=10, stop_sequences=[b"stop"]))
        cast("TextPart", response.parts[0]).content = "stopnext"

        binder.bind_response(response)
        later = user("later")
        binder.prepare([request, response, later])

        provenance = binder.export_segments()
        eos = next(p for p in provenance if p.kind == "eos")
        assert eos.key == cast("Any", response.metadata)["ds4.key"]
        assert eos.content_hash
        assert eos.source == "sampled"
        assert binder._records[1].end == eos.end
        assert len(binder._records) == 3
        assert provenance[0].start == 0
        assert provenance[-1].end == len(session.transcript)
        assert all(left.end == right.start for left, right in pairwise(provenance))
        old_tokens = session.transcript.tokens
        later_start = next(p.start for p in provenance if p.key == cast("Any", later.metadata)["ds4.key"])

        change = binder.prepare([request, response, user("replacement")])

        assert change.kind == "rewrite"
        assert change.start == later_start
        assert session.transcript.tokens[:later_start] == old_tokens[:later_start]
        restored_eos = next(p for p in binder.export_segments() if p.kind == "eos")
        assert restored_eos.key == cast("Any", response.metadata)["ds4.key"]
        assert restored_eos.source == "sampled"


def test_append_suffix_retires_paused_response_with_sampled_eos_provenance() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        request = user("hello")
        response = ModelResponse([TextPart("ok")])
        binder.prepare([request])
        list(session.run_round(max_tokens=10, stop_sequences=[b"ok"]))
        binder.bind_response(response)

        change = binder.prepare([request, response, user("next")])

        assert change.kind == "append"
        assert session._paused_stream is None
        assert [s.kind for s in session.transcript.segments][-4:] == [
            "assistant_prefix",
            "assistant",
            "eos",
            "user",
        ]
        provenance = binder.export_segments()
        eos = next(p for p in provenance if p.kind == "eos")
        assert eos.key == cast("Any", response.metadata)["ds4.key"]
        assert eos.content_hash
        assert eos.source == "sampled"
        assert provenance[0].start == 0
        assert provenance[-1].end == len(session.transcript)
        assert all(left.end == right.start for left, right in pairwise(provenance))
        list(session.run_round(max_tokens=10, stop_sequences=[b"ok"]))
        assert [s.kind for s in session.transcript.segments][-2:] == [
            "assistant_prefix",
            "assistant",
        ]
        assert [s.kind for s in session.transcript.segments].count("assistant_prefix") == 2


def test_truncate_drops_only_checkpoints_beyond_the_cut() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.append_tokens([1, 2, 3, 4], "user")
        list(session.run_round(max_tokens=0))
        session.checkpoint("before")
        session.append_tokens([5, 6], "user")
        list(session.run_round(max_tokens=0))
        session.checkpoint("after")

        session.truncate(4)

        assert session.transcript.tokens == (1, 2, 3, 4)
        assert session.plan().kind == "restore"
        assert session.plan().checkpoint == "before"
        with pytest.raises(CheckpointError, match="after"):
            session.drop_checkpoint("after")

        session.truncate(2)
        assert session.plan().kind == "rebuild"
        with pytest.raises(CheckpointError, match="before"):
            session.drop_checkpoint("before")


@pytest.mark.parametrize(
    ("family", "parts"),
    [
        ("qwen", [ToolCallPart("typed", {"value": 10**100})]),
        *[
            (
                family,
                [
                    ToolCallPart("first", {"value": "one"}),
                    TextPart("between"),
                    ToolCallPart("second", {"value": "two"}),
                ],
            )
            for family in ("deepseek4", "deepseek41", "glm", "qwen")
        ],
        *[
            (family, [ToolCallPart("typed", {"value": value})])
            for family in ("deepseek4", "deepseek41", "glm", "qwen")
            for value in ({"nested": ("not", "JSON")}, {"nested": {1: "non-string key"}})
        ],
    ],
)
def test_rejected_replacement_is_atomic(family: Any, parts: Any) -> None:
    with fake_engine(family) as engine:
        session = engine.session()
        binder = MessageBinder(session)
        original = [user("first"), ModelResponse([TextPart("old")]), user("later")]
        binder.prepare(original)
        list(session._worker.submit(SyncJob(session.transcript.tokens)))
        session.checkpoint("kept")
        before_tokens = session.transcript.tokens
        before_segments = session.transcript.segments
        before_records = tuple(binder._records)
        before_checkpoints = dict(session._checkpoint_tokens)
        changed = deepcopy(original)
        changed[1].parts = deepcopy(parts)

        with pytest.raises(ValueError, match=r"cannot|JSON"):
            binder.prepare(changed)

        assert session.transcript.tokens == before_tokens
        assert session.transcript.segments == before_segments
        assert tuple(binder._records) == before_records
        assert session._checkpoint_tokens == before_checkpoints
        assert session.plan().kind == "extend"
        session.drop_checkpoint("kept")


def test_rewriting_sampled_response_rebuilds_native_prefix() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        request = user("hello")
        response = ModelResponse([TextPart("ok")])
        binder.prepare([request])
        list(session.run_round(max_tokens=10))
        binder.bind_response(response)
        old_prefix = next(p for p in binder.export_segments() if p.kind == "assistant_prefix")
        old_prefix_tokens = session.transcript.tokens[old_prefix.start : old_prefix.end]

        changed = deepcopy(response)
        cast("TextPart", changed.parts[0]).content = "replacement"
        change = binder.prepare([request, changed])

        assert change.kind == "rewrite"
        assert change.start == old_prefix.start
        prefixes = [p for p in binder.export_segments() if p.kind == "assistant_prefix"]
        assert len(prefixes) == 1
        assert session.transcript.tokens[prefixes[0].start : prefixes[0].end] == old_prefix_tokens
        assert len([p for p in binder.export_segments() if p.kind == "assistant"]) == 1
        decoded = b"".join(engine.token_text(token) for token in session.transcript.tokens)
        assert b"replacement" in decoded


def test_invalid_truncate_preserves_checkpoints() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.append_tokens([1, 2, 3, 4], "user")
        list(session.run_round(max_tokens=0))
        session.checkpoint("kept")

        with pytest.raises(ValueError, match="outside"):
            session.truncate(-1)

        assert session.plan().kind == "extend"
        session.drop_checkpoint("kept")


def test_truncate_refuses_an_active_round() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.append_tokens([1, 2], "user")
        round_events = session.run_round()
        next(round_events)
        try:
            with pytest.raises(EngineError, match="round is open"):
                session.truncate(1)
            assert session.transcript.tokens == (1, 2)
        finally:
            round_events.close()
