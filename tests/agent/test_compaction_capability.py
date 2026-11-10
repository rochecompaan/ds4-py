"""Persistent compaction uses real keyed ranges, exact tokens, and ordered commits."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from test_context_strategy import RecordingOwner, _tool_agent, fake_engine, prepared_tool_binder

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.binding import MessageBinder
from ds4py.capabilities.compaction import LLMCompaction
from ds4py.capabilities.compaction_tail import plan_tail, tail_boundary
from ds4py.capabilities.context import ContextRequest, RoomResult
from ds4py.engine.lifecycle import SessionEvent
from ds4py.errors import EngineError


def script(prompt: Any, generated: Any) -> int:
    tokens = tokenize_rendered("durable state")
    return tokens[len(generated)] if len(generated) < len(tokens) else FAKE_EOS


def user(text: str) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)])


def history() -> list[ModelMessage]:
    return [
        user("old " * 500),
        ModelResponse([TextPart("old answer")]),
        user("recent"),
        ModelResponse([TextPart("answer")]),
    ]


def test_compaction_rewrites_live_history_and_continues_without_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
        session = engine.session()
        binder = MessageBinder(session)
        messages = history()
        binder.prepare(messages, instructions="keep coding")
        tail = deepcopy(messages[2:])
        original = session.transcript.tokens
        bottom = binder.records()[2].start
        events: list[SessionEvent] = []
        session.on_event(events.append, kinds={"before_compact", "compacted"})
        prepare = session.prepare_summary
        offsets: list[int] = []

        def record_prepare(offset: int) -> Any:
            offsets.append(offset)
            return prepare(offset)

        monkeypatch.setattr(session, "prepare_summary", record_prepare)
        compaction = LLMCompaction(binder)
        result = asyncio.run(compaction.compact_messages(messages, reason="manual"))

        assert result.summary == "durable state"
        assert result.generated_tokens == 13
        assert (result.dropped, result.kept) == (2, 2)
        assert offsets == [bottom]
        assert isinstance(messages[0], ModelRequest)
        assert isinstance(messages[0].parts[0], UserPromptPart)
        assert messages[0].parts[0].content == "durable state"
        assert messages[1:] == tail
        assert session.transcript.tokens[-len(original[bottom:]) :] == original[bottom:]
        assert [event.kind for event in events] == ["before_compact", "compacted"]
        assert events[0].payload == {"reason": "manual", "old_tokens": len(original)}
        assert events[1].payload == {
            "reason": "manual",
            "old_tokens": len(original),
            "new_tokens": len(session.transcript),
            "generated_tokens": 13,
        }
        unchanged = binder.prepare(messages, instructions="keep coding")
        assert unchanged.kind == "unchanged"
        tokens = session.transcript.tokens
        binder.prepare([*messages, user("next")], instructions="keep coding")
        assert session.transcript.tokens[: len(tokens)] == tokens
        assert session.plan().to_prefill == len(session.transcript) - len(tokens)


@pytest.mark.parametrize(
    ("users", "bottom", "system", "budget", "want"),
    [
        ([90, 170], 200, 10, 50, 170),
        ([110], 200, 10, 50, 110),  # prefer user in the 2x scan, even before target
        ([100], 200, 10, 50, 100),  # inclusive 2x boundary
        ([99], 200, 10, 50, 150),  # outside scan: bounded fallback
        ([], 40, 10, 50, 10),
    ],
)
def test_c_tail_span(users: list[int], bottom: int, system: int, budget: int, want: int) -> None:
    assert tail_boundary(bottom, system, budget, users) == want


@pytest.mark.parametrize(("ctx", "budget"), [(8192, 819), (1_000_000, 50_000)])
def test_tail_budget_and_whole_tool_exchange(ctx: int, budget: int) -> None:
    with ds4py.open("f.gguf", ctx=ctx, think=ThinkMode.NONE, native=FakeNative()) as engine:
        binder = MessageBinder(engine.session())
        messages: list[ModelMessage] = [
            user("old"),
            ModelResponse([TextPart("a" * (budget * 2 + 10))]),
            ModelResponse([ToolCallPart("read", {}, tool_call_id="call")]),
            ModelRequest([ToolReturnPart("read", "x" * (budget + 10), tool_call_id="call")]),
            ModelResponse([TextPart("done")]),
        ]
        binder.prepare(messages)
        planned = plan_tail(binder, messages)
        # Fallback target lands inside the return; it must include the call too.
        assert planned.messages == tuple(messages[2:])
        assert planned.bottom_tokens == binder.records()[2].start
        assert planned.retained.tokens == binder.session.transcript.tokens[planned.bottom_tokens :]


@pytest.mark.parametrize("phase", ["summary", "replacement"])
def test_native_failure_leaves_messages_and_provenance_unchanged(phase: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
        session = engine.session()
        binder = MessageBinder(session)
        messages = history()
        binder.prepare(messages)
        before = (deepcopy(messages), binder.records(), binder.export_segments(), session.transcript.tokens)
        events: list[SessionEvent] = []
        session.on_event(events.append, kinds={"before_compact", "compacted"})
        if phase == "summary":
            session.cancel()
        else:
            prepare = session.prepare_summary

            def cancel_after_summary(bottom: int) -> Any:
                result = prepare(bottom)
                session.cancel()
                return result

            monkeypatch.setattr(session, "prepare_summary", cancel_after_summary)
        with pytest.raises(EngineError):
            asyncio.run(LLMCompaction(binder).compact_messages(messages, reason="manual"))
        assert (messages, binder.records(), binder.export_segments(), session.transcript.tokens) == before
        assert [event.kind for event in events] == ["before_compact"]


@pytest.mark.parametrize(
    ("size", "used", "soft", "reserve", "expected"),
    [
        (65536, 55705, 0.85, 5156, True),
        (65536, 55704, 0.85, 5156, False),
        (8192, 5980, 0.99, 2084, True),
        (8192, 5979, 0.99, 2084, False),
        (100000, 91808, 0.99, 5156, True),
        (100000, 91807, 0.99, 5156, False),
        (8192, 0, 0.85, 2084, False),
    ],
)
def test_exact_c_triggers(size: int, used: int, soft: float, reserve: int, expected: bool) -> None:
    with ds4py.open("f.gguf", ctx=size, think=ThinkMode.NONE, native=FakeNative()) as engine:
        session = engine.session()
        if used:
            session.append_tokens([1] * used, "user")
        compactor = LLMCompaction(MessageBinder(session), soft=soft)
        # 921 prompt bytes + 72 C reserve-reason bytes + 3 delimiters,
        # plus C's 64 and the clamped summary budget.
        assert compactor.summary_reserve() == reserve
        assert compactor.should_compact() is expected


def test_before_request_mutates_authoritative_list_and_counts_summary_once() -> None:
    with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
        binder = MessageBinder(engine.session())
        messages = history()
        messages[0] = user("x" * 6000)
        binder.prepare(messages)
        ctx = SimpleNamespace(messages=messages, usage=SimpleNamespace(output_tokens=7))
        request = SimpleNamespace(messages=list(messages))
        result = asyncio.run(LLMCompaction(binder).before_model_request(cast("Any", ctx), cast("Any", request)))
        assert result is request
        assert result.messages is messages
        assert isinstance(messages[0].parts[0], UserPromptPart)
        assert messages[0].parts[0].content == "durable state"
        assert ctx.usage.output_tokens == 20


@pytest.mark.parametrize(("needed", "want"), [(10, True), (20_000, False)])
def test_make_room_checks_actual_freed_tokens(needed: int, want: bool) -> None:
    with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
        binder = MessageBinder(engine.session())
        messages = history()
        binder.prepare(messages)
        room = asyncio.run(LLMCompaction(binder).make_room(messages, ContextRequest(needed, "tool_result")))
        assert room.made_room is want
        assert room.generated_tokens == 13


def test_tool_overflow_adds_summary_usage_once() -> None:
    async def run(generated: int) -> int:
        with fake_engine() as engine:
            binder = prepared_tool_binder(engine.session())
            owner = RecordingOwner(binder, room=RoomResult(False, generated))
            result = await _tool_agent(owner, "x" * 7287).run("go")
            return result.usage.output_tokens

    assert asyncio.run(run(13)) - asyncio.run(run(0)) == 13


def test_tail_detects_unkeyed_native_assistant_without_a_fake_message() -> None:
    with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=FakeNative()) as engine:
        binder = MessageBinder(engine.session())
        messages = history()
        binder.prepare(messages)
        start = len(binder.session.transcript)
        binder.session.append_tokens([260, 267], "assistant_prefix")
        binder.session.append_tokens([98, 99], "assistant")
        planned = plan_tail(binder, messages)
        assert len(planned.messages) == 2
        assert planned.retained.tokens[-4:] == (260, 267, 98, 99)
        assert planned.retained.segments[-2].start == start - planned.bottom_tokens


def test_compaction_preserves_system_and_instruction_parts() -> None:
    from pydantic_ai.messages import InstructionPart, SystemPromptPart

    with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=FakeNative(script=script)) as engine:
        binder = MessageBinder(engine.session())
        messages = history()
        first = messages[0]
        assert isinstance(first, ModelRequest)
        # The binder accepts InstructionPart although upstream ModelRequestPart omits it.
        first.parts = cast(
            "Any", [SystemPromptPart("system constraint"), InstructionPart("instruction constraint"), *first.parts]
        )
        first.instructions = "configured instructions"
        binder.prepare(messages, instructions="configured instructions")
        preamble = binder.session.transcript.tokens[: binder.records()[0].start]
        asyncio.run(LLMCompaction(binder).compact_messages(messages, reason="manual"))
        summary = messages[0]
        assert isinstance(summary, ModelRequest)
        assert isinstance(summary.parts[0], SystemPromptPart)
        assert summary.parts[0].content == "system constraint"
        assert isinstance(summary.parts[1], InstructionPart)
        assert summary.parts[1].content == "instruction constraint"
        assert summary.instructions == "configured instructions"
        assert binder.session.transcript.tokens[: len(preamble)] == preamble
        assert binder.prepare(messages, instructions="configured instructions").kind == "unchanged"
