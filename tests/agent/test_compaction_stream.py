"""Reserve compaction keeps one native assistant and one public response alive."""

import asyncio
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any, cast, override

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import SPECIAL_TOKENS, tokenize_rendered
from ds4py.binding import MessageBinder
from ds4py.capabilities.compaction import LLMCompaction
from ds4py.capabilities.compaction_tail import plan_tail
from ds4py.capabilities.context import ContextRequest, ContextStrategy, RoomResult
from ds4py.compaction_commit import commit_compaction
from ds4py.engine.events import SummaryToken, Synced
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.stream import RoundStream
from ds4py.engine.transcript import Segment, TranscriptReplacement
from ds4py.errors import EngineError
from ds4py.model import DS4Model

TOOL = "<tool_call>\n<function=read>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>"
PREFIX = tuple(tokenize_rendered("<|im_start|>assistant\n<think>\n\n</think>\n\n"))


@pytest.mark.parametrize("early_close", [False, True])
@pytest.mark.parametrize("canonical_context", [False, True])
def test_reserve_compaction_continues_same_response(
    early_close: bool, canonical_context: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Losing/re-rendering the open tail, replacing the parser, or reconciling at
    # the pre-compaction offset must break the public response or exact tokens.
    output = tuple(tokenize_rendered("before after" + TOOL))
    marker = tuple(tokenize_rendered("Internal ds4-agent context compaction request."))
    summary_prompts: list[tuple[int, ...]] = []
    sampled = 0

    def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
        nonlocal sampled
        if any(tuple(prompt[i : i + len(marker)]) == marker for i in range(len(prompt))):
            if not generated:
                summary_prompts.append(tuple(prompt))
            summary = b"durable state"
            return summary[len(generated)] + 1 if len(generated) < len(summary) else FAKE_EOS
        token = output[sampled] if sampled < len(output) else FAKE_EOS
        sampled += 1
        return token

    finishes: list[RoundStream] = []
    finalize = RoundStream.finalize

    def counted_finalize(self: RoundStream, **kwargs: Any) -> None:
        finishes.append(self)
        finalize(self, **kwargs)

    monkeypatch.setattr(RoundStream, "finalize", counted_finalize)

    async def scenario() -> None:
        native = FakeNative(script=script, info=replace(FakeNative()._info, family="qwen", is_qwen4=True))
        with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=native) as engine:
            session = engine.session()
            binder = MessageBinder(session)
            model = DS4Model(session, binder=binder, sampling=engine.sampling_defaults)
            model.context_owner = LLMCompaction(binder)
            messages: list[ModelMessage] = [
                ModelRequest([UserPromptPart("x" * 4000)]),
                ModelResponse([TextPart("old answer")]),
                ModelRequest([UserPromptPart("r" * 1500)]),
            ]
            syncs: list[Synced] = []
            summaries: list[SummaryToken] = []
            committed: list[tuple[int, ...]] = []
            open_tails = []
            bound_keys = []
            pull_locked = []
            original_tokens: tuple[int, ...] = ()
            selected_bottom = 0

            def on_compacted(event: object) -> None:
                committed.append(session.transcript.tokens)
                open_tails.append(session.transcript.segments[-2:])
                bound_keys.append([record.key for record in binder.records()])
                pull_locked.append(stream._pull_lock.locked())

            session.on_event(on_compacted, kinds={"compacted"})

            def record_event(notification: SessionEvent) -> None:
                event = notification.payload.get("event")
                if isinstance(event, Synced):
                    syncs.append(event)
                elif isinstance(event, SummaryToken):
                    summaries.append(event)

            session.on_event(record_event, kinds={"Synced", "SummaryToken"})
            history = deepcopy(messages) if canonical_context else messages
            run_context = RunContext(deps=None, model=model, usage=RunUsage(), messages=history)
            async with model.request_stream(
                messages, None, ModelRequestParameters(), run_context if canonical_context else None
            ) as stream:
                metadata = stream.get().metadata
                assert metadata is not None
                original_key = metadata["ds4.key"]
                original_tokens = session.transcript.tokens
                selected_bottom = binder.records()[2].start
                # Seven text tokens are sampled before the reserve decision.
                session._worker.call(
                    lambda e, n: setattr(n, "_ctx", len(original_tokens) + len(PREFIX) + 7 + 1)
                ).result()
                iterator = stream.__aiter__()
                events = []
                async for event in iterator:
                    events.append(event)
                    if early_close and committed:
                        await cast("Any", iterator).aclose()
                        break
                response = stream.get()

            assert isinstance(history[0], ModelRequest)
            summary_part = history[0].parts[-1]
            assert isinstance(summary_part, UserPromptPart) and summary_part.content == "durable state"
            if canonical_context:
                assert isinstance(messages[0], ModelRequest)
                original_part = messages[0].parts[0]
                assert isinstance(original_part, UserPromptPart) and original_part.content == "x" * 4000
            assert len(committed) == 1
            assert pull_locked == [False]
            assert original_key not in bound_keys[0]
            tail = open_tails[0]
            assert [s.kind for s in tail] == ["assistant_prefix", "assistant"]
            assert all(not s.meta.get("key") for s in tail)
            assert committed[0][tail[0].start :] == (*PREFIX, *output[:7])
            assert len(summary_prompts) == 1
            assert summary_prompts[0][:selected_bottom] == original_tokens[:selected_bottom]
            assert summary_prompts[0][selected_bottom : selected_bottom + 6 + len(marker)] == (
                SPECIAL_TOKENS["<|im_start|>"],
                118,
                116,
                102,
                115,
                11,
                *marker,
            )
            assert len(summaries) == 13
            assert response.metadata is not None
            assert response.metadata["ds4.key"] == original_key
            if not early_close:
                assert len(finishes) == 1
            assert response.parts[0] == TextPart("before a" if early_close else "before after")
            if not early_close:
                assert response.parts[1:] == [
                    ToolCallPart(
                        "read",
                        {"path": "/tmp/x"},
                        tool_call_id=f"{original_key}:tool:0",
                        provider_details={"ds4py.raw": TOOL},
                    )
                ]
            assert response.usage.output_tokens == (8 if early_close else len(output)) + 13
            # Setup sync is outside the request; replacement and resumed syncs
            # both contribute their actual prefill/cache work exactly once.
            assert response.usage.input_tokens == sum(s.prefilled for s in syncs[1:])
            assert response.usage.cache_read_tokens == sum(s.plan.reused for s in syncs[1:])
            assert len(syncs) == 4
            assert syncs[-1].prefilled == 0
            assert syncs[-1].pos == len(committed[0])
            segments = session.transcript.segments
            assert [s.kind for s in segments].count("assistant_prefix") == 1
            assistant = next(s for s in segments if s.kind == "assistant")
            assert assistant.meta["key"] == original_key
            expected = output[:8] if early_close else output
            assert session.transcript.tokens[assistant.start : assistant.end] == expected
            assert binder.records()[-1].end == len(session.transcript)
            native_tokens = session._worker.call(lambda e, n: tuple(n.tokens())).result()
            # Normal EOS is transcript-only until the next sync; early cleanup
            # rewinds both representations to the published sampled boundary.
            assert native_tokens == (session.transcript.tokens if early_close else session.transcript.tokens[:-1])

    asyncio.run(scenario())


def test_tail_keeps_open_assistant_when_no_bound_message_is_selected() -> None:
    # A long open assistant can move the target beyond the final bound message.
    with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=FakeNative()) as engine:
        binder = MessageBinder(engine.session())
        messages: list[ModelMessage] = [ModelRequest([UserPromptPart("old")])]
        binder.prepare(messages)
        start = len(binder.session.transcript)
        binder.session.append_tokens([260, 267], "assistant_prefix")
        binder.session.append_tokens([98] * 2000, "assistant")
        tail = plan_tail(binder, messages)
        assert tail.messages == ()
        assert tail.bottom_tokens == start
        assert tail.retained.tokens == (260, 267, *([98] * 2000))
        assert [(s.start, s.end, s.kind) for s in tail.retained.segments] == [
            (0, 2, "assistant_prefix"),
            (2, 2002, "assistant"),
        ]


@pytest.mark.parametrize("sampled", [0, 3])
@pytest.mark.parametrize("outcome", ["raise", "cancel", "close"])
def test_owner_commit_then_failure_reconciles_exact_live_tail(sampled: int, outcome: str) -> None:
    async def scenario() -> None:
        native = FakeNative(
            script=lambda prompt, generated: 98,
            info=replace(FakeNative()._info, family="qwen", is_qwen4=True),
        )
        with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=native) as engine:
            session = engine.session()
            binder = MessageBinder(session)
            model = DS4Model(session, binder=binder, sampling=engine.sampling_defaults)
            messages: list[ModelMessage] = [ModelRequest([UserPromptPart("old history" * 100)])]
            committed = asyncio.Event()
            release = asyncio.Event()

            class FailingOwner(ContextStrategy):
                @override
                async def make_room(self, messages: list[ModelMessage], request: ContextRequest) -> RoomResult:
                    transcript = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
                    prefix = next(s for s in transcript.segments if s.kind == "assistant_prefix")
                    summary = ModelRequest([UserPromptPart("summary, not response")], metadata={"ds4.key": "summary"})
                    rendered = binder.render_request(summary, prefix=transcript.slice(0, binder.records()[0].start))
                    # An owner can supply the allowed unkeyed prefix/assistant/eos
                    # structure. EOS is not sampled output and must be removed
                    # by error/cancellation reconciliation, not counted backwards.
                    eos = TranscriptReplacement((FAKE_EOS,), (Segment(0, 1, "eos", {}),))
                    replacement = TranscriptReplacement.concat(rendered, transcript.slice(prefix.start), eos)
                    commit_compaction(binder, messages, [summary], replacement)
                    if outcome == "close":
                        committed.set()
                        await release.wait()
                        return RoomResult(True)
                    if outcome == "cancel":
                        raise asyncio.CancelledError
                    raise RuntimeError("owner failed after commit")

            model.context_owner = FailingOwner(binder)
            async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
                response_key = (stream.get().metadata or {})["ds4.key"]
                session._worker.call(
                    lambda e, n: setattr(n, "_ctx", len(session.transcript) + len(PREFIX) + sampled + 1)
                ).result()

                async def consume() -> None:
                    async for _ in stream:
                        pass

                if outcome == "close":
                    consumer = asyncio.create_task(consume())
                    try:
                        await asyncio.wait_for(committed.wait(), 5)
                        await asyncio.wait_for(stream.close_stream(), 5)
                    finally:
                        release.set()
                        await asyncio.wait_for(consumer, 5)
                else:
                    error = asyncio.CancelledError if outcome == "cancel" else RuntimeError
                    with pytest.raises(error):
                        await consume()
            response = stream.get()
            assert response.parts == ([TextPart("a" * sampled)] if sampled else [])
            assert response.usage.output_tokens == sampled
            assert (response.metadata or {})["ds4.key"] == response_key
            assert session.transcript.tokens[-sampled:] == (98,) * sampled if sampled else True
            assert session.transcript.segments[-1].kind == ("assistant" if sampled else "assistant_prefix")
            assert not any(s.kind == "eos" for s in session.transcript.segments)
            assert session._worker.call(lambda e, n: tuple(n.tokens())).result() == session.transcript.tokens
            assert binder.records()[-1].end == len(session.transcript)

    asyncio.run(scenario())


def test_missing_open_tail_rejects_instead_of_guessing() -> None:
    async def scenario() -> None:
        native = FakeNative(script=lambda prompt, generated: 98)
        with ds4py.open("f.gguf", ctx=1024, think=ThinkMode.NONE, native=native) as engine:
            session = engine.session()
            binder = MessageBinder(session)
            model = DS4Model(session, binder=binder, sampling=engine.sampling_defaults)

            class DestructiveOwner(ContextStrategy):
                @override
                async def make_room(self, messages: list[ModelMessage], request: ContextRequest) -> RoomResult:
                    system = session.transcript.segments[0].end
                    session.replace_transcript(
                        TranscriptReplacement(session.transcript.tokens, session.transcript.segments).slice(0, system),
                        reason="bad owner",
                    )
                    return RoomResult(True)

            model.context_owner = DestructiveOwner(binder)

            async def consume() -> None:
                async with model.request_stream(
                    [ModelRequest([UserPromptPart("go")])], None, ModelRequestParameters()
                ) as stream:
                    session._worker.call(lambda e, n: setattr(n, "_ctx", len(session.transcript) + 4)).result()
                    async for _ in stream:
                        pass

            with pytest.raises(EngineError, match="open assistant"):
                await consume()

    asyncio.run(scenario())


def test_summary_request_rendering_preserves_prefix_and_live_state() -> None:
    with ds4py.open(
        "f.gguf", native=FakeNative(info=replace(FakeNative()._info, family="qwen", is_qwen4=True))
    ) as engine:
        session = engine.session()
        binder = MessageBinder(session)
        session.append_tokens([42, 43], "system", {"source": "saved"})
        prefix = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
        request = ModelRequest([UserPromptPart("state")], metadata={"ds4.key": "summary"})
        before = (deepcopy(request), binder.records(), prefix, session.ctx_used)
        rendered = binder.render_request(request, prefix=prefix)
        assert rendered.tokens == (42, 43, *tokenize_rendered("<|im_start|>user\nstate<|im_end|>\n"))
        assert rendered.segments[0] == Segment(0, 2, "system", {"source": "saved"})
        assert [s.kind for s in rendered.segments] == ["system", "user"]
        assert rendered.segments[1].meta["key"] == "summary"
        assert (
            request,
            binder.records(),
            TranscriptReplacement(session.transcript.tokens, session.transcript.segments),
            session.ctx_used,
        ) == before


def test_repeated_custom_reserves_keep_exact_zero_sampled_boundary() -> None:
    async def scenario() -> None:
        sampled = 0

        def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
            nonlocal sampled
            sampled += 1
            return 98 if sampled <= 3 else FAKE_EOS

        native = FakeNative(script=script, info=replace(FakeNative()._info, family="qwen", is_qwen4=True))
        with ds4py.open("f.gguf", ctx=8192, think=ThinkMode.NONE, native=native) as engine:
            session = engine.session()
            binder = MessageBinder(session)
            model = DS4Model(session, binder=binder, sampling=engine.sampling_defaults)
            boundaries: list[int | None] = []

            class UnchangedOwner(ContextStrategy):
                @override
                async def make_room(self, messages: list[ModelMessage], request: ContextRequest) -> RoomResult:
                    boundaries.append(session.open_assistant_start())
                    # Change capacity, not the transcript or request-only view.
                    session._worker.call(lambda e, n: setattr(n, "_ctx", n.ctx() + 1)).result()
                    return RoomResult(True)

            model.context_owner = UnchangedOwner(binder)
            async with model.request_stream(
                [ModelRequest([UserPromptPart("go")])], None, ModelRequestParameters()
            ) as stream:
                start = len(session.transcript) + len(PREFIX)
                session._worker.call(lambda e, n: setattr(n, "_ctx", start + 1)).result()
                async for _ in stream:
                    pass
            assert boundaries == [start] * 4
            assert stream.get().parts == [TextPart("aaa")]
            assert stream.get().usage.output_tokens == 3
            assistant = next(s for s in session.transcript.segments if s.kind == "assistant")
            assert assistant.start == start
            assert assistant.meta["key"] == (stream.get().metadata or {})["ds4.key"]

    asyncio.run(scenario())


def test_tail_keeps_note_before_retained_message_and_pending_request() -> None:
    with ds4py.open("f.gguf", ctx=8192, native=FakeNative()) as engine:
        session = engine.session()
        binder = MessageBinder(session)
        messages: list[ModelMessage] = [
            ModelRequest([UserPromptPart("old" * 1000)]),
            ModelResponse([TextPart("old answer")]),
            ModelRequest([UserPromptPart("recent" * 200)]),
        ]
        binder.prepare(messages)
        note_start = binder.records()[2].start
        original = TranscriptReplacement(session.transcript.tokens, session.transcript.segments)
        note = TranscriptReplacement((101, 102), (Segment(0, 2, "note", {"text": "remember"}),))
        replacement = TranscriptReplacement.concat(original.slice(0, note_start), note, original.slice(note_start))
        commit_compaction(binder, messages, list(messages), replacement)
        pending = ModelRequest([UserPromptPart("pending")])
        messages.append(pending)
        tail = plan_tail(binder, messages)
        assert tail.cut_messages == 2
        assert tail.bottom_tokens == note_start
        assert tail.retained.tokens == (101, 102, *original.tokens[note_start:])
        assert tail.retained.segments[0] == Segment(0, 2, "note", {"text": "remember"})
        assert tail.messages[-1].parts == pending.parts
        assert session.transcript.tokens == replacement.tokens
