"""Reserve compaction keeps one native assistant and one public response alive."""

import asyncio
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

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
from ds4py.engine.events import SummaryToken, Synced
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.stream import RoundStream
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
        assert tail.messages == []
        assert tail.bottom == start
        assert tail.open_start == start
        assert tail.tokens == (260, 267, *([98] * 2000))
        assert [(s.start, s.end, s.kind) for s in tail.segments] == [(0, 2, "assistant_prefix"), (2, 2002, "assistant")]
