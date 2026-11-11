"""Real harness boundaries must retain compaction and fit grouped tool results."""

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import override

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart
from pydantic_ai_harness import HarnessDeprecationWarning
from test_compaction_persistence import dump
from test_context_strategy import fake_engine
from test_runtime import fake_engine as runtime_engine
from test_runtime import fixed_clock, make_model

from ds4py._native.fake import FAKE_EOS
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.capabilities.compaction import LLMCompaction
from ds4py.capabilities.context import ContextRequest, RoomResult
from ds4py.engine.events import Synced
from ds4py.engine.lifecycle import SessionEvent
from ds4py.errors import EngineError
from ds4py.runtime import AgentRuntime
from ds4py.store import SessionStore


@pytest.mark.parametrize("termination", ["success", "interrupted", "error"])
def test_runtime_reserve_history_survives_continuation_save_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, termination: str
) -> None:
    # A request-only commit resurrects the 14k user turn and saves absent keys.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    marker = tuple(tokenize_rendered("Internal ds4-agent context compaction request."))
    stage = "first"
    sampled = 0
    compacted = False
    interrupt: Callable[[], None] | None = None

    def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
        nonlocal sampled
        if any(tuple(prompt[i : i + len(marker)]) == marker for i in range(len(prompt))):
            text = b"durable state"
            return text[len(generated)] + 1 if len(generated) < len(text) else FAKE_EOS
        if compacted and stage == "reserve":
            if termination == "error":
                raise EngineError("resumed decode failed")
            if termination == "interrupted":
                assert interrupt is not None
                interrupt()
        text = {"first": b"first", "reserve": b"before after", "continue": b"continued"}[stage]
        token = text[sampled] + 1 if sampled < len(text) else FAKE_EOS
        sampled += 1
        return token

    async def scenario() -> None:
        nonlocal stage, sampled, compacted, interrupt
        with runtime_engine(script=script) as engine:
            model = make_model(engine)
            with pytest.warns(HarnessDeprecationWarning, match="deprecated and ignored: commands/paths start"):
                runtime = AgentRuntime(model, tmp_path, store=SessionStore(tmp_path / "store"), clock=fixed_clock)
            session = model.session
            interrupt = session.cancel
            armed = False

            def force_reserve(notification: SessionEvent) -> None:
                nonlocal armed, compacted
                event = notification.payload.get("event")
                if isinstance(event, Synced) and stage == "reserve" and not armed:
                    armed = True
                    session._worker.call(lambda e, n: setattr(n, "_ctx", event.pos + 8)).result()
                if notification.kind == "compacted":
                    compacted = True
                    session._worker.call(lambda e, n: setattr(n, "_ctx", 65536)).result()

            runtime.on_event(force_reserve, kinds={"Synced", "compacted"})
            try:
                first = await runtime.run("x" * 14000)
                original = dump(runtime.messages)
                stage, sampled = "reserve", 0
                if termination == "error":
                    with pytest.raises(EngineError, match="resumed decode failed"):
                        await runtime.run("r" * 1500)
                else:
                    result = await runtime.run("r" * 1500)
                    if termination == "success":
                        assert result.output == "before after"
                        assert result.usage.output_tokens == 25  # 12 response + 13 private summary.
                    else:
                        assert result.output == "before a"
                        assert result.usage.output_tokens == 21  # 8 partial + 13 summary.
                assert compacted
                saved = await runtime.save()
                messages, tokens = dump(runtime.messages), session.transcript.tokens
                records = model.binder.records()
                await runtime.new()
                await runtime.switch(saved.session_id)
                assert dump(runtime.messages) == messages
                assert session.transcript.tokens == tokens
                assert model.binder.records() == records
                assert session.plan().to_prefill == 0
                users = [p.content for m in runtime.messages for p in m.parts if isinstance(p, UserPromptPart)]
                assert users == ["durable state", "r" * 1500]
                assert [(m.metadata or {}).get("ds4.key") for m in runtime.messages] == [
                    record.key for record in records
                ]
                responses = [m for m in runtime.messages if isinstance(m, ModelResponse)]
                assert len(responses) == 1
                expected = {"success": "before after", "interrupted": "before a", "error": "before "}
                assert responses[0].parts == [TextPart(expected[termination])]
                stage, sampled = "continue", 0
                assert (await runtime.run("continue")).output == "continued"
                assert session.transcript.tokens[: len(tokens)] == tokens
                assert "x" * 14000 not in [
                    p.content for m in runtime.messages for p in m.parts if isinstance(p, UserPromptPart)
                ]
                continued = await runtime.save()
                expected = dump(runtime.messages)
                await runtime.new()
                await runtime.switch(continued.session_id)
                assert dump(runtime.messages) == expected
                old = await runtime._holder().current.latest_snapshot(run_id=first.run_id)
                assert old is not None and dump(old.messages) == original
            finally:
                await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("result_size", "needs_error", "needed", "old_history"),
    [
        (6000, True, 4813, True),
        (3900, True, 613, True),
        (3800, False, 413, True),
        (3594, False, 1, True),
        (3500, False, -187, True),
        (6000, True, 4199, False),
    ],
)
def test_combined_glm_tool_results_recover_before_native_sync(
    result_size: int, needs_error: bool, needed: int, old_history: bool
) -> None:
    # 6000-byte results each fit alone, but not together; 3800 also tests
    # combined framing/prefix overhead, rather than only result body sizes.
    calls = tuple(tokenize_rendered("<tool_call>read_big</tool_call><tool_call>read_big</tool_call>"))
    marker = tuple(tokenize_rendered("Internal ds4-agent context compaction request."))
    phase = 0

    def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
        nonlocal phase
        if any(tuple(prompt[i : i + len(marker)]) == marker for i in range(len(prompt))):
            text = b"durable state"
            return text[len(generated)] + 1 if len(generated) < len(text) else FAKE_EOS
        if phase == 0:
            if len(generated) < len(calls):
                return calls[len(generated)]
            phase = 1
            return FAKE_EOS
        text = b"done"
        return text[len(generated)] + 1 if len(generated) < len(text) else FAKE_EOS

    async def scenario() -> None:
        with fake_engine(family="glm", script=script) as engine:
            model = make_model(engine)
            requests: list[ContextRequest] = []

            class RecordingCompactor(LLMCompaction):
                @override
                async def make_room(self, messages: list[ModelMessage], request: ContextRequest) -> RoomResult:
                    requests.append(request)
                    return await super().make_room(messages, request)

            owner = RecordingCompactor(model.binder)
            model.context_owner = owner
            agent = Agent(model, capabilities=[owner])

            @agent.tool_plain
            def read_big() -> str:
                return "z" * result_size

            # Keep a bound old turn so aggregate compaction has something to drop.
            history = [ModelRequest([UserPromptPart("old" * 200)]), ModelResponse([TextPart("old answer")])]
            reasons = []
            model.session.on_event(lambda e: reasons.append(e.payload["reason"]), kinds={"compacted"})
            result = await agent.run("go", message_history=history if old_history else [])
            assert result.output == "done"
            returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
            assert len(returns) == 2
            assert reasons == (["tool_result"] if needed > 0 and old_history else [])
            # Native prefix 944 (330 without old turn) + bodies + headers 2*26 + separator 2
            # + grouped wrapper 3 + assistant prefix 3 + generation reserve 1.
            assert requests == ([ContextRequest(needed, "tool_result")] if needed > 0 else [])
            assert [p.content for p in returns] == [
                "z" * result_size,
                "Tool error: tool result still does not fit" if needs_error else "z" * result_size,
            ]
            assert result.usage.output_tokens == len(calls) + (13 if needed > 0 and old_history else 0) + 4
            assert len(model.session.transcript) < 8192

    asyncio.run(scenario())
