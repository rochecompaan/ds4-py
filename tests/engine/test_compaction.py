"""Private summaries must not become response text or change live history."""

import threading
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import pytest

import ds4py
from ds4py._native.fake import FAKE_EOS, FakeNative, FakeSession, reply
from ds4py._native.fake_chat import SPECIAL_TOKENS
from ds4py.engine.events import TextDelta, Token
from ds4py.engine.session import Session
from ds4py.errors import EngineError

# Independently copied from agent_compact_make_prompt at pinned ds4 5379c1c.
C_PROMPT = (
    "Internal ds4-agent context compaction request. This is not a user request.\n"
    "Write a durable task-state summary of the conversation so far. Preserve only facts that matter for continuing the work:\n"
    "- user goals, constraints, and preferences\n"
    "- files inspected or edited\n"
    "- commands run and important results\n"
    "- decisions, rejected approaches, known bugs, and pending next steps\n"
    "- reloadable bulky data with exact paths/ranges/commands when available\n\n"
    "Do not invent facts. Do not include generic narration. Do not include raw file contents unless they were essential to a conclusion.\n"
    "Do not solve unfinished tasks, calculate new results, or finish incomplete code in the summary. "
    "Record the user's requirements and what actually happened, leaving unfinished work pending.\n"
    "After the summary, stop. Do not continue the user task, do not call tools, and do not output thinking tags or DSML markup.\n"
    "Output only the compact summary.\n"
)


@pytest.mark.parametrize(("ctx", "expected"), [(1024, 256), (8192, 1024), (65536, 4096)])
def test_summary_budget_primitive(ctx: int, expected: int) -> None:
    from ds4py.engine.compaction import summary_budget

    assert summary_budget(ctx) == expected


@pytest.mark.parametrize(
    ("family", "opening", "closing"),
    [
        ("deepseek4", "<｜User｜>", "<｜Assistant｜></think>"),
        ("deepseek41", "<｜User｜>", "<｜Assistant｜></think>"),
        ("glm", "<|user|>", "<|assistant|><think></think>"),
        ("qwen", "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ],
)
def test_summary_suffix_exact_family_fixture(
    family: str, opening: str, closing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ds4py._native.fake_chat import tokenize_rendered
    from ds4py.engine.compaction import summary_suffix

    info = replace(
        FakeNative()._info,
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )
    with ds4py.open("f.gguf", ctx=4096, native=FakeNative(info=info)) as engine:
        prefix = [11, 12]
        contexts: list[tuple[int, ...]] = []
        assistant_prefix = engine._assistant_prefix

        def record_context(context: Sequence[int], mode: int) -> list[int]:
            assert mode == ds4py.ThinkMode.NONE
            contexts.append(tuple(context))
            return assistant_prefix(context, mode)

        monkeypatch.setattr(engine, "_assistant_prefix", record_context)
        expected = tuple(tokenize_rendered(opening + C_PROMPT + closing))
        assert summary_suffix(engine, prefix) == expected
        assert summary_suffix(engine, prefix, prompt="custom") == tuple(tokenize_rendered(opening + "custom" + closing))
        user_close = "<|im_end|>\n" if family == "qwen" else ""
        assert contexts == [
            (*prefix, *tokenize_rendered(opening + prompt + user_close)) for prompt in (C_PROMPT, "custom")
        ]
        assert prefix == [11, 12]


def state(session: Session) -> tuple[tuple[int, ...], bool, int]:
    return session._worker.call(
        lambda e, s: (tuple(s.tokens()), s.checkpoint_valid(), cast("FakeSession", s)._prompt_length)
    ).result()


def seed(session: Session, *, valid: bool = True) -> None:
    session.append_tokens([11, 12, 13, 14], "user", {"source": "original"})
    # Deliberately different from transcript: restoration must not sync history.
    session._worker.call(lambda e, s: s.sync([21, 22])).result()
    if not valid:
        session._worker.call(lambda e, s: s.invalidate()).result()


def test_summary_uses_only_caller_token_prefix() -> None:
    prompts: list[tuple[int, ...]] = []

    def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
        prompts.append(tuple(prompt))
        return reply(b"durable")(prompt, generated)

    with ds4py.open("f.gguf", ctx=4096, native=FakeNative(script=script)) as engine:
        session = engine.session()
        seed(session)
        before = (session.transcript.tokens, session.transcript.segments, state(session))
        result = session.prepare_summary(2)
        # DeepSeek4's native no-think opener is Assistant + </think>.
        suffix = [SPECIAL_TOKENS["<｜User｜>"]] + [b + 1 for b in C_PROMPT.encode()]
        suffix += [SPECIAL_TOKENS["<｜Assistant｜>"], SPECIAL_TOKENS["</think>"]]
        assert prompts[0] == (11, 12, *suffix)
        assert result.summary == "durable"
        assert result.tokens == tuple(b + 1 for b in b"durable")
        assert (session.transcript.tokens, session.transcript.segments, state(session)) == before


@pytest.mark.parametrize(
    "stop", [FAKE_EOS, SPECIAL_TOKENS["<think>"], SPECIAL_TOKENS["</think>"], SPECIAL_TOKENS["｜DSML｜"]]
)
def test_summary_stops_before_control_tokens(stop: int) -> None:
    def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
        return 98 if not generated else stop

    with ds4py.open("f.gguf", ctx=4096, native=FakeNative(script=script)) as engine:
        session = engine.session()
        seed(session)
        before = state(session)
        result = session.prepare_summary(2)
        assert (result.summary, result.tokens) == ("a", (98,))
        assert state(session) == before


@pytest.mark.parametrize(("ctx", "expected"), [(2048, 256), (4096, 512), (32768, 4096), (40000, 4096)])
def test_summary_obeys_c_budget(ctx: int, expected: int) -> None:
    with ds4py.open("f.gguf", ctx=ctx, native=FakeNative(script=lambda p, g: 98)) as engine:
        session = engine.session()
        seed(session)
        before = state(session)
        result = session.prepare_summary(2)
        assert result.summary == "a" * expected
        assert result.tokens == (98,) * expected
        assert state(session) == before


@pytest.mark.parametrize("leak", [b"safe<", "safe<｜DSML｜tool_calls>bad".encode(), "safe｜DSML｜bad".encode()])
def test_summary_does_not_leak_dsml(leak: bytes) -> None:
    def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
        if len(generated) == len(leak) and leak == b"safe<":
            return SPECIAL_TOKENS["｜DSML｜"]
        return reply(leak)(prompt, generated)

    with ds4py.open("f.gguf", ctx=4096, native=FakeNative(script=script)) as engine:
        session = engine.session()
        seed(session)
        observed: list[object] = []
        session.on_event(lambda e: observed.append(e.payload.get("event")))
        result = session.prepare_summary(2)
        assert (result.summary, result.tokens) == ("safe", (116, 98, 103, 102))
        assert b"DSML" not in b"".join(getattr(e, "text_bytes", b"") for e in observed)


def test_summary_masks_dedicated_tool_tokens_before_argmax(monkeypatch: pytest.MonkeyPatch) -> None:
    def distribution(self: FakeSession) -> list[float]:
        logits = [0.0] * self._engine.info().vocab_size
        generated = len(self._tokens) - self._prompt_length
        logits[FAKE_EOS if generated else 121] = 1.0
        if not generated:
            logits[SPECIAL_TOKENS["<tool_call>"]] = 2.0
        if self._logits is None:
            self._logits = logits
        return self._logits

    monkeypatch.setattr(FakeSession, "_distribution", distribution)
    with ds4py.open("f.gguf", ctx=4096, native=FakeNative()) as engine:
        session = engine.session()
        seed(session)
        result = session.prepare_summary(2)
        assert (result.summary, result.tokens) == ("x", (121,))
        assert session._worker.call(lambda e, s: cast("FakeSession", s).samples).result() == []


@pytest.mark.parametrize("phase", ["sync", "argmax", "eval"])
def test_summary_failure_restores_native_and_history(phase: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with ds4py.open("f.gguf", ctx=4096, native=FakeNative()) as engine:
        session = engine.session()
        seed(session)
        before = (session.transcript.tokens, session.transcript.segments, state(session))

        original_sync, original_eval = FakeSession.sync, FakeSession.eval

        def fail(native: FakeSession, *args: object, **kwargs: object) -> None:
            if phase == "sync":
                original_sync(native, [99, 100])
            elif phase == "eval":
                original_eval(native, 98)
            raise EngineError("injected failure")

        monkeypatch.setattr(FakeSession, phase, fail)
        with pytest.raises(EngineError, match="injected failure"):
            session.prepare_summary(2)
        assert (session.transcript.tokens, session.transcript.segments, state(session)) == before


@pytest.mark.parametrize("during_sync", [True, False])
def test_summary_cancellation_restores_native_and_history(during_sync: bool) -> None:
    cancel = threading.Event()

    def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
        cancel.set()
        return 98

    with ds4py.open("f.gguf", ctx=4096, cancel=cancel, native=FakeNative(script=script)) as engine:
        session = engine.session()
        seed(session)
        before = (session.transcript.tokens, session.transcript.segments, state(session))
        if during_sync:
            cancel.set()
        with pytest.raises(EngineError, match="interrupted"):
            session.prepare_summary(2)
        assert (session.transcript.tokens, session.transcript.segments, state(session)) == before
        assert not cancel.is_set()


@pytest.mark.parametrize("empty", [False, True])
def test_summary_restores_empty_or_invalid_native_state(empty: bool) -> None:
    with ds4py.open("f.gguf", ctx=4096, native=FakeNative()) as engine:
        session = engine.session()
        if not empty:
            seed(session, valid=False)
        before = state(session)
        assert session.prepare_summary(0).summary == "ok"
        assert state(session) == before


def test_summary_events_are_not_response_text(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models import ModelRequestParameters

    from ds4py.engine.events import SummaryToken
    from ds4py.model import DS4Model, DS4StreamedResponse

    with ds4py.open("f.gguf", ctx=4096, native=FakeNative()) as engine:
        session = engine.session()
        observed: list[object] = []
        session.on_event(lambda e: observed.append(e.payload.get("event")))
        session.prepare_summary(0)
        assert [e for e in observed if isinstance(e, SummaryToken)] == [
            SummaryToken(112, b"o"),
            SummaryToken(108, b"k"),
        ]
        assert not any(isinstance(e, (Token, TextDelta)) for e in observed)
        model = DS4Model(session, sampling=engine.sampling_defaults)
        response = DS4StreamedResponse(model, [], ModelRequestParameters(), engine.sampling_defaults, 10, ())

        def unexpected_delta(*args: object, **kwargs: object) -> None:
            pytest.fail("summary called handle_text_delta")

        monkeypatch.setattr(type(response._parts_manager), "handle_text_delta", unexpected_delta)
        for event in observed:
            if isinstance(event, SummaryToken):
                assert response._dispatch(event) == []
        assert response.get().parts == []
        assert response.usage.output_tokens == 0
        response._round.close()


@pytest.mark.parametrize("bottom", [-1, 5])
def test_summary_rejects_out_of_bounds_prefix(bottom: int) -> None:
    with ds4py.open("f.gguf", ctx=4096, native=FakeNative()) as engine:
        session = engine.session()
        seed(session)
        before = state(session)
        with pytest.raises(ValueError, match="outside the transcript"):
            session.prepare_summary(bottom)
        assert state(session) == before


def test_summary_rejects_insufficient_room_without_shortening_prefix() -> None:
    with ds4py.open("f.gguf", ctx=1100, native=FakeNative()) as engine:
        session = engine.session()
        seed(session)
        before = state(session)
        with pytest.raises(EngineError, match="context"):
            session.prepare_summary(4)
        assert state(session) == before


def test_empty_summary_is_an_error_and_restores_native() -> None:
    with ds4py.open("f.gguf", ctx=4096, native=FakeNative(script=lambda p, g: FAKE_EOS)) as engine:
        session = engine.session()
        seed(session)
        before = state(session)
        with pytest.raises(EngineError, match="empty"):
            session.prepare_summary(2)
        assert state(session) == before


def test_summary_is_allowed_only_at_reserve_boundary() -> None:
    from ds4py.engine.events import ReserveReached, Stopped

    with ds4py.open("f.gguf", ctx=2048, native=FakeNative(script=lambda p, g: 98)) as engine:
        session = engine.session()
        session.append_user("hi")
        round_ = session.run_round()
        try:
            # A sync event is an ordinary active-round boundary, not a reserve.
            next(round_)
            with pytest.raises(EngineError, match="round is open"):
                session.prepare_summary(0)
            pause = next(event for event in round_ if isinstance(event, ReserveReached))
            before = (session.transcript.tokens, session.transcript.segments, state(session))
            with session.reserve_window(pause):
                assert session.prepare_summary(0).summary == "a" * 256
            assert (session.transcript.tokens, session.transcript.segments, state(session)) == before
            with pytest.raises(EngineError, match="round is open"):
                session.append_tokens([99], "user")
            pause.fail()
            assert list(round_) == [Stopped("context_exhausted")]
        finally:
            round_.close()


@pytest.mark.parametrize("room", [255, 256, 300, 512, 513])
def test_summary_budget_is_bounded_by_remaining_room(room: int) -> None:
    with ds4py.open("f.gguf", ctx=4096, native=FakeNative(script=lambda p, g: 98)) as engine:
        session = engine.session()
        # Prefix + user marker + exact C prompt + two-token opener + one spare slot.
        prefix = 4096 - len(C_PROMPT.encode()) - 3 - 1 - room
        session.append_tokens([11] * prefix, "user")
        before = (session.transcript.tokens, session.transcript.segments, state(session))
        if room < 256:
            with pytest.raises(EngineError, match="not enough context"):
                session.prepare_summary(prefix)
        else:
            result = session.prepare_summary(prefix)
            assert result.summary == "a" * min(room, 512)
            assert result.tokens == (98,) * min(room, 512)
        assert (session.transcript.tokens, session.transcript.segments, state(session)) == before


@pytest.mark.parametrize("literal", [False, True])
def test_summary_retains_mixed_text_token_id_at_dsml_boundary(literal: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from ds4py._native.fake import FakeEngine
    from ds4py.engine.events import SummaryToken

    original = FakeEngine.token_text

    def text(engine: FakeEngine, token: int) -> bytes:
        if token == 121:
            return "safe<｜DSML｜bad".encode() if literal else b"safe<"
        return original(engine, token)

    def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
        return SPECIAL_TOKENS["｜DSML｜"] if generated else 121

    monkeypatch.setattr(FakeEngine, "token_text", text)
    with ds4py.open("f.gguf", ctx=4096, native=FakeNative(script=script)) as engine:
        session = engine.session()
        observed: list[object] = []
        session.on_event(lambda e: observed.append(e.payload.get("event")))
        result = session.prepare_summary(0)
        assert (result.summary, result.tokens, len(result.tokens)) == ("safe", (121,), 1)
        assert observed == [SummaryToken(121, b"safe")]


def test_summary_listener_closing_outer_round_does_not_resurrect_it() -> None:
    from ds4py.engine.events import ReserveReached, SummaryToken

    with ds4py.open("f.gguf", ctx=2048, native=FakeNative(script=lambda p, g: 98)) as engine:
        session = engine.session()
        session.append_user("hi")
        outer = session.run_round()
        closed: list[tuple[bool, bool]] = []
        try:
            pause = next(event for event in outer if isinstance(event, ReserveReached))

            def close_outer(event: object) -> None:
                outer.close()
                closed.append((session._operations.active_round, session._operations.reserve_event is not None))

            session.on_event(close_outer, kinds=(SummaryToken.__name__,))
            with session.reserve_window(pause):
                assert session.prepare_summary(0).summary == "a" * 256
            assert closed and all(flags == (False, False) for flags in closed)
            # Summary cleanup must not reclaim the expired outer-round flags.
            session.append_tokens([99], "user")
            assert (session._operations.active_round, session._operations.reserve_event is not None) == (False, False)
        finally:
            outer.close()


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel_sync", "cancel_decode"])
def test_summary_restores_full_invalid_native_state(outcome: str) -> None:
    cancel = threading.Event()

    def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
        if outcome == "failure":
            raise EngineError("injected full-state failure")
        if outcome == "cancel_decode":
            cancel.set()
        return reply(b"ok")(prompt, generated)

    with ds4py.open("f.gguf", ctx=2048, cancel=cancel, native=FakeNative(script=script)) as engine:
        session = engine.session()
        session.append_tokens([11, 12], "user")

        def fill(native: FakeSession) -> None:
            native.sync([31] * 2047)
            native.eval(32)
            native.invalidate()

        session._worker.call(lambda e, s: fill(cast("FakeSession", s))).result()
        before = (session.transcript.tokens, session.transcript.segments, state(session))
        if outcome == "cancel_sync":
            cancel.set()
        if outcome == "success":
            assert session.prepare_summary(0).summary == "ok"
        else:
            message = "injected full-state failure" if outcome == "failure" else "interrupted"
            with pytest.raises(EngineError, match=message):
                session.prepare_summary(0)
        assert (session.transcript.tokens, session.transcript.segments, state(session)) == before
        assert not cancel.is_set()
        # The worker remains usable, rather than being poisoned by cleanup.
        session._worker.call(lambda e, s: s.sync([41, 42])).result()
        assert state(session)[:2] == ((41, 42), True)


@pytest.mark.parametrize(
    ("family", "marker"),
    [
        ("deepseek4", "<tool_call>"),
        ("deepseek4", "</tool_call>"),
        ("deepseek4", "<arg_key>"),
        ("deepseek4", "</arg_key>"),
        ("deepseek4", "<arg_value>"),
        ("deepseek4", "</arg_value>"),
        ("deepseek4", "<tool_response>"),
        ("deepseek4", "</tool_response>"),
        ("qwen", "<function="),
        ("qwen", "</function>"),
        ("qwen", "<parameter="),
        ("qwen", "</parameter>"),
    ],
)
@pytest.mark.parametrize("layout", ["bytes", "mixed", "split"])
def test_summary_sanitizes_literal_tool_delimiters(
    family: str, marker: str, layout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from ds4py._native.fake import FakeEngine
    from ds4py.engine.events import SummaryToken

    original = FakeEngine.token_text
    # Exercise every interior split when two ordinary tokens straddle a marker.
    splits = range(1, len(marker)) if layout == "split" else (0,)
    for split in splits:
        chunks = (b"safe" + marker[:split].encode(), marker[split:].encode() + b"bad")

        def text(engine: FakeEngine, token: int, chunks: tuple[bytes, bytes] = chunks) -> bytes:
            if layout == "mixed" and token == 121:
                return b"safe" + marker.encode() + b"bad"
            if layout == "split" and token in (121, 122):
                return chunks[token - 121]
            return original(engine, token)

        def script(prompt: Sequence[int], generated: Sequence[int]) -> int:
            if layout == "bytes":
                return reply(b"safe" + marker.encode() + b"bad")(prompt, generated)
            if layout == "mixed":
                return 121 if not generated else FAKE_EOS
            return (121, 122, FAKE_EOS)[min(len(generated), 2)]

        monkeypatch.setattr(FakeEngine, "token_text", text)
        info = replace(FakeNative()._info, family=family, is_qwen4=family == "qwen")
        with ds4py.open("f.gguf", ctx=4096, native=FakeNative(script=script, info=info)) as engine:
            session = engine.session()
            observed: list[object] = []
            session.on_event(lambda e, sink=observed: sink.append(e.payload.get("event")))
            result = session.prepare_summary(0)
            expected_ids = (116, 98, 103, 102) if layout == "bytes" else (121,)
            assert (result.summary, result.tokens) == ("safe", expected_ids)
            events = [event for event in observed if isinstance(event, SummaryToken)]
            assert tuple(event.id for event in events) == expected_ids
            assert b"".join(event.text_bytes for event in events) == b"safe"
