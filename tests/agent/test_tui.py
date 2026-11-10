import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from test_tui_commands import FakeRuntime, make_tui

from ds4py.engine.events import Prefill
from ds4py.engine.lifecycle import SessionEvent
from ds4py.errors import EngineError
from ds4py.tui import TUI
from ds4py.tui_render import FooterModel, StreamRenderer, markdown_fragments


def test_dispatch_runs_one_prompt_and_preserves_pasted_newlines() -> None:
    runtime: Any = FakeRuntime()
    tui = make_tui(runtime)

    asyncio.run(tui.dispatch("first line\nsecond line"))

    assert runtime.calls == [("run", "first line\nsecond line")]


def test_command_error_is_reported_and_loop_remains_usable() -> None:
    runtime: Any = FakeRuntime()
    tui = make_tui(runtime)
    messages = []
    cast("Any", tui).show_message = messages.append

    async def scenario() -> None:
        await tui.dispatch("/switch bad")
        await tui.dispatch("/new")

    asyncio.run(scenario())
    assert messages[0] == "error: unknown save"
    assert messages[1] == "new session new-session"
    assert not tui.exiting


def test_prompt_loop_accepts_pipe_input_and_exit() -> None:
    runtime: Any = FakeRuntime()
    with create_pipe_input() as input_:
        input_.send_text("hello\n/exit\n")
        tui = TUI(runtime, input=input_, output=DummyOutput())
        asyncio.run(asyncio.wait_for(tui.run(), timeout=2))

    assert ("run", "hello") in runtime.calls
    assert tui.exiting


def test_model_question_uses_active_prompt_and_returns_typed_answer() -> None:
    from pydantic_ai_harness.ask_user import (
        AskUserRequest,
        AskUserResponse,
        Question,
        QuestionOption,
    )

    runtime: Any = FakeRuntime()
    request = AskUserRequest(
        questions=(
            Question(
                header="next",
                question="What next?",
                options=(QuestionOption(label="one"), QuestionOption(label="two")),
            ),
        )
    )
    with create_pipe_input() as input_:
        input_.send_text("ask\n2\n/exit\n")
        tui = TUI(runtime, input=input_, output=DummyOutput())

        async def ask_during_run(prompt: str) -> None:
            runtime.calls.append(("run", prompt))
            runtime.response = await tui.ask(request)

        runtime.run = ask_during_run
        asyncio.run(asyncio.wait_for(tui.run(), timeout=2))

    assert isinstance(runtime.response, AskUserResponse)
    assert runtime.response.answers[0].selected == ("two",)
    assert runtime.calls == [("run", "ask")]


@pytest.mark.parametrize(
    ("command", "feedback", "command_calls"),
    [
        ("/compact", "error: a run is already active", [("compact",)]),
        ("  /compact  ", "error: a run is already active", [("compact",)]),
        ("/compact extra", "error: usage: /compact", []),
        ('  /probe "two words"', "probe: two words", []),
        ('"/probe" "two words"', "probe: two words", []),
    ],
)
def test_immediate_command_does_not_consume_a_pending_question_answer(
    command: str, feedback: str, command_calls: list[tuple[str]]
) -> None:
    from pydantic_ai_harness.ask_user import AskUserRequest, AskUserResponse, Question, QuestionOption

    async def scenario() -> None:
        question_pending = asyncio.Event()
        feedback_received = asyncio.Event()
        question_done = asyncio.Event()
        responses: list[AskUserResponse] = []
        runtime = FakeRuntime()
        request = AskUserRequest(
            questions=(
                Question(
                    header="next",
                    question="What next?",
                    options=(QuestionOption(label="one"), QuestionOption(label="two")),
                ),
            )
        )
        with create_pipe_input() as input_:
            tui = TUI(cast("Any", runtime), input=input_, output=DummyOutput())
            messages: list[str] = []

            def show(text: object) -> None:
                messages.append(str(text))
                if str(text).startswith(("error:", "Invalid choice", "probe:")):
                    feedback_received.set()

            async def ask_during_run(prompt: str) -> None:
                runtime.calls.append(("run", prompt))
                question_pending.set()
                responses.append(await tui.ask(request))
                question_done.set()

            async def compact() -> None:
                runtime.calls.append(("compact",))
                raise EngineError("a run is already active")

            tui.commands.register(
                "probe", lambda text: f"probe: {text}", description="Probe", immediate_while_running=True
            )
            cast("Any", runtime).run = ask_during_run
            cast("Any", runtime).compact = compact
            cast("Any", tui).show_message = show
            task = asyncio.create_task(tui.run("ask"))
            try:
                await asyncio.wait_for(question_pending.wait(), 2)
                messages.clear()
                input_.send_text(f"{command}\n")
                await asyncio.wait_for(feedback_received.wait(), 2)
                assert messages == [feedback]
                assert runtime.calls == [("run", "ask"), *command_calls]
                assert tui._answers.empty()
                assert not question_done.is_set()
                assert responses == []
                assert tui.footer.status != "compacting"
                input_.send_text("1\n")
                await asyncio.wait_for(question_done.wait(), 2)
                assert responses[0].answers[0].selected == ("one",)
                assert not responses[0].cancelled
            finally:
                if not question_done.is_set():
                    input_.send_text("1\n")
                input_.send_text("/exit\n")
                await asyncio.wait_for(task, 2)

    asyncio.run(scenario())


@pytest.mark.parametrize("answer", ["/cancel", "/unknown", "/new", '/compact "unfinished'])
def test_non_immediate_slash_input_keeps_question_answer_handling(answer: str) -> None:
    from pydantic_ai_harness.ask_user import AskUserRequest, AskUserResponse, Question, QuestionOption

    runtime: Any = FakeRuntime()
    request = AskUserRequest(
        questions=(
            Question(
                header="next", question="What next?", options=(QuestionOption(label="one"), QuestionOption(label="two"))
            ),
        )
    )
    responses: list[AskUserResponse] = []
    messages: list[object] = []
    with create_pipe_input() as input_:
        tui = TUI(runtime, input=input_, output=DummyOutput())
        cast("Any", tui).show_message = messages.append

        async def ask_during_run(prompt: str) -> None:
            runtime.calls.append(("run", prompt))
            responses.append(await tui.ask(request))
            tui.exiting = True

        runtime.run = ask_during_run
        input_.send_text(f"{answer}\n" + ("" if answer == "/cancel" else "1\n"))
        asyncio.run(asyncio.wait_for(tui.run("ask"), 2))

    assert runtime.calls == [("run", "ask")]
    assert responses[0].cancelled is (answer == "/cancel")
    if answer != "/cancel":
        assert responses[0].answers[0].selected == ("one",)
        assert messages[-1] == "Invalid choice; try again or enter /cancel."


def test_unmarked_command_stays_queued_until_the_active_run_finishes() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        finish = asyncio.Event()
        immediate_done = asyncio.Event()
        queued_done = asyncio.Event()
        calls: list[str] = []
        runtime: Any = FakeRuntime()

        async def run(prompt: str) -> None:
            calls.append(prompt)
            started.set()
            await finish.wait()

        def later() -> None:
            calls.append("later")
            queued_done.set()

        runtime.run = run
        with create_pipe_input() as input_:
            tui = TUI(runtime, input=input_, output=DummyOutput())
            tui.commands.register("later", later, description="Queued command")
            tui.commands.register(
                "probe", immediate_done.set, description="Immediate barrier", immediate_while_running=True
            )
            task = asyncio.create_task(tui.run("hello"))
            try:
                await asyncio.wait_for(started.wait(), 2)
                input_.send_text("/later\n/probe\n")
                await asyncio.wait_for(immediate_done.wait(), 2)
                assert calls == ["hello"]
                assert not queued_done.is_set()
                finish.set()
                await asyncio.wait_for(queued_done.wait(), 2)
                assert calls == ["hello", "later"]
            finally:
                finish.set()
                input_.send_text("/exit\n")
                await asyncio.wait_for(task, 2)

    asyncio.run(scenario())


def test_prompt_loop_can_run_an_initial_prompt_inside_the_event_loop() -> None:
    runtime: Any = FakeRuntime()
    with create_pipe_input() as input_:
        input_.send_text("/exit\n")
        tui = TUI(runtime, input=input_, output=DummyOutput())
        asyncio.run(asyncio.wait_for(tui.run("initial"), timeout=2))

    assert runtime.calls[0] == ("run", "initial")
    assert tui.exiting


def test_enter_submits_and_alt_enter_inserts_newline() -> None:
    tui = make_tui()
    enter = tui.key_bindings.get_bindings_for_keys((Keys.Enter,))[0]
    alt_enter = tui.key_bindings.get_bindings_for_keys((Keys.Escape, Keys.Enter))[0]
    submitted = []
    inserted = []
    cast("Any", enter.handler)(
        SimpleNamespace(
            current_buffer=SimpleNamespace(
                validate_and_handle=lambda: submitted.append(True),
            )
        )
    )
    cast("Any", alt_enter.handler)(
        SimpleNamespace(
            current_buffer=SimpleNamespace(
                insert_text=inserted.append,
            )
        )
    )

    assert submitted == [True]
    assert inserted == ["\n"]


def test_ctrl_c_cancels_without_touching_partial_text_and_ctrl_d_exits() -> None:
    runtime: Any = FakeRuntime()
    tui = make_tui(runtime)
    cancel = tui.key_bindings.get_bindings_for_keys((Keys.ControlC,))[0]
    assert cast("Any", cancel.handler)(SimpleNamespace()) is None
    assert runtime.calls == [("cancel",)]

    exits = []

    def record_exit(exception: BaseException) -> None:
        exits.append(exception)

    ctrl_d = tui.key_bindings.get_bindings_for_keys((Keys.ControlD,))[0]
    cast("Any", ctrl_d.handler)(
        SimpleNamespace(
            app=SimpleNamespace(
                exit=record_exit,
            )
        )
    )
    assert len(exits) == 1
    assert isinstance(exits[0], EOFError)


def test_footer_distinguishes_prefill_from_token_generation() -> None:
    footer = FooterModel(cast("Any", FakeRuntime()))
    try:
        footer.handle_event(SessionEvent("run_start", "session", {}))
        assert footer.status == "running"
        footer.handle_event(
            SessionEvent(
                "Prefill",
                "session",
                {
                    "event": Prefill(done=2, total=2, cached_prefix=0, tps=10.0),
                },
            )
        )
        assert footer.status == "running"
        footer.handle_event(SessionEvent("Token", "session", {}))
        assert footer.status == "streaming"
        footer.handle_event(SessionEvent("Token", "session", {}))
        assert footer.status == "streaming"
        footer.handle_event(SessionEvent("run_end", "session", {}))
        assert footer.status == "idle"
    finally:
        footer.close()


def fragment_text(fragments: Any) -> str:
    return "".join(str(text) for _style, text in fragments)


def test_text_and_thinking_deltas_have_distinct_styles() -> None:
    renderer = StreamRenderer(DummyOutput())
    text = renderer.handle_event(
        PartDeltaEvent(
            index=0,
            delta=TextPartDelta("visible"),
        )
    )
    thinking = renderer.handle_event(
        PartDeltaEvent(
            index=0,
            delta=ThinkingPartDelta(content_delta="private"),
        )
    )

    assert fragment_text(text) == "visible"
    assert fragment_text(thinking) == "private"
    assert text[0][0] != thinking[0][0]
    assert "dim" in thinking[0][0]


def test_tool_previews_are_one_line_and_ellipsized() -> None:
    renderer = StreamRenderer(DummyOutput())
    long_value = "alpha\nbeta " * 40
    events = [
        FunctionToolCallEvent(
            part=ToolCallPart(
                tool_name="write_file",
                args={"path": "a.txt", "content": long_value},
            )
        ),
        FunctionToolResultEvent(
            part=ToolReturnPart(
                tool_name="write_file",
                content=long_value,
                tool_call_id="call-1",
            )
        ),
    ]

    for event in events:
        text = fragment_text(renderer.handle_event(event))
        assert "\n" not in text
        assert "…" in text
        assert len(text) <= 120
        assert "write_file" in text


def test_stream_renderer_applies_markdown_to_text_deltas() -> None:
    renderer = StreamRenderer(DummyOutput())
    fragments: Any = renderer.handle_event(
        PartDeltaEvent(
            index=0,
            delta=TextPartDelta("**bold** and `code`"),
        )
    )

    assert fragment_text(fragments) == "bold and code"
    assert any("bold" in style for style, _text in fragments)
    assert any("code" in style for style, _text in fragments)


def test_markdown_becomes_styled_prompt_toolkit_fragments() -> None:
    fragments: Any = markdown_fragments("# Heading\n**bold** and *emphasis* with `code`")
    styles = [style for style, _text in fragments]
    text = fragment_text(fragments)

    assert any("heading" in style for style in styles)
    assert any("bold" in style for style in styles)
    assert any("italic" in style for style in styles)
    assert any("code" in style for style in styles)
    assert "Heading" in text
    assert "**" not in text
    assert "`" not in text
