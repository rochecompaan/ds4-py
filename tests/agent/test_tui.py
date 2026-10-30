import asyncio
from types import SimpleNamespace
from typing import Any, cast

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

from ds4py.tui import TUI
from ds4py.tui_render import StreamRenderer, markdown_fragments


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
