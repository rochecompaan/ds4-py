"""Interactive prompt-toolkit shell backed only by ``AgentRuntime``."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterable

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.output import Output
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import PromptSession, print_formatted_text
from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserRequest, AskUserResponse

from ds4py.engine.lifecycle import SessionEvent
from ds4py.question_answers import selected_labels
from ds4py.runtime import AgentRuntime
from ds4py.tui_commands import SlashCommandRegistry, install_builtin_commands
from ds4py.tui_render import FooterModel, StreamRenderer


class _SlashCompleter(Completer):
    def __init__(self, registry: SlashCommandRegistry) -> None:
        self.registry = registry

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        yield from self.registry.completions(document.text_before_cursor)


class TUI:
    """Own prompt input, event rendering, commands, and the live footer."""

    def __init__(
        self,
        runtime: AgentRuntime,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.runtime = runtime
        self.output = output
        self.renderer = StreamRenderer(output)
        self.footer = FooterModel(runtime)
        self.commands = SlashCommandRegistry()
        self.exiting = False
        self.key_bindings = self._keys()
        self.session = PromptSession(
            input=input,
            output=output,
            completer=_SlashCompleter(self.commands),
            complete_while_typing=True,
            key_bindings=self.key_bindings,
            multiline=True,
            bottom_toolbar=self._bottom_toolbar,
        )
        install_builtin_commands(
            self.commands,
            runtime,
            request_exit=self._request_exit,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._events: asyncio.Queue[SessionEvent] = asyncio.Queue()
        self._prompt_task: asyncio.Task[str] | None = None
        self._queued: deque[str] = deque()
        self._answers: asyncio.Queue[str] = asyncio.Queue()
        self._answering = False
        self._in_loop = False
        self._closed = False
        self._unsubscribe: Callable[[], None] = runtime.on_event(self._receive_event)

    def _request_exit(self) -> None:
        self.exiting = True

    def _bottom_toolbar(self) -> StyleAndTextTuples:
        width = self.session.app.output.get_size().columns
        return self.footer.render(width)

    def _keys(self) -> KeyBindings:
        keys = KeyBindings()

        @keys.add("enter")
        def submit(event: KeyPressEvent) -> None:
            event.current_buffer.validate_and_handle()

        @keys.add("escape", "enter")
        def newline(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text("\n")

        @keys.add("c-c")
        def cancel(event: KeyPressEvent) -> None:
            self.runtime.cancel()

        @keys.add("c-d")
        def exit_(event: KeyPressEvent) -> None:
            event.app.exit(exception=EOFError())

        return keys

    def show_message(self, text: object) -> None:
        """Write a command result or error outside the model renderer."""
        print_formatted_text(str(text), output=self.output)

    def _receive_event(self, event: SessionEvent) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._events.put_nowait, event)

    async def _render_events(self) -> None:
        while True:
            event = await self._events.get()
            if event.kind == "agent_event":
                self.renderer.handle_event(event.payload["event"])
            self.session.app.invalidate()

    def _ensure_prompt(self) -> asyncio.Task[str]:
        if self._prompt_task is None:
            self._prompt_task = asyncio.create_task(
                self.session.prompt_async("ds4py> "),
            )
        return self._prompt_task

    def _take_prompt(self, task: asyncio.Task[str]) -> str:
        self._prompt_task = None
        return task.result()

    async def _next_line(self) -> str:
        task = self._ensure_prompt()
        try:
            return await task
        finally:
            if task.done():
                self._prompt_task = None

    async def _run_with_controls(self, prompt: str) -> None:
        run_task = asyncio.create_task(self.runtime.run(prompt))
        while True:
            prompt_task = self._ensure_prompt()
            done, _ = await asyncio.wait(
                {run_task, prompt_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if run_task in done:
                try:
                    run_task.result()
                except Exception as error:
                    self.show_message(f"error: {error}")
                return
            if prompt_task in done:
                try:
                    line = self._take_prompt(prompt_task)
                    if self._answering:
                        self._answers.put_nowait(line)
                    else:
                        self._queued.append(line)
                except EOFError:
                    self.exiting = True
                    self.runtime.cancel()
                    try:
                        await run_task
                    except Exception as error:
                        self.show_message(f"error: {error}")
                    return

    async def ask(self, request: AskUserRequest) -> AskUserResponse:
        """Answer model questions through the active prompt, never raw stdin."""
        if self._answering:
            raise RuntimeError("a model question is already active")
        self._answering = True
        answers = []
        try:
            for question in request.questions:
                self.show_message(question.question)
                for index, option in enumerate(question.options, 1):
                    description = f" — {option.description}" if option.description else ""
                    self.show_message(f"  {index}. {option.label}{description}")
                prompt = "Choices (comma-separated)" if question.multi_select else "Choice"
                self.show_message(f"{prompt}; enter /cancel to cancel.")
                while True:
                    if self._in_loop:
                        self._ensure_prompt()
                        answer = await self._answers.get()
                    else:
                        try:
                            answer = await self.session.prompt_async("answer> ")
                        except EOFError:
                            return AskUserResponse(cancelled=True)
                    if answer.strip() == "/cancel":
                        return AskUserResponse(cancelled=True)
                    selected = selected_labels(question, answer)
                    if selected is None:
                        self.show_message("Invalid choice; try again or enter /cancel.")
                        continue
                    answers.append(AskUserAnswer(
                        header=question.header, selected=selected,
                    ))
                    break
            return AskUserResponse(answers=tuple(answers))
        finally:
            self._answering = False
            while not self._answers.empty():
                self._queued.append(self._answers.get_nowait())

    async def dispatch(self, line: str) -> None:
        """Execute one submitted prompt or command without escaping the loop."""
        if not line.strip():
            return
        if line.lstrip().startswith("/"):
            try:
                result = await self.commands.execute(line.strip())
            except Exception as error:
                self.show_message(f"error: {error}")
            else:
                if result is not None:
                    self.show_message(result)
            return
        if self._in_loop:
            await self._run_with_controls(line)
        else:
            try:
                await self.runtime.run(line)
            except Exception as error:
                self.show_message(f"error: {error}")

    async def run(self, initial: str | None = None) -> int:
        """Run an optional initial prompt, then accept input until exit."""
        self._loop = asyncio.get_running_loop()
        self._in_loop = True
        event_task = asyncio.create_task(self._render_events())
        try:
            with patch_stdout(raw=True):
                if initial is not None:
                    self._queued.append(initial)
                while not self.exiting:
                    if self._queued:
                        line = self._queued.popleft()
                    else:
                        try:
                            line = await self._next_line()
                        except EOFError:
                            break
                    await self.dispatch(line)
        finally:
            self._in_loop = False
            self._loop = None
            event_task.cancel()
            if self._prompt_task is not None:
                self._prompt_task.cancel()
            await asyncio.gather(
                event_task,
                *((self._prompt_task,) if self._prompt_task is not None else ()),
                return_exceptions=True,
            )
            self.close()
        return 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unsubscribe()
        self.footer.close()
