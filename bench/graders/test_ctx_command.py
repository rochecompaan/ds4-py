"""Hidden behavioral grader for the bakeoff /ctx repository task."""

import asyncio
import re
from typing import TYPE_CHECKING, Any, Never, cast

if TYPE_CHECKING:
    from ds4py.runtime import AgentRuntime
from ds4py.tui_commands import SlashCommandRegistry, install_builtin_commands


class ForbiddenModel:
    async def request(self, *args: Any, **kwargs: Any) -> Never:
        raise AssertionError("/ctx must not call the model")


class ForbiddenSession:
    def run_round(self, *args: Any, **kwargs: Any) -> Never:
        raise AssertionError("/ctx must not generate native tokens")

    def append_user(self, *args: Any, **kwargs: Any) -> Never:
        raise AssertionError("/ctx must not append conversation history")


class FakeRuntime:
    ctx_used = 123
    ctx = 1000

    def __init__(self) -> None:
        self.model = ForbiddenModel()
        self.session = ForbiddenSession()

    async def new(self) -> Never:
        raise AssertionError("/ctx must not replace the session")

    async def save(self) -> Never:
        raise AssertionError("/ctx must not save the session")

    async def switch(self, prefix: str) -> Never:
        raise AssertionError("/ctx must not switch the session")

    def set_hints(self, enabled: bool) -> Never:
        raise AssertionError("/ctx must not change hints")


def test_ctx_command_reports_usage_without_generation_or_mutation() -> None:
    runtime = FakeRuntime()
    registry = SlashCommandRegistry()
    exited: list[bool] = []
    install_builtin_commands(
        registry,
        cast("AgentRuntime", runtime),
        request_exit=lambda: exited.append(True),
    )

    result = asyncio.run(registry.execute("/ctx"))

    assert isinstance(result, str)
    assert re.search(r"\b123\s+used\b", result, re.IGNORECASE)
    assert re.search(r"\b877\s+remaining\b", result, re.IGNORECASE)
    assert any(item.text == "/ctx " for item in registry.completions("/"))
    assert not exited

    for line in ("/ctx extra",):
        try:
            asyncio.run(registry.execute(line))
        except Exception as error:
            assert type(error).__name__ == "CommandError"
        else:
            raise AssertionError("/ctx must reject arguments")
