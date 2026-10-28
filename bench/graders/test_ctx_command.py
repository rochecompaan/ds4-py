"""Hidden behavioral grader for the M3 /ctx repository task."""

import asyncio
import re

from ds4py.tui_commands import SlashCommandRegistry, install_builtin_commands


class ForbiddenModel:
    async def request(self, *args, **kwargs):
        raise AssertionError("/ctx must not call the model")


class ForbiddenSession:
    def run_round(self, *args, **kwargs):
        raise AssertionError("/ctx must not generate native tokens")

    def append_user(self, *args, **kwargs):
        raise AssertionError("/ctx must not append conversation history")


class FakeRuntime:
    ctx_used = 123
    ctx = 1000

    def __init__(self):
        self.model = ForbiddenModel()
        self.session = ForbiddenSession()

    async def new(self):
        raise AssertionError("/ctx must not replace the session")

    async def save(self):
        raise AssertionError("/ctx must not save the session")

    async def switch(self, prefix):
        raise AssertionError("/ctx must not switch the session")

    def set_hints(self, enabled):
        raise AssertionError("/ctx must not change hints")


def test_ctx_command_reports_usage_without_generation_or_mutation():
    runtime = FakeRuntime()
    registry = SlashCommandRegistry()
    exited = []
    install_builtin_commands(registry, runtime, request_exit=lambda: exited.append(True))

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
