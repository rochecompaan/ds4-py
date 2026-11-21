"""Idle-only RollingWindow commands and their status publication."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_restore import fake_engine, make_model

from ds4py.capabilities.context import ContextComposition
from ds4py.capabilities.rolling_window import RollingWindow
from ds4py.context_paging.selection import ContextCutFrontier, ContextCutKey, SelectionState
from ds4py.runtime import AgentRuntime
from ds4py.store import SessionStore
from ds4py.tui_commands import CommandError, SlashCommandRegistry, install_builtin_commands


def runtime_for(path: Path, *, rolling: bool = True) -> tuple[Any, AgentRuntime, RollingWindow | None]:
    engine = fake_engine()
    model = make_model(engine)
    runtime = AgentRuntime(
        model,
        path,
        store=SessionStore(path / "store"),
        context=ContextComposition.rolling(token_budget=16000) if rolling else None,
    )
    owner = model.context_owner
    return engine, runtime, owner if isinstance(owner, RollingWindow) else None


def marked(owner: RollingWindow) -> None:
    owner._state.selection = SelectionState(ContextCutFrontier("completedTurn", ContextCutKey("old")), None)
    owner._state.accounting["requests"] = 3


def test_status_repeated_choices_and_invalid_arguments_preserve_state(tmp_path: Path) -> None:
    engine, runtime, owner = runtime_for(tmp_path)
    assert owner is not None
    registry = SlashCommandRegistry()
    install_builtin_commands(registry, runtime, request_exit=lambda: None)

    async def scenario() -> None:
        assert await registry.execute("/context-paging") == "paging on"
        assert await registry.execute("/context-paging status") == "paging on"
        marked(owner)
        selection = owner._state.selection
        accounting = owner._state.accounting.copy()
        assert await registry.execute("/context-paging on") == "paging on"
        assert owner._state.selection is selection
        assert owner._state.accounting == accounting

        with pytest.raises(CommandError, match="usage: /context-paging"):
            await registry.execute("/context-paging invalid")
        with pytest.raises(CommandError, match="usage: /context-paging"):
            await registry.execute("/context-paging on now")
        assert owner._state.selection is selection
        assert owner._state.accounting == accounting

        assert await registry.execute("/context-paging off") == "paging off"
        assert not owner.enabled
        assert owner._state.selection is None
        assert owner._state.accounting == {}
        marked(owner)
        assert await registry.execute("/context-paging off") == "paging off"
        assert owner._state.selection is not None
        assert owner._state.accounting == {"requests": 3}

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(runtime.close())
        engine.close()


def test_context_paging_command_is_owner_specific_and_idle_only(tmp_path: Path) -> None:
    engine, runtime, owner = runtime_for(tmp_path, rolling=False)
    assert owner is None
    registry = SlashCommandRegistry()
    install_builtin_commands(registry, runtime, request_exit=lambda: None)
    with pytest.raises(CommandError, match="unknown command"):
        asyncio.run(registry.execute("/context-paging"))
    asyncio.run(runtime.close())
    engine.close()

    engine, runtime, owner = runtime_for(tmp_path / "rolling")
    assert owner is not None
    cast("Any", runtime)._active = SimpleNamespace(kind="run")
    with pytest.raises(Exception, match="active"):
        runtime.context_paging("on")
    runtime._active = None
    asyncio.run(runtime.close())
    engine.close()
