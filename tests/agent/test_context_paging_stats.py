"""RollingWindow statistics stay read-only across live and restored sessions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

from pydantic_ai.messages import ModelResponse
from test_restore import fake_engine, fixed_clock, make_model, project_dir, scripted_rounds

from ds4py.capabilities.context import ContextComposition
from ds4py.capabilities.rolling_window import RollingWindow
from ds4py.context_paging.stats import InputTokenReading, read_request_input
from ds4py.message_metadata import DS4_PAGING_INPUT_TOKENS, DS4_REQUEST_SYNC_COUNT
from ds4py.runtime import AgentRuntime
from ds4py.store import SessionStore
from ds4py.tui_commands import SlashCommandRegistry, install_builtin_commands


def test_live_exact_input_and_repeated_stats_preserve_all_owner_state(tmp_path: Path, monkeypatch: Any) -> None:
    async def scenario() -> None:
        with fake_engine(script=scripted_rounds("done")) as engine:
            model = make_model(engine)
            runtime = AgentRuntime(
                model,
                project_dir(tmp_path),
                store=SessionStore(tmp_path / "home"),
                context=ContextComposition.rolling(token_budget=16_000),
                clock=fixed_clock,
            )
            owner = model.context_owner
            assert isinstance(owner, RollingWindow)
            try:
                await runtime.run("hello")
                # A cached append is still one selected request, not aggregate work.
                await runtime.run("again")
                saved_response = runtime.messages[-1]
                assert isinstance(saved_response, ModelResponse)
                metadata = saved_response.metadata or {}
                exact = metadata[DS4_PAGING_INPUT_TOKENS]
                assert isinstance(exact, int)
                assert metadata[DS4_REQUEST_SYNC_COUNT] == 1
                assert read_request_input(saved_response) == InputTokenReading("exact", exact)

                cast("Any", owner._state).selection = "marked selection"
                before = (
                    owner._state.selection,
                    owner._state.settings,
                    tuple(model.session.transcript.tokens),
                    list(runtime.messages),
                )

                def no_native_or_provider_work(*args: Any, **kwargs: Any) -> None:
                    raise AssertionError("statistics must not start model or native work")

                monkeypatch.setattr(model.session, "run_round", no_native_or_provider_work)
                monkeypatch.setattr(owner, "_select", no_native_or_provider_work)
                registry = SlashCommandRegistry()
                install_builtin_commands(registry, runtime, request_exit=lambda: None)
                first = await registry.execute("/context-paging stats")
                second = await registry.execute("/context-paging stats")

                assert first == second
                assert isinstance(first, str)
                assert f"Input tokens         {exact:,}" in first
                assert "Paging budget        16,000" in first
                assert f"({exact / 16_000 * 100:.1f}% used)" in first
                after = (
                    owner._state.selection,
                    owner._state.settings,
                    tuple(model.session.transcript.tokens),
                    list(runtime.messages),
                )
                assert after == before
            finally:
                await runtime.close()

    asyncio.run(scenario())


def test_exact_input_survives_save_and_view_restore(tmp_path: Path) -> None:
    project = project_dir(tmp_path)
    store = SessionStore(tmp_path / "home")
    with fake_engine(script=scripted_rounds("done")) as engine:
        runtime = AgentRuntime(
            make_model(engine),
            project,
            store=store,
            context=ContextComposition.rolling(token_budget=16_000),
            clock=fixed_clock,
        )

        async def save() -> tuple[str, int]:
            try:
                await runtime.run("hello")
                saved_response = runtime.messages[-1]
                assert isinstance(saved_response, ModelResponse)
                reading = read_request_input(saved_response)
                assert reading.tokens is not None
                record = await runtime.save()
                return record.session_id, reading.tokens
            finally:
                await runtime.close()

        session_id, exact = asyncio.run(save())

    with fake_engine() as engine:
        restored = AgentRuntime(
            make_model(engine),
            project,
            store=store,
            context=ContextComposition.rolling(token_budget=16_000),
            clock=fixed_clock,
        )

        async def restore() -> None:
            try:
                await restored.switch(session_id)
                restored_response = restored.messages[-1]
                assert isinstance(restored_response, ModelResponse)
                assert read_request_input(restored_response) == InputTokenReading("exact", exact)
                report = str(await restored.context_paging_stats())
                assert f"Input tokens         {exact:,}" in report
            finally:
                await restored.close()

        asyncio.run(restore())
