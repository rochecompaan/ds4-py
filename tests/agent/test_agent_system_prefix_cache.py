import asyncio
from pathlib import Path
from typing import Any

from test_runtime import fake_engine, fixed_clock, make_model, project_dir, scripted_rounds

from ds4py.engine.worker import SYSTEM
from ds4py.runtime import AgentRuntime
from ds4py.store import SessionStore


def engine_events(events: list[Any], kind: str) -> list[Any]:
    return [event.payload["event"] for event in events if event.kind == kind]


def test_runtime_new_uses_the_same_prefix_cache(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "store")
    with fake_engine(script=scripted_rounds("first answer", "second answer")) as engine:
        model = make_model(engine)
        runtime = AgentRuntime(model, project_dir(tmp_path), store=store, clock=fixed_clock)

        async def scenario() -> None:
            first = await runtime.run("first")
            assert first.output == "first answer"
            session = model.session
            system_length = max(segment.end for segment in session.transcript.segments if segment.kind == "system")
            prefix = session.transcript.tokens[:system_length]
            assert len(list((tmp_path / "ds4py-home" / "sysprompt").glob("*.kv"))) == 1

            await runtime.new()
            events: list[Any] = []
            session.on_event(events.append)
            second = await runtime.run("second")

            assert second.output == "second answer"
            assert session.transcript.tokens[:system_length] == prefix
            assert session.history.checkpoint_tokens == {SYSTEM: system_length}
            assert len(engine_events(events, "Loaded")) == 1
            system_prefills = [
                event
                for event in engine_events(events, "Prefill")
                if event.total > 0 and event.cached_prefix < system_length
            ]
            assert system_prefills == []

        try:
            asyncio.run(scenario())
        finally:
            asyncio.run(runtime.close())
