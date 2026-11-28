"""CLI RollingWindow selection, native-window preflight, and warning delivery."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any, cast, override

import pytest
from test_restore import fake_engine

from ds4py import agent_cli
from ds4py.agent_composition import ConfiguredSession, DefaultAgentExtension, EngineOpenOptions
from ds4py.capabilities.rolling_window import RollingWindow
from ds4py.engine.lifecycle import SessionEvent
from ds4py.engine.sampling import SamplingPolicy
from ds4py.noninteractive import run_noninteractive


class RecordingExtension(DefaultAgentExtension):
    def __init__(self) -> None:
        self.options: EngineOpenOptions | None = None
        self.open_calls = 0

    @override
    def open_session(self, options: EngineOpenOptions, project_dir: Path, extension_config: object) -> Any:
        self.open_calls += 1
        self.options = options
        engine = fake_engine()
        return ConfiguredSession(engine.session(), SamplingPolicy(seed=1), engine.close)


def test_rolling_preflight_reserves_budget_while_compaction_keeps_ctx(tmp_path: Path) -> None:
    rolling = RecordingExtension()
    args = agent_cli.parser(rolling).parse_args(
        ["--context-strategy", "rolling", "--context-token-budget", "4000", "--ctx", "10000"]
    )
    with agent_cli.open_runtime(args, tmp_path, extension=rolling) as runtime:
        assert rolling.options is not None
        assert rolling.options.ctx == 4000
        assert isinstance(runtime.model.context_owner, RollingWindow)

    compaction = RecordingExtension()
    args = agent_cli.parser(compaction).parse_args(["--ctx", "10000"])
    with agent_cli.open_runtime(args, tmp_path, extension=compaction):
        assert compaction.options is not None
        assert compaction.options.ctx == 10000


@pytest.mark.parametrize(
    ("trim_arguments", "expected_explicit"),
    [([], False), (["--context-trim-to", "1"], True)],
    ids=["implicit", "explicit"],
)
def test_rolling_preflight_preserves_trim_provenance(
    tmp_path: Path, trim_arguments: list[str], expected_explicit: bool
) -> None:
    extension = RecordingExtension()
    args = agent_cli.parser(extension).parse_args(
        ["--context-strategy", "rolling", "--context-token-budget", "1", *trim_arguments]
    )

    with agent_cli.open_runtime(args, tmp_path, extension=extension) as runtime:
        owner = runtime.model.context_owner
        assert isinstance(owner, RollingWindow)
        assert owner._base_settings.trim_to == 1
        assert owner._base_settings.trim_to_explicit is expected_explicit
        warnings: list[SessionEvent] = []
        unsubscribe = runtime.on_event(warnings.append, kinds={"context_paging_warning"})
        runtime._refresh_context_paging()
        unsubscribe()

    assert len(warnings) == int(expected_explicit)


def test_initially_disabled_rolling_still_reserves_the_smaller_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension = RecordingExtension()
    store = tmp_path / "global"
    store.mkdir()
    (store / "settings.json").write_text('{"contextPaging":{"enabled":false}}')
    monkeypatch.setenv("DS4PY_HOME", str(store))
    args = agent_cli.parser(extension).parse_args(["--context-strategy", "rolling", "--context-token-budget", "5000"])

    with agent_cli.open_runtime(args, tmp_path, extension=extension) as runtime:
        assert extension.options is not None
        assert extension.options.ctx == 5000
        owner = runtime.model.context_owner
        assert isinstance(owner, RollingWindow)
        assert not owner.enabled


def test_bad_settings_abort_before_engine_or_event_bus_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.json").write_text("{broken")
    monkeypatch.setenv("DS4PY_HOME", str(home))
    extension = RecordingExtension()
    args = agent_cli.parser(extension).parse_args(["--context-strategy", "rolling"])

    with (
        pytest.raises(json.JSONDecodeError, match="Expecting property name"),
        agent_cli.open_runtime(args, tmp_path, extension=extension),
    ):
        pass
    assert extension.open_calls == 0


def test_noninteractive_replays_initial_paging_warnings_and_unsubscribes() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.listeners = []

        def on_event(self, listener: Any, *, kinds: Any = None) -> Any:
            self.listeners.append(listener)

            def unsubscribe() -> None:
                self.listeners.remove(listener)

            return unsubscribe

        def _refresh_context_paging(self) -> None:
            for listener in tuple(self.listeners):
                listener(SessionEvent("context_paging_warning", "session", {"message": "trim warning"}))

        async def run(self, prompt: str) -> Any:
            return type("Result", (), {"output": "answer"})()

        async def save(self) -> None:
            assert self.listeners == []

    runtime = Runtime()
    out = io.StringIO()
    err = io.StringIO()
    code = asyncio.run(run_noninteractive(cast("Any", runtime), "hello", out=out, err=err))

    assert code == 0
    assert out.getvalue() == "answer\n"
    assert err.getvalue() == "ds4py: warning: trim warning\n"
    assert runtime.listeners == []
