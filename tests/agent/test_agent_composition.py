import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, override

import pytest

import ds4py
from ds4py.agent_composition import ConfiguredSession
from ds4py.engine.sampling import SamplingPolicy


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], 0), (["--top-k", "0"], 0), (["--top-k", "40"], 40)],
)
def test_default_open_runtime_maps_top_k(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    expected: int,
) -> None:
    import ds4py.model as model_module
    import ds4py.runtime as runtime_module
    from ds4py import agent_cli

    captured = []

    class Engine:
        info = SimpleNamespace(is_glm_dsa=False)

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def session(self) -> object:
            return object()

    class Runtime:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def close(self) -> None:
            pass

    def make_model(session: Any, *, sampling: SamplingPolicy, **kwargs: Any) -> Any:
        captured.append(sampling)
        return SimpleNamespace(session=session)

    monkeypatch.setattr(ds4py, "open", lambda *args, **kwargs: Engine())
    monkeypatch.setattr(model_module, "DS4Model", make_model)
    monkeypatch.setattr(runtime_module, "AgentRuntime", Runtime)

    args = agent_cli.parser().parse_args(argv)
    with agent_cli.open_runtime(args, tmp_path):
        pass

    assert [policy.top_k for policy in captured] == [expected]


def test_configured_session_exit_closes_wrapper() -> None:
    from ds4py.agent_composition import ConfiguredSession

    closed = []
    configured = ConfiguredSession(
        session=cast("Any", object()),
        sampling=SamplingPolicy(),
        close=lambda: closed.append("closed"),
    )

    with configured as entered:
        assert entered is configured
        assert closed == []

    assert closed == ["closed"]


class RecordingExtension:
    program = "companion-agent"

    def __init__(self, session: object, sampling: SamplingPolicy, events: list[Any]) -> None:
        self.session = session
        self.sampling = sampling
        self.events = events
        self.config = object()
        self.options = []
        self.project_dirs = []
        self.extension_configs = []
        self.command_calls = []

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.parse_args(["--top-k", "0"])
        parser.add_argument("--extension-value")

    def resolve_extension_config(self, args: argparse.Namespace, project_dir: Path) -> object:
        self.events.append(("resolve", args.extension_value, project_dir))
        return self.config

    def open_session(
        self,
        options: Any,
        project_dir: Path,
        extension_config: object,
    ) -> ConfiguredSession:
        self.events.append("open")
        self.options.append(options)
        self.project_dirs.append(project_dir)
        self.extension_configs.append(extension_config)
        return ConfiguredSession(
            session=cast("Any", self.session),
            sampling=self.sampling,
            close=lambda: self.events.append("configured-close"),
        )

    def configure_commands(self, registry: Any, runtime: Any) -> None:
        self.events.append("commands")
        self.command_calls.append((registry, runtime))

    def before_identity_change(self) -> None:
        self.events.append("before")

    def after_identity_change(self) -> None:
        self.events.append("after")


@pytest.mark.parametrize(
    ("think_argv", "expected_think"),
    [(["--think-level", "25"], 1025), (["--nothink"], 0)],
)
def test_noninteractive_extension_receives_one_typed_options_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    think_argv: list[str],
    expected_think: int,
) -> None:
    import ds4py.model as model_module
    import ds4py.noninteractive as noninteractive_module
    import ds4py.runtime as runtime_module
    from ds4py import agent_cli
    from ds4py.agent_composition import EngineOpenOptions

    events = []
    session = object()
    sampling = SamplingPolicy(temperature=0.75, top_k=90)
    extension = RecordingExtension(session, sampling, events)
    models = []
    runtimes = []

    class Runtime:
        def __init__(self, model: Any, project_dir: Path, **kwargs: Any) -> None:
            events.append("runtime")
            self.model = model
            self.project_dir = project_dir
            self.kwargs = kwargs
            runtimes.append(self)

        async def close(self) -> None:
            events.append("runtime-close")

    def make_model(actual_session: Any, **kwargs: Any) -> Any:
        events.append("model")
        model = SimpleNamespace(session=actual_session, kwargs=kwargs)
        models.append(model)
        return model

    async def run_fake(runtime: Any, prompt: str, **kwargs: Any) -> int:
        events.append(("run", runtime, prompt, kwargs))
        return 0

    def forbidden_merge(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("custom sampling must not be merged again")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_module, "DS4Model", make_model)
    monkeypatch.setattr(runtime_module, "AgentRuntime", Runtime)
    monkeypatch.setattr(noninteractive_module, "run_noninteractive", run_fake)
    monkeypatch.setattr(SamplingPolicy, "for_engine", classmethod(forbidden_merge))
    lib = tmp_path / "libds4.dylib"

    result = agent_cli.main(
        [
            "--non-interactive",
            "--prompt",
            "hello",
            "--model",
            "custom.gguf",
            "--ctx",
            "4096",
            "--backend",
            "test",
            *think_argv,
            "--seed",
            "0",
            "--threads",
            "3",
            "--lib",
            str(lib),
            "--tokens",
            "123",
            "--temp",
            "0",
            "--top-k",
            "0",
            "--top-p",
            "0",
            "--min-p",
            "0",
            "--request-limit",
            "7",
            "--system",
            "rules",
            "--read-only",
            "--extension-value",
            "resolved",
        ],
        extension=extension,
    )

    assert result == 0
    assert extension.options == [
        EngineOpenOptions(
            model="custom.gguf",
            ctx=4096,
            backend="test",
            think=expected_think,
            seed=0,
            threads=3,
            lib=lib,
            tokens=123,
            interactive=False,
            temperature=0.0,
            top_k=0,
            top_p=0.0,
            min_p=0.0,
        )
    ]
    assert extension.project_dirs == [tmp_path]
    assert extension.extension_configs == [extension.config]
    assert events[0] == ("resolve", "resolved", tmp_path)
    assert models[0].session is session
    assert models[0].kwargs == {"sampling": sampling, "max_tokens": 123}
    runtime = runtimes[0]
    assert runtime.project_dir == tmp_path
    assert runtime.kwargs == {
        "read_only": True,
        "system": "rules",
        "interactive": False,
        "question_answerer": None,
        "request_limit": 7,
        "before_identity_change": extension.before_identity_change,
        "after_identity_change": extension.after_identity_change,
    }
    assert extension.command_calls == []
    assert events[-3:] == [
        ("run", runtime, "hello", {"program": "companion-agent"}),
        "runtime-close",
        "configured-close",
    ]


def test_interactive_extension_opens_session_and_configures_tui(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import ds4py.model as model_module
    import ds4py.runtime as runtime_module
    import ds4py.tui as tui_module
    from ds4py import agent_cli

    events = []
    extension = RecordingExtension(object(), SamplingPolicy(), events)
    runtimes = []

    class Runtime:
        def __init__(self, model: Any, project_dir: Path, **kwargs: Any) -> None:
            events.append("runtime")
            self.kwargs = kwargs
            runtimes.append(self)

        async def close(self) -> None:
            events.append("runtime-close")

    class FakeTUI:
        def __init__(self, runtime: Any, *, configure_commands: Any = None) -> None:
            self.runtime = runtime
            assert configure_commands is not None
            registry = object()
            configure_commands(registry, runtime)

        async def run(self) -> int:
            events.append("tui-run")
            return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_module, "DS4Model", lambda session, **kwargs: SimpleNamespace(session=session))
    monkeypatch.setattr(runtime_module, "AgentRuntime", Runtime)
    monkeypatch.setattr(tui_module, "TUI", FakeTUI)

    assert agent_cli.main(["--extension-value", "interactive"], extension=extension) == 0
    assert len(extension.options) == 1
    assert extension.options[0].interactive is True
    assert len(extension.command_calls) == 1
    assert extension.command_calls[0][1] is runtimes[0]
    assert events[-4:] == ["commands", "tui-run", "runtime-close", "configured-close"]


def test_duplicate_generic_parser_option_fails() -> None:
    from ds4py import agent_cli
    from ds4py.agent_composition import DefaultAgentExtension

    class DuplicateOption(DefaultAgentExtension):
        @override
        def configure_parser(self, parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--model")

    with pytest.raises(argparse.ArgumentError):
        agent_cli.parser(DuplicateOption())


def test_extension_program_labels_parser_and_top_level_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from ds4py import agent_cli
    from ds4py.agent_composition import DefaultAgentExtension

    class FailingExtension(DefaultAgentExtension):
        program = "companion-agent"

        @override
        def resolve_extension_config(self, args: argparse.Namespace, project_dir: Path) -> object:
            raise ValueError("bad extension config")

    extension = FailingExtension()
    assert agent_cli.parser(extension).prog == "companion-agent"
    assert agent_cli.main([], extension=extension) == 1
    assert capsys.readouterr().err == "companion-agent: bad extension config\n"


def test_default_open_session_closes_engine_when_sampling_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ds4py.agent_composition import DefaultAgentExtension, EngineOpenOptions

    events = []
    failure = ValueError("sampling failed")

    class Engine:
        info = SimpleNamespace(is_glm_dsa=False)

        def __enter__(self) -> Any:
            events.append("engine-enter")
            return self

        def __exit__(self, *args: object) -> None:
            events.append("engine-exit")

        def session(self) -> object:
            return object()

    def fail_sampling(*args: Any, **kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(ds4py, "open", lambda *args, **kwargs: Engine())
    monkeypatch.setattr(SamplingPolicy, "for_engine", classmethod(fail_sampling))
    options = EngineOpenOptions(
        model="model.gguf",
        ctx=1000,
        backend=None,
        think=0,
        seed=None,
        threads=0,
        lib=None,
        tokens=100,
        interactive=False,
        temperature=None,
        top_k=None,
        top_p=None,
        min_p=None,
    )

    with (
        pytest.raises(ValueError, match="sampling failed") as caught,
        DefaultAgentExtension().open_session(options, Path("/project"), None),
    ):
        pass

    assert caught.value is failure
    assert events == ["engine-enter", "engine-exit"]
