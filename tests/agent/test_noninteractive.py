import asyncio
import builtins
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never, cast

import pytest

import ds4py
from ds4py import cli
from ds4py.engine.sampling import SamplingPolicy
from ds4py.noninteractive import run_noninteractive


class FakeRunRuntime:
    def __init__(self, *, output: Any = "answer", run_error: Any = None, save_error: Any = None) -> None:
        self.output = output
        self.run_error = run_error
        self.save_error = save_error
        self.saves = 0

    async def run(self, prompt: Any) -> Any:
        if self.run_error is not None:
            raise self.run_error
        return SimpleNamespace(output=self.output)

    async def save(self) -> None:
        self.saves += 1
        if self.save_error is not None:
            raise self.save_error


def invoke(runtime: Any) -> Any:
    out, err = io.StringIO(), io.StringIO()
    code = asyncio.run(run_noninteractive(runtime, "prompt", out=out, err=err))
    return code, out.getvalue(), err.getvalue()


def test_noninteractive_prints_only_the_final_answer_and_saves() -> None:
    runtime = FakeRunRuntime()
    assert invoke(runtime) == (0, "answer\n", "")
    assert runtime.saves == 1


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (RuntimeError("run failed"), 1, "run failed"),
        (asyncio.CancelledError(), 130, "interrupted"),
        (KeyboardInterrupt(), 130, "interrupted"),
    ],
)
def test_noninteractive_saves_on_run_failure_and_interruption(error: Any, code: Any, message: Any) -> None:
    runtime = FakeRunRuntime(run_error=error)
    actual, out, err = invoke(runtime)
    assert actual == code and out == "" and message in err
    assert runtime.saves == 1


def test_noninteractive_save_failure_is_nonzero() -> None:
    assert invoke(FakeRunRuntime(save_error=OSError("disk full"))) == (
        1,
        "answer\n",
        "ds4py: save failed: disk full\n",
    )


def test_noninteractive_saves_and_reports_save_failure_after_output_failure() -> None:
    class BrokenOutput:
        def write(self, text: Any) -> Never:
            raise BrokenPipeError("closed pipe")

    runtime = FakeRunRuntime(save_error=OSError("disk full"))
    err = io.StringIO()
    code = asyncio.run(
        run_noninteractive(
            cast("Any", runtime),
            "prompt",
            out=cast("Any", BrokenOutput()),
            err=err,
        )
    )

    assert code == 1
    assert runtime.saves == 1
    assert "output failed: closed pipe" in err.getvalue()
    assert "save failed: disk full" in err.getvalue()


def test_noninteractive_reports_both_run_and_save_failures() -> None:
    code, out, err = invoke(
        FakeRunRuntime(
            run_error=RuntimeError("run failed"),
            save_error=OSError("disk full"),
        )
    )
    assert code == 1 and out == ""
    assert "run failed" in err and "save failed: disk full" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["--non-interactive"],
        ["--non-interactive", "--prompt", "x", "--prompt-file", "x.txt"],
        ["--non-interactive", "--prompt", "x", "--edit-upto", "1"],
    ],
)
def test_agent_cli_validates_noninteractive_prompt_and_supported_flags(argv: Any) -> None:
    with pytest.raises(SystemExit):
        cli.main(argv)


def test_agent_cli_reports_value_errors_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ds4py import agent_cli

    def fail(_args: Any) -> Never:
        raise ValueError("invalid prefix")

    monkeypatch.setattr(agent_cli, "command", fail)

    assert cli.main([]) == 1
    assert capsys.readouterr().err == "ds4py: invalid prefix\n"


def test_agent_cli_maps_engine_sampling_policy_history_and_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import ds4py.model as model_module
    import ds4py.noninteractive as noninteractive_module
    import ds4py.runtime as runtime_module

    created = []

    class Engine:
        def __init__(self, think: Any) -> None:
            self.info = SimpleNamespace(is_glm_dsa=False)
            self.think = think
            self.exited = False

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            self.exited = True

        def session(self) -> object:
            return object()

    class Runtime:
        def __init__(self, model: Any, project_dir: Any, **kwargs: Any) -> None:
            self.model = model
            self.project_dir = project_dir
            self.kwargs = kwargs
            self.loaded = None
            self.closed = False
            created.append(self)

        def load_prefix(self, turns: Any) -> None:
            self.loaded = turns

        async def close(self) -> None:
            self.closed = True

    def open_fake(model: Any, **kwargs: Any) -> Any:
        created.append(SimpleNamespace(kind="open", model=model, kwargs=kwargs))
        engine = Engine(kwargs["think"])
        created.append(SimpleNamespace(kind="engine", engine=engine))
        return engine

    def model_fake(session: Any, **kwargs: Any) -> Any:
        value = SimpleNamespace(session=session, kwargs=kwargs)
        created.append(SimpleNamespace(kind="model", model=value))
        return value

    async def run_fake(runtime: Any, prompt: Any) -> int:
        created.append(SimpleNamespace(kind="run", runtime=runtime, prompt=prompt))
        return 0

    monkeypatch.setattr(ds4py, "open", open_fake)
    monkeypatch.setattr(model_module, "DS4Model", model_fake)
    monkeypatch.setattr(runtime_module, "AgentRuntime", Runtime)
    monkeypatch.setattr(noninteractive_module, "run_noninteractive", run_fake)
    prefix = tmp_path / "prefix.txt"
    prefix.write_text("USER: earlier\nASSISTANT: answer\n")

    assert (
        cli.main(
            [
                "--non-interactive",
                "--prompt",
                "hello",
                "--model",
                "m.gguf",
                "--ctx",
                "100000",
                "--tokens",
                "123",
                "--temp",
                "0",
                "--top-p",
                "0.8",
                "--min-p",
                "0",
                "--seed",
                "1",
                "--think-level",
                "25",
                "--cpu",
                "--threads",
                "2",
                "--system",
                "persistent rules",
                "--prefix-file",
                str(prefix),
                "--read-only",
            ]
        )
        == 0
    )

    opened = next(item for item in created if item.kind == "open")
    assert opened.model == "m.gguf"
    assert opened.kwargs == {
        "ctx": 100000,
        "backend": "cpu",
        "think": 1025,
        "seed": 1,
        "threads": 2,
        "lib": None,
    }
    made = next(item for item in created if item.kind == "model").model
    assert made.kwargs == {
        "sampling": SamplingPolicy(temperature=0.0, top_p=0.8, min_p=0.0, seed=1),
        "max_tokens": 123,
    }
    runtime = next(item for item in created if isinstance(item, Runtime))
    assert runtime.kwargs == {
        "read_only": True,
        "system": "persistent rules",
        "interactive": False,
        "question_answerer": None,
    }
    assert [(turn.role, turn.content) for turn in cast("Any", runtime.loaded)] == [
        ("user", "earlier"),
        ("assistant", "answer"),
    ]
    run = next(item for item in created if getattr(item, "kind", None) == "run")
    assert run.runtime is runtime and run.prompt == "hello"
    assert runtime.closed
    assert next(item for item in created if getattr(item, "kind", None) == "engine").engine.exited


def test_agent_cli_interactive_constructs_and_runs_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    import ds4py.model as model_module
    import ds4py.runtime as runtime_module
    import ds4py.tui as tui_module

    created = []

    class Engine:
        info = SimpleNamespace(is_glm_dsa=False)

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            created.append("engine-closed")

        def session(self) -> object:
            return object()

    class Runtime:
        def __init__(self, model: Any, project_dir: Any, **kwargs: Any) -> None:
            self.kwargs = kwargs
            created.append(self)

        async def close(self) -> None:
            created.append("runtime-closed")

    class FakeTUI:
        def __init__(self, runtime: Any) -> None:
            self.runtime = runtime
            created.append(self)

        async def run(self) -> int:
            created.append("tui-ran")
            return 0

    monkeypatch.setattr(ds4py, "open", lambda *args, **kwargs: Engine())
    monkeypatch.setattr(model_module, "DS4Model", lambda session, **kwargs: object())
    monkeypatch.setattr(runtime_module, "AgentRuntime", Runtime)
    monkeypatch.setattr(tui_module, "TUI", FakeTUI)

    assert cli.main([]) == 0
    runtime = next(item for item in created if isinstance(item, Runtime))
    tui = next(item for item in created if isinstance(item, FakeTUI))
    assert tui.runtime is runtime
    assert runtime.kwargs["interactive"] is True
    assert "yolo" not in runtime.kwargs
    assert "approval_answerer" not in runtime.kwargs
    assert callable(runtime.kwargs["question_answerer"])
    assert "tui-ran" in created
    assert created[-2:] == ["runtime-closed", "engine-closed"]


def test_agent_cli_rejects_removed_yolo_flag() -> None:
    from ds4py import agent_cli

    with pytest.raises(SystemExit):
        agent_cli.parser().parse_args(["--yolo"])


def test_agent_cli_question_answerer_returns_a_typed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai_harness.ask_user import (
        AskUserRequest,
        AskUserResponse,
        Question,
        QuestionOption,
    )

    from ds4py import agent_cli

    request = AskUserRequest(
        questions=(
            Question(
                header="next",
                question="What next?",
                options=(QuestionOption(label="one"), QuestionOption(label="two")),
            ),
        )
    )
    monkeypatch.setattr(sys.stdin, "readline", lambda: "2\n")

    response = asyncio.run(agent_cli._question_answerer(request))

    assert isinstance(response, AskUserResponse)
    assert response.answers[0].header == "next"
    assert response.answers[0].selected == ("two",)


def test_raw_alias_and_core_commands_do_not_import_the_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import ds4py.raw as raw_module

    calls = []
    monkeypatch.setattr(raw_module, "command", lambda args: calls.append(args.command) or 0)
    monkeypatch.setattr(cli, "build_library", lambda *args, **kwargs: tmp_path / "libds4.so")
    monkeypatch.setattr(cli, "check_library", lambda lib: SimpleNamespace(manifest=None, error=None))
    monkeypatch.setattr(cli, "format_report", lambda report: "ABI ok")
    real_import = builtins.__import__

    def guarded_import(name: Any, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("pydantic_ai", "pydantic_ai_harness")):
            raise AssertionError(f"core command imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert cli.main(["raw", "hello"]) == 0
    assert cli.main(["--raw", "hello"]) == 0
    assert cli.main(["build", "--src", str(tmp_path)]) == 0
    assert cli.main(["check"]) == 0
    assert calls == ["raw", "raw"]
    assert "ABI ok" in capsys.readouterr().out
