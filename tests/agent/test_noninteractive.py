import asyncio
import builtins
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

import ds4py
from ds4py import ThinkMode, cli
from ds4py.engine.sampling import SamplingPolicy
from ds4py.noninteractive import run_noninteractive


class FakeRunRuntime:
    def __init__(self, *, output="answer", run_error=None, save_error=None):
        self.output = output
        self.run_error = run_error
        self.save_error = save_error
        self.saves = 0

    async def run(self, prompt):
        if self.run_error is not None:
            raise self.run_error
        return SimpleNamespace(output=self.output)

    async def save(self):
        self.saves += 1
        if self.save_error is not None:
            raise self.save_error


def invoke(runtime):
    out, err = io.StringIO(), io.StringIO()
    code = asyncio.run(run_noninteractive(runtime, "prompt", out=out, err=err))
    return code, out.getvalue(), err.getvalue()


def test_noninteractive_prints_only_the_final_answer_and_saves():
    runtime = FakeRunRuntime()
    assert invoke(runtime) == (0, "answer\n", "")
    assert runtime.saves == 1


@pytest.mark.parametrize(("error", "code", "message"), [
    (RuntimeError("run failed"), 1, "run failed"),
    (asyncio.CancelledError(), 130, "interrupted"),
    (KeyboardInterrupt(), 130, "interrupted"),
])
def test_noninteractive_saves_on_run_failure_and_interruption(error, code, message):
    runtime = FakeRunRuntime(run_error=error)
    actual, out, err = invoke(runtime)
    assert actual == code and out == "" and message in err
    assert runtime.saves == 1


def test_noninteractive_save_failure_is_nonzero():
    assert invoke(FakeRunRuntime(save_error=OSError("disk full"))) == (
        1, "answer\n", "ds4py: save failed: disk full\n",
    )


def test_noninteractive_saves_and_reports_save_failure_after_output_failure():
    class BrokenOutput:
        def write(self, text):
            raise BrokenPipeError("closed pipe")

    runtime = FakeRunRuntime(save_error=OSError("disk full"))
    err = io.StringIO()
    code = asyncio.run(run_noninteractive(
        runtime, "prompt", out=BrokenOutput(), err=err,
    ))

    assert code == 1
    assert runtime.saves == 1
    assert "output failed: closed pipe" in err.getvalue()
    assert "save failed: disk full" in err.getvalue()


def test_noninteractive_reports_both_run_and_save_failures():
    code, out, err = invoke(FakeRunRuntime(
        run_error=RuntimeError("run failed"), save_error=OSError("disk full"),
    ))
    assert code == 1 and out == ""
    assert "run failed" in err and "save failed: disk full" in err


@pytest.mark.parametrize("argv", [
    ["--non-interactive"],
    ["--non-interactive", "--prompt", "x", "--prompt-file", "x.txt"],
    ["--non-interactive", "--prompt", "x", "--edit-upto", "1"],
])
def test_agent_cli_validates_noninteractive_prompt_and_supported_flags(argv):
    with pytest.raises(SystemExit):
        cli.main(argv)


def test_agent_cli_maps_engine_sampling_policy_history_and_workspace(monkeypatch, tmp_path):
    import ds4py.model as model_module
    import ds4py.noninteractive as noninteractive_module
    import ds4py.runtime as runtime_module

    created = []

    class Engine:
        def __init__(self, think):
            self.info = SimpleNamespace(is_glm_dsa=False)
            self.think = think
            self.exited = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.exited = True

        def session(self):
            return object()

    class Runtime:
        def __init__(self, model, project_dir, **kwargs):
            self.model = model
            self.project_dir = project_dir
            self.kwargs = kwargs
            self.loaded = None
            self.closed = False
            created.append(self)

        def load_prefix(self, turns):
            self.loaded = turns

        async def close(self):
            self.closed = True

    def open_fake(model, **kwargs):
        created.append(SimpleNamespace(kind="open", model=model, kwargs=kwargs))
        engine = Engine(kwargs["think"])
        created.append(SimpleNamespace(kind="engine", engine=engine))
        return engine

    def model_fake(session, **kwargs):
        value = SimpleNamespace(session=session, kwargs=kwargs)
        created.append(SimpleNamespace(kind="model", model=value))
        return value

    async def run_fake(runtime, prompt):
        created.append(SimpleNamespace(kind="run", runtime=runtime, prompt=prompt))
        return 0

    monkeypatch.setattr(ds4py, "open", open_fake)
    monkeypatch.setattr(model_module, "DS4Model", model_fake)
    monkeypatch.setattr(runtime_module, "AgentRuntime", Runtime)
    monkeypatch.setattr(noninteractive_module, "run_noninteractive", run_fake)
    prefix = tmp_path / "prefix.txt"
    prefix.write_text("USER: earlier\nASSISTANT: answer\n")

    assert cli.main([
        "--non-interactive", "--prompt", "hello", "--model", "m.gguf",
        "--ctx", "100000", "--tokens", "123", "--temp", "0", "--top-p", "0.8",
        "--min-p", "0", "--seed", "1", "--think-level", "25", "--cpu",
        "--threads", "2", "--system", "persistent rules", "--prefix-file", str(prefix),
        "--read-only", "--yolo",
    ]) == 0

    opened = next(item for item in created if item.kind == "open")
    assert opened.model == "m.gguf"
    assert opened.kwargs == {
        "ctx": 100000, "backend": "cpu", "think": 1025, "seed": 1,
        "threads": 2, "lib": None,
    }
    made = next(item for item in created if item.kind == "model").model
    assert made.kwargs == {
        "sampling": SamplingPolicy(temperature=0.0, top_p=0.8, min_p=0.0, seed=1),
        "max_tokens": 123,
    }
    runtime = next(item for item in created if isinstance(item, Runtime))
    assert runtime.kwargs == {
        "read_only": True, "yolo": True, "system": "persistent rules",
    }
    assert [(turn.role, turn.content) for turn in runtime.loaded] == [
        ("user", "earlier"), ("assistant", "answer"),
    ]
    run = next(item for item in created if getattr(item, "kind", None) == "run")
    assert run.runtime is runtime and run.prompt == "hello"
    assert runtime.closed
    assert next(item for item in created if getattr(item, "kind", None) == "engine").engine.exited


def test_raw_alias_and_core_commands_do_not_import_the_harness(monkeypatch, tmp_path, capsys):
    import ds4py.raw as raw_module

    calls = []
    monkeypatch.setattr(raw_module, "command", lambda args: calls.append(args.command) or 0)
    monkeypatch.setattr(cli, "build_library", lambda *args, **kwargs: tmp_path / "libds4.so")
    monkeypatch.setattr(cli, "check_library", lambda lib: SimpleNamespace(manifest=None, error=None))
    monkeypatch.setattr(cli, "format_report", lambda report: "ABI ok")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
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
