import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from ds4py import cli
from ds4py._native.bindings import CORE_BINDINGS
from ds4py._native.loader import LIB_NAME
from ds4py._native.typemap import TYPE_MAP_VERSION
from ds4py.abi.check import CheckReport, check_library, format_report
from ds4py.abi.manifest import BindingRecord, Manifest, Validation
from ds4py.abi.pins import KNOWN_HEADERS, UPSTREAM

EXT_SITE = str((Path(__file__).parents[1] / "fixtures" / "ext_site").resolve())
FORK_HASH = "b1e0ad190897de401780d02d246e610882bca676a531bd22d2e06b3769c52d9b"


def run_check(lib: Any, extra_env: Any = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ds4py", "check", "--lib", str(lib)],
        env={**os.environ, **(extra_env or {})},
        capture_output=True,
        text=True,
    )


def report_with(
    *, headers_known: Any = True, absent: Any = frozenset(), extensions: Any = None, live_checks: Any = None
) -> Any:
    extensions = extensions or {}
    records = {b.name: BindingRecord(b.normalized, b.required, b.source, True) for b in CORE_BINDINGS}
    for name, source in extensions.items():
        records[name] = BindingRecord("", False, source, True)
    for name in absent:
        records[name] = BindingRecord("", False, extensions.get(name, "core"), True)
    headers = dict(KNOWN_HEADERS[0])
    if not headers_known:
        headers["ds4.h"] = FORK_HASH
    manifest = Manifest(
        TYPE_MAP_VERSION,
        LIB_NAME,
        "",
        "/src/ds4",
        UPSTREAM.commit,
        "cpu",
        headers,
        headers_known,
        records,
        {},
        live_checks=live_checks or {},
    )
    warnings = () if headers_known else (f"unknown header ds4.h: {FORK_HASH[:12]}",)
    validation = Validation(frozenset(records) - frozenset(absent), frozenset(absent), warnings)
    return CheckReport(Path("/lib") / LIB_NAME, manifest, validation, extensions, None)


def test_report_known_headers_and_no_optional_bindings() -> None:
    assert format_report(report_with()).splitlines() == [
        f"library:  /lib/{LIB_NAME} (cpu)",
        "ds4 tree: /src/ds4 at 5379c1c9ac31",
        "headers:  known",
        "layouts:  ok",
        f"bindings: {len(CORE_BINDINGS)} required ok",
        "optional: none",
        "restore:  not verified (run `ds4py check --live --model PATH`)",
    ]


def test_report_lines_for_unknown_headers_and_optional_bindings() -> None:
    text = format_report(
        report_with(
            headers_known=False,
            absent={"ds4_engine_sampling_defaults"},
            extensions={"ds4_engine_layer_count": "test_ext", "ds4_engine_sampling_defaults": "test_ext"},
        )
    )
    assert (
        "warning: unknown header ds4.h b1e0ad190897; bindings and layouts were checked, but this tree is not a tested pin"
        in text
    )
    assert "ds4_engine_sampling_defaults  absent  (test_ext)" in text
    assert "ds4_engine_layer_count  present  (test_ext)" in text
    assert "restore:  not verified" in text


def test_report_non_git_tree_and_core_optional() -> None:
    report = report_with(absent={"optional_core"})
    report.manifest = replace(report.manifest, source_commit=None)
    text = format_report(report)
    assert "ds4 tree: /src/ds4 (not a git checkout)" in text
    assert "optional: optional_core  absent  (core)" in text


def test_refused_library_exits_1(tmp_path: Path) -> None:
    lib = tmp_path / LIB_NAME
    lib.write_bytes(b"not a library")
    result = run_check(lib)
    assert result.returncode == 1
    assert result.stdout.startswith(f"library:  {lib}\nerror:    ")
    assert "libds4.abi.json" in result.stdout
    assert result.stderr == ""


def test_check_resolves_environment_path_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib = tmp_path / LIB_NAME
    monkeypatch.setenv("DS4_LIB", str(lib))
    report = check_library(None)
    assert report.lib_path == lib
    assert report.manifest is None and report.validation is None
    assert "library not found" in cast("Any", report.error)


@pytest.mark.build
def test_check_upstream_build(built_upstream: Any) -> None:
    result = run_check(built_upstream[0])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "headers:  known" in result.stdout and "layouts:  ok" in result.stdout


@pytest.mark.build
def test_check_fork_build_lists_extension(built_fork: Any) -> None:
    result = run_check(built_fork[0], {"PYTHONPATH": EXT_SITE})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "headers:  known" in result.stdout
    assert "ds4_engine_sampling_defaults  present  (test_ext)" in result.stdout
    assert "ds4_engine_layer_count  present  (test_ext)" in result.stdout


def test_report_shows_the_live_checks() -> None:
    text = format_report(
        report_with(
            live_checks={
                "sync_plans": {"ok": True, "message": "extend of 64 tokens took 0.120 s"},
                "restore": {"ok": False, "message": "the runs differ at token 3: 7 after a restore, 9 after a rebuild"},
            }
        )
    )
    assert "sync:     ok (extend of 64 tokens took 0.120 s)" in text
    assert "restore:  disabled (the runs differ at token 3: 7 after a restore, 9 after a rebuild)" in text
    assert "sync:" not in format_report(report_with())


def test_live_needs_a_model(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["check", "--live"])
    assert e.value.code == 2 and "--live needs --model" in capsys.readouterr().err


@pytest.mark.parametrize(
    "sync_ok,restore_ok,error,expected",
    [
        (True, True, None, 0),
        (True, False, None, 0),
        (False, True, None, 1),
        (True, True, "refused", 1),
    ],
)
def test_live_cli_exit_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sync_ok: Any,
    restore_ok: Any,
    error: Any,
    expected: Any,
) -> None:
    report = report_with(
        live_checks={
            "sync_plans": {"ok": sync_ok, "message": "counts"},
            "restore": {"ok": restore_ok, "message": "tokens"},
        }
    )
    report.error = error

    def live(lib: Any, model: Any, *, ctx: Any) -> Any:
        assert (lib, model, ctx) == (Path("lib.so"), Path("model.gguf"), 512)
        return report

    monkeypatch.setattr(cli, "check_live", live, raising=False)
    assert cli.main(["check", "--live", "--lib", "lib.so", "--model", "model.gguf", "--ctx", "512"]) == expected
    assert ("error:    refused" if error else "sync:") in capsys.readouterr().out


@pytest.mark.parametrize("fail", [False, True])
def test_live_persists_checks_only_after_success_and_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail: Any
) -> None:
    from types import SimpleNamespace

    from ds4py._native.fake import FakeEngine, FakeNative
    from ds4py.abi import check
    from ds4py.abi.manifest import manifest_path
    from ds4py.errors import EngineError

    report = report_with(live_checks={"restore": {"ok": False}})
    lib = tmp_path / LIB_NAME
    lib.touch()
    library = SimpleNamespace(path=lib, manifest=report.manifest, validation=report.validation)
    library.manifest.write(manifest_path(lib))
    native = FakeNative(script=lambda p, g: 98, restore_enabled=False)
    monkeypatch.setattr(check, "load_library", lambda path: library)
    monkeypatch.setattr(check, "LibNative", lambda lib: native)
    engines = []
    original_close = FakeEngine.close

    def close(engine: Any) -> None:
        engines.append(engine)
        original_close(engine)

    monkeypatch.setattr(FakeEngine, "close", close)
    run = check.run_live_checks

    def reduced(worker: Any) -> Any:
        if fail:
            raise EngineError("sync failed")
        return run(worker, prefix=128, tail=16, generate=32)

    monkeypatch.setattr(check, "run_live_checks", reduced)
    if fail:
        with pytest.raises(EngineError, match="sync failed"):
            check.check_live(lib, Path("f.gguf"), ctx=512)
        assert Manifest.read(manifest_path(lib)).live_checks == {"restore": {"ok": False}}
    else:
        result = check.check_live(lib, Path("f.gguf"), ctx=512)
        saved = Manifest.read(manifest_path(lib)).live_checks
        assert cast("Any", result.manifest).live_checks == saved
        assert saved["sync_plans"]["ok"] and saved["restore"]["ok"]
    assert len(engines) == 1
