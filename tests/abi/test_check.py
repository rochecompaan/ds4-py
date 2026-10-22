from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ds4py._native.bindings import CORE_BINDINGS
from ds4py._native.loader import LIB_NAME
from ds4py._native.typemap import TYPE_MAP_VERSION
from ds4py.abi.check import CheckReport, check_library, format_report
from ds4py.abi.manifest import BindingRecord, Manifest, Validation
from ds4py.abi.pins import KNOWN_HEADERS, UPSTREAM

EXT_SITE = str((Path(__file__).parents[1] / "fixtures" / "ext_site").resolve())
FORK_HASH = "b1e0ad190897de401780d02d246e610882bca676a531bd22d2e06b3769c52d9b"


def run_check(lib, extra_env=None):
    return subprocess.run(
        [sys.executable, "-m", "ds4py", "check", "--lib", str(lib)],
        env={**os.environ, **(extra_env or {})}, capture_output=True, text=True,
    )


def report_with(*, headers_known=True, absent=frozenset(), extensions=None):
    extensions = extensions or {}
    records = {b.name: BindingRecord(b.normalized, b.required, b.source, True)
               for b in CORE_BINDINGS}
    for name, source in extensions.items():
        records[name] = BindingRecord("", False, source, True)
    for name in absent:
        records[name] = BindingRecord("", False, extensions.get(name, "core"), True)
    headers = dict(KNOWN_HEADERS[0])
    if not headers_known:
        headers["ds4.h"] = FORK_HASH
    manifest = Manifest(TYPE_MAP_VERSION, LIB_NAME, "", "/src/ds4", UPSTREAM.commit,
                        "cpu", headers, headers_known, records, {})
    warnings = () if headers_known else (f"unknown header ds4.h: {FORK_HASH[:12]}",)
    validation = Validation(frozenset(records) - frozenset(absent), frozenset(absent), warnings)
    return CheckReport(Path("/lib") / LIB_NAME, manifest, validation, extensions, None)


def test_report_known_headers_and_no_optional_bindings():
    assert format_report(report_with()).splitlines() == [
        f"library:  /lib/{LIB_NAME} (cpu)",
        "ds4 tree: /src/ds4 at 5379c1c9ac31",
        "headers:  known",
        "layouts:  ok",
        f"bindings: {len(CORE_BINDINGS)} required ok",
        "optional: none",
        "restore:  not verified (run `ds4py check --live --model PATH`)",
    ]


def test_report_lines_for_unknown_headers_and_optional_bindings():
    text = format_report(report_with(
        headers_known=False, absent={"ds4_engine_sampling_defaults"},
        extensions={"ds4_engine_layer_count": "test_ext",
                    "ds4_engine_sampling_defaults": "test_ext"},
    ))
    assert "warning: unknown header ds4.h b1e0ad190897; bindings and layouts were checked, but this tree is not a tested pin" in text
    assert "ds4_engine_sampling_defaults  absent  (test_ext)" in text
    assert "ds4_engine_layer_count  present  (test_ext)" in text
    assert "restore:  not verified" in text


def test_report_non_git_tree_and_core_optional():
    report = report_with(absent={"optional_core"})
    report.manifest = replace(report.manifest, source_commit=None)
    text = format_report(report)
    assert "ds4 tree: /src/ds4 (not a git checkout)" in text
    assert "optional: optional_core  absent  (core)" in text


def test_refused_library_exits_1(tmp_path):
    lib = tmp_path / LIB_NAME
    lib.write_bytes(b"not a library")
    result = run_check(lib)
    assert result.returncode == 1
    assert result.stdout.startswith(f"library:  {lib}\nerror:    ")
    assert "libds4.abi.json" in result.stdout
    assert result.stderr == ""


def test_check_resolves_environment_path_on_failure(tmp_path, monkeypatch):
    lib = tmp_path / LIB_NAME
    monkeypatch.setenv("DS4_LIB", str(lib))
    report = check_library(None)
    assert report.lib_path == lib
    assert report.manifest is None and report.validation is None
    assert "library not found" in report.error


@pytest.mark.build
def test_check_upstream_build(built_upstream):
    result = run_check(built_upstream[0])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "headers:  known" in result.stdout and "layouts:  ok" in result.stdout


@pytest.mark.build
def test_check_fork_build_lists_extension(built_fork):
    result = run_check(built_fork[0], {"PYTHONPATH": EXT_SITE})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "warning: unknown header ds4.h b1e0ad190897" in result.stdout
    assert "ds4_engine_sampling_defaults  present  (test_ext)" in result.stdout
    assert "ds4_engine_layer_count  present  (test_ext)" in result.stdout
