from dataclasses import replace
import fcntl
import io
import os
from pathlib import Path
import shutil
import sys

import pytest

from ds4py.abi import build, pins
from ds4py.abi.build import build_library
from ds4py.abi.manifest import Manifest, file_sha256, manifest_path
from ds4py._native.loader import LIB_NAME
from ds4py.errors import BuildError
from test_probe import doctored


@pytest.fixture
def fake_tree(tmp_path, upstream_headers):
    tree = tmp_path / "source"
    shutil.copytree(upstream_headers, tree)
    (tree / "Makefile").write_text("")
    (tree / "ds4.c").write_text("")
    return tree


def test_not_a_ds4_tree(tmp_path):
    (tmp_path / "Makefile").touch()
    with pytest.raises(BuildError, match="not a ds4 source tree: missing ds4.h"):
        build_library(tmp_path, backend="cpu", out_dir=tmp_path / "out")


@pytest.mark.parametrize("missing", ["Makefile", "ds4.h", "ds4_kvstore.h", "ds4.c"])
def test_missing_source_file_is_reported_first(fake_tree, tmp_path, monkeypatch, missing):
    (fake_tree / missing).unlink()
    monkeypatch.setenv("CC", str(tmp_path / "no-compiler"))
    with pytest.raises(BuildError, match=f"missing {missing}"):
        build_library(fake_tree, backend="cpu", out_dir=tmp_path / "out")


def test_no_compiler(tmp_path, fake_tree, monkeypatch):
    monkeypatch.setenv("CC", str(tmp_path / "no-such-cc"))
    with pytest.raises(BuildError, match="no C compiler"):
        build_library(fake_tree, backend="cpu", out_dir=tmp_path / "out")


def only_tools(tmp_path, monkeypatch, *names):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in names:
        (bin_dir / name).write_text("#!/bin/sh\nexit 0\n")
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("CC", raising=False)


def test_no_make(tmp_path, fake_tree, monkeypatch):
    only_tools(tmp_path, monkeypatch, "cc")
    with pytest.raises(BuildError, match="make not found on PATH"):
        build_library(fake_tree, backend="cpu", out_dir=tmp_path / "out")


def test_rocm_is_refused(tmp_path, fake_tree, monkeypatch):
    only_tools(tmp_path, monkeypatch, "cc", "make")
    with pytest.raises(BuildError, match="backend rocm is not supported by ds4py build yet"):
        build_library(fake_tree, backend="rocm", out_dir=tmp_path / "out")


@pytest.mark.skipif(sys.platform == "darwin", reason="metal builds on macOS")
def test_metal_needs_macos(tmp_path, fake_tree, monkeypatch):
    only_tools(tmp_path, monkeypatch, "cc", "make")
    with pytest.raises(BuildError, match="backend metal needs macOS"):
        build_library(fake_tree, backend="metal", out_dir=tmp_path / "out")


def test_changed_argument_type_fails_before_make(cc, tmp_path, monkeypatch):
    tree = doctored(tmp_path, "int ds4_session_eval(ds4_session *s, int token,",
                              "int ds4_session_eval(ds4_session *s, long token,")
    (tree / "Makefile").write_text("")
    (tree / "ds4.c").write_text("")
    monkeypatch.setattr(build, "_run_make", lambda *a, **k: pytest.fail("make ran"))
    with pytest.raises(BuildError) as e:
        build_library(tree, backend="cpu", out_dir=tmp_path / "out", cc=cc)
    assert "ds4_session_eval" in str(e.value) and "incompatible" in str(e.value)
    assert not (tmp_path / "out" / LIB_NAME).exists()


def test_one_build_per_out_dir(cc, tmp_path, fake_tree, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    held = os.open(out / ".build.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(held, fcntl.LOCK_EX)
        monkeypatch.setattr(build, "_run_make", lambda *a, **k: pytest.fail("make ran"))
        with pytest.raises(BuildError, match="another ds4py build is using"):
            build_library(fake_tree, backend="cpu", out_dir=out, cc=cc)
    finally:
        os.close(held)


def installed(out):
    lib = out / LIB_NAME
    return lib.read_bytes(), Manifest.read(manifest_path(lib)).library_sha256 == file_sha256(lib)


def test_install_switches_the_library_and_manifest_together(make_manifest, tmp_path, upstream_headers, monkeypatch):
    def pair(name, **changes):
        (tmp_path / name).mkdir()
        lib, m = make_manifest(tmp_path / name, upstream_headers)
        lib.write_bytes(name.encode())
        return lib, replace(m, library_sha256=file_sha256(lib), **changes)

    out = tmp_path / "out"
    one, two, four = pair("one"), pair("two"), pair("four")
    three = pair("three", type_map_version=0)
    assert build.install(*one, out) == out / LIB_NAME and installed(out) == (b"one", True)
    with monkeypatch.context() as mp:
        mp.setattr(Manifest, "write", lambda self, path: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            build.install(*two, out)
    with pytest.raises(BuildError):
        build.install(*three, out)
    assert installed(out) == (b"one", True)
    build.install(*four, out)
    assert installed(out) == (b"four", True)
    assert len(list((out / "installs").iterdir())) == 2


def test_mirror_skips_git_and_objects(tmp_path):
    src, farm = tmp_path / "src", tmp_path / "farm"
    for rel in (".git/HEAD", "ds4.c", "metal/ds4.metal", "ds4.o", "metal/ds4.o", "libds4.a", "libds4.so", "libds4.dylib"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(rel)
    build._mirror(src, farm)
    build._mirror(src, farm)
    files = {str(Path(d, f).relative_to(farm)) for d, _, fs in os.walk(farm) for f in fs}
    assert files == {"ds4.c", "metal/ds4.metal"} and not (farm / ".git").exists()
    assert all((farm / f).is_symlink() and (farm / f).resolve() == (src / f).resolve() for f in files)
    assert not (farm / "metal").is_symlink()


@pytest.mark.parametrize("entry", ["entry", "nested/entry"])
def test_mirror_file_to_directory_preserves_source(tmp_path, entry):
    src, farm = tmp_path / "src", tmp_path / "farm"
    source_entry = src / entry
    source_entry.parent.mkdir(parents=True)
    source_entry.write_text("original file")
    build._mirror(src, farm)

    source_entry.unlink()
    source_entry.mkdir()
    child = source_entry / "child.c"
    child.write_text("source contents")
    build._mirror(src, farm)
    build._mirror(src, farm)

    assert not child.is_symlink()
    assert child.read_text() == "source contents"
    mirrored = farm / entry
    assert mirrored.is_dir() and not mirrored.is_symlink()
    assert (mirrored / "child.c").is_symlink()
    assert (mirrored / "child.c").resolve() == child.resolve()


def test_mirror_directory_to_file_preserves_source(tmp_path):
    src, farm = tmp_path / "src", tmp_path / "farm"
    source_entry = src / "entry"
    source_entry.mkdir(parents=True)
    child = source_entry / "child.c"
    child.write_text("source contents")
    build._mirror(src, farm)
    child.unlink()
    source_entry.rmdir()
    source_entry.write_text("replacement file")

    build._mirror(src, farm)
    build._mirror(src, farm)

    assert not source_entry.is_symlink()
    assert source_entry.read_text() == "replacement file"
    assert (farm / "entry").is_symlink()
    assert (farm / "entry").resolve() == source_entry.resolve()


def test_mirror_does_not_recurse_into_itself(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "ds4.c").write_text("source")
    farm = src / "farm"
    build._mirror(src, farm)
    build._mirror(src, farm)
    assert (farm / "ds4.c").read_text() == "source"
    assert not (farm / "farm").exists()


def test_failed_switch_keeps_old_pair_and_cleans_staging(make_manifest, tmp_path, upstream_headers, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    pair = make_manifest(src, upstream_headers)
    out = tmp_path / "out"
    current = build.install(*pair, out)
    held = current.resolve()
    # A killed build may leave its unpublished symlink at this fixed name.
    temporary = out / f".{LIB_NAME}.new"
    temporary.symlink_to("missing-generation")
    original_replace = os.replace

    def fail_switch(source, target):
        if Path(target) == current:
            raise OSError("switch failed")
        return original_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_switch)
    with pytest.raises(OSError, match="switch failed"):
        build.install(*pair, out)
    assert current.resolve() == held and installed(out)[1]
    assert not temporary.is_symlink()
    assert len(list((out / "installs").iterdir())) == 1


def test_make_failure_streams_output_and_reports_tail(tmp_path):
    (tmp_path / "Makefile.lib").write_text(
        "libds4:\n\t@i=0; while [ $$i -lt 45 ]; do echo line$$i; i=$$((i+1)); done; exit 1\n"
    )
    log = io.StringIO()
    with pytest.raises(BuildError) as error:
        build._run_make(tmp_path, "cpu", "cc", 1, log)
    assert "line0\n" in log.getvalue() and "line44\n" in log.getvalue()
    assert "line0\n" not in str(error.value) and "line44\n" in str(error.value)


def test_cli_build_reports_errors(tmp_path, capsys):
    from ds4py.cli import main
    assert main(["build", "--src", str(tmp_path)]) == 1
    assert "ds4py:" in capsys.readouterr().err


@pytest.mark.build
def test_cpu_build_of_upstream(built_upstream, capsys):
    from ds4py.cli import main

    lib, m = built_upstream
    assert lib.name == LIB_NAME and m.library_sha256 == file_sha256(lib)
    assert all(r.ok for r in m.bindings.values()) and m.headers_known and m.backend == "cpu"
    if (Path(os.environ["DS4_SRC_UPSTREAM"]) / ".git").exists():
        assert m.source_commit == pins.UPSTREAM.commit
    # The CLI rebuild shares the farm: objects survive and are not recompiled.
    objects = {p: p.stat().st_mtime_ns for p in (lib.parent / "build").rglob("*.o")}
    held = lib.resolve()
    assert objects
    assert main(["build", "--src", m.source_path, "--backend", "cpu", "--out", str(lib.parent), "-j", "1"]) == 0
    assert str(lib) in capsys.readouterr().out.splitlines()
    assert all(p.stat().st_mtime_ns == timestamp for p, timestamp in objects.items())
    assert lib.resolve() != held and held.exists()
    assert Manifest.read(manifest_path(lib)).library_sha256 == file_sha256(lib)


@pytest.mark.build
def test_cpu_build_of_fork_with_extension(built_fork):
    lib, m = built_fork
    assert not m.headers_known and m.bindings["ds4_engine_sampling_defaults"].ok
