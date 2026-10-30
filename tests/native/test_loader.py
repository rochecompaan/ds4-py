import ctypes
import re
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest

from ds4py._native import loader
from ds4py._native.bindings import all_bindings
from ds4py._native.loader import LIB_NAME, load_library, resolve_lib_path
from ds4py.abi import build, manifest
from ds4py.abi.manifest import manifest_path
from ds4py.errors import AbiMismatch, LibraryAlreadyLoaded, LibraryNotFound, MissingBinding


@pytest.mark.parametrize("reader_state", ["resolved", "loaded"])
def test_multiple_installs_keep_a_readers_generation(
    reader_state: Any, make_manifest: Any, tmp_path: Path, upstream_headers: Any, stub_dlopen: Any
) -> None:
    pairs = []
    for name in ("one", "two", "three"):
        src = tmp_path / name
        src.mkdir()
        pairs.append(make_manifest(src, upstream_headers))
    out = tmp_path / "out"
    current = build.install(pairs[0][0], pairs[0][1], out)
    held = load_library(current).path if reader_state == "loaded" else current.resolve(strict=True)
    expected = held.read_bytes()
    build.install(pairs[1][0], pairs[1][1], out)
    build.install(pairs[2][0], pairs[2][1], out)
    assert held.read_bytes() == expected
    assert manifest.Manifest.read(manifest_path(held)).library_sha256 == manifest.file_sha256(held)
    assert load_library(held).path == held


@pytest.fixture(autouse=True)
def reset_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "_loaded", None)


@pytest.fixture
def stub_dlopen(monkeypatch: pytest.MonkeyPatch) -> Any:
    handle = mock.MagicMock()
    monkeypatch.setattr(loader, "_dlopen", lambda p: handle)
    return handle


@pytest.mark.parametrize(
    "root, expected",
    [
        (None, ".ds4py/lib"),
        ("", ".ds4py/lib"),
        ("~/custom", "custom/lib"),
    ],
)
def test_default_root_is_expanded_at_call_time(
    root: Any, expected: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DS4_LIB", raising=False)
    if root is None:
        monkeypatch.delenv("DS4PY_HOME", raising=False)
    else:
        monkeypatch.setenv("DS4PY_HOME", root)
    assert loader.default_lib_dir() == tmp_path / expected
    assert not (tmp_path / expected).exists()
    monkeypatch.setenv("DS4PY_HOME", str(tmp_path / "another"))
    assert loader.default_lib_dir() == tmp_path / "another/lib"


def test_custom_root_loses_to_explicit_library_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "custom"
    monkeypatch.setenv("DS4PY_HOME", str(root))
    monkeypatch.delenv("DS4_LIB", raising=False)
    assert resolve_lib_path(None) == root / "lib" / LIB_NAME
    monkeypatch.setenv("DS4_LIB", str(tmp_path / "env.so"))
    assert resolve_lib_path(None) == tmp_path / "env.so"
    assert resolve_lib_path(tmp_path / "explicit.so") == tmp_path / "explicit.so"
    assert not root.exists()


def test_resolution_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DS4PY_HOME", raising=False)
    monkeypatch.setenv("DS4_LIB", str(tmp_path / "env" / LIB_NAME))
    assert resolve_lib_path(tmp_path / "x.so") == tmp_path / "x.so"
    assert resolve_lib_path(None) == tmp_path / "env" / LIB_NAME
    monkeypatch.delenv("DS4_LIB")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_lib_path(None) == tmp_path / ".ds4py" / "lib" / LIB_NAME


def test_missing_library_says_how_to_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DS4_LIB", raising=False)
    with pytest.raises(LibraryNotFound, match="ds4py build --src"):
        load_library()


def test_missing_manifest_is_refused_before_dlopen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / LIB_NAME).write_bytes(b"x")
    monkeypatch.setattr(loader, "_dlopen", lambda p: pytest.fail("dlopen called"))
    with pytest.raises(AbiMismatch, match="libds4.abi.json"):
        load_library(tmp_path / LIB_NAME)


def test_one_library_per_process(make_manifest: Any, tmp_path: Path, upstream_headers: Any, stub_dlopen: Any) -> None:
    paths = []
    for name in ("a", "b"):  # two valid pairs in two directories
        (tmp_path / name).mkdir()
        lib, m = make_manifest(tmp_path / name, upstream_headers)
        m.write(manifest_path(lib))
        paths.append(lib)
    lib_a, lib_b = paths
    assert load_library(lib_a) is load_library(lib_a)
    with pytest.raises(LibraryAlreadyLoaded, match=re.escape(str(lib_a.resolve()))):  # the loader reports real paths
        load_library(lib_b)


def test_symlink_pairs_with_the_manifest_beside_the_real_file(
    make_manifest: Any, tmp_path: Path, upstream_headers: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "installs").mkdir()
    real, m = make_manifest(tmp_path / "installs", upstream_headers)
    m.write(manifest_path(real))  # no manifest beside the symlink
    link = tmp_path / LIB_NAME
    link.symlink_to(real)
    opened = []
    monkeypatch.setattr(loader, "_dlopen", lambda p: opened.append(p) or mock.MagicMock())
    real = real.resolve()  # tmp_path may itself sit under a symlink
    assert load_library(link).path == real and opened == [real]


def test_absent_optional_binding_raises_missing_binding(
    make_manifest: Any, tmp_path: Path, upstream_headers: Any, ext_site: Any, stub_dlopen: Any
) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers, bindings=all_bindings())
    m.write(manifest_path(lib))
    loaded = load_library(lib)
    assert loaded.has("ds4_engine_layer_count") and not loaded.has("ds4_engine_sampling_defaults")
    with pytest.raises(MissingBinding, match="ds4_engine_sampling_defaults is absent from this library"):
        loaded.fn("ds4_engine_sampling_defaults")


def test_undeclared_name_raises_missing_binding(
    make_manifest: Any, tmp_path: Path, upstream_headers: Any, stub_dlopen: Any
) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    m.write(manifest_path(lib))
    loaded = load_library(lib)
    assert not loaded.has("ds4_session_tp_spec_cycle")  # in ds4.h, but no binding declares it
    with pytest.raises(MissingBinding, match="ds4py does not declare ds4_session_tp_spec_cycle"):
        loaded.fn("ds4_session_tp_spec_cycle")


@pytest.mark.parametrize("damage", ["hash", "prototype", "json"])
def test_invalid_manifest_never_reaches_dlopen(
    make_manifest: Any, tmp_path: Path, upstream_headers: Any, monkeypatch: pytest.MonkeyPatch, damage: Any
) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    if damage == "hash":
        m.library_sha256 = "wrong"
    elif damage == "prototype":
        m.bindings["ds4_session_eval"].prototype = "wrong"
    m.write(manifest_path(lib))
    if damage == "json":
        manifest_path(lib).write_text("{")
    monkeypatch.setattr(loader, "_dlopen", lambda p: pytest.fail("dlopen called"))
    with pytest.raises(AbiMismatch, match="ds4py build"):
        load_library(lib)


def test_old_cpu_manifest_still_loads(
    make_manifest: Any, tmp_path: Path, upstream_headers: Any, stub_dlopen: Any
) -> None:
    import json

    lib, m = make_manifest(tmp_path, upstream_headers)
    path = manifest_path(lib)
    m.write(path)
    data = json.loads(path.read_text())
    data.pop("metal_sources", None)
    path.write_text(json.dumps(data))
    assert load_library(lib).has("ds4_engine_open")


@pytest.mark.parametrize("damage", ["missing", "changed", "empty", "path", "hash"])
def test_bad_metal_bundle_is_refused_before_dlopen(
    make_manifest: Any,
    tmp_path: Path,
    upstream_headers: Any,
    monkeypatch: pytest.MonkeyPatch,
    damage: Any,
) -> None:
    import hashlib

    lib, m = make_manifest(tmp_path, upstream_headers)
    m.backend = "metal"
    m.metal_sources = {"metal/dense.metal": hashlib.sha256(b"shader").hexdigest()}
    (tmp_path / "metal").mkdir()
    shader = tmp_path / "metal/dense.metal"
    shader.write_bytes(b"shader")
    if damage == "missing":
        shader.unlink()
    elif damage == "changed":
        shader.write_bytes(b"wrong")
    elif damage == "empty":
        m.metal_sources = {}
    elif damage == "path":
        m.metal_sources = {"../dense.metal": hashlib.sha256(b"shader").hexdigest()}
    elif damage == "hash":
        m.metal_sources = {"metal/dense.metal": "bad"}
    m.write(manifest_path(lib))
    monkeypatch.setattr(loader, "_dlopen", lambda p: pytest.fail("dlopen called"))
    with pytest.raises(AbiMismatch, match="ds4py build"):
        load_library(lib)
    assert loader._loaded is None


def test_missing_symbol_refuses_without_caching(
    make_manifest: Any, tmp_path: Path, upstream_headers: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    m.write(manifest_path(lib))
    monkeypatch.setattr(loader, "_dlopen", lambda p: object())
    with pytest.raises(AbiMismatch, match="ds4_engine_open.*ds4py build"):
        load_library(lib)
    assert loader._loaded is None


def test_functions_are_typed_and_free_uses_library_handle(
    make_manifest: Any, tmp_path: Path, upstream_headers: Any, stub_dlopen: Any
) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    m.write(manifest_path(lib))
    loaded = load_library(lib)
    fn = loaded.fn("ds4_session_eval")
    assert fn is stub_dlopen.ds4_session_eval
    assert cast("Any", fn).restype is ctypes.c_int
    assert cast("Any", fn).argtypes == [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
    ptr = ctypes.c_void_p(123)
    loaded.free(ptr)
    assert stub_dlopen.free.argtypes == [ctypes.c_void_p]
    assert stub_dlopen.free.restype is None
    stub_dlopen.free.assert_called_once_with(ptr)


@pytest.mark.parametrize("ok", ["false", "true", 1, {"ok": False}, [False]])
def test_serialized_non_boolean_probe_result_is_refused_before_dlopen(
    make_manifest: Any,
    tmp_path: Path,
    upstream_headers: Any,
    monkeypatch: pytest.MonkeyPatch,
    ok: Any,
) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    m.bindings["ds4_session_eval"].ok = ok
    m.write(manifest_path(lib))
    monkeypatch.setattr(loader, "_dlopen", lambda p: pytest.fail("dlopen called"))
    with pytest.raises(AbiMismatch, match="ds4_session_eval.*boolean.*ds4py build"):
        load_library(lib)
    assert loader._loaded is None


def test_library_basename_does_not_change_manifest_requirements(
    make_manifest: Any,
    tmp_path: Path,
    upstream_headers: Any,
    stub_dlopen: Any,
) -> None:
    lib, metadata = make_manifest(tmp_path, upstream_headers)
    renamed = lib.with_name("libds4_m2.so")
    lib.rename(renamed)
    metadata.write(manifest_path(renamed))
    loaded = load_library(renamed)
    assert loaded.has("ds4_session_copy_logits")
    assert loaded.has("ds4_session_set_logits")
