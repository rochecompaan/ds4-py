from typing import Any
from pathlib import Path
from dataclasses import replace

import pytest

from ds4py.abi import manifest
from ds4py._native.bindings import CORE_BINDINGS, all_bindings
from ds4py._native.structs import STRUCTS, struct_layout
from ds4py._native.typemap import TYPE_MAP_VERSION
from ds4py.errors import AbiMismatch
from test_probe import doctored
from typing import Never

LAYOUTS = {name: struct_layout(cls) for name, cls in STRUCTS.items()}


def test_matching_manifest_is_accepted(make_manifest: Any, tmp_path: Path, upstream_headers: Any) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    v = manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)
    assert v.available == {b.name for b in CORE_BINDINGS} and v.warnings == ()

def test_unknown_headers_only_warn(make_manifest: Any, tmp_path: Path, fork_headers: Any) -> None:
    lib, m = make_manifest(tmp_path, fork_headers)
    m.headers["ds4.h"] = "0" * 64
    v = manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)
    assert "unknown header ds4.h: 000000000000" in v.warnings

def test_changed_prototype_is_refused(make_manifest: Any, tmp_path: Path, upstream_headers: Any) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    m.bindings["ds4_session_eval"].prototype = "int ds4_session_eval(ds4_session *, long, char *, size_t)"
    with pytest.raises(AbiMismatch) as e:
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)
    msg = str(e.value)
    assert "ds4_session_eval(ds4_session *, long, char *, size_t)" in msg
    assert "ds4_session_eval(ds4_session *, int, char *, size_t)" in msg
    assert "ds4py build" in msg

def test_moved_field_is_refused(make_manifest: Any, tmp_path: Path) -> None:
    inc = doctored(tmp_path, "    int n_threads;\n    int context_size;", "    int context_size;\n    int n_threads;")
    lib, m = make_manifest(tmp_path, inc)
    with pytest.raises(AbiMismatch, match=r"ds4_engine_options\.(n_threads|context_size)"):
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)

def test_failed_required_probe_is_refused(make_manifest: Any, tmp_path: Path) -> None:
    inc = doctored(tmp_path, "int ds4_session_eval(ds4_session *s, int token,",
                             "int ds4_session_eval(ds4_session *s, long token,")
    lib, m = make_manifest(tmp_path, inc)
    assert not m.bindings["ds4_session_eval"].ok
    with pytest.raises(AbiMismatch, match="ds4_session_eval"):
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)

def test_library_hash_mismatch_is_refused(make_manifest: Any, tmp_path: Path, upstream_headers: Any) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    lib.write_bytes(lib.read_bytes() + b"rebuilt")            # the library changed after its manifest was written
    with pytest.raises(AbiMismatch, match="ds4py build"):
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)

def test_type_map_version_mismatch_is_refused(make_manifest: Any, tmp_path: Path, upstream_headers: Any) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    m = replace(m, type_map_version=TYPE_MAP_VERSION + 1)
    with pytest.raises(AbiMismatch, match="ds4py build"):
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)

def test_unprobed_extension_binding_is_refused(make_manifest: Any, tmp_path: Path, upstream_headers: Any, ext_site: Any) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)        # probed before the extension was installed
    with pytest.raises(AbiMismatch, match="ds4_engine_layer_count"):
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=all_bindings(), layouts=LAYOUTS)

def test_absent_optional_binding_is_listed(make_manifest: Any, tmp_path: Path, upstream_headers: Any, ext_site: Any) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers, bindings=all_bindings())
    v = manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=all_bindings(), layouts=LAYOUTS)
    assert v.absent == {"ds4_engine_sampling_defaults"} and "ds4_engine_layer_count" in v.available

@pytest.mark.parametrize("records", [{}, None, []])
def test_metal_manifest_requires_shader_records(make_manifest: Any, tmp_path: Path, upstream_headers: Any, records: Any) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    m.backend = "metal"
    m.metal_sources = records
    with pytest.raises(AbiMismatch, match="Metal.*ds4py build"):
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)


def test_manifest_round_trips(make_manifest: Any, tmp_path: Path, upstream_headers: Any) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    m.write(tmp_path / "m.json")
    assert manifest.Manifest.read(tmp_path / "m.json") == m


@pytest.mark.parametrize("damage, match", [
    ("missing", "ds4_engine_options"),
    ("size", "ds4_engine_options.*size"),
    ("field", "ds4_engine_options.n_threads"),
])
def test_incomplete_or_wrong_layout_is_refused(make_manifest: Any, tmp_path: Path, upstream_headers: Any, damage: Any, match: Any) -> None:
    from ds4py._native.structs import StructLayout

    lib, m = make_manifest(tmp_path, upstream_headers)
    m.layouts = dict(m.layouts)
    layout = m.layouts["ds4_engine_options"]
    if damage == "missing":
        del m.layouts["ds4_engine_options"]
    elif damage == "size":
        m.layouts["ds4_engine_options"] = StructLayout(layout.size + 1, layout.fields)
    else:
        fields = dict(layout.fields)
        del fields["n_threads"]
        m.layouts["ds4_engine_options"] = StructLayout(layout.size, fields)
    with pytest.raises(AbiMismatch, match=match):
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=CORE_BINDINGS, layouts=LAYOUTS)


def test_failed_atomic_write_preserves_old_manifest(make_manifest: Any, tmp_path: Path, upstream_headers: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    path = tmp_path / "m.json"
    m.write(path)
    previous = path.read_bytes()

    def fail_replace(src: Any, dst: Any) -> Never:
        assert manifest.Manifest.read(src) == m
        assert src.parent == path.parent
        raise OSError("cannot replace")

    monkeypatch.setattr(manifest.os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        m.write(path)
    assert path.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {lib, path}


@pytest.mark.parametrize("ok", ["false", "true", 1, 0, {"ok": False}, [False], None, ""])
@pytest.mark.parametrize("required", [True, False])
def test_non_boolean_probe_result_is_refused(make_manifest: Any, tmp_path: Path, upstream_headers: Any, ok: Any, required: Any) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    binding = replace(CORE_BINDINGS[0], required=required)
    m.bindings[binding.name].ok = ok
    with pytest.raises(AbiMismatch, match=rf"{binding.name}.*boolean.*ds4py build"):
        manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=[binding], layouts=LAYOUTS)


@pytest.mark.parametrize("ok, required", [(True, True), (False, True), (True, False), (False, False)])
def test_boolean_probe_results_preserve_required_and_optional_semantics(
    make_manifest: Any, tmp_path: Path, upstream_headers: Any, ok: Any, required: Any,
) -> None:
    lib, m = make_manifest(tmp_path, upstream_headers)
    binding = replace(CORE_BINDINGS[0], required=required)
    m.bindings[binding.name].ok = ok
    path = tmp_path / "m.json"
    m.write(path)
    m = manifest.Manifest.read(path)
    if required and not ok:
        with pytest.raises(AbiMismatch, match=rf"{binding.name}.*required signature probe failed"):
            manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=[binding], layouts=LAYOUTS)
    else:
        v = manifest.validate(m, lib_sha256=manifest.file_sha256(lib), bindings=[binding], layouts=LAYOUTS)
        assert v.available == ({binding.name} if ok else set())
        assert v.absent == (set() if ok else {binding.name})
