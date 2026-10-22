from pathlib import Path
import shutil

import pytest

from ds4py.abi import probe
from ds4py._native.bindings import CORE_BINDINGS, discover_extension_bindings
from ds4py._native.structs import EngineOptions, STRUCTS, struct_layout


HEADERS = Path(__file__).parents[1] / "fixtures" / "ds4-headers"


@pytest.fixture
def upstream_headers():
    return HEADERS / "upstream"


@pytest.fixture
def fork_headers():
    return HEADERS / "fork"


def doctored(tmp_path, old, new):
    include_dir = tmp_path / "include"
    shutil.copytree(HEADERS / "upstream", include_dir)
    header = include_dir / "ds4.h"
    text = header.read_text()
    assert text.count(old) == 1
    header.write_text(text.replace(old, new, 1))
    return include_dir


def test_core_bindings_compile_against_both_pins(cc, tmp_path, upstream_headers, fork_headers):
    for inc in (upstream_headers, fork_headers):
        results = probe.probe_signatures(inc, CORE_BINDINGS, cc, tmp_path / inc.name)
        assert set(results) == {binding.name for binding in CORE_BINDINGS}
        assert {n for n, r in results.items() if not r.ok} == set()


def test_fork_only_binding_fails_upstream(cc, tmp_path, upstream_headers, fork_headers, ext_site):
    ext = discover_extension_bindings()
    up = probe.probe_signatures(upstream_headers, ext, cc, tmp_path / "u")
    fk = probe.probe_signatures(fork_headers, ext, cc, tmp_path / "f")
    assert up["ds4_engine_layer_count"].ok and fk["ds4_engine_sampling_defaults"].ok
    assert not up["ds4_engine_sampling_defaults"].ok
    assert "ds4_engine_sampling_defaults" in up["ds4_engine_sampling_defaults"].message


def test_changed_argument_type_fails_its_probe(cc, tmp_path):
    inc = doctored(tmp_path, "int ds4_session_eval(ds4_session *s, int token,",
                             "int ds4_session_eval(ds4_session *s, long token,")
    results = probe.probe_signatures(inc, CORE_BINDINGS, cc, tmp_path / "w")
    assert not results["ds4_session_eval"].ok
    assert "incompatible" in results["ds4_session_eval"].message
    assert all(r.ok for n, r in results.items() if n != "ds4_session_eval")


def test_ctypes_structs_match_the_compiled_layout(cc, tmp_path, upstream_headers, fork_headers):
    for inc in (upstream_headers, fork_headers):
        compiled = probe.probe_layouts(inc, cc, tmp_path / inc.name)
        assert compiled == {name: struct_layout(cls) for name, cls in STRUCTS.items()}


def test_moved_field_changes_the_compiled_layout(cc, tmp_path):
    inc = doctored(tmp_path, "    int n_threads;\n    int context_size;",
                             "    int context_size;\n    int n_threads;")
    compiled = probe.probe_layouts(inc, cc, tmp_path / "w")
    assert compiled["ds4_engine_options"] != struct_layout(EngineOptions)


def test_header_hashes(upstream_headers, fork_headers):
    assert probe.headers_known(probe.header_hashes(upstream_headers))
    assert not probe.headers_known(probe.header_hashes(fork_headers))
