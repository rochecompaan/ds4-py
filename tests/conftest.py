import importlib
import os
from pathlib import Path
import shutil
from importlib.metadata import MetadataPathFinder

import pytest


@pytest.fixture(scope="session")
def built_upstream(tmp_path_factory):
    from ds4py.abi.build import build_library
    from ds4py.abi.manifest import Manifest, manifest_path

    src = os.environ.get("DS4_SRC_UPSTREAM")
    if not src:
        pytest.skip("DS4_SRC_UPSTREAM is not set")
    lib = build_library(Path(src), backend="cpu", out_dir=tmp_path_factory.mktemp("upstream-build"))
    return lib, Manifest.read(manifest_path(lib))


@pytest.fixture(scope="session")
def built_fork(tmp_path_factory):
    from ds4py.abi.build import build_library
    from ds4py.abi.manifest import Manifest, manifest_path

    src = os.environ.get("DS4_SRC_FORK")
    if not src:
        pytest.skip("DS4_SRC_FORK is not set")
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str((Path(__file__).parent / "fixtures" / "ext_site").resolve()))
        importlib.invalidate_caches()
        MetadataPathFinder().invalidate_caches()
        try:
            lib = build_library(Path(src), backend="cpu", out_dir=tmp_path_factory.mktemp("fork-build"))
            return lib, Manifest.read(manifest_path(lib))
        finally:
            MetadataPathFinder().invalidate_caches()


@pytest.fixture
def cc():
    compiler = os.environ.get("CC") or shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler not available")
    return compiler


@pytest.fixture
def ext_site(monkeypatch):
    site = Path(__file__).parent / "fixtures" / "ext_site"
    monkeypatch.syspath_prepend(str(site.resolve()))
    importlib.invalidate_caches()
    MetadataPathFinder().invalidate_caches()
    yield site
    MetadataPathFinder().invalidate_caches()


@pytest.fixture
def upstream_headers():
    return Path(__file__).parent / "fixtures" / "ds4-headers" / "upstream"


@pytest.fixture
def fork_headers():
    return Path(__file__).parent / "fixtures" / "ds4-headers" / "fork"


@pytest.fixture(scope="session")
def probed():
    return {}


@pytest.fixture
def make_manifest(cc, probed, tmp_path_factory):
    from ds4py.abi import probe
    from ds4py.abi.manifest import BindingRecord, Manifest, file_sha256
    from ds4py._native.bindings import CORE_BINDINGS
    from ds4py._native.loader import LIB_NAME
    from ds4py._native.typemap import TYPE_MAP_VERSION

    def make(where, headers_dir, bindings=CORE_BINDINGS):
        lib = where / LIB_NAME
        lib.write_bytes(b"dummy library")
        key = (headers_dir.resolve(), tuple(b.name for b in bindings))
        if key not in probed:
            work = tmp_path_factory.mktemp("probe")
            probed[key] = (
                probe.probe_signatures(headers_dir, bindings, cc, work / "signatures"),
                probe.probe_layouts(headers_dir, cc, work / "layouts"),
            )
        signatures, layouts = probed[key]
        hashes = probe.header_hashes(headers_dir)
        return lib, Manifest(
            type_map_version=TYPE_MAP_VERSION,
            library_file=LIB_NAME, library_sha256=file_sha256(lib),
            source_path=str(headers_dir), source_commit=None, backend="cpu",
            headers=hashes, headers_known=probe.headers_known(hashes),
            bindings={b.name: BindingRecord(b.normalized, b.required, b.source,
                                          signatures[b.name].ok, signatures[b.name].message)
                      for b in bindings},
            layouts=layouts,
        )

    return make


@pytest.fixture
def fresh_loader():
    from ds4py._native import loader
    loader._loaded = None
    yield
    loader._loaded = None


@pytest.fixture(scope="session")
def live_lib():
    from ds4py._native.loader import resolve_lib_path

    if not os.environ.get("DS4_TEST_MODEL"):
        pytest.skip("DS4_TEST_MODEL is not set")
    lib = resolve_lib_path(None)
    if not lib.is_file():
        pytest.skip(f"library not found: {lib}")
    return lib
