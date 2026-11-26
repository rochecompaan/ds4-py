"""Compiled boundary double shared by the native adapter tests."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest


def _stub_engine(cc: Any, upstream_headers: Any, tmp_path: Path, *, mellum: bool) -> Iterator[Any]:
    # A compiled boundary double is needed because real sessions require a model.
    import ctypes
    import subprocess
    from types import SimpleNamespace

    from ds4py._native.api import InstanceLock, NativeEngine
    from ds4py._native.bindings import CORE_BINDINGS
    from ds4py._native.loader import Library
    from ds4py._native.structs import EngineOptions

    path = tmp_path / ("stub-mellum.so" if mellum else "stub.so")
    subprocess.run(
        [
            cc,
            "-shared",
            "-fPIC",
            "-I",
            str(upstream_headers),
            *(["-DSTUB_MELLUM"] if mellum else []),
            str(Path(__file__).with_name("api_stub.c")),
            "-o",
            str(path),
        ],
        check=True,
    )
    handle = ctypes.CDLL(str(path))
    bindings = [b for b in CORE_BINDINGS if hasattr(handle, b.name)]
    lib = Library(
        path,
        cast("Any", None),
        cast("Any", SimpleNamespace(available={b.name for b in bindings}, absent=set())),
        handle,
        bindings,
    )
    lock = InstanceLock.acquire(tmp_path / ("mellum.lock" if mellum else "lock"))
    engine = NativeEngine(lib, 0x1234, EngineOptions(), lock)
    yield engine, handle
    engine.close()


@pytest.fixture
def stub_engine(cc: Any, upstream_headers: Any, tmp_path: Path) -> Iterator[Any]:
    yield from _stub_engine(cc, upstream_headers, tmp_path, mellum=False)


@pytest.fixture
def stub_mellum_engine(cc: Any, upstream_headers: Any, tmp_path: Path) -> Iterator[Any]:
    yield from _stub_engine(cc, upstream_headers, tmp_path, mellum=True)
