"""Compiled boundary double shared by the native adapter tests."""
from typing import Any, cast
from collections.abc import Iterator

from pathlib import Path

import pytest


@pytest.fixture
def stub_engine(cc: Any, upstream_headers: Any, tmp_path: Path) -> Iterator[Any]:
    # A compiled boundary double is needed because real sessions require a model.
    import ctypes
    import subprocess
    from types import SimpleNamespace
    from ds4py._native.api import InstanceLock, NativeEngine
    from ds4py._native.bindings import CORE_BINDINGS
    from ds4py._native.loader import Library
    from ds4py._native.structs import EngineOptions

    path = tmp_path / "stub.so"
    subprocess.run([cc, "-shared", "-fPIC", "-I", str(upstream_headers),
                    str(Path(__file__).with_name("api_stub.c")), "-o", str(path)], check=True)
    handle = ctypes.CDLL(str(path))
    bindings = [b for b in CORE_BINDINGS if hasattr(handle, b.name)]
    lib = Library(
        path, cast(Any, None),
        cast(Any, SimpleNamespace(available={b.name for b in bindings}, absent=set())),
        handle, bindings,
    )
    engine = NativeEngine(lib, 0x1234, EngineOptions(), InstanceLock.acquire(tmp_path / "lock"))
    yield engine, handle
    engine.close()
