"""Resolve one immutable library generation and bind only validated symbols."""

import ctypes
import os
from pathlib import Path
import sys
import threading
from collections.abc import Callable, Iterable
from typing import Any

from ds4py.abi.manifest import Manifest, Validation, file_sha256, manifest_path, validate
from ds4py.abi.shaders import metal_source_env
from ds4py.errors import AbiMismatch, LibraryAlreadyLoaded, LibraryNotFound, MissingBinding
from .bindings import Binding, all_bindings
from .structs import STRUCTS, struct_layout
from .typemap import ctypes_signature, parse_prototype

LIB_NAME = "libds4.dylib" if sys.platform == "darwin" else "libds4.so"
_REBUILD = " Run `ds4py build` to rebuild the library and its manifest. "
_loaded: Library | None = None
_load_lock = threading.Lock()


def default_lib_dir() -> Path:
    root = os.environ.get("DS4PY_HOME")
    return (Path(root).expanduser() if root else Path.home() / ".ds4py") / "lib"


def resolve_lib_path(lib: str | Path | None) -> Path:
    if lib is not None:
        return Path(lib).expanduser()
    if "DS4_LIB" in os.environ:
        return Path(os.environ["DS4_LIB"]).expanduser()
    return default_lib_dir() / LIB_NAME


def _dlopen(path: Path) -> ctypes.CDLL:
    return ctypes.CDLL(str(path), mode=os.RTLD_LOCAL)


class Library:
    def __init__(
        self,
        path: Path,
        manifest: Manifest,
        validation: Validation,
        handle: ctypes.CDLL,
        bindings: Iterable[Binding],
    ) -> None:
        self.path = path
        self.manifest = manifest
        self.validation = validation
        self._handle = handle
        self._functions: dict[str, Callable[..., Any]] = {}
        for binding in bindings:
            if binding.name not in validation.available:
                continue
            try:
                fn = getattr(handle, binding.name)
            except AttributeError as exc:
                raise AbiMismatch(f"{binding.name}: symbol missing from {path}." + _REBUILD) from exc
            fn.restype, fn.argtypes = ctypes_signature(parse_prototype(binding.prototype))
            self._functions[binding.name] = fn
        try:
            self._free = handle.free
        except AttributeError as exc:
            raise AbiMismatch(f"free: symbol missing from {path}." + _REBUILD) from exc
        self._free.argtypes = [ctypes.c_void_p]
        self._free.restype = None

    def fn(self, name: str) -> Callable[..., Any]:
        if name in self.validation.absent:
            raise MissingBinding(f"{name} is absent from this library")
        if name not in self._functions:
            raise MissingBinding(f"ds4py does not declare {name}")
        return self._functions[name]

    def has(self, name: str) -> bool:
        return name in self._functions

    def free(self, ptr: int | ctypes.c_void_p | None) -> None:
        self._free(ptr)


def load_library(lib: str | Path | None = None) -> Library:
    global _loaded
    path = resolve_lib_path(lib)
    try:
        real = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LibraryNotFound(
            f"library not found: {path}. Run `ds4py build --src <ds4 tree>` to build it."
        ) from exc
    with _load_lock:
        if _loaded is not None:
            if _loaded.path != real:
                raise LibraryAlreadyLoaded(
                    f"library already loaded from {_loaded.path}; cannot load {real}. Restart the process."
                )
            return _loaded
        manifest_file = manifest_path(real)
        try:
            manifest = Manifest.read(manifest_file)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise AbiMismatch(f"cannot read {manifest_file}: {exc}." + _REBUILD) from exc
        bindings = all_bindings()
        layouts = STRUCTS
        validation = validate(
            manifest, lib_sha256=file_sha256(real), bindings=bindings,
            layouts={name: struct_layout(cls) for name, cls in layouts.items()},
        )
        if manifest.backend == "metal":
            metal_source_env(real.parent, manifest.metal_sources, os.environ)
        try:
            handle = _dlopen(real)
        except OSError as exc:
            raise AbiMismatch(f"cannot load {real}: {exc}." + _REBUILD) from exc
        loaded = Library(real, manifest, validation, handle, bindings)
        _loaded = loaded
        return loaded
