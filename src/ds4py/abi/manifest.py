"""Persist probe evidence and validate it against installed declarations."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from ds4py._native.bindings import Binding
from ds4py._native.structs import StructLayout
from ds4py._native.typemap import TYPE_MAP_VERSION
from ds4py.errors import AbiMismatch
from .pins import KNOWN_HEADERS

MANIFEST_NAME = "libds4.abi.json"
_REBUILD = " Run `ds4py build` to rebuild the library and its manifest. "


def manifest_path(lib: Path) -> Path:
    return lib.resolve().parent / MANIFEST_NAME


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class BindingRecord:
    prototype: str
    required: bool
    source: str
    ok: bool
    message: str = ""


@dataclass
class Manifest:
    type_map_version: int
    library_file: str
    library_sha256: str
    source_path: str
    source_commit: str | None
    backend: str
    headers: dict[str, str]
    headers_known: bool
    bindings: dict[str, BindingRecord]
    layouts: dict[str, StructLayout]
    live_checks: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        data = json.loads(path.read_text())
        data["bindings"] = {name: BindingRecord(**record)
                            for name, record in data["bindings"].items()}
        data["layouts"] = {name: StructLayout(**layout)
                           for name, layout in data["layouts"].items()}
        return cls(**data)

    def write(self, path: Path) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(asdict(self), stream, indent=2)
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Validation:
    available: frozenset[str]
    absent: frozenset[str]
    warnings: tuple[str, ...]


def validate(
    m: Manifest, *, lib_sha256: str, bindings: Sequence[Binding],
    layouts: dict[str, StructLayout],
) -> Validation:
    if m.type_map_version != TYPE_MAP_VERSION:
        raise AbiMismatch("type-map version differs from the installed version." + _REBUILD)
    if m.library_sha256 != lib_sha256:
        raise AbiMismatch("library SHA-256 differs from its manifest." + _REBUILD)
    available, absent = set(), set()
    for binding in bindings:
        record = m.bindings.get(binding.name)
        if record is None:
            raise AbiMismatch(f"{binding.name}: no probe result recorded." + _REBUILD)
        if record.prototype != binding.normalized:
            raise AbiMismatch(
                f"{binding.name}: the library was probed with\n  {record.prototype}"
                f"\nbut ds4py declares\n  {binding.normalized}" + _REBUILD
            )
        if not record.ok and binding.required:
            raise AbiMismatch(
                f"{binding.name}: required signature probe failed: {record.message}" + _REBUILD
            )
        (available if record.ok else absent).add(binding.name)
    for name, expected in layouts.items():
        actual = m.layouts.get(name)
        if actual is None:
            raise AbiMismatch(f"{name}: no layout recorded." + _REBUILD)
        if actual.size != expected.size:
            raise AbiMismatch(
                f"{name}: size {actual.size} differs from {expected.size}." + _REBUILD
            )
        for field_name, offset in expected.fields.items():
            recorded = actual.fields.get(field_name)
            if recorded != offset:
                raise AbiMismatch(
                    f"{name}.{field_name}: offset {recorded} differs from {offset}." + _REBUILD
                )
    warnings = tuple(
        f"unknown header {name}: {digest[:12]}"
        for name, digest in m.headers.items()
        if not any(known.get(name) == digest for known in KNOWN_HEADERS)
    )
    return Validation(frozenset(available), frozenset(absent), warnings)
