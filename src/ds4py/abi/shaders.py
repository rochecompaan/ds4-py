"""Copy and verify the Metal source bundle; never change the process environment."""

from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
import re
import shutil

from ds4py.errors import AbiMismatch, BuildError
from .manifest import file_sha256

_REBUILD = " Run `ds4py build` to rebuild the library and its Metal sources."


def copy_metal_sources(source_root: Path, destination_root: Path) -> dict[str, str]:
    """Capture real file bytes, not links back into the source checkout."""
    try:
        sources = sorted((source_root / "metal").glob("*.metal"))
        if not sources:
            raise BuildError(f"no Metal sources in {source_root / 'metal'}." + _REBUILD)
        destination = destination_root / "metal"
        destination.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for source in sources:
            target = destination / source.name
            shutil.copy2(source, target)
            hashes[f"metal/{source.name}"] = file_sha256(target)
        return hashes
    except (OSError, RuntimeError) as exc:
        raise BuildError(f"cannot copy Metal sources: {exc}." + _REBUILD) from exc


def _records(hashes: Mapping[str, str]) -> Iterator[tuple[str, str]]:
    if not isinstance(hashes, Mapping) or not hashes:
        raise AbiMismatch("Metal shader records are missing or invalid." + _REBUILD)
    for relative, digest in hashes.items():
        if not isinstance(relative, str):
            raise AbiMismatch("Metal shader paths must be relative strings." + _REBUILD)
        path = PurePosixPath(relative)
        if (len(path.parts) != 2 or path.parts[0] != "metal"
                or path.suffix != ".metal" or path.as_posix() != relative):
            raise AbiMismatch(f"invalid Metal shader path: {relative!r}." + _REBUILD)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise AbiMismatch(f"invalid Metal shader hash for {relative}." + _REBUILD)
        yield relative, digest


def _check_hash(path: Path, digest: str, label: str) -> None:
    try:
        if not path.is_file():
            raise AbiMismatch(f"{label} is not a regular Metal source file." + _REBUILD)
        actual = file_sha256(path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise AbiMismatch(f"cannot read {label}: {exc}." + _REBUILD) from exc
    if actual != digest:
        raise AbiMismatch(f"{label}: Metal shader SHA-256 differs from the manifest." + _REBUILD)


def metal_source_env(
    lib_dir: Path, hashes: Mapping[str, str], overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Validate all bundled and effective files, preserving explicit override strings."""
    result = {}
    root = lib_dir.resolve()
    for relative, digest in _records(hashes):
        try:
            bundled = (root / relative).resolve(strict=True)
        except (OSError, ValueError, RuntimeError) as exc:
            raise AbiMismatch(f"cannot resolve Metal shader {relative}: {exc}." + _REBUILD) from exc
        if not bundled.is_relative_to(root):
            raise AbiMismatch(f"Metal shader {relative} escapes its install directory." + _REBUILD)
        _check_hash(bundled, digest, relative)
        key = f"DS4_METAL_{Path(relative).stem.upper()}_SOURCE"
        if overrides is not None and key in overrides:
            value = overrides[key]
            _check_hash(Path(value), digest, key)
            result[key] = value
        else:
            result[key] = str(bundled)
    return result
