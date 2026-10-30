"""Build a source mirror and publish immutable library/manifest generations."""

import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from ds4py._native.bindings import Binding, all_bindings
from ds4py._native.loader import LIB_NAME
from ds4py._native.structs import STRUCTS, struct_layout
from ds4py._native.typemap import TYPE_MAP_VERSION
from ds4py.errors import AbiMismatch, BuildError

from . import probe
from .manifest import BindingRecord, Manifest, file_sha256, manifest_path, validate
from .shaders import copy_metal_sources, metal_source_env


def default_backend() -> str:
    return "metal" if sys.platform == "darwin" else "cpu"


def _preflight(src: Path, backend: str, cc: str | None) -> str:
    for name in ("Makefile", "ds4.h", "ds4_kvstore.h", "ds4.c"):
        if not (src / name).is_file():
            raise BuildError(f"{src} is not a ds4 source tree: missing {name}")
    compiler = shutil.which(cc or os.environ.get("CC") or "cc")
    if compiler is None:
        raise BuildError("no C compiler: set CC or put cc on PATH")
    if shutil.which("make") is None:
        raise BuildError("make not found on PATH")
    if backend == "rocm":
        raise BuildError("backend rocm is not supported by ds4py build yet")
    if backend == "metal" and sys.platform != "darwin":
        raise BuildError("backend metal needs macOS")
    if backend not in ("cpu", "metal", "cuda"):
        raise BuildError(f"unknown backend: {backend}")
    return str(Path(compiler).resolve())


def _mirror(src: Path, farm: Path) -> None:
    """Keep build products in real directories, never in the source checkout."""
    src = src.resolve()
    farm = farm.resolve()
    farm.mkdir(parents=True, exist_ok=True)
    for directory, dirs, names in os.walk(src):
        dirs[:] = [name for name in dirs if name != ".git" and Path(directory) / name != farm]
        target = farm / Path(directory).relative_to(src)
        # A previously mirrored file may now be a source directory. Never
        # traverse its stale symlink, which points back into the source tree.
        if target.is_symlink():
            target.unlink()
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            if name == ".git" or Path(name).suffix in (".o", ".a", ".so", ".dylib"):
                continue
            link = target / name
            if not link.is_symlink() and link.is_dir():
                shutil.rmtree(link)
            else:
                link.unlink(missing_ok=True)
            link.symlink_to(Path(directory) / name)


def _run_make(farm: Path, backend: str, cc: str, jobs: int | None, log: TextIO) -> None:
    make = shutil.which("make")
    if make is None:
        raise BuildError("make not found on PATH")
    command = [
        make,
        "-f",
        "Makefile.lib",
        "libds4",
        f"LIBDS4_BACKEND={backend}",
        f"CC={cc}",
        f"-j{jobs or os.cpu_count() or 1}",
    ]
    tail: deque[str] = deque(maxlen=40)
    with subprocess.Popen(command, cwd=farm, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        if process.stdout is None:
            raise BuildError("make stdout is unavailable")
        for line in process.stdout:
            log.write(line)
            log.flush()
            tail.append(line)
        if process.wait():
            raise BuildError("make failed:\n" + "".join(tail))


def install(
    lib: Path,
    manifest: Manifest,
    out_dir: Path,
    *,
    metal_root: Path | None = None,
) -> Path:
    """Publish one bundle; caller holds .build.lock. Never prune reader generations.

    Offline cleanup requires stopping all readers, engine processes and builds;
    retain the directory named by the current library symlink.
    """
    installs = out_dir / "installs"
    installs.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(tempfile.mkdtemp(dir=installs))
    temporary = out_dir / f".{LIB_NAME}.new"
    switched: bool = False
    try:
        staged = staged_dir / LIB_NAME
        shutil.copy2(lib, staged)
        manifest.write(manifest_path(staged))
        try:
            if manifest.backend == "metal":
                if metal_root is None:
                    raise BuildError("Metal sources are missing. Run `ds4py build` to rebuild.")
                hashes = copy_metal_sources(metal_root, staged_dir)
                if hashes != manifest.metal_sources:
                    raise BuildError("Metal sources differ from the manifest. Run `ds4py build` to rebuild.")
                metal_source_env(staged_dir, manifest.metal_sources)
            validate(
                Manifest.read(manifest_path(staged)),
                lib_sha256=file_sha256(staged),
                bindings=all_bindings(),
                layouts={name: struct_layout(cls) for name, cls in STRUCTS.items()},
            )
        except AbiMismatch as exc:
            raise BuildError(str(exc)) from exc
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(Path("installs") / staged_dir.name / LIB_NAME)
        Path(temporary).replace(out_dir / LIB_NAME)
        switched = True
        return out_dir / LIB_NAME
    finally:
        if not switched:  # ty: ignore[redundant-condition-strict]  # Exceptions bypass the normal assignment.
            temporary.unlink(missing_ok=True)
            # replace can publish before an interrupt prevents recording switched.
            # Never delete a generation that is current or cannot be ruled out.
            try:
                published = (out_dir / LIB_NAME).resolve() == staged_dir.resolve() / LIB_NAME
            except OSError, RuntimeError:
                published = True
            if not published:
                shutil.rmtree(staged_dir)


def _build_manifest(  # noqa: PLR0913, PLR0917 - Manifest construction mirrors the native build context.
    src: Path,
    lib: Path,
    backend: str,
    cc: str,
    work: Path,
    bindings: Sequence[Binding],
    signatures: Mapping[str, probe.ProbeResult],
    log: TextIO,
    metal_sources: dict[str, str],
) -> Manifest:
    layouts = probe.probe_layouts(src, cc, work / "layouts")
    hashes = probe.header_hashes(src)
    known = probe.headers_known(hashes)
    if not known:
        print("warning: unknown ds4 header hashes", file=log)
    try:
        git = shutil.which("git")
        if git is None:
            raise BuildError("git not found on PATH")
        commit = subprocess.run(
            [git, "-C", str(src), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        commit = None
    return Manifest(
        type_map_version=TYPE_MAP_VERSION,
        library_file=LIB_NAME,
        library_sha256=file_sha256(lib),
        source_path=str(src),
        source_commit=commit,
        backend=backend,
        headers=hashes,
        headers_known=known,
        bindings={
            b.name: BindingRecord(b.normalized, b.required, b.source, signatures[b.name].ok, signatures[b.name].message)
            for b in bindings
        },
        layouts=layouts,
        metal_sources=metal_sources,
    )


def build_library(  # noqa: PLR0913 - Build options mirror the public CLI.
    src: Path,
    *,
    backend: str,
    out_dir: Path,
    jobs: int | None = None,
    cc: str | None = None,
    log: TextIO = sys.stderr,
) -> Path:
    compiler = _preflight(src, backend, cc)
    src = src.resolve()
    bindings = all_bindings()
    with tempfile.TemporaryDirectory(prefix="ds4py-probe-") as directory:
        work = Path(directory)
        signatures = probe.probe_signatures(src, bindings, compiler, work / "signatures")
        failed = [
            f"{b.name}: {signatures[b.name].message}" for b in bindings if b.required and not signatures[b.name].ok
        ]
        if failed:
            raise BuildError("required signature probes failed:\n" + "\n".join(failed))
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / ".build.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BuildError(f"another ds4py build is using {out_dir}") from exc
            key = hashlib.sha256(os.fsencode(src)).hexdigest()[:12]
            farm = out_dir / "build" / f"{backend}-{key}"
            _mirror(src, farm)
            (farm / "Makefile.lib").unlink(missing_ok=True)
            (farm / "Makefile.lib").write_bytes(files(__package__).joinpath("Makefile.lib").read_bytes())
            metal_root = work / "sources" if backend == "metal" else None
            metal_sources = copy_metal_sources(src, metal_root) if metal_root is not None else {}
            _run_make(farm, backend, compiler, jobs, log)
            try:
                manifest = _build_manifest(
                    src, farm / LIB_NAME, backend, compiler, work, bindings, signatures, log, metal_sources
                )
            except subprocess.CalledProcessError as exc:
                raise BuildError(f"layout probe failed: {exc.stderr}") from exc
            return install(farm / LIB_NAME, manifest, out_dir, metal_root=metal_root)
