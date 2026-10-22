"""Load a checked library and render its ABI diagnostics without a model."""

from dataclasses import dataclass
from pathlib import Path

from ds4py._native.loader import load_library, resolve_lib_path
from ds4py.errors import AbiError
from .manifest import Manifest, Validation


@dataclass
class CheckReport:
    lib_path: Path
    manifest: Manifest | None
    validation: Validation | None
    extensions: dict[str, str]
    error: str | None


def check_library(lib: str | Path | None) -> CheckReport:
    path = resolve_lib_path(lib)
    try:
        library = load_library(lib)
    except AbiError as exc:
        return CheckReport(path, None, None, {}, str(exc))
    checked = library.validation.available | library.validation.absent
    extensions = {name: record.source for name, record in library.manifest.bindings.items()
                  if name in checked and record.source != "core"}
    return CheckReport(library.path, library.manifest, library.validation, extensions, None)


def format_report(r: CheckReport) -> str:
    if r.error is not None:
        return f"library:  {r.lib_path}\nerror:    {r.error}"
    m, v = r.manifest, r.validation
    assert m is not None and v is not None
    tree = (f"{m.source_path} at {m.source_commit[:12]}" if m.source_commit is not None
            else f"{m.source_path} (not a git checkout)")
    lines = [f"library:  {r.lib_path} ({m.backend})", f"ds4 tree: {tree}",
             f"headers:  {'known' if m.headers_known else 'unknown'}"]
    for warning in v.warnings:
        lines.append(
            f"          warning: {warning.replace(': ', ' ', 1)}; "
            "bindings and layouts were checked, but this tree is not a tested pin"
        )
    required = sum(m.bindings[name].required for name in v.available)
    lines.extend(["layouts:  ok", f"bindings: {required} required ok"])
    optional = [name for name in m.bindings
                if name in v.available | v.absent
                and (not m.bindings[name].required or name in r.extensions)]
    for name in optional:
        status = "absent" if name in v.absent else "present"
        source = r.extensions.get(name, m.bindings[name].source)
        lines.append(f"optional: {name}  {status}  ({source})")
    if not optional:
        lines.append("optional: none")
    lines.append("restore:  not verified (run `ds4py check --live --model PATH`)")
    return "\n".join(lines)
