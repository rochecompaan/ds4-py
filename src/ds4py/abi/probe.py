"""Compile isolated signature checks and measure public C struct layouts."""

import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ds4py._native.structs import STRUCTS, StructLayout, struct_layout

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ds4py._native.bindings import Binding
from ds4py._native.typemap import parse_prototype, probe_declaration

from .pins import HEADER_FILES, KNOWN_HEADERS


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    message: str


def header_hashes(include_dir: Path) -> dict[str, str]:
    return {name: hashlib.sha256((include_dir / name).read_bytes()).hexdigest() for name in HEADER_FILES}


def headers_known(hashes: dict[str, str]) -> bool:
    return hashes in KNOWN_HEADERS


def probe_signatures(
    include_dir: Path,
    bindings: Sequence[Binding],
    cc: str,
    workdir: Path,
) -> dict[str, ProbeResult]:
    """Compile each declaration independently, retaining compiler diagnostics."""
    workdir.mkdir(parents=True, exist_ok=True)

    def compile_binding(binding: Binding) -> tuple[str, ProbeResult]:
        source = workdir / f"{binding.name}.c"
        declaration = probe_declaration(parse_prototype(binding.prototype))
        source.write_text(f'#include "ds4.h"\n#include "ds4_kvstore.h"\n{declaration} = {binding.name};\n')
        result = subprocess.run(
            [
                cc,
                "-std=c11",
                "-c",
                "-o",
                "/dev/null",
                "-Werror=incompatible-pointer-types",
                "-Werror=implicit-function-declaration",
                f"-I{include_dir}",
                str(source),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return binding.name, ProbeResult(result.returncode == 0, result.stderr)

    with ThreadPoolExecutor() as executor:
        return dict(executor.map(compile_binding, bindings))


def probe_layouts(include_dir: Path, cc: str, workdir: Path) -> dict[str, StructLayout]:
    """Build and run a program reporting sizeof and nested offsetof values."""
    workdir.mkdir(parents=True, exist_ok=True)
    lines = [
        '#include "ds4.h"',
        '#include "ds4_kvstore.h"',
        "#include <stddef.h>",
        "#include <stdio.h>",
        "int main(void) {",
    ]
    for name, cls in STRUCTS.items():
        lines.append(f'printf("{name} size %zu\\n", sizeof({name}));')
        for field in struct_layout(cls).fields:
            lines.append(f'printf("{name} field {field} %zu\\n", offsetof({name}, {field}));')
    lines.append("return 0;\n}")
    source = workdir / "layouts.c"
    executable = (workdir / "layouts").resolve()
    source.write_text("\n".join(lines) + "\n")
    subprocess.run(
        [cc, "-std=c11", f"-I{include_dir}", str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        check=True,
    )
    result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        check=True,
    )
    layouts = {}
    for line in result.stdout.splitlines():
        name, kind, *values = line.split()
        if kind == "size":
            layouts[name] = StructLayout(int(values[0]), {})
        else:
            field, offset = values
            layouts[name].fields[field] = int(offset)
    return layouts
