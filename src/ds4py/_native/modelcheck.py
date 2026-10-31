"""Isolate ds4's inspection-time exits in a child process."""

import ctypes
import os
import subprocess
import sys
import traceback
from pathlib import Path

from ds4py.errors import EngineError, EngineLockedError, ModelFileError

from .protocol import EngineConfig

NATIVE_REFUSED = 3
PYTHON_ERROR = 4


def exit_error(status: int, stderr: str, model: Path, lock_file: Path) -> EngineError | None:
    if status == 0:
        return None
    lines = stderr.splitlines()
    native = [line[5:] for line in lines if line.startswith("ds4: ")]
    if native:
        if status == 2:
            return EngineLockedError(f"{lock_file}: {native[-1]}")
        if status == 1:
            return ModelFileError(f"{model}: {native[-1]}")
        if status == NATIVE_REFUSED:
            return EngineError(native[-1])
    return EngineError(f"model inspection exited with status {status}:\n" + "\n".join(lines[-20:]))


def inspect_model(lib: Path, config: EngineConfig, lock_file: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-P", "-m", "ds4py._native.modelcheck", str(lib), str(config.model), config.backend],
        env={**os.environ, "DS4_LOCK_FILE": str(lock_file)},
        capture_output=True,
        text=True,
    )
    error = exit_error(result.returncode, result.stderr, config.model, lock_file)
    if error is not None:
        raise error


def main(argv: list[str]) -> int:
    try:
        from .api import engine_options
        from .loader import load_library

        lib = load_library(Path(argv[0]))
        options = engine_options(EngineConfig(Path(argv[1]), backend=argv[2]), inspect_only=True)
        raw = ctypes.c_void_p()
        if lib.fn("ds4_engine_open")(ctypes.byref(raw), ctypes.byref(options)):
            return NATIVE_REFUSED
        lib.fn("ds4_engine_close")(raw)
        return 0
    except Exception:
        traceback.print_exc()
        return PYTHON_ERROR


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
