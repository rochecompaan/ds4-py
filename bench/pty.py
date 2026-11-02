"""Pseudo-terminal process control for bake-off acceptance checks."""

from __future__ import annotations

import os
import pty
import re
import select
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from bench.restore_check import CheckResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


def drive_pty(  # noqa: C901, PLR0912, PLR0915 - A PTY session is one stateful lifecycle.
    command: Sequence[str],
    *,
    steps: Iterable[tuple[str, str | None]],
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 30,
) -> CheckResult:
    """Drive timed expect/send steps through a controlling pseudo-terminal."""
    deadline = time.monotonic() + timeout
    output = bytearray()
    pid, master = pty.fork()
    if pid == 0:
        try:
            if cwd is not None:
                os.chdir(cwd)
            os.execvpe(command[0], list(command), os.environ.copy() if env is None else env)
        except BaseException as error:  # noqa: BLE001 - The forked child must report and exit, never raise into the parent.
            print(f"PTY exec failed: {error}", file=sys.stderr)
            os._exit(127)
    wait_status = None

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("PTY timed out")
        return left

    def read_available(wait: float) -> bool:
        ready, _, _ = select.select([master], [], [], min(wait, 0.1))
        if not ready:
            return True
        try:
            data = os.read(master, 8192)
        except OSError:
            return False
        if not data:
            return False
        output.extend(data)
        return True

    try:
        cursor = 0
        for pattern, response in steps:
            regex = re.compile(pattern, re.IGNORECASE)
            while True:
                text = output.decode("utf-8", errors="replace")
                match = regex.search(text, cursor)
                if match:
                    cursor = match.end()
                    break
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    wait_status = status
                    raise RuntimeError(f"process exited while waiting for {pattern!r}")  # noqa: TRY301
                read_available(remaining())
            if response is not None:
                os.write(master, response.encode("utf-8"))
        while wait_status is None:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                wait_status = status
                break
            if not read_available(remaining()):
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    wait_status = status
        # Drain anything written between the final expectation and process exit.
        while read_available(0):
            pass
        code = os.waitstatus_to_exitcode(wait_status)
        passed = code == 0
    except (OSError, RuntimeError, TimeoutError) as error:
        output.extend(f"\nPTY driver: {error}".encode())
        passed = False
        if wait_status is None:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            with suppress(ChildProcessError):
                _, wait_status = os.waitpid(pid, 0)
    finally:
        os.close(master)
    detail = output.decode("utf-8", errors="replace")[-8000:]
    return CheckResult(passed, detail)
