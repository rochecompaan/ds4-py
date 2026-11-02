"""Pseudo-terminal process control for bake-off acceptance checks."""

from __future__ import annotations

import json
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
from ds4py._json import JsonValue, is_json_object

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class _PtyChild:
    """Own one child process, its output, and its terminal cleanup."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None,
        env: Mapping[str, str] | None,
        timeout: float,
    ) -> None:
        self.deadline = time.monotonic() + timeout
        self.output = bytearray()
        self.wait_status: int | None = None
        self.pid, self.master = pty.fork()
        if self.pid == 0:
            try:
                if cwd is not None:
                    os.chdir(cwd)
                os.execvpe(command[0], list(command), os.environ.copy() if env is None else env)  # noqa: S606
            except BaseException as error:  # noqa: BLE001 - The forked child must report and exit, never raise into the parent.
                print(f"PTY exec failed: {error}", file=sys.stderr)
                os._exit(127)

    def remaining(self) -> float:
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("PTY timed out")
        return left

    def read_available(self, wait: float) -> bool:
        ready, _, _ = select.select([self.master], [], [], min(wait, 0.1))
        if not ready:
            return True
        try:
            data = os.read(self.master, 8192)
        except OSError:
            return False
        if not data:
            return False
        self.output.extend(data)
        return True

    def poll(self) -> int | None:
        if self.wait_status is None:
            done, status = os.waitpid(self.pid, os.WNOHANG)
            if done:
                self.wait_status = status
        return self.wait_status

    def write(self, text: str) -> None:
        os.write(self.master, text.encode("utf-8"))

    def exit_code(self) -> int:
        if self.wait_status is None:
            raise RuntimeError("process status requested before the child exited")
        return os.waitstatus_to_exitcode(self.wait_status)

    def terminate(self) -> None:
        if self.wait_status is None:
            with suppress(ProcessLookupError):
                os.kill(self.pid, signal.SIGKILL)
            with suppress(ChildProcessError):
                _, self.wait_status = os.waitpid(self.pid, 0)

    def close(self) -> None:
        os.close(self.master)

    def result(self, *, passed: bool) -> CheckResult:
        return CheckResult(passed, self.output.decode("utf-8", errors="replace")[-8000:])


class _EventFollower:
    """Read one append-only session event stream without consuming partial lines."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.path: Path | None = None
        self.offset = 0
        self.pending = b""

    def read(self) -> list[dict[str, JsonValue]]:
        if self.path is None:
            paths = sorted(self.home.glob("sessions/*/events.jsonl"))
            if len(paths) > 1:
                raise RuntimeError(f"expected one session event stream, found {len(paths)}")
            if not paths:
                return []
            self.path = paths[0]
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read()
                self.offset = stream.tell()
        except FileNotFoundError:
            return []
        self.pending += data
        *complete, self.pending = self.pending.split(b"\n")
        records = []
        for line in complete:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid complete session event: {error.msg}") from error
            if not is_json_object(record):
                raise RuntimeError("session event must be a JSON object")
            records.append(record)
        return records


def _run_id(record: dict[str, JsonValue]) -> str | None:
    value = record.get("run_id")
    return value if isinstance(value, str) else None


def _payload(record: dict[str, JsonValue]) -> dict[str, JsonValue]:
    value = record.get("payload")
    return value if is_json_object(value) else {}


def _stopped_interrupt(record: dict[str, JsonValue]) -> bool:
    if record.get("kind") != "Stopped":
        return False
    inner = _payload(record).get("event")
    return is_json_object(inner) and inner.get("reason") == "interrupted"


def drive_pty(
    command: Sequence[str],
    *,
    steps: Iterable[tuple[str, str | None]],
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 30,
) -> CheckResult:
    """Drive timed expect/send steps through a controlling pseudo-terminal."""
    child = _PtyChild(command, cwd=cwd, env=env, timeout=timeout)
    try:
        cursor = 0
        for pattern, response in steps:
            regex = re.compile(pattern, re.IGNORECASE)
            while True:
                text = child.output.decode("utf-8", errors="replace")
                match = regex.search(text, cursor)
                if match:
                    cursor = match.end()
                    break
                if child.poll() is not None:
                    raise RuntimeError(f"process exited while waiting for {pattern!r}")  # noqa: TRY301
                child.read_available(child.remaining())
            if response is not None:
                child.write(response)
        while child.poll() is None:
            if not child.read_available(child.remaining()) and child.poll() is not None:
                break
        # Drain anything written between the final expectation and process exit.
        while child.read_available(0):
            pass
        passed = child.exit_code() == 0
    except (OSError, RuntimeError, TimeoutError) as error:
        child.output.extend(f"\nPTY driver: {error}".encode())
        child.terminate()
        passed = False
    finally:
        child.close()
    return child.result(passed=passed)


def drive_interruption_pty(  # noqa: C901, PLR0912, PLR0913, PLR0915 - The structured lifecycle is one state machine.
    command: Sequence[str],
    *,
    home: str | Path,
    follow_up: str,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 30,
) -> CheckResult:
    """Interrupt real generation using session events, then prove a follow-up and exit."""
    child = _PtyChild(command, cwd=cwd, env=env, timeout=timeout)
    events = _EventFollower(Path(home))
    initial_run: str | None = None
    follow_up_run: str | None = None
    interrupted = False
    awaiting = "initial Token"
    try:
        while True:
            for record in events.read():
                run_id = _run_id(record)
                kind = record.get("kind")
                if initial_run is None:
                    if kind == "run_start" and run_id is not None:
                        initial_run = run_id
                        awaiting = "initial Token"
                    continue
                if run_id != initial_run and awaiting in {
                    "initial Token",
                    "interrupted Stopped",
                    "initial run_end",
                }:
                    continue
                if awaiting == "initial Token":
                    if kind == "Token":
                        child.write("\x03")
                        awaiting = "interrupted Stopped"
                elif awaiting == "interrupted Stopped":
                    if _stopped_interrupt(record):
                        interrupted = True
                        awaiting = "initial run_end"
                elif awaiting == "initial run_end":
                    if kind == "run_end":
                        if _payload(record).get("status") == "error":
                            raise RuntimeError("interrupted run ended with an error")  # noqa: TRY301
                        child.write(follow_up + "\n")
                        awaiting = "follow-up run_start"
                elif awaiting == "follow-up run_start":
                    if kind == "run_start" and run_id is not None and run_id != initial_run:
                        follow_up_run = run_id
                        awaiting = "follow-up run_end"
                elif awaiting == "follow-up run_end" and run_id == follow_up_run and kind == "run_end":
                    if _payload(record).get("status") != "success":
                        raise RuntimeError("follow-up run did not succeed")  # noqa: TRY301
                    child.write("/exit\n")
                    awaiting = "clean process exit"
            if child.poll() is not None:
                while child.read_available(0):
                    pass
                if awaiting == "clean process exit" and child.exit_code() == 0:
                    passed = True
                    break
                raise RuntimeError(f"process exited while waiting for {awaiting}")  # noqa: TRY301
            child.read_available(child.remaining())
    except (OSError, RuntimeError, TimeoutError) as error:
        suffix = f" while waiting for {awaiting}" if isinstance(error, TimeoutError) else ""
        child.output.extend(f"\nPTY driver: {error}{suffix}".encode())
        child.terminate()
        passed = False
    finally:
        child.close()
    if not interrupted:
        passed = False
    return child.result(passed=passed)
