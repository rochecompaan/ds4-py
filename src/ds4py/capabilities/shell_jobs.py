"""Keep background shell jobs alive across prompts and stop them with the session."""

import json
import os
import signal
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

from pydantic_ai.capabilities import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import RunContext, ToolDefinition

    from ds4py.engine.session import Session

_TERM_GRACE = 2.0
_KILL_WAIT = 2.0
_POLL = 0.05
_HANDLE_FIELDS = 2


class ShellJobState:
    """Session-scoped set of ``start_command`` IDs with best-effort cleanup."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.job_ids: set[str] = set()

    def add(self, job_id: str) -> None:
        self.job_ids.add(job_id)

    def close(self) -> None:
        for job_id in tuple(self.job_ids):
            base = self.workspace_dir / ".pydantic-ai-harness" / "shell" / job_id
            handle = base / "handle"
            if not handle.exists():
                continue
            try:
                fields = handle.read_text().split()
                if len(fields) != _HANDLE_FIELDS or not fields[0].isdigit():
                    continue
                pid = int(fields[0])
                group = int(fields[1]) if fields[1].isdigit() else None
                if _published_exit(base / "status.json") is not None:
                    # The command ended, but children left in its group may survive it.
                    if group is not None and _alive(pid, group):
                        _signal(pid, group, signal.SIGKILL)
                        _wait_for_exit(pid, group, _KILL_WAIT)
                    continue
                # The wrapper checks this marker before it starts the command;
                # it also closes the fork/signal race the pinned harness documents.
                with suppress(OSError):
                    (base / "stop").touch()
                _signal(pid, group, signal.SIGTERM)
                if _wait_for_exit(pid, group, _TERM_GRACE):
                    continue
                _signal(pid, group, signal.SIGKILL)
                _wait_for_exit(pid, group, _KILL_WAIT)
            except OSError, ValueError:
                continue


class DS4ShellJobs(AbstractCapability):
    """Record ``start_command`` job IDs so session end can stop them."""

    def __init__(self, session: Session, workspace: object) -> None:
        self.session = session
        self.workspace = workspace

    def _state(self) -> ShellJobState:
        workspace_dir = getattr(self.workspace, "working_dir", self.workspace)
        return self.session.state(
            "shell_jobs",
            lambda: ShellJobState(cast("str | Path", workspace_dir)),
        )

    @override
    async def after_tool_execute(
        self,
        ctx: RunContext[object],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, object],
        result: object,
    ) -> object:
        if call.tool_name == "start_command" and isinstance(result, str):
            for line in result.splitlines():
                if line.startswith("ID: "):
                    job_id = line[4:].strip()
                    if job_id:
                        self._state().add(job_id)
                    break
        return result


def _published_exit(status_path: Path) -> int | None:
    if not status_path.exists():
        return None
    try:
        value = json.loads(status_path.read_text())
    except OSError, ValueError:
        return None
    if not isinstance(value, dict):
        return None
    exit_code = value.get("exit_code")
    return exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None


def _signal(pid: int, group: int | None, number: int) -> None:
    if group is not None:
        os.killpg(group, number)
    else:
        os.kill(pid, number)


def _alive(pid: int, group: int | None) -> bool:
    try:
        if group is not None:
            os.killpg(group, 0)
        else:
            os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_exit(pid: int, group: int | None, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if not _alive(pid, group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL)
