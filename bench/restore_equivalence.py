"""Run fresh-process restore drivers and capture C equivalence evidence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.restore_c_reference import prepare_c_rebuild, write_c_rebuild_metadata
from bench.restore_fidelity import _step_from_metadata
from bench.restore_plan import CReferenceRun, RestoreError, RestoreRun

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.restore_plan import RestoreConfig, RestoreOperations


def _restore_run(trace_path: Path) -> RestoreRun:
    metadata = trace_path.with_suffix(".json")
    value = json.loads(metadata.read_text())
    return RestoreRun(
        trace_path=trace_path,
        terminal_reason=value["terminal_reason"],
        switch_prefill=value["switch_prefill"],
        followup_plan=value["followup_plan"],
        followup_position=value["followup_position"],
        first_step=_step_from_metadata(value["first_step"]),
        provenance_path=metadata,
        followup_prefill=value["followup_prefill"],
    )


@dataclass(frozen=True)
class ProcessRestoreOperations:
    config: RestoreConfig
    workspace: Path | None = None
    provenance_environment: Mapping[str, str] | None = None
    c_driver: str | Path | None = None

    def _run(
        self,
        command: list[str],
        *,
        home: Path,
        trace_path: Path,
        operation: str,
    ) -> subprocess.CompletedProcess[str]:
        source_workspace = (self.workspace if self.workspace is not None else Path.cwd()).resolve()
        environment = os.environ.copy()
        if self.provenance_environment is not None:
            environment.update(self.provenance_environment)
        environment["DS4PY_HOME"] = str(home.resolve())
        with tempfile.TemporaryDirectory(prefix=f"ds4py-restore-{operation}-") as temporary:
            restore_workspace = Path(temporary) / "workspace"
            shutil.copytree(source_workspace, restore_workspace)
            completed = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                cwd=restore_workspace,
                check=False,
            )
        trace_path.with_suffix(".log").write_text((completed.stdout or "") + (completed.stderr or ""))
        if completed.returncode:
            raise RestoreError(f"{operation} exited {completed.returncode}")
        return completed

    def _python_command(
        self,
        operation: Literal["restore", "same-history"],
        *,
        session_id: str,
        trace_path: Path,
    ) -> list[str]:
        executable = self.config.executable[:-1] if self.config.executable[-1] == "attempt" else self.config.executable
        return [
            *executable,
            operation,
            "--session-id",
            session_id,
            "--metadata",
            str(trace_path.with_suffix(".json")),
            "--model",
            self.config.model,
            "--backend",
            self.config.backend,
            "--ctx",
            str(self.config.ctx),
            "--temp",
            str(self.config.temperature),
            "--seed",
            str(self.config.seed),
            "--request-limit",
            str(self.config.request_limit),
            "--tokens",
            str(self.config.tokens),
            "--trace",
            str(trace_path),
        ]

    def restore(self, home: Path, session_id: str, trace_path: Path) -> RestoreRun:
        trace_path = trace_path.resolve()
        command = self._python_command("restore", session_id=session_id, trace_path=trace_path)
        self._run(command, home=home, trace_path=trace_path, operation="restore")
        return _restore_run(trace_path)

    def replay_same_history(
        self,
        home: Path,
        session_id: str,
        trace_path: Path,
        restore_trace: Path,
    ) -> RestoreRun:
        trace_path = trace_path.resolve()
        command = self._python_command("same-history", session_id=session_id, trace_path=trace_path)
        command.extend(["--restore-trace", str(restore_trace.resolve())])
        self._run(command, home=home, trace_path=trace_path, operation="same-history")
        return _restore_run(trace_path)

    def c_rebuild(self, home: Path, case2_trace: Path, trace_path: Path) -> CReferenceRun:
        if self.c_driver is None:
            raise RestoreError("Gate 5 rebuild equivalence requires the C replay driver")
        case2_trace = case2_trace.resolve()
        trace_path = trace_path.resolve()
        command = prepare_c_rebuild(self.config, self.c_driver, case2_trace, trace_path)
        completed = self._run(command, home=home, trace_path=trace_path, operation="c-rebuild")
        trace_path.write_text(completed.stdout or "")
        return write_c_rebuild_metadata(case2_trace, trace_path, trace_path.with_suffix(".json"))


def process_restore_runner(
    config: RestoreConfig,
    *,
    workspace: Path | None = None,
    provenance_environment: Mapping[str, str] | None = None,
    c_driver: str | Path | None = None,
) -> RestoreOperations:
    """Return explicit operations that run in disposable task-workspace copies."""
    return ProcessRestoreOperations(config, workspace, provenance_environment, c_driver)
