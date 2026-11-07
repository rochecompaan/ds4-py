"""Run fresh-process restore drivers and capture C equivalence evidence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.restore_c_reference import prepare_c_rebuild, write_c_rebuild_metadata
from bench.restore_fidelity import _step_from_metadata
from bench.restore_plan import RestoreError, RestoreRun

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.restore_plan import RestoreConfig, RestoreRunner


def process_restore_runner(
    config: RestoreConfig,
    *,
    workspace: Path | None = None,
    provenance_environment: Mapping[str, str] | None = None,
    c_driver: str | Path | None = None,
) -> RestoreRunner:
    """Return a runner that starts each restore from a disposable task-workspace copy."""

    def run(
        home: Path,
        session_id: str,
        case: int,
        trace_path: Path,
    ) -> RestoreRun:
        home = home.resolve()
        trace_path = trace_path.resolve()
        metadata = trace_path.with_suffix(".json")
        source_workspace = (workspace if workspace is not None else Path.cwd()).resolve()
        executable = config.executable[:-1] if config.executable[-1] == "attempt" else config.executable
        case2_trace = trace_path.with_name("case-2.jsonl")
        if case == 4:
            if c_driver is None:
                raise RestoreError("Gate 5 rebuild equivalence requires the C replay driver")
            command = prepare_c_rebuild(config, c_driver, case2_trace, trace_path)
        else:
            command = [
                *executable,
                "same-history" if case == 3 else "restore",
                "--session-id",
                session_id,
                "--metadata",
                str(metadata),
                "--model",
                config.model,
                "--backend",
                config.backend,
                "--ctx",
                str(config.ctx),
                "--temp",
                str(config.temperature),
                "--seed",
                str(config.seed),
                "--request-limit",
                str(config.request_limit),
                "--tokens",
                str(config.tokens),
                "--trace",
                str(trace_path),
            ]
            if case == 3:
                command.extend(["--restore-trace", str(trace_path.with_name("case-1.jsonl"))])
        environment = os.environ.copy()
        if provenance_environment is not None:
            environment.update(provenance_environment)
        environment["DS4PY_HOME"] = str(home)
        with tempfile.TemporaryDirectory(prefix=f"ds4py-restore-case-{case}-") as temporary:
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
            raise RestoreError(f"restore case {case} exited {completed.returncode}")
        if case == 4:
            trace_path.write_text(completed.stdout or "")
            write_c_rebuild_metadata(case2_trace, trace_path, metadata)
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

    return run
