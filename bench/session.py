"""One-session host, model, library, and trace provenance."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bench.driver_cases import load_case_tokens
from bench.provenance import (
    CUDA_DISABLE_HOST_REGISTER_ENV,
    NATIVE_PATCHES_ENV,
    Q8_F16_CACHE_ENV,
    Q8_F16_CACHE_MB,
    NativePatch,
    RoundProvenance,
    TraceProvenance,
    write_trace_provenance,
)
from ds4py._native.loader import resolve_lib_path
from ds4py.abi.manifest import file_sha256
from ds4py.engine.replay import read_replay

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from bench.driver_cases import DriverRoundEvidence
    from bench.models import AgentConfig, TaskRunResult


def token_sha256(tokens: Sequence[int]) -> str:
    """Hash canonical token IDs without token text or prompt rendering."""
    payload = json.dumps(list(tokens), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def native_patch_provenance(repo_root: Path, specification: str) -> tuple[NativePatch, ...]:
    """Hash repo-relative native patch paths in stable name order."""
    patches = []
    for name in specification.split(os.pathsep):
        if not name:
            continue
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("native patch paths must be relative to the ds4-py repository")
        patches.append(NativePatch(name=path.as_posix(), sha256=file_sha256(repo_root / path)))
    return tuple(sorted(patches, key=lambda patch: patch.name))


def _git_revision(path: Path) -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    completed = subprocess.run(  # noqa: S603 - Fixed git command with a repository path.
        [git, "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _gpu_details(backend: str) -> tuple[str, str]:
    from_env = (
        os.environ.get("DS4_BAKEOFF_GPU_MODEL"),
        os.environ.get("DS4_BAKEOFF_GPU_DRIVER_VERSION"),
    )
    if all(from_env):
        return from_env[0] or "unknown", from_env[1] or "unknown"
    if backend != "cuda":
        return "cpu", "not-applicable"
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return "unknown", "unknown"
    try:
        completed = subprocess.run(  # noqa: S603 - Fixed nvidia-smi query without shell input.
            [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except OSError, subprocess.SubprocessError:
        return "unknown", "unknown"
    if completed.returncode != 0 or not completed.stdout.strip():
        return "unknown", "unknown"
    model, separator, driver = completed.stdout.splitlines()[0].partition(",")
    return (model.strip(), driver.strip()) if separator else ("unknown", "unknown")


@dataclass(frozen=True)
class BakeoffSession:
    """Immutable inputs shared by every phase of one bake-off."""

    bakeoff_session_id: str
    instance_id: str
    gpu_model: str
    gpu_driver_version: str
    ds4_revision: str
    ds4py_revision: str
    libds4_sha256: str
    model_path: str
    model_sha256: str
    backend: str
    ctx: int
    temperature: float
    seed: int
    response_budget: int
    q8_f16_cache_mb: int
    native_patches: tuple[NativePatch, ...] = ()
    cuda_disable_host_register: str = ""

    def __post_init__(self) -> None:
        if not self.bakeoff_session_id or not self.instance_id:
            raise ValueError("bake-off session and instance IDs are required")

    @classmethod
    def capture(
        cls,
        configs: Sequence[AgentConfig],
        instance_id: str,
        *,
        repo_root: Path,
    ) -> BakeoffSession:
        """Hash the selected artifacts once and capture host/revision identity."""
        c = next(config for config in configs if config.agent == "c")
        python = next(config for config in configs if config.agent == "ds4py")
        shared = ("model", "backend", "ctx", "temperature", "seed", "tokens")
        if any(getattr(c, name) != getattr(python, name) for name in shared):
            raise ValueError("C and ds4-py configurations differ")
        model = Path(python.model).expanduser().resolve(strict=True)
        library = resolve_lib_path(None).expanduser().resolve(strict=True)
        gpu_model, gpu_driver = _gpu_details(python.backend)
        return cls(
            bakeoff_session_id=uuid.uuid4().hex,
            instance_id=instance_id,
            gpu_model=gpu_model,
            gpu_driver_version=gpu_driver,
            ds4_revision=os.environ.get("DS4_BAKEOFF_DS4_REVISION")
            or _git_revision(Path(c.executable[0]).expanduser().resolve().parent),
            ds4py_revision=os.environ.get("DS4_BAKEOFF_DS4PY_REVISION") or _git_revision(repo_root),
            libds4_sha256=file_sha256(library),
            model_path=str(model),
            model_sha256=file_sha256(model),
            backend=python.backend,
            ctx=python.ctx,
            temperature=python.temperature,
            seed=python.seed,
            response_budget=python.tokens,
            q8_f16_cache_mb=Q8_F16_CACHE_MB,
            native_patches=native_patch_provenance(repo_root, os.environ.get(NATIVE_PATCHES_ENV, "")),
            cuda_disable_host_register=os.environ.get(CUDA_DISABLE_HOST_REGISTER_ENV, ""),
        )

    def environment(self) -> dict[str, str]:
        """Return the provenance inherited by fresh restore processes."""
        return {
            "BAKEOFF_SESSION_ID": self.bakeoff_session_id,
            "DS4_BAKEOFF_INSTANCE_ID": self.instance_id,
            "DS4_BAKEOFF_GPU_MODEL": self.gpu_model,
            "DS4_BAKEOFF_GPU_DRIVER_VERSION": self.gpu_driver_version,
            "DS4_BAKEOFF_DS4_REVISION": self.ds4_revision,
            "DS4_BAKEOFF_DS4PY_REVISION": self.ds4py_revision,
            "DS4_BAKEOFF_LIBDS4_SHA256": self.libds4_sha256,
            "DS4_BAKEOFF_MODEL_SHA256": self.model_sha256,
            Q8_F16_CACHE_ENV: str(self.q8_f16_cache_mb),
            NATIVE_PATCHES_ENV: os.pathsep.join(patch.name for patch in self.native_patches),
            CUDA_DISABLE_HOST_REGISTER_ENV: self.cuda_disable_host_register,
        }

    def provenance(
        self,
        trace_path: Path,
        source_trace_path: Path,
        *,
        rounds: Sequence[RoundProvenance],
        reset_policy: str,
    ) -> TraceProvenance:
        return TraceProvenance(
            bakeoff_session_id=self.bakeoff_session_id,
            instance_id=self.instance_id,
            gpu_model=self.gpu_model,
            gpu_driver_version=self.gpu_driver_version,
            ds4_revision=self.ds4_revision,
            ds4py_revision=self.ds4py_revision,
            libds4_sha256=self.libds4_sha256,
            model_path=self.model_path,
            model_sha256=self.model_sha256,
            backend=self.backend,
            ctx=self.ctx,
            q8_f16_cache_mb=self.q8_f16_cache_mb,
            temperature=self.temperature,
            seed=self.seed,
            response_budget=self.response_budget,
            reset_policy=reset_policy,
            trace_sha256=file_sha256(trace_path),
            source_trace_sha256=file_sha256(source_trace_path),
            rounds=tuple(rounds),
            native_patches=self.native_patches,
            cuda_disable_host_register=self.cuda_disable_host_register,
        )

    def task_provenance(self, trace_path: Path) -> TraceProvenance:
        trace = read_replay(trace_path)
        rounds = tuple(
            RoundProvenance(round_.number, round_.think, token_sha256(round_.prompt), (), ()) for round_ in trace.rounds
        )
        return self.provenance(trace_path, trace_path, rounds=rounds, reset_policy="sequential")

    def reference_provenance(
        self,
        output_path: Path,
        source_trace_path: Path,
        evidence: Sequence[DriverRoundEvidence],
    ) -> TraceProvenance:
        rounds = tuple(
            RoundProvenance(
                round_index=item.case.round_index,
                think=item.case.think,
                prompt_sha256=token_sha256(load_case_tokens(item.case)[0]),
                sync_calls=item.sync_calls,
                progress_chunks=item.progress_chunks,
            )
            for item in evidence
        )
        return self.provenance(
            output_path,
            source_trace_path,
            rounds=rounds,
            reset_policy="fresh-per-round",
        )


def report_provenance_environment(data: Mapping[str, object]) -> dict[str, str]:
    """Recover process provenance from a schema-2 report for standalone rechecks."""
    pairs = data.get("provenance")
    if not isinstance(pairs, list) or not pairs or not isinstance(pairs[0], dict):
        return {}
    c = pairs[0].get("c")
    if not isinstance(c, dict):
        return {}
    names = {
        "BAKEOFF_SESSION_ID": "bakeoff_session_id",
        "DS4_BAKEOFF_INSTANCE_ID": "instance_id",
        "DS4_BAKEOFF_GPU_MODEL": "gpu_model",
        "DS4_BAKEOFF_GPU_DRIVER_VERSION": "gpu_driver_version",
        "DS4_BAKEOFF_DS4_REVISION": "ds4_revision",
        "DS4_BAKEOFF_DS4PY_REVISION": "ds4py_revision",
        "DS4_BAKEOFF_LIBDS4_SHA256": "libds4_sha256",
        "DS4_BAKEOFF_MODEL_SHA256": "model_sha256",
        Q8_F16_CACHE_ENV: "q8_f16_cache_mb",
        CUDA_DISABLE_HOST_REGISTER_ENV: "cuda_disable_host_register",
    }
    environment: dict[str, str] = {}
    for env_name, field in names.items():
        value = data.get(field) if field in {"bakeoff_session_id", "instance_id"} else c.get(field)
        if field == "q8_f16_cache_mb" and isinstance(value, int) and value > 0:
            environment[env_name] = str(value)
        elif isinstance(value, str) and value:
            environment[env_name] = value
    patches = c.get("native_patches")
    if isinstance(patches, list):
        names: list[str] = []
        for patch in patches:
            if isinstance(patch, dict):
                name = patch.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
        if len(names) == len(patches):
            environment[NATIVE_PATCHES_ENV] = os.pathsep.join(names)
    return environment


def record_task_provenance(
    tasks: Sequence[TaskRunResult],
    session: BakeoffSession,
) -> dict[Path, TraceProvenance]:
    """Write one sidecar per raw attempt trace and return the same records."""
    recorded: dict[Path, TraceProvenance] = {}
    for item in tasks:
        for raw_path in item.attempt_trace_paths:
            path = Path(raw_path)
            provenance = session.task_provenance(path)
            write_trace_provenance(path, provenance)
            recorded[path] = provenance
    return recorded
