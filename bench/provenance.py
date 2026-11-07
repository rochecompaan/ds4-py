"""Host, model, library, and prefill provenance for bake-off comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


Q8_F16_CACHE_ENV = "DS4_CUDA_Q8_F16_CACHE_MB"
Q8_F16_CACHE_MB = 5120
NATIVE_PATCHES_ENV = "DS4_BAKEOFF_NATIVE_PATCHES"
CUDA_DISABLE_HOST_REGISTER_ENV = "DS4_CUDA_DISABLE_HOST_REGISTER"


@dataclass(frozen=True)
class NativePatch:
    name: str
    sha256: str


@dataclass(frozen=True)
class SyncCall:
    label: str
    input_tokens: int
    cached_tokens: int
    prefill_tokens: int
    position: int


@dataclass(frozen=True)
class ProgressChunk:
    done: int
    total: int


@dataclass(frozen=True)
class RoundProvenance:
    round_index: int
    think: int
    prompt_sha256: str
    sync_calls: tuple[SyncCall, ...]
    progress_chunks: tuple[ProgressChunk, ...]
    observed_think: int | None = None


@dataclass(frozen=True)
class TraceProvenance:
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
    q8_f16_cache_mb: int
    temperature: float
    seed: int
    response_budget: int
    reset_policy: str
    trace_sha256: str
    source_trace_sha256: str
    rounds: tuple[RoundProvenance, ...]
    observed_libds4_sha256: str | None = None
    observed_model_path: str | None = None
    observed_backend: str | None = None
    observed_ctx: int | None = None
    observed_seed: int | None = None
    observed_temperature: float | None = None
    native_patches: tuple[NativePatch, ...] = ()
    cuda_disable_host_register: str = ""
    invalid_reason: str | None = None


_IDENTITY_FIELDS = (
    "bakeoff_session_id",
    "instance_id",
    "gpu_model",
    "gpu_driver_version",
    "ds4_revision",
    "ds4py_revision",
    "libds4_sha256",
    "model_path",
    "model_sha256",
)

_SESSION_FIELDS = (
    "bakeoff_session_id",
    "instance_id",
    "gpu_model",
    "gpu_driver_version",
    "ds4_revision",
    "ds4py_revision",
    "libds4_sha256",
    "model_path",
    "model_sha256",
    "backend",
    "ctx",
    "q8_f16_cache_mb",
    "temperature",
    "seed",
    "response_budget",
    "native_patches",
    "cuda_disable_host_register",
)

_OBSERVED_FIELDS = ("libds4_sha256", "model_path", "backend", "ctx", "seed", "temperature")
_OBSERVED_PROVENANCE_FIELDS = tuple(f"observed_{field}" for field in _OBSERVED_FIELDS)


def write_trace_provenance(trace_path: Path, provenance: TraceProvenance) -> Path:
    """Write the sidecar associated with one raw trace."""
    output = trace_path.with_name(f"{trace_path.name}.provenance.json")
    output.write_text(json.dumps(asdict(provenance), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _field_mismatches(c: object, ds4py: object, fields: tuple[str, ...], prefix: str = "") -> list[str]:
    issues: list[str] = []
    for field in fields:
        c_value = getattr(c, field)
        ds4py_value = getattr(ds4py, field)
        if c_value != ds4py_value:
            issues.append(f"{prefix}{field} differs: C={c_value!r}, ds4-py={ds4py_value!r}")
    return issues


def _has_observed_identity(provenance: TraceProvenance) -> bool:
    return any(getattr(provenance, field) is not None for field in _OBSERVED_PROVENANCE_FIELDS)


def _validate_observed_identity(label: str, provenance: TraceProvenance) -> list[str]:
    if not _has_observed_identity(provenance):
        return []
    issues = []
    for field in _OBSERVED_FIELDS:
        observed = getattr(provenance, f"observed_{field}")
        expected = getattr(provenance, field)
        if observed is None:
            issues.append(f"{label} observed_{field} is missing")
        elif observed != expected:
            issues.append(
                f"{label} observed_{field} differs from expected: observed={observed!r}, expected={expected!r}"
            )
    return issues


def validate_session_pair(c: TraceProvenance, ds4py: TraceProvenance) -> tuple[str, ...]:
    """Return missing identity and every host, model, library, and sampling mismatch."""
    issues = [
        f"{label} {provenance.invalid_reason}"
        for label, provenance in (("C", c), ("ds4-py", ds4py))
        if provenance.invalid_reason is not None
    ]
    issues.extend(_validate_observed_identity("C", c))
    issues.extend(_validate_observed_identity("ds4-py", ds4py))
    if _has_observed_identity(c) and _has_observed_identity(ds4py):
        issues.extend(_field_mismatches(c, ds4py, _OBSERVED_PROVENANCE_FIELDS))
    issues.extend(_field_mismatches(c, ds4py, _SESSION_FIELDS))
    for field in _IDENTITY_FIELDS:
        for label, provenance in (("C", c), ("ds4-py", ds4py)):
            value = getattr(provenance, field)
            if not value or value == "unknown":
                issues.append(f"{label} {field} is missing")
    return tuple(issues)


def validate_pair(c: TraceProvenance, ds4py: TraceProvenance) -> tuple[str, ...]:
    """Return every provenance mismatch that invalidates a direct pair."""
    issues = list(validate_session_pair(c, ds4py))
    for label, provenance in (("C", c), ("ds4-py", ds4py)):
        if not _has_observed_identity(provenance):
            issues.append(f"{label} observed identity is missing")
        for round_ in provenance.rounds:
            if round_.observed_think is None:
                issues.append(f"{label} rounds[{round_.round_index}].observed_think is missing")
            elif round_.observed_think != round_.think:
                issues.append(
                    f"{label} rounds[{round_.round_index}].observed_think differs from expected: "
                    f"observed={round_.observed_think!r}, expected={round_.think!r}"
                )
    issues.extend(_field_mismatches(c, ds4py, ("source_trace_sha256", "reset_policy")))
    if len(c.rounds) != len(ds4py.rounds):
        issues.append(f"rounds length differs: C={len(c.rounds)}, ds4-py={len(ds4py.rounds)}")

    for c_round, ds4py_round in zip(c.rounds, ds4py.rounds, strict=False):
        prefix = f"rounds[{c_round.round_index}]."
        issues.extend(
            _field_mismatches(c_round, ds4py_round, ("round_index", "think", "observed_think", "prompt_sha256"), prefix)
        )
        if len(c_round.sync_calls) != len(ds4py_round.sync_calls):
            issues.append(
                f"{prefix}sync_calls length differs: C={len(c_round.sync_calls)}, ds4-py={len(ds4py_round.sync_calls)}"
            )
        for call_index, (c_call, ds4py_call) in enumerate(
            zip(c_round.sync_calls, ds4py_round.sync_calls, strict=False)
        ):
            issues.extend(
                _field_mismatches(
                    c_call,
                    ds4py_call,
                    ("label", "input_tokens", "cached_tokens", "prefill_tokens", "position"),
                    f"{prefix}sync_calls[{call_index}].",
                )
            )
        if c_round.progress_chunks is None or ds4py_round.progress_chunks is None:
            issues.append(f"{prefix}progress_chunks are required for a reference-driver pair")
        elif c_round.progress_chunks != ds4py_round.progress_chunks:
            issues.append(
                f"{prefix}progress_chunks differ: C={c_round.progress_chunks!r}, ds4-py={ds4py_round.progress_chunks!r}"
            )
    return tuple(issues)
