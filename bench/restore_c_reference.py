"""C split-zero reference evidence for the Gate 5 rebuilt restore path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Protocol

from bench.driver_cases import export_trace_cases, read_driver_ndjson
from bench.provenance import TraceProvenance
from ds4py.engine.replay import read_replay


class CConfig(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def backend(self) -> str: ...

    @property
    def ctx(self) -> int: ...


class CRebuildError(ValueError):
    """C rebuild evidence is absent, malformed, or inconsistent."""


def prepare_c_rebuild(
    config: CConfig,
    c_driver: str | Path,
    case2_trace: Path,
    output_trace: Path,
) -> list[str]:
    """Export case 2's exact first prompt as a one-step split-zero C case."""
    trace = read_replay(case2_trace)
    if not trace.rounds or not trace.rounds[0].sampled:
        raise CRebuildError("case 2 needs a prompt and first sampled token for the C reference")
    first = replace(trace.rounds[0], cached=0, sampled=trace.rounds[0].sampled[:1])
    reference = replace(trace, rounds=(first,))
    manifest = export_trace_cases(
        reference,
        task_id="gate-5-rebuild",
        attempt=1,
        destination=output_trace.parent / "case-4-driver",
        max_steps=1,
    )
    return [
        str(c_driver),
        "--model",
        config.model,
        "--cases",
        str(manifest),
        "--source-trace",
        str(case2_trace),
        "--backend",
        config.backend,
        "--ctx",
        str(config.ctx),
    ]


def write_c_rebuild_metadata(case2_trace: Path, output_trace: Path, metadata: Path) -> None:
    """Validate one C round and write the common restore-run metadata schema."""
    rounds = read_driver_ndjson(output_trace)
    if len(rounds) != 1 or len(rounds[0].steps) != 1:
        raise CRebuildError("C rebuild reference must contain exactly one decode step")
    evidence = rounds[0]
    if evidence.case.split_at != 0 or evidence.case.steps != 1:
        raise CRebuildError("C rebuild reference did not use split-at 0 with one step")
    case2_metadata = json.loads(case2_trace.with_suffix(".json").read_text())
    provenance_value = case2_metadata["provenance"]
    provenance_value["rounds"] = ()
    provenance = replace(
        TraceProvenance(**provenance_value),
        reset_policy="c-rebuild-split-0",
        trace_sha256=hashlib.sha256(output_trace.read_bytes()).hexdigest(),
        source_trace_sha256=hashlib.sha256(case2_trace.read_bytes()).hexdigest(),
    )
    capture = evidence.steps[0]
    metadata.write_text(
        json.dumps(
            {
                "switch_prefill": 0,
                "followup_plan": "c-rebuild",
                "followup_prefill": sum(call.prefill_tokens for call in evidence.sync_calls),
                "followup_position": capture.position,
                "first_step": asdict(capture),
                "terminal_reason": evidence.terminal_reason,
                "trace_path": str(output_trace),
                "provenance": asdict(provenance),
                "save_record_sha256": case2_metadata["save_record_sha256"],
                "followup_prompt_sha256": case2_metadata["followup_prompt_sha256"],
            }
        )
        + "\n"
    )
