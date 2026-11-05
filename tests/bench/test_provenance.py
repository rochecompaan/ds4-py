from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.provenance import (
    ProgressChunk,
    RoundProvenance,
    SyncCall,
    TraceProvenance,
    validate_pair,
    validate_session_pair,
    write_trace_provenance,
)


def _round() -> RoundProvenance:
    return RoundProvenance(
        round_index=0,
        think=75,
        prompt_sha256="prompt-sha",
        sync_calls=(
            SyncCall(label="prefix", input_tokens=8, cached_tokens=0, prefill_tokens=8, position=8),
            SyncCall(label="full", input_tokens=12, cached_tokens=8, prefill_tokens=4, position=12),
        ),
        progress_chunks=(ProgressChunk(done=8, total=8), ProgressChunk(done=4, total=4)),
    )


def _provenance() -> TraceProvenance:
    return TraceProvenance(
        bakeoff_session_id="session-1",
        instance_id="instance-1",
        gpu_model="NVIDIA H100",
        gpu_driver_version="580.00",
        ds4_revision="ds4-revision",
        ds4py_revision="ds4py-revision",
        libds4_sha256="library-sha",
        model_path="/models/ds4.gguf",
        model_sha256="model-sha",
        backend="cuda",
        ctx=100000,
        q8_f16_cache_mb=5120,
        temperature=0.0,
        seed=1,
        response_budget=8192,
        reset_policy="fresh-per-round",
        trace_sha256="c-output-sha",
        source_trace_sha256="source-sha",
        rounds=(_round(),),
    )


def test_provenance_rejects_different_instance_session_model_or_library() -> None:
    c = _provenance()
    ds4py = replace(
        c,
        instance_id="instance-2",
        bakeoff_session_id="session-2",
        model_sha256="other-model",
        libds4_sha256="other-library",
        trace_sha256="ds4py-output-sha",
    )

    issues = validate_session_pair(c, ds4py)
    text = "\n".join(issues)
    for field in ("instance_id", "bakeoff_session_id", "model_sha256", "libds4_sha256"):
        assert field in text


def test_provenance_rejects_unknown_identity_even_when_both_sides_match() -> None:
    c = replace(_provenance(), ds4_revision="unknown")

    assert "ds4_revision" in "\n".join(validate_session_pair(c, c))


def test_provenance_rejects_different_q8_f16_cache_budgets() -> None:
    c = _provenance()
    ds4py = replace(c, q8_f16_cache_mb=4096)

    assert "q8_f16_cache_mb" in "\n".join(validate_session_pair(c, ds4py))


def test_provenance_rejects_different_session_shape_or_missing_chunk_plan() -> None:
    c = _provenance()
    changed_round = replace(
        c.rounds[0],
        think=100,
        prompt_sha256="other-prompt",
        sync_calls=(c.rounds[0].sync_calls[0], replace(c.rounds[0].sync_calls[1], prefill_tokens=3, position=11)),
        progress_chunks=(),
    )
    ds4py = replace(
        c,
        source_trace_sha256="other-source",
        reset_policy="sequential",
        trace_sha256="ds4py-output-sha",
        rounds=(changed_round,),
    )

    text = "\n".join(validate_pair(c, ds4py))
    for field in (
        "source_trace_sha256",
        "reset_policy",
        "think",
        "prompt_sha256",
        "prefill_tokens",
        "position",
        "progress_chunks",
    ):
        assert field in text


def test_provenance_requires_matching_round_coverage_and_chunk_boundaries() -> None:
    c = _provenance()
    missing_round = replace(c, rounds=())
    changed_chunks = replace(c, rounds=(replace(c.rounds[0], progress_chunks=(ProgressChunk(done=4, total=4),)),))

    assert "rounds" in "\n".join(validate_pair(c, missing_round))
    assert "progress_chunks" in "\n".join(validate_pair(c, changed_chunks))


def test_session_pair_allows_different_task_prompts_and_trace_hashes() -> None:
    c = _provenance()
    ds4py = replace(
        c,
        trace_sha256="ds4py-output-sha",
        source_trace_sha256="different-task-history",
        reset_policy="sequential",
        rounds=(replace(c.rounds[0], prompt_sha256="different-prompt", progress_chunks=()),),
    )

    assert validate_session_pair(c, ds4py) == ()


def test_trace_provenance_sidecar_round_trip(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"kind":"Stopped"}\n')
    provenance = _provenance()

    output = write_trace_provenance(trace, provenance)

    assert output == tmp_path / "trace.jsonl.provenance.json"
    data = json.loads(output.read_text())
    assert data["trace_sha256"] == "c-output-sha"
    assert data["source_trace_sha256"] == "source-sha"
    assert data["q8_f16_cache_mb"] == 5120
    assert data["rounds"][0]["think"] == 75
    assert data["rounds"][0]["sync_calls"][1]["position"] == 12
