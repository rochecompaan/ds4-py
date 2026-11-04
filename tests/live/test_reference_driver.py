"""Opt-in one-trace reference-driver check with session provenance."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.models import AgentConfig
from bench.run_bakeoff import run_reference_equivalence
from bench.session import BakeoffSession
from bench.workspaces import load_manifest

pytestmark = pytest.mark.live


def test_retained_c_trace_matches_reference_drivers() -> None:
    names = (
        "DS4_TEST_MODEL",
        "DS4_REPLAY_TRACE",
        "DS4_BAKEOFF_INSTANCE_ID",
        "DS4_C_DRIVER",
    )
    required = {name: os.environ.get(name) for name in names}
    if os.environ.get("DS4_RUN_DRIVER_EQUIVALENCE") != "1" or not all(required.values()):
        pytest.skip("set DS4_RUN_DRIVER_EQUIVALENCE=1 and " + ", ".join(names) + " for driver equivalence")

    root = Path(__file__).resolve().parents[2]
    model = Path(cast("str", required["DS4_TEST_MODEL"])).expanduser().resolve()
    trace = Path(cast("str", required["DS4_REPLAY_TRACE"])).expanduser().resolve()
    c_driver = Path(cast("str", required["DS4_C_DRIVER"])).expanduser().resolve()
    manifest = load_manifest(root / "bench/bakeoff.yaml")
    common: dict[str, Any] = dict(  # noqa: C408 - Shared reference-driver configuration is clearer here.
        model=str(model),
        backend=os.environ.get("DS4_BAKEOFF_BACKEND", "cuda"),
        ctx=manifest.ctx,
        temperature=manifest.temperature,
        seed=manifest.seed,
        tokens=8192,
    )
    configs = (
        AgentConfig("c", (str(c_driver),), **common),
        AgentConfig("ds4py", (sys.executable, "unused"), **common),
    )
    session = BakeoffSession.capture(
        configs,
        cast("str", required["DS4_BAKEOFF_INSTANCE_ID"]),
        repo_root=root,
    )
    task = SimpleNamespace(
        task_id=trace.parent.parent.name,
        agent="c",
        attempts=1,
        attempt_trace_paths=(trace,),
    )
    results = (
        Path(os.environ.get("DS4_DRIVER_EQUIVALENCE_RESULTS", str(root / "bench/driver-equivalence-results")))
        / session.bakeoff_session_id
    )
    phase = run_reference_equivalence(
        cast("Any", [task]),
        configs[1],
        c_driver,
        session,
        results,
    )

    assert phase.result.valid, phase.result.detail
    assert phase.result.passed, phase.result.detail
    assert phase.result.decode_steps > 0
    assert phase.provenance
    assert len(list(results.rglob("*.provenance.json"))) == 2
