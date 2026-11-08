"""Explicit live bakeoff acceptance; the live marker alone never starts it."""

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.run_bakeoff import AgentConfig, run_bakeoff
from bench.session import BakeoffSession
from bench.workspaces import load_manifest

pytestmark = pytest.mark.live


def test_complete_bakeoff_passes_all_six_gates() -> None:
    names = (
        "DS4_TEST_MODEL",
        "DS4_AGENT",
        "POLYGLOT_BENCHMARK",
        "DS4_BAKEOFF_INSTANCE_ID",
        "DS4_C_DRIVER",
    )
    required = {name: os.environ.get(name) for name in names}
    if os.environ.get("DS4_RUN_BAKEOFF") != "1" or not all(required.values()):
        pytest.skip("set DS4_RUN_BAKEOFF=1 and " + ", ".join(names) + " for the complete live bake-off")

    root = Path(__file__).resolve().parents[2]
    model = Path(cast("str", required["DS4_TEST_MODEL"])).expanduser().resolve()
    cli = Path(cast("str", required["DS4_AGENT"])).expanduser().resolve()
    c_driver = Path(cast("str", required["DS4_C_DRIVER"])).expanduser().resolve()
    manifest = load_manifest(root / "bench/bakeoff.yaml")
    common: dict[str, Any] = dict(  # noqa: C408 - Keyword-built agent context is clearer here.
        model=str(model),
        backend=os.environ.get("DS4_BAKEOFF_BACKEND", "cuda"),
        ctx=manifest.ctx,
        temperature=manifest.temperature,
        seed=manifest.seed,
        tokens=int(os.environ.get("DS4_BAKEOFF_TOKENS", "8192")),
    )
    configs = (
        AgentConfig("c", (str(cli),), **common),
        AgentConfig(
            "ds4py",
            (sys.executable, str(root / "bench/restore_check.py"), "attempt"),
            **common,
        ),
    )
    session = BakeoffSession.capture(
        configs,
        cast("str", required["DS4_BAKEOFF_INSTANCE_ID"]),
        repo_root=root,
    )
    results = Path(os.environ.get("DS4_BAKEOFF_RESULTS", str(root / "bench/results")))
    report = run_bakeoff(
        root / "bench/bakeoff.yaml",
        Path(cast("str", required["POLYGLOT_BENCHMARK"])).expanduser().resolve(),
        root,
        configs,
        results,
        session=session,
        c_driver=c_driver,
    )

    written = json.loads((results / "bakeoff.json").read_text())
    gates = written["gates"]
    assert set(gates) == {f"gate-{index}" for index in range(1, 7)}
    failed = {name: gate["detail"] for name, gate in gates.items() if not gate["passed"] or not gate["valid"]}
    assert not failed, f"live bakeoff gates failed: {failed}"
    assert written["schema_version"] == 2
    assert written["certification"] == "full-12-task"
    assert written["bakeoff_session_id"] == session.bakeoff_session_id
    assert written["instance_id"] == session.instance_id
    assert written["driver_equivalence"]["decode_steps"] > 0
    assert written["self_replay"]
    assert written["provenance"]
    assert report.passed
