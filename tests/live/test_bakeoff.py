"""Explicit live bakeoff acceptance; the live marker alone never starts it."""

import json
import os
from pathlib import Path
import sys
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.workspaces import load_manifest
from bench.run_bakeoff import AgentConfig, run_bakeoff


pytestmark = pytest.mark.live


def test_complete_bakeoff_passes_all_six_gates() -> None:
    required = {
        name: os.environ.get(name)
        for name in ("DS4_TEST_MODEL", "DS4_AGENT", "POLYGLOT_BENCHMARK")
    }
    if os.environ.get("DS4_RUN_BAKEOFF") != "1" or not all(required.values()):
        pytest.skip(
            "set DS4_RUN_BAKEOFF=1, DS4_TEST_MODEL, DS4_AGENT, "
            "and POLYGLOT_BENCHMARK for the complete live bake-off"
        )

    root = Path(__file__).resolve().parents[2]
    model = Path(cast(str, required["DS4_TEST_MODEL"])).expanduser().resolve()
    cli = Path(cast(str, required["DS4_AGENT"])).expanduser().resolve()
    manifest = load_manifest(root / "bench/bakeoff.yaml")
    common: dict[str, Any] = dict(
        model=str(model), backend="cpu", ctx=manifest.ctx,
        temperature=manifest.temperature, seed=manifest.seed, tokens=512,
    )
    configs = (
        AgentConfig("c", (str(cli),), **common),
        AgentConfig(
            "ds4py", (sys.executable, str(root / "bench/restore_check.py"), "attempt"),
            **common,
        ),
    )
    report = run_bakeoff(
        root / "bench/bakeoff.yaml",
        Path(cast(str, required["POLYGLOT_BENCHMARK"])).expanduser().resolve(),
        root,
        configs,
        root / "bench/results",
    )

    written = json.loads((root / "bench/results/bakeoff.json").read_text())
    gates = written["gates"]
    assert set(gates) == {f"gate-{index}" for index in range(1, 7)}
    failed = {
        name: gate["detail"] for name, gate in gates.items() if not gate["passed"]
    }
    assert not failed, f"live bakeoff gates failed: {failed}"
    assert len(written["replay"]) > 0
    assert report.passed
