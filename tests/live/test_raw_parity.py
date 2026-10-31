import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ds4py.abi.manifest import Manifest, manifest_path
from ds4py.abi.shaders import metal_source_env
from ds4py.raw import read_trace_tokens

PROMPT = "The three primary colors are"


@pytest.mark.live
def test_raw_matches_ds4_agent(live_lib: Any, tmp_path: Path) -> None:
    agent = os.environ.get("DS4_AGENT") or pytest.skip("DS4_AGENT is not set")
    manifest = Manifest.read(manifest_path(live_lib))
    backend = manifest.backend
    c_env = os.environ.copy()
    if backend == "metal":
        c_env.update(metal_source_env(live_lib.resolve().parent, manifest.metal_sources, c_env))
    common = ["-m", os.environ["DS4_TEST_MODEL"], "-c", "4096", "-n", "64", "--temp", "0", "--seed", "1"]
    c = subprocess.run(
        [
            agent,
            "--non-interactive",
            "--raw",
            "-p",
            PROMPT,
            *common,
            f"--{backend}",
            "--trace",
            str(tmp_path / "c.log"),
        ],
        capture_output=True,
        env=c_env,
        check=False,
    )
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "ds4py",
            "raw",
            PROMPT,
            *common,
            "--backend",
            backend,
            "--lib",
            str(live_lib),
            "--trace",
            str(tmp_path / "py.log"),
        ],
        capture_output=True,
        check=False,
    )
    assert c.returncode == 0 and p.returncode == 0, (c.stderr, p.stderr)
    assert read_trace_tokens(tmp_path / "py.log") == read_trace_tokens(tmp_path / "c.log")
    assert p.stdout == c.stdout
