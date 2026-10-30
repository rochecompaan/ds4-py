import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ds4py.abi.manifest import Manifest, manifest_path


@pytest.mark.live
def test_check_live_records_its_results(live_lib: Any, tmp_path: Path) -> None:
    lib = tmp_path / live_lib.name
    shutil.copy(live_lib, lib)
    shutil.copy(manifest_path(live_lib), manifest_path(lib))
    r = subprocess.run(
        [sys.executable, "-m", "ds4py", "check", "--live", "--lib", str(lib), "--model", os.environ["DS4_TEST_MODEL"]],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sync:     ok" in r.stdout and "restore:  not verified" not in r.stdout
    checks = Manifest.read(manifest_path(lib)).live_checks
    assert bool(checks["sync_plans"]["ok"]) and isinstance(checks["restore"]["ok"], bool)
