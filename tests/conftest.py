import os
import shutil

import pytest


@pytest.fixture
def cc():
    compiler = os.environ.get("CC") or shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler not available")
    return compiler
