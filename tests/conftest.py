import importlib
import os
from pathlib import Path
import shutil
from importlib.metadata import MetadataPathFinder

import pytest


@pytest.fixture
def cc():
    compiler = os.environ.get("CC") or shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler not available")
    return compiler


@pytest.fixture
def ext_site(monkeypatch):
    site = Path(__file__).parent / "fixtures" / "ext_site"
    monkeypatch.syspath_prepend(str(site.resolve()))
    importlib.invalidate_caches()
    MetadataPathFinder().invalidate_caches()
    yield site
    MetadataPathFinder().invalidate_caches()
