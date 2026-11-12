"""Optional native file-payload capability policy and worker confinement."""

import threading
from pathlib import Path
from typing import Any

import pytest

import ds4py
from ds4py._native import api
from ds4py._native.fake import FakeNative


class ProbeLibrary:
    def __init__(self, result: bool | None) -> None:
        self.result = result

    def has(self, name: str) -> bool:
        assert name == "ds4_session_supports_payload"
        return self.result is not None

    def fn(self, name: str) -> Any:
        assert name == "ds4_session_supports_payload"

        def probe(session: int) -> bool:
            assert session == 42
            assert self.result is not None
            return self.result

        return probe


@pytest.mark.parametrize(
    ("native_result", "expected"),
    [(None, True), (False, False), (True, True)],
    ids=["absent", "native_false", "native_true"],
)
def test_file_payload_policy_has_only_legacy_true_or_the_native_boolean(
    native_result: bool | None,
    expected: bool,
) -> None:
    assert api.file_payloads_supported(ProbeLibrary(native_result), 42) is expected


@pytest.mark.parametrize(
    ("capability", "expected"),
    [(None, True), (False, False), (True, True)],
    ids=["absent", "native_false", "native_true"],
)
def test_fake_file_payload_probe_runs_on_the_engine_worker(
    capability: bool | None,
    expected: bool,
) -> None:
    native = FakeNative(payload_support=capability)
    with ds4py.open(Path("fake.gguf"), native=native) as engine:
        session = engine.session()
        assert session.file_payloads_supported() is expected

    assert len(native.payload_support_threads) == 1
    assert native.payload_support_threads[0] != threading.get_ident()
