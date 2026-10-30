import pytest

from ds4py._native._results import bool_result, int_result, optional_text_result
from ds4py.errors import EngineError


@pytest.mark.parametrize("value", [-1, 0, 1, 42])
def test_int_result_accepts_native_integers(value: int) -> None:
    assert int_result(value) == value


@pytest.mark.parametrize("value", [None, "1", 1.0, object()])
def test_int_result_rejects_non_integers(value: object) -> None:
    with pytest.raises(EngineError):
        int_result(value)


@pytest.mark.parametrize("value", [True, False])
def test_bool_result_accepts_native_booleans(value: bool) -> None:
    assert bool_result(value) is value


@pytest.mark.parametrize("value", [None, 0, 1, "true", object()])
def test_bool_result_rejects_non_booleans(value: object) -> None:
    with pytest.raises(EngineError):
        bool_result(value)


def test_optional_text_result_decodes_borrowed_bytes() -> None:
    assert optional_text_result(b"reasoning") == "reasoning"
    assert optional_text_result(None) is None


@pytest.mark.parametrize("value", [1, "text", object()])
def test_optional_text_result_rejects_other_values(value: object) -> None:
    with pytest.raises(EngineError):
        optional_text_result(value)
