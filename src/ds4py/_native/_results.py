"""Validate scalar values returned through dynamic ctypes boundaries."""

from ds4py.errors import EngineError


def int_result(value: object) -> int:
    """Return a native integer result, rejecting unexpected dynamic values."""
    if not isinstance(value, int):
        raise EngineError(f"native call returned a non-integer value: {value!r}")
    return value


def bool_result(value: object) -> bool:
    """Return a native boolean result, rejecting unexpected dynamic values."""
    if not isinstance(value, bool):
        raise EngineError(f"native call returned a non-boolean value: {value!r}")
    return value


def optional_text_result(value: object) -> str | None:
    """Decode borrowed native text, preserving a null result as ``None``."""
    if value is None:
        return None
    if not isinstance(value, bytes):
        raise EngineError(f"native call returned non-text value: {value!r}")
    return value.decode()
