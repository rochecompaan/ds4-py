"""ctypes marshalling for the live session logits vector."""

import ctypes as C
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ds4py.errors import EngineError

if TYPE_CHECKING:
    from .api import NativeSession


def _buffer(session: NativeSession, values: Any) -> tuple[Any, int]:
    size = session._engine.info().vocab_size
    if len(values) != size:
        raise ValueError(f"logits buffer must equal the vocabulary size ({size})")
    return (C.c_float * size), size


def _native_float32_view(values: Any) -> memoryview | None:
    try:
        view = memoryview(values)
    except TypeError:
        return None
    if view.format == "f" and view.itemsize == C.sizeof(C.c_float) and view.c_contiguous:
        return view
    return None


def copy_logits(session: NativeSession, out: Any) -> None:
    """Copy native logits into a writable contiguous float32 buffer."""
    buffer_type, size = _buffer(session, out)
    if _native_float32_view(out) is None:
        raise TypeError("logits output must be a contiguous float32 buffer")
    try:
        out_buffer = buffer_type.from_buffer(out)
    except (BufferError, TypeError) as error:
        raise TypeError("logits output must be a writable contiguous float32 buffer") from error
    copied = session._call("ds4_session_copy_logits", out_buffer, size)
    if copied != size:
        raise EngineError(f"ds4_session_copy_logits failed: returned {copied}, expected {size}")


def set_logits(session: NativeSession, values: Sequence[float]) -> None:
    """Replace the native logits from a same-sized Python float sequence."""
    buffer_type, size = _buffer(session, values)
    if _native_float32_view(values) is not None:
        try:
            buffer = buffer_type.from_buffer(values)
        except BufferError, TypeError:
            buffer = buffer_type(*values)
    else:
        buffer = buffer_type(*values)
    if session._call("ds4_session_set_logits", buffer, size):
        raise EngineError("ds4_session_set_logits failed")
