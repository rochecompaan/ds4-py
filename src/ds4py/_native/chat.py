"""ctypes adapters for the pinned native chat-rendering calls."""
import ctypes as C

from ds4py.errors import EngineError
from . import structs


def append_delta(handle, prefix, function: str, *args) -> list[int]:
    """Call a native append function with C-owned storage and return its delta."""
    original = tuple(prefix)
    tokens = structs.Tokens()
    try:
        # Native append may realloc the vector. Python-owned arrays cannot be
        # passed here, so let ds4 allocate even the initial prefix.
        push = handle._lib.fn("ds4_tokens_push")
        for token in original:
            push(C.byref(tokens), token)
        handle._call(function, C.byref(tokens), *args)
        size = len(original)
        if tokens.len < size or (
            original and tuple(tokens.v[:size]) != original
        ):
            raise EngineError(f"{function} changed the supplied chat prefix")
        if tokens.len == size:
            return []
        return list(tokens.v[size:tokens.len])
    finally:
        handle._lib.fn("ds4_tokens_free")(C.byref(tokens))


def qwen_reasoning_effort(handle, mode) -> str | None:
    """Borrow ds4's static effort string; a null result is a borrowed success."""
    handle._assert_thread()
    if not handle.raw:
        raise EngineError("native handle is closed")
    # Unlike the chat append calls, this C function has no engine parameter.
    text = handle._lib.fn("ds4_qwen4_reasoning_effort_text")(mode)
    return text.decode() if text is not None else None
