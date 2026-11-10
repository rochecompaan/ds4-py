"""Worker-confined, fail-closed restoration of the actual native token prefix."""

from typing import TYPE_CHECKING

from ds4py._native.protocol import SessionHandle, SyncStatus
from ds4py.errors import EngineError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def restore_native_tokens(
    native: SessionHandle,
    tokens: Sequence[int],
    *,
    valid: bool,
    poison: Callable[[BaseException], None],
) -> None:
    """Rebuild without cancellation or events; poison if restoration is uncertain."""
    try:
        if not valid:
            native.invalidate()
        if not tokens:
            native.rewind(0)
        else:
            _sync_tokens(native, tokens)
        if not valid:
            native.invalidate()
    except BaseException as error:
        poison(error)
        raise


def _sync_tokens(native: SessionHandle, tokens: Sequence[int]) -> None:
    full = len(tokens) == native.ctx()
    status = native.sync(tokens[:-1] if full else tokens)
    if status is not SyncStatus.DONE:
        raise EngineError("native token restoration did not complete")
    if full:
        native.eval(tokens[-1])
