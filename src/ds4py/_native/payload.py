"""Bridge Python-owned files to native payload calls through a duplicated stream.

The engine's payload API takes a libc ``FILE *``. Python keeps ownership of the
caller's file object: only the duplicated descriptor is passed to ``fdopen`` and
only that stream is closed.
"""

import ctypes
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import BinaryIO

_libc = ctypes.CDLL(None, use_errno=True)
_fdopen = _libc.fdopen
_fdopen.argtypes = [ctypes.c_int, ctypes.c_char_p]
_fdopen.restype = ctypes.c_void_p
_fclose = _libc.fclose
_fclose.argtypes = [ctypes.c_void_p]
_fclose.restype = ctypes.c_int
_ftell = _libc.ftell
_ftell.argtypes = [ctypes.c_void_p]
_ftell.restype = ctypes.c_long


def _os_error() -> OSError:
    error = ctypes.get_errno()
    return OSError(error, os.strerror(error))


@contextmanager
def native_stream(file: BinaryIO, mode: str) -> Iterator[int]:
    """Yield a libc stream for ``file``; close only the duplicated descriptor.

    The Python object is flushed and its logical position is made authoritative
    before duplication, so a buffered read-ahead cannot leave the shared file
    description ahead of the caller. Afterwards the caller's logical position is
    moved to the native stream's position and its buffer is discarded.
    """
    if mode not in ("rb", "wb"):
        raise ValueError("native stream mode must be 'rb' or 'wb'")
    file.flush()
    # A buffered reader may hold data read ahead of its logical position, and
    # seeking to that same position can leave the shared description at EOF,
    # so move the descriptor itself.
    os.lseek(file.fileno(), file.tell(), os.SEEK_SET)
    descriptor = os.dup(file.fileno())
    stream = _fdopen(descriptor, mode.encode("ascii"))
    if not stream:
        error = _os_error()
        os.close(descriptor)
        raise error

    failure = None
    try:
        yield stream
    except BaseException as error:
        failure = error
        raise
    finally:
        position = _ftell(stream)
        status = _fclose(stream)
        error = _os_error()
        if position >= 0:
            try:
                file.seek(0, os.SEEK_END)   # discard read-ahead before adopting it
                file.seek(position)
            except BaseException:
                if failure is None:
                    raise
        if status and failure is None:
            raise error
