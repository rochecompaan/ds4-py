"""Allocate the native sampler's mutable uint64 state."""

import ctypes


def rng_state(seed: int) -> ctypes.c_uint64:
    return ctypes.c_uint64(seed)
