"""Family-aware sampling parameters and native-compatible random state."""
from dataclasses import dataclass
import os
import time

from ds4py._native.protocol import EngineInfo
from ds4py._native.rng import rng_state


@dataclass(frozen=True)
class SamplingPolicy:
    temperature: float = 1.0
    top_p: float = 1.0
    min_p: float = 0.05
    top_k: int = 0
    seed: int | None = None

    @classmethod
    def for_engine(
        cls, info: EngineInfo, *, temperature: float | None = None,
        top_p: float | None = None, min_p: float | None = None,
        top_k: int | None = None, seed: int | None = None,
    ) -> "SamplingPolicy":
        return cls(
            temperature=1.0 if temperature is None else temperature,
            top_p=(0.95 if info.is_glm_dsa else 1.0) if top_p is None else top_p,
            min_p=(0.0 if info.is_glm_dsa else 0.05) if min_p is None else min_p,
            top_k=0 if top_k is None else top_k,
            seed=seed,
        )


class Rng:
    def __init__(self, seed: int | None) -> None:
        if not seed:
            seed = int(time.time()) ^ (os.getpid() << 32) ^ time.process_time_ns()
            seed = (seed & ((1 << 64) - 1)) or 1
        self.state = rng_state(seed)
