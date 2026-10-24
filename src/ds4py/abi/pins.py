"""Source pins and informational header fingerprints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pin:
    repo: str
    commit: str


UPSTREAM = Pin(
    "https://github.com/antirez/ds4.git",
    "5379c1c9ac315f1540371b06a3db88491d4059ff",
)
FORK = Pin(
    "https://github.com/pauleveritt/ds4.git",
    "3e61e09bbe4599ca71197d8ec21933a93e24f022",
)
HEADER_FILES = ("ds4.h", "ds4_kvstore.h")
KNOWN_HEADERS: tuple[dict[str, str], ...] = (
    {
        "ds4.h": "fd34c9485866cbf9b55e6e9cefeb0c19db4023de567d9c2196ddc3b91d6bc0d4",
        "ds4_kvstore.h": "e6d0e2836385285718436d89292d1052099b7cc9a2c8a1c79d3d59e214d81c3b",
    },
    {
        "ds4.h": "b5140bc6bd398321aa17b2fd97a130dcd9cebac3f91eb0b04d83aa710df8110b",
        "ds4_kvstore.h": "e6d0e2836385285718436d89292d1052099b7cc9a2c8a1c79d3d59e214d81c3b",
    },
)
