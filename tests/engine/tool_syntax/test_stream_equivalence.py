from typing import Any
import random


def test_every_single_split_and_bytewise(parser_case: Any, parse_chunks: Any) -> None:
    data = parser_case["input"].encode()
    expected = parse_chunks(parser_case, [data])
    assert parse_chunks(parser_case, [bytes([byte]) for byte in data]) == expected
    for split in range(len(data) + 1):
        observed = parse_chunks(parser_case, [data[:split], data[split:]])
        assert observed == expected, f"split {split} of {len(data)}"


def test_100_seeded_random_partitions(parser_case: Any, parse_chunks: Any) -> None:
    data = parser_case["input"].encode()
    expected = parse_chunks(parser_case, [data])
    rng = random.Random(f"0:{parser_case['id']}")
    for sample in range(100):
        cuts = sorted(rng.randint(0, len(data)) for _ in range(24))
        bounds = [0, *cuts, len(data)]
        chunks = [data[start:end] for start, end in zip(bounds, bounds[1:])]
        observed = parse_chunks(parser_case, chunks)
        assert observed == expected, f"random partition {sample}: {cuts}"
