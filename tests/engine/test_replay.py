import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from ds4py import ThinkMode

FIXTURES = Path(__file__).parents[1] / "fixtures" / "replay"
BASIC = FIXTURES / "basic.trace"
COMPACTED = FIXTURES / "compacted.trace"
EARLY_EDIT = FIXTURES / "early_edit.trace"
STAMP = "2026-10-21 09:05:07.042 "
FIRST_PREFILL = "prefill tool_round=1 transcript=4 prompt=4 cached=2 suffix=2 think=high"


def read(path: Any) -> Any:
    try:
        module = importlib.import_module("ds4py.engine.replay")
    except ModuleNotFoundError as error:
        if error.name != "ds4py.engine.replay":
            raise
        raise AssertionError("replay reader is missing") from error
    return module.read_replay(path)


def assert_replay_error(path: Any, fragment: Any) -> None:
    with pytest.raises(ValueError) as caught:
        read(path)
    assert type(caught.value).__name__ == "ReplayError"
    message = str(caught.value)
    assert fragment in message
    assert re.search(re.escape(str(path)) + r":\d+:", message)


def mutate(tmp_path: Path, source: Any, old: Any, new: Any) -> Any:
    data = source.read_text()
    assert old in data
    path = tmp_path / f"{source.stem}-invalid.trace"
    path.write_text(data.replace(old, new, 1))
    return path


def test_basic_trace_rebuilds_rounds_eos_tool_results_and_rewinds() -> None:
    trace = read(BASIC)

    assert trace.ctx == 4096
    assert trace.model == "/models/fake.gguf"
    assert [round.number for round in trace.rounds] == [1, 2, 3, 4]
    assert [tuple(round.prompt) for round in trace.rounds] == [
        (10, 11, 12, 13),
        (10, 11, 12, 13, 20, 21, 99, 30, 31),
        (10, 11, 12, 13, 20, 21, 99, 30, 50, 51, 52),
        (10, 11, 12, 13, 20, 21, 99, 30, 50, 51, 52),
    ]
    assert [tuple(round.sampled) for round in trace.rounds] == [
        (20, 21),
        (40,),
        (),
        (),
    ]
    assert [round.cached for round in trace.rounds] == [2, 6, 8, 11]
    assert [round.think for round in trace.rounds] == [
        ThinkMode.HIGH,
        ThinkMode.NONE,
        1075,
        ThinkMode.MEDIUM,
    ]
    assert [round.context_limited for round in trace.rounds] == [False] * 4
    assert [round.prefix_only for round in trace.rounds] == [False] * 4


def test_compaction_replaces_the_base_and_records_context_limit() -> None:
    trace = read(COMPACTED)

    assert trace.ctx == 8192
    assert trace.model == "/models/compacted.gguf"
    assert len(trace.rounds) == 2
    first, second = trace.rounds
    assert tuple(first.prompt) == (5, 6, 7, 8, 9)
    assert tuple(first.sampled) == (15, 16)
    assert first.context_limited is True
    assert first.prefix_only is False
    assert tuple(second.prompt) == (100, 101, 15, 16, 102)
    assert tuple(second.sampled) == (17,)
    assert second.cached == 4
    assert second.think == ThinkMode.LOW
    assert second.context_limited is False


def test_early_edit_round_is_prefix_only() -> None:
    trace = read(EARLY_EDIT)

    assert len(trace.rounds) == 1
    round = trace.rounds[0]
    assert tuple(round.prompt) == (1, 2, 3)
    assert tuple(round.sampled) == (60, 61, 62)
    assert round.think == ThinkMode.MAX
    assert round.prefix_only is True
    assert round.context_limited is False


def test_ignores_prefill_sync_status(tmp_path: Path) -> None:
    path = mutate(
        tmp_path,
        BASIC,
        FIRST_PREFILL,
        "prefill sync done prompt=4 cached=0 suffix=4 rc=0 1.0 ms\n" + STAMP + FIRST_PREFILL,
    )

    trace = read(path)

    assert len(trace.rounds) == 4


def test_accepts_c_round_numbers_across_user_turns(tmp_path: Path) -> None:
    data = BASIC.read_text()
    for old, new in ((1, 0), (2, 1), (3, 2), (4, 3)):
        data = data.replace(f"tool_round={old}", f"tool_round={new}")
    data += (
        STAMP
        + "prefill tool_round=0 transcript=12 prompt=12 cached=11 suffix=1 think=none\n"
        + STAMP
        + "tokens label=prefill_suffix start=11 len=12\n"
        + STAMP
        + 'token index=11 id=70 bytes=1 text="F" hex=46\n'
        + STAMP
        + 'token index=1 id=71 bytes=1 text="G" hex=47\n'
        + STAMP
        + "generation finished tool_round=0 generated=1 carried=0 context_limited=0\n"
    )
    path = tmp_path / "zero-based.trace"
    path.write_text(data)

    trace = read(path)

    assert [round.number for round in trace.rounds] == [0, 1, 2, 3, 0]
    assert trace.rounds[-1].prompt == (10, 11, 12, 13, 20, 21, 99, 30, 50, 51, 52, 70)
    assert trace.rounds[-1].sampled == (71,)


def test_missing_initial_prompt_is_rejected(tmp_path: Path) -> None:
    lines = BASIC.read_text().splitlines(keepends=True)
    initial = next(i for i, line in enumerate(lines) if "initial_system_prompt" in line)
    prefill = next(i for i, line in enumerate(lines) if "prefill tool_round=" in line)
    path = tmp_path / "missing-initial.trace"
    path.write_text("".join(lines[:initial] + lines[prefill:]))
    assert_replay_error(path, "missing initial system prompt")


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        (
            FIRST_PREFILL,
            "tokens label=initial_system_prompt start=0 len=2\n" + STAMP + FIRST_PREFILL,
            "duplicate initial system prompt",
        ),
        (FIRST_PREFILL, "", "prefill suffix without prefill"),
        (
            FIRST_PREFILL,
            'token index=1 id=90 bytes=1 text="x" hex=78\n' + STAMP + FIRST_PREFILL,
            "sampled token outside a round",
        ),
        (
            "tool_round=1 transcript=4 prompt=4 cached=2 suffix=2",
            "tool_round=1 transcript=4 prompt=4 cached=99 suffix=2",
            "rebuilt history",
        ),
        (
            "tool_round=1 transcript=4 prompt=4 cached=2 suffix=2",
            "tool_round=1 transcript=4 prompt=4 cached=-1 suffix=2",
            "negative cached",
        ),
        (
            "tool_round=2 transcript=9 prompt=9 cached=6 suffix=3",
            "tool_round=2 transcript=5 prompt=5 cached=6 suffix=3",
            "beyond traced transcript",
        ),
        (
            "tool_round=1 transcript=4 prompt=4 cached=2 suffix=2",
            "tool_round=1 transcript=4 prompt=4 cached=2 suffix=3",
            "suffix length",
        ),
        (
            "tool_round=1 transcript=4 prompt=4 cached=2 suffix=2",
            "tool_round=1 transcript=5 prompt=4 cached=2 suffix=2",
            "transcript/prompt length",
        ),
        (
            "tokens label=initial_system_prompt start=0 len=2",
            "tokens label=initial_system_prompt start=0 len=3",
            "wrong token count",
        ),
        ("token index=1 id=11", "token index=2 id=11", "non-contiguous token index"),
        ("token index=2 id=21", "token index=3 id=21", "non-contiguous token index"),
        (
            "generation finished tool_round=1 generated=2",
            "generation finished tool_round=1 generated=3",
            "generated count",
        ),
        (FIRST_PREFILL, FIRST_PREFILL + "\n" + STAMP + FIRST_PREFILL, "second prefill before generation finished"),
        (FIRST_PREFILL, "tokens label=unexpected start=0 len=0\n" + STAMP + FIRST_PREFILL, "unexpected token block"),
        ("generation finished tool_round=4 generated=0 carried=0 context_limited=0", "", "truncated final round"),
        ("tokens label=prefill_suffix start=2 len=4", "tokens label=prefill_suffix start=3 len=4", "suffix start"),
        (
            "tokens label=prefill_suffix start=2 len=4",
            "tokens label=prefill_suffix start=2 len=5",
            "suffix block length",
        ),
        ("think=high", "think=ultra", "unsupported think name"),
        ("think=75", "think=0", "unsupported think name"),
    ],
)
def test_malformed_basic_traces_are_rejected(tmp_path: Path, old: Any, new: Any, fragment: Any) -> None:
    assert_replay_error(mutate(tmp_path, BASIC, old, new), fragment)


def test_missing_context_is_rejected(tmp_path: Path) -> None:
    path = mutate(
        tmp_path,
        BASIC,
        "agent worker start ctx=4096 backend=cpu model=/models/fake.gguf trace=c.trace",
        "agent worker status unknown=true",
    )
    assert_replay_error(path, "missing context")


def test_invalid_context_is_rejected(tmp_path: Path) -> None:
    path = mutate(tmp_path, BASIC, "agent worker start ctx=4096", "agent worker start ctx=0")
    assert_replay_error(path, "invalid context")


def test_duplicate_compaction_is_rejected(tmp_path: Path) -> None:
    path = mutate(
        tmp_path,
        COMPACTED,
        "prefill tool_round=2 transcript=5 prompt=5 cached=4 suffix=1 think=low",
        "tokens label=compacted_transcript start=0 len=4\n"
        + STAMP
        + "prefill tool_round=2 transcript=5 prompt=5 cached=4 suffix=1 think=low",
    )
    assert_replay_error(path, "unexpected compacted transcript")


@pytest.mark.parametrize(
    "raw_line",
    [
        "raw one-shot prompt mode: skipping agent system prompt",
        "tokens label=raw_prompt_tokens start=0 len=0",
    ],
)
def test_raw_traces_are_not_replayable(tmp_path: Path, raw_line: Any) -> None:
    path = tmp_path / "raw.trace"
    path.write_text(
        STAMP
        + "agent worker start ctx=4096 backend=cpu model=/models/raw.gguf trace=raw.trace\n"
        + STAMP
        + raw_line
        + "\n"
    )
    assert_replay_error(path, "raw traces are not replayable")


@pytest.mark.parametrize(
    "header",
    [
        "agent worker start ctx=4096 backend=cpu model=/models/fake.gguf trace=c.trace",
        "agent worker start ctx=1 backend=cuda model=/models/other.gguf trace=other.trace",
    ],
)
def test_duplicate_worker_start_cannot_overwrite_replay_metadata(tmp_path: Path, header: Any) -> None:
    path = tmp_path / "duplicate.trace"
    path.write_text(BASIC.read_text() + STAMP + header + "\n")
    assert_replay_error(path, "duplicate worker start")


def test_initial_state_without_any_round_cannot_claim_replay_parity(tmp_path: Path) -> None:
    path = tmp_path / "empty.trace"
    path.write_text("\n".join(BASIC.read_text().splitlines()[:5]) + "\n")
    assert_replay_error(path, "no replay rounds")
