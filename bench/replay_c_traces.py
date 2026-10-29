"""Gate 6: replay complete C token traces with no tool execution."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from bench.run_bakeoff import AgentConfig, TaskRunResult

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ds4py
from ds4py.engine.decode import GenerateJob
from ds4py.errors import Ds4pyError
from ds4py.engine.events import Event, Stopped, Token
from ds4py.engine.replay import read_replay
from ds4py.engine.replay_types import ReplayRound, ReplayTrace
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.session import Engine
from ds4py.engine.stream import RoundStream
from ds4py.engine.tool_syntax import make_parser
from ds4py.engine.tool_syntax.model import Snapshot
from ds4py.engine.transcript import Transcript
from ds4py.engine.worker import SyncJob


class ReplayMismatch(AssertionError):
    """A generated token or terminal boundary differs from the C trace."""


class ReplaySelectionError(ValueError):
    """The C trace set is incomplete or inconsistent with the selected model."""


@dataclass(frozen=True)
class ReplayResult:
    task_id: str
    round_index: int
    passed: bool
    detail: str
    tokens: int = 0
    tool_call: bool = False
    terminal_reason: str | None = None
    trace_path: Path | None = None


def select_c_traces(
    task_results: Iterable[TaskRunResult], *, expected_tasks: int = 12,
) -> list[tuple[str, Path]]:
    """Select every C attempt in manifest order; ds4-py self-replay is forbidden."""
    tasks = [item for item in task_results if item.agent == "c"]
    identifiers = [item.task_id for item in tasks]
    if len(tasks) != expected_tasks or len(set(identifiers)) != expected_tasks:
        raise ReplaySelectionError(
            f"gate 6 needs {expected_tasks} distinct C tasks, found {len(set(identifiers))}"
        )
    selected = []
    for item in tasks:
        paths = item.attempt_trace_paths
        if len(paths) != item.attempts or any(path is None for path in paths):
            raise ReplaySelectionError(
                f"{item.task_id}: {len(paths)}/{item.attempts} C attempt traces are present"
            )
        selected.extend(
            (item.task_id, Path(path)) for path in paths if path is not None
        )
    return selected


def _parse_c(
    engine: Engine, round_: ReplayRound, chunks: Iterable[bytes],
) -> Snapshot:
    parser = make_parser(engine.info)
    mode = int(round_.think)
    parser.in_think = mode in (1, 2, 3, 4) or 1001 <= mode <= 1100
    streamed = False
    for chunk in chunks:
        streamed = True
        parser.feed(chunk)
    if not streamed:
        parser.feed(b"")
    return parser.finish()


def _token_description(
    engine: Engine, tokens: Sequence[int], index: int,
) -> str:
    if index >= len(tokens):
        return "id=<missing> bytes=<missing>"
    return f"id={tokens[index]} bytes={engine.token_text(tokens[index])!r}"


def _assert_tokens(
    engine: Engine, task_id: str, ordinal: int, round_: ReplayRound,
    actual: Sequence[int], source: str,
) -> None:
    expected = round_.sampled
    if actual == expected:
        return
    index = next(
        (i for i, (left, right) in enumerate(zip(expected, actual)) if left != right),
        min(len(expected), len(actual)),
    )
    raise ReplayMismatch(
        f"{task_id} trace round {ordinal} (tool_round={round_.number}), "
        f"token {index}; prompt={len(round_.prompt)} "
        f"C sampled={len(expected)} ds4py={len(actual)}\n"
        f"C:      {_token_description(engine, expected, index)}\n"
        f"ds4py:  {_token_description(engine, actual, index)}"
    )


def _check_round(
    engine: Engine, task_id: str, ordinal: int, round_: ReplayRound,
    events: Sequence[Event], transcript: Transcript,
) -> tuple[bool, str]:
    if not events or not isinstance(events[-1], Stopped):
        raise ReplayMismatch(f"{task_id} trace round {ordinal} did not drain to Stopped")
    emitted = tuple(event.id for event in events if isinstance(event, Token))
    additions = transcript.tokens[len(round_.prompt):]
    _assert_tokens(engine, task_id, ordinal, round_, emitted, "emitted token")
    _assert_tokens(engine, task_id, ordinal, round_, additions, "transcript addition")

    data = b"".join(engine.token_text(token) for token in round_.sampled)
    c_snapshot = _parse_c(engine, round_, [data])
    stopped = events[-1]
    if not round_.prefix_only and not round_.context_limited:
        if stopped.reason not in ("stop_token", "tool_call_complete", "parser_error"):
            raise ReplayMismatch(
                f"{task_id} trace round {ordinal} did not stop naturally at C's "
                f"boundary: {stopped.reason}"
            )
        if stopped.reason == "tool_call_complete":
            if c_snapshot.state != "DONE" or c_snapshot.dsml_in_think:
                raise ReplayMismatch(
                    f"{task_id} trace round {ordinal}: tool_call_complete but "
                    f"C text parsed as {c_snapshot.state}: {c_snapshot.error}"
                )
        elif stopped.reason == "parser_error":
            if c_snapshot.state != "ERROR" and not c_snapshot.dsml_in_think:
                raise ReplayMismatch(
                    f"{task_id} trace round {ordinal}: parser_error but C text "
                    f"parsed as {c_snapshot.state}"
                )
    tool_call = c_snapshot.state == "DONE" and bool(c_snapshot.calls)
    if tool_call:
        bytewise = _parse_c(
            engine, round_, (data[index:index + 1] for index in range(len(data)))
        )
        if bytewise != c_snapshot:
            raise ReplayMismatch(f"{task_id} trace round {ordinal}: bytewise parser mismatch")
        for split in range(len(data) + 1):
            observed = _parse_c(engine, round_, [data[:split], data[split:]])
            if observed != c_snapshot:
                raise ReplayMismatch(
                    f"{task_id} trace round {ordinal}: tool stanza differs at byte split {split}"
                )
    # Parsing a call is the terminal boundary; replay never executes it.
    return tool_call, stopped.reason


def _reserve(engine: Engine, round_: ReplayRound) -> int:
    if not round_.context_limited:
        return 1
    reserve = engine.config.ctx - len(round_.prompt) - len(round_.sampled)
    if reserve < 0:
        raise ReplayMismatch(f"negative derived context reserve {reserve}")
    return reserve


def generation_limit(engine: Engine, round_: ReplayRound) -> int:
    """Natural rounds need one EOS/token slot; bounded rounds use the exact prefix."""
    return len(round_.sampled) if round_.prefix_only or round_.context_limited else len(round_.sampled) + 1


def replay_round(
    engine: Engine, policy: SamplingPolicy, rng: Rng, task_id: str,
    ordinal: int, round_: ReplayRound,
) -> ReplayResult:
    """Replay one C-owned prompt without preflight, translation, or tool calls."""
    transcript = Transcript()
    if round_.prompt:
        transcript.append(round_.prompt, "system")
    list(engine._worker.submit(SyncJob(round_.prompt)))
    stream = RoundStream(engine.info, round_.think)
    events = list(engine._worker.submit(GenerateJob(
        policy, rng, generation_limit(engine, round_), transcript,
        reserve=_reserve(engine, round_), think_mode=round_.think, stream=stream,
        control=stream.control,
    )))
    tool_call, terminal = _check_round(
        engine, task_id, ordinal, round_, events, transcript,
    )
    return ReplayResult(
        task_id, ordinal, True,
        f"{task_id} round {ordinal}: {len(round_.sampled)} tokens; {terminal}",
        tokens=len(round_.sampled), tool_call=tool_call, terminal_reason=terminal,
    )


def replay_trace(
    engine: Engine, trace: ReplayTrace, *, task_id: str = "trace",
) -> tuple[ReplayResult, ...]:
    """Reuse one C-capture RNG across rounds, resetting it for each C attempt."""
    policy = SamplingPolicy.for_engine(engine.info, temperature=0.0, seed=1)
    rng = Rng(1)
    return tuple(
        replay_round(
            engine, policy, rng, task_id, ordinal, round_,
        )
        for ordinal, round_ in enumerate(trace.rounds)
    )


def _validate_selection(selected: Sequence[tuple[str, Path]]) -> None:
    identifiers = {task_id for task_id, _path in selected}
    if len(selected) == 0 or len(identifiers) != 12:
        raise ReplaySelectionError(
            f"gate 6 needs 12 C tasks, found {len(identifiers)}"
        )


def replay_with_engine(
    engine: Engine, selected: Sequence[tuple[str, Path]], *,
    require_full_set: bool = True,
) -> tuple[ReplayResult, ...]:
    """Replay selected C files sequentially on one already-open engine."""
    if require_full_set:
        _validate_selection(selected)
    model = Path(engine.config.model).expanduser().resolve()
    results: list[ReplayResult] = []
    for task_id, path in selected:
        path = Path(path)
        try:
            trace = read_replay(path)
            if trace.model is None or trace.ctx is None:
                raise ReplaySelectionError(f"trace {path} is missing model or context metadata")
            if Path(trace.model).expanduser().resolve() != model:
                raise ReplaySelectionError(
                    f"trace model {trace.model} does not match selected {model}"
                )
            if trace.ctx != engine.config.ctx:
                raise ReplaySelectionError(
                    f"trace ctx {trace.ctx} does not match engine ctx {engine.config.ctx}"
                )
        except (OSError, ValueError) as error:
            results.append(ReplayResult(
                task_id, -1, False,
                f"{task_id} round -1: malformed C trace {path}: {error}",
                trace_path=path,
            ))
            continue
        policy = SamplingPolicy.for_engine(engine.info, temperature=0.0, seed=1)
        rng = Rng(1)
        for ordinal, round_ in enumerate(trace.rounds):
            try:
                result = replay_round(
                    engine, policy, rng, task_id, ordinal, round_,
                )
            except (Ds4pyError, ReplayMismatch) as error:
                result = ReplayResult(
                    task_id, ordinal, False, str(error), trace_path=path,
                )
            results.append(ReplayResult(**{**result.__dict__, "trace_path": path}))
    if require_full_set and not any(result.tool_call for result in results):
        results.append(ReplayResult(
            "<selection>", -1, False,
            "full 12-run C set contains no tool-call round to replay",
        ))
    return tuple(results)


def replay_c_traces(
    task_results: Iterable[TaskRunResult], model: str | Path, *,
    lib: str | Path | None = None, backend: str = "cpu",
) -> tuple[ReplayResult, ...]:
    """Load the selected GGUF once and close it after the sequential C phase."""
    selected = select_c_traces(task_results)
    first = read_replay(selected[0][1])
    if first.model is None or first.ctx is None:
        raise ReplaySelectionError("C trace is missing model or context metadata")
    if Path(first.model).expanduser().resolve() != Path(model).expanduser().resolve():
        raise ReplaySelectionError(
            f"trace model {first.model} does not match selected {model}"
        )
    with ds4py.open(model, ctx=first.ctx, backend=backend, lib=lib, seed=1) as engine:
        return replay_with_engine(engine, selected)


def replay_task_results(
    task_results: Iterable[TaskRunResult], config: AgentConfig, *,
    lib: str | Path | None = None,
) -> tuple[ReplayResult, ...]:
    """Convert selection/loading failures into gate evidence instead of skipping."""
    try:
        return replay_c_traces(
            task_results, config.model, lib=lib, backend=config.backend,
        )
    except (Ds4pyError, OSError, ValueError) as error:
        return (ReplayResult("<selection>", -1, False, str(error)),)


def main(argv: Sequence[str] | None = None) -> int:
    """Replay the C traces named by an existing report and update gate 6."""
    from bench.report import update_replay_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=os.environ.get("DS4_TEST_MODEL"))
    parser.add_argument("--backend", default=os.environ.get("DS4_BAKEOFF_BACKEND", "cpu"))
    parser.add_argument("--lib")
    args = parser.parse_args(argv)
    if args.model is None:
        parser.error("--model or DS4_TEST_MODEL is required")
    report_path = args.report / "bakeoff.json" if args.report.is_dir() else args.report
    import json
    data = json.loads(report_path.read_text())
    tasks = cast(
        list[TaskRunResult],
        [SimpleNamespace(**item) for item in data.get("tasks", [])],
    )
    config = cast(AgentConfig, SimpleNamespace(model=args.model, backend=args.backend))
    results = replay_task_results(tasks, config, lib=args.lib)
    gate = update_replay_report(report_path, results)
    for result in results:
        print(
            f"{'PASS' if result.passed else 'FAIL'} {result.task_id} "
            f"round {result.round_index}: {result.detail}"
        )
    print(f"gate-6: {'PASS' if gate.passed else 'FAIL'} — {gate.detail}")
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
