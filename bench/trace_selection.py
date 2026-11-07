"""Select recorded traces and replay their rounds without tool execution."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ds4py.engine.decode import GenerateJob
from ds4py.engine.events import Event, Stopped, Token
from ds4py.engine.stream import RoundStream
from ds4py.engine.tool_syntax import make_parser
from ds4py.engine.transcript import Transcript
from ds4py.engine.worker import SyncJob

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from bench.models import TaskRunResult
    from ds4py.engine.replay_types import ReplayRound
    from ds4py.engine.sampling import Rng, SamplingPolicy
    from ds4py.engine.session import Engine
    from ds4py.engine.tool_syntax.model import Snapshot


class ReplayMismatch(AssertionError):
    """A generated token or terminal boundary differs from the recorded trace."""


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
    task_results: Iterable[TaskRunResult],
    *,
    expected_tasks: int | None = None,
) -> list[tuple[str, Path]]:
    """Select every C attempt in manifest order, without a certification rule."""
    tasks = [item for item in task_results if item.agent == "c"]
    identifiers = [item.task_id for item in tasks]
    if not tasks or len(set(identifiers)) != len(identifiers):
        raise ReplaySelectionError("C replay needs distinct task identifiers and at least one C task")
    if expected_tasks is not None and len(tasks) != expected_tasks:
        raise ReplaySelectionError(f"C replay needs {expected_tasks} tasks, found {len(tasks)}")
    selected = []
    for item in tasks:
        paths = item.attempt_trace_paths
        if len(paths) != item.attempts or any(path is None for path in paths):
            raise ReplaySelectionError(f"{item.task_id}: {len(paths)}/{item.attempts} C attempt traces are present")
        selected.extend((item.task_id, Path(path)) for path in paths)
    return selected


def _parse_c(
    engine: Engine,
    round_: ReplayRound,
    chunks: Iterable[bytes],
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
    engine: Engine,
    tokens: Sequence[int],
    index: int,
) -> str:
    if index >= len(tokens):
        return "id=<missing> bytes=<missing>"
    return f"id={tokens[index]} bytes={engine.token_text(tokens[index])!r}"


def _assert_tokens(  # noqa: PLR0913, PLR0917 - Replay comparison needs the complete round context.
    engine: Engine,
    task_id: str,
    ordinal: int,
    round_: ReplayRound,
    actual: Sequence[int],
    source: str,
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


def _check_round(  # noqa: PLR0913, PLR0917 - Replay comparison needs the complete round context.
    engine: Engine,
    task_id: str,
    ordinal: int,
    round_: ReplayRound,
    events: Sequence[Event],
    transcript: Transcript,
    response_budget: int | None = None,
) -> tuple[bool, str]:
    if not events or not isinstance(events[-1], Stopped):
        raise ReplayMismatch(f"{task_id} trace round {ordinal} did not drain to Stopped")
    emitted = tuple(event.id for event in events if isinstance(event, Token))
    additions = transcript.tokens[len(round_.prompt) :]
    _assert_tokens(engine, task_id, ordinal, round_, emitted, "emitted token")
    _assert_tokens(engine, task_id, ordinal, round_, additions, "transcript addition")

    data = b"".join(engine.token_text(token) for token in round_.sampled)
    c_snapshot = _parse_c(engine, round_, [data])
    stopped = events[-1]
    if not round_.prefix_only and not round_.context_limited:
        at_response_budget = response_budget is not None and len(round_.sampled) == response_budget
        terminal_reasons = ("stop_token", "tool_call_complete", "parser_error")
        if at_response_budget:
            terminal_reasons += ("max_tokens",)
        if stopped.reason not in terminal_reasons:
            raise ReplayMismatch(
                f"{task_id} trace round {ordinal} did not stop naturally at C's boundary: {stopped.reason}"
            )
        if stopped.reason == "tool_call_complete":
            if c_snapshot.state != "DONE" or c_snapshot.dsml_in_think:
                raise ReplayMismatch(
                    f"{task_id} trace round {ordinal}: tool_call_complete but "
                    f"C text parsed as {c_snapshot.state}: {c_snapshot.error}"
                )
        elif stopped.reason == "parser_error" and c_snapshot.state != "ERROR" and not c_snapshot.dsml_in_think:
            raise ReplayMismatch(
                f"{task_id} trace round {ordinal}: parser_error but C text parsed as {c_snapshot.state}"
            )
    tool_call = c_snapshot.state == "DONE" and bool(c_snapshot.calls)
    if tool_call:
        # ToolParser.feed consumes every chunk byte by byte, so bytewise equivalence
        # covers every two-chunk split without reparsing the full response per byte.
        bytewise = _parse_c(engine, round_, (data[index : index + 1] for index in range(len(data))))
        if bytewise != c_snapshot:
            raise ReplayMismatch(f"{task_id} trace round {ordinal}: bytewise parser mismatch")
    # Parsing a call is the terminal boundary; replay never executes it.
    return tool_call, stopped.reason


def _reserve(engine: Engine, round_: ReplayRound) -> int:
    if not round_.context_limited:
        return 1
    reserve = engine.config.ctx - len(round_.prompt) - len(round_.sampled)
    if reserve < 0:
        raise ReplayMismatch(f"negative derived context reserve {reserve}")
    return reserve


def generation_limit(engine: Engine, round_: ReplayRound, response_budget: int | None = None) -> int:
    """Natural rounds need one EOS/token slot, bounded by the recorded response budget."""
    limit = len(round_.sampled) if round_.prefix_only or round_.context_limited else len(round_.sampled) + 1
    return min(limit, response_budget) if response_budget is not None else limit


def replay_round(  # noqa: PLR0913, PLR0917 - Replay requires model, sampling, and trace context.
    engine: Engine,
    policy: SamplingPolicy,
    rng: Rng,
    task_id: str,
    ordinal: int,
    round_: ReplayRound,
    response_budget: int | None = None,
) -> ReplayResult:
    """Replay one recorded prompt without preflight, translation, or tool calls."""
    transcript = Transcript()
    if round_.prompt:
        transcript.append(round_.prompt, "system")
    for tokens in (round_.prompt[: round_.cached], round_.prompt):
        if tokens:
            list(engine._worker.submit(SyncJob(tokens)))
    stream = RoundStream(engine.info, round_.think)
    events = list(
        engine._worker.submit(
            GenerateJob(
                policy,
                rng,
                generation_limit(engine, round_, response_budget),
                transcript,
                reserve=_reserve(engine, round_),
                think_mode=round_.think,
                stream=stream,
                control=stream.control,
            )
        )
    )
    tool_call, terminal = _check_round(
        engine,
        task_id,
        ordinal,
        round_,
        events,
        transcript,
        response_budget,
    )
    return ReplayResult(
        task_id,
        ordinal,
        True,
        f"{task_id} round {ordinal}: {len(round_.sampled)} tokens; {terminal}",
        tokens=len(round_.sampled),
        tool_call=tool_call,
        terminal_reason=terminal,
    )
