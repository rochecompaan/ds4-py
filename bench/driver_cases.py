"""Shared case manifests and validated NDJSON evidence for replay drivers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from bench.comparison import MAX_LOGIT_STEPS, TOP_K, StepCapture, TopLogit
from bench.provenance import ProgressChunk, SyncCall

if TYPE_CHECKING:
    from ds4py._json import JsonValue
    from ds4py.engine.replay_types import ReplayTrace

_MANIFEST_FIELDS = (
    "task_id",
    "attempt",
    "round_index",
    "token_file",
    "prompt_tokens",
    "split_at",
    "steps",
    "think",
)


class DriverEvidenceError(ValueError):
    """A manifest or evidence stream is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class DriverCase:
    task_id: str
    attempt: int
    round_index: int
    token_file: Path
    prompt_tokens: int
    split_at: int
    steps: int
    think: int


@dataclass(frozen=True)
class DriverRoundEvidence:
    source_trace: Path
    case: DriverCase
    sync_calls: tuple[SyncCall, ...]
    progress_chunks: tuple[ProgressChunk, ...]
    steps: tuple[StepCapture, ...]
    prefill_seconds: float
    decode_seconds: float
    terminal_reason: str | None
    observed_libds4_sha256: str | None = None
    observed_model_path: str | None = None
    observed_backend: str | None = None
    observed_ctx: int | None = None
    observed_seed: int | None = None
    observed_temperature: float | None = None
    observed_think: int | None = None


@dataclass
class _RoundBuilder:
    case: DriverCase
    source_trace: Path
    sync_calls: list[SyncCall]
    progress_chunks: list[ProgressChunk]
    steps: list[StepCapture]
    seen_steps: set[int]
    observed_libds4_sha256: str
    observed_model_path: str
    observed_backend: str
    observed_ctx: int
    observed_seed: int
    observed_temperature: float
    observed_think: int


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\t\r\n"):
        raise DriverEvidenceError(f"{field} must be non-empty and contain no tabs or newlines")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise DriverEvidenceError(f"{field} must be an integer >= {minimum}")
    return value


def _text_integer(value: str, field: str, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise DriverEvidenceError(f"{field} must be an integer") from exc
    return _integer(parsed, field, minimum=minimum)


def _finite_number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DriverEvidenceError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DriverEvidenceError(f"{field} must be finite")
    return number


def _seconds(value: object, field: str) -> float:
    seconds = _finite_number(value, field)
    if seconds < 0:
        raise DriverEvidenceError(f"{field} must be non-negative")
    return seconds


def _field(record: dict[str, JsonValue], name: str) -> JsonValue:
    if name not in record:
        raise DriverEvidenceError(f"missing field: {name}")
    return record[name]


def _validate_case(case: DriverCase) -> DriverCase:
    _text(case.task_id, "task_id")
    _text(str(case.token_file), "token_file")
    if case.split_at > case.prompt_tokens:
        raise DriverEvidenceError("cached prefix exceeds prompt length")
    return case


def _case_from_record(
    record: dict[str, JsonValue],
    *,
    round_field: str = "round",
) -> DriverCase:
    case = DriverCase(
        task_id=_text(_field(record, "task_id"), "task_id"),
        attempt=_integer(_field(record, "attempt"), "attempt"),
        round_index=_integer(_field(record, round_field), "round_index"),
        token_file=Path(_text(_field(record, "token_file"), "token_file")),
        prompt_tokens=_integer(_field(record, "prompt_tokens"), "prompt_tokens", minimum=1),
        split_at=_integer(_field(record, "split_at"), "split_at"),
        steps=_integer(_field(record, "steps"), "steps"),
        think=_integer(_field(record, "think"), "think"),
    )
    return _validate_case(case)


def read_case_manifest(path: Path) -> tuple[DriverCase, ...]:
    """Read the headered TSV manifest consumed by both reference drivers."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "\t".join(_MANIFEST_FIELDS):
        raise DriverEvidenceError("invalid case manifest header")
    cases = []
    seen = set()
    for line_number, line in enumerate(lines[1:], 2):
        columns = line.split("\t")
        if len(columns) != len(_MANIFEST_FIELDS):
            raise DriverEvidenceError(f"{path}:{line_number}: wrong manifest column count")
        record: dict[str, JsonValue] = dict(zip(_MANIFEST_FIELDS, columns, strict=True))
        for name in ("attempt", "round_index", "prompt_tokens", "split_at", "steps", "think"):
            record[name] = _text_integer(_text(record[name], name), name)
        case = _case_from_record(record, round_field="round_index")
        key = (case.task_id, case.attempt, case.round_index)
        if key in seen:
            raise DriverEvidenceError(f"duplicate case: {key}")
        seen.add(key)
        cases.append(case)
    if not cases:
        raise DriverEvidenceError("case manifest contains no cases")
    return tuple(cases)


def load_case_tokens(case: DriverCase) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Load prompt IDs followed by exactly the requested recorded sampled IDs."""
    lines = case.token_file.read_text(encoding="utf-8").splitlines()
    expected_count = case.prompt_tokens + case.steps
    if len(lines) != expected_count:
        raise DriverEvidenceError(f"expected {expected_count} token IDs, found {len(lines)}")
    tokens = tuple(_text_integer(line, "token") for line in lines)
    return tokens[: case.prompt_tokens], tokens[case.prompt_tokens :]


def export_trace_cases(
    trace: ReplayTrace,
    *,
    task_id: str,
    attempt: int,
    destination: Path,
    max_steps: int = MAX_LOGIT_STEPS,
) -> Path:
    """Export every selected round to one TSV manifest and one token file per round."""
    _text(task_id, "task_id")
    _integer(attempt, "attempt")
    _integer(max_steps, "max_steps")
    rows = []
    payloads = []
    for ordinal, round_ in enumerate(trace.rounds):
        token_file = (destination / f"case-{ordinal:05d}.tokens").resolve()
        case = _validate_case(
            DriverCase(
                task_id=task_id,
                attempt=attempt,
                round_index=_integer(round_.number, "round_index"),
                token_file=token_file,
                prompt_tokens=len(round_.prompt),
                split_at=_integer(round_.cached, "split_at"),
                steps=min(max_steps, len(round_.sampled)),
                think=_integer(round_.think, "think"),
            )
        )
        if not case.prompt_tokens:
            raise DriverEvidenceError("prompt_tokens must be positive")
        tokens = tuple(_integer(token, "token") for token in (*round_.prompt, *round_.sampled[: case.steps]))
        values = (
            case.task_id,
            case.attempt,
            case.round_index,
            case.token_file,
            case.prompt_tokens,
            case.split_at,
            case.steps,
            case.think,
        )
        rows.append("\t".join(str(value) for value in values))
        payloads.append((token_file, "".join(f"{token}\n" for token in tokens)))
    if not rows:
        raise DriverEvidenceError("trace contains no selected rounds")
    destination.mkdir(parents=True, exist_ok=True)
    for token_file, payload in payloads:
        token_file.write_text(payload, encoding="utf-8")
    manifest = destination / "cases.tsv"
    manifest.write_text("\t".join(_MANIFEST_FIELDS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def export_round_zero_cap_case(
    trace: ReplayTrace,
    *,
    task_id: str,
    attempt: int,
    destination: Path,
    response_budget: int,
) -> Path:
    """Export round 0 with padding available only for unseen expected IDs."""
    if response_budget < 1:
        raise DriverEvidenceError("response budget must be positive")
    if not trace.rounds or trace.rounds[0].number != 0:
        raise DriverEvidenceError("cap parity requires a round-0 trace")
    first = trace.rounds[0]
    if len(first.sampled) > response_budget:
        raise DriverEvidenceError("recorded round 0 exceeds the response budget")
    padding = (0,) * (response_budget - len(first.sampled))
    padded = replace(first, sampled=(*first.sampled, *padding))
    cap_trace = replace(trace, rounds=(padded,))
    return export_trace_cases(
        cap_trace,
        task_id=task_id,
        attempt=attempt,
        destination=destination,
        max_steps=response_budget,
    )


def _start_round(record: dict[str, JsonValue]) -> _RoundBuilder:
    return _RoundBuilder(
        case=_case_from_record(record),
        source_trace=Path(_text(_field(record, "source_trace"), "source_trace")),
        sync_calls=[],
        progress_chunks=[],
        steps=[],
        seen_steps=set(),
        observed_libds4_sha256=_text(_field(record, "observed_libds4_sha256"), "observed_libds4_sha256"),
        observed_model_path=_text(_field(record, "observed_model_path"), "observed_model_path"),
        observed_backend=_text(_field(record, "observed_backend"), "observed_backend"),
        observed_ctx=_integer(_field(record, "observed_ctx"), "observed_ctx", minimum=1),
        observed_seed=_integer(_field(record, "observed_seed"), "observed_seed"),
        observed_temperature=_finite_number(_field(record, "observed_temperature"), "observed_temperature"),
        observed_think=_integer(_field(record, "observed_think"), "observed_think"),
    )


def _add_sync(builder: _RoundBuilder, record: dict[str, JsonValue]) -> None:
    if builder.steps:
        raise DriverEvidenceError("sync record follows decode steps")
    case = builder.case
    expected_labels = ("full",) if case.split_at == 0 else ("prefix", "full")
    index = len(builder.sync_calls)
    if index >= len(expected_labels):
        raise DriverEvidenceError("too many sync records")
    label = _text(_field(record, "label"), "label")
    if label != expected_labels[index]:
        raise DriverEvidenceError(f"expected {expected_labels[index]} sync, found {label}")
    expected_input = case.split_at if label == "prefix" else case.prompt_tokens
    call = SyncCall(
        label=label,
        input_tokens=_integer(_field(record, "input_tokens"), "input_tokens"),
        cached_tokens=_integer(_field(record, "cached_tokens"), "cached_tokens"),
        prefill_tokens=_integer(_field(record, "prefill_tokens"), "prefill_tokens"),
        position=_integer(_field(record, "position"), "position"),
    )
    _seconds(_field(record, "elapsed_seconds"), "elapsed_seconds")
    if call.input_tokens != expected_input or call.position != expected_input:
        raise DriverEvidenceError(f"{label} sync has wrong input length or position")
    if call.cached_tokens + call.prefill_tokens != call.input_tokens:
        raise DriverEvidenceError(f"{label} sync cached/prefill lengths do not sum to input")
    builder.sync_calls.append(call)


def _add_prefill(builder: _RoundBuilder, record: dict[str, JsonValue]) -> None:
    if not builder.sync_calls or builder.steps:
        raise DriverEvidenceError("prefill progress is outside a sync phase")
    chunk = ProgressChunk(
        done=_integer(_field(record, "done"), "done"),
        total=_integer(_field(record, "total"), "total"),
    )
    if chunk.done > chunk.total:
        raise DriverEvidenceError("prefill done exceeds total")
    builder.progress_chunks.append(chunk)


def _add_step(builder: _RoundBuilder, record: dict[str, JsonValue]) -> None:
    if not builder.sync_calls or builder.sync_calls[-1].label != "full":
        raise DriverEvidenceError("decode step precedes full sync")
    index = _integer(_field(record, "index"), "index")
    if index in builder.seen_steps:
        raise DriverEvidenceError(f"duplicate step index: {index}")
    if index != len(builder.steps):
        raise DriverEvidenceError(f"out-of-order step index: {index}")
    position = _integer(_field(record, "position"), "position")
    if position != builder.case.prompt_tokens + index:
        raise DriverEvidenceError("step has wrong pre-sample position")
    raw_top = _field(record, "top")
    if not isinstance(raw_top, list) or len(raw_top) != TOP_K:
        raise DriverEvidenceError(f"step must contain exactly top-{TOP_K} logits")
    top = []
    for entry in raw_top:
        if not isinstance(entry, dict):
            raise DriverEvidenceError("top logit entry must be an object")
        item = entry
        logit = _finite_number(_field(item, "logit"), "logit")
        top.append(
            TopLogit(
                token=_integer(_field(item, "token"), "top token"),
                logit=logit,
            )
        )
    builder.steps.append(
        StepCapture(
            position=position,
            top=tuple(top),
            selected_token=_integer(_field(record, "selected_token"), "selected_token"),
        )
    )
    builder.seen_steps.add(index)


def _finish_round(builder: _RoundBuilder, record: dict[str, JsonValue]) -> DriverRoundEvidence:
    expected_syncs = 1 if builder.case.split_at == 0 else 2
    if len(builder.sync_calls) != expected_syncs:
        raise DriverEvidenceError("round ended before required sync evidence")
    prefill_tokens = _integer(_field(record, "prefill_tokens"), "prefill_tokens")
    decode_steps = _integer(_field(record, "decode_steps"), "decode_steps")
    if prefill_tokens != sum(call.prefill_tokens for call in builder.sync_calls):
        raise DriverEvidenceError("round_end prefill token total differs from sync evidence")
    if decode_steps != len(builder.steps):
        raise DriverEvidenceError("round_end decode total differs from captured steps")
    terminal = _field(record, "terminal_reason")
    if terminal is not None:
        terminal = _text(terminal, "terminal_reason")
    return DriverRoundEvidence(
        source_trace=builder.source_trace,
        case=builder.case,
        sync_calls=tuple(builder.sync_calls),
        progress_chunks=tuple(builder.progress_chunks),
        steps=tuple(builder.steps),
        prefill_seconds=_seconds(_field(record, "prefill_seconds"), "prefill_seconds"),
        decode_seconds=_seconds(_field(record, "decode_seconds"), "decode_seconds"),
        terminal_reason=terminal,
        observed_libds4_sha256=builder.observed_libds4_sha256,
        observed_model_path=builder.observed_model_path,
        observed_backend=builder.observed_backend,
        observed_ctx=builder.observed_ctx,
        observed_seed=builder.observed_seed,
        observed_temperature=builder.observed_temperature,
        observed_think=builder.observed_think,
    )


def _object(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DriverEvidenceError("NDJSON record must be an object")
    return cast("dict[str, JsonValue]", value)


def _begin_round(
    record: dict[str, JsonValue],
    current: _RoundBuilder | None,
    seen_rounds: set[tuple[str, int, int]],
) -> _RoundBuilder:
    if current is not None:
        raise DriverEvidenceError("new round started before round_end")
    current = _start_round(record)
    key = (current.case.task_id, current.case.attempt, current.case.round_index)
    if key in seen_rounds:
        raise DriverEvidenceError(f"duplicate round: {key}")
    seen_rounds.add(key)
    return current


def _process_record(
    record: dict[str, JsonValue],
    current: _RoundBuilder | None,
    seen_rounds: set[tuple[str, int, int]],
    rounds: list[DriverRoundEvidence],
) -> _RoundBuilder | None:
    kind = _text(_field(record, "kind"), "kind")
    if kind == "round_start":
        return _begin_round(record, current, seen_rounds)
    if current is None:
        raise DriverEvidenceError(f"{kind} record outside a round")
    if kind == "sync":
        _add_sync(current, record)
    elif kind == "prefill":
        _add_prefill(current, record)
    elif kind == "step":
        _add_step(current, record)
    elif kind == "round_end":
        rounds.append(_finish_round(current, record))
        return None
    else:
        raise DriverEvidenceError(f"unknown record kind: {kind}")
    return current


def read_driver_ndjson(path: Path) -> tuple[DriverRoundEvidence, ...]:
    """Parse all rounds, rejecting missing, duplicate, or inconsistent evidence."""
    rounds: list[DriverRoundEvidence] = []
    current: _RoundBuilder | None = None
    seen_rounds: set[tuple[str, int, int]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = _object(json.loads(line))
            current = _process_record(record, current, seen_rounds, rounds)
        except (json.JSONDecodeError, DriverEvidenceError) as exc:
            raise DriverEvidenceError(f"{path}:{line_number}: {exc}") from exc
    if current is not None:
        raise DriverEvidenceError("evidence ended before round_end")
    if not rounds:
        raise DriverEvidenceError("evidence contains no rounds")
    return tuple(rounds)
