"""Replay a saved attempt's exact history for the Gate 5 restore reference."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from array import array
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import ds4py
from bench.comparison import TOP_K, StepCapture, TopLogit
from bench.provenance import provenance_from_environment
from bench.trace_selection import replay_round
from ds4py.agent_cli import _think_mode, parser
from ds4py.engine.replay import read_replay
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.worker import SyncJob
from ds4py.store import SaveRecord, SessionStore, token_digest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ds4py._json import JsonValue
    from ds4py._native.protocol import EngineHandle, SessionHandle
    from ds4py.engine.replay_types import ReplayRound, ReplayTrace
    from ds4py.engine.session import Engine


class RestoreHistoryError(ValueError):
    """Recorded history and save evidence are incomplete or inconsistent."""


def _json_object(value: object, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise RestoreHistoryError(f"{label} must be an object")
    return cast("dict[str, JsonValue]", value)


def _original_events(save_dir: Path, record: SaveRecord) -> list[dict[str, JsonValue]]:
    events = [
        _json_object(json.loads(line), f"{save_dir / 'events.jsonl'} event")
        for line in (save_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    saved_index = None
    original_run_id = None
    for index, event in enumerate(events):
        if event.get("kind") != "saved":
            continue
        payload = _json_object(event.get("payload"), "saved payload")
        saved_record = _json_object(payload.get("record"), "saved record")
        if saved_record.get("run_id") == record.run_id and saved_record.get("tokens_sha256") == record.tokens_sha256:
            saved_index = index
            original_run_id = event.get("run_id")
            break
    if saved_index is None or not isinstance(original_run_id, str) or not original_run_id:
        raise RestoreHistoryError("save events do not identify the original saved run")
    return [event for event in events[:saved_index] if event.get("run_id") == original_run_id]


def _event_payload(event: dict[str, JsonValue], kind: str) -> dict[str, JsonValue]:
    payload = _json_object(event.get("payload"), f"{kind} payload")
    return _json_object(payload.get("event"), f"{kind} event")


def _validate_event_history(events: Sequence[dict[str, JsonValue]], trace: ReplayTrace, saved_tokens: int) -> None:
    syncs = [_event_payload(event, "Synced") for event in events if event.get("kind") == "Synced"]
    sampled: list[tuple[int, ...]] = []
    current: list[int] = []
    for event in events:
        kind = event.get("kind")
        if kind == "Token":
            token = _event_payload(event, "Token").get("id")
            if type(token) is not int:
                raise RestoreHistoryError("Token event has a non-integer ID")
            current.append(token)
        elif kind == "Stopped":
            sampled.append(tuple(current))
            current = []
    if len(syncs) != len(trace.rounds) + 1:
        raise RestoreHistoryError(f"expected {len(trace.rounds) + 1} original syncs, found {len(syncs)}")
    if len(sampled) != len(trace.rounds):
        raise RestoreHistoryError(f"expected {len(trace.rounds)} original decode rounds, found {len(sampled)}")
    for index, (round_, sync, tokens) in enumerate(zip(trace.rounds, syncs[: len(trace.rounds)], sampled, strict=True)):
        plan = _json_object(sync.get("plan"), f"sync {index} plan")
        expected = (len(round_.prompt), round_.cached, len(round_.prompt) - round_.cached)
        observed = (sync.get("pos"), plan.get("reused"), plan.get("to_prefill"))
        if observed != expected:
            raise RestoreHistoryError(f"round {index} sync evidence differs from the original trace")
        if tokens != round_.sampled:
            raise RestoreHistoryError(f"round {index} sampled IDs differ from the original save events")
    if syncs[-1].get("pos") != saved_tokens:
        raise RestoreHistoryError("final save sync position differs from the save record")


def _validate_token_history(trace: ReplayTrace, saved_tokens: tuple[int, ...]) -> None:
    if not trace.rounds:
        raise RestoreHistoryError("original trace contains no rounds")
    for index, round_ in enumerate(trace.rounds):
        end = len(round_.prompt) + len(round_.sampled)
        if end > len(saved_tokens):
            raise RestoreHistoryError(f"round {index} extends beyond the saved token list")
        if round_.prompt != saved_tokens[: len(round_.prompt)]:
            raise RestoreHistoryError(f"round {index} prompt differs from the saved token list")
        if round_.sampled != saved_tokens[len(round_.prompt) : end]:
            raise RestoreHistoryError(f"round {index} sampled IDs differ from the saved token list")


def _capture(engine: Engine) -> StepCapture:
    def on_worker(native_engine: EngineHandle, session: SessionHandle) -> StepCapture:
        values = array("f", [0.0]) * native_engine.info().vocab_size
        session.copy_logits(values)
        entries = heapq.nlargest(TOP_K, ((float(value), index) for index, value in enumerate(values)))
        top = tuple(TopLogit(token, logit) for logit, token in entries)
        return StepCapture(session.pos(), top, top[0].token)

    return engine._worker.call(on_worker).result()


def _trace_record(
    record: SaveRecord,
    round_: ReplayRound,
    index: int,
) -> str:
    return json.dumps(
        {
            "session_id": record.session_id,
            "run_id": "same-history",
            "step_index": index,
            "round_index": index,
            "prompt": list(round_.prompt),
            "think": round_.think,
            "cached": round_.cached,
            "prefill_suffix": list(round_.prompt[round_.cached :]),
            "sampled": list(round_.sampled),
        }
    )


def _write_replay_trace(
    output: Path,
    record: SaveRecord,
    original: ReplayTrace,
    followup: ReplayRound,
) -> None:
    lines = [_trace_record(record, round_, index) for index, round_ in enumerate(original.rounds)]
    lines.append(_trace_record(record, followup, len(lines)))
    output.write_text("\n".join(lines) + "\n")


def replay_saved_history(
    engine: Engine,
    save_dir: Path,
    restore_trace_path: Path,
    output_trace_path: Path,
    *,
    response_budget: int,
) -> StepCapture:
    """Replay every original round, prefill the fixed follow-up, and capture step 0."""
    save_dir = save_dir.resolve()
    record = SessionStore().read_record(save_dir / "save.json")
    original_path = save_dir.parents[2] / "trace.jsonl"
    original = read_replay(original_path)
    restored = read_replay(restore_trace_path)
    if not restored.rounds:
        raise RestoreHistoryError("restore trace contains no follow-up round")
    followup = restored.rounds[0]
    if record.tokens > len(followup.prompt) or followup.cached != record.tokens:
        raise RestoreHistoryError("restore follow-up does not extend the complete saved token list")
    saved_tokens = tuple(followup.prompt[: record.tokens])
    if token_digest(saved_tokens) != record.tokens_sha256:
        raise RestoreHistoryError("restore prompt token hash differs from the save record")
    _validate_token_history(original, saved_tokens)
    _validate_event_history(_original_events(save_dir, record), original, record.tokens)

    policy = SamplingPolicy.for_engine(engine.info, temperature=0.0, seed=1)
    rng = Rng(1)
    for index, round_ in enumerate(original.rounds):
        replay_round(
            engine,
            policy,
            rng,
            "same-history",
            index,
            round_,
            response_budget=response_budget,
        )
    for tokens in (saved_tokens, followup.prompt):
        if tokens:
            list(engine._worker.submit(SyncJob(tokens)))
    capture = _capture(engine)
    if capture.position != len(followup.prompt):
        raise RestoreHistoryError("same-history follow-up capture has the wrong position")
    reference_round = type(followup)(
        number=len(original.rounds),
        prompt=followup.prompt,
        sampled=(),
        think=followup.think,
        cached=record.tokens,
        context_limited=False,
        prefix_only=True,
    )
    _write_replay_trace(output_trace_path, record, original, reference_round)
    return capture


def same_history_command(argv: Sequence[str]) -> int:
    """Run the fresh-process same-history reference used by ``restore check``."""
    custom = argparse.ArgumentParser()
    custom.add_argument("--session-id", required=True)
    custom.add_argument("--metadata", type=Path, required=True)
    custom.add_argument("--restore-trace", type=Path, required=True)
    own, remaining = custom.parse_known_args(argv)
    args = parser().parse_args(remaining)
    if args.trace is None:
        raise RestoreHistoryError("same-history evidence requires --trace")
    save_dir = SessionStore().session_dir(own.session_id)
    with ds4py.open(
        args.model,
        ctx=args.ctx,
        backend=args.backend,
        lib=args.lib,
        seed=args.seed,
        think=_think_mode(args),
    ) as engine:
        capture = replay_saved_history(
            engine,
            save_dir,
            own.restore_trace,
            Path(args.trace),
            response_budget=args.tokens,
        )
    trace_path = Path(args.trace)
    original_trace = save_dir.parents[2] / "trace.jsonl"
    provenance = provenance_from_environment(
        args,
        reset_policy="same-history-replay",
        trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        source_trace_sha256=hashlib.sha256(original_trace.read_bytes()).hexdigest(),
    )
    followup = read_replay(own.restore_trace).rounds[0]
    own.metadata.write_text(
        json.dumps(
            {
                "switch_prefill": 0,
                "followup_plan": "same-history",
                "followup_prefill": len(followup.prompt) - followup.cached,
                "followup_position": capture.position,
                "first_step": asdict(capture),
                "terminal_reason": "capture_only",
                "trace_path": str(trace_path),
                "provenance": asdict(provenance),
                "save_record_sha256": hashlib.sha256((save_dir / "save.json").read_bytes()).hexdigest(),
                "followup_prompt_sha256": hashlib.sha256(
                    b"Continue the previous task: state the next concrete action, then stop."
                ).hexdigest(),
            }
        )
        + "\n"
    )
    return 0
