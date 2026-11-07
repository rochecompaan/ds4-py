"""Replay manifest cases through ds4-py and emit strict NDJSON evidence."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ds4py
from bench.comparison import TOP_K, StepCapture, TopLogit
from bench.driver_cases import DriverCase, load_case_tokens, read_case_manifest
from bench.provenance import Q8_F16_CACHE_ENV, Q8_F16_CACHE_MB, ProgressChunk
from ds4py._native.loader import resolve_lib_path
from ds4py._native.protocol import SyncStatus
from ds4py.abi.manifest import file_sha256
from ds4py.engine.decode import GenerateJob, generate
from ds4py.engine.events import Token
from ds4py.engine.sampling import Rng, SamplingPolicy
from ds4py.engine.transcript import Transcript
from ds4py.engine.worker import ResetJob
from ds4py.errors import EngineError

if TYPE_CHECKING:
    from ds4py._json import JsonValue
    from ds4py._native.protocol import EngineHandle, SessionHandle
    from ds4py.engine.events import Event
    from ds4py.engine.logits import LogitsView


@dataclass(frozen=True)
class _SyncResult:
    cached: int
    prefilled: int
    position: int
    elapsed: float
    chunks: tuple[ProgressChunk, ...]


@dataclass(frozen=True)
class _CapturedStep:
    capture: StepCapture
    expected: int
    expected_logit: float


@dataclass(frozen=True)
class _DecodeResult:
    steps: tuple[_CapturedStep, ...]
    terminal_reason: str
    elapsed: float


@dataclass(frozen=True)
class _DriverIdentity:
    libds4_sha256: str
    model_path: str
    backend: str
    ctx: int


def _emit(record: dict[str, JsonValue]) -> None:
    print(json.dumps(record, separators=(",", ":"), allow_nan=False), flush=True)


def _sync(engine: ds4py.Engine, tokens: tuple[int, ...], label: str) -> _SyncResult:
    def run(_engine: EngineHandle, session: SessionHandle) -> _SyncResult:
        common = session.common_prefix(tokens)
        if common < 0:
            raise EngineError("native common-prefix calculation failed")
        cached = common if common == session.pos() else 0
        prefilled = len(tokens) - cached
        chunks: list[ProgressChunk] = []

        def progress(event: str, current: int, _total: int) -> None:
            if event == "prefill_chunk":
                done = max(0, min(current - cached, prefilled))
                chunks.append(ProgressChunk(done, prefilled))

        start = time.monotonic()
        status = session.sync(tokens, progress=progress)
        elapsed = time.monotonic() - start
        if status is not SyncStatus.DONE:
            raise EngineError(f"{label} sync was interrupted")
        return _SyncResult(cached, prefilled, session.pos(), elapsed, tuple(chunks))

    # Run native timing on the owning worker.
    return engine._worker.call(run).result()


def _decode(
    engine: ds4py.Engine,
    case: DriverCase,
    prompt: tuple[int, ...],
    expected: tuple[int, ...],
) -> _DecodeResult:
    policy = SamplingPolicy.for_engine(engine.info, temperature=0.0, seed=1)
    transcript = Transcript()
    transcript.append(prompt, "system")
    captures: list[_CapturedStep] = []

    def hook(view: LogitsView) -> None:
        values = view.logits()
        top = heapq.nlargest(TOP_K, enumerate(values), key=lambda item: item[1])
        if len(top) != TOP_K:
            raise EngineError(f"native vocabulary returned fewer than {TOP_K} logits")
        entries = tuple(TopLogit(token, float(logit)) for token, logit in top)
        index = len(captures)
        expected_token = expected[index]
        captures.append(
            _CapturedStep(
                capture=StepCapture(view.position, entries, entries[0].token),
                expected=expected_token,
                expected_logit=float(values[expected_token]),
            )
        )

    job = GenerateJob(
        policy,
        Rng(1),
        case.steps,
        transcript,
        reserve=1,
        think_mode=case.think,
        logits_hooks=(hook,),
    )
    events: list[Event] = []

    def run(native_engine: EngineHandle, session: SessionHandle) -> tuple[str, float]:
        start = time.monotonic()
        reason = generate(
            native_engine,
            session,
            job,
            cancelled=lambda: False,
            emit=events.append,
        )
        return reason, time.monotonic() - start

    # Match the C native timing window.
    reason, elapsed = engine._worker.call(run).result()
    actual = [event.id for event in events if isinstance(event, Token)]
    if len(actual) > len(captures) or any(
        token != captures[index].capture.selected_token for index, token in enumerate(actual)
    ):
        raise EngineError("greedy top-logit capture differs from committed decode tokens")
    expected_capture_count = len(actual) + 1 if reason == "stop_token" else len(actual)
    if len(captures) != expected_capture_count:
        raise EngineError("logits hook count differs from generated tokens")
    return _DecodeResult(tuple(captures), reason, elapsed)


def _round_start(source_trace: Path, case: DriverCase, identity: _DriverIdentity) -> dict[str, JsonValue]:
    return {
        "kind": "round_start",
        "source_trace": str(source_trace),
        "task_id": case.task_id,
        "attempt": case.attempt,
        "round": case.round_index,
        "token_file": str(case.token_file),
        "prompt_tokens": case.prompt_tokens,
        "split_at": case.split_at,
        "steps": case.steps,
        "think": case.think,
        "observed_libds4_sha256": identity.libds4_sha256,
        "observed_model_path": identity.model_path,
        "observed_backend": identity.backend,
        "observed_ctx": identity.ctx,
        "observed_seed": 1,
        "observed_temperature": 0.0,
        "observed_think": case.think,
    }


def _run_sync(tokens: tuple[int, ...], case: DriverCase, engine: ds4py.Engine) -> tuple[int, float]:
    prefill_tokens = 0
    prefill_seconds = 0.0
    syncs = []
    if case.split_at:
        syncs.append(("prefix", tokens[: case.split_at]))
    syncs.append(("full", tokens))
    for label, input_tokens in syncs:
        result = _sync(engine, input_tokens, label)
        _emit(
            {
                "kind": "sync",
                "label": label,
                "input_tokens": len(input_tokens),
                "cached_tokens": result.cached,
                "prefill_tokens": result.prefilled,
                "position": result.position,
                "elapsed_seconds": result.elapsed,
            }
        )
        for chunk in result.chunks:
            _emit({"kind": "prefill", "done": chunk.done, "total": chunk.total})
        prefill_tokens += result.prefilled
        prefill_seconds += result.elapsed
    return prefill_tokens, prefill_seconds


def _run_case(engine: ds4py.Engine, source_trace: Path, case: DriverCase, identity: _DriverIdentity) -> None:
    prompt, expected = load_case_tokens(case)
    # Reset the one native session between cases.
    list(engine._worker.submit(ResetJob()))
    _emit(_round_start(source_trace, case, identity))
    prefill_tokens, prefill_seconds = _run_sync(prompt, case, engine)
    result = _decode(engine, case, prompt, expected)
    for index, step in enumerate(result.steps):
        capture = step.capture
        _emit(
            {
                "kind": "step",
                "index": index,
                "position": capture.position,
                "selected_token": capture.selected_token,
                "expected": step.expected,
                "expected_logit": step.expected_logit,
                "top": [{"token": item.token, "logit": item.logit} for item in capture.top],
            }
        )
    _emit(
        {
            "kind": "round_end",
            "prefill_tokens": prefill_tokens,
            "decode_steps": len(result.steps),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": result.elapsed,
            "terminal_reason": result.terminal_reason,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--ctx", type=int, default=100_000)
    parser.add_argument("--lib", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    cases = read_case_manifest(args.cases)
    os.environ[Q8_F16_CACHE_ENV] = str(Q8_F16_CACHE_MB)
    model_path = args.model.expanduser().resolve()
    library = resolve_lib_path(args.lib).expanduser().resolve(strict=True) if cases else None
    identity = _DriverIdentity(
        libds4_sha256=file_sha256(library) if library is not None else "",
        model_path=str(model_path),
        backend=args.backend,
        ctx=args.ctx,
    )
    with ds4py.open(
        model_path,
        ctx=args.ctx,
        backend=args.backend,
        lib=library,
        seed=1,
    ) as engine:
        for case in cases:
            _run_case(engine, args.source_trace, case, identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
