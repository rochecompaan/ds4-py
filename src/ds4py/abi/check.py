"""ABI diagnostics and opt-in model-backed sync/restore checks."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

from ds4py._native.api import LibNative
from ds4py._native.protocol import EngineConfig
from ds4py.engine.worker import EngineWorker, SyncJob, CheckpointJob, GenerateJob
from ds4py.engine.events import Synced, Token, Stopped
from ds4py.engine.sampling import SamplingPolicy, Rng
from ds4py.engine.transcript import Transcript

from ds4py._native.loader import load_library, resolve_lib_path
from ds4py.errors import AbiError, EngineError
from .manifest import Manifest, Validation, manifest_path


@dataclass
class CheckReport:
    lib_path: Path
    manifest: Manifest | None
    validation: Validation | None
    extensions: dict[str, str]
    error: str | None


def check_library(lib: str | Path | None) -> CheckReport:
    path = resolve_lib_path(lib)
    try:
        library = load_library(lib)
    except AbiError as exc:
        return CheckReport(path, None, None, {}, str(exc))
    checked = library.validation.available | library.validation.absent
    extensions = {name: record.source for name, record in library.manifest.bindings.items()
                  if name in checked and record.source != "core"}
    return CheckReport(library.path, library.manifest, library.validation, extensions, None)


def format_report(r: CheckReport) -> str:
    if r.error is not None:
        return f"library:  {r.lib_path}\nerror:    {r.error}"
    m, v = r.manifest, r.validation
    assert m is not None and v is not None
    tree = (f"{m.source_path} at {m.source_commit[:12]}" if m.source_commit is not None
            else f"{m.source_path} (not a git checkout)")
    lines = [f"library:  {r.lib_path} ({m.backend})", f"ds4 tree: {tree}",
             f"headers:  {'known' if m.headers_known else 'unknown'}"]
    for warning in v.warnings:
        lines.append(
            f"          warning: {warning.replace(': ', ' ', 1)}; "
            "bindings and layouts were checked, but this tree is not a tested pin"
        )
    required = sum(m.bindings[name].required for name in v.available)
    lines.extend(["layouts:  ok", f"bindings: {required} required ok"])
    optional = [name for name in m.bindings
                if name in v.available | v.absent
                and (not m.bindings[name].required or name in r.extensions)]
    for name in optional:
        status = "absent" if name in v.absent else "present"
        source = r.extensions.get(name, m.bindings[name].source)
        lines.append(f"optional: {name}  {status}  ({source})")
    if not optional:
        lines.append("optional: none")
    if "sync_plans" in m.live_checks:
        sync = m.live_checks["sync_plans"]
        lines.append(f"sync:     {'ok' if sync['ok'] else 'failed'} ({sync['message']})")
    if "restore" in m.live_checks:
        restore = m.live_checks["restore"]
        lines.append(f"restore:  {'ok' if restore['ok'] else 'disabled'} ({restore['message']})")
    else:
        lines.append("restore:  not verified (run `ds4py check --live --model PATH`)")
    return "\n".join(lines)


LIVE_TEXT = (
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of "
    "foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, "
    "it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had "
    "everything before us, we had nothing before us, we were all going direct to Heaven, we were all going "
    "direct the other way -- in short, the period was so far like the present period, that some of its "
    "noisiest authorities insisted on its being received, for good or for evil, in the superlative degree "
    "of comparison only. "
)

LIVE_CHECKPOINT = "live-check"


def _timed_sync(worker, tokens, clock):
    start = clock()
    synced = None
    seconds = 0.0
    for event in worker.submit(SyncJob(tuple(tokens))):
        if isinstance(event, Synced):
            synced, seconds = event, clock() - start
    if synced is None:
        raise EngineError("ds4py check --live sync was interrupted")
    return synced, seconds


def _greedy_run(worker, count):
    tokens = []
    for event in worker.submit(GenerateJob(SamplingPolicy(temperature=0.0), Rng(1),
                                          count, Transcript())):
        if isinstance(event, Token):
            tokens.append(event.id)
        elif isinstance(event, Stopped):
            return tokens, event.reason
    raise EngineError("ds4py check --live generation did not stop")


def _compare_runs(a, b):
    left, left_reason = a
    right, right_reason = b
    if a == b:
        return {"ok": True,
                "message": f"{len(left)} greedy tokens after a restore match a rebuild",
                "tokens": len(left)}
    i = 0
    while i < min(len(left), len(right)) and left[i] == right[i]:
        i += 1
    x = left[i] if i < len(left) else f"end ({left_reason})"
    y = right[i] if i < len(right) else f"end ({right_reason})"
    return {"ok": False, "message": f"the runs differ at token {i}: {x} after a restore, "
            f"{y} after a rebuild", "tokens": len(left)}


def run_live_checks(worker: EngineWorker, *, clock: Callable[[], float] = time.monotonic,
                    prefix: int = 2048, tail: int = 64, generate: int = 256) -> dict[str, dict]:
    """Compare planned reuse counts and greedy restore/rebuild output, not speed."""
    ctx = worker.call(lambda e, s: s.ctx()).result()
    if prefix + generate + 1 > ctx:
        raise EngineError(f"ds4py check --live needs a context of at least {prefix + generate + 1}")
    text = LIVE_TEXT
    tokens = worker.call(lambda e, s: e.tokenize(text)).result()
    while len(tokens) < prefix:
        text += LIVE_TEXT
        tokens = worker.call(lambda e, s: e.tokenize(text)).result()
    tokens = tokens[:prefix]
    _timed_sync(worker, tokens[:prefix - tail], clock)
    list(worker.submit(CheckpointJob(LIVE_CHECKPOINT)))
    extend = _timed_sync(worker, tokens, clock)
    worker.call(lambda e, s: s.invalidate()).result()
    restore = _timed_sync(worker, tokens, clock)
    a = _greedy_run(worker, generate)
    list(worker.submit(CheckpointJob(LIVE_CHECKPOINT, drop=True)))
    worker.call(lambda e, s: s.invalidate()).result()
    rebuild = _timed_sync(worker, tokens, clock)
    b = _greedy_run(worker, generate)

    steps = {"extend": extend, "restore": restore, "rebuild": rebuild}
    message = None
    for step, (event, _) in steps.items():
        expected = prefix if step == "rebuild" else tail
        plan = event.plan
        if plan.kind != step or plan.to_prefill != expected or (
                step == "restore" and plan.checkpoint != LIVE_CHECKPOINT):
            message = (f"the {step} sync planned {plan.kind} of {plan.to_prefill} tokens, "
                       f"expected {step} of {expected}")
        elif event.prefilled != plan.to_prefill:
            message = (f"the {step} sync prefilled {event.prefilled} tokens; "
                       f"its plan prefills {plan.to_prefill}")
        if message is not None:
            break
    ok = message is None
    if ok:
        message = (f"extend of {tail} tokens took {extend[1]:.3f} s, "
                   f"restore of {tail} took {restore[1]:.3f} s, "
                   f"rebuild of {prefix} took {rebuild[1]:.3f} s")
    return {"sync_plans": {"ok": ok, "message": message,
                           "prefilled": {k: v[0].prefilled for k, v in steps.items()},
                           "seconds": {k: v[1] for k, v in steps.items()}},
            "restore": _compare_runs(a, b)}


def check_live(lib: str | Path | None, model: Path, *, ctx: int = 4096) -> CheckReport:
    """Run checks with restore enabled, then persist their result beside the library."""
    library = load_library(lib)
    native = LibNative(library)
    native.restore_enabled = True
    worker = EngineWorker(native, EngineConfig(Path(model), ctx=ctx, backend=native.backend))
    try:
        checks = run_live_checks(worker)
    finally:
        worker.close()
    library.manifest.live_checks = checks
    library.manifest.write(manifest_path(library.path))
    return check_library(library.path)
