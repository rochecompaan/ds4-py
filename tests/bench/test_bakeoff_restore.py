"""Bakeoff save evidence and fresh-process restore checks."""
from typing import Any, cast

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.run_bakeoff import drive_pty
from bench.restore_check import (
    RestoreError,
    RestoreRun,
    check_restore,
    make_stripped_copy,
    restore_command,
    run_attempt_with_save,
    validate_save,
)
import ds4py
from ds4py.engine.session import Engine
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.engine.sampling import SamplingPolicy
from ds4py.model import DS4Model
from ds4py.runtime import AgentRuntime
from ds4py.store import (
    KV_HEADER,
    SaveRecord,
    SegmentProvenance,
    SessionStore,
    token_digest,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord as StepRunRecord,
)
from typing import Never


def restore_fake_engine(script: Any) -> Engine:
    info = replace(
        FakeNative()._info,
        name="fake-ds4", model_id=7, family="qwen",
        is_deepseek41=False, is_glm_dsa=False, is_qwen4=True,
    )
    return ds4py.open(
        "f.gguf", ctx=65536, think=ThinkMode.NONE,
        native=FakeNative(info=info, script=script),
    )


def restore_script(*texts: Any) -> Any:
    rounds = [tuple(tokenize_rendered(text)) for text in texts]
    state = {"round": -1}

    def next_token(prompt: Any, generated: Any) -> Any:
        if not generated:
            state["round"] += 1
        sequence = rounds[min(state["round"], len(rounds) - 1)]
        return sequence[len(generated)] if len(generated) < len(sequence) else FAKE_EOS

    return next_token


def make_save(root: Any, session_id: Any="session-1") -> Any:
    store = SessionStore(root / "home")
    save_dir = store.session_dir(session_id)
    save_dir.mkdir(parents=True)
    run_id = session_id + ".save-1"
    record = SaveRecord(
        session_id=session_id, conversation_id=session_id, run_id=run_id,
        step_index=0, model_id=1, quant_bits=2, ctx=100000, title="task",
        created_at="2026-01-01T00:00:00+00:00",
        last_used="2026-01-01T00:00:00+00:00", payload_bytes=3, tokens=1,
        tokens_sha256=token_digest([1]),
        segments=(SegmentProvenance(0, 1, "user", "key", "a" * 64, "test"),),
    )
    store.write_kv_atomic(save_dir / "session.kv", b"kv!")
    store.write_record_atomic(save_dir / "save.json", record)
    (save_dir / "events.jsonl").write_text(
        json.dumps({"kind": "run_start", "run_id": "run-1", "payload": {}}) + "\n" +
        json.dumps({"kind": "run_end", "run_id": "run-1", "payload": {"status": "success"}}) + "\n"
    )

    async def snapshot() -> None:
        steps = FileStepStore(save_dir / "steps")
        await steps.register_run(StepRunRecord(
            run_id=run_id, conversation_id=session_id, agent_name="ds4py",
        ))
        await steps.save_snapshot(ContinuableSnapshot(
            run_id=run_id, step_index=0, messages=[], conversation_id=session_id,
            agent_name="ds4py",
        ))

    asyncio.run(snapshot())
    return save_dir


def write_trace(path: Any, sampled: Any) -> None:
    path.write_text(json.dumps({
        "session_id": "restore", "run_id": "followup", "step_index": 0,
        "round_index": 0, "prompt": [1, 2], "think": 0, "cached": 0,
        "prefill_suffix": [1, 2], "sampled": list(sampled),
    }) + "\n")


def successful_driver(calls: Any, tokens: Any=(1, 2, 3), reason: Any="stop_token") -> Any:
    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        calls.append((home, session_id, case))
        write_trace(trace_path, tokens)
        return RestoreRun(
            trace_path=trace_path, terminal_reason=reason, switch_prefill=0,
            followup_plan="extend" if case == 1 else "rebuild",
            followup_prefill=2 if case == 1 else 12,
        )
    return drive


def test_validate_save_accepts_complete_record_snapshot_events_and_kv(tmp_path: Path) -> None:
    save = make_save(tmp_path)
    evidence = validate_save(save)

    assert evidence.record.session_id == "session-1"
    assert evidence.events_path.is_file()
    assert evidence.snapshot_path.is_file()
    assert SessionStore().read_kv_header(save / "session.kv") == 3


@pytest.mark.parametrize("missing", ["events", "snapshot", "kv"])
def test_validate_save_rejects_missing_required_evidence(tmp_path: Path, missing: Any) -> None:
    save = make_save(tmp_path)
    if missing == "events":
        (save / "events.jsonl").unlink()
    elif missing == "snapshot":
        shutil.rmtree(save / "steps")
    else:
        (save / "session.kv").unlink()

    with pytest.raises(RestoreError):
        validate_save(save)


def test_stripped_copy_has_zero_payload_and_preserves_snapshot_and_record(tmp_path: Path) -> None:
    save = make_save(tmp_path)
    stripped = make_stripped_copy(save, tmp_path / "stripped-home")

    assert stripped != save
    assert SessionStore().read_kv_header(stripped / "session.kv") == 0
    assert (stripped / "save.json").read_bytes() == (save / "save.json").read_bytes()
    assert (stripped / "steps").is_dir()
    assert (stripped / "session.kv").stat().st_size == KV_HEADER.size


def test_restore_cases_compare_tokens_and_enforce_prefill_paths(tmp_path: Path) -> None:
    save = make_save(tmp_path)
    calls = []

    result = check_restore(save, tmp_path / "work", runner=successful_driver(calls))

    assert result.passed, result.detail
    assert [call[2] for call in calls] == [1, 2]
    assert all(call[1] == "session-1" for call in calls)
    stripped = cast(Any, result.stripped_save_dir)
    assert SessionStore().read_kv_header(stripped / "session.kv") == 0


def test_restore_token_mismatch_fails_with_index(tmp_path: Path) -> None:
    save = make_save(tmp_path)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        write_trace(trace_path, (1, 2, 3) if case == 1 else (1, 9, 3))
        return RestoreRun(
            trace_path, "stop_token", 0,
            "extend" if case == 1 else "rebuild", 2 if case == 1 else 12,
        )

    result = check_restore(save, tmp_path / "work", runner=drive)

    assert not result.passed
    assert "token 1" in result.detail


@pytest.mark.parametrize("case1_prefill,case2_prefill", [(1, 12), (0, 0)])
def test_restore_requires_no_case1_prefill_and_real_case2_prefill(
    tmp_path: Path, case1_prefill: Any, case2_prefill: Any,
) -> None:
    save = make_save(tmp_path)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        write_trace(trace_path, (1, 2))
        return RestoreRun(
            trace_path, "stop_token", case1_prefill if case == 1 else 0,
            "extend" if case == 1 else "rebuild",
            2 if case == 1 else case2_prefill,
        )

    result = check_restore(save, tmp_path / "work", runner=drive)
    assert not result.passed


def test_short_restore_sequences_must_share_terminal_reason(tmp_path: Path) -> None:
    save = make_save(tmp_path)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        write_trace(trace_path, (1, 2))
        return RestoreRun(
            trace_path, "stop_token" if case == 1 else "max_tokens", 0,
            "extend" if case == 1 else "rebuild", 2 if case == 1 else 12,
        )

    result = check_restore(save, tmp_path / "work", runner=drive)
    assert not result.passed
    assert "terminal" in result.detail


def test_restore_driver_preserves_noninteractive_prompt_composition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "home")
    with restore_fake_engine(restore_script("saved answer")) as engine:
        model = DS4Model(engine.session(), sampling=SamplingPolicy(seed=1))
        runtime = AgentRuntime(
            model, project, store=store, interactive=False, read_only=True,
        )

        async def save_attempt() -> Any:
            await runtime.run("question")
            return await runtime.save()

        try:
            record = asyncio.run(save_attempt())
        finally:
            asyncio.run(runtime.close())

    replacement_engine = restore_fake_engine(restore_script("follow-up answer"))
    metadata = tmp_path / "restore.json"
    trace = tmp_path / "restore.jsonl"
    monkeypatch.chdir(project)
    monkeypatch.setenv("DS4PY_HOME", str(store.root))
    monkeypatch.setattr(ds4py, "open", lambda *args, **kwargs: replacement_engine)

    status = restore_command([
        "--session-id", record.session_id, "--metadata", str(metadata),
        "--model", "f.gguf", "--backend", "cpu", "--ctx", "65536",
        "--temp", "0", "--seed", "1", "--tokens", "256", "--trace", str(trace),
    ])

    assert status == 0
    assert json.loads(metadata.read_text())["followup_plan"] == "extend"


def test_default_noninteractive_attempt_uses_its_existing_single_save(monkeypatch: pytest.MonkeyPatch) -> None:
    class Runtime:
        def __init__(self) -> None:
            self.saved = 0

        async def save(self) -> None:
            self.saved += 1

    runtime: Any = Runtime()

    async def noninteractive(runtime: Any, prompt: Any) -> int:
        await runtime.save()
        return 0

    monkeypatch.setattr("ds4py.noninteractive.run_noninteractive", noninteractive)
    assert asyncio.run(run_attempt_with_save(runtime, "task")) == 0
    assert runtime.saved == 1


def test_attempt_saves_in_finally_even_when_prompt_fails() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.saved = 0

        async def save(self) -> None:
            self.saved += 1

    runtime: Any = Runtime()

    async def fail() -> Never:
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        asyncio.run(run_attempt_with_save(runtime, "task", operation=fail))
    assert runtime.saved == 1


def interactive_script(root: Any) -> Any:
    path = root / "interactive.py"
    path.write_text('''\
print("PREFILL", flush=True)
while True:
    try:
        line = input("> ")
    except KeyboardInterrupt:
        print("\\nINTERRUPTED", flush=True)
        continue
    if line == "continue":
        print("RESUMED", flush=True)
        break
''')
    return path


def test_pty_interrupt_resumes(tmp_path: Path) -> None:
    script = interactive_script(tmp_path)
    interrupted = drive_pty(
        [sys.executable, str(script)],
        steps=[("PREFILL", "\x03"), ("INTERRUPTED", "continue\n"), ("RESUMED", None)],
        timeout=5,
    )

    assert interrupted.passed, interrupted.detail
    assert "RESUMED" in interrupted.detail
