"""M3 save evidence and fresh-process restore checks."""

import asyncio
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.run_m3 import drive_pty
from bench.restore_check import (
    RestoreError,
    RestoreRun,
    check_restore,
    make_stripped_copy,
    run_attempt_with_save,
    validate_save,
)
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


def make_save(root, session_id="session-1"):
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

    async def snapshot():
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


def write_trace(path, sampled):
    path.write_text(json.dumps({
        "session_id": "restore", "run_id": "followup", "step_index": 0,
        "round_index": 0, "prompt": [1, 2], "think": 0, "cached": 0,
        "prefill_suffix": [1, 2], "sampled": list(sampled),
    }) + "\n")


def successful_driver(calls, tokens=(1, 2, 3), reason="stop_token"):
    def drive(home, session_id, case, trace_path):
        calls.append((home, session_id, case))
        write_trace(trace_path, tokens)
        return RestoreRun(
            trace_path=trace_path, terminal_reason=reason, switch_prefill=0,
            followup_plan="extend" if case == 1 else "rebuild",
            followup_prefill=2 if case == 1 else 12,
        )
    return drive


def test_validate_save_accepts_complete_record_snapshot_events_and_kv(tmp_path):
    save = make_save(tmp_path)
    evidence = validate_save(save)

    assert evidence.record.session_id == "session-1"
    assert evidence.events_path.is_file()
    assert evidence.snapshot_path.is_file()
    assert SessionStore().read_kv_header(save / "session.kv") == 3


@pytest.mark.parametrize("missing", ["events", "snapshot", "kv"])
def test_validate_save_rejects_missing_required_evidence(tmp_path, missing):
    save = make_save(tmp_path)
    if missing == "events":
        (save / "events.jsonl").unlink()
    elif missing == "snapshot":
        shutil.rmtree(save / "steps")
    else:
        (save / "session.kv").unlink()

    with pytest.raises(RestoreError):
        validate_save(save)


def test_stripped_copy_has_zero_payload_and_preserves_snapshot_and_record(tmp_path):
    save = make_save(tmp_path)
    stripped = make_stripped_copy(save, tmp_path / "stripped-home")

    assert stripped != save
    assert SessionStore().read_kv_header(stripped / "session.kv") == 0
    assert (stripped / "save.json").read_bytes() == (save / "save.json").read_bytes()
    assert (stripped / "steps").is_dir()
    assert (stripped / "session.kv").stat().st_size == KV_HEADER.size


def test_restore_cases_compare_tokens_and_enforce_prefill_paths(tmp_path):
    save = make_save(tmp_path)
    calls = []

    result = check_restore(save, tmp_path / "work", runner=successful_driver(calls))

    assert result.passed, result.detail
    assert [call[2] for call in calls] == [1, 2]
    assert all(call[1] == "session-1" for call in calls)
    stripped = result.stripped_save_dir
    assert SessionStore().read_kv_header(stripped / "session.kv") == 0


def test_restore_token_mismatch_fails_with_index(tmp_path):
    save = make_save(tmp_path)

    def drive(home, session_id, case, trace_path):
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
    tmp_path, case1_prefill, case2_prefill,
):
    save = make_save(tmp_path)

    def drive(home, session_id, case, trace_path):
        write_trace(trace_path, (1, 2))
        return RestoreRun(
            trace_path, "stop_token", case1_prefill if case == 1 else 0,
            "extend" if case == 1 else "rebuild",
            2 if case == 1 else case2_prefill,
        )

    result = check_restore(save, tmp_path / "work", runner=drive)
    assert not result.passed


def test_short_restore_sequences_must_share_terminal_reason(tmp_path):
    save = make_save(tmp_path)

    def drive(home, session_id, case, trace_path):
        write_trace(trace_path, (1, 2))
        return RestoreRun(
            trace_path, "stop_token" if case == 1 else "max_tokens", 0,
            "extend" if case == 1 else "rebuild", 2 if case == 1 else 12,
        )

    result = check_restore(save, tmp_path / "work", runner=drive)
    assert not result.passed
    assert "terminal" in result.detail


def test_default_noninteractive_attempt_uses_its_existing_single_save(monkeypatch):
    class Runtime:
        def __init__(self):
            self.saved = 0

        async def save(self):
            self.saved += 1

    runtime = Runtime()

    async def noninteractive(runtime, prompt):
        await runtime.save()
        return 0

    monkeypatch.setattr("ds4py.noninteractive.run_noninteractive", noninteractive)
    assert asyncio.run(run_attempt_with_save(runtime, "task")) == 0
    assert runtime.saved == 1


def test_attempt_saves_in_finally_even_when_prompt_fails():
    class Runtime:
        def __init__(self):
            self.saved = 0

        async def save(self):
            self.saved += 1

    runtime = Runtime()

    async def fail():
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        asyncio.run(run_attempt_with_save(runtime, "task", operation=fail))
    assert runtime.saved == 1


def interactive_script(root):
    path = root / "interactive.py"
    path.write_text('''\
import sys

if sys.argv[1] == "interrupt":
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
else:
    try:
        answer = input("approval? [y/N] ")
    except EOFError:
        print("DENIED", flush=True)
        raise SystemExit(2)
    if answer != "y":
        print("DENIED", flush=True)
        raise SystemExit(2)
    print("CONTINUED", flush=True)
''')
    return path


def test_pty_interrupt_resumes_and_approval_continues(tmp_path):
    script = interactive_script(tmp_path)
    interrupted = drive_pty(
        [sys.executable, str(script), "interrupt"],
        steps=[("PREFILL", "\x03"), ("INTERRUPTED", "continue\n"), ("RESUMED", None)],
        timeout=5,
    )
    approved = drive_pty(
        [sys.executable, str(script), "approval"],
        steps=[("approval", "y\n"), ("CONTINUED", None)], timeout=5,
    )

    assert interrupted.passed, interrupted.detail
    assert "RESUMED" in interrupted.detail
    assert approved.passed, approved.detail
    assert "CONTINUED" in approved.detail


def test_pty_eof_denies_approval_instead_of_auto_approving(tmp_path):
    script = interactive_script(tmp_path)
    denied = drive_pty(
        [sys.executable, str(script), "approval"],
        steps=[("approval", "\x04")], timeout=5,
    )

    assert not denied.passed
    assert "DENIED" in denied.detail
