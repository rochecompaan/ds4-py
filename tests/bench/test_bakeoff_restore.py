"""Bakeoff save evidence and fresh-process restore checks."""

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from typing import Never

from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
)
from pydantic_ai_harness.step_persistence import (
    RunRecord as StepRunRecord,
)

import ds4py
from bench.comparison import StepCapture, TopLogit
from bench.provenance import (
    CUDA_DISABLE_HOST_REGISTER_ENV,
    NATIVE_PATCHES_ENV,
    Q8_F16_CACHE_ENV,
    NativePatch,
    TraceProvenance,
    validate_session_pair,
)
from bench.restore_check import (
    FOLLOWUP,
    CheckResult,
    CReferenceRun,
    RestoreError,
    RestoreRun,
    check_restore,
    check_result_saves,
    make_stripped_copy,
    restore_command,
    run_attempt_with_save,
    validate_save,
)
from bench.restore_equivalence import RestoreProvenance
from bench.run_bakeoff import drive_pty, interruption_acceptance
from bench.session import BakeoffSession
from ds4py import ThinkMode
from ds4py._native.fake import FAKE_EOS, FakeNative
from ds4py._native.fake_chat import tokenize_rendered
from ds4py.engine.sampling import SamplingPolicy
from ds4py.engine.session import Engine
from ds4py.model import DS4Model
from ds4py.runtime import AgentRuntime
from ds4py.store import (
    KV_HEADER,
    SaveRecord,
    SegmentProvenance,
    SessionStore,
    token_digest,
)


def test_restore_check_script_loads_outside_repository(tmp_path: Path) -> None:
    """A direct worker must import bench without inheriting runner sys.path changes."""
    root = Path(__file__).parents[2]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(root / "bench" / "restore_check.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1, completed.stderr
    assert completed.stderr.startswith("usage: restore_check.py attempt|restore|same-history|check [options]")


def restore_fake_engine(script: Any) -> Engine:
    info = replace(
        FakeNative()._info,
        name="fake-ds4",
        model_id=7,
        family="qwen",
        is_deepseek41=False,
        is_glm_dsa=False,
        is_qwen4=True,
    )
    return ds4py.open(
        "f.gguf",
        ctx=65536,
        think=ThinkMode.NONE,
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


def make_save(root: Any, session_id: Any = "session-1") -> Any:
    store = SessionStore(root / "home")
    save_dir = store.session_dir(session_id)
    save_dir.mkdir(parents=True)
    run_id = session_id + ".save-1"
    record = SaveRecord(
        session_id=session_id,
        conversation_id=session_id,
        run_id=run_id,
        step_index=0,
        model_id=1,
        quant_bits=2,
        ctx=100000,
        title="task",
        created_at="2026-01-01T00:00:00+00:00",
        last_used="2026-01-01T00:00:00+00:00",
        payload_bytes=3,
        tokens=1,
        tokens_sha256=token_digest([1]),
        segments=(SegmentProvenance(0, 1, "user", "key", "a" * 64, "test"),),
    )
    store.write_kv_atomic(save_dir / "session.kv", b"kv!")
    store.write_record_atomic(save_dir / "save.json", record)
    (save_dir / "events.jsonl").write_text(
        json.dumps({"kind": "run_start", "run_id": "run-1", "payload": {}})
        + "\n"
        + json.dumps({"kind": "run_end", "run_id": "run-1", "payload": {"status": "success"}})
        + "\n"
    )

    async def snapshot() -> None:
        steps = FileStepStore(save_dir / "steps")
        await steps.register_run(
            StepRunRecord(
                run_id=run_id,
                conversation_id=session_id,
                agent_name="ds4py",
            )
        )
        await steps.save_snapshot(
            ContinuableSnapshot(
                run_id=run_id,
                step_index=0,
                messages=[],
                conversation_id=session_id,
                agent_name="ds4py",
            )
        )

    asyncio.run(snapshot())
    return save_dir


def write_trace(path: Any, sampled: Any) -> None:
    path.write_text(
        json.dumps(
            {
                "session_id": "restore",
                "run_id": "followup",
                "step_index": 0,
                "round_index": 0,
                "prompt": [1, 2],
                "think": 0,
                "cached": 0,
                "prefill_suffix": [1, 2],
                "sampled": list(sampled),
            }
        )
        + "\n"
    )


def step_capture(*, delta: float = 0.0, reorder: bool = False, position: int = 2) -> StepCapture:
    entries = [TopLogit(index, 6.0 - index) for index in range(12)]
    if reorder:
        entries[0], entries[1] = entries[1], entries[0]
    entries[3] = replace(entries[3], logit=entries[3].logit + delta)
    return StepCapture(position, tuple(entries), entries[0].token)


def write_restore_metadata(
    trace_path: Path,
    *,
    case: int = 1,
    first_step: StepCapture | None = None,
    followup_position: int = 2,
    provenance_changes: dict[str, object] | None = None,
    save_record_hash: str = "save",
) -> Path:
    first_step = step_capture() if first_step is None else first_step
    provenance = {
        "bakeoff_session_id": "session-1",
        "instance_id": "instance-1",
        "gpu_model": "gpu",
        "gpu_driver_version": "driver",
        "ds4_revision": "ds4",
        "ds4py_revision": "python",
        "libds4_sha256": "library",
        "model_path": "model.gguf",
        "model_sha256": "model",
        "backend": "cpu",
        "ctx": 65536,
        "q8_f16_cache_mb": 5120,
        "temperature": 0.0,
        "seed": 1,
        "response_budget": 8192,
        "reset_policy": "restore-switch",
        "trace_sha256": f"trace-{case}",
        "source_trace_sha256": f"source-{case}",
        "rounds": [],
    }
    provenance.update(provenance_changes or {})
    metadata = trace_path.with_suffix(".json")
    value = {
        "followup_prefill": 2 if case == 1 else 12,
        "followup_position": followup_position,
        "first_step": {
            "position": first_step.position,
            "selected_token": first_step.selected_token,
            "top": [{"token": item.token, "logit": item.logit} for item in first_step.top],
        },
        "terminal_reason": "stop_token",
        "trace_path": str(trace_path),
        "provenance": provenance,
        "save_record_sha256": save_record_hash,
        "followup_prompt_sha256": hashlib.sha256(FOLLOWUP.encode()).hexdigest(),
    }
    if case != 4:
        value["switch_prefill"] = 0
        value["followup_plan"] = {1: "extend", 2: "rebuild", 3: "same-history"}[case]
    metadata.write_text(json.dumps(value))
    return metadata


def c_driver_stdout(command: list[str]) -> str:
    """Emit one valid C manifest round with hand-derived top-12 logits."""
    from bench.driver_cases import read_case_manifest

    manifest = Path(command[command.index("--cases") + 1])
    source_trace = command[command.index("--source-trace") + 1]
    case = read_case_manifest(manifest)[0]
    capture = step_capture(position=case.prompt_tokens)
    records = [
        {
            "kind": "round_start",
            "source_trace": source_trace,
            "task_id": case.task_id,
            "attempt": case.attempt,
            "round": case.round_index,
            "token_file": str(case.token_file),
            "prompt_tokens": case.prompt_tokens,
            "split_at": case.split_at,
            "steps": case.steps,
            "think": case.think,
            "observed_libds4_sha256": "library",
            "observed_model_path": command[command.index("--model") + 1],
            "observed_backend": command[command.index("--backend") + 1],
            "observed_ctx": int(command[command.index("--ctx") + 1]),
            "observed_seed": 1,
            "observed_temperature": 0.0,
            "observed_think": case.think,
        },
        {
            "kind": "sync",
            "label": "full",
            "input_tokens": case.prompt_tokens,
            "cached_tokens": 0,
            "prefill_tokens": case.prompt_tokens,
            "position": case.prompt_tokens,
            "elapsed_seconds": 0.0,
        },
        {
            "kind": "step",
            "index": 0,
            "position": capture.position,
            "selected_token": capture.selected_token,
            "top": [{"token": item.token, "logit": item.logit} for item in capture.top],
        },
        {
            "kind": "round_end",
            "prefill_tokens": case.prompt_tokens,
            "decode_steps": 1,
            "prefill_seconds": 0.0,
            "decode_seconds": 0.0,
            "terminal_reason": "stop_token",
        },
    ]
    return "".join(json.dumps(record) + "\n" for record in records)


def restore_run(case: int, trace_path: Path, **options: Any) -> RestoreRun | CReferenceRun:
    write_trace(trace_path, options.get("tokens", (1, 2, 3)))
    first_step = options.get("first_step", step_capture())
    metadata = write_restore_metadata(
        trace_path,
        case=case,
        first_step=first_step,
        followup_position=options.get("followup_position", 2),
        provenance_changes=options.get("provenance_changes"),
        save_record_hash=options.get("save_record_hash", "save"),
    )
    if case == 4:
        return CReferenceRun(
            trace_path=trace_path,
            followup_prefill=options.get("followup_prefill", 12),
            first_step=first_step,
            provenance_path=metadata,
        )
    return RestoreRun(
        trace_path=trace_path,
        terminal_reason=options.get("reason", "stop_token"),
        switch_prefill=options.get("switch_prefill", 0),
        followup_plan=options.get(
            "plan",
            {1: "extend", 2: "rebuild", 3: "same-history"}[case],
        ),
        followup_position=options.get("followup_position", 2),
        first_step=first_step,
        provenance_path=metadata,
        followup_prefill=options.get("followup_prefill", 2 if case == 1 else 12),
    )


class FakeRestoreOperations:
    def __init__(self, drive: Any) -> None:
        self.drive = drive
        self.restore_count = 0
        self.session_id = ""
        self.replay_restore_trace: Path | None = None
        self.c_case2_trace: Path | None = None

    def restore(self, home: Path, session_id: str, trace_path: Path) -> RestoreRun:
        self.restore_count += 1
        self.session_id = session_id
        return cast("RestoreRun", self.drive(home, session_id, self.restore_count, trace_path))

    def replay_same_history(
        self,
        home: Path,
        session_id: str,
        trace_path: Path,
        restore_trace: Path,
    ) -> RestoreRun:
        self.replay_restore_trace = restore_trace
        return cast("RestoreRun", self.drive(home, session_id, 3, trace_path))

    def c_rebuild(self, home: Path, case2_trace: Path, trace_path: Path) -> CReferenceRun:
        self.c_case2_trace = case2_trace
        return cast("CReferenceRun", self.drive(home, self.session_id, 4, trace_path))


def test_validate_save_accepts_complete_record_snapshot_events_and_kv(
    tmp_path: Path,
) -> None:
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


def test_stripped_copy_has_zero_payload_and_preserves_snapshot_and_record(
    tmp_path: Path,
) -> None:
    save = make_save(tmp_path)
    stripped = make_stripped_copy(save, tmp_path / "stripped-home")

    assert stripped != save
    assert SessionStore().read_kv_header(stripped / "session.kv") == 0
    assert (stripped / "save.json").read_bytes() == (save / "save.json").read_bytes()
    assert (stripped / "steps").is_dir()
    assert (stripped / "session.kv").stat().st_size == KV_HEADER.size


def test_restore_uses_same_history_and_c_rebuild_references(tmp_path: Path) -> None:
    """Different KV histories may differ; each restore path must match its own reference."""
    save = make_save(tmp_path)
    calls = []
    extended = step_capture()
    rebuilt = step_capture(delta=0.25)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        calls.append((home, session_id, case))
        return restore_run(
            case,
            trace_path,
            first_step=extended if case in (1, 3) else rebuilt,
        )

    operations = FakeRestoreOperations(drive)
    result = check_restore(save, tmp_path / "work", operations=operations)

    assert result.passed, result.detail
    assert result.valid
    assert result.position.passed
    assert result.restore_fidelity.passed
    assert result.rebuild_equivalence.passed
    assert [call[2] for call in calls] == [1, 2, 3, 4]
    assert all(call[1] == "session-1" for call in calls)
    assert operations.replay_restore_trace == tmp_path / "work/case-1.jsonl"
    assert operations.c_case2_trace == tmp_path / "work/case-2.jsonl"
    assert len(result.trace_paths) == 4
    stripped = cast("Any", result.stripped_save_dir)
    assert SessionStore().read_kv_header(stripped / "session.kv") == 0


@pytest.mark.parametrize("reference_case", [3, 4])
@pytest.mark.parametrize("bad_reference", [step_capture(reorder=True), step_capture(delta=1e-5)])
def test_restore_reference_ranking_and_tolerance_fail(
    tmp_path: Path,
    reference_case: int,
    bad_reference: StepCapture,
) -> None:
    """A reference mismatch must fail its own check, not the other restore path."""
    save = make_save(tmp_path)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        return restore_run(
            case,
            trace_path,
            first_step=bad_reference if case == reference_case else step_capture(),
        )

    result = check_restore(save, tmp_path / "work", operations=FakeRestoreOperations(drive))

    assert not result.passed
    assert result.valid
    assert result.restore_fidelity.passed == (reference_case != 3)
    assert result.rebuild_equivalence.passed == (reference_case != 4)


def test_restore_position_uses_fixed_followup_round_not_later_tool_round(tmp_path: Path) -> None:
    save = make_save(tmp_path)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        run = restore_run(case, trace_path)
        first = json.loads(trace_path.read_text())
        later = {
            **first,
            "step_index": 1,
            "round_index": 1,
            "prompt": [1, 2, 3],
            "cached": 2,
            "prefill_suffix": [3],
            "sampled": [],
        }
        with trace_path.open("a") as stream:
            stream.write(json.dumps(later) + "\n")
        return run

    result = check_restore(save, tmp_path / "work", operations=FakeRestoreOperations(drive))

    assert result.passed, result.detail
    assert result.valid


def test_restore_rejects_equal_length_prompts_with_different_tokens(tmp_path: Path) -> None:
    save = make_save(tmp_path)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        run = restore_run(case, trace_path)
        if case == 2:
            record = json.loads(trace_path.read_text())
            record["prompt"] = [8, 9]
            record["prefill_suffix"] = [8, 9]
            trace_path.write_text(json.dumps(record) + "\n")
        return run

    result = check_restore(save, tmp_path / "work", operations=FakeRestoreOperations(drive))

    assert not result.passed
    assert not result.valid
    assert "5.1" in result.detail
    assert "index 0" in result.detail


@pytest.mark.parametrize(
    "changes",
    [
        {"instance_id": "other-instance"},
        {"bakeoff_session_id": "other-session"},
        {"libds4_sha256": "other-library"},
        {"q8_f16_cache_mb": 4096},
    ],
)
def test_restore_provenance_mismatch_is_invalid(tmp_path: Path, changes: dict[str, object]) -> None:
    save = make_save(tmp_path)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        return restore_run(case, trace_path, provenance_changes=changes if case == 2 else None)

    result = check_restore(save, tmp_path / "work", operations=FakeRestoreOperations(drive))

    assert not result.passed
    assert not result.valid


def test_restore_requires_matching_save_and_fixed_followup_hashes(tmp_path: Path) -> None:
    save = make_save(tmp_path)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        return restore_run(case, trace_path, save_record_hash="save" if case == 1 else "other-save")

    result = check_restore(save, tmp_path / "work", operations=FakeRestoreOperations(drive))

    assert not result.passed
    assert not result.valid
    assert "save-record" in result.detail


@pytest.mark.parametrize(("case1_prefill", "case2_prefill"), [(1, 12), (0, 0)])
def test_restore_requires_no_case1_prefill_and_real_case2_prefill(
    tmp_path: Path,
    case1_prefill: Any,
    case2_prefill: Any,
) -> None:
    save = make_save(tmp_path)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        return restore_run(
            case,
            trace_path,
            switch_prefill=case1_prefill if case == 1 else 0,
            followup_prefill=2 if case == 1 else case2_prefill,
        )

    result = check_restore(save, tmp_path / "work", operations=FakeRestoreOperations(drive))
    assert not result.passed
    assert not result.valid


def test_restore_requires_extend_rebuild_and_first_step_position(tmp_path: Path) -> None:
    save = make_save(tmp_path)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        return restore_run(
            case,
            trace_path,
            plan="rebuild" if case == 1 else "extend",
            followup_position=3 if case == 2 else 2,
            first_step=step_capture(position=3) if case == 2 else step_capture(),
        )

    result = check_restore(save, tmp_path / "work", operations=FakeRestoreOperations(drive))

    assert not result.passed
    assert not result.valid
    assert "case 1" in result.detail


def test_restore_requires_first_step_evidence(tmp_path: Path) -> None:
    save = make_save(tmp_path)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        return restore_run(case, trace_path, first_step=None if case == 2 else step_capture())

    result = check_restore(save, tmp_path / "work", operations=FakeRestoreOperations(drive))

    assert not result.passed
    assert not result.valid
    assert "first-step" in result.detail


def test_different_later_tokens_and_terminal_reasons_are_diagnostic(tmp_path: Path) -> None:
    save = make_save(tmp_path)

    def drive(home: Any, session_id: Any, case: Any, trace_path: Any) -> Any:
        return restore_run(
            case,
            trace_path,
            tokens=(1, 2) if case == 1 else (9,),
            reason="stop_token" if case == 1 else "max_tokens",
        )

    result = check_restore(save, tmp_path / "work", operations=FakeRestoreOperations(drive))

    assert result.passed, result.detail
    assert "terminal" in result.detail
    assert len(result.trace_paths) == 4


@pytest.mark.parametrize("copy_workspace", [False, True], ids=["original", "copied"])
@pytest.mark.parametrize(("response_budget", "expected_terminal"), [(256, "stop_token"), (1, "max_tokens")])
def test_restore_driver_preserves_noninteractive_prompt_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    copy_workspace: bool,
    response_budget: int,
    expected_terminal: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "home")
    with restore_fake_engine(restore_script("saved answer")) as engine:
        model = DS4Model(engine.session(), sampling=SamplingPolicy(seed=1))
        runtime = AgentRuntime(
            model,
            project,
            store=store,
            interactive=False,
            read_only=False,
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
    restore_project = tmp_path / "restored-project"
    if copy_workspace:
        shutil.copytree(project, restore_project)
    else:
        restore_project = project
    monkeypatch.chdir(restore_project)
    monkeypatch.setenv("DS4PY_HOME", str(store.root))
    monkeypatch.setattr(ds4py, "open", lambda *args, **kwargs: replacement_engine)

    status = restore_command(
        [
            "--session-id",
            record.session_id,
            "--metadata",
            str(metadata),
            "--model",
            "f.gguf",
            "--backend",
            "cpu",
            "--ctx",
            "65536",
            "--temp",
            "0",
            "--seed",
            "1",
            "--tokens",
            str(response_budget),
            "--trace",
            str(trace),
        ]
    )

    assert status == 0
    value = json.loads(metadata.read_text())
    assert value["followup_plan"] == "extend"
    assert value["followup_position"] == value["first_step"]["position"] > 0
    assert value["terminal_reason"] == expected_terminal
    assert len(value["first_step"]["top"]) == 12
    assert "first_step" not in json.loads(trace.read_text())


def test_same_history_replays_saved_attempt_and_matches_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same-history reference must reproduce the saved attempt without loading its KV."""
    from bench.restore_check import same_history_command
    from ds4py.engine.replay import read_replay
    from ds4py.prefill_trace import PrefillTraceWriter

    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "home")
    original_trace = tmp_path / "trace.jsonl"
    with restore_fake_engine(restore_script("saved answer")) as engine:
        model = DS4Model(engine.session(), sampling=SamplingPolicy(seed=1))
        runtime = AgentRuntime(model, project, store=store, interactive=False, read_only=False)
        try:
            with PrefillTraceWriter(model.session, original_trace, think=model.session.engine.think):
                asyncio.run(runtime.run("question"))
                record = asyncio.run(runtime.save())
        finally:
            asyncio.run(runtime.close())

    monkeypatch.chdir(project)
    monkeypatch.setenv("DS4PY_HOME", str(store.root))
    patch_name = "docs/patches/cuda-no-copy-host-register-opt-out.patch"
    patch = NativePatch(
        patch_name,
        hashlib.sha256((Path(__file__).parents[2] / patch_name).read_bytes()).hexdigest(),
    )
    environment = {
        "BAKEOFF_SESSION_ID": "session-1",
        "DS4_BAKEOFF_INSTANCE_ID": "instance-1",
        "DS4_BAKEOFF_GPU_MODEL": "gpu",
        "DS4_BAKEOFF_GPU_DRIVER_VERSION": "driver",
        "DS4_BAKEOFF_DS4_REVISION": "ds4",
        "DS4_BAKEOFF_DS4PY_REVISION": "python",
        "DS4_BAKEOFF_LIBDS4_SHA256": "library",
        "DS4_BAKEOFF_MODEL_SHA256": "model",
        Q8_F16_CACHE_ENV: "5120",
        NATIVE_PATCHES_ENV: patch_name,
        CUDA_DISABLE_HOST_REGISTER_ENV: "1",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    session = BakeoffSession(
        bakeoff_session_id="session-1",
        instance_id="instance-1",
        gpu_model="gpu",
        gpu_driver_version="driver",
        ds4_revision="ds4",
        ds4py_revision="python",
        libds4_sha256="library",
        model_path=str(Path("f.gguf").resolve()),
        model_sha256="model",
        backend="cpu",
        ctx=65536,
        temperature=0.0,
        seed=1,
        response_budget=8192,
        q8_f16_cache_mb=5120,
        native_patches=(patch,),
        cuda_disable_host_register="1",
    )
    restore_trace = tmp_path / "case-1.jsonl"
    restore_metadata = tmp_path / "case-1.json"
    restore_engine = restore_fake_engine(restore_script("follow-up answer"))
    replay_engine = restore_fake_engine(restore_script("saved answer", "follow-up answer"))
    monkeypatch.setattr(ds4py, "open", lambda *args, **kwargs: restore_engine)
    assert (
        restore_command(
            [
                "--session-id",
                record.session_id,
                "--metadata",
                str(restore_metadata),
                "--model",
                "f.gguf",
                "--backend",
                "cpu",
                "--ctx",
                "65536",
                "--temp",
                "0",
                "--seed",
                "1",
                "--tokens",
                "8192",
                "--trace",
                str(restore_trace),
            ]
        )
        == 0
    )

    replay_trace = tmp_path / "case-3.jsonl"
    replay_metadata = tmp_path / "case-3.json"
    monkeypatch.setattr(ds4py, "open", lambda *args, **kwargs: replay_engine)
    assert (
        same_history_command(
            [
                "--session-id",
                record.session_id,
                "--metadata",
                str(replay_metadata),
                "--restore-trace",
                str(restore_trace),
                "--model",
                "f.gguf",
                "--backend",
                "cpu",
                "--ctx",
                "65536",
                "--temp",
                "0",
                "--seed",
                "1",
                "--tokens",
                "8192",
                "--trace",
                str(replay_trace),
            ]
        )
        == 0
    )

    restored = json.loads(restore_metadata.read_text())["first_step"]
    replayed = json.loads(replay_metadata.read_text())["first_step"]
    assert replayed == restored
    replay_rounds = read_replay(replay_trace).rounds
    restore_rounds = read_replay(restore_trace).rounds
    assert len(replay_rounds) == len(read_replay(original_trace).rounds) + 1
    assert replay_rounds[-1].prompt == restore_rounds[0].prompt
    assert replay_rounds[-1].cached == record.tokens
    assert replay_rounds[-1].sampled == ()
    for metadata_path, output_trace, source_trace, reset_policy in (
        (restore_metadata, restore_trace, restore_trace, "restore-switch"),
        (replay_metadata, replay_trace, original_trace, "same-history-replay"),
    ):
        expected = session.provenance(output_trace, source_trace, rounds=(), reset_policy=reset_policy)
        raw = json.loads(metadata_path.read_text())["provenance"]
        raw["native_patches"] = tuple(NativePatch(**item) for item in raw.get("native_patches", ()))
        actual = TraceProvenance(**raw)
        assert validate_session_pair(expected, actual) == ()


def test_default_noninteractive_attempt_uses_its_existing_single_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    path.write_text("""\
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
""")
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


def test_interruption_acceptance_uses_structured_event_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bench.phases as phase_runner

    task = tmp_path / "task"
    task.mkdir()
    (task / "TASK.md").write_text("Solve the task.")
    observed: dict[str, Any] = {}

    def drive(command: list[str], **kwargs: Any) -> CheckResult:
        observed["command"] = command
        observed.update(kwargs)
        return CheckResult(True, "interrupted and resumed")

    monkeypatch.setattr(phase_runner, "drive_interruption_pty", drive)
    config = SimpleNamespace(
        agent="ds4py",
        executable=(sys.executable, "driver.py"),
        model="model.gguf",
        backend="cpu",
        ctx=100,
        temperature=0.0,
        seed=1,
        tokens=32,
        request_limit=100,
    )

    result = interruption_acceptance(
        cast("Any", config),
        cast("Any", SimpleNamespace(agent_dir=task)),
        tmp_path / "results",
    )

    assert result.passed, result.detail
    home = tmp_path / "results" / "gate-4" / "interruption" / "home"
    assert observed["home"] == home
    assert observed["follow_up"] == "State the next concrete action in one sentence, then stop."
    assert observed["cwd"] == task
    assert observed["env"]["DS4PY_HOME"] == str(home)
    assert observed["timeout"] == 180
    assert "--non-interactive" not in observed["command"]


def test_restore_processes_use_disposable_workspace_copies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save = make_save(tmp_path)
    workspace = tmp_path / "task-workspace"
    workspace.mkdir()
    (workspace / "input.txt").write_text("original")
    working_directories: list[Path] = []
    copied_inputs: list[str] = []
    commands: list[list[str]] = []
    c_cases = []
    config = SimpleNamespace(
        executable=(sys.executable, "restore-driver.py", "attempt"),
        model="model.gguf",
        backend="cpu",
        ctx=65536,
        temperature=0.0,
        seed=1,
        tokens=8192,
        request_limit=100,
    )

    def complete(command: list[str], **kwargs: Any) -> Any:
        working_directory = cast("Path", kwargs["cwd"])
        working_directories.append(working_directory)
        copied_inputs.append((working_directory / "input.txt").read_text())
        (working_directory / "restore-write.txt").write_text("isolated")
        commands.append(command)
        if "--cases" in command:
            from bench.driver_cases import read_case_manifest

            c_cases.extend(read_case_manifest(Path(command[command.index("--cases") + 1])))
            return SimpleNamespace(returncode=0, stdout=c_driver_stdout(command), stderr="")
        trace = Path(command[command.index("--trace") + 1])
        case = int(trace.stem.rsplit("-", 1)[1])
        write_trace(trace, (1,))
        write_restore_metadata(trace, case=case)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", complete)
    item = SimpleNamespace(
        agent="ds4py",
        task_id="tiny",
        attempt_save_dirs=(save,),
        workspace=workspace,
    )

    results = check_result_saves(
        cast("Any", (item,)),
        cast("Any", config),
        tmp_path / "restores",
        provenance=RestoreProvenance(
            session=BakeoffSession(
                bakeoff_session_id="session-1",
                instance_id="instance-1",
                gpu_model="gpu",
                gpu_driver_version="driver",
                ds4_revision="ds4",
                ds4py_revision="python",
                libds4_sha256="library",
                model_path="model.gguf",
                model_sha256="model",
                backend="cpu",
                ctx=65536,
                temperature=0.0,
                seed=1,
                response_budget=8192,
                q8_f16_cache_mb=5120,
            ),
        ),
        c_driver=tmp_path / "c-driver",
    )

    assert results[0].passed, results[0].detail
    assert len(working_directories) == 4
    assert all(path != workspace for path in working_directories)
    assert len(set(working_directories)) == 4
    assert copied_inputs == ["original"] * 4
    assert not (workspace / "restore-write.txt").exists()
    assert all(not path.exists() for path in working_directories)
    assert len(commands) == 4
    for command in commands[:3]:
        assert command[command.index("--tokens") + 1] == "8192"
        assert command[command.index("--request-limit") + 1] == "100"
    assert "same-history" in commands[2]
    assert len(c_cases) == 1
    assert c_cases[0].split_at == 0
    assert c_cases[0].steps == 1
    assert c_cases[0].prompt_tokens == 2
