import json
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.pty import _EventFollower, drive_interruption_pty
from bench.restore_check import CheckResult

FOLLOW_UP = "State the next concrete action in one sentence, then stop."


def write_script(root: Path, source: str) -> Path:
    path = root / "child.py"
    path.write_text(textwrap.dedent(source))
    return path


def test_output_only_tui_text_does_not_satisfy_interruption(tmp_path: Path) -> None:
    """A footer-scraping implementation must fail without a structured Token."""
    received = tmp_path / "received"
    script = write_script(
        tmp_path,
        f"""
        import sys
        import time
        from pathlib import Path

        print("streaming", flush=True)
        print("idle", flush=True)
        line = sys.stdin.readline()
        if line:
            Path({str(received)!r}).write_text(line)
        time.sleep(5)
        """,
    )

    result = drive_interruption_pty(
        [sys.executable, str(script)],
        home=tmp_path / "home",
        follow_up=FOLLOW_UP,
        timeout=0.4,
    )

    assert not result.passed
    assert "initial Token" in result.detail
    assert not received.exists()


def test_fragmented_events_drive_one_interrupt_and_ordered_follow_up(tmp_path: Path) -> None:
    """Wrong-run events, partial JSON, or duplicate Ctrl+C must fail this exchange."""
    home = tmp_path / "home"
    record = tmp_path / "record.json"
    script = write_script(
        tmp_path,
        """
        import atexit
        import json
        import select
        import signal
        import sys
        import time
        from pathlib import Path

        home = Path(sys.argv[1])
        record = Path(sys.argv[2])
        events = home / "sessions" / "test-session" / "events.jsonl"
        events.parent.mkdir(parents=True)
        interrupts = []
        inputs = []

        def save_record():
            record.write_text(json.dumps({"interrupts": len(interrupts), "inputs": inputs}))

        atexit.register(save_record)
        signal.signal(signal.SIGINT, lambda *_: interrupts.append(1))

        def emit(run_id, kind, payload, pieces=1):
            data = (json.dumps({
                "ts": "2026-11-02T00:00:00+00:00",
                "session_id": "test-session",
                "run_id": run_id,
                "kind": kind,
                "payload": payload,
            }) + "\\n").encode()
            start = 0
            with events.open("ab", buffering=0) as stream:
                for index in range(pieces):
                    end = round(len(data) * (index + 1) / pieces)
                    stream.write(data[start:end])
                    start = end
                    time.sleep(0.01)

        emit("active", "run_start", {})
        emit("unrelated", "Token", {"event": {"token": 1, "text": "x"}})
        time.sleep(0.15)
        if interrupts:
            raise SystemExit("Ctrl+C followed an unrelated Token")

        emit("active", "Token", {"event": {"token": 2, "text": "a"}}, pieces=4)
        deadline = time.monotonic() + 2
        while not interrupts and time.monotonic() < deadline:
            time.sleep(0.01)
        if len(interrupts) != 1:
            raise SystemExit("the fragmented Token did not trigger exactly one Ctrl+C")
        emit("active", "Token", {"event": {"token": 3, "text": "b"}})
        time.sleep(0.05)
        if len(interrupts) != 1:
            raise SystemExit(f"expected one Ctrl+C, got {len(interrupts)}")

        emit("unrelated", "Stopped", {"event": {"reason": "interrupted"}})
        emit("unrelated", "run_end", {"status": "success"})
        ready, _, _ = select.select([sys.stdin], [], [], 0.15)
        if ready:
            raise SystemExit("follow-up followed an unrelated run")

        emit("active", "Stopped", {"event": {"reason": "interrupted"}})
        emit("active", "run_end", {"status": "success"})
        inputs.append(sys.stdin.readline().strip())

        emit(None, "run_start", {})
        emit("follow-up", "run_start", {})
        emit("follow-up", "run_end", {"status": "success"})
        inputs.append(sys.stdin.readline().strip())
        time.sleep(0.05)
        """,
    )

    result = drive_interruption_pty(
        [sys.executable, str(script), str(home), str(record)],
        home=home,
        follow_up=FOLLOW_UP,
        timeout=5,
    )

    assert result.passed, result.detail
    assert json.loads(record.read_text()) == {
        "interrupts": 1,
        "inputs": [FOLLOW_UP, "/exit"],
    }


def test_early_exit_reports_the_expected_initial_token(tmp_path: Path) -> None:
    """A dead child must produce a prompt diagnostic instead of an event-wait hang."""
    script = write_script(tmp_path, "raise SystemExit(0)\n")

    result = drive_interruption_pty(
        [sys.executable, str(script)],
        home=tmp_path / "home",
        follow_up=FOLLOW_UP,
        timeout=1,
    )

    assert not result.passed
    assert "initial Token" in result.detail
    assert len(result.detail) <= 8100


def run_barrier_scenario(tmp_path: Path, mode: str) -> tuple[CheckResult, Path]:
    home = tmp_path / "home"
    record = tmp_path / "record.json"
    script = write_script(
        tmp_path,
        """
        import json
        import signal
        import sys
        import time
        from pathlib import Path

        home = Path(sys.argv[1])
        record = Path(sys.argv[2])
        mode = sys.argv[3]
        events = home / "sessions" / "test-session" / "events.jsonl"
        events.parent.mkdir(parents=True)
        interrupts = []
        inputs = []

        def save_record():
            record.write_text(json.dumps({"inputs": inputs}))

        save_record()
        signal.signal(signal.SIGINT, lambda *_: interrupts.append(1))

        def event(run_id, kind, payload):
            return json.dumps({
                "ts": "2026-11-02T00:00:00+00:00",
                "session_id": "test-session",
                "run_id": run_id,
                "kind": kind,
                "payload": payload,
            }) + "\\n"

        def emit_batch(items):
            data = "".join(event(*item) for item in items).encode()
            with events.open("ab", buffering=0) as stream:
                stream.write(data)

        emit_batch([("active", "run_start", {})])
        if mode == "pre_interrupt":
            emit_batch([
                ("active", "Token", {"event": {"token": 2, "text": "a"}}),
                ("active", "Stopped", {"event": {"reason": "interrupted"}}),
                ("active", "run_end", {"status": "success"}),
            ])
        else:
            emit_batch([("active", "Token", {"event": {"token": 2, "text": "a"}})])

        deadline = time.monotonic() + 2
        while not interrupts and time.monotonic() < deadline:
            time.sleep(0.01)
        if not interrupts:
            raise SystemExit("driver did not send Ctrl+C")

        if mode == "pre_follow_up":
            emit_batch([
                ("active", "Stopped", {"event": {"reason": "interrupted"}}),
                ("active", "run_end", {"status": "success"}),
                ("stale", "run_start", {}),
                ("stale", "run_end", {"status": "success"}),
            ])
        elif mode == "ordinary_success":
            emit_batch([
                ("active", "Stopped", {"event": {"reason": "stop_token"}}),
                ("active", "run_end", {"status": "success"}),
            ])

        for _ in range(2):
            line = sys.stdin.readline()
            if not line:
                break
            inputs.append(line.strip())
            save_record()
        """,
    )
    result = drive_interruption_pty(
        [sys.executable, str(script), str(home), str(record), mode],
        home=home,
        follow_up=FOLLOW_UP,
        timeout=1,
    )
    return result, record


@pytest.mark.parametrize(
    ("mode", "awaited"),
    [
        ("pre_interrupt", "interrupted Stopped"),
        ("pre_follow_up", "follow-up run_start"),
    ],
)
def test_pre_input_events_cannot_satisfy_lifecycle(
    tmp_path: Path,
    mode: str,
    awaited: str,
) -> None:
    """Events buffered before a control write must not satisfy the state that write enters."""
    result, record = run_barrier_scenario(tmp_path, mode)

    assert not result.passed
    assert awaited in result.detail
    expected_inputs = [] if mode == "pre_interrupt" else [FOLLOW_UP]
    assert json.loads(record.read_text())["inputs"] == expected_inputs


def test_uninterrupted_success_is_rejected(tmp_path: Path) -> None:
    """Ordinary generation success after Ctrl+C must not masquerade as an interruption."""
    result, record = run_barrier_scenario(tmp_path, "ordinary_success")

    assert not result.passed
    assert "interrupted Stopped" in result.detail
    assert json.loads(record.read_text())["inputs"] == []


def test_follower_completes_one_fragment_without_skipping_the_next(tmp_path: Path) -> None:
    """Splitting a line must defer exactly that event, then preserve the following event."""
    home = tmp_path / "home"
    events = home / "sessions" / "test-session" / "events.jsonl"
    events.parent.mkdir(parents=True)
    first = {
        "ts": "2026-11-02T00:00:00+00:00",
        "session_id": "test-session",
        "run_id": "active",
        "kind": "Token",
        "payload": {"event": {"token": 2, "text": "a"}},
    }
    second = {**first, "kind": "Stopped", "payload": {"event": {"reason": "interrupted"}}}
    data = (json.dumps(first) + "\n").encode()
    follower = _EventFollower(home)

    with events.open("ab", buffering=0) as stream:
        stream.write(data[: len(data) // 2])
    assert follower.read() == []
    with events.open("ab", buffering=0) as stream:
        stream.write(data[len(data) // 2 :])
    assert follower.read() == [first]
    with events.open("ab", buffering=0) as stream:
        stream.write((json.dumps(second) + "\n").encode())
    assert follower.read() == [second]
