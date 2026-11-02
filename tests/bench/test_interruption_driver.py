import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from bench.pty import drive_interruption_pty  # noqa: I001 - The repository path precedes bench imports.


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
        emit("active", "Token", {"event": {"token": 3, "text": "b"}})
        deadline = time.monotonic() + 2
        while not interrupts and time.monotonic() < deadline:
            time.sleep(0.01)
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
