"""Bounded real-SIGINT regression, isolated from pytest's signal handling."""
from typing import Any
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("phase", ["sync", "generation"])
@pytest.mark.parametrize("signals", [1, 2])
def test_caller_sigint_drains_before_releasing_round(phase: Any, signals: Any) -> None:
    probe = textwrap.dedent(r'''
        import os
        import signal
        import sys
        import threading
        from queue import SimpleQueue

        import ds4py
        from ds4py._native.fake import FakeNative, FakeSession, reply
        from ds4py.engine import worker
        from ds4py.engine.events import Stopped, Token

        phase, signals = sys.argv[1], int(sys.argv[2])
        main = threading.current_thread()
        blocked, release = threading.Event(), threading.Event()
        waiting, draining = threading.Event(), threading.Event()
        cancelling, second_handled = threading.Event(), threading.Event()
        errors, signal_count, waits = [], 0, 0

        class Cancel(threading.Event):
            def set(self):
                super().set()
                cancelling.set()

        class ObservedQueue(SimpleQueue):
            def get(self, *args, **kwargs):
                global waits
                if threading.current_thread() is main and blocked.is_set() and self.empty():
                    waits += 1
                    (waiting if waits == 1 else draining).set()
                return super().get(*args, **kwargs)

        worker.SimpleQueue = ObservedQueue
        cancel = Cancel()
        original_wait = threading.Event.wait

        def observe_wait(self, *args, **kwargs):
            if threading.current_thread() is main and blocked.is_set() and cancel.is_set():
                draining.set()
            return original_wait(self, *args, **kwargs)

        threading.Event.wait = observe_wait
        original_sync, original_eval = FakeSession.sync, FakeSession.eval

        def sync(self, tokens, *, progress=None, cancel=None):
            if phase == "sync":
                blocked.set()
                progress("prefill_chunk", 0, len(tokens))
                assert release.wait(5), "sync release timed out"
            return original_sync(self, tokens, progress=progress, cancel=cancel)

        def eval(self, token):
            if phase == "generation" and token == 99:
                blocked.set()
                assert release.wait(5), "eval release timed out"
            return original_eval(self, token)

        FakeSession.sync, FakeSession.eval = sync, eval

        def handler(signum, frame):
            global signal_count
            signal_count += 1
            if signal_count == 2:
                second_handled.set()
            signal.default_int_handler(signum, frame)

        signal.signal(signal.SIGINT, handler)
        with ds4py.open("f.gguf", ctx=64, native=FakeNative(script=reply(b"abc")), cancel=cancel) as e:
            session = e.session()
            session.append_tokens([5, 6], "user")
            rnd = session.run_round()
            if phase == "generation":
                assert next(ev for ev in rnd if isinstance(ev, Token)) == Token(98, b"a")

            def interrupt_and_release():
                try:
                    assert waiting.wait(3), "caller never waited for a worker event"
                    os.kill(os.getpid(), signal.SIGINT)
                    assert cancelling.wait(3), "interruption did not cancel the job"
                    assert cancel.is_set(), "cancellation cleared before native completion"
                    try:
                        session.append_tokens([7], "user")
                    except ds4py.EngineError as error:
                        assert str(error) == "a round is open"
                    else:
                        raise AssertionError("append accepted before native completion")
                    try:
                        next(session.run_round())
                    except ds4py.EngineError as error:
                        assert str(error) == "a round is open"
                    else:
                        raise AssertionError("overlap accepted before native completion")
                    if signals == 2:
                        assert draining.wait(3), "cleanup never waited for completion"
                        os.kill(os.getpid(), signal.SIGINT)
                        assert second_handled.wait(3), "second SIGINT was not handled"
                except BaseException as error:
                    errors.append(repr(error))
                finally:
                    release.set()

            helper = threading.Thread(target=interrupt_and_release)
            helper.start()
            try:
                with_interruption = False
                try:
                    list(rnd)
                except KeyboardInterrupt:
                    with_interruption = True
                assert with_interruption, "SIGINT was swallowed"
            finally:
                release.set()
                helper.join(5)
                rnd.close()
            assert not helper.is_alive()
            assert not errors, errors
            assert signal_count == signals
            expected = (5, 6) if phase == "sync" else (5, 6, 98, 99)
            assert session.transcript.tokens == expected, session.transcript.tokens
            assert session.ctx_used == (0 if phase == "sync" else 4)
            assert not cancel.is_set()
            session.append_tokens([7], "user")
            assert list(session.run_round(max_tokens=0))[-1] == Stopped("max_tokens")
        print("SIGINT drain OK", phase, signals)
    ''')
    result = subprocess.run(
        [sys.executable, "-c", probe, phase, str(signals)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"SIGINT drain OK {phase} {signals}"


@pytest.mark.parametrize("phase", ["sync", "generation"])
def test_interrupt_after_end_dequeue_releases_round(phase: Any) -> None:
    probe = textwrap.dedent(r'''
        import sys
        import ds4py
        from ds4py._native.fake import FakeNative, reply
        from ds4py.engine import worker
        from ds4py.engine.events import Stopped

        phase = sys.argv[1]
        original_submit = worker.EngineWorker.submit
        target = None
        injected = False

        def submit(self, job):
            global target
            events = original_submit(self, job)
            wanted = worker.SyncJob if phase == "sync" else worker.GenerateJob
            if isinstance(job, wanted) and target is None:
                target = events
                # Fence the job and end-marker publication before consuming it.
                self.call(lambda e, s: None).result()
            return events

        worker.EngineWorker.submit = submit

        def interrupt_after_dequeue(frame, event, arg):
            global injected
            if (event == "line" and frame.f_code is worker._JobEvents.__next__.__code__
                    and frame.f_locals.get("self") is target
                    and frame.f_locals.get("event") is worker._END and not injected):
                injected = True
                assert target.finished.is_set() and target.queue.empty()
                assert not target._exhausted
                raise KeyboardInterrupt("after end dequeue")
            return interrupt_after_dequeue

        with ds4py.open("f.gguf", ctx=64, native=FakeNative(script=reply(b"ok"))) as e:
            session = e.session()
            session.append_tokens([5, 6], "user")
            sys.settrace(interrupt_after_dequeue)
            try:
                try:
                    list(session.run_round())
                except KeyboardInterrupt as error:
                    assert str(error) == "after end dequeue"
                else:
                    raise AssertionError("caller interrupt was swallowed")
            finally:
                sys.settrace(None)
            assert injected
            expected = (5, 6) if phase == "sync" else (5, 6, 112, 108)
            assert session.transcript.tokens == expected
            assert session.ctx_used == len(expected)
            assert not e._worker.cancel_event.is_set()
            session.append_tokens([7], "user")
            assert list(session.run_round(max_tokens=0))[-1] == Stopped("max_tokens")
        print("lost END recovered", phase)
    ''')
    result = subprocess.run(
        [sys.executable, "-c", probe, phase], capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"lost END recovered {phase}"


def test_lost_end_with_completion_after_receive_wait_still_terminates() -> None:
    probe = textwrap.dedent(r'''
        import threading
        from queue import SimpleQueue
        from ds4py.engine.worker import _END, _JobEvents

        entered_wait = threading.Event()

        class InterruptingQueue(SimpleQueue):
            def get(self, *args, **kwargs):
                if self.empty():
                    entered_wait.set()
                value = super().get(*args, **kwargs)
                if value is _END:
                    raise KeyboardInterrupt("after dequeue, before completion")
                return value

        channel = _JobEvents()
        channel.queue = InterruptingQueue()
        channel.queue.put(_END)
        try:
            next(channel)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("interruption was not injected")
        assert channel.queue.empty() and not channel.finished.is_set()

        def finish():
            assert entered_wait.wait(3), "receiver never attempted another read"
            channel.finished.set()

        finisher = threading.Thread(target=finish)
        finisher.start()
        try:
            assert list(channel) == []
        finally:
            finisher.join(3)
        assert not finisher.is_alive()
        print("late completion recovered")
    ''')
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "late completion recovered"
