"""Startup Ctrl+C must stop the owner thread before propagating to the caller."""
from typing import Any
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("startup_fails", [False, True])
def test_startup_sigint_closes_resources_on_owner_before_reraising(startup_fails: Any) -> None:
    probe = textwrap.dedent(r'''
        import os
        import signal
        import sys
        import threading
        from concurrent.futures import Future

        import ds4py
        from ds4py._native.fake import FakeNative
        from ds4py.engine import worker

        startup_fails = sys.argv[1] == "True"
        waiting, initialized = threading.Event(), threading.Event()
        interrupted, release = threading.Event(), threading.Event()
        main = threading.current_thread()
        handles, owners, closed, errors = [], [], [], []
        original = KeyboardInterrupt("startup interrupted")

        class Ready(Future):
            def result(self, *args, **kwargs):
                if threading.current_thread() is main:
                    waiting.set()
                return super().result(*args, **kwargs)

        worker.Future = Ready

        class TrackingNative(FakeNative):
            def open_engine(self, config):
                engine = super().open_engine(config)
                handles.append(engine)
                owners.append(threading.current_thread())
                create, info, engine_close = engine.create_session, engine.info, engine.close

                def create_session(ctx):
                    session = create(ctx)
                    handles.append(session)
                    session_close = session.close
                    def close_session():
                        closed.append(("session", threading.current_thread()))
                        session_close()
                    session.close = close_session
                    return session

                def blocked_info():
                    initialized.set()
                    assert release.wait(3), "native initialization release timed out"
                    if startup_fails:
                        raise ds4py.EngineError("initialization failed after interrupt")
                    return info()

                def close_engine():
                    closed.append(("engine", threading.current_thread()))
                    engine_close()

                engine.create_session, engine.info, engine.close = create_session, blocked_info, close_engine
                return engine

        def handler(signum, frame):
            interrupted.set()
            raise original

        signal.signal(signal.SIGINT, handler)

        def interrupt_startup():
            try:
                assert initialized.wait(3), "native resources were not initialized"
                assert waiting.wait(3), "constructor never waited for readiness"
                os.kill(os.getpid(), signal.SIGINT)
                assert interrupted.wait(3), "SIGINT was not handled"
            except BaseException as error:
                errors.append(repr(error))
            finally:
                release.set()

        helper = threading.Thread(target=interrupt_startup)
        helper.start()
        try:
            try:
                ds4py.open("f.gguf", ctx=64, native=TrackingNative())
            except KeyboardInterrupt as error:
                assert error is original, "startup cleanup replaced the caller interrupt"
            else:
                raise AssertionError("startup swallowed SIGINT")
            assert len(owners) == 1 and owners[0] is not main
            assert not owners[0].is_alive(), "constructor returned before owner shutdown"
            assert closed == [("session", owners[0]), ("engine", owners[0])], closed
            assert len(handles) == 2 and all(handle.raw == 0 for handle in handles)
        finally:
            release.set()
            helper.join(3)
        assert not helper.is_alive() and not errors, errors
        print("startup SIGINT cleanup OK", startup_fails)
    ''')
    result = subprocess.run(
        [sys.executable, "-c", probe, str(startup_fails)],
        capture_output=True, text=True, timeout=8,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"startup SIGINT cleanup OK {startup_fails}"
