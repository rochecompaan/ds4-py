"""One-shot agent execution with clean stdout and unconditional saving."""

import asyncio
import sys


def _report(err, message) -> None:
    """Best-effort diagnostics must not prevent the unconditional save."""
    try:
        print(f"ds4py: {message}", file=err)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    except Exception:
        pass


async def run_noninteractive(runtime, prompt, *, out=None, err=None) -> int:
    """Run one prompt, print its final output, and attempt a save in all cases.

    The caller owns the runtime and engine and must close them after this
    function returns. Diagnostics never share the final answer's stream.
    """
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    code = 0
    try:
        try:
            result = await runtime.run(prompt)
        except (asyncio.CancelledError, KeyboardInterrupt):
            code = 130
            _report(err, "interrupted")
        except Exception as exc:
            code = 1
            _report(err, str(exc))
        else:
            try:
                print(result.output, file=out)
            except (asyncio.CancelledError, KeyboardInterrupt):
                code = 130
                _report(err, "output interrupted")
            except Exception as exc:
                code = 1
                _report(err, f"output failed: {exc}")
    finally:
        try:
            await runtime.save()
        except (asyncio.CancelledError, KeyboardInterrupt):
            if code == 0:
                code = 130
            _report(err, "save interrupted")
        except Exception as exc:
            code = 1
            _report(err, f"save failed: {exc}")
    return code
