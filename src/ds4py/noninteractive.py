"""One-shot agent execution with clean stdout and unconditional saving."""

import asyncio
import sys
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from ds4py.runtime import AgentRuntime


def _report(err: TextIO, message: str) -> None:
    """Best-effort diagnostics must not prevent the unconditional save."""
    try:
        print(f"ds4py: {message}", file=err)
    except asyncio.CancelledError, KeyboardInterrupt:
        pass
    except Exception:
        return


async def run_noninteractive(
    runtime: AgentRuntime,
    prompt: str,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Run one prompt, print its final output, and attempt a save in all cases.

    The caller owns the runtime and engine and must close them after this
    function returns. Diagnostics never share the final answer's stream.
    """
    output = sys.stdout if out is None else out
    errors = sys.stderr if err is None else err
    code = 0
    try:
        try:
            result = await runtime.run(prompt)
        except asyncio.CancelledError, KeyboardInterrupt:
            code = 130
            _report(errors, "interrupted")
        except Exception as exc:
            code = 1
            _report(errors, str(exc))
        else:
            try:
                print(result.output, file=output)
            except asyncio.CancelledError, KeyboardInterrupt:
                code = 130
                _report(errors, "output interrupted")
            except Exception as exc:
                code = 1
                _report(errors, f"output failed: {exc}")
    finally:
        try:
            await runtime.save()
        except asyncio.CancelledError, KeyboardInterrupt:
            if code == 0:
                code = 130
            _report(errors, "save interrupted")
        except Exception as exc:
            code = 1
            _report(errors, f"save failed: {exc}")
    return code
