"""Raw token streaming and ds4-agent-compatible binary traces."""
import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import re
import signal
import sys
import threading
from typing import BinaryIO, Callable, TextIO

from ds4py import Engine, open as open_engine
from ds4py.engine.events import Interrupted, Stopped, Token
from ds4py.engine.sampling import SamplingPolicy
from ds4py.errors import EngineError


def _escape(data: bytes) -> bytes:
    escapes = {92: b"\\\\", 10: b"\\n", 13: b"\\r", 9: b"\\t", 34: b'\\"'}
    return b"".join(escapes.get(b, f"\\x{b:02x}".encode() if b < 32 or b == 127
                                else bytes([b])) for b in data)


class Trace:
    def __init__(self, path: Path, *, now: Callable[[], datetime] = datetime.now):
        self.path, self.now = path, now

    def __enter__(self) -> "Trace":
        self._file = self.path.open("ab")
        return self

    def __exit__(self, *exc) -> None:
        self._file.close()

    def _line(self, message: bytes) -> None:
        stamp = self.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._file.write(stamp.encode() + b" " + message + b"\n")
        self._file.flush()

    def line(self, message: str) -> None:
        self._line(message.encode("utf-8"))

    def text(self, label: str, data: bytes) -> None:
        self._line(label.encode() + b'="' + _escape(data) + b'"')

    def token(self, index: int, token: int, text: bytes) -> None:
        self._line(f"token index={index} id={token} bytes={len(text)} text=\"".encode()
                   + _escape(text) + b'" hex=' + text.hex().encode())


def read_trace_tokens(path: Path) -> tuple[list[int], list[int]]:
    tokens: list[int] | None = None
    prompt_length = 0
    for line in path.read_bytes().splitlines():
        start = re.fullmatch(
            rb"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} "
            rb"tokens label=raw_prompt_tokens start=0 len=(\d+)", line,
        )
        if start:
            prompt_length = int(start[1])
            tokens = []
        elif tokens is not None:
            token = re.search(rb" token index=\d+ id=(-?\d+) ", line)
            if token:
                tokens.append(int(token[1]))
    if tokens is None:
        raise ValueError("trace contains no raw run")
    return tokens[:prompt_length], tokens[prompt_length:]


def run_raw(engine: Engine, prompt: str, *, max_tokens: int,
            sampling: SamplingPolicy, out: BinaryIO, err: TextIO,
            trace: Trace | None = None) -> int:
    tokens = engine.tokenize(prompt)
    if not tokens:
        raise EngineError("raw prompt tokenized to no tokens")
    if trace is not None:
        trace.text("raw-prompt", prompt.encode("utf-8"))
        trace.line(f"tokens label=raw_prompt_tokens start=0 len={len(tokens)}")
        for index, token in enumerate(tokens):
            trace.token(index, token, engine.token_text(token))
    session = engine.session()
    session.append_tokens(tokens, "user")
    if not session.plan().fits:
        err.write(f"ds4py: prompt of {len(tokens)} tokens does not fit a context of {engine.config.ctx}\n")
        return 2
    main_thread = threading.current_thread() is threading.main_thread()
    if main_thread:
        previous = signal.getsignal(signal.SIGINT)

        def interrupt(signum, frame):
            session.cancel()
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        signal.signal(signal.SIGINT, interrupt)
    try:
        index = 0
        for event in session.run_round(max_tokens=max_tokens, sampling=sampling):
            if isinstance(event, Token):
                out.write(event.text_bytes)
                out.flush()
                index += 1
                if trace is not None:
                    trace.token(index, event.id, event.text_bytes)
            elif isinstance(event, Interrupted):
                return 130
            elif isinstance(event, Stopped):
                return 130 if event.reason == "interrupted" else 0
        return 0
    finally:
        if main_thread:
            signal.signal(signal.SIGINT, previous)


def command(args: argparse.Namespace) -> int:
    prompt = args.prompt if args.prompt is not None else args.prompt_text
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_bytes().decode("utf-8")
    with open_engine(args.model, ctx=args.ctx, backend=args.backend, seed=args.seed,
                     threads=args.threads, lib=args.lib) as engine:
        sampling = SamplingPolicy.for_engine(engine.info, temperature=args.temp,
                                             top_p=args.top_p, min_p=args.min_p,
                                             seed=args.seed)
        with Trace(args.trace) if args.trace is not None else nullcontext() as trace:
            return run_raw(engine, prompt, max_tokens=args.tokens, sampling=sampling,
                           out=sys.stdout.buffer, err=sys.stderr, trace=trace)
