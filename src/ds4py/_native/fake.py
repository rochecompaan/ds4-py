"""Deterministic, thread-confined native handles for model-free tests."""
import threading
from typing import Callable, Iterable, Mapping, Sequence

from ds4py.errors import EngineError, MissingBinding
from .bindings import all_bindings
from .protocol import EngineConfig, EngineInfo, SyncStatus, ThinkMode

FAKE_EOS = 0
Script = Callable[[Sequence[int], Sequence[int]], int]


def reply(text: bytes) -> Script:
    def next_token(prompt, generated):
        return text[len(generated)] + 1 if len(generated) < len(text) else FAKE_EOS
    return next_token


class FakeClock:
    def __init__(self):
        self._time = 0.0

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds):
        self._time += seconds


class FakeNative:
    backend = "cpu"

    def __init__(self, *, script: Script = reply(b"ok"), chunk: int = 4,
                 clock: FakeClock | None = None, chunk_seconds: float = 0.0,
                 bindings: Mapping[str, Callable] = {}, absent: Iterable[str] = (),
                 restore_enabled: bool = True, info: EngineInfo | None = None):
        if chunk <= 0:
            raise ValueError("chunk must be positive")
        self.script, self.chunk = script, chunk
        self.clock = clock if clock is not None else FakeClock()
        self.chunk_seconds = chunk_seconds
        self.bindings, self.absent = dict(bindings), set(absent)
        self.restore_enabled = restore_enabled
        self._info = info if info is not None else EngineInfo(
            name="fake", model_id=0, family="deepseek4", is_deepseek41=False,
            is_glm_dsa=False, is_qwen4=False, has_vision=False,
            routed_quant_bits=0, mtp_draft_tokens=1, vocab_size=257,
        )

    def open_engine(self, config: EngineConfig) -> "FakeEngine":
        return FakeEngine(self)

    def bind(self, name):
        if name not in {binding.name for binding in all_bindings()}:
            raise MissingBinding(f"ds4py does not declare {name}")
        if name in self.absent:
            raise MissingBinding(f"{name} is absent from this library")
        return self.bindings[name]

    def parse_think_level(self, text):
        level = 0
        if not text:
            return None
        for char in text:
            if not "0" <= char <= "9" or level > 100:
                return None
            level = level * 10 + ord(char) - ord("0")
        return 1000 + level if level <= 100 else None

    def think_mode_for_context(self, mode, ctx):
        return ThinkMode.HIGH if mode == ThinkMode.MAX and ctx < 393216 else mode


class _Handle:
    def __init__(self):
        self.raw = id(self)
        self._thread = threading.get_ident()

    def _assert_thread(self):
        assert threading.get_ident() == self._thread, "fake handle used on another thread"

    def _check(self):
        self._assert_thread()
        if not self.raw:
            raise EngineError("fake handle is closed")


class FakeEngine(_Handle):
    def __init__(self, native):
        super().__init__()
        self._native = native
        self._sessions = set()

    def info(self):
        self._check()
        return self._native._info

    def tokenize(self, text, *, rendered=False):
        self._check()
        return [byte + 1 for byte in text.encode()]

    def token_text(self, token):
        self._check()
        return b"</s>" if token == FAKE_EOS else bytes([token - 1])

    def eos(self):
        self._check()
        return FAKE_EOS

    def is_stop(self, token):
        self._check()
        return token == FAKE_EOS

    def create_session(self, ctx):
        self._check()
        session = FakeSession(self, ctx)
        self._sessions.add(session)
        return session

    def close(self):
        self._assert_thread()
        for session in list(self._sessions):
            session.close()
        self.raw = 0


class FakeSnapshot:
    def __init__(self, tokens, prompt_length):
        self.tokens = list(tokens)
        self.prompt_length = prompt_length
        self.nbytes = 1024 * len(tokens)
        self.closed = False

    def close(self):
        self.closed = True


class FakeSession(_Handle):
    def __init__(self, engine, ctx):
        super().__init__()
        self._engine, self._native, self._ctx = engine, engine._native, ctx
        self._tokens = []
        self._prompt_length = 0
        self._valid = False
        self.calls: list[tuple] = []

    def sync(self, tokens, *, progress=None, cancel=None):
        self._check()
        if not tokens or len(tokens) >= self._ctx:
            raise EngineError("prompt must be nonempty and shorter than context")
        target = list(tokens)
        start = len(self._tokens)
        if not self._valid or target[:start] != self._tokens:
            self._tokens = []
            start = 0
        self._valid = True
        status = SyncStatus.DONE
        if progress:
            progress("prefill_chunk", start, len(target))
        while len(self._tokens) < len(target):
            if cancel and cancel():
                status = SyncStatus.INTERRUPTED
                break
            end = min(len(self._tokens) + self._native.chunk, len(target))
            self._tokens.extend(target[len(self._tokens):end])
            self._native.clock.advance(self._native.chunk_seconds)
            if progress:
                progress("prefill_chunk", end, len(target))
        self._prompt_length = len(self._tokens)
        self.calls.append(("sync", len(self._tokens) - start))
        return status

    def sample(self, temperature, top_k, top_p, min_p, rng):
        self._check()
        if not self._valid:
            return -1
        # SplitMix64: deterministic state evolution, independent of the script.
        mask = (1 << 64) - 1
        value = (rng.value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        rng.value = value ^ (value >> 31)
        return self.argmax()

    def argmax(self):
        self._check()
        if not self._valid:
            return -1
        return self._native.script(self._tokens[:self._prompt_length],
                                   self._tokens[self._prompt_length:])

    def eval(self, token):
        self._check()
        if not self._valid:
            raise EngineError("session checkpoint is invalid")
        if len(self._tokens) >= self._ctx:
            raise EngineError("context is full")
        self._tokens.append(token)
        self.calls.append(("eval", token))

    def rewind(self, pos):
        self._check()
        if pos < len(self._tokens):
            del self._tokens[max(0, pos):]
            self._valid = False
            self.calls.append(("rewind", pos))

    def pos(self):
        self._check()
        return len(self._tokens)

    def ctx(self):
        self._check()
        return self._ctx

    def tokens(self):
        self._check()
        return list(self._tokens)

    def common_prefix(self, tokens):
        self._check()
        if not self._valid:
            return 0
        count = 0
        for live, requested in zip(self._tokens, tokens):
            if live != requested:
                break
            count += 1
        return count

    def checkpoint_valid(self):
        self._check()
        return self._valid

    def invalidate(self):
        self._check()
        self._valid = False

    def payload_bytes(self):
        self._check()
        return 1024 * len(self._tokens)

    def save_snapshot(self):
        self._check()
        if not self._valid:
            raise EngineError("session checkpoint is invalid")
        return FakeSnapshot(self._tokens, self._prompt_length)

    def load_snapshot(self, snap):
        self._check()
        if snap.closed:
            raise EngineError("snapshot is closed")
        self._tokens = list(snap.tokens)
        self._prompt_length = snap.prompt_length
        self._valid = True

    def close(self):
        self._assert_thread()
        self.raw = 0
        self._engine._sessions.discard(self)
