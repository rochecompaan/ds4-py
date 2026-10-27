"""Deterministic, thread-confined native handles for model-free tests."""
import struct
import threading
from typing import Callable, Iterable, Mapping, Sequence
import zlib

from ds4py.errors import EngineError, MissingBinding
from . import fake_chat
from .bindings import all_bindings
from .protocol import EngineConfig, EngineInfo, SyncStatus, ThinkMode

FAKE_EOS = 0
PAYLOAD_MAGIC = b"DFKP"
PAYLOAD_VERSION = 1
PAYLOAD_STRIDE = 1024
PAYLOAD_HEADER = struct.Struct("<4sIII")
Script = Callable[[Sequence[int], Sequence[int]], int]


def encode_payload(tokens: Sequence[int]) -> bytes:
    """Encode a versioned token payload padded to the fake's declared byte size."""
    tokens = list(tokens)
    if not tokens:
        return b""
    raw = b"".join(struct.pack("<i", token) for token in tokens)
    size = PAYLOAD_STRIDE * len(tokens)
    encoded = PAYLOAD_HEADER.pack(
        PAYLOAD_MAGIC, PAYLOAD_VERSION, len(tokens), zlib.crc32(raw),
    ) + raw
    return encoded + bytes(size - len(encoded))


def decode_payload(data: bytes, payload_bytes: int) -> list[int]:
    """Validate and decode a fake payload; never accept a corrupt token list."""
    if payload_bytes == 0:
        return []
    if len(data) != payload_bytes:
        raise EngineError(
            f"truncated payload: declared {payload_bytes} bytes, read {len(data)}"
        )
    if payload_bytes % PAYLOAD_STRIDE:
        raise EngineError(f"corrupt payload: invalid size {payload_bytes}")
    if payload_bytes < PAYLOAD_HEADER.size:
        raise EngineError(f"corrupt payload: header is truncated at {payload_bytes} bytes")
    magic, version, count, checksum = PAYLOAD_HEADER.unpack_from(data)
    if magic != PAYLOAD_MAGIC:
        raise EngineError("corrupt payload: bad magic")
    if version != PAYLOAD_VERSION:
        raise EngineError(f"corrupt payload: unsupported version {version}")
    if count != payload_bytes // PAYLOAD_STRIDE:
        raise EngineError(f"corrupt payload: token count {count} does not match its size")
    end = PAYLOAD_HEADER.size + count * 4
    raw = data[PAYLOAD_HEADER.size:end]
    if data[end:] != bytes(payload_bytes - end):
        raise EngineError("corrupt payload: nonzero padding")
    if zlib.crc32(raw) != checksum:
        raise EngineError("corrupt payload: token checksum mismatch")
    return list(struct.unpack(f"<{count}i", raw))


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
            routed_quant_bits=0, mtp_draft_tokens=1, vocab_size=fake_chat.VOCAB_SIZE,
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
        if rendered:
            return fake_chat.tokenize_rendered(text)
        return [byte + 1 for byte in text.encode()]

    def token_text(self, token):
        self._check()
        if token == FAKE_EOS:
            return b"</s>"
        special = fake_chat.special_token_text(token)
        return special if special is not None else bytes([token - 1])

    def chat_begin(self):
        self._check()
        return fake_chat.chat_begin(self)

    def think_prefix(self, prefix, mode):
        self._check()
        return fake_chat.think_prefix(self, prefix, mode)

    def render_message(self, prefix, role, content):
        self._check()
        return fake_chat.render_message(self, prefix, role, content)

    def assistant_prefix(self, prefix, mode):
        self._check()
        return fake_chat.assistant_prefix(self, prefix, mode)

    def qwen_reasoning_effort(self, mode):
        self._check()
        return fake_chat.qwen_reasoning_effort(mode)

    def eos(self):
        self._check()
        return FAKE_EOS

    def is_stop(self, token):
        self._check()
        return token == FAKE_EOS

    def is_stop_for_think_mode(self, token, mode):
        self._check()
        enabled = mode in (1, 2, 3, 4) or 1001 <= mode <= 1100
        return self.is_stop(token) or (
            not enabled and token in (
                fake_chat.SPECIAL_TOKENS["<think>"], fake_chat.SPECIAL_TOKENS["</think>"],
            )
        )

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


class FakeStagedPayload:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.bytes = PAYLOAD_STRIDE * len(self.tokens)
        self._data = encode_payload(self.tokens)
        self.closed = False

    def write(self, file):
        if self.closed:
            raise EngineError("staged payload is closed")
        file.write(self._data)
        file.flush()

    def close(self):
        self.closed = True


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
        self._logits: list[float] | None = None
        self.logit_copies = self.logit_sets = 0
        self._prompt_length = 0
        self._valid = False
        self.calls: list[tuple] = []
        self.samples: list[tuple] = []

    def sync(self, tokens, *, progress=None, cancel=None):
        self._check()
        self._logits = None
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

    def _distribution(self):
        self._check()
        if not self._valid:
            return None
        if self._logits is None:
            token = self._native.script(self._tokens[:self._prompt_length],
                                        self._tokens[self._prompt_length:])
            distribution = [0.0] * self._engine.info().vocab_size
            if 0 <= token < len(distribution):
                distribution[token] = 1.0
            self._logits = distribution
        return self._logits

    def copy_logits(self, out):
        self._check()
        size = self._engine.info().vocab_size
        if len(out) != size:
            raise ValueError(f"logits buffer must equal the vocabulary size ({size})")
        distribution = self._distribution()
        if distribution is None:
            raise EngineError("session checkpoint is invalid; sync first")
        for index, value in enumerate(distribution):
            out[index] = value
        self.logit_copies += 1

    def set_logits(self, values):
        self._check()
        size = self._engine.info().vocab_size
        if len(values) != size:
            raise ValueError(f"logits buffer must equal the vocabulary size ({size})")
        if not self._valid:
            raise EngineError("session checkpoint is invalid; sync first")
        self._logits = [float(value) for value in values]
        self.logit_sets += 1

    def sample(self, temperature, top_k, top_p, min_p, rng):
        self._check()
        if not self._valid:
            return -1
        self.samples.append((temperature, top_k, top_p, min_p))
        # SplitMix64: deterministic state evolution, independent of the script.
        mask = (1 << 64) - 1
        value = (rng.value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        rng.value = value ^ (value >> 31)
        distribution = self._distribution()
        if distribution is None:
            return -1
        return max(range(len(distribution)), key=distribution.__getitem__)

    def argmax(self):
        distribution = self._distribution()
        if distribution is None:
            return -1
        return max(range(len(distribution)), key=distribution.__getitem__)

    def eval(self, token):
        self._check()
        if not self._valid:
            raise EngineError("session checkpoint is invalid")
        if len(self._tokens) >= self._ctx:
            raise EngineError("context is full")
        self._tokens.append(token)
        self._logits = None
        self.calls.append(("eval", token))

    def rewind(self, pos):
        self._check()
        if pos < len(self._tokens):
            del self._tokens[max(0, pos):]
            self._valid = False
            self.calls.append(("rewind", pos))
        self._logits = None

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
        self._logits = None

    def payload_bytes(self):
        self._check()
        return PAYLOAD_STRIDE * len(self._tokens)

    def stage_payload(self):
        self._check()
        if not self._valid:
            raise EngineError("session checkpoint is invalid")
        return FakeStagedPayload(self._tokens)

    def load_payload(self, file, payload_bytes):
        self._check()
        data = file.read(payload_bytes)
        tokens = decode_payload(data, payload_bytes)
        self._tokens = tokens
        self._prompt_length = len(tokens)
        self._valid = True
        self._logits = None
        self.calls.append(("load_payload", len(tokens)))

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
        self._logits = None

    def close(self):
        self._assert_thread()
        self.raw = 0
        self._engine._sessions.discard(self)
