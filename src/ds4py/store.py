"""Stdlib-only save records, KV files, and atomic file replacement.

The engine owns the KV payload bytes. This module owns the fixed file header,
the save record, prefix lookup, engine compatibility checks, and the
destination-directory temporary file protocol used to publish both files.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
import hashlib
import json
import operator
import os
from pathlib import Path
import struct
import tempfile

from ds4py.errors import StoreError

KV_HEADER = struct.Struct(">8sIQ")
KV_MAGIC = b"DS4PYKV\0"
KV_VERSION = 1
SAVE_FORMAT_VERSION = 1
_KV_FILE = "session.kv"
_SAVE_FILE = "save.json"
_MAX_UINT32 = 2**32 - 1


def token_digest(tokens: Iterable[int]) -> str:
    """SHA-256 over the fixed big-endian uint32 encoding of the token IDs."""
    values = []
    for token in tokens:
        if isinstance(token, bool):
            raise ValueError("token IDs must be integers")
        try:
            value = operator.index(token)
        except TypeError as error:
            raise ValueError("token IDs must be integers") from error
        if not 0 <= value <= _MAX_UINT32:
            raise ValueError("token IDs must fit an unsigned 32-bit integer")
        values.append(value)
    encoded = struct.pack(f">{len(values)}I", *values)
    return hashlib.sha256(encoded).hexdigest()


def _integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _nonnegative(value, name: str) -> int:
    value = _integer(value, name)
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _string(value, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


@dataclass(frozen=True)
class SegmentProvenance:
    """One token range and the message identity that produced it."""

    start: int
    end: int
    kind: str
    key: str
    content_hash: str
    source: str

    def __post_init__(self):
        start = _nonnegative(self.start, "segment start")
        end = _nonnegative(self.end, "segment end")
        if end <= start:
            raise ValueError("a segment must end after it starts")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        for name in ("kind", "key", "content_hash", "source"):
            object.__setattr__(self, name, _string(getattr(self, name), f"segment {name}"))

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "key": self.key,
            "content_hash": self.content_hash,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SegmentProvenance":
        if not isinstance(data, Mapping):
            raise ValueError("a segment must be an object")
        try:
            return cls(
                data["start"], data["end"], data["kind"], data["key"],
                data["content_hash"], data["source"],
            )
        except KeyError as error:
            raise ValueError(f"segment is missing {error.args[0]!r}") from error


@dataclass(frozen=True)
class SaveRecord:
    """The KV file's model identity and token provenance for one snapshot."""

    session_id: str
    conversation_id: str
    run_id: str
    step_index: int
    model_id: int
    quant_bits: int
    ctx: int
    title: str
    created_at: str
    last_used: str
    payload_bytes: int
    tokens: int
    tokens_sha256: str
    segments: tuple[SegmentProvenance, ...]
    version: int = SAVE_FORMAT_VERSION

    def __post_init__(self):
        object.__setattr__(self, "version", _integer(self.version, "version"))
        if self.version != SAVE_FORMAT_VERSION:
            raise ValueError(f"unsupported save record version {self.version}")
        for name in ("session_id", "conversation_id", "run_id", "title",
                     "created_at", "last_used", "tokens_sha256"):
            object.__setattr__(self, name, _string(getattr(self, name), name))
        for name in ("step_index", "model_id", "quant_bits", "ctx", "payload_bytes", "tokens"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        digest = self.tokens_sha256
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise ValueError("tokens_sha256 must be a 64-character SHA-256 digest")
        segments = tuple(self.segments)
        if not all(isinstance(segment, SegmentProvenance) for segment in segments):
            raise ValueError("segments must contain SegmentProvenance values")
        object.__setattr__(self, "segments", segments)
        self._check_coverage()

    def _check_coverage(self) -> None:
        if self.tokens == 0:
            if self.segments:
                raise ValueError(
                    "segments must be empty for a zero-token transcript"
                )
            return
        if not self.segments:
            raise ValueError(
                f"segments do not cover {self.tokens} tokens: the list is empty"
            )
        expected = 0
        for segment in self.segments:
            if segment.start != expected:
                raise ValueError(
                    "segments overlap or have a gap: expected start token "
                    f"{expected}, got start token {segment.start}"
                )
            if segment.end > self.tokens:
                raise ValueError(
                    f"segments extend beyond {self.tokens} tokens: end token "
                    f"{segment.end}"
                )
            expected = segment.end
        if expected != self.tokens:
            raise ValueError(
                f"segments have a gap: they end at token {expected}, "
                f"not {self.tokens}"
            )

    def to_dict(self) -> dict[str, object]:
        self._check_coverage()
        return {
            "version": self.version,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "run_id": self.run_id,
            "step_index": self.step_index,
            "model_id": self.model_id,
            "quant_bits": self.quant_bits,
            "ctx": self.ctx,
            "title": self.title,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "payload_bytes": self.payload_bytes,
            "tokens": self.tokens,
            "tokens_sha256": self.tokens_sha256,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SaveRecord":
        if not isinstance(data, Mapping):
            raise ValueError("a save record must be an object")
        names = (
            "version", "session_id", "conversation_id", "run_id", "step_index",
            "model_id", "quant_bits", "ctx", "title", "created_at", "last_used",
            "payload_bytes", "tokens", "tokens_sha256", "segments",
        )
        missing = [name for name in names if name not in data]
        if missing:
            raise ValueError(f"save record is missing {', '.join(missing)}")
        raw_segments = data["segments"]
        if not isinstance(raw_segments, (list, tuple)):
            raise ValueError("segments must be a list")
        return cls(
            **{name: data[name] for name in names if name != "segments"},
            segments=tuple(SegmentProvenance.from_dict(item) for item in raw_segments),
        )


def _default_root() -> Path:
    configured = os.environ.get("DS4PY_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path("~/.ds4py").expanduser()


def _atomic_write(path: Path, chunks: Sequence[bytes]) -> None:
    """Publish ``chunks`` through a destination-directory temp file and rename."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise StoreError(f"{path}: cannot create directory: {error}") from error
    temporary = None
    replaced = False
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as file:
            temporary = Path(file.name)
            for chunk in chunks:
                file.write(chunk)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        replaced = True
    except OSError as error:
        raise StoreError(f"{path}: cannot write atomically: {error}") from error
    finally:
        if temporary is not None and not replaced:
            with suppress(OSError):
                temporary.unlink()


class SessionStore:
    """Session directories under a resolved root, with atomic file primitives.

    An explicit root is fixed. Without one, ``DS4PY_HOME`` or ``~/.ds4py`` is
    read on every access so an environment change applies to the next call.
    Resolution never creates directories or moves files.
    """

    def __init__(self, root: str | os.PathLike | None = None):
        self._root = None if root is None else Path(root).expanduser()

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else _default_root()

    def session_dir(self, session_id: str) -> Path:
        return self.root / "sessions" / session_id

    def write_kv_atomic(self, path: str | os.PathLike, payload: bytes) -> None:
        """Write the fixed header and exactly ``len(payload)`` payload bytes."""
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        payload = bytes(payload)
        header = KV_HEADER.pack(KV_MAGIC, KV_VERSION, len(payload))
        _atomic_write(Path(path), (header, payload))

    def read_kv(self, path: str | os.PathLike) -> bytes:
        """Read one payload, rejecting a bad header, every truncation, and extras."""
        path = Path(path)
        try:
            with path.open("rb") as file:
                raw = file.read(KV_HEADER.size)
                if len(raw) != KV_HEADER.size:
                    raise StoreError(
                        f"{path}: truncated header: expected {KV_HEADER.size} "
                        f"bytes, got {len(raw)}"
                    )
                magic, version, payload_bytes = KV_HEADER.unpack(raw)
                if magic != KV_MAGIC:
                    raise StoreError(f"{path}: bad KV magic {magic!r}")
                if version != KV_VERSION:
                    raise StoreError(
                        f"{path}: unsupported KV version {version}, "
                        f"expected {KV_VERSION}"
                    )
                remaining = max(0, os.fstat(file.fileno()).st_size - KV_HEADER.size)
                if payload_bytes > remaining:
                    raise StoreError(
                        f"{path}: truncated payload: expected {payload_bytes} "
                        f"bytes, got {remaining}"
                    )
                payload = file.read(payload_bytes)
                if len(payload) != payload_bytes:
                    raise StoreError(
                        f"{path}: truncated payload: expected {payload_bytes} "
                        f"bytes, got {len(payload)}"
                    )
                if file.read(1):
                    raise StoreError(f"{path}: extra undeclared payload data")
                return payload
        except OSError as error:
            raise StoreError(f"{path}: cannot read KV file: {error}") from error

    def write_record_atomic(self, path: str | os.PathLike, record: SaveRecord) -> None:
        """Serialize and atomically publish one save record."""
        path = Path(path)
        try:
            encoded = json.dumps(
                record.to_dict(), indent=2, ensure_ascii=False,
            ).encode("utf-8") + b"\n"
        except ValueError as error:
            raise StoreError(f"{path}: invalid save record: {error}") from error
        _atomic_write(path, (encoded,))

    def read_record(self, path: str | os.PathLike) -> SaveRecord:
        """Parse one save record, wrapping every malformation with its path."""
        path = Path(path)
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise StoreError(f"{path}: cannot read save record: {error}") from error
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, ValueError, RecursionError) as error:
            raise StoreError(f"{path}: malformed save record: {error}") from error
        if not isinstance(data, Mapping):
            raise StoreError(f"{path}: malformed save record: expected an object")
        try:
            return SaveRecord.from_dict(data)
        except (KeyError, TypeError, ValueError) as error:
            raise StoreError(f"{path}: malformed save record: {error}") from error

    def find(self, prefix: str) -> tuple[SaveRecord, Path]:
        """Resolve a unique save by session-id prefix and return its record path."""
        sessions = self.root / "sessions"
        names = sorted(
            entry.name for entry in sessions.iterdir()
            if entry.is_dir() and entry.name.startswith(prefix)
            and (entry / _SAVE_FILE).is_file()
        ) if sessions.is_dir() else []
        if not names:
            raise StoreError(f"unknown session prefix {prefix!r}")
        if len(names) > 1:
            raise StoreError(
                f"ambiguous session prefix {prefix!r}: {', '.join(names)}"
            )
        path = sessions / names[0] / _SAVE_FILE
        return self.read_record(path), path

    def validate_for_engine(self, record: SaveRecord, info) -> None:
        """Refuse a record saved by another model identity or quantization."""
        if record.model_id != info.model_id:
            raise StoreError(
                f"session {record.session_id}: model id {record.model_id} does "
                f"not match the engine model id {info.model_id}"
            )
        if record.quant_bits != info.routed_quant_bits:
            raise StoreError(
                f"session {record.session_id}: quant bits {record.quant_bits} "
                f"do not match the engine routed quant bits "
                f"{info.routed_quant_bits}"
            )
