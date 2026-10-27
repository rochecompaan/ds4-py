"""Parse ds4-agent-compatible assistant prefix files."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PrefixTurn:
    """One plain-text historical user or assistant turn."""
    role: Literal["user", "assistant"]
    content: str


class PrefixError(ValueError):
    """Invalid prefix-file syntax, encoding, or file access."""
    pass


def _marker_at(data: bytes, position: int):
    if data.startswith(b"USER:", position):
        return "user", 5
    if data.startswith(b"ASSISTANT:", position):
        return "assistant", 10
    return None


def _decode_content(data: bytes, start: int, end: int, role: str, line: int, turn_start: int) -> str:
    try:
        return data[start:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        invalid_line = line + data.count(b"\n", turn_start, start + exc.start)
        raise PrefixError(f"line {invalid_line}: invalid UTF-8 in {role.upper()} turn") from exc


def parse_prefix(data: bytes) -> tuple[PrefixTurn, ...]:
    """Parse UTF-8 USER:/ASSISTANT: pairs; require a final assistant turn."""
    if not data:
        raise PrefixError("prefix file is empty")
    if b"\0" in data:
        raise PrefixError("prefix file contains a NUL byte")

    position = 3 if data.startswith(b"\xef\xbb\xbf") else 0
    line = 1
    turns = []
    while position < len(data):
        turn_start = position
        marker = _marker_at(data, position)
        if marker is None:
            raise PrefixError(f"line {line} must start with USER: or ASSISTANT:")
        role, marker_length = marker
        expected = "user" if len(turns) % 2 == 0 else "assistant"
        if role != expected:
            raise PrefixError(f"line {line}: expected {expected.upper()}:")

        content_start = position + marker_length
        if content_start < len(data) and data[content_start] in (32, 9):
            content_start += 1

        next_position = len(data)
        next_line = line
        scan_line = line
        for index in range(content_start, len(data)):
            if data[index] != 10:
                continue
            scan_line += 1
            if index + 1 < len(data) and _marker_at(data, index + 1) is not None:
                next_position = index + 1
                next_line = scan_line
                break

        content_end = next_position
        if content_end > content_start and data[content_end - 1] == 10:
            content_end -= 1
            if content_end > content_start and data[content_end - 1] == 13:
                content_end -= 1
        if content_end == content_start:
            raise PrefixError(f"line {line} has an empty {role.upper()} turn")

        content = _decode_content(data, content_start, content_end, role, line, turn_start)
        turns.append(PrefixTurn(role, content))
        position = next_position
        line = next_line

    if not turns:
        raise PrefixError("prefix file contains no turns")
    if turns[-1].role != "assistant":
        raise PrefixError("prefix file must end with an ASSISTANT turn")
    return tuple(turns)


def load_prefix(path: str | Path) -> tuple[PrefixTurn, ...]:
    """Read and validate a prefix file, reporting path-aware PrefixError failures."""
    if not str(path):
        raise PrefixError("invalid prefix file path")
    file_path = Path(path)
    try:
        stream = file_path.open("rb")
    except OSError as exc:
        raise PrefixError(f"cannot open prefix file {file_path}: {exc}") from exc
    try:
        with stream:
            data = stream.read()
    except OSError as exc:
        raise PrefixError(f"cannot read prefix file {file_path}: {exc}") from exc
    try:
        return parse_prefix(data)
    except PrefixError as exc:
        raise PrefixError(f"{file_path}: {exc}") from exc
