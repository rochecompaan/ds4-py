"""UTF-16 code-unit operations shared by paging and recovery."""


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def utf16_offset(text: str, index: int) -> int:
    if not 0 <= index <= len(text):
        raise ValueError("Python string index is out of range")
    return utf16_length(text[:index])


def utf16_slice(text: str, start: int, end: int | None = None) -> str:
    """Slice whole characters inside UTF-16 bounds without splitting pairs."""
    limit = utf16_length(text) if end is None else end
    if start < 0 or limit < 0:
        raise ValueError("UTF-16 offsets must be non-negative")
    if start >= limit:
        return ""
    parts: list[str] = []
    position = 0
    for character in text:
        width = utf16_length(character)
        if position + width > limit:
            break
        if position >= start:
            parts.append(character)
        position += width
    return "".join(parts)


def clip_utf16(text: str, limit: int) -> tuple[str, int]:
    clipped = utf16_slice(text, 0, limit)
    return clipped, utf16_length(clipped)


__all__ = ["clip_utf16", "utf16_length", "utf16_offset", "utf16_slice"]
