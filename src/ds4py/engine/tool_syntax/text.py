"""Escape only tool-body text that collides with its closing delimiter."""


def _escaped_close_at(text: str, position: int, closing_tag: str) -> bool:
    if text[position] != "&":
        return False
    cursor = position + 1
    while text.startswith("amp;", cursor):
        cursor += 4
    return text.startswith("lt;" + closing_tag[1:], cursor)


def _validate_closing_tag(closing_tag: str) -> None:
    if not closing_tag:
        raise ValueError("closing_tag must not be empty")


def escape_tool_text(text: str, closing_tag: str) -> str:
    _validate_closing_tag(closing_tag)
    output = []
    copied_from = 0
    for position, character in enumerate(text):
        if text.startswith(closing_tag, position) or _escaped_close_at(text, position, closing_tag):
            output.append(text[copied_from:position])
            output.append("&lt;" if character == "<" else "&amp;")
            copied_from = position + 1
    output.append(text[copied_from:])
    return "".join(output)


def unescape_tool_text(text: str, closing_tag: str) -> str:
    _validate_closing_tag(closing_tag)
    output = []
    position = 0
    while position < len(text):
        if _escaped_close_at(text, position, closing_tag):
            if text.startswith("&amp;", position):
                output.append("&")
                position += 5
            else:
                output.append("<")
                position += 4
        else:
            output.append(text[position])
            position += 1
    return "".join(output)
