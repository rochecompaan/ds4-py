"""Canonical native syntax for assistant calls reconstructed from history."""
import json
from collections.abc import Mapping

from .qwen import value_is_json
from .text import escape_tool_text


def _arguments(args) -> dict[str, object]:
    if args is None:
        return {}
    if isinstance(args, str):
        value = json.loads(args)
    else:
        value = args
    if not isinstance(value, Mapping):
        raise ValueError("tool call arguments must be a mapping")
    return dict(value)


def _validate_json(value):
    if value is None or type(value) in (str, bool, int, float):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"JSON object keys must be strings, got {type(key).__name__}")
            _validate_json(item)
        return
    raise ValueError(f"cannot preserve non-JSON value of type {type(value).__name__}")


def _json(value) -> str:
    _validate_json(value)
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).replace("<", "\\u003c")


def _validate_name(family, value, kind):
    invalid = not isinstance(value, str) or not value
    if family in ("deepseek4", "deepseek41"):
        invalid = invalid or (isinstance(value, str) and any(c in value for c in '">'))
    elif family == "glm":
        invalid = invalid or (
            isinstance(value, str)
            and (value != value.strip(" \t\r\n") or (kind == "tool name" and "<" in value))
        )
    elif family == "qwen":
        invalid = invalid or (
            isinstance(value, str) and (value != value.strip(" \t\r\n") or ">" in value)
        )
    if invalid:
        raise ValueError(f"{family} cannot preserve {kind} {value!r}")


def _dsml(name, args, *, v41):
    calls = " calls" if v41 else "tool_calls"
    invoke = " invoke" if v41 else "invoke"
    parameter = " parameter" if v41 else "parameter"
    closing = f"</｜DSML｜{parameter}>"
    lines = [f"<｜DSML｜{calls}>", f'<｜DSML｜{invoke} name="{name}">']
    for key, value in args.items():
        if isinstance(value, str):
            rendered = escape_tool_text(value, closing)
            lines.append(
                f'<｜DSML｜{parameter} name="{key}" string="true">{rendered}{closing}'
            )
        else:
            lines.append(
                f'<｜DSML｜{parameter} name="{key}" string="false">{_json(value)}{closing}'
            )
    lines.extend([f"</｜DSML｜{invoke}>", f"</｜DSML｜{calls}>"])
    return "\n".join(lines)


def _glm(name, args):
    spans = [f"<tool_call>{name}"]
    for key, value in args.items():
        if not isinstance(value, str):
            raise ValueError(f"GLM cannot preserve non-string argument {key!r}")
        key_text = escape_tool_text(str(key), "</arg_key>")
        spans.append(
            f"<arg_key>{key_text}</arg_key><arg_value>"
            f"{escape_tool_text(value, '</arg_value>')}</arg_value>"
        )
    spans.append("</tool_call>")
    return "".join(spans)


def _qwen(name, args):
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in args.items():
        value_text = value if isinstance(value, str) else _json(value)
        if isinstance(value, str):
            if value_is_json(value.encode()):
                raise ValueError(f"Qwen cannot preserve JSON-looking string argument {key!r}")
            value_text = escape_tool_text(value_text, "</parameter>")
        elif not value_is_json(value_text.encode()):
            raise ValueError(f"Qwen cannot preserve non-JSON argument {key!r}")
        lines.extend([
            f"<parameter={key}>", value_text, "</parameter>",
        ])
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def render_tool_call(family, name, args) -> str:
    """Render one call in the model family's existing native syntax."""
    arguments = _arguments(args)
    if family not in ("deepseek4", "deepseek41", "glm", "qwen"):
        raise ValueError(f"unknown tool syntax family: {family}")
    _validate_name(family, name, "tool name")
    for key in arguments:
        _validate_name(family, key, "argument key")
    if family == "deepseek4":
        return _dsml(name, arguments, v41=False)
    if family == "deepseek41":
        return _dsml(name, arguments, v41=True)
    if family == "glm":
        return _glm(name, arguments)
    if family == "qwen":
        return _qwen(name, arguments)
    raise ValueError(f"unknown tool syntax family: {family}")
