"""Deterministic fake-vocabulary mirror of ds4's native chat rendering."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .protocol import ThinkMode

if TYPE_CHECKING:
    from .fake import FakeEngine


_SPECIAL_TEXTS = (
    "<｜begin▁of▁sentence｜>",
    "<｜System｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<sop>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
    "<|im_start|>",
    "<|im_end|>",
    "｜DSML｜",
)
SPECIAL_TOKENS = {text: 257 + index for index, text in enumerate(_SPECIAL_TEXTS)}
_SPECIAL_TEXT_BY_ID = {token: text.encode() for text, token in SPECIAL_TOKENS.items()}
_RENDERED_TOKENS = sorted(
    ((text.encode(), token) for text, token in SPECIAL_TOKENS.items()),
    key=lambda item: len(item[0]),
    reverse=True,
)
VOCAB_SIZE = max(SPECIAL_TOKENS.values()) + 1

_QWEN_XHIGH = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
_QWEN_LOW = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration."
)
_DEEPSEEK_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
)


def tokenize_rendered(text: str) -> list[int]:
    data = text.encode()
    tokens = []
    position = 0
    while position < len(data):
        for literal, token in _RENDERED_TOKENS:
            if data.startswith(literal, position):
                tokens.append(token)
                position += len(literal)
                break
        else:
            tokens.append(data[position] + 1)
            position += 1
    return tokens


def special_token_text(token: int) -> bytes | None:
    return _SPECIAL_TEXT_BY_ID.get(token)


def think_enabled(mode: int) -> bool:
    value = int(mode)
    return value in (ThinkMode.LOW, ThinkMode.MEDIUM, ThinkMode.HIGH, ThinkMode.MAX) or 1001 <= value <= 1100


def chat_begin(engine: FakeEngine) -> list[int]:
    info = engine.info()
    if info.is_qwen4:
        return []
    if info.is_glm_dsa:
        # The fake GLM vocabulary uses <sop> as both its fallback BOS and SOP.
        return engine.tokenize("<sop><sop>", rendered=True)
    return engine.tokenize("<｜begin▁of▁sentence｜>", rendered=True)


def think_prefix(engine: FakeEngine, prefix: Sequence[int], mode: int) -> list[int]:
    info = engine.info()
    if info.is_glm_dsa:
        effort = None
        if mode in (ThinkMode.LOW, ThinkMode.MEDIUM, ThinkMode.HIGH):
            effort = "Reasoning Effort: High"
        elif mode == ThinkMode.MAX:
            effort = "Reasoning Effort: Max"
        return engine.tokenize("<|system|>", rendered=True) + engine.tokenize(effort) if effort else []
    if info.is_deepseek41:
        value = int(mode)
        level = value - 1000 if 1000 <= value <= 1100 else -1
        if mode == ThinkMode.HIGH:
            level = 75
        elif mode == ThinkMode.MAX:
            level = 100
        if level <= 0:
            return []
        effort = f"Reasoning Effort: {level} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        return engine.tokenize("<｜System｜>", rendered=True) + engine.tokenize(effort)
    return engine.tokenize(_DEEPSEEK_MAX) if mode == ThinkMode.MAX else []


def _escaped_close_at(text: str, position: int, closing: str) -> bool:
    if text[position] != "&":
        return False
    position += 1
    while text.startswith("amp;", position):
        position += 4
    return text.startswith("lt;", position) and text.startswith(closing[1:], position + 3)


def _escape_closing(text: str, closing: str) -> str:
    output = []
    start = position = 0
    while position < len(text):
        if text.startswith(closing, position) or _escaped_close_at(text, position, closing):
            output.append(text[start:position])
            output.append("&lt;" if text[position] == "<" else "&amp;")
            position += 1
            start = position
        else:
            position += 1
    output.append(text[start:])
    return "".join(output)


def _qwen_open(engine: FakeEngine, role: str) -> list[int]:
    return engine.tokenize("<|im_start|>", rendered=True) + engine.tokenize(role + "\n")


def _qwen_close(engine: FakeEngine) -> list[int]:
    return engine.tokenize("<|im_end|>", rendered=True) + engine.tokenize("\n")


def _qwen_message(engine: FakeEngine, role: str, content: str) -> list[int]:
    if role in ("tool", "function"):
        return (
            _qwen_open(engine, "user")
            + engine.tokenize("<tool_response>\n")
            + engine.tokenize(_escape_closing(content, "</tool_response>"))
            + engine.tokenize("\n</tool_response>")
            + _qwen_close(engine)
        )
    name = "system" if role in ("system", "developer") else "assistant" if role == "assistant" else "user"
    body = engine.tokenize(content, rendered=True) if name == "assistant" else engine.tokenize(content)
    return _qwen_open(engine, name) + body + _qwen_close(engine)


def _glm_message(engine: FakeEngine, role: str, content: str) -> list[int]:
    if role in ("system", "developer"):
        return engine.tokenize("<|system|>", rendered=True) + engine.tokenize(content, rendered=True)
    if role == "assistant":
        opening = engine.tokenize("<|assistant|>", rendered=True)
        if not content.startswith(("<think>", "</think>")):
            opening += engine.tokenize("<think></think>", rendered=True)
        return opening + engine.tokenize(content, rendered=True)
    if role in ("tool", "function"):
        return (
            engine.tokenize("<|observation|><tool_response>", rendered=True)
            + engine.tokenize(_escape_closing(content, "</tool_response>"))
            + engine.tokenize("</tool_response>", rendered=True)
        )
    return engine.tokenize("<|user|>", rendered=True) + engine.tokenize(content)


def _deepseek41_system_marker(prefix: Sequence[int]) -> list[int]:
    roles = {text: SPECIAL_TOKENS[text] for text in ("<｜System｜>", "<｜User｜>", "<｜Assistant｜>")}
    if any(token in (roles["<｜User｜>"], roles["<｜Assistant｜>"]) for token in prefix):
        return [roles["<｜System｜>"]]
    return [] if roles["<｜System｜>"] in prefix else [roles["<｜System｜>"]]


def _deepseek_message(
    engine: FakeEngine,
    prefix: Sequence[int],
    role: str,
    content: str,
) -> list[int]:
    if role in ("system", "developer"):
        marker = _deepseek41_system_marker(prefix) if engine.info().is_deepseek41 else []
        return marker + engine.tokenize(content)
    if role == "assistant":
        opening = engine.tokenize("<｜Assistant｜>", rendered=True)
        if not content.startswith(("<think>", "</think>")):
            opening += engine.tokenize("</think>", rendered=True)
        return opening + engine.tokenize(content)
    if role in ("tool", "function"):
        return engine.tokenize("<｜User｜>", rendered=True) + engine.tokenize(
            "<tool_result>" + _escape_closing(content, "</tool_result>") + "</tool_result>"
        )
    return engine.tokenize("<｜User｜>", rendered=True) + engine.tokenize(content)


def render_message(
    engine: FakeEngine,
    prefix: Sequence[int],
    role: str,
    content: str,
) -> list[int]:
    info = engine.info()
    if info.is_qwen4:
        return _qwen_message(engine, role, content)
    if info.is_glm_dsa:
        return _glm_message(engine, role, content)
    return _deepseek_message(engine, prefix, role, content)


def assistant_prefix(
    engine: FakeEngine,
    prefix: Sequence[int],
    mode: int,
) -> list[int]:
    info = engine.info()
    if info.is_qwen4:
        suffix = engine.tokenize("\n" if think_enabled(mode) else "\n\n")
        if not think_enabled(mode):
            suffix += engine.tokenize("</think>", rendered=True) + engine.tokenize("\n\n")
        return _qwen_open(engine, "assistant") + engine.tokenize("<think>", rendered=True) + suffix
    if info.is_glm_dsa:
        opening = engine.tokenize("<|assistant|>", rendered=True)
        if not think_enabled(mode):
            return opening + engine.tokenize("<think></think>", rendered=True)
    else:
        opening = engine.tokenize("<｜Assistant｜>", rendered=True)
    marker = "<think>" if think_enabled(mode) else "</think>"
    return opening + engine.tokenize(marker, rendered=True)


def qwen_reasoning_effort(mode: int) -> str | None:
    if mode in (ThinkMode.HIGH, ThinkMode.MAX):
        return _QWEN_XHIGH
    if mode == ThinkMode.LOW:
        return _QWEN_LOW
    return None
