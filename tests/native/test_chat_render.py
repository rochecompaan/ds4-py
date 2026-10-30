import ctypes as C
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from ds4py._native.api import NativeEngine, _Handle
from ds4py._native.fake import FakeNative
from ds4py._native.protocol import EngineConfig, ThinkMode
from ds4py._native.structs import Tokens
from ds4py.engine.session import open as open_engine
from ds4py.errors import EngineError


class ChatLib:
    """Small callable boundary double; no model or native inference is needed."""

    def __init__(self) -> None:
        self.calls = []
        self.keepalive = []
        self.token_frees = 0
        self.effort = None
        self._functions = {
            "ds4_tokens_push": self.push,
            "ds4_tokens_free": self.free_tokens,
            "ds4_chat_begin": self.chat_begin,
            "ds4_chat_append_think_prefix": self.think_prefix,
            "ds4_chat_append_message": self.render_message,
            "ds4_chat_append_assistant_prefix": self.assistant_prefix,
            "ds4_qwen4_reasoning_effort_text": self.qwen_effort,
        }

    def fn(self, name: Any) -> Any:
        return self._functions[name]

    @staticmethod
    def tokens(pointer: Any) -> Any:
        return C.cast(pointer, C.POINTER(Tokens)).contents

    def push(self, pointer: Any, token: Any) -> None:
        tokens = self.tokens(pointer)
        values = [*list(tokens.v[: tokens.len]), token]
        array = (C.c_int * len(values))(*values)
        self.keepalive.append(array)
        tokens.v = C.cast(array, C.POINTER(C.c_int))
        tokens.len = tokens.cap = len(values)

    def append(self, pointer: Any, values: Any) -> None:
        for value in values:
            self.push(pointer, value)

    def free_tokens(self, pointer: Any) -> None:
        self.token_frees += 1
        tokens = self.tokens(pointer)
        tokens.v = C.POINTER(C.c_int)()
        tokens.len = tokens.cap = 0

    def chat_begin(self, raw: Any, tokens: Any) -> None:
        self.calls.append(("begin", raw))
        self.append(tokens, [90])

    def think_prefix(self, raw: Any, tokens: Any, mode: Any) -> None:
        self.calls.append(("think", raw, mode))
        self.append(tokens, [91, 92])

    def render_message(self, raw: Any, tokens: Any, role: Any, content: Any) -> None:
        self.calls.append(("message", raw, role, content))
        self.append(tokens, [93])

    def assistant_prefix(self, raw: Any, tokens: Any, mode: Any) -> None:
        self.calls.append(("assistant", raw, mode))
        self.append(tokens, [94])

    def qwen_effort(self, mode: Any) -> Any:
        self.calls.append(("effort", mode))
        return self.effort


def native_handle() -> Any:
    lib = ChatLib()
    engine = NativeEngine.__new__(NativeEngine)
    _Handle.__init__(engine, cast("Any", lib), 0x1234)
    return engine, lib


def fake_engine(family: Any = "deepseek4") -> Any:
    native = FakeNative()
    info = replace(
        native._info,
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )
    return FakeNative(info=info).open_engine(EngineConfig(Path("fake.gguf")))


def test_native_chat_deltas_preserve_prefix_and_free_token_storage() -> None:
    engine, lib = native_handle()
    prefix = [4, 5]

    assert engine.chat_begin() == [90]
    assert engine.think_prefix(prefix, ThinkMode.HIGH) == [91, 92]
    assert engine.render_message(prefix, "user", "hello") == [93]
    assert engine.assistant_prefix(prefix, ThinkMode.NONE) == [94]
    assert prefix == [4, 5]
    assert lib.token_frees == 4
    assert lib.calls == [
        ("begin", 0x1234),
        ("think", 0x1234, ThinkMode.HIGH),
        ("message", 0x1234, b"user", b"hello"),
        ("assistant", 0x1234, ThinkMode.NONE),
    ]


def test_native_chat_rejects_a_closed_handle() -> None:
    engine, _ = native_handle()
    engine.raw = 0
    for call in (
        engine.chat_begin,
        lambda: engine.think_prefix([], ThinkMode.HIGH),
        lambda: engine.render_message([], "user", "x"),
        lambda: engine.assistant_prefix([], ThinkMode.HIGH),
        lambda: engine.qwen_reasoning_effort(ThinkMode.HIGH),
    ):
        with pytest.raises(EngineError, match="closed"):
            call()


def test_native_qwen_effort_is_borrowed_and_may_be_null() -> None:
    engine, lib = native_handle()
    assert engine.qwen_reasoning_effort(ThinkMode.MEDIUM) is None
    lib.effort = b"xhigh effort"
    assert engine.qwen_reasoning_effort(ThinkMode.HIGH) == "xhigh effort"
    assert lib.calls == [("effort", ThinkMode.MEDIUM), ("effort", ThinkMode.HIGH)]
    assert lib.token_frees == 0


def test_native_chat_detects_a_rewritten_prefix() -> None:
    engine, lib = native_handle()

    def corrupt(raw: Any, pointer: Any, mode: Any) -> None:
        tokens = lib.tokens(pointer)
        tokens.v[0] = 999

    lib._functions["ds4_chat_append_think_prefix"] = corrupt
    with pytest.raises(EngineError, match="prefix"):
        engine.think_prefix([1], ThinkMode.HIGH)


def test_fake_rendered_specials_never_leak_into_plain_tokenization() -> None:
    engine = fake_engine()
    special = engine.tokenize("<tool_call>", rendered=True)
    assert len(special) == 1
    assert special[0] > 256
    assert engine.token_text(special[0]) == b"<tool_call>"
    text = "user <tool_call> <|im_start|> <think> ｜DSML｜"
    plain = engine.tokenize(text)
    assert plain == [byte + 1 for byte in text.encode()]
    assert special[0] not in plain
    assert b"".join(engine.token_text(token) for token in plain) == text.encode()


def test_fake_chat_rendering_appends_only_the_native_delta() -> None:
    engine = fake_engine()
    prefix = engine.tokenize("prefix")
    assert engine.chat_begin() == engine.tokenize("<｜begin▁of▁sentence｜>", rendered=True)
    assert engine.think_prefix(prefix, ThinkMode.NONE) == []
    assert engine.render_message(prefix, "user", "hello") == (
        engine.tokenize("<｜User｜>", rendered=True) + engine.tokenize("hello")
    )
    assert engine.assistant_prefix(prefix, ThinkMode.HIGH) == (engine.tokenize("<｜Assistant｜><think>", rendered=True))
    assert prefix == engine.tokenize("prefix")


def test_fake_qwen_native_tool_markup_is_plain_like_pinned_upstream() -> None:
    engine = fake_engine("qwen")
    delta = engine.render_message([], "tool", "result")
    tool_open = engine.tokenize("<tool_response>", rendered=True)
    assert len(tool_open) == 1
    assert tool_open[0] not in delta
    assert engine.tokenize("<|im_start|>", rendered=True)[0] in delta
    assert b"".join(engine.token_text(token) for token in delta) == (
        b"<|im_start|>user\n<tool_response>\nresult\n</tool_response><|im_end|>\n"
    )


def test_public_engine_chat_counterparts_run_on_the_worker() -> None:
    engine = open_engine(Path("fake.gguf"), native=FakeNative())
    try:
        assert engine._chat_begin() == engine.tokenize("<｜begin▁of▁sentence｜>", rendered=True)
        assert engine._think_prefix([], ThinkMode.NONE) == []
        assert engine._render_message([], "user", "x")
        assert engine._assistant_prefix([], ThinkMode.HIGH)
        assert engine._qwen_reasoning_effort(ThinkMode.MEDIUM) is None
    finally:
        engine.close()
