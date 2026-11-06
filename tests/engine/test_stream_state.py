from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from ds4py import ThinkMode
from ds4py._native.fake import FakeNative
from ds4py._native.protocol import EngineInfo
from ds4py.engine.decoder import Decoder
from ds4py.engine.events import Event
from ds4py.engine.pause import ControlSpan
from ds4py.engine.stream import RoundStream
from ds4py.engine.tool_syntax.base import ToolParser
from ds4py.engine.tool_syntax.delimiter import DelimiterParser
from ds4py.engine.tool_syntax.dsml import Dsml
from ds4py.engine.tool_syntax.glm import Glm
from ds4py.engine.tool_syntax.qwen import Qwen

ParserFactory = Callable[[], ToolParser]


def dsml_stanza(*, v41: bool, name: str = "echo", value: str = '{"outer":[1,{"inner":"value"}]}') -> bytes:
    calls, invoke, parameter = (" calls", " invoke", " parameter") if v41 else ("tool_calls", "invoke", "parameter")
    return (
        f'<｜DSML｜{calls}><｜DSML｜{invoke} name="{name}">'
        f'<｜DSML｜{parameter} name="value">{value}</｜DSML｜{parameter}>'
        f"</｜DSML｜{invoke}></｜DSML｜{calls}>"
    ).encode()


def delimiter_stanza(family: str, *, name: str = "echo", value: str = '{"outer":[1,{"inner":"value"}]}') -> bytes:
    if family == "glm":
        return f"<tool_call>{name}<arg_key>value</arg_key><arg_value>{value}</arg_value></tool_call>".encode()
    return f"<tool_call><function={name}><parameter=value>{value}</parameter></function></tool_call>".encode()


def parser_boundaries(
    factory: ParserFactory, full: bytes, structural: bytes, opener: bytes
) -> list[tuple[ParserFactory, bytes, bytes]]:
    value_split = full.index(b'"value"') + 4
    return [
        (factory, opener, full[len(opener) :]),
        (factory, structural, full[len(structural) :]),
        (factory, full[:value_split], full[value_split:]),
    ]


PARSER_BOUNDARIES = [
    *parser_boundaries(
        lambda: Dsml(v41=False),
        dsml_stanza(v41=False),
        "<｜DSML｜tool_calls>".encode(),
        "<｜DS".encode(),
    ),
    *parser_boundaries(
        lambda: Dsml(v41=True),
        dsml_stanza(v41=True),
        "<｜DSML｜ calls>".encode(),
        "<｜DS".encode(),
    ),
    *parser_boundaries(Glm, delimiter_stanza("glm"), b"<tool_call>", b"<tool"),
    *parser_boundaries(Qwen, delimiter_stanza("qwen"), b"<tool_call>", b"<tool"),
]


@pytest.mark.parametrize(
    ("factory", "prefix", "suffix"),
    PARSER_BOUNDARIES,
    ids=lambda value: getattr(value, "__name__", "") if callable(value) else "",
)
def test_parser_fork_and_restore_match_uninterrupted_parser(
    factory: ParserFactory,
    prefix: bytes,
    suffix: bytes,
) -> None:
    reference = factory()
    reference.feed(prefix + suffix)

    parser = factory()
    parser.feed(prefix)
    captured = parser.capture()
    fork = parser.fork()
    fork.feed(suffix)
    assert fork.feed(b"") == reference.feed(b"")
    assert fork.wants_greedy_sampling() == reference.wants_greedy_sampling()

    parser.feed(b"<different>suffix")
    parser.restore(captured)
    parser.feed(suffix)
    assert parser.feed(b"") == reference.feed(b"")
    assert parser.wants_greedy_sampling() == reference.wants_greedy_sampling()


def second_call_data(factory: ParserFactory) -> tuple[bytes, bytes]:
    if factory is Glm:
        return delimiter_stanza("glm", name="first") + b"<tool", b"_call>second</tool_call>"
    if factory is Qwen:
        return delimiter_stanza("qwen", name="first") + b"<tool", (
            b"_call><function=second><parameter=value>two</parameter></function></tool_call>"
        )
    v41 = cast("Dsml", factory()).v41
    calls, invoke, parameter = (" calls", " invoke", " parameter") if v41 else ("tool_calls", "invoke", "parameter")
    first = dsml_stanza(v41=v41, name="first")
    prefix = first[: -len(f"</｜DSML｜{calls}>".encode())] + f'<｜DSML｜{invoke} name="second">'.encode()
    suffix = (
        f'<｜DSML｜{parameter} name="value" string="true">two</｜DSML｜{parameter}>'
        f"</｜DSML｜{invoke}></｜DSML｜{calls}>"
    ).encode()
    return prefix, suffix


@pytest.mark.parametrize("factory", [lambda: Dsml(v41=False), lambda: Dsml(v41=True), Glm, Qwen])
def test_parser_fork_preserves_completed_calls_and_partial_next_call(factory: ParserFactory) -> None:
    prefix, suffix = second_call_data(factory)
    reference = factory()
    reference.feed(prefix + suffix)

    parser = factory()
    parser.feed(prefix)
    captured = parser.capture()
    fork = parser.fork()
    fork.feed(suffix)
    assert fork.feed(b"") == reference.feed(b"")

    parser.feed(b"broken")
    parser.restore(captured)
    parser.feed(suffix)
    assert parser.feed(b"") == reference.feed(b"")


@pytest.mark.parametrize("factory", [lambda: Dsml(v41=False), lambda: Dsml(v41=True), Qwen])
def test_parser_state_does_not_share_nested_argument_containers(factory: ParserFactory) -> None:
    if factory is Qwen:
        full = delimiter_stanza("qwen")
        call_close = full.index(b"</function>")
    else:
        v41 = cast("Dsml", factory()).v41
        full = dsml_stanza(v41=v41)
        invoke = " invoke" if v41 else "invoke"
        call_close = full.index(f"</｜DSML｜{invoke}>".encode())
    parser = factory()
    parser.feed(full[:call_close])
    captured = parser.capture()
    fork = parser.fork()

    stateful = cast("Dsml | DelimiterParser", parser)
    original = cast("dict[str, object]", stateful._args["value"].value)
    outer = cast("list[object]", original["outer"])
    nested = cast("dict[str, object]", outer[1])
    nested["inner"] = "mutated"

    forked = cast("dict[str, object]", cast("Dsml | DelimiterParser", fork)._args["value"].value)
    assert cast("dict[str, object]", cast("list[object]", forked["outer"])[1])["inner"] == "value"
    parser.restore(captured)
    restored = cast("dict[str, object]", stateful._args["value"].value)
    assert cast("dict[str, object]", cast("list[object]", restored["outer"])[1])["inner"] == "value"


@pytest.mark.parametrize(
    ("thinking", "prefix", "suffix"),
    [
        (False, b"A\xf0\x9f", b"\x98\x80B"),
        (False, b"before<th", b"ink>inside</think>after"),
        (False, b"before<think>inside</th", b"ink>after"),
    ],
)
def test_decoder_fork_and_restore_preserve_utf8_and_marker_state(
    thinking: bool,
    prefix: bytes,
    suffix: bytes,
) -> None:
    reference = Decoder(thinking=thinking)
    expected = reference.feed(prefix) + reference.feed(suffix) + reference.finish()

    decoder = Decoder(thinking=thinking)
    prefix_deltas = decoder.feed(prefix)
    expected_suffix = expected[len(prefix_deltas) :]
    captured = decoder.capture()
    fork = decoder.fork()
    assert fork.feed(suffix) + fork.finish() == expected_suffix

    decoder.feed(b"different")
    decoder.finish()
    decoder.restore(captured)
    assert decoder.feed(suffix) + decoder.finish() == expected_suffix
    assert decoder.thinking == reference.thinking


def test_control_span_fork_and_restore_preserve_overlap_and_resume_state() -> None:
    reference = ControlSpan([b"abab"])
    reference.feed(b"ab")
    assert reference.feed(b"ab") == b"abab"
    reference.resume()
    assert reference.feed(b"ab") == b"abab"

    control = ControlSpan([b"abab"])
    control.feed(b"ab")
    captured = control.capture()
    fork = control.fork()
    assert fork.feed(b"ab") == b"abab"
    fork.resume()
    assert fork.feed(b"ab") == b"abab"
    assert fork.stop_sequence == reference.stop_sequence
    assert fork.active_stop_sequences == reference.active_stop_sequences
    assert fork.matcher.pending == reference.matcher.pending

    control.feed(b"different")
    control.restore(captured)
    assert control.feed(b"ab") == b"abab"
    control.resume()
    assert control.feed(b"ab") == b"abab"
    assert control.stop_sequence == reference.stop_sequence
    assert control.active_stop_sequences == reference.active_stop_sequences
    assert control.matcher.pending == reference.matcher.pending


BASE_INFO = FakeNative()._info
ROUND_STREAM_INFOS = [
    BASE_INFO,
    replace(BASE_INFO, family="deepseek41", is_deepseek41=True),
    replace(BASE_INFO, family="glm", is_glm_dsa=True),
    replace(BASE_INFO, family="qwen", is_qwen4=True),
]


def observe_bytes(stream: RoundStream, data: bytes, events: list[Event]) -> None:
    for byte in data:
        stream.observe(byte, bytes([byte]), events.append)


def finish_stream(stream: RoundStream, events: list[Event]) -> None:
    stream.finalize(parser=True, emit=events.append)
    stream.emit_calls(events.append)


def assert_stream_matches(
    actual: RoundStream, expected: RoundStream, events: list[Event], expected_events: list[Event]
) -> None:
    assert actual.sampled_tokens == expected.sampled_tokens
    assert actual.snapshot == expected.snapshot
    assert actual.think_tokens == expected.think_tokens
    assert actual.finalized == expected.finalized
    assert actual.control.stop_sequence == expected.control.stop_sequence
    assert actual.active_stop_sequences == expected.active_stop_sequences
    assert actual.control.matcher.pending == expected.control.matcher.pending
    assert actual._control_pending == expected._control_pending
    assert actual._last_state == expected._last_state
    assert actual._last_thinking == expected._last_thinking
    assert actual._calls_emitted == expected._calls_emitted
    assert events == expected_events


@pytest.mark.parametrize("info", ROUND_STREAM_INFOS, ids=lambda info: info.family)
def test_round_stream_fork_and_restore_match_uninterrupted_reference(info: EngineInfo) -> None:
    if info.is_glm_dsa:
        stanza = delimiter_stanza("glm")
    elif info.is_qwen4:
        stanza = delimiter_stanza("qwen")
    else:
        stanza = dsml_stanza(v41=info.is_deepseek41)
    prefix = b"before<think>reason</th"
    suffix = b"ink>" + stanza + b"st"

    reference = RoundStream(info, ThinkMode.NONE, control=ControlSpan([b"stop"]))
    reference_events: list[Event] = []
    observe_bytes(reference, prefix, reference_events)
    observe_bytes(reference, suffix, reference_events)
    finish_stream(reference, reference_events)

    control = ControlSpan([b"stop"])
    stream = RoundStream(info, ThinkMode.NONE, control=control)
    sampled_tokens = stream.sampled_tokens
    events: list[Event] = []
    observe_bytes(stream, prefix, events)
    captured = stream.capture()
    fork = stream.fork()
    fork_events = list(events)
    observe_bytes(fork, suffix, fork_events)
    finish_stream(fork, fork_events)
    assert_stream_matches(fork, reference, fork_events, reference_events)
    assert stream.sampled_tokens == list(prefix)
    assert stream.snapshot.in_think
    assert stream._control_pending == b"</th"

    prefix_events = list(events)
    observe_bytes(stream, b"different", events)
    stream.finalize(parser=True, emit=events.append)
    stream.restore(captured)
    assert stream.control is control
    assert stream.sampled_tokens is sampled_tokens
    events = prefix_events
    observe_bytes(stream, suffix, events)
    finish_stream(stream, events)
    assert_stream_matches(stream, reference, events, reference_events)
