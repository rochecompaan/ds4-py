from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import pytest
from pydantic_ai.messages import (
    BinaryImage,
    InstructionBaselinePart,
    InstructionPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition

import ds4py
from ds4py import ThinkMode
from ds4py._native.fake import FakeNative
from ds4py.binding import MessageBinder
from ds4py.binding_provenance import BoundMessage, _content_hash, prepare_provenance
from ds4py.engine.session import Engine, Session
from ds4py.errors import EngineError


def fake_engine(family: Any = "deepseek4", **fake: Any) -> Engine:
    info = replace(
        FakeNative()._info,
        family=family,
        is_deepseek41=family == "deepseek41",
        is_glm_dsa=family == "glm",
        is_qwen4=family == "qwen",
    )
    return ds4py.open(
        "f.gguf",
        ctx=8192,
        think=ThinkMode.NONE,
        native=FakeNative(info=info, **fake),
    )


def user(text: Any) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)])


def tool(name: Any = "read_file") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Read a file.",
        parameters_json_schema={"type": "object", "properties": {}},
    )


def decode(engine: Any, tokens: Any) -> str:
    return b"".join(engine.token_text(token) for token in tokens).decode()


@pytest.mark.parametrize(
    ("native_reason", "finish_reason"),
    [("max_tokens", "length"), ("stop_token", "stop"), ("tool_call_complete", "tool_call")],
)
def test_adopt_preserves_sampled_record_in_serialized_snapshot(native_reason: str, finish_reason: Any) -> None:
    parts = (
        [ToolCallPart(tool_name="read_file", args={"path": "hello.txt"}, tool_call_id="call-1")]
        if native_reason == "tool_call_complete"
        else [TextPart("same")]
    )
    candidate = ModelResponse(
        parts,
        finish_reason=finish_reason,
        metadata={"ds4.key": "response", "ds4.source": "sampled"},
    )
    bound = deepcopy(candidate)
    content_hash = _content_hash(bound)
    sampled: dict[str, Any] = {
        "content_hash": content_hash,
        "segments": [{"kind": "assistant", "tokens": [17, 68]}],
    }
    if native_reason != "max_tokens":
        sampled["segments"].append({"kind": "eos", "tokens": [99]})
    bound.metadata = dict(bound.metadata or {})
    bound.metadata["ds4.sampled"] = sampled
    binder = MessageBinder.__new__(MessageBinder)
    binder._records = [BoundMessage("response", content_hash, 0, 3)]
    binder._bound_messages = [bound]

    adopted = cast("ModelResponse", binder.adopt([candidate])[0])

    assert adopted is candidate
    assert (adopted.metadata or {}).get("ds4.sampled") == sampled
    assert (adopted.metadata or {})["ds4.sampled"] is not sampled
    restored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json([adopted]))[0]
    assert isinstance(restored, ModelResponse)
    assert (restored.metadata or {}).get("ds4.sampled") == sampled


def test_first_request_renders_tools_and_binds_each_message() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        messages = [
            user("read it"),
            ModelResponse([TextPart("done")]),
            ModelRequest([ToolReturnPart("read_file", "contents")]),
        ]

        change = binder.prepare(messages, tools=[tool()], instructions="rules")

        assert change.kind == "append"
        assert change.start == 0
        assert change.continuation is True
        assert "# Tools" in decode(engine, session.transcript.tokens)
        keyed = [segment for segment in binder.export_segments() if segment.key]
        assert len({segment.key for segment in keyed}) == 3
        assert all(segment.content_hash for segment in keyed)


def test_restore_binds_an_ordered_subsequence_of_canonical_history() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        canonical = [
            user("first"),
            ModelResponse([TextPart("omitted answer")]),
            user("omitted request"),
            ModelResponse([TextPart("last answer")]),
        ]
        bound = [canonical[0], canonical[3]]
        binder.prepare(bound)
        provenance = binder.export_segments()

        restored = MessageBinder(session)
        restored.restore(canonical, provenance)

        assert len(canonical) == 4
        assert restored._bound_messages == bound
        assert [record.key for record in restored._records] == [
            cast("Any", message.metadata)["ds4.key"] for message in bound
        ]
        assert restored._signature is None
        assert restored._restored is True


def test_restore_uses_session_segment_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        canonical = [user("first"), ModelResponse([TextPart("answer")])]
        binder.prepare(canonical)
        saved_segments = binder.export_segments()
        saved_records = binder.records()

        restored = MessageBinder(session)
        calls: list[tuple[Any, ...]] = []

        def install(segments: Any) -> None:
            # Preparation is pure: no record swap may happen before installation.
            assert restored.records() == ()
            calls.append(tuple(segments))
            Session.install_segments(session, segments)

        monkeypatch.setattr(session, "install_segments", install)
        restored.restore(canonical, saved_segments)

        assert calls == [session.transcript.segments]
        assert restored.records() == saved_records
        assert restored.export_segments() == saved_segments
        assert restored._signature is None
        assert restored._restored is True


def test_prepare_provenance_deep_copies_bound_messages() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        messages = [user("first")]
        binder.prepare(messages)
        key = cast("Any", messages[0].metadata)["ds4.key"]
        prepared = prepare_provenance(messages, binder.export_segments(), loaded=len(session.transcript))

        messages[0].metadata = {"ds4.key": "changed"}

        assert cast("Any", prepared.bound_messages[0].metadata)["ds4.key"] == key


def test_prepare_provenance_accepts_replacement_segments() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        messages = [user("first")]
        binder.prepare(messages)

        prepared = prepare_provenance(messages, session.transcript.segments, loaded=len(session.transcript))

        assert prepared.installed == session.transcript.segments
        assert prepared.records == binder.records()


def test_apply_compaction_assigns_records_without_restore_state() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        messages = [user("first"), ModelResponse([TextPart("answer")])]
        binder.prepare(messages)
        signature = binder._signature
        prepared = prepare_provenance(messages, binder.export_segments(), loaded=len(session.transcript))

        binder.apply_compaction(prepared)

        assert binder.records() == prepared.records
        assert binder._signature == signature
        assert binder._restored is False


def test_system_and_instruction_parts_are_separate_system_segments() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        before = len(session.transcript.segments)
        binder.prepare(
            [
                ModelRequest(
                    cast(
                        "Any",
                        [
                            SystemPromptPart("system text"),
                            InstructionPart("instruction text"),
                        ],
                    )
                )
            ],
            instructions="global",
        )

        decoded = decode(engine, session.transcript.tokens)
        assert decoded.index("global") < decoded.index("# Tools")
        assert decoded.index("# Tools") < decoded.index("system text")
        segments = session.transcript.segments[before:][-2:]
        assert [segment.kind for segment in segments] == ["system", "system"]
        assert "system text" in decode(engine, session.transcript.tokens[segments[0].start : segments[0].end])
        assert "instruction text" in decode(engine, session.transcript.tokens[segments[1].start : segments[1].end])


def test_user_text_is_plain_and_tool_markup_is_not_rendered() -> None:
    payload = "user <tool_call> <|im_start|> <think> ｜DSML｜"
    with fake_engine("qwen") as engine:
        session = engine.session()
        MessageBinder(session).prepare([user(payload)])
        segment = session.transcript.segments[-1]
        tokens = list(session.transcript.tokens[segment.start : segment.end])
        plain = engine.tokenize(payload)
        assert any(tokens[index : index + len(plain)] == plain for index in range(len(tokens) - len(plain) + 1))


def test_tool_results_join_in_history_order_as_one_tool_message() -> None:
    with fake_engine() as engine:
        session = engine.session()
        MessageBinder(session).prepare(
            [
                ModelRequest(
                    [
                        ToolReturnPart("read_file", "result 1"),
                        ToolReturnPart("edit_file", "result 2"),
                    ]
                )
            ]
        )

        tool_segments = [s for s in session.transcript.segments if s.kind == "tool"]
        assert len(tool_segments) == 1
        text = decode(
            engine,
            session.transcript.tokens[tool_segments[0].start : tool_segments[0].end],
        )
        assert "Tool result 1 (read_file):\nresult 1\n\nTool result 2 (edit_file):\nresult 2" in text


def test_scalar_and_structured_tool_results_have_stable_text() -> None:
    with fake_engine() as engine:
        session = engine.session()
        MessageBinder(session).prepare(
            [
                ModelRequest(
                    [
                        ToolReturnPart("bytes", b"a\xffb"),
                        ToolReturnPart("mapping", {"b": 2, "a": 1}),
                        ToolReturnPart("sequence", [1, "x"]),
                        ToolReturnPart("scalar", 3),
                    ]
                )
            ]
        )

        text = decode(engine, session.transcript.tokens)
        assert "Tool result 1 (bytes):\na�b" in text
        assert 'Tool result 2 (mapping):\n{"a":1,"b":2}' in text
        assert 'Tool result 3 (sequence):\n[1,"x"]' in text
        assert "Tool result 4 (scalar):\n3" in text


def test_unknown_tool_result_uses_tool_error_text() -> None:
    with fake_engine() as engine:
        session = engine.session()
        MessageBinder(session).prepare(
            [
                ModelRequest(
                    [
                        RetryPromptPart(
                            "Tool error: unknown tool missing",
                            tool_name="missing",
                            tool_call_id="call-1",
                        )
                    ]
                )
            ]
        )
        assert "Tool error: unknown tool missing" in decode(engine, session.transcript.tokens)


def test_images_fail_before_mutating_the_transcript() -> None:
    with fake_engine() as engine:
        session = engine.session()
        binder = MessageBinder(session)
        message = ModelRequest([UserPromptPart([BinaryImage(data=b"not really a png", media_type="image/png")])])

        with pytest.raises(EngineError, match="M5"):
            binder.prepare([message], tools=[tool()], instructions="rules")

        assert session.transcript.tokens == ()
        assert session.transcript.segments == ()


def test_rejected_non_integer_truncate_leaves_checkpoint_usable() -> None:
    with fake_engine() as engine:
        session = engine.session()
        session.append_tokens([1, 2, 3, 4], "user")
        list(session.run_round(max_tokens=0))
        session.checkpoint("kept")
        session.append_tokens([5, 6], "user")
        list(session.run_round(max_tokens=0))
        before = session.transcript.tokens

        with pytest.raises(TypeError):
            session.truncate(cast("Any", 2.0))

        assert session.transcript.tokens == before
        assert session.history.checkpoint_tokens == {"kept": 4}
        session.truncate(4)
        plan = session.plan()
        assert plan.kind == "restore"
        assert plan.checkpoint == "kept"
        list(session.run_round(max_tokens=0))


def test_canonicalize_exact_merged_requests_restores_bound_keys() -> None:
    with fake_engine() as engine:
        binder = MessageBinder(engine.session())
        first, second = user("summary"), user("retained constraints")
        messages: list[ModelMessage] = [first, second, ModelResponse([TextPart("answer")])]
        binder.prepare(messages)
        merged = ModelRequest([*first.parts, *second.parts])
        canonical = binder.canonicalize([merged, messages[2]])
        assert canonical == messages
        assert canonical[0] is not messages[0]
        assert canonical[1] is not messages[1]
        assert binder.prepare(canonical).kind == "unchanged"
        canonical[0].parts = []
        assert binder.canonicalize([merged])[0].parts == messages[0].parts


def test_canonicalize_leaves_nonmatching_view_only_requests_untouched() -> None:
    with fake_engine() as engine:
        binder = MessageBinder(engine.session())
        binder.prepare([user("summary"), user("retained constraints")])
        changed = ModelRequest([UserPromptPart("summary"), UserPromptPart("different view")])
        view: list[ModelMessage] = [changed]
        assert binder.canonicalize(view) is view
        assert view[0] is changed


def test_canonicalize_does_not_discard_changed_framework_parts() -> None:
    with fake_engine() as engine:
        binder = MessageBinder(engine.session())
        first, second = user("summary"), user("retained")
        binder.prepare([first, second])
        changed = ModelRequest([*first.parts, *second.parts, InstructionBaselinePart(instructions=[])])
        view: list[ModelMessage] = [changed]
        assert binder.canonicalize(view) is view
