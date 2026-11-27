import asyncio
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool
from pydantic_ai.usage import RunUsage
from test_navigator import _fixture_item

from ds4py.context_paging.public_history import (
    PublicHistoryItem,
    PublicModelTurn,
    PublicText,
    PublicToolResult,
    PublicUser,
)
from ds4py.context_paging.tools import create_context_paging_toolset
from ds4py.errors import EngineError
from ds4py.message_metadata import DS4_KEY

FIXTURES = json.loads((Path(__file__).parents[1] / "fixtures/context_paging/recovery.json").read_text())["fixtures"]


def _call(toolset: FunctionToolset[Any], name: str, arguments: dict[str, Any]) -> Any:
    async def scenario() -> Any:
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        tools = await toolset.get_tools(ctx)
        tool: ToolsetTool[Any] = tools[name]
        return await toolset.call_tool(name, arguments, ctx, tool)

    return asyncio.run(scenario())


def _value(result: object) -> Any:
    return result.return_value if isinstance(result, ToolReturn) else result


def _payload(result: object) -> Any:
    value = _value(result)
    block = value["content"][0]
    return json.loads(block["text"])


def _toolset(items: Sequence[PublicHistoryItem], *, enabled: Callable[[], bool] = lambda: True) -> FunctionToolset[Any]:
    return create_context_paging_toolset(is_enabled=enabled, public_items=lambda: items)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture["id"])
def test_generated_recovery_contract(fixture: dict[str, Any]) -> None:
    items = [item for value in fixture["input"]["items"] if (item := _fixture_item(value)) is not None]
    toolset = _toolset(items)
    expected = fixture["expected"]
    if "value" in expected and "ds4Error" in expected["value"]:
        with pytest.raises(EngineError, match=expected["value"]["ds4Error"]["message"]):
            _call(toolset, fixture["input"]["tool"], fixture["input"]["params"])
        return

    result = _value(_call(toolset, fixture["input"]["tool"], fixture["input"]["params"]))
    if "error" in expected:
        assert result == {
            "content": [{"type": "text", "text": f"Recovery input error: {expected['error']['message']}"}]
        }
    else:
        assert result == expected["value"]


def test_real_tool_schemas_are_strict_and_use_camel_case_aliases() -> None:
    toolset = _toolset([])

    read = toolset.tools["read_context_output"].tool_def.parameters_json_schema
    load = toolset.tools["load_history"].tool_def.parameters_json_schema
    browse = toolset.tools["browse_history"].tool_def.parameters_json_schema

    assert set(toolset.tools) == {"search_history", "browse_history", "load_history", "read_context_output"}
    assert read["additionalProperties"] is False
    assert {"historyId", "contentIndex", "toolCallId"} <= set(read["properties"])
    assert "historyIds" in load["properties"]
    assert "historyId" in browse["properties"]
    with pytest.raises((ValueError, TypeError)):
        _call(toolset, "browse_history", {"direction": "backward", "unexpected": True})


def test_disabled_refusal_is_exact_for_all_four_tools() -> None:
    toolset = _toolset([], enabled=lambda: False)
    calls = {
        "search_history": {"query": "x"},
        "browse_history": {"direction": "backward"},
        "load_history": {"historyIds": ["x"]},
        "read_context_output": {"historyId": "x", "source": "assistant"},
    }

    for name, arguments in calls.items():
        result = _call(toolset, name, arguments)
        assert _value(result) == "Context paging recovery tools are disabled."


def test_load_defers_large_items_but_loads_smaller_later_items_in_request_order() -> None:
    items = [
        PublicUser("large", 0, 0, ("x" * 30_000,)),
        PublicUser("small", 1, 1, ("small body",)),
        PublicModelTurn(
            "image",
            2,
            1,
            (PublicText("image metadata"),),
            (
                PublicToolResult(
                    "image-call", "read", [{"type": "image", "data": "AAECAwQ=", "mimeType": "image/png"}], False, 1, 0
                ),
            ),
            (),
            (),
            False,
            False,
        ),
    ]
    toolset = _toolset(items)

    payload = _payload(_call(toolset, "load_history", {"historyIds": ["large", "small", "image"]}))
    loaded = payload["items"]

    assert [item["id"] for item in loaded] == ["large", "small", "image"]
    assert loaded[0]["deferred"] is True
    assert "userMessage" in loaded[1]
    assert loaded[1]["userMessage"]["content"] == "small body"
    assert loaded[2]["deferred"] is True
    assert "AAECAwQ=" not in json.dumps(payload)
    assert loaded[0]["readParts"] == {
        "historyId": "large",
        "mode": "parts",
        "offset": loaded[0]["nextPartOffset"] or 0,
    }
    assert len(json.dumps(payload).encode()) < 16_384


def test_search_load_lists_unloaded_ids_and_read_modes_preserve_handles() -> None:
    items = [PublicUser("user-1", 0, 0, ("needle " + "x" * 20_000,)), PublicUser("user-2", 1, 1, ("other",))]
    toolset = _toolset(items)

    searched = _payload(_call(toolset, "search_history", {"query": "needle", "load": True}))
    assert searched["references"][0]["historyId"] == "user-1"
    assert searched["items"][0]["id"] == "user-1"

    parts = _payload(
        _call(
            toolset,
            "read_context_output",
            {"historyId": "user-1", "mode": "parts", "limit": 1},
        )
    )
    assert parts["offset"] == 0
    assert parts["totalParts"] >= 1
    handle = parts["parts"][0]["read"]

    page = _call(toolset, "read_context_output", dict(handle))
    blocks = _value(page)["content"]
    header = json.loads(blocks[0]["text"])
    assert header["historyId"] == "user-1"
    assert blocks[1]["text"]


def test_bytes_tool_results_are_metadata_only_across_recovery_tools() -> None:
    messages = [
        ModelResponse([ToolCallPart("read", {}, tool_call_id="bytes-call")], metadata={DS4_KEY: "bytes-turn"}),
        ModelRequest(
            [ToolReturnPart("read", b"\x00\x01\x02\x03", tool_call_id="bytes-call")],
            metadata={DS4_KEY: "bytes-result"},
        ),
    ]
    toolset = create_context_paging_toolset(is_enabled=lambda: True, history_source=lambda: messages)

    calls = {
        "search_history": {"query": "read"},
        "load_history": {"historyIds": ["bytes-turn"]},
        "read_context_output": {"historyId": "bytes-turn", "mode": "parts"},
    }
    for name, arguments in calls.items():
        payload = _payload(_call(toolset, name, arguments))
        encoded = json.dumps(payload, separators=(",", ":"))
        assert '"type":"bytes"' in encoded or '\\"type\\":\\"bytes\\"' in encoded
        assert '"bytes":4' in encoded or '\\"bytes\\":4' in encoded
        assert "b'\\x00" not in encoded


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_non_image_binary_tool_results_are_metadata_only_across_recovery_tools(nested: bool) -> None:
    binary = BinaryContent(b"%PDF", media_type="application/pdf")
    content: object = {"document": [binary]} if nested else binary
    messages = [
        ModelResponse([ToolCallPart("read", {}, tool_call_id="binary-call")], metadata={DS4_KEY: "binary-turn"}),
        ModelRequest(
            [ToolReturnPart("read", content, tool_call_id="binary-call")],
            metadata={DS4_KEY: "binary-result"},
        ),
    ]
    toolset = create_context_paging_toolset(is_enabled=lambda: True, history_source=lambda: messages)

    searched = _payload(_call(toolset, "search_history", {"query": "read"}))
    loaded = _payload(_call(toolset, "load_history", {"historyIds": ["binary-turn"]}))
    parts = _payload(_call(toolset, "read_context_output", {"historyId": "binary-turn", "mode": "parts"}))
    binary_part = next(part for part in parts["parts"] if "application/pdf" in json.dumps(part))
    page = _value(_call(toolset, "read_context_output", dict(binary_part["read"], mode="text")))

    encoded = json.dumps([searched, loaded, parts, page], separators=(",", ":"))
    assert '"type":"bytes"' in encoded or '\\"type\\":\\"bytes\\"' in encoded
    assert '"bytes":4' in encoded or '\\"bytes\\":4' in encoded
    assert "application/pdf" in encoded
    assert "%PDF" not in encoded
    assert "BinaryContent" not in encoded


@pytest.mark.parametrize(
    ("content", "expected_type", "expected_reference"),
    [
        (ImageUrl("https://example.test/shot.png"), "image", "https://example.test/shot.png"),
        (AudioUrl("https://example.test/sound.mp3"), "audio", "https://example.test/sound.mp3"),
        (DocumentUrl("https://example.test/doc.pdf"), "document", "https://example.test/doc.pdf"),
        (VideoUrl("https://example.test/movie.mp4"), "video", "https://example.test/movie.mp4"),
        (UploadedFile("file-123", provider_name="openai", media_type="application/pdf"), "file", "file-123"),
    ],
    ids=["image-url", "audio-url", "document-url", "video-url", "uploaded-file"],
)
@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_multimodal_tool_results_are_metadata_only_across_recovery_tools(
    content: object, expected_type: str, expected_reference: str, nested: bool
) -> None:
    wrapped = {"attachment": [content]} if nested else content
    messages = [
        ModelResponse([ToolCallPart("read", {}, tool_call_id="media-call")], metadata={DS4_KEY: "media-turn"}),
        ModelRequest(
            [ToolReturnPart("read", wrapped, tool_call_id="media-call")],
            metadata={DS4_KEY: "media-result"},
        ),
    ]
    toolset = create_context_paging_toolset(is_enabled=lambda: True, history_source=lambda: messages)

    searched = _payload(_call(toolset, "search_history", {"query": "read"}))
    loaded = _payload(_call(toolset, "load_history", {"historyIds": ["media-turn"]}))
    parts = _payload(_call(toolset, "read_context_output", {"historyId": "media-turn", "mode": "parts"}))
    if isinstance(content, ImageUrl) and nested:
        media_part = next(part for part in parts["parts"] if part["part"] == "/toolResults/0/content")
    else:
        media_part = next(part for part in parts["parts"] if expected_reference in json.dumps(part))
    page = _value(_call(toolset, "read_context_output", dict(media_part["read"], mode="text")))

    encoded = json.dumps([searched, loaded, parts, page], separators=(",", ":"))
    assert f'"type":"{expected_type}"' in encoded or f'\\"type\\":\\"{expected_type}\\"' in encoded
    assert expected_reference in encoded
    assert type(content).__name__ not in encoded
    if isinstance(content, ImageUrl):
        assert loaded["items"][0]["deferred"] is True
        if nested:
            recovered = json.loads(page["content"][1]["text"])
            image_read = recovered["attachment"][0]["read"]
        else:
            assert media_part["type"] == "image"
            image_read = media_part["read"]
        with pytest.raises(EngineError, match="image recovery requires M5 vision support"):
            _call(toolset, "read_context_output", image_read)


@pytest.mark.parametrize(
    "content",
    [
        ImageUrl("data:image/png;base64,AAECAwQ="),
        DocumentUrl("data:application/pdf;base64,AAECAwQ="),
    ],
    ids=["image", "document"],
)
@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_data_url_tool_results_never_expose_embedded_payloads(content: object, nested: bool) -> None:
    wrapped = {"attachment": [content]} if nested else content
    messages = [
        ModelResponse([ToolCallPart("read", {}, tool_call_id="embedded-call")], metadata={DS4_KEY: "embedded-turn"}),
        ModelRequest(
            [ToolReturnPart("read", wrapped, tool_call_id="embedded-call")],
            metadata={DS4_KEY: "embedded-result"},
        ),
    ]
    toolset = create_context_paging_toolset(is_enabled=lambda: True, history_source=lambda: messages)

    searched = _payload(_call(toolset, "search_history", {"query": "read"}))
    loaded = _payload(_call(toolset, "load_history", {"historyIds": ["embedded-turn"]}))
    parts = _payload(_call(toolset, "read_context_output", {"historyId": "embedded-turn", "mode": "parts"}))
    content_part = next(part for part in parts["parts"] if part["part"] == "/toolResults/0/content")
    page = _value(_call(toolset, "read_context_output", dict(content_part["read"], mode="text")))

    encoded = json.dumps([searched, loaded, parts, page], separators=(",", ":"))
    assert "data:" not in encoded
    assert "AAECAwQ=" not in encoded
    assert "base64" not in encoded
    if isinstance(content, ImageUrl):
        assert loaded["items"][0]["deferred"] is True
        if nested:
            recovered = json.loads(page["content"][1]["text"])
            image_read = recovered["attachment"][0]["read"]
        else:
            assert content_part["type"] == "image"
            image_read = content_part["read"]
        with pytest.raises(EngineError, match="image recovery requires M5 vision support"):
            _call(toolset, "read_context_output", image_read)
    else:
        assert '"type":"bytes"' in encoded or '\\"type\\":\\"bytes\\"' in encoded
        assert "application/pdf" in encoded


@pytest.mark.parametrize("content", [datetime(2026, 11, 27, tzinfo=UTC), Decimal("1.5")])
def test_search_rejects_non_json_results_with_a_recovery_reply(content: object) -> None:
    messages = [
        ModelResponse([ToolCallPart("read", {}, tool_call_id="value-call")], metadata={DS4_KEY: "value-turn"}),
        ModelRequest(
            [ToolReturnPart("read", {"value": content}, tool_call_id="value-call")],
            metadata={DS4_KEY: "value-result"},
        ),
    ]
    toolset = create_context_paging_toolset(is_enabled=lambda: True, history_source=lambda: messages)

    for name, arguments in [
        ("search_history", {"query": "read"}),
        ("load_history", {"historyIds": ["value-turn"]}),
    ]:
        result = _call(toolset, name, arguments)
        text = _value(result)["content"][0]["text"]
        assert "Recovery input error: Recovery part contains a value that is not public JSON data." in text


@pytest.mark.parametrize("image_type", [BinaryContent, BinaryImage])
def test_real_image_tool_results_are_metadata_only_across_recovery_tools(image_type: Any) -> None:
    payload = b"IMAGEBYTES"
    messages = [
        ModelResponse([ToolCallPart("read", {}, tool_call_id="image-call")], metadata={DS4_KEY: "image-turn"}),
        ModelRequest(
            [
                ToolReturnPart(
                    "read",
                    image_type(payload, media_type="image/png"),
                    tool_call_id="image-call",
                )
            ],
            metadata={DS4_KEY: "image-result"},
        ),
    ]
    toolset = create_context_paging_toolset(is_enabled=lambda: True, history_source=lambda: messages)

    searched = _payload(_call(toolset, "search_history", {"query": "read"}))
    assert searched["references"][0]["historyId"] == "image-turn"
    loaded = _payload(_call(toolset, "load_history", {"historyIds": ["image-turn"]}))
    assert loaded["items"][0]["deferred"] is True
    parts = _payload(_call(toolset, "read_context_output", {"historyId": "image-turn", "mode": "parts"}))
    image = next(part for part in parts["parts"] if part["type"] == "image")

    page = _value(_call(toolset, "read_context_output", dict(image["read"], mode="text")))
    encoded = json.dumps([searched, loaded, parts, page], separators=(",", ":"))
    assert "SU1BR0VCWVRFUw==" not in encoded
    assert "IMAGEBYTES" not in encoded
    assert "image/png" in encoded
    with pytest.raises(EngineError, match="image recovery requires M5 vision support"):
        _call(toolset, "read_context_output", image["read"])


def test_model_incorrect_read_input_returns_guidance_instead_of_crashing() -> None:
    toolset = _toolset([PublicUser("user-1", 0, 0, ("needle",))])

    result = _call(
        toolset,
        "read_context_output",
        {"historyId": "user-1", "source": "assistant"},
    )

    text = _value(result)["content"][0]["text"]
    assert "Recovery input error:" in text
    assert "not a model turn" in text
    assert "part handle" in text


def test_valid_image_mode_raises_m5_error_without_exposing_bytes() -> None:
    image = PublicModelTurn(
        "turn-image",
        0,
        0,
        (PublicText("image"),),
        (
            PublicToolResult(
                "call-image", "read", [{"type": "image", "data": "AAECAwQ=", "mimeType": "image/png"}], False, 1, 0
            ),
        ),
        (),
        (),
        False,
        False,
    )
    toolset = _toolset([image])

    with pytest.raises(EngineError, match="image recovery requires M5 vision support"):
        _call(
            toolset,
            "read_context_output",
            {"historyId": "turn-image", "part": "/toolResults/0/content/0", "mode": "image"},
        )


def test_nested_image_handle_uses_the_stable_m5_error() -> None:
    messages = [
        ModelResponse([ToolCallPart("read", {}, tool_call_id="nested-image")], metadata={DS4_KEY: "nested-turn"}),
        ModelRequest(
            [
                ToolReturnPart(
                    "read",
                    {"nested": BinaryContent(b"IMAGEBYTES", media_type="image/png")},
                    tool_call_id="nested-image",
                )
            ],
            metadata={DS4_KEY: "nested-result"},
        ),
    ]
    toolset = create_context_paging_toolset(is_enabled=lambda: True, history_source=lambda: messages)

    page = _value(
        _call(
            toolset,
            "read_context_output",
            {"historyId": "nested-turn", "part": "/toolResults/0/content", "mode": "text"},
        )
    )
    nested = json.loads(page["content"][1]["text"])["nested"]
    assert nested["read"]["part"] == "/toolResults/0/content/nested"
    assert "data" not in nested
    assert "IMAGEBYTES" not in json.dumps(page)

    with pytest.raises(EngineError, match="image recovery requires M5 vision support"):
        _call(toolset, "read_context_output", nested["read"])


def test_paging_tool_calls_are_excluded_from_real_canonical_history() -> None:
    messages = [
        ModelResponse(
            parts=[ToolCallPart("search_history", {}, tool_call_id="paging")], metadata={DS4_KEY: "paging-turn"}
        ),
        ModelRequest(
            parts=[ToolReturnPart("search_history", {"items": [{"id": "secret"}]}, tool_call_id="paging")],
            metadata={DS4_KEY: "paging-result"},
        ),
        ModelRequest(parts=[UserPromptPart("ordinary work")], metadata={DS4_KEY: "ordinary-user"}),
    ]
    toolset = create_context_paging_toolset(is_enabled=lambda: True, history_source=lambda: messages)

    result = _payload(_call(toolset, "search_history", {"query": "secret"}))
    assert result["references"] == []
    browsed = _payload(_call(toolset, "browse_history", {"direction": "backward"}))
    assert len(browsed["references"]) == 1
    assert browsed["references"][0]["preview"] == "ordinary work"
