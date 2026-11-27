import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturn, ToolReturnPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool
from pydantic_ai.usage import RunUsage

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
