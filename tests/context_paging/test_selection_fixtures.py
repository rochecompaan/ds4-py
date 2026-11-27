import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from test_history import _message, _timestamp

from ds4py.binding_plan import BindingChange
from ds4py.binding_projection import RequestProjection
from ds4py.context_paging.history import HistoryProjection, project_history
from ds4py.context_paging.notice import PagingNotice
from ds4py.context_paging.selection import (
    ContextCutFrontier,
    ContextCutKey,
    RollingWindowSettings,
    SelectionResult,
    SelectionState,
    select_view,
)
from ds4py.context_paging.text_units import utf16_length
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY, DS4_TIMESTAMP

FIXTURES = json.loads((Path(__file__).parents[1] / "fixtures/context_paging/selection.json").read_text())["fixtures"]


def _signature(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class FixtureMeter:
    def __init__(self, input_value: dict[str, Any], raw_costs: dict[int, int]) -> None:
        self.resident = input_value["tokenEstimates"]["residentTokens"]
        self.raw_costs = raw_costs

    @staticmethod
    def _generated_cost(message: ModelMessage) -> int:
        characters = 0
        for part in message.parts:
            content = part.content if isinstance(part, (UserPromptPart, ToolReturnPart)) else ""
            if isinstance(content, str):
                characters += utf16_length(content)
            elif isinstance(content, list):
                characters += sum(utf16_length(child) for child in content if isinstance(child, str))
        return (characters + 3) // 4

    def measure(
        self,
        *,
        canonical_messages: Sequence[ModelMessage],
        candidate_messages: Sequence[ModelMessage],
    ) -> RequestProjection:
        del canonical_messages
        costs = [self.raw_costs.get(id(message)) or self._generated_cost(message) for message in candidate_messages]
        total = self.resident + sum(costs)
        notice_tokens = sum(
            cost
            for message, cost in zip(candidate_messages, costs, strict=True)
            if (message.metadata or {}).get(DS4_CONTEXT_PAGING) == "notice"
        )
        return RequestProjection(
            tuple(range(total)),
            total,
            BindingChange("append", 0, False),
            self.resident,
            notice_tokens,
            (),
        )


def _history(input_value: dict[str, Any]) -> tuple[HistoryProjection, FixtureMeter, dict[int, dict[str, Any]]]:
    identifiers: dict[str, str] = {}
    timestamps: dict[str, str] = {}
    for item in input_value.get("rawHistoryItems", []):
        message = item["userMessage"] if item["kind"] == "user" else item["assistantMessage"]
        identifiers[_signature(message)] = item["id"]
        timestamps[_signature(message)] = item["timestamp"]
    messages: list[ModelMessage] = []
    raw_by_id: dict[int, dict[str, Any]] = {}
    raw_costs: dict[int, int] = {}
    keyless: list[ModelRequest] = []
    for index, raw in enumerate(input_value["messages"]):
        signature = _signature(raw)
        if raw["role"] == "custom":
            message: ModelMessage = ModelRequest(
                [UserPromptPart(raw["content"], timestamp=_timestamp(raw["timestamp"]))],
                metadata={DS4_KEY: f"keyless-{index}", DS4_TIMESTAMP: "2026-10-19T00:00:00.000Z"},
                timestamp=_timestamp(raw["timestamp"]),
            )
            keyless.append(message)
        else:
            key = identifiers.get(signature, f"message-{index}")
            message = _message(
                {
                    "id": key,
                    "timestamp": timestamps.get(signature, "2026-10-19T00:00:00.000Z"),
                    "message": raw,
                }
            )
        messages.append(message)
        raw_by_id[id(message)] = raw
        raw_costs[id(message)] = input_value["tokenEstimates"]["messageTokens"][index]
    history = (
        project_history(messages) if input_value.get("rawHistoryItems") else HistoryProjection(tuple(messages), ())
    )
    for message in keyless:
        message.metadata = {}
    if keyless:
        history = replace(
            history,
            items=tuple(
                replace(item, history_id="") if hasattr(item, "message") and item.message in keyless else item
                for item in history.items
            ),
        )
    return history, FixtureMeter(input_value, raw_costs), raw_by_id


def _raw_message(message: ModelMessage, raw_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if id(message) in raw_by_id:
        return deepcopy(raw_by_id[id(message)])
    metadata = message.metadata or {}
    if (
        isinstance(message, ModelRequest)
        and isinstance(message.parts[0], UserPromptPart)
        and metadata.get(DS4_CONTEXT_PAGING) in {"notice", "recovery"}
    ):
        part = message.parts[0]
        assert isinstance(part.content, str)
        return {"role": "user", "content": part.content, "timestamp": 0}
    tool_parts = [part for part in message.parts if isinstance(part, ToolReturnPart)]
    if tool_parts:
        raw = deepcopy(
            next(
                raw
                for raw in raw_by_id.values()
                if raw["role"] == "toolResult" and raw["toolCallId"] == tool_parts[0].tool_call_id
            )
        )
        raw["content"] = [
            {"type": "text", "text": child}
            for part in tool_parts
            for child in (part.content if isinstance(part.content, list) else [part.content])
            if isinstance(child, str)
        ]
        return raw
    raise AssertionError(f"Unexpected generated fixture message: {message!r}")


def _notice_value(content: str) -> dict[str, object]:
    return {"role": "user", "content": content, "timestamp": 0}


def _result_value(
    result: SelectionResult,
    raw_by_id: dict[int, dict[str, Any]],
) -> dict[str, object]:
    value: dict[str, object] = {
        "messages": [_raw_message(message, raw_by_id) for message in result.candidate_messages],
        "estimatedTokens": result.measurement.total_tokens,
        "budgetTokens": result.effective_budget,
        "mode": result.mode,
    }
    if result.state is not None:
        last_evicted: dict[str, object] = {"historyId": result.state.frontier.last_evicted.history_id}
        if result.state.frontier.last_evicted.tool_call_id is not None:
            last_evicted["toolCallId"] = result.state.frontier.last_evicted.tool_call_id
        frontier: dict[str, object] = {"kind": result.state.frontier.kind, "lastEvicted": last_evicted}
        if result.state.frontier.user_history_id is not None:
            frontier["userHistoryId"] = result.state.frontier.user_history_id
        value["cutState"] = {
            "frontier": frontier,
            "notice": _notice_value(result.state.notice.content) if result.state.notice is not None else None,
        }
    if result.fallback_reason is not None:
        value["cutFallbackReason"] = result.fallback_reason
    return value


def _remembered(input_value: dict[str, Any]) -> SelectionState:
    raw = input_value["cutState"]
    frontier = raw["frontier"]
    last_evicted = frontier["lastEvicted"]
    return SelectionState(
        ContextCutFrontier(
            frontier["kind"],
            ContextCutKey(last_evicted["historyId"], last_evicted.get("toolCallId")),
            frontier.get("userHistoryId"),
        ),
        PagingNotice(raw["notice"]["content"], "fixture-replay") if raw["notice"] is not None else None,
    )


def _settings(input_value: dict[str, Any]) -> RollingWindowSettings:
    return RollingWindowSettings(
        enabled=True,
        token_budget=input_value["tokenBudget"],
        trim_to=input_value["trimToTokens"],
        trim_to_explicit=True,
    )


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture["id"])
def test_generated_selection_contract(fixture: dict[str, Any]) -> None:
    input_value = fixture["input"]
    history, meter, raw_by_id = _history(input_value)
    result = select_view(
        history,
        settings=_settings(input_value),
        model_context=input_value["modelContextWindow"],
        meter=meter,
    )

    assert _result_value(result, raw_by_id) == fixture["expected"]["value"]

    if "replay" in fixture:
        replay = fixture["replay"]
        replay_input = replay["input"]
        replay_history, replay_meter, replay_raw_by_id = _history(replay_input)
        replayed = select_view(
            replay_history,
            settings=_settings(replay_input),
            model_context=replay_input["modelContextWindow"],
            meter=replay_meter,
            remembered=_remembered(replay_input),
        )
        actual_replay = _result_value(replayed, replay_raw_by_id)
        expected_replay = deepcopy(replay["expected"]["value"])
        # Pi replays an additive estimate; ds4 measures the exact rebuilt request.
        actual_replay.pop("estimatedTokens")
        expected_replay.pop("estimatedTokens")
        assert actual_replay == expected_replay
