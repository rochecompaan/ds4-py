from collections.abc import Sequence
from dataclasses import replace

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ds4py.binding_plan import BindingChange
from ds4py.binding_projection import RequestProjection
from ds4py.context_paging.history import HistoryProjection, UserHistoryItem, project_history
from ds4py.context_paging.selection import (
    ContextCutFrontier,
    ContextCutKey,
    ContextSelectionError,
    RollingWindowSettings,
    SelectionResult,
    SelectionState,
    select_view,
)
from ds4py.context_paging.settings import default_trim_to
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY


class AdditiveMeter:
    def __init__(self, resident: int = 10, *, notice_cost: int = 5, dynamic_notice: bool = False) -> None:
        self.resident = resident
        self.notice_cost = notice_cost
        self.dynamic_notice = dynamic_notice

    def measure(
        self,
        *,
        canonical_messages: Sequence[ModelMessage],
        candidate_messages: Sequence[ModelMessage],
    ) -> RequestProjection:
        del canonical_messages
        costs = []
        notice_tokens = 0
        for message in candidate_messages:
            metadata = message.metadata or {}
            if metadata.get(DS4_CONTEXT_PAGING) == "notice":
                part = message.parts[0]
                assert isinstance(part, SystemPromptPart)
                notice_tokens = len(part.content) // 4 if self.dynamic_notice else self.notice_cost
                costs.append(notice_tokens)
            elif metadata.get(DS4_CONTEXT_PAGING) == "recovery":
                costs.append(5)
            else:
                costs.append(int(metadata.get("test.cost", 0)))
        total = self.resident + sum(costs)
        return RequestProjection(
            tuple(range(total)),
            total,
            BindingChange("append", 0, False),
            self.resident,
            notice_tokens,
            (),
        )


def user(key: str, cost: int, text: str | None = None) -> ModelRequest:
    return ModelRequest([UserPromptPart(text or key)], metadata={DS4_KEY: key, "test.cost": cost})


def response(key: str, cost: int, call: str | None = None) -> ModelResponse:
    parts = [ToolCallPart("read", {}, tool_call_id=call)] if call is not None else [TextPart(key)]
    return ModelResponse(parts, metadata={DS4_KEY: key, "test.cost": cost})


def result(key: str, call: str, cost: int) -> ModelRequest:
    return ModelRequest(
        [ToolReturnPart("read", key, tool_call_id=call)],
        metadata={DS4_KEY: key, "test.cost": cost},
    )


def make_settings(budget: int, target: int | None) -> RollingWindowSettings:
    trim_to = default_trim_to(budget) if target is None and budget > 0 else target
    try:
        return RollingWindowSettings(
            enabled=True,
            token_budget=budget,
            trim_to=trim_to or 0,
            trim_to_explicit=target is not None,
        )
    except ValueError:
        settings = object.__new__(RollingWindowSettings)
        object.__setattr__(settings, "enabled", True)
        object.__setattr__(settings, "token_budget", budget)
        object.__setattr__(settings, "trim_to", trim_to or 0)
        object.__setattr__(settings, "trim_to_explicit", target is not None)
        return settings


def select(
    messages: list[ModelMessage],
    *,
    budget: int = 100,
    target: int | None = 50,
    model: int = 1000,
    meter: AdditiveMeter | None = None,
    remembered: SelectionState | None = None,
) -> SelectionResult:
    return select_view(
        project_history(messages),
        settings=make_settings(budget, target),
        model_context=model,
        meter=meter or AdditiveMeter(),
        remembered=remembered,
    )


def keys(messages: tuple[ModelMessage, ...]) -> list[str]:
    return [str((message.metadata or {}).get(DS4_KEY)) for message in messages]


def test_equality_and_values_between_target_and_budget_do_not_advance() -> None:
    for final_cost, expected in [(50, "within-budget"), (70, "within-budget"), (90, "within-budget")]:
        selected = select([user("u1", 0), response("r1", final_cost)], budget=100, target=50)
        assert selected.mode == expected
        assert selected.state is None
        assert keys(selected.candidate_messages) == ["u1", "r1"]


def test_reserve_recovery_advances_a_remembered_view_at_its_target() -> None:
    messages = [
        user("evicted-user", 5),
        response("evicted-response", 5),
        user("retained-user", 15),
        response("retained-response", 15),
        user("active-user", 5),
    ]
    initial = select(messages, budget=50, target=50)
    assert initial.state is not None
    assert initial.state.frontier.last_evicted.history_id == "evicted-response"
    assert keys(initial.candidate_messages) == [
        "context-paging-notice",
        "retained-user",
        "retained-response",
        "active-user",
    ]

    recovered = select_view(
        project_history(messages),
        settings=make_settings(50, 50),
        model_context=1000,
        meter=AdditiveMeter(),
        remembered=initial.state,
        overflow="recovery",
    )

    assert recovered.frontier is not None
    assert recovered.frontier.last_evicted.history_id == "retained-response"
    assert keys(recovered.candidate_messages) == ["context-paging-notice", "active-user"]


def test_fifo_protects_active_request_and_newest_unread_exchange() -> None:
    messages = [
        response("prefix", 400),
        user("old-user", 400),
        response("old-turn", 400),
        user("active-user", 10),
        response("old-active", 10, "old-call"),
        result("old-result", "old-call", 10),
        response("newest", 30, "new-call"),
        result("new-result", "new-call", 20),
    ]

    selected = select(messages, budget=1000, target=50)

    assert selected.mode == "paged"
    assert keys(selected.candidate_messages) == ["context-paging-notice", "active-user", "newest", "new-result"]
    assert selected.state is not None
    assert selected.state.frontier.kind == "partialTurn"
    assert selected.state.frontier.last_evicted.history_id == "old-active"
    assert selected.state.notice is not None
    assert '--- user message, historyId "old-user" ---' in selected.state.notice.content
    assert '--- user message, historyId "active-user" ---' not in selected.state.notice.content
    assert selected.measurement.total_tokens == 75


def test_partial_completed_remainder_is_atomic_on_a_later_cut() -> None:
    initial = [
        user("active-user", 10),
        response("first", 30, "c1"),
        result("first-result", "c1", 10),
        response("second", 20, "c2"),
        result("second-result", "c2", 10),
    ]
    first = select(initial, budget=80, target=50)
    assert first.state is not None
    assert first.state.frontier.kind == "partialTurn"

    later = [*initial, user("new-active", 10)]
    frozen = select(later, budget=100, target=50, remembered=first.state)
    assert keys(frozen.candidate_messages) == [
        "context-paging-notice",
        "active-user",
        "second",
        "second-result",
        "new-active",
    ]

    evicted = select(later, budget=60, target=30, remembered=first.state)
    assert keys(evicted.candidate_messages) == ["context-paging-notice", "new-active"]
    assert evicted.state is not None
    assert evicted.state.frontier.kind == "completedTurn"


def test_keyless_ending_snaps_back_to_last_safe_keyed_boundary() -> None:
    projection = project_history([response("p1", 30), response("p2", 30), user("active", 20)])
    items = list(projection.items)
    items[1] = replace(items[1], history_id="")
    projection = HistoryProjection(projection.canonical_messages, tuple(items))

    selected = select_view(
        projection,
        settings=make_settings(80, 30),
        model_context=1000,
        meter=AdditiveMeter(),
    )

    assert keys(selected.candidate_messages) == ["context-paging-notice", "p2", "active"]
    assert selected.state is not None
    assert selected.state.frontier.last_evicted.history_id == "p1"
    assert selected.fallback_reason is None


def test_smaller_later_costs_do_not_reopen_a_committed_cut() -> None:
    messages = [response("prefix", 40), user("active", 30)]
    first = select(messages, budget=70, target=40)
    assert first.state is not None

    second = select(messages, budget=200, target=100, remembered=first.state)

    assert keys(second.candidate_messages) == ["context-paging-notice", "active"]
    assert second.state is not None
    assert second.state.frontier == first.state.frontier
    assert second.state.notice == first.state.notice


def test_unresolved_remembered_key_uses_stateless_fifo_fallback() -> None:
    remembered = SelectionState(ContextCutFrontier("prefix", ContextCutKey("missing")), None)

    selected = select([response("p1", 50), user("active", 40)], budget=80, target=50, remembered=remembered)

    assert selected.fallback_reason == "cut-key-unresolved"
    assert selected.state is None
    assert keys(selected.candidate_messages) == ["context-paging-notice", "active"]


@pytest.mark.parametrize(
    ("budget", "target", "model", "code"),
    [(0, 50, 100, "INVALID_TOKEN_BUDGET"), (100, 0, 100, "INVALID_TRIM_TARGET"), (100, 50, 0, "INVALID_MODEL_CONTEXT")],
)
def test_invalid_limits_keep_baseline_codes(budget: int, target: int, model: int, code: str) -> None:
    with pytest.raises(ContextSelectionError) as caught:
        select([user("u", 1)], budget=budget, target=target, model=model)

    assert caught.value.code == code
    assert str(caught.value).startswith(f"[{code}] ")


def test_oversized_resident_and_active_request_have_distinct_errors() -> None:
    with pytest.raises(ContextSelectionError) as resident:
        select([user("u", 1)], budget=20, meter=AdditiveMeter(resident=21))
    assert resident.value.code == "RESIDENT_INPUT_TOO_LARGE"

    with pytest.raises(ContextSelectionError) as active:
        select([response("prefix", 20), user("active", 90)], budget=100)
    assert active.value.code == "ACTIVE_REQUEST_TOO_LARGE"


def test_protected_overflow_can_fit_above_normal_budget() -> None:
    messages = [user("active", 10), response("turn", 20, "call"), result("result", "call", 110)]

    selected = select(messages, budget=80, target=40, model=200)

    assert selected.mode == "protected-overflow"
    assert selected.measurement.total_tokens == 155
    assert keys(selected.candidate_messages) == ["context-paging-notice", "active", "turn", "result"]
    assert selected.state is None


def test_oversized_protected_result_uses_recovery_without_fabricating_a_cut() -> None:
    messages = [user("active", 10), response("turn", 20, "call"), result("result", "call", 110)]

    selected = select(messages, budget=80, target=40, model=150)

    assert selected.mode == "recovery"
    assert selected.state is None
    assert keys(selected.candidate_messages) == ["context-paging-notice", "active", "turn", "result:recovery"]
    recovered = selected.candidate_messages[-1]
    assert recovered.metadata is not None
    assert recovered.metadata[DS4_CONTEXT_PAGING] == "recovery"
    part = recovered.parts[0]
    assert isinstance(part, ToolReturnPart)
    assert "read_context_output" in str(part.content)
    assert "historyId" in str(part.content)


def test_remembered_notice_is_resized_when_the_active_request_grows() -> None:
    messages = [
        user("old-user", 3000, "x" * 1800),
        response("old-turn", 20),
        user("active", 10),
    ]
    meter = AdditiveMeter(dynamic_notice=True)
    first = select(messages, budget=3000, target=200, model=3000, meter=meter)
    assert first.state is not None
    assert first.state.notice is not None

    later = [*messages[:2], user("active", 2800)]
    resized = select(later, budget=3000, target=200, model=3000, meter=meter, remembered=first.state)

    assert resized.measurement.total_tokens <= 3000
    assert resized.state is not None
    assert resized.state.frontier == first.state.frontier
    assert resized.state.notice != first.state.notice
    assert resized.state.notice is not None
    assert len(resized.state.notice.content) < len(first.state.notice.content)


def test_explicit_target_at_budget_trims_to_budget_and_warns_once_per_pair() -> None:
    messages = [response("prefix", 50), user("active", 40)]
    settings = make_settings(80, 80)

    first = select_view(project_history(messages), settings=settings, model_context=1000, meter=AdditiveMeter())
    second = select_view(project_history(messages), settings=settings, model_context=1000, meter=AdditiveMeter())

    assert first.effective_target == 80
    assert first.warning is not None
    assert second.warning is None
    assert first.mode == "paged"


def test_selection_failure_does_not_mutate_history_or_commit_state() -> None:
    messages = [user("active", 100)]
    projection = project_history(messages)
    before = list(messages)

    with pytest.raises(ContextSelectionError):
        select_view(
            projection,
            settings=make_settings(50, None),
            model_context=1000,
            meter=AdditiveMeter(),
        )

    assert messages == before
    item = projection.items[0]
    assert isinstance(item, UserHistoryItem)
    assert item.message is messages[0]
