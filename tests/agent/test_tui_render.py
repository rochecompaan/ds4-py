"""The footer consumes validated success data and a balanced attempt lifecycle."""

from copy import deepcopy
from typing import Any, cast

import pytest
from test_tui_commands import FakeRuntime

from ds4py.engine.lifecycle import SessionEvent
from ds4py.tui_render import FooterModel


def event(kind: str, **payload: object) -> SessionEvent:
    return SessionEvent(kind, "session", payload)


@pytest.mark.parametrize("initial", ["idle", "running"])
@pytest.mark.parametrize("outcome", ["compacted", "noop", "failed", "cancelled"])
def test_terminal_event_restores_pre_attempt_status(initial: str, outcome: str) -> None:
    footer = FooterModel(cast("Any", FakeRuntime()))
    try:
        if initial == "running":
            footer.handle_event(event("run_start"))
        footer.handle_event(event("before_compact", reason="manual", old_tokens=900))
        if outcome == "compacted":
            footer.handle_event(
                event("compacted", reason="manual", old_tokens=900, new_tokens=200, generated_tokens=13)
            )
            assert footer.ctx_used == 200
            assert footer.custom["compact"] == "900 → 200 tokens, 13 generated"
        assert footer.status == "compacting"
        footer.handle_event(event("compaction_finished", reason="manual", outcome=outcome))
        assert footer.status == initial
        # Duplicate/late delivery cannot restore stale attempt state.
        footer.handle_event(event("run_end"))
        footer.handle_event(event("compaction_finished", reason="manual", outcome=outcome))
        assert footer.status == "idle"
    finally:
        footer.close()


@pytest.mark.parametrize("field", ["old_tokens", "new_tokens", "generated_tokens"])
@pytest.mark.parametrize("invalid", [None, -1, True, "200", 200.0])
def test_malformed_compacted_counts_do_not_mutate_footer(field: str, invalid: object) -> None:
    footer = FooterModel(cast("Any", FakeRuntime()))
    try:
        footer.handle_event(event("before_compact", reason="manual", old_tokens=900))
        payload: dict[str, object] = {"reason": "manual", "old_tokens": 900, "new_tokens": 200, "generated_tokens": 13}
        if invalid is None:
            del payload[field]
        else:
            payload[field] = invalid
        before = deepcopy({key: value for key, value in vars(footer).items() if key != "_unsubscribe"})
        with pytest.raises((TypeError, ValueError)):
            footer.handle_event(event("compacted", **payload))
        assert {key: value for key, value in vars(footer).items() if key != "_unsubscribe"} == before
    finally:
        footer.close()


@pytest.mark.parametrize("invalid", [None, "success", True, 0, [], {}])
def test_malformed_finished_outcome_leaves_attempt_active(invalid: object) -> None:
    footer = FooterModel(cast("Any", FakeRuntime()))
    try:
        footer.handle_event(event("run_start"))
        footer.handle_event(event("before_compact", reason="manual", old_tokens=900))
        payload = {} if invalid is None else {"outcome": invalid}
        with pytest.raises((TypeError, ValueError)):
            footer.handle_event(event("compaction_finished", reason="manual", **payload))
        assert footer.status == "compacting"
        footer.handle_event(event("compaction_finished", reason="manual", outcome="failed"))
        assert footer.status == "running"
    finally:
        footer.close()


@pytest.mark.parametrize("outcome", ["compacted", "noop", "failed", "cancelled"])
def test_run_end_takes_precedence_over_late_attempt_finish(outcome: str) -> None:
    footer = FooterModel(cast("Any", FakeRuntime()))
    try:
        footer.handle_event(event("run_start"))
        footer.handle_event(event("before_compact", reason="threshold", old_tokens=900))
        footer.handle_event(event("run_end"))
        footer.handle_event(event("compaction_finished", reason="threshold", outcome=outcome))
        assert footer.status == "idle"
    finally:
        footer.close()


def test_zero_compacted_counts_are_valid() -> None:
    footer = FooterModel(cast("Any", FakeRuntime()))
    try:
        footer.handle_event(event("compacted", reason="manual", old_tokens=0, new_tokens=0, generated_tokens=0))
        assert footer.ctx_used == 0
        assert footer.custom["compact"] == "0 → 0 tokens, 0 generated"
    finally:
        footer.close()
