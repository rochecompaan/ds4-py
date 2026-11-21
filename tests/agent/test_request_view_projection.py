from copy import deepcopy
from typing import Any, cast

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from test_binding import fake_engine

from ds4py.binding_provenance import message_content_hash
from ds4py.capabilities.system_notes import DS4SystemNotes
from ds4py.engine.prompt import PromptBuilder
from ds4py.errors import EngineError
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY, DS4_VIEW_OF


def _user(text: str, key: str, **metadata: object) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)], metadata={DS4_KEY: key, **metadata})


def _response(text: str, key: str) -> ModelResponse:
    return ModelResponse([TextPart(text)], metadata={DS4_KEY: key})


def _notice(text: str = "Earlier history is paged.", key: str = "notice-view") -> ModelRequest:
    return ModelRequest(
        [SystemPromptPart(text)],
        metadata={DS4_KEY: key, DS4_CONTEXT_PAGING: "notice"},
    )


def _project(binder: Any, canonical: list[ModelMessage], candidate: list[ModelMessage]) -> Any:
    return binder.project_request_view(
        canonical_messages=canonical,
        candidate_messages=candidate,
        params=ModelRequestParameters(),
    )


def _prepared_tokens(binder: Any, messages: list[ModelMessage]) -> tuple[int, ...]:
    binder.prepare_model_request(messages, ModelRequestParameters())
    session = binder.session
    prefix = PromptBuilder(session.engine).assistant_prefix(session.transcript.tokens)
    return cast("tuple[int, ...]", (*session.transcript.tokens, *prefix))


def test_exact_tokens_and_unit_spans_match_the_following_prepared_request() -> None:
    messages: list[ModelMessage] = [_user("first", "u1"), _response("answer", "r1"), _user("next", "u2")]
    with fake_engine() as engine:
        from ds4py.binding import MessageBinder

        session = engine.session()
        binder = MessageBinder(session)
        candidate = deepcopy(messages)

        projection = _project(binder, messages, candidate)
        plan = session.plan(projection.tokens)
        expected = _prepared_tokens(binder, candidate)

        assert projection.tokens == expected
        assert projection.total_tokens == len(expected)
        assert plan.to_prefill == len(expected)
        assert plan.kind == "rebuild"
        assert projection.change.kind == "append"
        assert projection.notice_tokens == 0
        assert [(unit.unit_key, unit.request_keys, unit.response_key) for unit in projection.units] == [
            ("r1", ("u1",), "r1"),
            ("u2", ("u2",), None),
        ]
        assert all(0 <= unit.start < unit.end <= len(session.transcript) for unit in projection.units)
        assistant_prefix = len(expected) - len(session.transcript)
        covered = projection.resident_tokens + sum(unit.end - unit.start for unit in projection.units)
        assert covered + assistant_prefix == projection.total_tokens
        assert candidate == messages


def test_notice_and_replacement_costs_remain_exact_and_separate() -> None:
    canonical: list[ModelMessage] = [_user("old", "u1"), _response("old answer", "r1"), _user("live request", "u2")]
    replacement = _user(
        "short",
        "u2-view",
        **{DS4_CONTEXT_PAGING: "recovery", DS4_VIEW_OF: "u2"},
    )
    with fake_engine() as engine:
        from ds4py.binding import MessageBinder

        binder = MessageBinder(engine.session())
        with_notice = _project(binder, canonical, [_notice(), deepcopy(replacement)])
        without_notice = _project(binder, canonical, [deepcopy(replacement)])

        assert with_notice.notice_tokens > 0
        assert without_notice.notice_tokens == 0
        assert with_notice.total_tokens - without_notice.total_tokens == with_notice.notice_tokens
        assert len(with_notice.units) == 1
        assert with_notice.units[0].unit_key == "u2"
        assert with_notice.units[0].request_keys == ("u2",)
        assert with_notice.units[0].response_key is None
        assert with_notice.units[0].end - with_notice.units[0].start > 0
        assert with_notice.tokens != without_notice.tokens


def test_removing_a_completed_unit_removes_exactly_its_own_cost() -> None:
    canonical: list[ModelMessage] = [_user("old request", "u1"), _response("old answer", "r1"), _user("live", "u2")]
    with fake_engine() as engine:
        from ds4py.binding import MessageBinder

        binder = MessageBinder(engine.session())
        full = _project(binder, canonical, deepcopy(canonical))
        paged = _project(binder, canonical, [_notice(), deepcopy(canonical[2])])

        removed = full.units[0]
        assert full.total_tokens - paged.total_tokens + paged.notice_tokens == removed.end - removed.start
        assert paged.units[0].unit_key == "u2"
        assert paged.units[0].end - paged.units[0].start > 0


def test_incremental_edit_rewrites_only_the_affected_unit() -> None:
    original: list[ModelMessage] = [_user("first", "u1"), _response("same answer", "r1"), _user("next", "u2")]
    with fake_engine() as engine:
        from ds4py.binding import MessageBinder

        binder = MessageBinder(engine.session())
        prepared = deepcopy(original)
        binder.prepare_model_request(prepared, ModelRequestParameters())
        baseline = _project(binder, original, deepcopy(original))
        changed = deepcopy(original)
        cast("UserPromptPart", changed[2].parts[0]).content = "a considerably longer next request"

        projection = _project(binder, changed, deepcopy(changed))

        assert baseline.change.kind == "unchanged"
        assert projection.change.kind == "rewrite"
        assert projection.change.start == baseline.units[-1].start
        assert projection.units[0] == baseline.units[0]
        assert projection.units[-1].unit_key == "u2"
        assert projection.tokens[: projection.change.start] == baseline.tokens[: projection.change.start]


def test_note_preview_recomputes_without_applying_live_notes_or_keys() -> None:
    with fake_engine() as engine:
        from ds4py.binding import MessageBinder

        session = engine.session()
        binder = MessageBinder(session)
        notes = DS4SystemNotes(session, hints=False, clock=lambda: "first clock value")
        binder.note_provider = notes
        canonical: list[ModelMessage] = [_user("request", "u1")]
        before = deepcopy(canonical)
        native_before = session.transcript.tokens

        first = _project(binder, canonical, deepcopy(canonical))
        notes.clock = lambda: "a much longer second clock value that changes tokenization"
        notes.hints_enabled = True
        second = _project(binder, canonical, deepcopy(canonical))

        assert first.tokens != second.tokens
        assert first.total_tokens != second.total_tokens
        assert notes._date_emitted is False
        assert notes._hints_full_emitted is False
        assert session.transcript.tokens == native_before
        assert canonical == before


def test_failed_projection_publishes_no_keys_notes_or_native_changes() -> None:
    with fake_engine() as engine:
        from ds4py.binding import MessageBinder

        session = engine.session()
        binder = MessageBinder(session)

        def fail_clock() -> str:
            raise ValueError("clock failed")

        notes = DS4SystemNotes(session, hints=True, clock=fail_clock)
        binder.note_provider = notes
        messages: list[ModelMessage] = [ModelRequest([UserPromptPart("request")])]

        with pytest.raises(ValueError, match="clock failed"):
            _project(binder, messages, messages)

        assert messages[0].metadata is None
        assert binder.records() == ()
        assert session.transcript.tokens == ()
        assert notes._date_emitted is False
        assert notes._hints_full_emitted is False


def test_replacement_target_and_unmarked_hash_use_canonical_provenance() -> None:
    canonical: list[ModelMessage] = [_user("original", "u1")]
    replacement = _user("short", "view", **{DS4_CONTEXT_PAGING: "recovery", DS4_VIEW_OF: "u1"})
    with fake_engine() as engine:
        from ds4py.binding import MessageBinder

        binder = MessageBinder(engine.session())
        projection = _project(binder, canonical, [replacement])

        assert projection.units[0].unit_key == "u1"
        assert message_content_hash(canonical[0]) != message_content_hash(replacement)

        invalid = _user("changed", "u1")
        with pytest.raises(EngineError, match="canonical content"):
            _project(binder, canonical, [invalid])
