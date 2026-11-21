from copy import deepcopy
from typing import cast

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from ds4py.binding_provenance import (
    PreparedRequestViewProvenance,
    RequestViewMessageProvenance,
    _content_hash,
    message_content_hash,
    prepare_request_view_provenance,
    validate_messages,
)
from ds4py.errors import EngineError
from ds4py.message_metadata import DS4_CONTEXT_PAGING, DS4_KEY, DS4_VIEW_OF
from ds4py.store import SegmentProvenance


def _request(text: str, key: str, **metadata: object) -> ModelRequest:
    return ModelRequest([UserPromptPart(text)], metadata={DS4_KEY: key, **metadata})


def _response(text: str, key: str) -> ModelResponse:
    return ModelResponse([TextPart(text)], metadata={DS4_KEY: key})


def _segment(message: ModelRequest | ModelResponse, start: int, kind: str = "user") -> SegmentProvenance:
    metadata = cast("dict[str, object]", message.metadata)
    return SegmentProvenance(start, start + 1, kind, cast("str", metadata[DS4_KEY]), _content_hash(message), "test")


def _valid_input() -> tuple[
    list[ModelRequest | ModelResponse], list[ModelRequest | ModelResponse], list[SegmentProvenance]
]:
    canonical = [
        _request("old request", "old"),
        _response("omitted answer", "answer"),
        _request("live request", "live"),
    ]
    notice = _request("paging notice", "notice-view", **{DS4_CONTEXT_PAGING: "notice"})
    retained = deepcopy(canonical[0])
    replacement = _request(
        "short live request",
        "live-view",
        **{DS4_CONTEXT_PAGING: "recovery", DS4_VIEW_OF: "live"},
    )
    request_view = [notice, retained, replacement]
    segments = [
        _segment(notice, 0, "system"),
        _segment(retained, 1),
        _segment(replacement, 2),
    ]
    return canonical, request_view, segments


def _prepare(
    canonical: list[ModelRequest | ModelResponse],
    request_view: list[ModelRequest | ModelResponse],
    segments: list[SegmentProvenance],
) -> PreparedRequestViewProvenance:
    return prepare_request_view_provenance(
        canonical_messages=canonical,
        request_view=request_view,
        segments=segments,
        loaded=len(segments),
    )


def test_prepare_request_view_validates_segments_and_builds_immutable_mapping() -> None:
    canonical, request_view, segments = _valid_input()

    prepared = _prepare(canonical, request_view, segments)

    assert tuple(record.key for record in prepared.prepared.records) == ("notice-view", "old", "live-view")
    assert prepared.prepared.bound_messages == tuple(request_view)
    assert prepared.messages == (
        RequestViewMessageProvenance("notice-view", None, "notice"),
        RequestViewMessageProvenance("old", "old", None),
        RequestViewMessageProvenance("live-view", "live", "recovery"),
    )
    assert canonical[1].metadata == {DS4_KEY: "answer"}


def test_public_hash_and_validator_are_canonical_operations() -> None:
    message = _request("content", "key")

    assert message_content_hash(message) == _content_hash(message)
    assert message_content_hash(message) != message_content_hash(_request("changed", "key"))
    assert validate_messages([message]) is None


def test_arbitrary_additional_canonical_messages_are_valid() -> None:
    canonical, request_view, segments = _valid_input()
    canonical.insert(0, _request("separate old request", "separate-old"))
    canonical.append(_response("future canonical answer", "future-answer"))

    prepared = _prepare(canonical, request_view, segments)

    assert [item.canonical_key for item in prepared.messages] == [None, "old", "live"]


def test_duplicate_view_key_fails_atomically() -> None:
    canonical, request_view, segments = _valid_input()
    original = deepcopy(request_view)
    request_view[2].metadata = {DS4_KEY: "old", DS4_CONTEXT_PAGING: "recovery", DS4_VIEW_OF: "live"}

    with pytest.raises(EngineError, match="duplicate message key"):
        _prepare(canonical, request_view, segments)

    assert request_view[0] == original[0]
    assert request_view[1] == original[1]


def test_out_of_order_segment_keys_fail() -> None:
    canonical, request_view, segments = _valid_input()
    request_view[1], request_view[2] = request_view[2], request_view[1]

    with pytest.raises(EngineError, match="out of snapshot order"):
        _prepare(canonical, request_view, segments)


def test_mismatched_segment_hash_fails() -> None:
    canonical, request_view, segments = _valid_input()
    segments[1] = SegmentProvenance(1, 2, "user", "old", "0" * 64, "test")

    with pytest.raises(EngineError, match=r"content hash.*does not match"):
        _prepare(canonical, request_view, segments)


def test_keyed_segment_must_resolve_to_one_view_message() -> None:
    canonical, request_view, segments = _valid_input()
    del request_view[1]

    with pytest.raises(EngineError, match="absent from the snapshot"):
        _prepare(canonical, request_view, segments)


@pytest.mark.parametrize("marker", [None, "", "future", 1])
def test_malformed_paging_markers_fail(marker: object) -> None:
    canonical, request_view, segments = _valid_input()
    request_view[0].metadata = {DS4_KEY: "notice-view", DS4_CONTEXT_PAGING: marker}

    with pytest.raises(EngineError, match="paging marker"):
        _prepare(canonical, request_view, segments)


def test_notice_cannot_have_a_canonical_target() -> None:
    canonical, request_view, segments = _valid_input()
    request_view[0].metadata = {DS4_KEY: "notice-view", DS4_CONTEXT_PAGING: "notice", DS4_VIEW_OF: "old"}

    with pytest.raises(EngineError, match=r"notice.*target"):
        _prepare(canonical, request_view, segments)


@pytest.mark.parametrize("target", [None, "", "missing", 1])
def test_recovery_requires_an_existing_canonical_target(target: object) -> None:
    canonical, request_view, segments = _valid_input()
    metadata = {DS4_KEY: "live-view", DS4_CONTEXT_PAGING: "recovery"}
    if target is not None:
        metadata[DS4_VIEW_OF] = target
    request_view[2].metadata = metadata

    with pytest.raises(EngineError, match=r"recovery.*target"):
        _prepare(canonical, request_view, segments)


def test_replacement_requires_a_fresh_view_only_key() -> None:
    canonical, request_view, segments = _valid_input()
    request_view[2].metadata = {DS4_KEY: "live", DS4_CONTEXT_PAGING: "recovery", DS4_VIEW_OF: "live"}
    segments[2] = _segment(request_view[2], 2)

    with pytest.raises(EngineError, match="fresh view key"):
        _prepare(canonical, request_view, segments)


def test_unmarked_message_must_have_the_same_canonical_key_and_hash() -> None:
    canonical, request_view, segments = _valid_input()
    request_view[1] = _request("tampered", "old")
    segments[1] = _segment(request_view[1], 1)

    with pytest.raises(EngineError, match=r"unmarked view message.*canonical"):
        _prepare(canonical, request_view, segments)


def test_unmarked_message_cannot_claim_a_replacement_target() -> None:
    canonical, request_view, segments = _valid_input()
    request_view[1].metadata = {DS4_KEY: "old", DS4_VIEW_OF: "old"}

    with pytest.raises(EngineError, match=r"unmarked view message.*target"):
        _prepare(canonical, request_view, segments)


def test_duplicate_canonical_keys_fail_independently_of_segments() -> None:
    canonical, request_view, segments = _valid_input()
    canonical[1].metadata = {DS4_KEY: "old"}

    with pytest.raises(EngineError, match=r"canonical.*duplicate message key"):
        _prepare(canonical, request_view, segments)
