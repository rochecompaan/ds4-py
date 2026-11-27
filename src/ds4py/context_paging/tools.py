"""Strict Pydantic tools for public paged-history recovery."""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Literal, cast

from pydantic import Field
from pydantic_ai import Tool
from pydantic_ai.messages import ModelMessage, ToolReturn
from pydantic_ai.toolsets import FunctionToolset

from ds4py.context_paging.history import HistoryProjection, project_history
from ds4py.context_paging.navigator import HistoryNavigator, HistoryReference
from ds4py.context_paging.public_history import PublicHistoryItem, PublicUser, project_public_history
from ds4py.context_paging.recovery_parts import (
    HistoryPartDescription,
    RecoveryInputError,
    context_output_texts,
    describe_history_part,
    history_parts,
    is_recovery_image,
    public_history_item,
    public_item_value,
    select_history_part,
    serialize_recovery_part,
)
from ds4py.context_paging.recovery_parts import (
    read_context_output as read_output_page,
)
from ds4py.context_paging.recovery_response import (
    MAXIMUM_HISTORY_ID_CHARACTERS,
    MAXIMUM_INLINE_PART_BYTES,
    MAXIMUM_PAGE_LIMIT,
    MAXIMUM_PART_CHARACTERS,
    RecoveryTextReply,
    compact_json,
    fits_recovery_response,
    recovery_response,
    recovery_text_reply,
)
from ds4py.context_paging.text_units import utf16_length
from ds4py.errors import EngineError

_DISABLED = "Context paging recovery tools are disabled."


def _tool_return(reply: RecoveryTextReply) -> ToolReturn:
    return ToolReturn(return_value=reply)


def _contains_image(value: object) -> bool:
    if is_recovery_image(value):
        return True
    if isinstance(value, Mapping):
        return any(_contains_image(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_image(child) for child in value)
    return False


def _deferred_item(item: PublicHistoryItem) -> dict[str, object]:
    return {
        "id": item.history_id,
        "kind": "user" if isinstance(item, PublicUser) else "modelTurn",
        "deferred": True,
        "parts": [],
        "totalParts": len(history_parts(item)),
        "nextPartOffset": 0,
        "readParts": {"historyId": item.history_id, "mode": "parts", "offset": 0},
    }


def bounded_history_reply(  # noqa: PLR0912 - Whole-item fitting and deferred continuation have separate branches.
    items: Sequence[PublicHistoryItem] | None = None,
    references: Sequence[HistoryReference] | None = None,
) -> ToolReturn:
    """Fit whole public items smallest-first while retaining request order and handles."""
    loaded: list[dict[str, object]] = [_deferred_item(item) for item in items or ()]
    compact: list[HistoryReference | dict[str, object]] = [dict(reference) for reference in references or ()]
    payload: dict[str, object] = {}
    if references is not None:
        payload["references"] = compact
    if items is not None:
        payload["items"] = loaded
    if items is not None and references is not None:
        payload["unloadedHistoryIds"] = [reference["historyId"] for reference in references[len(items) :]]

    largest_first = sorted(
        ((len(compact_json(reference).encode("utf-8")), index) for index, reference in enumerate(compact)),
        reverse=True,
    )
    for _, index in largest_first:
        if fits_recovery_response(payload):
            break
        compact[index] = {"historyId": compact[index]["historyId"], "detailsDeferred": True}
    if not fits_recovery_response(payload):
        return _tool_return(recovery_response(payload))

    candidates = [
        (len(serialize_recovery_part(item, "").encode("utf-8")), index, item)
        for index, item in enumerate(items or ())
        if not _contains_image(public_item_value(item))
    ]
    for _, index, item in sorted(candidates, key=lambda candidate: candidate[0]):
        deferred = loaded[index]
        loaded[index] = json.loads(serialize_recovery_part(item, ""))
        if not fits_recovery_response(payload):
            loaded[index] = deferred

    for index, item in enumerate(items or ()):
        deferred = loaded[index]
        if deferred.get("deferred") is not True:
            continue
        raw_descriptions = deferred["parts"]
        raw_read_parts = deferred["readParts"]
        if not isinstance(raw_descriptions, list) or not isinstance(raw_read_parts, dict):
            raise TypeError("Deferred recovery items must contain part lists and read handles.")
        descriptions = cast("list[HistoryPartDescription]", raw_descriptions)
        read_parts = cast("dict[str, object]", raw_read_parts)
        parts = history_parts(item)
        for part in parts:
            description = describe_history_part(item, part)
            descriptions.append(description)
            deferred["nextPartOffset"] = None if len(descriptions) == len(parts) else len(descriptions)
            read_parts["offset"] = deferred["nextPartOffset"] or 0
            if len(compact_json(description).encode("utf-8")) > MAXIMUM_INLINE_PART_BYTES or not fits_recovery_response(
                payload
            ):
                descriptions.pop()
                deferred["nextPartOffset"] = len(descriptions)
                read_parts["offset"] = len(descriptions)
                break
    return _tool_return(recovery_response(payload))


def history_parts_page(item: PublicHistoryItem, offset: int = 0, limit: int = MAXIMUM_PAGE_LIMIT) -> dict[str, object]:
    parts = history_parts(item)
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= len(parts):
        raise ValueError("Invalid part-list offset.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAXIMUM_PAGE_LIMIT:
        raise ValueError("Invalid part-list limit.")
    descriptions: list[HistoryPartDescription] = []
    page: dict[str, object] = {
        "parts": descriptions,
        "offset": offset,
        "nextOffset": offset if offset < len(parts) else None,
        "totalParts": len(parts),
    }
    for index in range(offset, min(len(parts), offset + limit)):
        descriptions.append(describe_history_part(item, parts[index]))
        page["nextOffset"] = None if index + 1 == len(parts) else index + 1
        if not fits_recovery_response(page):
            descriptions.pop()
            page["nextOffset"] = index
            break
    if not descriptions and offset < len(parts):
        raise ValueError("Recovery part description exceeds the reply budget.")
    return page


def create_context_paging_toolset(  # noqa: C901, PLR0915 - The four public tools share one strict snapshot and refusal boundary.
    *,
    is_enabled: Callable[[], bool],
    public_items: Callable[[], Sequence[PublicHistoryItem]] | None = None,
    history_source: Callable[[], HistoryProjection | Sequence[ModelMessage]] | None = None,
) -> FunctionToolset[object]:
    """Build the four real tools; history is read at execution time, not registration."""
    if (public_items is None) == (history_source is None):
        raise ValueError("Supply exactly one public item or canonical history source.")
    navigator = HistoryNavigator()

    def snapshot() -> HistoryNavigator:
        if public_items is not None:
            items = public_items()
        else:
            if history_source is None:
                raise RuntimeError("A canonical history source was not configured.")
            source = history_source()
            history = source if isinstance(source, HistoryProjection) else project_history(source)
            items = project_public_history(history)
        navigator.rebuild(items)
        return navigator

    def disabled_result(operation: Callable[[], ToolReturn]) -> ToolReturn:
        if not is_enabled():
            return ToolReturn(return_value=_DISABLED)
        try:
            return operation()
        except (ValueError, RecoveryInputError) as error:
            return _tool_return(recovery_text_reply([f"Recovery input error: {error}"]))

    def search_history(  # noqa: PLR0913, PLR0917 - The schema fixes all public arguments.
        query: Annotated[str, Field(max_length=200)],
        files: Annotated[list[str], Field(max_length=10)] | None = None,
        tools: Annotated[list[str], Field(max_length=10)] | None = None,
        failed: bool | None = None,  # noqa: FBT001 - The public tool schema requires this positional boolean.
        limit: Annotated[int, Field(ge=1, le=10)] = 5,
        load: bool = False,  # noqa: FBT001, FBT002 - The public tool schema requires this boolean default.
    ) -> ToolReturn:
        """Find public recovery references, optionally loading the first three items."""

        def execute() -> ToolReturn:
            current = snapshot()
            parameters: dict[str, object] = {"query": query, "limit": limit}
            if files is not None:
                parameters["files"] = files
            if tools is not None:
                parameters["tools"] = tools
            if failed is not None:
                parameters["failed"] = failed
            references = current.search(parameters)
            if not load:
                return bounded_history_reply(references=references)
            ids = [reference["historyId"] for reference in references[:3]]
            return bounded_history_reply(current.load(ids) if ids else [], references)

        return disabled_result(execute)

    def browse_history(
        direction: Literal["backward", "forward", "around"],
        historyId: Annotated[str | None, Field(max_length=128)] = None,
        sequence: Annotated[int | None, Field(ge=0, le=1_000_000)] = None,
        count: Annotated[int, Field(ge=1, le=10)] = 5,
        stride: Annotated[int, Field(ge=1, le=10)] = 1,
    ) -> ToolReturn:
        """Return compact public references around paged history."""

        def execute() -> ToolReturn:
            parameters: dict[str, object] = {"direction": direction, "count": count, "stride": stride}
            if historyId is not None:
                parameters["historyId"] = historyId
            if sequence is not None:
                parameters["sequence"] = sequence
            return bounded_history_reply(references=snapshot().browse(parameters))

        return disabled_result(execute)

    def load_history(
        historyIds: Annotated[list[str], Field(min_length=1, max_length=3)],
    ) -> ToolReturn:
        """Load one to three public items with deferred parts for items that do not fit."""
        return disabled_result(lambda: bounded_history_reply(snapshot().load(historyIds)))

    def read_context_output(  # noqa: PLR0913, PLR0917 - The schema fixes all public arguments.
        historyId: Annotated[str, Field(max_length=MAXIMUM_HISTORY_ID_CHARACTERS)],
        source: Literal["assistant", "toolResult"] | None = None,
        part: Annotated[str | None, Field(max_length=MAXIMUM_PART_CHARACTERS)] = None,
        mode: Literal["text", "parts", "image"] | None = None,
        contentIndex: Annotated[int | None, Field(ge=0)] = None,
        toolCallId: str | None = None,
        offset: Annotated[int | None, Field(ge=0)] = None,
        limit: Annotated[int | None, Field(ge=1, le=MAXIMUM_PAGE_LIMIT)] = None,
    ) -> ToolReturn:
        """Read exact text, JSON, part descriptions, or validate an image handle."""

        def execute() -> ToolReturn:
            if utf16_length(historyId) > MAXIMUM_HISTORY_ID_CHARACTERS:
                raise ValueError(f"historyId must be at most {MAXIMUM_HISTORY_ID_CHARACTERS} characters.")
            if part is not None and utf16_length(part) > MAXIMUM_PART_CHARACTERS:
                raise ValueError(f"part must be at most {MAXIMUM_PART_CHARACTERS} characters.")
            parameters: dict[str, object] = {"historyId": historyId}
            for key, value in (
                ("source", source),
                ("part", part),
                ("mode", mode),
                ("contentIndex", contentIndex),
                ("toolCallId", toolCallId),
                ("offset", offset),
                ("limit", limit),
            ):
                if value is not None:
                    parameters[key] = value
            current = snapshot()
            selected_mode = mode or "text"
            if selected_mode == "text":
                return _tool_return(
                    recovery_text_reply(context_output_texts(read_output_page(current.items, parameters)))
                )
            if source is not None or contentIndex is not None or toolCallId is not None:
                raise ValueError("Part-list and image modes do not accept legacy output selectors.")
            item = public_history_item(current.items, historyId)
            if selected_mode == "parts":
                if part is not None:
                    raise ValueError("Part-list mode does not accept a part selector.")
                return _tool_return(
                    recovery_response(history_parts_page(item, offset or 0, limit or MAXIMUM_PAGE_LIMIT))
                )
            if selected_mode != "image":
                raise ValueError("Unknown recovery read mode.")
            if part is None or offset is not None or limit is not None:
                raise ValueError("Image mode requires a part handle without text-page bounds.")
            image = select_history_part(item, part)
            if not is_recovery_image(image):
                raise ValueError("The selected public part is not an image.")
            describe_history_part(item, next(candidate for candidate in history_parts(item) if candidate.part == part))
            raise EngineError("image recovery requires M5 vision support")

        return disabled_result(execute)

    tools: list[Tool[object]] = [
        Tool(search_history, takes_ctx=False),
        Tool(browse_history, takes_ctx=False),
        Tool(load_history, takes_ctx=False),
        Tool(read_context_output, takes_ctx=False),
    ]
    return FunctionToolset(tools, strict=True, id="context-paging-recovery")


__all__ = ["bounded_history_reply", "create_context_paging_toolset", "history_parts_page"]
