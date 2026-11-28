"""Read-only saved-history, request-input, and cache statistics."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    BinaryImage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, FileStepStore, RunRecord

from ds4py.context_paging.stats import (
    ContextPagingStats,
    InputTokenReading,
    RollingWindowStatisticsState,
    collect_statistics,
    read_request_input,
)
from ds4py.message_metadata import (
    DS4_CONTEXT_PAGING,
    DS4_KEY,
    DS4_PAGING_INPUT_TOKENS,
    DS4_REQUEST_SYNC_COUNT,
)


def keyed(message: ModelRequest | ModelResponse, key: str, **metadata: object) -> ModelRequest | ModelResponse:
    message.metadata = {DS4_KEY: key, **metadata}
    return message


def response(
    key: str,
    *,
    input_tokens: int | None = 2,
    cache_read: int | None = 3,
    cache_write: int | None = 4,
    syncs: object = 1,
    exact: object = None,
) -> ModelResponse:
    metadata: dict[str, object] = {}
    if syncs is not None:
        metadata[DS4_REQUEST_SYNC_COUNT] = syncs
    if exact is not None:
        metadata[DS4_PAGING_INPUT_TOKENS] = exact
    saved = ModelResponse(
        [TextPart("ijkl")],
        usage=RequestUsage(
            input_tokens=input_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        ),
    )
    saved.metadata = {DS4_KEY: key, **metadata}
    return saved


def current(*, enabled: bool = True, input_tokens: int | None = None) -> RollingWindowStatisticsState:
    return RollingWindowStatisticsState(
        enabled=enabled,
        token_budget=80,
        model_window=100,
        input_tokens=input_tokens,
    )


async def save_snapshot(
    store: FileStepStore,
    run_id: str,
    messages: list[ModelRequest | ModelResponse],
    key: str,
) -> None:
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id=run_id,
            step_index=0,
            messages=deepcopy(messages),
            timestamp=datetime(2026, 11, 21, tzinfo=UTC),
            idempotency_key=key,
        )
    )


def test_real_store_rehydrates_media_deduplicates_and_skips_damaged_snapshots(tmp_path: Path) -> None:
    async def scenario() -> None:
        steps = tmp_path / "steps"
        store = FileStepStore(steps, media_threshold_bytes=8)
        await store.register_run(RunRecord(run_id="run"))
        user = keyed(ModelRequest([SystemPromptPart("resident"), UserPromptPart("abcd")]), "user")
        turn = response("response", cache_read=3, cache_write=4, syncs=2, exact=42)
        turn.parts = [
            ThinkingPart("efgh"),
            TextPart("ijkl"),
            ToolCallPart("read", {"x": "y"}, tool_call_id="call"),
        ]
        result = keyed(
            ModelRequest(
                [
                    ToolReturnPart(
                        "read",
                        "mnop" * 20,
                        tool_call_id="call",
                        metadata={"usage": {"cacheRead": 1, "cacheWrite": 2}},
                    )
                ]
            ),
            "result",
        )
        notice = keyed(
            ModelRequest([UserPromptPart("transient notice")]),
            "notice",
            **{DS4_CONTEXT_PAGING: "notice"},
        )
        image = keyed(
            ModelRequest([UserPromptPart([BinaryImage(data=b"image", media_type="image/png")])]),
            "image",
        )
        messages = [user, turn, result, notice, image]
        await save_snapshot(store, "run", messages, "one")
        await save_snapshot(store, "run", messages, "two")
        revised = deepcopy(messages)
        revised_user = revised[0]
        assert isinstance(revised_user, ModelRequest)
        revised_part = revised_user.parts[-1]
        assert isinstance(revised_part, UserPromptPart)
        revised_part.content = "abcdefgh"
        await save_snapshot(store, "run", revised, "three")
        (steps / "run" / "events.jsonl").write_text("excluded telemetry")
        (steps / "run" / "snapshots" / "999.json").write_text("{damaged")
        (tmp_path / "session.kv").write_bytes(b"excluded kv")
        (tmp_path / "session.json").write_text("excluded save record")

        stats = await collect_statistics(store=store, session_dir=tmp_path, current=current())

        assert isinstance(stats, ContextPagingStats)
        # user 1 + response ceil(21/4) + tool result 20 + image 1200 + revision 2.
        assert stats.history_tokens == 1229
        assert stats.input == InputTokenReading("exact", 42)
        assert stats.cache_read_tokens == 4
        assert stats.cache_write_tokens == 6
        assert stats.stored_bytes == sum(
            path.stat().st_size for path in steps.rglob("*") if path.is_file() and path.name != "events.jsonl"
        )
        assert len(list((steps / "media").glob("*.bin"))) >= 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        (
            response("exact", input_tokens=100, cache_read=200, cache_write=300, syncs=2, exact=11),
            InputTokenReading("exact", 11),
        ),
        (
            response("old", input_tokens=2, cache_read=3, cache_write=4, syncs=1),
            InputTokenReading("reconstructible", 9),
        ),
        (
            response("zero", input_tokens=0, cache_read=0, cache_write=0, syncs=1),
            InputTokenReading("reconstructible", 0),
        ),
        (
            response("multi", input_tokens=2, cache_read=3, cache_write=4, syncs=2),
            InputTokenReading("unavailable", None),
        ),
        (
            response("bool", input_tokens=2, cache_read=3, cache_write=4, syncs=True),
            InputTokenReading("unavailable", None),
        ),
        (
            response("float", input_tokens=2, cache_read=3, cache_write=4, syncs=1.0),
            InputTokenReading("unavailable", None),
        ),
        (
            response("unknown", input_tokens=2, cache_read=3, cache_write=4, syncs=None),
            InputTokenReading("unavailable", None),
        ),
        (
            response("missing", input_tokens=2, cache_read=3, cache_write=None, syncs=1),
            InputTokenReading("unavailable", None),
        ),
        (
            response("bad-exact", input_tokens=2, cache_read=3, cache_write=4, syncs=1, exact=-1),
            InputTokenReading("unavailable", None),
        ),
    ],
    ids=[
        "exact-multi-sync",
        "old-single-sync",
        "zero",
        "multi-sync",
        "bool-sync-count",
        "float-sync-count",
        "no-sync-count",
        "missing-counter",
        "bad-exact",
    ],
)
def test_request_input_provenance(saved: ModelResponse, expected: InputTokenReading) -> None:
    assert read_request_input(saved) == expected


def test_missing_cache_counter_affects_only_that_total_and_zero_is_known(tmp_path: Path) -> None:
    class DamagedStore:
        async def list_runs(
            self, *, parent_run_id: str | None = None, conversation_id: str | None = None
        ) -> list[RunRecord]:
            return [RunRecord(run_id="run")]

        async def list_snapshots(self, *, run_id: str, include_interrupted: bool = False) -> list[ContinuableSnapshot]:
            zero = response("zero", cache_read=0, cache_write=0)
            missing = response("missing-read", cache_read=0, cache_write=5)
            object.__setattr__(missing.usage, "cache_read_tokens", None)
            snapshot = ContinuableSnapshot(run_id=run_id, step_index=0, messages=[zero, missing])
            return [snapshot, deepcopy(snapshot)]

    async def scenario() -> None:
        stats = await collect_statistics(store=DamagedStore(), session_dir=tmp_path, current=current())
        assert stats.cache_read_tokens is None
        assert stats.cache_write_tokens == 5

    asyncio.run(scenario())


def test_disabled_report_marks_budget_inactive_and_only_history_estimated(tmp_path: Path) -> None:
    async def scenario() -> None:
        stats = await collect_statistics(
            store=FileStepStore(tmp_path / "steps"),
            session_dir=tmp_path,
            current=current(enabled=False),
        )
        report = stats.render()
        assert report.splitlines() == [
            "Context paging: off",
            "",
            "SESSION — complete saved history",
            "  Stored size          0 B",
            "  History tokens       ~0",
            "",
            "CONTEXT — latest model request",
            "  Input tokens         unavailable",
            "  Paging budget        80   (inactive)",
            "  Model window         100",
            "",
            "CACHE — whole-session totals",
            "  Tokens read          0",
            "  Tokens written       0",
        ]
        assert report.count("~") == 1

    asyncio.run(scenario())
