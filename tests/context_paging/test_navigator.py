import pytest

from ds4py.context_paging.navigator import HistoryNavigator, HistoryNavigatorError
from ds4py.context_paging.public_history import (
    PublicModelTurn,
    PublicText,
    PublicToolCall,
    PublicToolResult,
    PublicUser,
)


def _items() -> list[PublicUser | PublicModelTurn]:
    return [
        PublicUser("user-1", 0, 0, ("Please inspect the needle at src/nested/needle.py",)),
        PublicModelTurn(
            "turn-1",
            1,
            0,
            (
                PublicText("I found the needle in the nested module."),
                PublicToolCall("call-1", "read", {"path": "src/nested/needle.py"}, 1),
            ),
            (PublicToolResult("call-1", "read", [{"type": "text", "text": "needle result"}], False, 1, 0),),
            ("read",),
            ("src/nested/needle.py",),
            False,
            False,
        ),
        PublicModelTurn("turn-2", 2, 1, (PublicText("unrelated response"),), (), ("shell",), (), True, False),
    ]


def test_search_ranks_matches_and_preserves_context_filters() -> None:
    navigator = HistoryNavigator(_items())

    matches = navigator.search({"query": "needle", "limit": 10})

    assert [match["historyId"] for match in matches] == ["user-1", "turn-1"]
    assert "needle" in matches[0]["preview"]
    assert matches[1]["previousHistoryId"] == "user-1"
    assert matches[1]["nextHistoryId"] == "turn-2"
    assert [match["historyId"] for match in navigator.search({"query": "needle", "tools": ["read"]})] == ["turn-1"]
    assert [
        match["historyId"] for match in navigator.search({"query": "needle", "files": ["src/nested/needle.py"]})
    ] == ["turn-1"]
    assert navigator.search({"query": "needle", "failed": True}) == []


def test_browse_defaults_directions_stride_and_anchor_validation() -> None:
    navigator = HistoryNavigator(_items())

    assert [item["historyId"] for item in navigator.browse({"direction": "backward", "count": 2})] == [
        "turn-2",
        "turn-1",
    ]
    assert [item["historyId"] for item in navigator.browse({"direction": "forward", "count": 2})] == [
        "user-1",
        "turn-1",
    ]
    around = navigator.browse({"direction": "around", "historyId": "turn-1", "count": 3})
    assert [item["historyId"] for item in around] == ["user-1", "turn-1", "turn-2"]
    assert navigator.browse({"direction": "backward", "historyId": "turn-2", "count": 2})[0]["historyId"] == "turn-1"
    with pytest.raises(HistoryNavigatorError, match="Unknown history anchor") as error:
        navigator.browse({"direction": "around", "historyId": "missing"})
    assert error.value.code == "UNKNOWN_HISTORY_ANCHOR"


def test_load_retains_requested_order_and_validates_one_to_three_ids() -> None:
    navigator = HistoryNavigator(_items())

    loaded = navigator.load(["turn-2", "user-1"])
    assert [item.history_id for item in loaded] == ["turn-2", "user-1"]
    with pytest.raises(HistoryNavigatorError) as error:
        navigator.load(["user-1", "missing"])
    assert error.value.code == "UNKNOWN_HISTORY_ID"
    with pytest.raises(HistoryNavigatorError) as error:
        navigator.load([])
    assert error.value.code == "INVALID_HISTORY_NAVIGATOR_INPUT"


def test_rebuild_cache_identity_changes_only_for_visible_stable_ids() -> None:
    items = _items()
    navigator = HistoryNavigator(items)
    identity = navigator.cache_identity
    first_corpus = navigator.search({"query": "needle"})

    navigator.rebuild(items)
    assert navigator.cache_identity == identity
    assert navigator.search({"query": "needle"}) == first_corpus

    navigator.rebuild([*items, PublicUser("user-2", 3, 2, ("new",))])
    assert navigator.cache_identity == identity + 1
