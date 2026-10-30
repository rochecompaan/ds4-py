from typing import Any

import pytest

from ds4py.engine.plan import Plan, plan_sync

P = list(range(1, 21))


def plan(tokens: Any, live: Any = (), valid: Any = True, cps: Any = None, ctx: Any = 100, restore: Any = True) -> Any:
    return plan_sync(tokens, live=list(live), live_valid=valid, checkpoints=cps or {}, ctx=ctx, restore_enabled=restore)


def test_extend() -> None:
    assert plan(P, live=P[:5]) == Plan("extend", 5, 15, True, None)
    assert plan(P, live=P) == Plan("extend", 20, 0, True, None)
    assert plan(P, live=P[:5], cps={"longer": P[:8]}).kind == "extend"


def test_restore_picks_the_longest_checkpoint() -> None:
    p = plan(P, live=[9, 9], cps={"system": P[:3], "turn": P[:8], "other": [7]})
    assert p == Plan("restore", 8, 12, True, "turn")
    assert plan(P, valid=False, cps={"first": P[:8], "second": P[:8]}).checkpoint == "first"


def test_rebuild() -> None:
    assert plan(P, live=P[:5], valid=False) == Plan("rebuild", 0, 20, True, None)
    assert plan(P).kind == "rebuild"
    assert plan(P, live=[9], cps={"system": P[:3]}, restore=False).kind == "rebuild"
    assert plan(P[:3], live=P[:5], cps={"empty": [], "too_long": P}).reused == 0
    assert plan([]) == Plan("rebuild", 0, 0, True, None)


def test_fits_leaves_the_reserve() -> None:
    assert plan(P, ctx=21).fits and not plan(P, ctx=20).fits


@pytest.mark.parametrize("reserve,ctx,fits", [(0, 20, False), (-1, 20, False), (3, 22, False), (3, 23, True)])
def test_reserve_has_minimum_one(reserve: Any, ctx: Any, fits: Any) -> None:
    assert plan_sync(P, live=(), live_valid=True, checkpoints={}, ctx=ctx, reserve=reserve).fits is fits
