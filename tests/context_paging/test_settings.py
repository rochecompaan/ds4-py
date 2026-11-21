import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Literal, cast

import pytest

from ds4py.context_paging.settings import RollingWindowSettings, default_trim_to, resolve_settings

FIXTURES = json.loads((Path(__file__).parents[1] / "fixtures/context_paging/settings.json").read_text())["fixtures"]
FILE_FIXTURES = [fixture for fixture in FIXTURES if not fixture["id"].startswith("settings-adaptive-")]
ADAPTIVE_FIXTURES = [fixture for fixture in FIXTURES if fixture["id"].startswith("settings-adaptive-")]


def _write_settings(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _resolve_sources(tmp_path: Path, sources: dict[str, object]) -> RollingWindowSettings:
    project_dir = tmp_path / "project"
    store_root = tmp_path / "store"
    global_settings = sources["globalSettings"]
    _write_settings(store_root / "settings.json", global_settings)
    if "projectSettings" in sources:
        _write_settings(project_dir / ".ds4py/settings.json", sources["projectSettings"])
    return resolve_settings(
        project_dir=project_dir,
        store_root=store_root,
        project_trusted=cast("bool", sources["projectTrusted"]),
    )


def _assert_matches(settings: RollingWindowSettings, expected: dict[str, object]) -> None:
    assert settings.enabled == expected["enabled"]
    assert settings.token_budget == expected["tokenBudget"]
    assert settings.trim_to == expected["trimToTokens"]
    assert settings.trim_to_explicit == expected["trimToTokensExplicit"]


@pytest.mark.parametrize("fixture", FILE_FIXTURES, ids=lambda fixture: fixture["id"])
def test_baseline_file_resolution(fixture: dict[str, object], tmp_path: Path) -> None:
    settings = _resolve_sources(tmp_path, cast("dict[str, object]", fixture["input"]))

    expected = cast("dict[str, object]", fixture["expected"])
    _assert_matches(settings, cast("dict[str, object]", expected["value"]))


@pytest.mark.parametrize("fixture", ADAPTIVE_FIXTURES, ids=lambda fixture: fixture["id"])
def test_baseline_adaptive_targets(fixture: dict[str, object]) -> None:
    token_budget = cast("dict[str, int]", fixture["input"])["tokenBudget"]

    expected = cast("dict[str, object]", fixture["expected"])
    assert default_trim_to(token_budget) == expected["value"]


def test_literal_defaults(tmp_path: Path) -> None:
    settings = resolve_settings(project_dir=tmp_path / "project", store_root=tmp_path / "store", project_trusted=False)

    assert settings == RollingWindowSettings(enabled=True, token_budget=128_000, trim_to=80_000, trim_to_explicit=False)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"enabled": False}, (False, 64_000, 30_000, True)),
        ({"token_budget": 32_000}, (True, 32_000, 30_000, True)),
        ({"trim_to": 20_000}, (True, 64_000, 20_000, True)),
    ],
)
def test_each_programmatic_override_wins_independently(
    overrides: dict[str, object],
    expected: tuple[bool, int, int, bool],
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    store_root = tmp_path / "store"
    _write_settings(project_dir / ".ds4py/settings.json", {"contextPaging": {"enabled": True}})
    _write_settings(store_root / "settings.json", {"contextPaging": {"tokenBudget": 64_000, "trimToTokens": 30_000}})

    settings = resolve_settings(
        project_dir=project_dir,
        store_root=store_root,
        project_trusted=True,
        enabled=cast("bool", overrides.get("enabled")),
        token_budget=cast("int | None", overrides.get("token_budget")),
        trim_to=cast("int | None", overrides.get("trim_to")),
    )

    assert (settings.enabled, settings.token_budget, settings.trim_to, settings.trim_to_explicit) == expected


def test_programmatic_budget_recomputes_only_an_adaptive_target(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store_root = tmp_path / "store"
    _write_settings(store_root / "settings.json", {"contextPaging": {"tokenBudget": 64_000}})

    adaptive = resolve_settings(
        project_dir=project_dir, store_root=store_root, project_trusted=False, token_budget=32_000
    )

    assert adaptive.trim_to == 20_000
    assert adaptive.trim_to_explicit is False


@pytest.mark.parametrize("field", ["token_budget", "trim_to"])
@pytest.mark.parametrize("value", [True, 1.5, 0, -1, 2**53])
def test_direct_integer_fields_reject_invalid_values(
    field: Literal["token_budget", "trim_to"], value: object, tmp_path: Path
) -> None:
    kwargs = {field: cast("int", value)}

    with pytest.raises(ValueError, match="positive safe integer"):
        resolve_settings(project_dir=tmp_path, store_root=tmp_path / "store", project_trusted=False, **kwargs)


@pytest.mark.parametrize("value", [1, 0, "true"])
def test_direct_enabled_rejects_non_booleans(value: object, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="enabled must be a boolean"):
        resolve_settings(
            project_dir=tmp_path,
            store_root=tmp_path / "store",
            project_trusted=False,
            enabled=cast("bool", value),
        )


@pytest.mark.parametrize("value", [True, 1.5, 0, -1, 2**53])
def test_adaptive_target_rejects_invalid_direct_budget(value: object) -> None:
    with pytest.raises(ValueError, match="positive safe integer"):
        default_trim_to(cast("int", value))


def test_maximum_safe_integer_is_exact() -> None:
    assert default_trim_to(2**53 - 1) == 5_629_499_534_213_119


def test_records_are_immutable() -> None:
    settings = RollingWindowSettings(enabled=True, token_budget=1, trim_to=1, trim_to_explicit=False)

    with pytest.raises(FrozenInstanceError):
        settings.__setattr__("enabled", False)


def test_malformed_trusted_project_file_raises_original_error(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    settings_path = project_dir / ".ds4py/settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        resolve_settings(project_dir=project_dir, store_root=tmp_path / "store", project_trusted=True)


def test_malformed_global_file_raises_even_with_project_overrides(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    store_root = tmp_path / "store"
    _write_settings(project_dir / ".ds4py/settings.json", {"contextPaging": {"enabled": True}})
    settings_path = store_root / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        resolve_settings(project_dir=project_dir, store_root=store_root, project_trusted=True)


def test_unreadable_file_raises_original_error(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    (store_root / "settings.json").mkdir(parents=True)

    with pytest.raises(IsADirectoryError):
        resolve_settings(project_dir=tmp_path / "project", store_root=store_root, project_trusted=False)


def test_untrusted_project_file_is_not_read(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    (project_dir / ".ds4py/settings.json").mkdir(parents=True)

    settings = resolve_settings(project_dir=project_dir, store_root=tmp_path / "store", project_trusted=False)

    assert settings.enabled is True
