"""RollingWindow settings records and file/override resolution."""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeIs

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

DEFAULT_TOKEN_BUDGET = 128_000
MAXIMUM_SAFE_INTEGER = 2**53 - 1


def _positive_safe_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAXIMUM_SAFE_INTEGER:
        raise ValueError(f"{name} must be a positive safe integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class RollingWindowSettings:
    """Resolved RollingWindow policy values."""

    enabled: bool
    token_budget: int
    trim_to: int
    trim_to_explicit: bool

    def __post_init__(self) -> None:
        _boolean(self.enabled, "enabled")
        _positive_safe_integer(self.token_budget, "token_budget")
        _positive_safe_integer(self.trim_to, "trim_to")
        _boolean(self.trim_to_explicit, "trim_to_explicit")


def default_trim_to(token_budget: int) -> int:
    """Return five eighths of a budget without floating-point rounding."""

    budget = _positive_safe_integer(token_budget, "token_budget")
    return max(1, budget * 5 // 8)


def _read_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return json.loads(text)


def _object_mapping(value: object) -> TypeIs[Mapping[str, object]]:
    return isinstance(value, dict)


def _paging_section(settings: object) -> Mapping[str, object] | None:
    if not _object_mapping(settings):
        return None
    section = settings.get("contextPaging")
    return section if _object_mapping(section) else None


def _read_enabled(settings: object) -> bool | None:
    section = _paging_section(settings)
    if section is None:
        return None
    value = section.get("enabled")
    return value if isinstance(value, bool) else None


def _read_integer(settings: object, key: str) -> int | None:
    section = _paging_section(settings)
    if section is None:
        return None
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAXIMUM_SAFE_INTEGER:
        return None
    return value


def resolve_settings(  # noqa: PLR0913 - The public resolver owns three sources and three independent overrides.
    *,
    project_dir: Path,
    store_root: Path,
    project_trusted: bool,
    enabled: bool | None = None,
    token_budget: int | None = None,
    trim_to: int | None = None,
) -> RollingWindowSettings:
    """Resolve trusted project, global, and explicit programmatic settings."""

    direct_enabled = _boolean(enabled, "enabled") if enabled is not None else None
    direct_budget = _positive_safe_integer(token_budget, "token_budget") if token_budget is not None else None
    direct_trim = _positive_safe_integer(trim_to, "trim_to") if trim_to is not None else None

    project = _read_json(project_dir / ".ds4py/settings.json") if project_trusted else None
    global_settings = _read_json(store_root / "settings.json")

    resolved_enabled = _read_enabled(project)
    if resolved_enabled is None:
        resolved_enabled = _read_enabled(global_settings)
    if direct_enabled is not None:
        resolved_enabled = direct_enabled
    if resolved_enabled is None:
        resolved_enabled = True

    resolved_budget = _read_integer(project, "tokenBudget")
    if resolved_budget is None:
        resolved_budget = _read_integer(global_settings, "tokenBudget")
    if direct_budget is not None:
        resolved_budget = direct_budget
    if resolved_budget is None:
        resolved_budget = DEFAULT_TOKEN_BUDGET

    file_trim = _read_integer(project, "trimToTokens")
    if file_trim is None:
        file_trim = _read_integer(global_settings, "trimToTokens")
    trim_to_explicit = file_trim is not None or direct_trim is not None
    resolved_trim = direct_trim if direct_trim is not None else file_trim
    if resolved_trim is None:
        resolved_trim = default_trim_to(resolved_budget)

    return RollingWindowSettings(
        enabled=resolved_enabled,
        token_budget=resolved_budget,
        trim_to=resolved_trim,
        trim_to_explicit=trim_to_explicit,
    )
