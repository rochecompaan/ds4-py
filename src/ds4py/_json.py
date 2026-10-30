"""JSON-shaped types and runtime validators for dynamic input boundaries."""

from collections.abc import Mapping
from typing import TypeIs

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]


def is_json_value(value: object, _seen: set[int] | None = None) -> TypeIs[JsonValue]:
    """Return whether *value* contains only JSON-shaped runtime values."""
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if _seen is None:
        _seen = set()
    identity = id(value)
    if identity in _seen:
        return False
    _seen.add(identity)
    try:
        if isinstance(value, list):
            return all(is_json_value(item, _seen) for item in value)
        if isinstance(value, dict):
            return all(isinstance(key, str) and is_json_value(item, _seen) for key, item in value.items())
        return False
    finally:
        _seen.remove(identity)


def is_string_key_mapping(value: object) -> TypeIs[Mapping[str, object]]:
    """Return whether *value* is a mapping with only string keys."""
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def is_json_object(value: object, _seen: set[int] | None = None) -> TypeIs[JsonObject]:
    """Return whether *value* is a string-keyed JSON object."""
    if not isinstance(value, dict):
        return False
    if _seen is None:
        _seen = set()
    identity = id(value)
    if identity in _seen:
        return False
    _seen.add(identity)
    try:
        return all(isinstance(key, str) and is_json_value(item, _seen) for key, item in value.items())
    finally:
        _seen.remove(identity)
