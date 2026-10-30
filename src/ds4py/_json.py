"""JSON-shaped types and runtime validators for dynamic input boundaries."""

from typing import TypeIs

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]


def is_json_value(value: object) -> TypeIs[JsonValue]:
    """Return whether *value* contains only JSON-shaped runtime values."""
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(is_json_value(item) for item in value)
    return is_json_object(value)


def is_json_object(value: object) -> TypeIs[JsonObject]:
    """Return whether *value* is a string-keyed JSON object."""
    return isinstance(value, dict) and all(isinstance(key, str) and is_json_value(item) for key, item in value.items())
