import pytest

from ds4py._json import JsonObject, JsonValue, is_json_object, is_json_value


@pytest.mark.parametrize("value", [None, True, False, 0, 1, -1, 0.5, "", "text"])
def test_is_json_value_accepts_scalars(value: JsonValue) -> None:
    assert is_json_value(value)


@pytest.mark.parametrize("value", [[], {}])
def test_is_json_value_accepts_empty_containers(value: JsonValue) -> None:
    assert is_json_value(value)


def test_is_json_value_accepts_nested_containers() -> None:
    value = {"outer": [1, {"inner": [None, True, "text"]}]}
    assert is_json_value(value)
    assert is_json_object(value)


@pytest.mark.parametrize("value", [(1, 2), object(), [object()], {"nested": (1, 2)}])
def test_is_json_value_rejects_non_json_values(value: object) -> None:
    assert not is_json_value(value)


def test_is_json_value_rejects_non_string_object_keys() -> None:
    assert not is_json_value({1: "not a JSON object"})
    assert not is_json_object({1: "not a JSON object"})


def test_is_json_value_rejects_one_invalid_deeply_nested_value() -> None:
    assert not is_json_value({"outer": [{"inner": [1, {"invalid": object()}]}]})


def _accept_json(value: JsonValue) -> JsonValue:
    return value


def _accept_json_object(value: JsonObject) -> JsonObject:
    return value


def _narrow_json(value: object) -> JsonValue | None:
    if is_json_value(value):
        return _accept_json(value)
    return None


def _narrow_json_object(value: object) -> JsonObject | None:
    if is_json_object(value):
        return _accept_json_object(value)
    return None


def test_typeis_validators_narrow_objects_for_json_consumers() -> None:
    value = {"items": [1, "two", None]}
    assert _narrow_json(value) == value
    assert _narrow_json_object(value) == value
    assert _narrow_json(object()) is None
    assert _narrow_json_object([1, 2]) is None
