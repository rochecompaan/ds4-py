"""Parse typed answers to harness multiple-choice questions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai_harness.ask_user import Question


def selected_labels(question: Question, answer: str) -> tuple[str, ...] | None:
    """Return unique valid labels, or ``None`` when the answer is invalid."""
    selected: list[str] = []
    for value in (item.strip() for item in answer.split(",")):
        if not value:
            continue
        if value.isdecimal() and 1 <= int(value) <= len(question.options):
            label = question.options[int(value) - 1].label
        else:
            label = next(
                (option.label for option in question.options if option.label == value),
                None,
            )
        if label is None or label in selected:
            return None
        selected.append(label)
    if not selected or (len(selected) > 1 and not question.multi_select):
        return None
    return tuple(selected)
