from dataclasses import replace
from typing import Any

from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage


def run_context(**attributes: Any) -> RunContext[object]:
    """Build a real harness RunContext with only test-specific overrides."""
    context = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    return replace(context, **attributes)
