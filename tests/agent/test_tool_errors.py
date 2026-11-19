import asyncio

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai import CallToolsNode
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from ds4py.capabilities.tool_errors import ToolErrorRecovery


class Result(BaseModel):
    value: int


def test_output_retry_exhaustion_still_propagates() -> None:
    try:
        Result.model_validate({"value": "bad"})
    except ValidationError as cause:
        error = UnexpectedModelBehavior("Exceeded maximum output retries (1)")
        error.__cause__ = cause
    else:
        raise AssertionError("the invalid output fixture must fail validation")

    call = ToolCallPart("final_result", {"value": "bad"}, tool_call_id="output")
    node = CallToolsNode(ModelResponse(parts=[call]))
    interrupted = ModelRequest(parts=[], state="interrupted")
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[]))
    ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=[interrupted])

    with pytest.raises(UnexpectedModelBehavior) as caught:
        asyncio.run(ToolErrorRecovery().on_node_run_error(ctx, node=node, error=error))

    assert caught.value is error
    assert ctx.messages == [interrupted]
